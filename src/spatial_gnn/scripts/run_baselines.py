import numpy as np
import pandas as pd
import pickle
import os
import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set_style("ticks")

import torch
from torch_geometric import profile
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

from spatial_gnn.utils.dataset_utils import load_dataset_config, parse_center_celltypes, parse_gene_list
from spatial_gnn.models.gnn_model import GNN
from spatial_gnn.datasets.spatial_dataset import SpatialAgingCellDataset
from spatial_gnn.models.mean_baselines import khop_mean_baseline_batch, center_celltype_mean_baseline_batch, global_mean_baseline_batch


def main():
    # set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset to use (aging_coronal, aging_sagittal, exercise, reprogramming, allen, kukanja, pilot)", type=str, required=True)
    parser.add_argument("--base_path", help="Base path to the data directory", type=str, required=True)
    parser.add_argument("--exp_name", help="Experiment name", type=str, required=True)
    parser.add_argument("--k_hop", help="k-hop neighborhood size", type=int, required=True)
    parser.add_argument("--augment_hop", help="number of hops to take for graph augmentation", type=int, required=True)
    parser.add_argument("--center_celltypes", help="cell type labels to center graphs on, separated by comma. Use 'all' for all cell types or 'none' for no cell type filtering", type=str, required=True)
    parser.add_argument("--node_feature", help="node features key, e.g. 'celltype_age_region'", type=str, required=True)
    parser.add_argument("--inject_feature", help="inject features key, e.g. 'center_celltype'", type=str, required=True)
    args = parser.parse_args()

    # Load dataset configurations
    DATASET_CONFIGS = load_dataset_config()
    
    # set which model to use
    use_model = "best_model"

    # Validate dataset choice
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Dataset must be one of: {', '.join(DATASET_CONFIGS.keys())}")
    print(f"\n {args.dataset}", flush=True)
    
    # load parameters from arguments
    dataset_config = DATASET_CONFIGS[args.dataset]
    train_ids = dataset_config['train_ids']
    test_ids = dataset_config['test_ids']
    file_path = os.path.join(args.base_path, dataset_config['file_name'])
    k_hop = args.k_hop
    augment_hop = args.augment_hop
    
    # Handle center_celltypes
    center_celltypes = parse_center_celltypes(args.center_celltypes)
    node_feature = args.node_feature
    inject_feature = args.inject_feature

    if inject_feature.lower() == "none":
        inject_feature = None
        inject=False
    else:
        inject=True

    # determine gpu / cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)


    # Build cell type index
    celltypes_to_index = {}
    for ci, cellt in enumerate(dataset_config["celltypes"]):
        celltypes_to_index[cellt] = ci
    
    # init dataset with settings
    train_dataset = SpatialAgingCellDataset(subfolder_name="train",
                                            dataset_prefix=args.exp_name,
                                            target="expression",
                                            k_hop=k_hop,
                                            augment_hop=augment_hop,
                                            node_feature=node_feature,
                                            inject_feature=inject_feature,
                                            num_cells_per_ct_id=100,
                                            center_celltypes=center_celltypes,
                                            use_ids=train_ids,
                                            raw_filepaths=[file_path],
                                            celltypes_to_index=celltypes_to_index)

    test_dataset = SpatialAgingCellDataset(subfolder_name="test",
                                        dataset_prefix=args.exp_name,
                                        target="expression",
                                        k_hop=k_hop,
                                        augment_hop=augment_hop,
                                        node_feature=node_feature,
                                        inject_feature=inject_feature,
                                        num_cells_per_ct_id=100,
                                        center_celltypes=center_celltypes,
                                        use_ids=test_ids,
                                        raw_filepaths=[file_path],
                                        celltypes_to_index=celltypes_to_index)

    test_dataset.process()  
    print("Finished processing test dataset", flush=True)
    train_dataset.process()
    print("Finished processing train dataset", flush=True)
    
    all_test_data, all_train_data = [], []
    
    # for f in tqdm(train_dataset.processed_file_names):
    #     batch_list = torch.load(os.path.join(train_dataset.processed_dir, f), weights_only=False)
    #     all_train_data.extend(batch_list)
    for f in tqdm(test_dataset.processed_file_names):
        batch_list = torch.load(os.path.join(test_dataset.processed_dir, f), weights_only=False)
        all_test_data.extend(batch_list)
    
    # train_loader = DataLoader(all_train_data, batch_size=512, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=None, persistent_workers=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=None, persistent_workers=False)
    print(len(all_train_data), len(all_test_data), flush=True)
    
    save_dir = os.path.join("results/baselines",train_dataset.processed_dir.split("/")[-2])
    os.makedirs(save_dir, exist_ok=True)
    
    print("Running k-hop baseline...", flush=True)
    eval_model(None, test_loader, save_dir, "khop_mean")
    breakpoint()
    print("Running center cell type baseline...", flush=True)
    eval_model(None, test_loader, save_dir, "khop_celltype_mean")
    print("Running global baseline...", flush=True)
    eval_model(None, test_loader, save_dir, "global_mean")


def eval_model(train_loader, test_loader, save_dir, baseline_type, device="cuda"):

    save_dir = os.path.join(save_dir, baseline_type)
    os.makedirs(save_dir, exist_ok=True)

    print("Measuring baseline performance bulk and by cell type...", flush=True)

    preds = []
    actuals = []
    celltypes = []
    
    if baseline_type == "global_mean":
        global_mean = global_mean_baseline_batch(train_loader)
        for data in tqdm(test_loader):
            preds.append(global_mean)
            actuals.append(train_loader.dataset.y.float())
        preds = np.concatenate([pred.detach().cpu().numpy() for pred in preds])
        celltypes = np.concatenate((celltypes,np.concatenate(train_loader.dataset.center_celltype))) # [512]
    else:
        for data in tqdm(test_loader):
            data = data.to(device)

            if baseline_type == "khop_mean":
                out = khop_mean_baseline_batch(data) # [512, 300]
            elif baseline_type == "khop_celltype_mean":
                out = center_celltype_mean_baseline_batch(data) # [512, 300]   
            else:
                raise ValueError(f"Baseline type {baseline_type} not recognized!")
                
            preds.append(out)
            
            if data.y.shape != out.shape:
                actuals.append(torch.reshape(data.y.float(), out.shape))
            else:
                actuals.append(data.y.float()) # [[512, 300]]
            
            # get cell type
            celltypes = np.concatenate((celltypes,np.concatenate(data.center_celltype))) # [512]
        
    preds = np.concatenate([pred.detach().cpu().numpy() for pred in preds])
    actuals = np.concatenate([act.detach().cpu().numpy() for act in actuals])
    celltypes = np.array(celltypes)
    
    # drop genes that are missing everywhere
    preds = preds[:, actuals.max(axis=0)>=0]
    actuals = actuals[:, actuals.max(axis=0)>=0]
    
    # gene stats
    gene_r = []
    gene_s = []
    gene_r2 = []
    gene_mae = []
    gene_rmse = []

    for g in range(preds.shape[1]):
        
        r, p = pearsonr(preds[:,g], actuals[:,g])
        gene_r.append(r)
        
        s, p = spearmanr(preds[:,g], actuals[:,g])
        gene_s.append(s)
        
        r2 = r2_score(actuals[:,g], preds[:,g])
        gene_r2.append(r2)
        
        gene_mae.append(np.mean(np.abs(preds[:,g]-actuals[:,g])))
        gene_rmse.append(np.sqrt(np.mean((preds[:,g]-actuals[:,g])**2)))
        

    # cell stats
    cell_r = []
    cell_s = []
    cell_r2 = []
    cell_mae = []
    cell_rmse = []

    for c in range(preds.shape[0]):
        r, p = pearsonr(preds[c,:], actuals[c,:])
        cell_r.append(r)
        
        s, p = spearmanr(preds[c,:], actuals[c,:])
        cell_s.append(s)
        
        r2 = r2_score(actuals[c,:], preds[c,:])
        cell_r2.append(r2)
        
        cell_mae.append(np.mean(np.abs(preds[c,:]-actuals[c,:])))
        cell_rmse.append(np.sqrt(np.mean((preds[c,:]-actuals[c,:])**2)))
    
    # save gene stats dataframe
    df_gene = pd.DataFrame(np.vstack((gene_r, gene_s, gene_r2, gene_mae, gene_rmse)).T,
                           columns=["Pearson","Spearman","R2","MAE", "RMSE"])
    df_gene.to_csv(os.path.join(save_dir, "test_evaluation_stats_gene.csv"), index=False)

    # save cell stats dataframe
    df_cell = pd.DataFrame(np.vstack((cell_r, cell_s, cell_r2, cell_mae, cell_rmse)).T,
                           columns=["Pearson","Spearman","R2","MAE", "RMSE"])
    df_cell.to_csv(os.path.join(save_dir, "test_evaluation_stats_cell.csv"), index=False)
    
    # print bulk results
    print("Finished bulk analysis:", flush=True)
    print("Cell:", flush=True)
    print(df_cell.median(axis=0), flush=True)
    print("Gene:", flush=True)
    print(df_gene.median(axis=0), flush=True)
    
    # stats broken down by cell type
    ct_stats_dict = {}

    for ct in np.unique(celltypes):
        
        ct_stats_dict[ct] = {}

        # gene stats
        gene_r = []
        gene_s = []
        gene_r2 = []
        gene_mae = []
        gene_rmse = []

        for g in range(preds.shape[1]):
            
            if len(preds[celltypes==ct,g]) > 1:
                r, p = pearsonr(preds[celltypes==ct,g], actuals[celltypes==ct,g])
                s, p = spearmanr(preds[celltypes==ct,g], actuals[celltypes==ct,g])
                r2 = r2_score(actuals[celltypes==ct,g], preds[celltypes==ct,g])
                gene_mae.append(np.mean(np.abs(preds[celltypes==ct,g]-actuals[celltypes==ct,g])))
                gene_rmse.append(np.sqrt(np.mean((preds[celltypes==ct,g]-actuals[celltypes==ct,g])**2)))
            else:
                r = np.nan
                s = np.nan
                r2 = np.nan
                gene_mae.append(np.nan)
                gene_rmse.append(np.nan)
                
            gene_r.append(r)
            gene_s.append(s)
            gene_r2.append(r2)

        
        # cell stats
        cell_r = []
        cell_s = []
        cell_r2 = []
        cell_mae = []
        cell_rmse = []

        for c in np.where(celltypes==ct)[0]:
            
            if len(preds[c,:]) > 1:
                r, p = pearsonr(preds[c,:], actuals[c,:])
                s, p = spearmanr(preds[c,:], actuals[c,:])
                r2 = r2_score(actuals[c,:], preds[c,:])
                cell_mae.append(np.mean(np.abs(preds[c,:]-actuals[c,:])))
                cell_rmse.append(np.sqrt(np.mean((preds[c,:]-actuals[c,:])**2)))
            else:
                r = np.nan
                s = np.nan
                r2 = np.nan
                gene_mae.append(np.nan)
                gene_rmse.append(np.nan)
            
            cell_r.append(r)
            cell_s.append(s)
            cell_r2.append(r2)

            #pred_ct = celltypes==ct
            
        # add results to dictionary
        ct_stats_dict[ct]["Gene - Pearson (mean)"] = robust_nanmean(gene_r) 
        ct_stats_dict[ct]["Gene - Pearson (median)"] = robust_nanmedian(gene_r)
        ct_stats_dict[ct]["Gene - Spearman (mean)"] = robust_nanmean(gene_s)
        ct_stats_dict[ct]["Gene - Spearman (median)"] = robust_nanmedian(gene_s)
        ct_stats_dict[ct]["Gene - R2 (mean)"] = robust_nanmean(gene_r2)
        ct_stats_dict[ct]["Gene - R2 (median)"] = robust_nanmedian(gene_r2)
        ct_stats_dict[ct]["Gene - MAE (mean)"] = robust_nanmean(gene_mae)
        ct_stats_dict[ct]["Gene - RMSE (mean)"] = robust_nanmean(gene_rmse)
        
        ct_stats_dict[ct]["Cell - Pearson (mean)"] = robust_nanmean(cell_r)
        ct_stats_dict[ct]["Cell - Pearson (median)"] = robust_nanmedian(cell_r)
        ct_stats_dict[ct]["Cell - Spearman (mean)"] = robust_nanmean(cell_s)
        ct_stats_dict[ct]["Cell - Spearman (median)"] = robust_nanmedian(cell_s)
        ct_stats_dict[ct]["Cell - R2 (mean)"] = robust_nanmean(cell_r2)
        ct_stats_dict[ct]["Cell - R2 (median)"] = robust_nanmedian(cell_r2)
        ct_stats_dict[ct]["Cell - MAE (mean)"] = robust_nanmean(cell_mae)
        ct_stats_dict[ct]["Cell - RMSE (mean)"] = robust_nanmean(cell_rmse)
    
    # save cell type results
    with open(os.path.join(save_dir, "test_evaluation_stats_bycelltype.pkl"), 'wb') as f:
        pickle.dump(ct_stats_dict, f)
    
    # make cell type plots
    
    # Cell stat plots
    with open(os.path.join(save_dir, "test_evaluation_stats_bycelltype.pkl"), 'rb') as handle:
        ct_stats_dict = pickle.load(handle)

    columns_to_plot = ["Cell - Pearson (median)", "Cell - Spearman (median)", "Cell - R2 (median)"]
        
    #--------------------------------
    metric_col = []
    ct_col = []
    val_col = []

    for col in columns_to_plot:
        for ct in ct_stats_dict.keys():
            val = ct_stats_dict[ct][col]
            
            metric_col.append(col)
            ct_col.append(ct)
            val_col.append(val)

    plot_df = pd.DataFrame(np.vstack((metric_col, ct_col, val_col)).T, columns=["Metric","Cell type","Value"])
    plot_df["Value"] = plot_df["Value"].astype(float)

    # plot
    fig, ax = plt.subplots(figsize=(12,4))
    sns.barplot(plot_df, x="Cell type", y="Value", hue="Metric", palette="Reds", ax=ax)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.7))
    plt.title(save_dir.split("/")[-2], fontsize=14)
    plt.xticks(rotation=30, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Cell type", fontsize=14)
    plt.ylabel("Metric Value", fontsize=14)
    plt.setp(ax.get_legend().get_texts(), fontsize='14')
    plt.setp(ax.get_legend().get_title(), fontsize='16')
    plt.tight_layout()
    #plt.savefig("plots/expression_prediction_performance/"+save_dir.split("/")[-2]+"_CELL.pdf", bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "prediction_performance_CELL.pdf"), bbox_inches='tight')
    plt.close()
    
    
    # Gene stats plots
    with open(os.path.join(save_dir, "test_evaluation_stats_bycelltype.pkl"), 'rb') as handle:
        ct_stats_dict = pickle.load(handle)

    columns_to_plot = ["Gene - Pearson (median)", "Gene - Spearman (median)"]
        
    #--------------------------------
    metric_col = []
    ct_col = []
    val_col = []

    for col in columns_to_plot:
        for ct in ct_stats_dict.keys():
            val = ct_stats_dict[ct][col]
            
            metric_col.append(col)
            ct_col.append(ct)
            val_col.append(val)

    plot_df = pd.DataFrame(np.vstack((metric_col, ct_col, val_col)).T, columns=["Metric","Cell type","Value"])
    plot_df["Value"] = plot_df["Value"].astype(float)

    # plot
    fig, ax = plt.subplots(figsize=(12,4))
    sns.barplot(plot_df, x="Cell type", y="Value", hue="Metric", palette="Reds", ax=ax)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.7))
    plt.title(save_dir.split("/")[-2], fontsize=14)
    plt.xticks(rotation=30, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Cell type", fontsize=14)
    plt.ylabel("Metric Value", fontsize=14)
    plt.setp(ax.get_legend().get_texts(), fontsize='14')
    plt.setp(ax.get_legend().get_title(), fontsize='16')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "prediction_performance_GENE.pdf"))
    plt.close()
    
    print("Finished cell type analysis.", flush=True)
    

def robust_nanmean(x):
    nmx = np.nanmean(x) if np.count_nonzero(~np.isnan(x))>1 else np.mean(x)
    return (nmx)

def robust_nanmedian(x):
    nmx = np.nanmedian(x) if np.count_nonzero(~np.isnan(x))>1 else np.median(x)
    return (nmx)


if __name__ == "__main__":
    main()
