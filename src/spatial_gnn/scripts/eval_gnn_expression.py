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
from spatial_gnn.models.gnn_model import GNN, CellTypeGNN
from spatial_gnn.models.mean_baselines import khop_mean_baseline_batch
from spatial_gnn.datasets.spatial_dataset import SpatialAgingCellDataset
from spatial_gnn.utils.plot_utils import plot_loss_curves


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
    parser.add_argument("--gene_list", help="Path to file containing list of genes to use (optional)", type=str, default=None)
    parser.add_argument("--debug", help="Enable debug mode with subset of data for quick testing", action="store_true")
    args = parser.parse_args()

    # Load dataset configurations
    DATASET_CONFIGS = load_dataset_config()
    
    # set which model to use
    use_model = "model"

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

    # Load gene list if provided
    gene_list = parse_gene_list(args.gene_list)

    # Build cell type index
    celltypes_to_index = {}
    for ci, cellt in enumerate(dataset_config["celltypes"]):
        celltypes_to_index[cellt] = ci

    if args.dataset is not None:
        exp_name = args.dataset

    # init dataset with settings
    train_dataset = SpatialAgingCellDataset(subfolder_name="train",
                                            dataset_prefix=exp_name,
                                            target="expression",
                                            k_hop=k_hop,
                                            augment_hop=augment_hop,
                                            node_feature=node_feature,
                                            inject_feature=inject_feature,
                                            num_cells_per_ct_id=100,
                                            center_celltypes=center_celltypes,
                                            use_ids=train_ids,
                                            raw_filepaths=[file_path],
                                            gene_list=gene_list,
                                            celltypes_to_index=celltypes_to_index)

    test_dataset = SpatialAgingCellDataset(subfolder_name="test",
                                        dataset_prefix=exp_name,
                                        target="expression",
                                        k_hop=k_hop,
                                        augment_hop=augment_hop,
                                        node_feature=node_feature,
                                        inject_feature=inject_feature,
                                        num_cells_per_ct_id=100,
                                        center_celltypes=center_celltypes,
                                        use_ids=test_ids,
                                        raw_filepaths=[file_path],
                                        gene_list=gene_list,
                                        celltypes_to_index=celltypes_to_index)

    test_dataset.process()
    print("Finished processing test dataset", flush=True)
    train_dataset.process()
    print("Finished processing train dataset", flush=True)
    
    all_test_data = []
    for f in tqdm(test_dataset.processed_file_names):
        batch_list = torch.load(os.path.join(test_dataset.processed_dir, f), weights_only=False)
        all_test_data.extend(batch_list)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=None, persistent_workers=False)
    print(len(test_dataset), flush=True)

    cell_type_model = CellTypeGNN(
        hidden_channels=64,
        input_dim=int(train_dataset.get(0).x.shape[1]),
        num_layers=k_hop,
        method="GIN",
        pool="add",
        celltypes_to_index=celltypes_to_index,
    )

    cell_type_model_path = "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/output/final_exps/expression_with_celltype_decoupled_no_genept_softmax_ct/zeng_expression_2hop_2augment_expression_none/weightedl1_1en04/celltype_model.pth"
    cell_type_model.load_state_dict(torch.load(cell_type_model_path))
    
    model = GNN(
        hidden_channels=64,
        input_dim=int(train_dataset.get(0).x.shape[1]),
        output_dim=len(train_dataset.get(0).y),
        inject_dim=int(train_dataset.get(0).inject.shape[1]) if inject is True else 0,
        method="GIN", 
        pool="add", 
        num_layers=k_hop,
        celltypes_to_index=celltypes_to_index,
        predict_celltype=True,
        train_multitask=False,
        celltype_model=cell_type_model,  # Pass pre-trained model for decoupled training
        ablate_gene_expression=False,
        use_one_hot_ct=False,
        attention_pool=None,
    )
    print(f"Model initialized on {device}")

    gene_names = [gene.upper() for gene in train_dataset.gene_names]
    model_save_dir = "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/output/final_exps/expression_with_celltype_decoupled_no_genept_softmax_ct/zeng_expression_2hop_2augment_expression_none/weightedl1_1en04/"

    model.load_state_dict(torch.load(os.path.join(model_save_dir, f"{use_model}.pth")), strict=False)
    model.to(device)
    print(profile.count_parameters(model), flush=True)

    eval_model(model, test_loader, model_save_dir, device, inject, gene_names)


def eval_model(model, test_loader, save_dir, device="cuda", inject=False, gene_names=None):

    ### LOSS CURVES
    print("Plotting training and validation loss curves...", flush=True)
    plot_loss_curves(save_dir)
    
    ### MODEL PERFORMANCE
    print("Measuring model predictive performance bulk and by cell type...", flush=True)

    model.eval()
    preds = []
    actuals = []
    celltypes = []
    
    for data in tqdm(test_loader):
        data = data.to(device)
        
        # Compute baseline if predicting residuals
        if model.predict_residuals:
            k_hop_baseline = khop_mean_baseline_batch(
                x=data.x,
                batch=data.batch,
                center_nodes=data.center_node,
            )
        
        out = model(
            x=data.x,
            edge_index=data.edge_index,
            batch=data.batch,
            center_cell_idx=data.center_node,
            inject=data.inject if inject else None,
            gene_names=gene_names,
        )

        if model.predict_celltype:
            out, _ = out  # Unpack tuple, use expression output for prediction
        
        # If predicting residuals, add baseline back to get final expression prediction
        if model.predict_residuals:
            out = out + k_hop_baseline
        
        # Move to CPU immediately to free GPU memory
        preds.append(out.detach().cpu())
        
        if data.y.shape != out.shape:
            actuals.append(torch.reshape(data.y.float(), out.shape).detach().cpu())
        else:
            actuals.append(data.y.float().detach().cpu()) # [[512, 300]]

        # get cell type
        celltypes = np.concatenate((celltypes,np.concatenate(data.center_celltype))) # [512]
        
        # Clear GPU memory after each batch
        del out, data
        torch.cuda.empty_cache()
    
    preds = np.concatenate([pred.numpy() for pred in preds]) # [num_cells, num_genes]
    actuals = np.concatenate([act.numpy() for act in actuals]) # [num_cells, num_genes]
    celltypes = np.array(celltypes) # [num_cells,]

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
    
    # Calculate micro and macro averages
    print("Computing micro and macro averages...", flush=True)
        
    # Micro averages: computed over all individual cell-gene pairs
    preds_flat = preds.flatten()
    actuals_flat = actuals.flatten()

    # also compute over non-zero values
    preds_flat_nonzero = preds_flat[actuals_flat != 0]
    actuals_flat_nonzero = actuals_flat[actuals_flat != 0]

    micro_r, _ = pearsonr(preds_flat, actuals_flat)
    micro_s, _ = spearmanr(preds_flat, actuals_flat)
    micro_r2 = r2_score(actuals_flat, preds_flat)
    micro_mae = np.mean(np.abs(preds_flat - actuals_flat))
    micro_rmse = np.sqrt(np.mean((preds_flat - actuals_flat)**2))

    micro_r_nonzero, _ = pearsonr(preds_flat_nonzero, actuals_flat_nonzero)
    micro_s_nonzero, _ = spearmanr(preds_flat_nonzero, actuals_flat_nonzero)
    micro_r2_nonzero = r2_score(actuals_flat_nonzero, preds_flat_nonzero)
    micro_mae_nonzero = np.mean(np.abs(preds_flat_nonzero - actuals_flat_nonzero))
    micro_rmse_nonzero = np.sqrt(np.mean((preds_flat_nonzero - actuals_flat_nonzero)**2))

    # Macro averages: computed over all cell types
    ct_mean_stats_dict = {}
    ct_mean_stats_dict_nonzero = {}

    for ct in np.unique(celltypes):

        ct_mean_stats_dict[ct] = {}
        ct_mean_stats_dict_nonzero[ct] = {}

        cell_r = []
        cell_s = []
        cell_r2 = []
        cell_mae = []
        cell_rmse = []

        cell_r_nonzero = []
        cell_s_nonzero = []
        cell_r2_nonzero = []
        cell_mae_nonzero = []
        cell_rmse_nonzero = []

        for c in np.where(celltypes==ct)[0]:

            if len(preds[c,:]) > 1:
                r, p = pearsonr(preds[c,:], actuals[c,:])
                s, p = spearmanr(preds[c,:], actuals[c,:])
                r2 = r2_score(actuals[c,:], preds[c,:])
                mae = np.mean(np.abs(preds[c,:]-actuals[c,:]))
                rmse = np.sqrt(np.mean((preds[c,:]-actuals[c,:])**2))
            else:
                r = np.nan
                s = np.nan
                r2 = np.nan
                mae = np.nan
                rmse = np.nan
            
            cell_r.append(r)
            cell_s.append(s)
            cell_r2.append(r2)
            cell_mae.append(mae)
            cell_rmse.append(rmse)

            # non-zero values
            if len(preds[c,:][actuals[c,:] != 0]) > 1:
                r_nonzero, p_nonzero = pearsonr(preds[c,:][actuals[c,:] != 0], actuals[c,:][actuals[c,:] != 0])
                s_nonzero, p_nonzero = spearmanr(preds[c,:][actuals[c,:] != 0], actuals[c,:][actuals[c,:] != 0])
                r2_nonzero = r2_score(actuals[c,:][actuals[c,:] != 0], preds[c,:][actuals[c,:] != 0])
                mae_nonzero = np.mean(np.abs(preds[c,:][actuals[c,:] != 0]-actuals[c,:][actuals[c,:] != 0]))
                rmse_nonzero = np.sqrt(np.mean((preds[c,:][actuals[c,:] != 0]-actuals[c,:][actuals[c,:] != 0])**2))
            else:
                r_nonzero = np.nan
                s_nonzero = np.nan
                r2_nonzero = np.nan
                mae_nonzero = np.nan
                rmse_nonzero = np.nan
            
            cell_r_nonzero.append(r_nonzero)
            cell_s_nonzero.append(s_nonzero)
            cell_r2_nonzero.append(r2_nonzero)
            cell_mae_nonzero.append(mae_nonzero)
            cell_rmse_nonzero.append(rmse_nonzero)

        ct_mean_stats_dict[ct]["Cell - Pearson (mean)"] = robust_nanmean(cell_r)
        ct_mean_stats_dict[ct]["Cell - Spearman (mean)"] = robust_nanmean(cell_s)
        ct_mean_stats_dict[ct]["Cell - R2 (mean)"] = robust_nanmean(cell_r2)
        ct_mean_stats_dict[ct]["Cell - MAE (mean)"] = robust_nanmean(cell_mae)
        ct_mean_stats_dict[ct]["Cell - RMSE (mean)"] = robust_nanmean(cell_rmse)

        ct_mean_stats_dict_nonzero[ct]["Cell - Pearson (mean)"] = robust_nanmean(cell_r_nonzero)
        ct_mean_stats_dict_nonzero[ct]["Cell - Spearman (mean)"] = robust_nanmean(cell_s_nonzero)
        ct_mean_stats_dict_nonzero[ct]["Cell - R2 (mean)"] = robust_nanmean(cell_r2_nonzero)
        ct_mean_stats_dict_nonzero[ct]["Cell - MAE (mean)"] = robust_nanmean(cell_mae_nonzero)
        ct_mean_stats_dict_nonzero[ct]["Cell - RMSE (mean)"] = robust_nanmean(cell_rmse_nonzero)

    # get macro average as average over cell type in 
    macro_r = robust_nanmean(np.array([ct_mean_stats_dict[ct]["Cell - Pearson (mean)"] for ct in ct_mean_stats_dict.keys()]))
    macro_s = robust_nanmean(np.array([ct_mean_stats_dict[ct]["Cell - Spearman (mean)"] for ct in ct_mean_stats_dict.keys()]))
    macro_r2 = robust_nanmean(np.array([ct_mean_stats_dict[ct]["Cell - R2 (mean)"] for ct in ct_mean_stats_dict.keys()]))
    macro_mae = robust_nanmean(np.array([ct_mean_stats_dict[ct]["Cell - MAE (mean)"] for ct in ct_mean_stats_dict.keys()]))
    macro_rmse = robust_nanmean(np.array([ct_mean_stats_dict[ct]["Cell - RMSE (mean)"] for ct in ct_mean_stats_dict.keys()]))

    macro_r_nonzero = robust_nanmean(np.array([ct_mean_stats_dict_nonzero[ct]["Cell - Pearson (mean)"] for ct in ct_mean_stats_dict_nonzero.keys()]))
    macro_s_nonzero = robust_nanmean(np.array([ct_mean_stats_dict_nonzero[ct]["Cell - Spearman (mean)"] for ct in ct_mean_stats_dict_nonzero.keys()]))
    macro_r2_nonzero = robust_nanmean(np.array([ct_mean_stats_dict_nonzero[ct]["Cell - R2 (mean)"] for ct in ct_mean_stats_dict_nonzero.keys()]))
    macro_mae_nonzero = robust_nanmean(np.array([ct_mean_stats_dict_nonzero[ct]["Cell - MAE (mean)"] for ct in ct_mean_stats_dict_nonzero.keys()]))
    macro_rmse_nonzero = robust_nanmean(np.array([ct_mean_stats_dict_nonzero[ct]["Cell - RMSE (mean)"] for ct in ct_mean_stats_dict_nonzero.keys()]))

    # save macro and micro averages in one dataframe
    overall_stats_dict = {
        "Macro - Pearson": macro_r,
        "Macro - Spearman": macro_s,
        "Macro - R2": macro_r2,
        "Macro - MAE": macro_mae,
        "Macro - RMSE": macro_rmse,
        "Macro (Nonzero) - Pearson": macro_r_nonzero,
        "Macro (Nonzero) - Spearman": macro_s_nonzero,
        "Macro (Nonzero) - R2": macro_r2_nonzero,
        "Macro (Nonzero) - MAE": macro_mae_nonzero,
        "Macro (Nonzero) - RMSE": macro_rmse_nonzero,
        "Micro - Pearson": micro_r,
        "Micro - Spearman": micro_s,
        "Micro - R2": micro_r2,
        "Micro - MAE": micro_mae,
        "Micro - RMSE": micro_rmse,
        "Micro (Nonzero) - Pearson": micro_r_nonzero,
        "Micro (Nonzero) - Spearman": micro_s_nonzero,
        "Micro (Nonzero) - R2": micro_r2_nonzero,
        "Micro (Nonzero) - MAE": micro_mae_nonzero,
        "Micro (Nonzero) - RMSE": micro_rmse_nonzero,
    }
    overall_stats_dict = pd.DataFrame(
        list(overall_stats_dict.items()),
        columns=["Metric", "Value"]
    )
    overall_stats_dict.to_csv(os.path.join(save_dir, "test_evaluation_stats_macro_micro.csv"), index=False)

    # save the dictionary for each cell type
    with open(os.path.join(save_dir, "test_evaluation_stats_bycelltype.pkl"), 'wb') as f:
        pickle.dump(ct_mean_stats_dict, f)
    with open(os.path.join(save_dir, "test_evaluation_stats_bycelltype_nonzero_genes.pkl"), 'wb') as f:
        pickle.dump(ct_mean_stats_dict_nonzero, f)        
    print("Finished cell type analysis.", flush=True)
    

def robust_nanmean(x):
    nmx = np.nanmean(x) if np.count_nonzero(~np.isnan(x))>1 else np.mean(x)
    return (nmx)

def robust_nanmedian(x):
    nmx = np.nanmedian(x) if np.count_nonzero(~np.isnan(x))>1 else np.median(x)
    return (nmx)


if __name__ == "__main__":
    main()
