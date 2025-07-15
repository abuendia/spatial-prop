import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata as ad
from scipy.stats import pearsonr, spearmanr, ttest_ind
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.set_style("ticks")
from sklearn.neighbors import BallTree
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
from decimal import Decimal
import copy
import json 

import random

#os.chdir("/oak/stanford/groups/jamesz/esun/GNNPerturbation/spatial-gnn/scripts")

from aging_gnn_model import *

import networkx as nx
import argparse
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph, one_hot, to_networkx
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool, global_add_pool, global_max_pool
from torch.nn.modules.loss import _Loss
from torch.distributions import MultivariateNormal as MVN

from sklearn.metrics import r2_score


### Load in dataset configs

def load_dataset_config():
    """Load dataset configurations from JSON file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'datasets.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset configuration file not found at {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in dataset configuration file at {config_path}")

def main():
    # set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset to use (aging_coronal, aging_sagittal, exercise, reprogramming, allen, kukanja, pilot)", type=str, required=True)
    parser.add_argument("--base_path", help="Base path to the data directory", type=str, required=True)
    parser.add_argument("--k_hop", help="k-hop neighborhood size", type=int, required=True)
    parser.add_argument("--augment_hop", help="number of hops to take for graph augmentation", type=int, required=True)
    parser.add_argument("--center_celltypes", help="cell type labels to center graphs on, separated by comma. Use 'all' for all cell types or 'none' for no cell type filtering", type=str, required=True)
    parser.add_argument("--node_feature", help="node features key, e.g. 'celltype_age_region'", type=str, required=True)
    parser.add_argument("--inject_feature", help="inject features key, e.g. 'center_celltype'", type=str, required=True)
    parser.add_argument("--learning_rate", help="learning rate", type=float, required=True)
    parser.add_argument("--loss", help="loss: balanced_mse, npcc, mse, l1", type=str, required=True)
    parser.add_argument("--epochs", help="number of epochs", type=int, required=True)
    parser.add_argument("--gene_list", help="Path to file containing list of genes to use (optional)", type=str, default=None)
    args = parser.parse_args()

    # Load dataset configurations
    DATASET_CONFIGS = load_dataset_config()
    
    # set which model to use
    use_model = "model" # "best_model"

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
    if args.center_celltypes.lower() == 'none':
        center_celltypes = None
    elif args.center_celltypes.lower() == 'all':
        center_celltypes = 'all'
    else:
        center_celltypes = args.center_celltypes.split(",")
    
    node_feature = args.node_feature
    inject_feature = args.inject_feature
    learning_rate = args.learning_rate
    loss = args.loss
    epochs = args.epochs

    if inject_feature.lower() == "none":
        inject_feature = None
        inject=False
    else:
        inject=True

    # determine gpu / cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)

    # train
    exp_name = f"{k_hop}hop_{augment_hop}augment_{node_feature}_{inject_feature}_{learning_rate:.0e}lr_{loss}_{epochs}epochs"

    # Load gene list if provided
    gene_list = None
    if args.gene_list is not None:
        try:
            with open(args.gene_list, 'r') as f:
                gene_list = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Gene list file not found at {args.gene_list}")

    # Build cell type index
    celltypes_to_index = {}
    for ci, cellt in enumerate(dataset_config["celltypes"]):
        celltypes_to_index[cellt] = ci
    
    
    # init dataset with settings
    train_dataset = SpatialAgingCellDataset(subfolder_name="train",
                                            dataset_prefix=args.dataset,
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
                                        dataset_prefix=args.dataset,
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

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=None, persistent_workers=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=None, persistent_workers=False)

    print(len(train_dataset), flush=True)
    print(len(test_dataset), flush=True)


    # init GNN model
    if inject is True:
        model = GNN(hidden_channels=64,
                    input_dim=int(train_dataset.get(0).x.shape[1]),
                    output_dim=len(train_dataset.get(0).y), # added for multivariate targets
                    inject_dim=int(train_dataset.get(0).inject.shape[1]), # added for injecting features into last layer (after pooling),
                    method="GIN", pool="add", num_layers=k_hop)
    else:
        model = GNN(hidden_channels=64,
                    input_dim=int(train_dataset.get(0).x.shape[1]),
                    output_dim=len(train_dataset.get(0).y), # added for multivariate targets
                    method="GIN", pool="add", num_layers=k_hop)
    model.to(device)
    print(device, flush=True)

    # create directory to save results
    model_dirname = loss+f"_{learning_rate:.0e}".replace("-","n")
    save_dir = os.path.join("results/gnn",train_dataset.processed_dir.split("/")[-2],model_dirname)
    
    ### RESULTS / VISUALIZATION ###
    
    # load model weights
    model.load_state_dict(torch.load(os.path.join(save_dir, f"{use_model}.pth"),
                                    map_location=torch.device('cpu')))
    from torch_geometric import profile
    print(profile.count_parameters(model), flush=True)
    
    ### LOSS CURVES
    print("Plotting training and validation loss curves...", flush=True)
    
    with open(os.path.join(save_dir, "training.pkl"), 'rb') as handle:
        b = pickle.load(handle)
    
    best_idx = np.argmin(b['test'])
    
    plt.figure(figsize=(4,4))
    plt.plot(b['epoch'],b['train'],label='Train',color='0.2',zorder=0)
    plt.plot(b['epoch'],b['test'],label='Test',color='green',zorder=1)
    plt.scatter(b['epoch'][best_idx],b['test'][best_idx],s=50,c='green',marker="D",zorder=2,label="Selected Model")
    plt.ylabel("Weighted L1 Loss", fontsize=16)
    plt.xlabel("Training Epochs", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    #plt.savefig(f"plots/gnn/{test_dataset.processed_dir.split('/')[-2]}_losscurves.pdf", bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "loss_curves.pdf"), bbox_inches='tight')
    plt.close()
    
    print("Finished plots.", flush=True)

    ### MODEL PERFORMANCE
    print("Measuring model predictive performance bulk and by cell type...", flush=True)
    
    model.eval()
    
    preds = []
    actuals = []
    celltypes = []
    
    for data in test_loader:
        if inject is False:
            out = model(data.x, data.edge_index, data.batch, None)
        else:
            out = model(data.x, data.edge_index, data.batch, data.inject)
        preds.append(out)
        
        if data.y.shape != out.shape:
            actuals.append(torch.reshape(data.y.float(), out.shape))
        else:
            actuals.append(data.y.float())
        
        # get cell type
        celltypes = np.concatenate((celltypes,np.concatenate(data.center_celltype)))
    
    preds = np.concatenate([pred.detach().numpy() for pred in preds])
    actuals = np.concatenate([act.detach().numpy() for act in actuals])
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

            r, p = pearsonr(preds[celltypes==ct,g], actuals[celltypes==ct,g])
            gene_r.append(r)

            s, p = spearmanr(preds[celltypes==ct,g], actuals[celltypes==ct,g])
            gene_s.append(s)

            r2 = r2_score(actuals[celltypes==ct,g], preds[celltypes==ct,g])
            gene_r2.append(r2)

            gene_mae.append(np.mean(np.abs(preds[celltypes==ct,g]-actuals[celltypes==ct,g])))
            gene_rmse.append(np.sqrt(np.mean((preds[celltypes==ct,g]-actuals[celltypes==ct,g])**2)))


        # cell stats
        cell_r = []
        cell_s = []
        cell_r2 = []
        cell_mae = []
        cell_rmse = []

        for c in np.where(celltypes==ct)[0]:

            r, p = pearsonr(preds[c,:], actuals[c,:])
            cell_r.append(r)

            s, p = spearmanr(preds[c,:], actuals[c,:])
            cell_s.append(s)

            r2 = r2_score(actuals[c,:], preds[c,:])
            cell_r2.append(r2)

            cell_mae.append(np.mean(np.abs(preds[c,:]-actuals[c,:])))
            cell_rmse.append(np.sqrt(np.mean((preds[c,:]-actuals[c,:])**2)))

            pred_ct = celltypes==ct
            
        # add results to dictionary
        ct_stats_dict[ct]["Gene - Pearson (mean)"] = np.mean(gene_r)
        ct_stats_dict[ct]["Gene - Pearson (median)"] = np.median(gene_r)
        ct_stats_dict[ct]["Gene - Spearman (mean)"] = np.mean(gene_s)
        ct_stats_dict[ct]["Gene - Spearman (median)"] = np.mean(gene_s)
        ct_stats_dict[ct]["Gene - R2 (mean)"] = np.mean(gene_r2)
        ct_stats_dict[ct]["Gene - R2 (median)"] = np.mean(gene_r2)
        ct_stats_dict[ct]["Gene - MAE (mean)"] = np.mean(gene_mae)
        ct_stats_dict[ct]["Gene - RMSE (mean)"] = np.mean(gene_rmse)
        
        ct_stats_dict[ct]["Cell - Pearson (mean)"] = np.mean(cell_r)
        ct_stats_dict[ct]["Cell - Pearson (median)"] = np.median(cell_r)
        ct_stats_dict[ct]["Cell - Spearman (mean)"] = np.mean(cell_s)
        ct_stats_dict[ct]["Cell - Spearman (median)"] = np.mean(cell_s)
        ct_stats_dict[ct]["Cell - R2 (mean)"] = np.mean(cell_r2)
        ct_stats_dict[ct]["Cell - R2 (median)"] = np.median(cell_r2)
        ct_stats_dict[ct]["Cell - MAE (mean)"] = np.mean(cell_mae)
        ct_stats_dict[ct]["Cell - RMSE (mean)"] = np.mean(cell_rmse)
    
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
    #plt.savefig("plots/expression_prediction_performance/"+save_dir.split("/")[-2]+"_GENE.pdf", bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "prediction_performance_GENE.pdf"))
    plt.close()
    
    
    print("Finished cell type analysis.", flush=True)
    
    print("DONE.", flush=True)

if __name__ == "__main__":
    main()