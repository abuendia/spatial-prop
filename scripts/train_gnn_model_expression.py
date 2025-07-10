import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata as ad
from scipy.stats import pearsonr, spearmanr, ttest_ind
import pickle
import copy
import os
import json
from sklearn.neighbors import BallTree
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
from decimal import Decimal
import random
import networkx as nx
import argparse

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph, one_hot, to_networkx
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool, global_add_pool, global_max_pool
from torch.nn.modules.loss import _Loss
import wandb    

from aging_gnn_model import *

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

    # Validate dataset choice
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Dataset must be one of: {', '.join(DATASET_CONFIGS.keys())}")

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
    run = wandb.init(
        project="spatial-gnn",
        name=exp_name,
    )

    # Load gene list if provided
    gene_list = None
    if args.gene_list is not None:
        try:
            with open(args.gene_list, 'r') as f:
                gene_list = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Gene list file not found at {args.gene_list}")

    # init dataset with settings
    raw_filepaths = [file_path]

    train_dataset = SpatialAgingCellDataset(subfolder_name="train",
                                            target="expression",
                                            k_hop=k_hop,
                                            augment_hop=augment_hop,
                                            node_feature=node_feature,
                                            inject_feature=inject_feature,
                                            num_cells_per_ct_id=100,
                                            center_celltypes=center_celltypes,
                                            use_ids=train_ids,
                                            raw_filepaths=raw_filepaths,
                                            gene_list=gene_list)

    test_dataset = SpatialAgingCellDataset(subfolder_name="test",
                                        target="expression",
                                        k_hop=k_hop,
                                        augment_hop=augment_hop,
                                        node_feature=node_feature,
                                        inject_feature=inject_feature,
                                        num_cells_per_ct_id=100,
                                        center_celltypes=center_celltypes,
                                        use_ids=test_ids,
                                        raw_filepaths=raw_filepaths,
                                        gene_list=gene_list)
                                            
    test_dataset.process()
    print("Finished processing test dataset", flush=True)
    train_dataset.process()
    print("Finished processing train dataset", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=None, persistent_workers=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=None, persistent_workers=False) # shuffle=True to reduce bias in batch-wise metric estimates

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
    print(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # get loss
    if loss == "mse":
        criterion = torch.nn.MSELoss()
    elif loss == "l1":
        criterion = torch.nn.L1Loss()
    elif loss == "weightedl1":
        criterion = WeightedL1Loss(zero_weight=1, nonzero_weight=10)
    elif loss == "balanced_mse":
        criterion = BMCLoss(0.1) # init noise_sigma
        optimizer.add_param_group({'params': criterion.noise_sigma, 'lr':learning_rate, 'name': 'noise_sigma'})
    elif loss == "npcc":
        criterion = Neg_Pearson_Loss()
    else:
        raise Exception ("loss is not recognized!")

    # train model
    training_results = {"metric":loss, "epoch":[], "train":[], "test":[]}
    best_mse = np.inf

    # create directory to save results
    model_dirname = loss+f"_{learning_rate:.0e}".replace("-","n")
    save_dir = os.path.join("results/gnn",train_dataset.processed_dir.split("/")[-2],model_dirname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(1, epochs):

        train(model, train_loader, criterion, optimizer, inject=inject, device=device)
        train_mse = test(model, train_loader, loss, criterion, inject=inject, device=device)
        test_mse = test(model, test_loader, loss, criterion, inject=inject, device=device)
        
        if test_mse < best_mse: # if model improved then save
            # save best model
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            #print("Saved best model")
            #best_model = copy.deepcopy(model)
            best_mse = test_mse
        
        if loss == "mse":
            print(f'Epoch: {epoch:03d}, Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}', flush=True)
        elif loss == "l1":
            print(f'Epoch: {epoch:03d}, Train L1: {train_mse:.4f}, Test L1: {test_mse:.4f}', flush=True)
        elif loss == "weightedl1":
            print(f'Epoch: {epoch:03d}, Train WL1: {train_mse:.4f}, Test WL1: {test_mse:.4f}', flush=True)
        elif loss == "balanced_mse":
            print(f'Epoch: {epoch:03d}, Train BMC: {train_mse:.4f}, Test BMC: {test_mse:.4f}', flush=True)
        elif loss == "npcc":
            print(f'Epoch: {epoch:03d}, Train NPCC: {train_mse:.4f}, Test NPCC: {test_mse:.4f}', flush=True)
            
        training_results["epoch"].append(epoch)
        training_results["train"].append(train_mse)    
        training_results["test"].append(test_mse)

        # log to wandb
        wandb.log({"train_loss":train_mse, "test_loss":test_mse})

    # save final model
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
    print("Saved final model")

    # save results
    with open(os.path.join(save_dir, "training.pkl"), 'wb') as f:
        pickle.dump(training_results, f)
    print("Saved training logs")


if __name__ == "__main__":
    main()
