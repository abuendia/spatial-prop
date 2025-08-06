use_wandb = False

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata as ad
from scipy.stats import pearsonr, spearmanr, ttest_ind
import pickle
import copy
import os
from sklearn.neighbors import BallTree
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
from decimal import Decimal
import random
import networkx as nx
import argparse
from collections import defaultdict
from sklearn.metrics import r2_score

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph, one_hot, to_networkx
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool, global_add_pool, global_max_pool
from torch.nn.modules.loss import _Loss
if use_wandb is True:
    import wandb    

from aging_gnn_model import *


def evaluate(model, test_loader, device="cuda"):
    """
    Evaluates the model on a held-out test set by computing
    Pearson and Spearman correlation for each cell type.
    """
    model.eval()  

    results = defaultdict(lambda: {"preds": [], "targets": []})

    with torch.no_grad():  
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch, getattr(data, "inject", None))

            preds = out.cpu().numpy().flatten()
            targets = data.y.cpu().numpy().flatten()
            cell_types = [item[0] for item in data.center_celltype] # center cell type

            for i, cell_type in enumerate(cell_types):
                results[cell_type]["preds"].append(preds[i])
                results[cell_type]["targets"].append(targets[i])

    correlation_results = {}
    for cell_type, values in results.items():
        preds = np.array(values["preds"])
        targets = np.array(values["targets"])

        if len(preds) > 1:  # Avoid computing correlation on a single value
            pearson_corr, _ = pearsonr(preds, targets)
            spearman_corr, _ = spearmanr(preds, targets)
            r2_corr = r2_score(targets, preds)

        else:
            pearson_corr, spearman_corr, r2_corr = np.nan, np.nan, np.nan

        correlation_results[cell_type] = {
            "pearson": pearson_corr,
            "spearman": spearman_corr,
            "r2": r2_corr,
        }

    # Print results
    print("\nEvaluation Results (Correlation by Cell Type):")
    for cell_type, metrics in correlation_results.items():
        print(f"{cell_type}: Pearson={metrics['Pearson']:.4f}, Spearman={metrics['Spearman']:.4f}, R2={metrics['R2']:.4f}")
        if use_wandb is True:
            wandb.log({f"barplot_{cell_type}": wandb.plots.bar({"Pearson": metrics['Pearson'], "Spearman": metrics['Spearman'], "R2": metrics['R2']})})

    return correlation_results  # Return results as a dictionary for further analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("k_hop", help="k-hop neighborhood size", type=int)
    parser.add_argument("augment_hop", help="number of hops to take for graph augmentation", type=int)
    parser.add_argument("center_celltypes", help="cell type labels to center graphs on, separated by comma", type=str)
    parser.add_argument("node_feature", help="node features key, e.g. 'celltype_age_region'", type=str)
    parser.add_argument("inject_feature", help="inject features key, e.g. 'center_celltype'", type=str)
    args = parser.parse_args()

    # load parameters from arguments
    k_hop = args.k_hop
    augment_hop = args.augment_hop
    center_celltypes = args.center_celltypes.split(",")
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

    # train
    train_ids = [
        ['1','101','14','19','30','38','42','46','53','61','7','70','75','80','86','97'], # aging coronal
        # ['2','34','39','62','68'], # aging sagittal
        ["YC1","YC2","YC3","OE1","OE2","OE3","OC1","OC2","OC3"], # exercise
        ["YC1982","YC1990","YC1975","OT902","OT1160","OT1084","OC903","OC1226","OC1083"], # reprogramming
        # ['MsBrainAgingSpatialDonor_10_0', 'MsBrainAgingSpatialDonor_10_1', 'MsBrainAgingSpatialDonor_10_2', 'MsBrainAgingSpatialDonor_11_0', 'MsBrainAgingSpatialDonor_11_1', 'MsBrainAgingSpatialDonor_11_2', 'MsBrainAgingSpatialDonor_12_0', 'MsBrainAgingSpatialDonor_12_1', 'MsBrainAgingSpatialDonor_13_1', 'MsBrainAgingSpatialDonor_13_2', 'MsBrainAgingSpatialDonor_14_1', 'MsBrainAgingSpatialDonor_15_0', 'MsBrainAgingSpatialDonor_15_1', 'MsBrainAgingSpatialDonor_16_0', 'MsBrainAgingSpatialDonor_16_1', 'MsBrainAgingSpatialDonor_17_0', 'MsBrainAgingSpatialDonor_17_1', 'MsBrainAgingSpatialDonor_18_0', 'MsBrainAgingSpatialDonor_18_1', 'MsBrainAgingSpatialDonor_19_0', 'MsBrainAgingSpatialDonor_19_1', 'MsBrainAgingSpatialDonor_19_2', 'MsBrainAgingSpatialDonor_2_0', 'MsBrainAgingSpatialDonor_2_1', 'MsBrainAgingSpatialDonor_3_0', 'MsBrainAgingSpatialDonor_3_1', 'MsBrainAgingSpatialDonor_4_0', 'MsBrainAgingSpatialDonor_4_1', 'MsBrainAgingSpatialDonor_4_2', 'MsBrainAgingSpatialDonor_5_0', 'MsBrainAgingSpatialDonor_5_1', 'MsBrainAgingSpatialDonor_5_2', 'MsBrainAgingSpatialDonor_6_0', 'MsBrainAgingSpatialDonor_6_1', 'MsBrainAgingSpatialDonor_6_2', 'MsBrainAgingSpatialDonor_7_0', 'MsBrainAgingSpatialDonor_7_1', 'MsBrainAgingSpatialDonor_7_2', 'MsBrainAgingSpatialDonor_8_0', 'MsBrainAgingSpatialDonor_8_1', 'MsBrainAgingSpatialDonor_8_2', 'MsBrainAgingSpatialDonor_9_1', 'MsBrainAgingSpatialDonor_9_2'], # allen
        # None, # androvic
        # ['CNTRL_PEAK_B_R2', 'CNTRL_PEAK_B_R3', 'CNTRL_PEAK_B_R4', 'EAE_PEAK_B_R2', 'EAE_PEAK_B_R3', 'EAE_PEAK_B_R4'], # kukanja
        # ['Middle1', 'Old1', 'Old2', 'Young1', 'Young2'], # pilot
    ]

    # test
    test_ids = [
        ["11","33","57","93"], # aging coronal
        # ['81'], # aging sagittal
        ["YC4","OE4","OC4"], # exercise
        ["YC1989","OT1125","OC1138"], # reprogramming
        # ["MsBrainAgingSpatialDonor_13_0","MsBrainAgingSpatialDonor_9_0","MsBrainAgingSpatialDonor_14_0","MsBrainAgingSpatialDonor_1_0"], # allen
        # [], # androvic
        # ['CNTRL_PEAK_B_R1', 'EAE_PEAK_B_R1'], # kukanja
        # ["Middle2"], # pilot
    ]

    exp_name = f"eval_{k_hop}hop_{augment_hop}augment_{node_feature}_{inject_feature}"
    if use_wandb is True:
        run = wandb.init(
            project="spatial-gnn",
            name=exp_name,
        )

    test_dataset = SpatialAgingCellDataset(subfolder_name="test",
                                        target="expression",
                                        k_hop=k_hop,
                                        augment_hop=augment_hop,
                                        node_feature=node_feature,
                                        inject_feature=inject_feature,
                                        num_cells_per_ct_id=100,
                                        center_celltypes=center_celltypes,
                                        use_ids=test_ids)  
    test_dataset.process()
    print("Finished processing test dataset", flush=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=None, persistent_workers=False) # shuffle=True to reduce bias in batch-wise metric estimates

    # init GNN model
    if inject is True:
        model = GNN(hidden_channels=64,
                    input_dim=int(test_dataset.get(0).x.shape[1]),
                    output_dim=len(test_dataset.get(0).y), # added for multivariate targets
                    inject_dim=int(test_dataset.get(0).inject.shape[1]), # added for injecting features into last layer (after pooling),
                    method="GIN", pool="add", num_layers=k_hop)
    else:
        model = GNN(hidden_channels=64,
                    input_dim=int(test_dataset.get(0).x.shape[1]),
                    output_dim=len(test_dataset.get(0).y), # added for multivariate targets
                    method="GIN", pool="add", num_layers=k_hop)
    model.to(device)

    # load model from checkpoint
    best_ckpt = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/repos/spatial-gnn/results/gnn/expression_100per_2hop_2C0aug_200delaunay_expressionFeat_TNP_NoneInject/weightedl1_1en04/best_model.pth"
    model.load_state_dict(torch.load(best_ckpt))
    evaluate(model, test_loader, device=device)

if __name__=="__main__":
    main()
