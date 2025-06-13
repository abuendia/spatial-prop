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


def main():
    # set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("k_hop", help="k-hop neighborhood size", type=int)
    parser.add_argument("augment_hop", help="number of hops to take for graph augmentation", type=int)
    parser.add_argument("center_celltypes", help="cell type labels to center graphs on, separated by comma", type=str)
    parser.add_argument("node_feature", help="node features key, e.g. 'celltype_age_region'", type=str)
    parser.add_argument("inject_feature", help="inject features key, e.g. 'center_celltype'", type=str)
    parser.add_argument("learning_rate", help="learning rate", type=float)
    parser.add_argument("loss", help="loss: balanced_mse, npcc, mse, l1", type=str)
    parser.add_argument("epochs", help="number of epochs", type=int)
    args = parser.parse_args()

    # load parameters from arguments
    k_hop = args.k_hop
    augment_hop = args.augment_hop
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

    exp_name = f"{k_hop}hop_{augment_hop}augment_{node_feature}_{inject_feature}_{learning_rate:.0e}lr_{loss}_{epochs}epochs"
    run = wandb.init(
        project="spatial-gnn",
        name=exp_name,
    )

    # init dataset with settings
    train_dataset = SpatialAgingCellDataset(subfolder_name="train",
                                            target="expression",
                                            k_hop=k_hop,
                                            augment_hop=augment_hop,
                                            node_feature=node_feature,
                                            inject_feature=inject_feature,
                                            num_cells_per_ct_id=100,
                                            center_celltypes=center_celltypes,
                                    use_ids=train_ids)

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
