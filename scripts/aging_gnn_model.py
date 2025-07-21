# import key packages and libraries
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata as ad
from scipy.stats import pearsonr, spearmanr, ttest_ind
import pickle
import os
from sklearn.neighbors import BallTree

from scipy.stats import mannwhitneyu, ttest_ind
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests
from decimal import Decimal

import random

from ageaccel_proximity import *

import networkx as nx

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph, one_hot, to_networkx
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool, global_add_pool, global_max_pool
from torch.nn.modules.loss import _Loss
from torch.distributions import MultivariateNormal as MVN


# Dataset Class

class SpatialAgingCellDataset(Dataset):
    '''
    Class for building spatial cell subgraphs from the MERFISH anndata file
        - Nodes are cells and featurized by one-hot encoding of cell type
        - Edges are trimmed Delaunay spatial graph connections
        - Graphs are k-hop neighborhoods around cells and labeled by average peripheral age acceleration
    
    Relies on build_spatial_graph() from ageaccel_proximity.py
        Use `from ageaccel_proximity import *` when importing libraries
        
    Arguments:
        root [str] - root directory path
        transform [None] - not implemented
        pre_transform [None] - not implemented
        normalize_total [bool] - whether or not to normalize gene expression data to total (don't use for scaled expression inputs)
        raw_filepaths [lst of str] - list of paths to anndata .h5ad files of spatial transcriptomics data
        gene_list [str or None] - path to file containing list of genes to use, or None to compute from AnnData
        processed_folder_name [str] - path to save processed data files
        subfolder_name [str] - name of subfolder to save in (e.g. "train")
        target [str] - name of target label to use ("aging", "age", "num_neuron")
        node_feature [str] - how to featurize nodes ("celltype", "celltype_age", "celltype_region", "celltype_age_region", "expression", "celltype_expression")
        inject_feature [None or str] - what features to inject as last layer of network for prediction ("center_celltype")
        sub_id [str] - key in adata.obs to separate graphs by id
        use_ids [lst of lst of str, None] - list of list of sub_id values to use for each anndata dataset; if None, then uses all data
        center_celltypes [str or lst] - 'all' to use all cell types, otherwise list of cell types to draw subgraphs from
        num_cells_per_ct_id [str] - number of cells per cell type per id to take
        k_hop [int] - k-hop neighborhood subgraphs to take
        augment_hop [int] - k-hop neighbors to also take induced subgraphs from (augments number of graphs)
        augment_cutoff ['auto', 0 <= float < 1] - quantile cutoff in absolute value of label to perform augmentation (to balance labels)
        dispersion_factor [0 <= float < 1] - factor for dispersion of augmentation sampling of rare graph labels; higher means more rare samples
        radius_cutoff [float] - radius cutoff for Delaunay triangulation edges
        celltypes_to_index [dict] - dictionary mapping cell type labels to integer index
    '''
    def __init__(self, 
                 root=".",
                 dataset_prefix="",
                 transform=None, 
                 pre_transform=None,
                 normalize_total=True,
                 raw_filepaths=None,
                 gene_list=None,
                 processed_folder_name="data/gnn_datasets",
                 subfolder_name=None,
                 target="expression",
                 node_feature="celltype",
                 inject_feature=None,
                 sub_id="mouse_id",
                 use_ids=None,
                 center_celltypes='all',
                 num_cells_per_ct_id=1,
                 k_hop=2,
                 augment_hop=0,
                 augment_cutoff=0,
                 dispersion_factor=0,
                 radius_cutoff=200,
                 celltypes_to_index = {
                                     'Neuron-Excitatory' : 0,
                                     'Neuron-Inhibitory' : 1,
                                     'Neuron-MSN' : 2, 
                                     'Astrocyte' : 3, 
                                     'Microglia' : 4, 
                                     'Oligodendrocyte' : 5, 
                                     'OPC' : 6,
                                     'Endothelial' : 7, 
                                     'Pericyte' : 8, 
                                     'VSMC' : 9, 
                                     'VLMC' : 10,
                                     'Ependymal' : 11, 
                                     'Neuroblast' : 12, 
                                     'NSC' : 13,  
                                     'Macrophage' : 14, 
                                     'Neutrophil' : 15,
                                     'T cell' : 16, 
                                     'B cell' : 17,
                                    },
                ):
    
        self.root=root
        self.dataset_prefix=dataset_prefix
        self.transform=transform
        self.pre_transform=pre_transform
        self.normalize_total=normalize_total
        self.raw_filepaths=raw_filepaths
        self.gene_list = gene_list
        self.processed_folder_name=processed_folder_name
        self.subfolder_name=subfolder_name
        self.target=target
        self.node_feature=node_feature
        self.inject_feature=inject_feature
        self.sub_id=sub_id
        self.use_ids=use_ids
        self.center_celltypes=center_celltypes
        self.num_cells_per_ct_id=num_cells_per_ct_id
        self.k_hop=k_hop
        self.augment_hop=augment_hop
        self.augment_cutoff=augment_cutoff
        self.dispersion_factor=dispersion_factor
        self.radius_cutoff=radius_cutoff
        self.celltypes_to_index=celltypes_to_index
        self._indices = None

    def indices(self):
        return range(self.len()) if self._indices is None else self._indices
    
    @property
    def processed_dir(self) -> str:
        if self.augment_cutoff == 'auto':
            aug_key = self.augment_cutoff
        else:
            aug_key = int(self.augment_cutoff*100)
        celltype_firstletters = "".join([x[0] for x in self.center_celltypes])
        data_dir = f"{self.dataset_prefix}_{self.target}_{self.num_cells_per_ct_id}per_{self.k_hop}hop_{self.augment_hop}C{aug_key}aug_{self.radius_cutoff}delaunay_{self.node_feature}Feat_{celltype_firstletters}_{self.inject_feature}Inject"
        if self.subfolder_name is not None:
            return os.path.join(self.root, self.processed_folder_name, data_dir, self.subfolder_name)
        else:
            return os.path.join(self.root, self.processed_folder_name, data_dir)

    @property
    def processed_file_names(self):
        return sorted([f for f in os.listdir(self.processed_dir) if f.endswith('.pt')])
    
    def process(self):
        
        # Create / overwrite directory
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        else:
            print ("Dataset already exists!")
            return()
        
        # define augmentation
        if self.augment_cutoff == 'auto':
            aug_key = self.augment_cutoff
        else:
            aug_key = int(self.augment_cutoff*100)
            
        # read in genes
        if self.gene_list is not None:
            # Load gene list from file
            gene_names = np.genfromtxt(self.gene_list, dtype='unicode')
        else:
            # Compute gene list from first AnnData file
            adata = sc.read_h5ad(self.raw_filepaths[0])
            gene_names = adata.var_names.values
        
        for rfi, raw_filepath in enumerate(self.raw_filepaths):
            # load raw data
            adata = sc.read_h5ad(raw_filepath)
            if issparse(adata.X):
                adata.X = adata.X.toarray()
            
            # filter to known cell type keys
            adata = adata[adata.obs.celltype.isin(self.celltypes_to_index.keys())]
            
            # normalize by total genes
            if self.normalize_total is True:
                print("Normalized data")
                sc.pp.normalize_total(adata, target_sum=adata.shape[1])
            
            # handle missing genes (-1 token, indicators added later)
            missing_genes = [gene for gene in gene_names if gene not in adata.var_names]
            missing_X = -np.ones((adata.shape[0],len(missing_genes)))
            orig_obs_names = adata.obs_names.copy()
            orig_var_names = adata.var_names.copy()
            adata = ad.AnnData(X = np.concatenate((adata.X, missing_X), axis=1),
                               obs = adata.obs,
                               obsm = adata.obsm)
            adata.obs_names = orig_obs_names
            adata.var_names = np.concatenate((orig_var_names, missing_genes))
            
            # order by gene_names
            adata = adata[:, gene_names]
            
            # make and save subgraphs
            subgraph_count = 0
            
            if self.use_ids is None:
                sub_ids_arr = np.unique(adata.obs[self.sub_id])
            elif self.use_ids[rfi] is None:
                sub_ids_arr = np.unique(adata.obs[self.sub_id])
            elif isinstance(self.use_ids[rfi], list):
                sub_ids_arr = np.intersect1d(np.unique(adata.obs[self.sub_id]), np.array(self.use_ids[rfi]))
            else:
                sub_ids_arr = np.intersect1d(np.unique(adata.obs[self.sub_id]), np.array(self.use_ids))
            
            for sid in sub_ids_arr:

                # subset to each sample
                sub_adata = adata[(adata.obs[self.sub_id]==sid)]

                # Delaunay triangulation with pruning of > 200um distances
                build_spatial_graph(sub_adata, method="delaunay")
                sub_adata.obsp['spatial_connectivities'][sub_adata.obsp['spatial_distances']>self.radius_cutoff] = 0
                sub_adata.obsp['spatial_distances'][sub_adata.obsp['spatial_distances']>self.radius_cutoff] = 0

                # convert graphs to PyG format
                edge_index, edge_att = from_scipy_sparse_matrix(sub_adata.obsp['spatial_connectivities'])
                
                ### Construct Node Labels
                if self.node_feature not in ["celltype", "expression", "celltype_expression", "gaussian"]:
                    raise Exception (f"'node_feature' value of {self.node_feature} not recognized")
                
                if "celltype" in self.node_feature:
                    # get cell type one hot encoding
                    node_labels = torch.tensor([self.celltypes_to_index[x] for x in sub_adata.obs["celltype"]])
                    node_labels = one_hot(node_labels, num_classes=len(self.celltypes_to_index.keys()))
                
                if "expression" in self.node_feature:
                    # get spatial expression
                    if self.node_feature == "expression":
                        node_labels = torch.tensor(sub_adata.X).float()
                    else:
                        node_labels = torch.cat((node_labels, torch.tensor(sub_adata.X).float()), 1).float()
                    # add missing gene indicators where == -1
                    #node_labels = torch.cat((node_labels, torch.tensor((sub_adata.X==-1).astype(float))), 1).float()
                
                if self.node_feature == "gaussian":
                    # random gaussian noise as features
                    node_labels = torch.normal(mean=0, std=1, size=sub_adata.X.shape).float()
                
                if "X_spatial" in sub_adata.obsm:
                    precomputed_embed = torch.tensor(sub_adata.obsm["spatial"]).float()
                    node_labels = torch.cat((node_labels, precomputed_embed), dim=1)
                
                ### Get Indices of Random Center Cells
                cell_idxs = []
                
                if self.center_celltypes == "all":
                    center_celltypes_to_use = np.unique(sub_adata.obs["celltype"])
                else:
                    center_celltypes_to_use = self.center_celltypes
                
                for ct in center_celltypes_to_use:
                    np.random.seed(444)
                    idxs = np.random.choice(np.arange(sub_adata.shape[0])[sub_adata.obs["celltype"]==ct],
                                            np.min([self.num_cells_per_ct_id, np.sum(sub_adata.obs["celltype"]==ct)]),
                                            replace=False)
                    cell_idxs = np.concatenate((cell_idxs, idxs))

                ### Extract K-hop Subgraphs
                
                graph_labels = [] # for computing quantiles later
                for cidx in cell_idxs:
                    # get subgraph
                    sub_node_labels, sub_edge_index, graph_label, center_id, subgraph_cct, subgraph_cts, subgraph_region, subgraph_age, subgraph_cond = self.subgraph_from_index(int(cidx), edge_index, node_labels, sub_adata)
                    
                    # filter out tiny subgraphs
                    if len(sub_node_labels) > 2*self.k_hop:
                        
                        # append graph_label (for computing augmentation quantiles)
                        graph_labels.append(graph_label)
                        
                        # get injected labels
                        if (self.inject_feature == "center_celltype"):
                            #injected_labels = sub_node_labels[center_id,:len(self.celltypes_to_index.keys())].detach().clone() # get center cell type vector
                            injected_labels = one_hot(torch.tensor([self.celltypes_to_index[subgraph_cct[0]]]), num_classes=len(self.celltypes_to_index.keys()))
                        
                        # zero out center cell node features
                        sub_node_labels[center_id,:] = 0
                        
                        # make PyG Data object
                        if self.inject_feature is None:
                            subgraph_data = Data(x = sub_node_labels,
                                                 edge_index = sub_edge_index,
                                                 y = torch.tensor([graph_label]).flatten(), # flatten used to handle uni- and multi-variate targets
                                                 center_node = center_id,
                                                 center_celltype = subgraph_cct,
                                                 celltypes = subgraph_cts,
                                                 region = subgraph_region,
                                                 age = subgraph_age,
                                                 condition = subgraph_cond,
                                                 dataset = raw_filepath)
                        else:
                            subgraph_data = Data(x = sub_node_labels,
                                             edge_index = sub_edge_index,
                                             y = torch.tensor([graph_label]).flatten(), # flatten used to handle uni- and multi-variate targets
                                             center_node = center_id,
                                             center_celltype = subgraph_cct,
                                             celltypes = subgraph_cts,
                                             region = subgraph_region,
                                             age = subgraph_age,
                                             condition = subgraph_cond,
                                             dataset = raw_filepath,
                                             inject = injected_labels) 

                        # save object
                        torch.save(subgraph_data,
                                   os.path.join(self.processed_dir,f"g{subgraph_count}.pt"))
                        subgraph_count += 1
                        
                ### Selective Graph Augmentation
                
                # get augmentation indices
                if self.augment_hop > 0:
                    augment_idxs = []
                    for cidx in cell_idxs:
                        # get subgraph and get node indices of all nodes
                        sub_nodes, sub_edge_index, center_node_idx, edge_mask = k_hop_subgraph(
                                                                                int(cidx),
                                                                                self.augment_hop, 
                                                                                edge_index,
                                                                                relabel_nodes=True)
                        augment_idxs = np.concatenate((augment_idxs,sub_nodes.detach().numpy()))
                    
                    augment_idxs = np.unique(augment_idxs) # remove redundancies
                    
                    avg_aug_size = len(augment_idxs)/len(cell_idxs) # get average number of augmentations per center cell
                
                    # compute augmentation cutoff
                    if self.augment_cutoff == "auto":
                        bins, bin_edges = np.histogram(graph_labels, bins=5)
                        bins = np.concatenate((bins[0:1], bins, bins[-1:])) # expand edge bins with duplicate counts
                    else:
                        absglcutoff = np.quantile(np.abs(graph_labels), self.augment_cutoff)

                    # get subgraphs and save for augmentation
                    for cidx in augment_idxs:
                                            
                        # get subgraph
                        sub_node_labels, sub_edge_index, graph_label, center_id, subgraph_cct, subgraph_cts, subgraph_region, subgraph_age, subgraph_cond = self.subgraph_from_index(int(cidx), edge_index, node_labels, sub_adata)
                                            
                        # augmentation selection conditions
                        if self.augment_cutoff == "auto": # probabilistic
                            curr_bin = bins[np.digitize(graph_label,bin_edges)] # get freq of current bin
                            prob_aug = (np.max(bins) - curr_bin) / (curr_bin * avg_aug_size * (1-self.dispersion_factor))
                            do_aug = (random.random() < prob_aug) # augment with probability based on max bin size
                        else: # by quantile cutoff
                            do_aug = (np.mean(np.abs(graph_label)) >= absglcutoff) # if pass graph label cutoff then augment
                        
                        # save augmented graphs if conditions met
                        if (len(sub_node_labels) > 2*self.k_hop) and (do_aug):
                        
                            # get injected labels
                            if (self.inject_feature == "center_celltype"):
                                injected_labels = one_hot(torch.tensor([self.celltypes_to_index[subgraph_cct[0]]]), num_classes=len(self.celltypes_to_index.keys()))
                            
                            # zero out center cell node features
                            sub_node_labels[center_id,:] = 0
                            
                            # make PyG Data object
                            if self.inject_feature is None:
                                subgraph_data = Data(x = sub_node_labels,
                                                     edge_index = sub_edge_index,
                                                     y = torch.tensor([graph_label]).flatten(), # flatten used to handle uni- and multi-variate targets
                                                     center_node = center_id,
                                                     center_celltype = subgraph_cct,
                                                     celltypes = subgraph_cts,
                                                     region = subgraph_region,
                                                     age = subgraph_age,
                                                     condition = subgraph_cond,
                                                     dataset = raw_filepath)
                            else:
                                subgraph_data = Data(x = sub_node_labels,
                                                     edge_index = sub_edge_index,
                                                     y = torch.tensor([graph_label]).flatten(), # flatten used to handle uni- and multi-variate targets
                                                     center_node = center_id,
                                                     center_celltype = subgraph_cct,
                                                     celltypes = subgraph_cts,
                                                     region = subgraph_region,
                                                     age = subgraph_age,
                                                     condition = subgraph_cond,
                                                     inject = injected_labels,
                                                     dataset = raw_filepath)

                            # save object
                            torch.save(subgraph_data,
                                       os.path.join(self.processed_dir,f"g{subgraph_count}.pt"))
                            subgraph_count += 1

    
    def subgraph_from_index (self, cidx, edge_index, node_labels, sub_adata):
        '''
        Method used by self.process to extract subgraph and properties based on a cell index (cidx) and edge_index and node_labels and sub_adata
        '''
        # get subgraph
        sub_nodes, sub_edge_index, center_node_id, edge_mask = k_hop_subgraph(
                                                                int(cidx),
                                                                self.k_hop, 
                                                                edge_index,
                                                                relabel_nodes=True)
        # get node values
        sub_node_labels = node_labels[sub_nodes,:]

        # label graphs
        if self.target == "expression": # EXPRESSION AS LABEL
            graph_label = np.array(sub_adata[cidx,:].X).flatten().astype('float32')
        else:
            raise Exception ("'target' not recognized")
        
        # get celltypes and center cell type
        subgraph_cts = np.array(sub_adata.obs["celltype"].values[sub_nodes.numpy()].copy())
        subgraph_cct = subgraph_cts[center_node_id.numpy()]
        
        # get brain region if exists
        if "region" in sub_adata.obs.keys():
            subgraph_region = np.array(sub_adata.obs["region"].values[sub_nodes.numpy()].copy())[center_node_id.numpy()]
        else:
            subgraph_region = "no region specified"
        
        # get age if exists
        if "age" in sub_adata.obs.keys():
            subgraph_age = np.array(sub_adata.obs["age"].values[sub_nodes.numpy()].copy())[center_node_id.numpy()]
        else:
            subgraph_age = "no age specified"
        
        # get cohort (condition) if exists
        if "cohort" in sub_adata.obs.keys():
            subgraph_cond = np.array(sub_adata.obs["cohort"].values[sub_nodes.numpy()].copy())[center_node_id.numpy()]
        else:
            subgraph_cond = "no cohort specified"
        
        
        return (sub_node_labels, sub_edge_index, graph_label, center_node_id, subgraph_cct, subgraph_cts, subgraph_region, subgraph_age, subgraph_cond)
    

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'g{idx}.pt'), weights_only=False)
        return data
		

# Model class

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, input_dim, output_dim=1, inject_dim=0,
                 num_layers=3, method="GCN", pool="add"):
        super(GNN, self).__init__()
        torch.manual_seed(444)
        
        self.method = method
        self.pool = pool
        
        if self.method == "GCN":
            self.conv1 = GCNConv(input_dim, hidden_channels)
            self.convs = ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)])
        
        elif self.method == "GIN":
            self.conv1 = GINConv(
                            Sequential(
                                Linear(input_dim, hidden_channels),
                                BatchNorm1d(hidden_channels),
                                ReLU(),
                                Linear(hidden_channels, hidden_channels)
                            )
                          )
            self.convs = ModuleList([GINConv(
                            Sequential(
                                Linear(hidden_channels, hidden_channels),
                                BatchNorm1d(hidden_channels),
                                ReLU(),
                                Linear(hidden_channels, hidden_channels)
                            )
                          ) for _ in range(num_layers - 1)])
        
        elif self.method == "SAGE":
            self.conv1 = SAGEConv(input_dim, hidden_channels)
            self.convs = ModuleList([SAGEConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)])
            
        else:
            raise Exception("'method' not recognized.")
        
        self.lin = Linear(hidden_channels+inject_dim, output_dim)

    def forward(self, x, edge_index, batch, inject=None):
                    
        # node embeddings 
        x = F.relu(self.conv1(x, edge_index))
        for layer_idx, conv in enumerate(self.convs):
            if layer_idx < len(self.convs) - 1:
                x = F.relu(conv(x, edge_index))
            else:
                x = conv(x, edge_index)

        # pooling and readout
        if self.pool == "mean":
            x = global_mean_pool(x, batch)
        elif self.pool == "add":
            x = global_add_pool(x, batch)
        elif self.pool == "max":
            x = global_max_pool(x, batch)
        else:
            raise Exception ("'pool' not recognized")

        # final prediction
        x = F.dropout(x, p=0.1, training=self.training)
        
        if inject is None: # use only embedding to predict
            x = self.lin(x)
        else: # inject features at last layer
            x = self.lin(torch.cat((x,inject),1))
        
        return x
    

def bmc_loss(pred, target, noise_var):
    """Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, d].
      target: A float tensor of size [batch, d].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    # reshape target to pred shape -- added 6/25/2024
    if target.shape != pred.shape:
        target = torch.reshape(target, pred.shape)
    
    I = torch.eye(pred.shape[-1])
    logits = MVN(pred.unsqueeze(1), noise_var*I).log_prob(target.unsqueeze(0))  # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))     # contrastive-like loss
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 
    
    return loss


class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)



    
# negative pearson correlation loss
def npcc_loss(pred, target):
    """
    Negative pearson correlation as loss
    """
    
    # Alternative formulation
    x = torch.flatten(pred)
    y = torch.flatten(target)
    
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    
    loss = 1-r_val

    return loss
    
class Neg_Pearson_Loss(_Loss):
    def __init__(self):
        super(Neg_Pearson_Loss, self).__init__()
        return

    def forward(self, pred, target):
        return npcc_loss(pred, target)




def weighted_l1_loss(pred, target, zero_weight, nonzero_weight):

    if target.shape != pred.shape:
        target = torch.reshape(target, pred.shape)
    
    abs_diff = torch.abs(pred - target)
    zero_mask = (target == 0).float()
    nonzero_mask = (target != 0).float()
    
    loss = (zero_weight * zero_mask * abs_diff +
            nonzero_weight * nonzero_mask * abs_diff)
    
    return loss.mean()

class WeightedL1Loss(_Loss):
    def __init__(self, zero_weight=1.0, nonzero_weight=1.0):
        super(WeightedL1Loss, self).__init__()
        self.zero_weight = zero_weight
        self.nonzero_weight = nonzero_weight

    def forward(self, predictions, targets):
        
        loss = weighted_l1_loss(predictions, targets, self.zero_weight, self.nonzero_weight)
        
        return loss






def train(model, loader, criterion, optimizer, inject=False, device="cuda"):
    model.train()
    
    import time
    start = time.time()
    
    
    for data in loader:  # Iterate in batches over the training dataset.
        
        end = time.time()
        print(f"Data load time: {end - start}", flush=True)
        
        start = time.time()
        
        data.to(device)
        
        end = time.time()
        print(f"To cuda time: {end - start}", flush=True)
        
        start = time.time()
        
        if inject is False:
            out = model(data.x, data.edge_index, data.batch, None)  # Perform a single forward pass.
        else:
            out = model(data.x, data.edge_index, data.batch, data.inject) # Perform a single forward pass.
        
        end = time.time()
        print(f"Forward pass time: {end - start}", flush=True)
        
        start = time.time()
        
        loss = criterion(out, data.y)  # Compute the loss.
        
        end = time.time()
        print(f"Loss time: {end - start}", flush=True)
        
        start = time.time()
        
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        
        end = time.time()
        print(f"Backward pass time: {end - start}", flush=True)
        
        start = time.time()
        
        print(data.x.device)
        

def test(model, loader, loss, criterion, inject=False, device="cuda"):
    model.eval()

    errors = []
    for data in loader:  # Iterate in batches over the training/test dataset.
        
        data.to(device)
        if inject is False:
            out = model(data.x, data.edge_index, data.batch, None)
        else:
            out = model(data.x, data.edge_index, data.batch, data.inject)
        
        if loss == "mse":
            errors.append(F.mse_loss(out, data.y.unsqueeze(1)).sqrt().item())
        elif loss == "l1":
            errors.append(F.l1_loss(out, data.y.unsqueeze(1)).item())
        elif loss == "weightedl1":
            errors.append(weighted_l1_loss(out, data.y.unsqueeze(1), criterion.zero_weight, criterion.nonzero_weight).item())
        elif loss == "balanced_mse":
            errors.append(bmc_loss(out, data.y.unsqueeze(1), criterion.noise_sigma**2).item())
        elif loss == "npcc":
            errors.append(npcc_loss(out, data.y.unsqueeze(1)).item())
        
    return np.mean(errors)  # Derive ratio of correct predictions.