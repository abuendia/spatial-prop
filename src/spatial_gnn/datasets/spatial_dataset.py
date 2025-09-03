import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata as ad
import os
from scipy.sparse import issparse
import pickle
import random

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import k_hop_subgraph, one_hot
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import json

from spatial_gnn.scripts.ageaccel_proximity import *


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
                 embedding_json=None,
                 genept_embeddings_path=None,
                 perturbation_mask_key="perturbation_mask"
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
        self.embedding_json = embedding_json
        self.cell_embeddings = None
        self.perturbation_mask_key=perturbation_mask_key
        if embedding_json is not None:
            with open(embedding_json, 'r') as f:
                self.cell_embeddings = json.load(f)
            # Convert all embeddings to np.array for efficient stacking
            for k in self.cell_embeddings:
                self.cell_embeddings[k] = np.array(self.cell_embeddings[k], dtype=np.float32)
        
        # Load GenePT embeddings if provided
        self.genept_embeddings_path = genept_embeddings_path
        self.genept_embeddings = None
        if genept_embeddings_path is not None:
            print(f"Loading GenePT embeddings from {genept_embeddings_path}")
            with open(genept_embeddings_path, 'rb') as f:
                self.genept_embeddings = pickle.load(f)
            # Convert all embeddings to np.array for efficient stacking
            for k in self.genept_embeddings:
                self.genept_embeddings[k] = np.array(self.genept_embeddings[k], dtype=np.float32)
            print(f"Loaded {len(self.genept_embeddings)} GenePT embeddings")

    def indices(self):
        return range(self.len()) if self._indices is None else self._indices
    
    def _combine_expression_with_genept(self, expression_matrix, gene_names):
        """
        Combine raw expression values with GenePT embeddings.
        
        For each gene, multiply the raw expression value by the corresponding GenePT embedding.
        This creates a feature vector that combines expression magnitude with semantic gene information.
        
        Parameters:
        -----------
        expression_matrix : np.ndarray
            Expression matrix (cells x genes)
        gene_names : np.ndarray
            Gene names corresponding to the expression matrix
            
        Returns:
        --------
        np.ndarray
            Combined features matrix (cells x (genes * embedding_dim))
        """
        if self.genept_embeddings is None:
            return expression_matrix
        
        # Get embedding dimension
        emb_dim = len(next(iter(self.genept_embeddings.values())))
        
        # Initialize output matrix
        n_cells, n_genes = expression_matrix.shape
        combined_features = np.zeros((n_cells, n_genes * emb_dim), dtype=np.float32)
        
        # For each gene, combine expression with embedding
        for i, gene_name in enumerate(gene_names):
            # Convert gene name to uppercase for lookup
            gene_name_upper = gene_name.upper()
            if gene_name_upper in self.genept_embeddings:
                # Get the GenePT embedding for this gene
                gene_embedding = self.genept_embeddings[gene_name_upper]
                
                # Multiply expression values by the embedding
                # This creates a feature vector where each element is expression * embedding_dim
                for j in range(emb_dim):
                    combined_features[:, i * emb_dim + j] = expression_matrix[:, i] * gene_embedding[j]
            else:
                # If gene not in embeddings, use zeros for that gene's embedding dimensions
                combined_features[:, i * emb_dim:(i + 1) * emb_dim] = 0.0
        
        return combined_features
    
    @property
    def processed_dir(self) -> str:
        if self.augment_cutoff == 'auto':
            aug_key = self.augment_cutoff
        else:
            aug_key = int(self.augment_cutoff*100)
        celltype_firstletters = "".join([x[0] for x in self.center_celltypes])
        
        # Add GenePT indicator to directory name if embeddings are used
        genept_suffix = "_GenePT" if self.genept_embeddings is not None else ""
        
        data_dir = f"{self.dataset_prefix}_{self.target}_{self.num_cells_per_ct_id}per_{self.k_hop}hop_{self.augment_hop}C{aug_key}aug_{self.radius_cutoff}delaunay_{self.node_feature}Feat_{celltype_firstletters}_{self.inject_feature}Inject{genept_suffix}"
        if self.subfolder_name is not None:
            return os.path.join(self.root, self.processed_folder_name, data_dir, self.subfolder_name)
        else:
            return os.path.join(self.root, self.processed_folder_name, data_dir)

    @property
    def processed_file_names(self):
        return sorted([f for f in os.listdir(self.processed_dir) if f.endswith('.pt')])
        
    @property
    def gene_names(self):
        if self.gene_list is not None:
            # Load gene list from file
            if isinstance(self.gene_list, str):
                gene_names = np.genfromtxt(self.gene_list, dtype='unicode')
            elif isinstance(self.gene_list, list):
                gene_names = np.array(self.gene_list)
            else:
                raise Exception ("gene_list argument not recognized")
        else:
            # Compute gene list from first AnnData file
            adata = sc.read_h5ad(self.raw_filepaths[0])
            gene_names = adata.var_names.values
        return gene_names
    
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
        gene_names = self.gene_names
        
        # Convert gene names to uppercase for consistency
        gene_names = np.array([gene.upper() for gene in gene_names])
        
        # save genes (already converted to uppercase)
        if self.subfolder_name is not None:
            genefn = self.processed_dir.split("/")[-2]
        else:
            genefn = self.processed_dir.split("/")[-1]
        pd.DataFrame(gene_names).to_csv(os.path.join(self.root,self.processed_folder_name,"gene_names",f"{genefn}.csv"), header=False, index=False)
        
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
            
            # Convert all gene names to uppercase for consistency
            adata.var_names = [gene.upper() for gene in adata.var_names]
            gene_names = np.array([gene.upper() for gene in gene_names])
            
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
                
                # Check if perturbation mask exists and use it for expression features
                use_perturbation_expression = False
                if self.perturbation_mask_key in sub_adata.obs.keys():
                    use_perturbation_expression = True
                    print(f"Using perturbation mask from {self.perturbation_mask_key}")
                
                if "celltype" in self.node_feature:
                    # get cell type one hot encoding
                    node_labels = torch.tensor([self.celltypes_to_index[x] for x in sub_adata.obs["celltype"]])
                    node_labels = one_hot(node_labels, num_classes=len(self.celltypes_to_index.keys()))
                
                if "expression" in self.node_feature:
                    # get spatial expression - use perturbation mask if available
                    if use_perturbation_expression:
                        # Use perturbation mask values instead of original expression
                        perturbation_expression = np.array([sub_adata.obs[self.perturbation_mask_key].iloc[i] for i in range(sub_adata.shape[0])])
                        
                        # Reshape to match gene dimensions if needed
                        if len(perturbation_expression.shape) == 1:
                            # Single value per cell, expand to match gene dimensions
                            perturbation_expression = np.tile(perturbation_expression[:, np.newaxis], (1, sub_adata.shape[1]))
                        
                        if self.node_feature == "expression":
                            node_labels = torch.tensor(perturbation_expression).float()
                        else:
                            node_labels = torch.cat((node_labels, torch.tensor(perturbation_expression).float()), 1).float()
                    else:
                        # Use original expression values
                        if self.node_feature == "expression":
                            node_labels = torch.tensor(sub_adata.X).float()
                        else:
                            node_labels = torch.cat((node_labels, torch.tensor(sub_adata.X).float()), 1).float()
                    
                    # Combine with GenePT embeddings if available
                    if self.genept_embeddings is not None:
                        if use_perturbation_expression:
                            # Use perturbation expression for GenePT combination
                            genept_augmented_features = self._combine_expression_with_genept(perturbation_expression, sub_adata.var_names)
                        else:
                            # Use original expression for GenePT combination
                            genept_augmented_features = self._combine_expression_with_genept(sub_adata.X, sub_adata.var_names)
                        
                        if self.node_feature == "expression":
                            node_labels = torch.tensor(genept_augmented_features).float()
                        else:
                            node_labels = torch.cat((node_labels, torch.tensor(genept_augmented_features).float()), 1).float()
                    
                    # add missing gene indicators where == -1
                    #node_labels = torch.cat((node_labels, torch.tensor((sub_adata.X==-1).astype(float))), 1).float()
                
                if self.node_feature == "gaussian":
                    # random gaussian noise as features
                    node_labels = torch.normal(mean=0, std=1, size=sub_adata.X.shape).float()
                
                if "X_spatial" in sub_adata.obsm:
                    precomputed_embed = torch.tensor(sub_adata.obsm["spatial"]).float()
                    node_labels = torch.cat((node_labels, precomputed_embed), dim=1)
                
                # If using embeddings, build embedding matrix for all cells in sub_adata
                if self.cell_embeddings is not None:
                    emb_dim = len(next(iter(self.cell_embeddings.values())))
                    emb_matrix = np.zeros((sub_adata.shape[0], emb_dim), dtype=np.float32)
                    for i, cid in enumerate(sub_adata.obs_names):
                        if cid in self.cell_embeddings:
                            emb_matrix[i] = self.cell_embeddings[cid]
                        else:
                            emb_matrix[i] = np.zeros(emb_dim, dtype=np.float32)
                    emb_tensor = torch.tensor(emb_matrix).float()
                else:
                    emb_tensor = None
                
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
                    cidx = int(cidx)
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
                                                 dataset = raw_filepath, 
                                                 original_cell_idx = cidx,
                                                 original_cell_id = sub_adata.obs_names[cidx])
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
                                             dataset = raw_filepath, 
                                             original_cell_idx = cidx,
                                             original_cell_id = sub_adata.obs_names[cidx])

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
                        cidx = int(cidx)
                        sub_node_labels, sub_edge_index, graph_label, center_id, subgraph_cct, subgraph_cts, subgraph_region, subgraph_age, subgraph_cond = self.subgraph_from_index(cidx, edge_index, node_labels, sub_adata)
                                            
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
                                                     dataset = raw_filepath, 
                                                     original_cell_idx = cidx,
                                                     original_cell_id = sub_adata.obs_names[cidx])
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
                                                     dataset = raw_filepath, 
                                                     original_cell_idx = cidx,
                                                     original_cell_id = sub_adata.obs_names[cidx])

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
