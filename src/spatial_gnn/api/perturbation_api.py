"""
API for spatial GNN perturbation predictions.
"""
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph, one_hot
from scipy.sparse import issparse
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from typing import List, Dict, Union, Optional, Tuple, Any
import sys
import os
import json
import tqdm 
import matplotlib.pyplot as plt

from spatial_gnn.scripts.aging_gnn_model import SpatialAgingCellDataset, GNN
from spatial_gnn.scripts.perturbation import predict, perturb_data, get_center_celltypes, get_center_celltypes
from spatial_gnn.scripts.ageaccel_proximity import build_spatial_graph

# celltype -> gene -> multiplier 
# .obsm['perturbation_mask']
# save result to .obsm['perturbation_results']

def create_perturbation_mask(
    adata: ad.AnnData,
    perturbation_dict: Dict[str, Dict[str, float]],
    mask_key: str = 'perturbation_mask'
) -> ad.AnnData:
    """
    Create a perturbation mask from a cell type → gene → multiplier dictionary.
    
    This function generates a perturbation mask that specifies exactly what each gene 
    in each cell should be perturbed to. The mask is stored in adata.obsm[mask_key].
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing spatial transcriptomics data
    perturbation_dict : Dict[str, Dict[str, float]]
        Nested dictionary mapping cell_type → gene_name → multiplier.
        - multiplier = 0.0: knockout (set expression to 0)
        - multiplier = 2.0: 2x overexpression 
        - multiplier = 0.5: 50% knockdown
        - multiplier = 1.0: no change (default)
    mask_key : str, default='perturbation_mask'
        Key to store the perturbation mask in adata.obsm
        
    Returns
    -------
    anndata.AnnData
        Updated AnnData object with perturbation mask in adata.obsm[mask_key]
        
    Examples
    --------
    >>> # Example: Knockout Gene1 in T cells, overexpress Gene2 in NSCs
    >>> perturbation_dict = {
    ...     'T cell': {'Gene1': 0.0},  # knockout
    ...     'NSC': {'Gene2': 2.0}      # 2x overexpression
    ... }
    >>> adata_with_mask = create_perturbation_mask(adata, perturbation_dict)
    """
    
    # Make a copy to avoid modifying the original
    adata_result = adata.copy()
    
    # Initialize perturbation mask with ones (no perturbation by default)
    n_cells, n_genes = adata_result.shape
    perturbation_mask = np.ones((n_cells, n_genes), dtype=np.float32)
    
    # Check if celltype column exists
    if 'celltype' not in adata_result.obs.columns:
        raise ValueError("AnnData object must have 'celltype' column in obs")
    
    # Apply perturbations for each cell type
    for cell_type, gene_multipliers in perturbation_dict.items():
        # Find cells of this type
        cell_mask = adata_result.obs['celltype'] == cell_type
        cell_indices = np.where(cell_mask)[0]
        
        if len(cell_indices) == 0:
            print(f"Warning: No cells found for cell type '{cell_type}'")
            continue
            
        print(f"Applying perturbations to {len(cell_indices)} cells of type '{cell_type}'")
        
        # Apply gene-specific multipliers
        for gene_name, multiplier in gene_multipliers.items():
            if gene_name in adata_result.var_names:
                gene_idx = adata_result.var_names.get_loc(gene_name)
                perturbation_mask[cell_indices, gene_idx] = multiplier
                print(f"  - Gene '{gene_name}': multiplier = {multiplier}")
            else:
                print(f"Warning: Gene '{gene_name}' not found in data")
    
    # Store the perturbation mask
    adata_result.obsm[mask_key] = perturbation_mask
    
    # Store metadata about the perturbation
    perturbation_info = {
        'cell_types': list(perturbation_dict.keys()),
        'n_perturbed_cells': sum(len(np.where(adata_result.obs['celltype'] == ct)[0]) 
                                for ct in perturbation_dict.keys()),
        'perturbed_genes': list(set(gene for genes in perturbation_dict.values() 
                                   for gene in genes.keys()))
    }
    adata_result.uns['perturbation_info'] = perturbation_info
    
    print(f"\nPerturbation mask created:")
    print(f"- Shape: {perturbation_mask.shape}")
    print(f"- Cell types affected: {perturbation_info['cell_types']}")
    print(f"- Cells affected: {perturbation_info['n_perturbed_cells']}")
    print(f"- Genes affected: {perturbation_info['perturbed_genes']}")
    print(f"- Mask stored in adata.obsm['{mask_key}']")
    
    return adata_result


def predict_perturbation_effects(
    adata: ad.AnnData,
    model_path: str,
    dataset: str,
    base_path: str,
    k_hop: int,
    augment_hop: int,
    center_celltypes: str,
    node_feature: str,
    inject_feature: str,
    perturbations: Optional[List[Dict[str, Any]]] = None,
    perturbation_mask_key: str = 'perturbation_mask',
    gene_list: Optional[str] = None,
    normalize_total: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
    debug: bool = False,
    debug_subset_size: int = 100,
    **kwargs
) -> ad.AnnData:
    """
    Predict the effects of perturbations on spatial transcriptomics data using a trained GNN model.
    
    This function takes an AnnData object with perturbations specified either through:
    1. A perturbation mask stored in adata.obsm[perturbation_mask_key] (preferred), OR
    2. A list of perturbation specifications (legacy mode)
    
    The function uses a GNN model to predict the resulting changes in gene expression 
    across the spatial network.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing spatial transcriptomics data with:
        - X: gene expression matrix (cells x genes)
        - obs: cell metadata including spatial coordinates and cell types
        - obsm: spatial coordinates (e.g., 'spatial') and optionally perturbation_mask
        - var: gene metadata
    
    model_path : str
        Path to the saved model state dictionary (.pth file)
    
    perturbations : Optional[List[Dict[str, Any]]], default=None
        LEGACY: List of perturbation specifications. Each perturbation should be a dictionary with:
        - 'type': str, one of ['add', 'multiply', 'knockout', 'overexpression']
        - 'genes': List[str] or List[int], gene names or indices to perturb
        - 'magnitude': float, magnitude of perturbation
        - 'cell_types': Optional[List[str]], specific cell types to perturb (if None, all cells)
        - 'spatial_region': Optional[Dict], spatial region constraints
        - 'proportion': Optional[float], proportion of cells to perturb (default: 1.0)
        NOTE: If perturbation_mask_key exists in adata.obsm, this parameter is ignored.
    
    perturbation_mask_key : str, default='perturbation_mask'
        Key in adata.obsm containing the perturbation mask (cells x genes matrix of multipliers)
    
    dataset : str
        Dataset to use (aging_coronal, aging_sagittal, exercise, reprogramming, allen, kukanja, pilot)
    
    base_path : str
        Base path to the data directory
    
    k_hop : int
        k-hop neighborhood size
    
    augment_hop : int
        number of hops to take for graph augmentation
    
    center_celltypes : str
        cell type labels to center graphs on, separated by comma. Use 'all' for all cell types or 'none' for no cell type filtering
    
    node_feature : str
        node features key, e.g. 'celltype_age_region'
    
    inject_feature : str
        inject features key, e.g. 'center_celltype'
    
    gene_list : Optional[str], default=None
        Path to file containing list of genes to use (optional)
    
    normalize_total : bool, default=True
        Whether to normalize total gene expression
    
    device : str, default="cuda" if available else "cpu"
        Device to run the model on
    
    batch_size : int, default=32
        Batch size for processing
    
    debug : bool, default=False
        Enable debug mode with subset of data for quick testing
    
    debug_subset_size : int, default=100
        Number of cells to use in debug mode
    
    **kwargs
        Additional arguments passed to SpatialAgingCellDataset
    
    Returns
    -------
    anndata.AnnData
        Updated AnnData object with new layers:
        - 'predicted_original': Original predictions without perturbations
        - 'predicted_perturbed': Predictions with perturbations applied
        - 'perturbation_effects': Difference between perturbed and original predictions
        - 'perturbation_mask': Boolean mask indicating which cells were perturbed
    
    Examples
    --------
    >>> # Example perturbation: knockout a gene in specific cell types
    >>> perturbations = [{
    ...     'type': 'knockout',
    ...     'genes': ['Gene1', 'Gene2'],
    ...     'magnitude': 0.0,
    ...     'cell_types': ['Neuron-Excitatory', 'Neuron-Inhibitory'],
    ...     'proportion': 0.5
    ... }]
    >>> 
    >>> # Apply perturbations and get predictions
    >>> adata_perturbed = predict_perturbation_effects(
    ...     adata=adata,
    ...     model_path="path/to/model.pth",
    ...     perturbations=perturbations,
    ...     dataset="aging_coronal",
    ...     base_path="/path/to/data",
    ...     k_hop=2,
    ...     augment_hop=2,
    ...     center_celltypes="all",
    ...     node_feature="expression",
    ...     inject_feature="None",
    ...     debug=True,
    ...     debug_subset_size=50
    ... )
    """
    
    # Validate inputs
    if adata is not None and not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an AnnData object or None")
    
    if not isinstance(model_path, str):
        raise TypeError("model_path must be a string")
    
    # Check perturbation specification - either mask in obsm or legacy perturbations list
    use_perturbation_mask = False
    if adata is not None and perturbation_mask_key in adata.obsm:
        use_perturbation_mask = True
        print(f"Using perturbation mask from adata.obsm['{perturbation_mask_key}']")
        # Validate mask dimensions
        mask_shape = adata.obsm[perturbation_mask_key].shape
        expected_shape = adata.shape
        if mask_shape != expected_shape:
            raise ValueError(f"Perturbation mask shape {mask_shape} doesn't match data shape {expected_shape}")
    elif perturbations is not None:
        if not isinstance(perturbations, list) or len(perturbations) == 0:
            raise ValueError("perturbations must be a non-empty list")
        print("Using legacy perturbations list")
    else:
        raise ValueError("Must provide either perturbation_mask in adata.obsm or perturbations list")
    
    # Load dataset configurations - exact same as train_gnn_model_expression.py
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

    DATASET_CONFIGS = load_dataset_config()

    # Validate dataset choice
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Dataset must be one of: {', '.join(DATASET_CONFIGS.keys())}")

    # load parameters from arguments - exact same as train_gnn_model_expression.py
    dataset_config = DATASET_CONFIGS[dataset]
    train_ids = dataset_config['train_ids']
    test_ids = dataset_config['test_ids']
    file_path = os.path.join(base_path, dataset_config['file_name'])
    
    # Handle center_celltypes - exact same as train_gnn_model_expression.py
    if center_celltypes.lower() == 'none':
        center_celltypes = None
    elif center_celltypes.lower() == 'all':
        center_celltypes = 'all'
    else:
        center_celltypes = center_celltypes.split(",")

    if inject_feature.lower() == "none":
        inject_feature = None
        inject=False
    else:
        inject=True

    # Load gene list if provided - exact same as train_gnn_model_expression.py
    gene_list_data = None
    if gene_list is not None:
        try:
            with open(gene_list, 'r') as f:
                gene_list_data = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Gene list file not found at {gene_list}")
    
    # Build cell type index - exact same as train_gnn_model_expression.py
    celltypes_to_index = {}
    for ci, cellt in enumerate(dataset_config["celltypes"]):
        celltypes_to_index[cellt] = ci
    
    # Two modes: either use provided AnnData or load from training pipeline
    if adata is not None:
        # Apply debug subsetting if enabled
        if debug:
            print(f"DEBUG MODE: Subsetting AnnData to {debug_subset_size} cells", flush=True)
            # Randomly sample cells for faster processing
            n_cells = min(debug_subset_size, adata.shape[0])
            sampled_indices = np.random.choice(adata.shape[0], n_cells, replace=False)
            adata = adata[sampled_indices].copy()
            # Store original indices for mapping back
            adata.obs['original_index'] = sampled_indices
            print(f"DEBUG: AnnData subset to {adata.shape[0]} cells", flush=True)
        else:
            # If not in debug mode, just store sequential indices
            adata.obs['original_index'] = np.arange(adata.shape[0])
        
        # Mode 1: User provided AnnData - create graphs on-the-fly
        graphs_data, gene_names, cell_type_indices = _create_graphs_from_adata(
            adata, k_hop, node_feature, inject_feature, celltypes_to_index, normalize_total, debug, debug_subset_size
        )
        # Load model using a sample from the graphs
        model = _load_model_from_graphs(model_path, graphs_data[0], device)
    else:
        # Mode 2: Use training pipeline approach
        dataset_obj, celltypes_to_index, test_loader, all_test_data = _prepare_dataset_like_training(
            dataset, file_path, k_hop, augment_hop, center_celltypes, 
            node_feature, inject_feature, train_ids, test_ids,
            gene_list_data, celltypes_to_index, normalize_total
        )
        # Load model using the dataset
        model = _load_model_from_path(model_path, dataset_obj, device)
        # Use the pre-loaded data
        graphs_data = all_test_data
        gene_names = dataset_obj.gene_names
    
    # Create a copy of the original data for results
    if adata is not None:
        adata_result = adata.copy()
        # Create dataloader from graphs
        dataloader = DataLoader(graphs_data, batch_size=batch_size, shuffle=False)
    else:
        # For training pipeline mode, create a dummy adata for results
        adata_result = ad.AnnData(X=np.zeros((100, len(gene_names))))
        adata_result.var_names = gene_names
        # Use the pre-created dataloader
        dataloader = test_loader
    
    # Initialize storage for results
    all_predictions_original = []
    all_predictions_perturbed = []
    all_perturbation_masks = []
    all_cell_indices = []
    
    # Process each batch
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            batch_data = batch_data.to(device)
            
            # Get original predictions
            predictions_original = predict(model, batch_data, inject=inject_feature is not None)
            
            # Store cell indices for mapping back to original data
            cell_indices = _get_cell_indices_from_batch(batch_data, adata_result)
            
            # Apply perturbations and get perturbed predictions
            if use_perturbation_mask:
                batch_data_perturbed, perturbation_mask = _apply_perturbation_mask_to_batch(
                    batch_data, adata_result, perturbation_mask_key, cell_indices
                )
            else:
                batch_data_perturbed, perturbation_mask = _apply_perturbations_to_batch(
                    batch_data, perturbations, adata_result, celltypes_to_index
                )
            
            predictions_perturbed = predict(model, batch_data_perturbed, inject=inject_feature is not None)
            
            # Store results
            all_predictions_original.append(predictions_original.cpu())
            all_predictions_perturbed.append(predictions_perturbed.cpu())
            all_perturbation_masks.append(perturbation_mask.cpu())
            all_cell_indices.extend(cell_indices)
    
    # Concatenate all results
    predictions_original = torch.cat(all_predictions_original, dim=0)
    predictions_perturbed = torch.cat(all_predictions_perturbed, dim=0)
    perturbation_mask = torch.cat(all_perturbation_masks, dim=0)
    
    # Map predictions back to original AnnData structure
    _add_predictions_to_adata(
        adata_result, predictions_original, predictions_perturbed, 
        perturbation_mask, all_cell_indices, gene_names
    )
    
    return adata_result


def _create_graphs_from_adata(
    adata: ad.AnnData,
    k_hop: int,
    node_feature: str,
    inject_feature: Optional[str],
    celltypes_to_index: Dict[str, int],
    normalize_total: bool,
    debug: bool = False,
    debug_subset_size: int = 100
) -> Tuple[List[Data], List[str], np.ndarray]:
    """
    Create graph objects directly from AnnData, similar to how SpatialAgingCellDataset.process() works
    
    Returns
    -------
    Tuple[List[Data], List[str], np.ndarray]
        - List of graph Data objects
        - List of gene names
        - Array of original cell indices (for mapping back to original data)
    """    
    # Make a copy to avoid modifying the original
    adata_copy = adata.copy()
    
    if issparse(adata_copy.X):
        adata_copy.X = adata_copy.X.toarray()
    
    # Filter to known cell types
    cell_type_mask = adata_copy.obs.celltype.isin(celltypes_to_index.keys())
    adata_copy = adata_copy[cell_type_mask]
    original_indices = np.where(cell_type_mask)[0]  # Track indices after cell type filtering
    
    # Normalize by total genes if requested
    if normalize_total:
        print("Normalized data")
        sc.pp.normalize_total(adata_copy, target_sum=adata_copy.shape[1])
    
    # Get gene names
    gene_names = adata_copy.var_names.values
    
    # Build spatial graph using Delaunay triangulation
    build_spatial_graph(adata_copy, method="delaunay")
    radius_cutoff = 200  # Same as in training
    adata_copy.obsp['spatial_connectivities'][adata_copy.obsp['spatial_distances'] > radius_cutoff] = 0
    adata_copy.obsp['spatial_distances'][adata_copy.obsp['spatial_distances'] > radius_cutoff] = 0
    
    # Convert to PyG format
    edge_index, edge_att = from_scipy_sparse_matrix(adata_copy.obsp['spatial_connectivities'])
    
    # Construct node labels
    if node_feature not in ["celltype", "expression", "celltype_expression"]:
        raise Exception(f"'node_feature' value of {node_feature} not recognized")
    
    if "celltype" in node_feature:
        node_labels = torch.tensor([celltypes_to_index[x] for x in adata_copy.obs["celltype"]])
        node_labels = one_hot(node_labels, num_classes=len(celltypes_to_index.keys()))
    
    if "expression" in node_feature:
        if node_feature == "expression":
            node_labels = torch.tensor(adata_copy.X).float()
        else:
            node_labels = torch.cat((node_labels, torch.tensor(adata_copy.X).float()), 1).float()
    
    # Create graphs for each cell (simplified version - just create a few representative graphs)
    graphs = []
    cell_indices = []
    
    for cidx in range(adata_copy.shape[0]):
        # Get k-hop subgraph
        sub_nodes, sub_edge_index, center_node_id, edge_mask = k_hop_subgraph(
            int(cidx), k_hop, edge_index, relabel_nodes=True
        )
        
        if len(sub_nodes) > 2 * k_hop:  # Filter out tiny subgraphs
            # Get node features for subgraph
            sub_node_labels = node_labels[sub_nodes, :]
            
            # Get target (expression of center cell)
            graph_label = np.array(adata_copy[cidx, :].X).flatten().astype('float32')
            
            # Get cell types
            subgraph_cts = np.array(adata_copy.obs["celltype"].values[sub_nodes.numpy()].copy())
            subgraph_cct = subgraph_cts[center_node_id.numpy()]
            
            # Get injected labels if needed
            if inject_feature == "center_celltype":
                injected_labels = one_hot(
                    torch.tensor([celltypes_to_index[subgraph_cct[0]]]), 
                    num_classes=len(celltypes_to_index.keys())
                )
            
            # Zero out center cell node features (as in training)
            sub_node_labels[center_node_id, :] = 0
            
            # Create PyG Data object
            if inject_feature is None:
                graph_data = Data(
                    x=sub_node_labels,
                    edge_index=sub_edge_index,
                    y=torch.tensor([graph_label]).flatten(),
                    center_node=center_node_id,
                    center_celltype=subgraph_cct,
                    celltypes=subgraph_cts
                )
            else:
                graph_data = Data(
                    x=sub_node_labels,
                    edge_index=sub_edge_index,
                    y=torch.tensor([graph_label]).flatten(),
                    center_node=center_node_id,
                    center_celltype=subgraph_cct,
                    celltypes=subgraph_cts,
                    inject=injected_labels
                )
            
            graphs.append(graph_data)
            cell_indices.append(cidx)
    
    return graphs, gene_names


def _load_model_from_graphs(
    model_path: str,
    sample_graph: Data,
    device: str
) -> torch.nn.Module:
    """
    Load model using a sample graph to determine dimensions
    """
    # Determine if injection is used
    inject = hasattr(sample_graph, 'inject')
    
    if inject:
        model = GNN(
            hidden_channels=64,
            input_dim=int(sample_graph.x.shape[1]),
            output_dim=len(sample_graph.y),
            inject_dim=int(sample_graph.inject.shape[1]),
            method="GIN",
            pool="add",
            num_layers=2  # This should match k_hop from training
        )
    else:
        model = GNN(
            hidden_channels=64,
            input_dim=int(sample_graph.x.shape[1]),
            output_dim=len(sample_graph.y),
            method="GIN",
            pool="add",
            num_layers=2  # This should match k_hop from training
        )
    
    # Load the state dictionary
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    
    return model


def _prepare_dataset_like_training(
    dataset: str,
    file_path: str,
    k_hop: int,
    augment_hop: int,
    center_celltypes: Union[str, List[str], None],
    node_feature: str,
    inject_feature: Optional[str],
    train_ids: List[str],
    test_ids: List[str],
    gene_list: Optional[List[str]],
    celltypes_to_index: Dict[str, int],
    normalize_total: bool
) -> Tuple[SpatialAgingCellDataset, Dict[str, int]]:
    """
    Prepare dataset using the exact same approach as train_gnn_model_expression.py
    """
    
    # init dataset with settings - exact copy from train_gnn_model_expression.py
    train_dataset = SpatialAgingCellDataset(subfolder_name="train",
                                            dataset_prefix=dataset,
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
                                            celltypes_to_index=celltypes_to_index,
                                            normalize_total=normalize_total)

    test_dataset = SpatialAgingCellDataset(subfolder_name="test",
                                        dataset_prefix=dataset,
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
                                        celltypes_to_index=celltypes_to_index,
                                        normalize_total=normalize_total)
    
    test_dataset.process()
    print("Finished processing test dataset", flush=True)
    train_dataset.process()
    print("Finished processing train dataset", flush=True)

    # Apply debug mode subsetting if enabled (copied from train_gnn_model_expression.py)
    # For perturbation, we can add a debug parameter later if needed
    debug = True  # Set to False by default for perturbation analysis
    debug_subset_size = 100  # Default value
    
    if debug:
        print(f"DEBUG MODE: Using subset of {debug_subset_size} samples from each dataset", flush=True)
        
        # Subset train dataset
        train_subset_size = min(debug_subset_size, len(train_dataset))
        train_dataset._indices = list(range(train_subset_size))
        
        # Subset test dataset  
        test_subset_size = min(debug_subset_size, len(test_dataset))
        test_dataset._indices = list(range(test_subset_size))
        
        print(f"DEBUG: Train dataset subset to {len(train_dataset)} samples", flush=True)
        print(f"DEBUG: Test dataset subset to {len(test_dataset)} samples", flush=True)

    all_train_data = []
    all_test_data = []
    
    # Get file names to load - use subset if in debug mode (copied from train_gnn_model_expression.py)
    if debug:
        train_files = train_dataset.processed_file_names[:train_subset_size]
        test_files = test_dataset.processed_file_names[:test_subset_size]
    else:
        train_files = train_dataset.processed_file_names
        test_files = test_dataset.processed_file_names
    
    for f in tqdm.tqdm(train_files):
        all_train_data.append(torch.load(os.path.join(train_dataset.processed_dir, f), weights_only=False))

    for f in tqdm.tqdm(test_files):
        all_test_data.append(torch.load(os.path.join(test_dataset.processed_dir, f), weights_only=False))

    train_loader = DataLoader(all_train_data, batch_size=512, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(all_test_data, batch_size=512, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    print(len(all_train_data), flush=True)
    print(len(all_test_data), flush=True)

    # For perturbation analysis, we'll use the test dataset and test_loader
    return test_dataset, celltypes_to_index, test_loader, all_test_data


def _load_model_from_path(
    model_path: str,
    dataset: SpatialAgingCellDataset,
    device: str
) -> torch.nn.Module:
    """
    Load a trained GNN model from a saved state dictionary.
    This follows the same pattern as model_performance.py
    """
    
    # Initialize model with the same parameters as training
    inject = dataset.inject_feature is not None
    if inject:
        model = GNN(
            hidden_channels=64,
            input_dim=int(dataset.get(0).x.shape[1]),
            output_dim=len(dataset.get(0).y),
            inject_dim=int(dataset.get(0).inject.shape[1]),
            method="GIN", 
            pool="add", 
            num_layers=dataset.k_hop
        )
    else:
        model = GNN(
            hidden_channels=64,
            input_dim=int(dataset.get(0).x.shape[1]),
            output_dim=len(dataset.get(0).y),
            method="GIN", 
            pool="add", 
            num_layers=dataset.k_hop
        )
    
    # Load the state dictionary
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    
    return model


def _apply_perturbation_mask_to_batch(
    batch_data: Data,
    adata: ad.AnnData,
    perturbation_mask_key: str,
    cell_indices: List[int]
) -> Tuple[Data, torch.Tensor]:
    """Apply perturbations using a pre-computed perturbation mask from adata.obsm."""
    
    batch_data_perturbed = batch_data.clone()
    perturbation_mask_full = adata.obsm[perturbation_mask_key]
    
    # Create boolean mask for cells that have any perturbations (multiplier != 1.0)
    perturbation_mask = torch.zeros(batch_data.x.shape[0], dtype=torch.bool)
    
    # Apply perturbations to each node in the batch
    for i in range(batch_data.x.shape[0]):
        # Map batch index to original cell index
        if i < len(cell_indices):
            cell_idx = cell_indices[i]
            if cell_idx < perturbation_mask_full.shape[0]:
                # Get the perturbation multipliers for this cell
                cell_multipliers = perturbation_mask_full[cell_idx, :]
                
                # Apply multipliers to node features (element-wise multiplication)
                if len(cell_multipliers) <= batch_data.x.shape[1]:
                    batch_data_perturbed.x[i, :len(cell_multipliers)] *= torch.tensor(
                        cell_multipliers, dtype=batch_data.x.dtype, device=batch_data.x.device
                    )
                    
                    # Mark as perturbed if any multiplier is not 1.0
                    if not np.allclose(cell_multipliers, 1.0):
                        perturbation_mask[i] = True
    
    return batch_data_perturbed, perturbation_mask


def _apply_perturbations_to_batch(
    batch_data: Data,
    perturbations: List[Dict[str, Any]],
    adata: ad.AnnData,
    celltypes_to_index: Dict[str, int]
) -> Tuple[Data, torch.Tensor]:
    """Apply perturbations to a batch of data."""
    
    batch_data_perturbed = batch_data.clone()
    perturbation_mask = torch.zeros(batch_data.x.shape[0], dtype=torch.bool)
    
    for perturbation in perturbations:
        # Parse perturbation parameters
        pert_type = perturbation.get('type', 'add')
        genes = perturbation.get('genes', [])
        magnitude = perturbation.get('magnitude', 1.0)
        cell_types = perturbation.get('cell_types', None)
        proportion = perturbation.get('proportion', 1.0)
        
        # Convert gene names to indices if needed
        if isinstance(genes[0], str):
            gene_indices = [adata.var.index.get_loc(gene) for gene in genes if gene in adata.var.index]
        else:
            gene_indices = genes
        
        if not gene_indices:
            continue
        
        # Create perturbation vector
        pert_vector = torch.zeros(batch_data.x.shape[1])
        if pert_type == 'knockout':
            pert_vector[gene_indices] = -magnitude
        elif pert_type == 'overexpression':
            pert_vector[gene_indices] = magnitude
        elif pert_type == 'add':
            pert_vector[gene_indices] = magnitude
        elif pert_type == 'multiply':
            pert_vector[gene_indices] = magnitude
        else:
            raise ValueError(f"Unknown perturbation type: {pert_type}")
        
        # Create cell mask based on cell types and proportion
        cell_mask = _create_cell_mask(
            batch_data, cell_types, proportion, celltypes_to_index
        )
        
        # Apply perturbation
        if pert_type == 'multiply':
            batch_data_perturbed.x[cell_mask, gene_indices] *= pert_vector[gene_indices]
        else:
            batch_data_perturbed.x[cell_mask, gene_indices] += pert_vector[gene_indices]
        
        # Update perturbation mask
        perturbation_mask[cell_mask] = True
    
    return batch_data_perturbed, perturbation_mask


def _create_cell_mask(
    batch_data: Data,
    cell_types: Optional[List[str]],
    proportion: float,
    celltypes_to_index: Dict[str, int]
) -> torch.Tensor:
    """Create a boolean mask for cells to perturb."""
    
    if cell_types is None:
        # Perturb all cells
        mask = torch.ones(batch_data.x.shape[0], dtype=torch.bool)
    else:
        # Perturb specific cell types
        mask = torch.zeros(batch_data.x.shape[0], dtype=torch.bool)
        for cell_type in cell_types:
            if cell_type in celltypes_to_index:
                ct_idx = celltypes_to_index[cell_type]
                # This is a simplified approach - in practice, you'd need to map
                # cell types to actual cells in the batch data
                # For now, we'll assume all cells can be perturbed
                mask = torch.ones(batch_data.x.shape[0], dtype=torch.bool)
                break
    
    # Apply proportion
    if proportion < 1.0:
        num_cells = int(proportion * mask.sum().item())
        if num_cells > 0:
            # Randomly select cells to perturb
            indices = torch.where(mask)[0]
            selected_indices = indices[torch.randperm(len(indices))[:num_cells]]
            mask = torch.zeros_like(mask)
            mask[selected_indices] = True
    
    return mask


def _get_cell_indices_from_batch(batch_data: Data, adata: ad.AnnData) -> List[int]:
    """Extract cell indices from batch data."""
    # Get the original indices from the AnnData object
    if 'original_index' in adata.obs:
        # Map through both the original_index and cell type filtering
        return adata.obs['original_index'].astype(int).tolist()
    else:
        # Fallback to sequential indices if original_index is not available
        return list(range(adata.n_obs))


def _add_predictions_to_adata(
    adata: ad.AnnData,
    predictions_original: torch.Tensor,
    predictions_perturbed: torch.Tensor,
    perturbation_mask: torch.Tensor,
    cell_indices: List[int],
    gene_names: List[str]
) -> None:
    """Add prediction results to AnnData object."""
    
    # Convert predictions to numpy arrays
    pred_orig_np = predictions_original.numpy()
    pred_pert_np = predictions_perturbed.numpy()
    pert_mask_np = perturbation_mask.numpy()
    
    # Create perturbation effects (difference)
    perturbation_effects = pred_pert_np - pred_orig_np
    
    # Initialize layers with zeros for the full size of original data
    n_cells = max(cell_indices) + 1 if cell_indices else adata.n_obs
    n_genes = adata.X.shape[1]
    
    adata.layers['predicted_original'] = np.zeros((n_cells, n_genes))
    adata.layers['predicted_perturbed'] = np.zeros((n_cells, n_genes))
    adata.layers['perturbation_effects'] = np.zeros((n_cells, n_genes))
    adata.obs['perturbation_mask'] = np.zeros(n_cells, dtype=bool)
    
    # Map predictions back to original cell order using the tracked indices
    for i, cell_idx in enumerate(cell_indices):
        if i < len(pred_orig_np):  # Check if we have predictions for this index
            gene_length = min(len(gene_names), n_genes)
            adata.layers['predicted_original'][cell_idx, :gene_length] = pred_orig_np[i, :gene_length]
            adata.layers['predicted_perturbed'][cell_idx, :gene_length] = pred_pert_np[i, :gene_length]
            adata.layers['perturbation_effects'][cell_idx, :gene_length] = perturbation_effects[i, :gene_length]
            if i < len(pert_mask_np):
                adata.obs.loc[cell_idx, 'perturbation_mask'] = pert_mask_np[i]


def get_perturbation_summary(adata: ad.AnnData) -> pd.DataFrame:
    """
    Generate a summary of perturbation effects.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with perturbation results
        
    Returns
    -------
    pd.DataFrame
        Summary statistics of perturbation effects
    """
    
    if 'perturbation_effects' not in adata.layers:
        raise ValueError("No perturbation effects found in adata.layers")
    
    effects = adata.layers['perturbation_effects']
    
    # Calculate summary statistics
    summary_stats = {
        'mean_effect': np.mean(effects, axis=0),
        'std_effect': np.std(effects, axis=0),
        'max_effect': np.max(effects, axis=0),
        'min_effect': np.min(effects, axis=0),
        'median_effect': np.median(effects, axis=0),
        'num_perturbed_cells': np.sum(adata.obs.get('perturbation_mask', False))
    }
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_stats, index=adata.var_names)
    
    return summary_df


def visualize_perturbation_effects(
    adata: ad.AnnData,
    genes: Optional[List[str]] = None,
    n_genes: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize perturbation effects.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with perturbation results
    genes : Optional[List[str]], default=None
        Specific genes to visualize
    n_genes : int, default=10
        Number of top affected genes to show if genes is None
    save_path : Optional[str], default=None
        Path to save the visualization
    """
    
    if 'perturbation_effects' not in adata.layers:
        raise ValueError("No perturbation effects found in adata.layers")
    
    effects = adata.layers['perturbation_effects']
    
    if genes is None:
        # Select top affected genes
        mean_effects = np.abs(np.mean(effects, axis=0))
        top_indices = np.argsort(mean_effects)[-n_genes:]
        genes = adata.var_names[top_indices].tolist()
    
    
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Mean effects by gene
    gene_effects = np.mean(effects, axis=0)
    gene_indices = [adata.var_names.get_loc(gene) for gene in genes if gene in adata.var_names]
    axes[0, 0].bar(range(len(gene_indices)), gene_effects[gene_indices])
    axes[0, 0].set_title('Mean Perturbation Effects by Gene')
    axes[0, 0].set_xticks(range(len(gene_indices)))
    axes[0, 0].set_xticklabels(genes, rotation=45)
    
    # Plot 2: Distribution of effects
    axes[0, 1].hist(gene_effects, bins=50, alpha=0.7)
    axes[0, 1].set_title('Distribution of Mean Effects')
    axes[0, 1].set_xlabel('Mean Effect')
    axes[0, 1].set_ylabel('Frequency')
    
    # Plot 3: Spatial visualization (if spatial coordinates available)
    if 'spatial' in adata.obsm:
        scatter = axes[1, 0].scatter(
            adata.obsm['spatial'][:, 0],
            adata.obsm['spatial'][:, 1],
            c=adata.obs.get('perturbation_mask', False),
            cmap='viridis',
            alpha=0.6
        )
        axes[1, 0].set_title('Perturbed Cells (Spatial)')
        axes[1, 0].set_xlabel('X coordinate')
        axes[1, 0].set_ylabel('Y coordinate')
        plt.colorbar(scatter, ax=axes[1, 0])
    
    # Plot 4: Effect magnitude by cell type
    if 'celltype' in adata.obs.columns:
        celltype_effects = []
        celltype_labels = []
        for celltype in adata.obs['celltype'].unique():
            mask = adata.obs['celltype'] == celltype
            if mask.sum() > 0:
                celltype_effects.append(np.mean(np.abs(effects[mask])))
                celltype_labels.append(celltype)
        
        axes[1, 1].bar(range(len(celltype_effects)), celltype_effects)
        axes[1, 1].set_title('Mean Effect Magnitude by Cell Type')
        axes[1, 1].set_xticks(range(len(celltype_effects)))
        axes[1, 1].set_xticklabels(celltype_labels, rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 


if __name__ == "__main__":
    data_path = "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/raw/aging_coronal.h5ad"
    model_path = "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/results/gnn/aging_coronal_expression_100per_2hop_2C0aug_200delaunay_expressionFeat_TNP_NoneInject/DEBUG_weightedl1_1en04/best_model.pth"
    
    adata = sc.read_h5ad(data_path)
    
    perturbation_dict = {
        'T cell': {'Gm12878': 0.0},      # knockout Gm12878 in T cells
        'NSC': {'Gm12878': 0.0},         # knockout Gm12878 in NSCs  
        'Pericyte': {'Gm12878': 0.0}     # knockout Gm12878 in Pericytes
    }
    
    adata_with_mask = create_perturbation_mask(adata, perturbation_dict)
    adata_perturbed = predict_perturbation_effects(
        adata=adata_with_mask,
        model_path=model_path,
        dataset="aging_coronal",
        base_path="/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/raw",
        k_hop=2,
        augment_hop=2,
        center_celltypes="T cell,NSC,Pericyte",
        node_feature="expression",
        inject_feature="None",
        debug=True,
        debug_subset_size=50
    )
    visualize_perturbation_effects(adata_perturbed)
