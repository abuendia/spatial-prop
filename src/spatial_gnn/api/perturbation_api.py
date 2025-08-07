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
import copy
import warnings
from typing import List, Dict, Union, Optional, Tuple, Any
import sys
import os
import json 


from spatial_gnn.scripts.aging_gnn_model import SpatialAgingCellDataset, GNN
from spatial_gnn.scripts.perturbation import predict, perturb_data, get_center_celltypes, get_center_celltypes


def predict_perturbation_effects(
    adata: ad.AnnData,
    model_path: str,
    perturbations: List[Dict[str, Any]],
    dataset: str,
    base_path: str,
    k_hop: int,
    augment_hop: int,
    center_celltypes: str,
    node_feature: str,
    inject_feature: str,
    learning_rate: float,
    loss: str,
    epochs: int,
    gene_list: Optional[str] = None,
    normalize_total: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
    **kwargs
) -> ad.AnnData:
    """
    Predict the effects of perturbations on spatial transcriptomics data using a trained GNN model.
    
    This function takes an AnnData object, applies specified perturbations, and uses a GNN model
    to predict the resulting changes in gene expression across the spatial network.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing spatial transcriptomics data with:
        - X: gene expression matrix (cells x genes)
        - obs: cell metadata including spatial coordinates and cell types
        - obsm: spatial coordinates (e.g., 'spatial')
        - var: gene metadata
    
    model_path : str
        Path to the saved model state dictionary (.pth file)
    
    perturbations : List[Dict[str, Any]]
        List of perturbation specifications. Each perturbation should be a dictionary with:
        - 'type': str, one of ['add', 'multiply', 'knockout', 'overexpression']
        - 'genes': List[str] or List[int], gene names or indices to perturb
        - 'magnitude': float, magnitude of perturbation
        - 'cell_types': Optional[List[str]], specific cell types to perturb (if None, all cells)
        - 'spatial_region': Optional[Dict], spatial region constraints
        - 'proportion': Optional[float], proportion of cells to perturb (default: 1.0)
    
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
    
    learning_rate : float
        learning rate
    
    loss : str
        loss: balanced_mse, npcc, mse, l1
    
    epochs : int
        number of epochs
    
    gene_list : Optional[str], default=None
        Path to file containing list of genes to use (optional)
    
    normalize_total : bool, default=True
        Whether to normalize total gene expression
    
    device : str, default="cuda" if available else "cpu"
        Device to run the model on
    
    batch_size : int, default=32
        Batch size for processing
    
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
    ...     learning_rate=1e-4,
    ...     loss="mse",
    ...     epochs=100
    ... )
    """
    
    # Validate inputs
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an AnnData object")
    
    if not isinstance(model_path, str):
        raise TypeError("model_path must be a string")
    
    if not isinstance(perturbations, list) or len(perturbations) == 0:
        raise ValueError("perturbations must be a non-empty list")
    
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
    
    # Prepare dataset using the exact same approach as train_gnn_model_expression.py
    dataset_obj, celltypes_to_index = _prepare_dataset_like_training(
        dataset, file_path, k_hop, augment_hop, center_celltypes, 
        node_feature, inject_feature, train_ids, test_ids,
        gene_list_data, celltypes_to_index, normalize_total
    )
    
    # Load model using the same approach as model_performance.py
    model = _load_model_from_path(model_path, dataset_obj, device)
    
    # Create a copy of the original data
    adata_result = adata.copy()
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize storage for results
    all_predictions_original = []
    all_predictions_perturbed = []
    all_perturbation_masks = []
    all_cell_indices = []
    
    # Process each batch
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # Get original predictions
            predictions_original = predict(model, batch_data, inject=inject_feature is not None)
            
            # Apply perturbations and get perturbed predictions
            batch_data_perturbed, perturbation_mask = _apply_perturbations_to_batch(
                batch_data, perturbations, adata_result, celltypes_to_index
            )
            
            predictions_perturbed = predict(model, batch_data_perturbed, inject=inject_feature is not None)
            
            # Store results
            all_predictions_original.append(predictions_original.cpu())
            all_predictions_perturbed.append(predictions_perturbed.cpu())
            all_perturbation_masks.append(perturbation_mask.cpu())
            
            # Store cell indices for mapping back to original data
            cell_indices = _get_cell_indices_from_batch(batch_data, adata_result)
            all_cell_indices.extend(cell_indices)
    
    # Concatenate all results
    predictions_original = torch.cat(all_predictions_original, dim=0)
    predictions_perturbed = torch.cat(all_predictions_perturbed, dim=0)
    perturbation_mask = torch.cat(all_perturbation_masks, dim=0)
    
    # Map predictions back to original AnnData structure
    _add_predictions_to_adata(
        adata_result, predictions_original, predictions_perturbed, 
        perturbation_mask, all_cell_indices, dataset.gene_names
    )
    
    return adata_result


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

    return test_dataset, celltypes_to_index


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
    # This is a simplified implementation
    # In practice, you'd need to map the batch data back to original cell indices
    return list(range(batch_data.x.shape[0]))


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
    
    # Add to AnnData layers
    adata.layers['predicted_original'] = np.zeros_like(adata.X)
    adata.layers['predicted_perturbed'] = np.zeros_like(adata.X)
    adata.layers['perturbation_effects'] = np.zeros_like(adata.X)
    
    # Map predictions back to original cell order
    for i, cell_idx in enumerate(cell_indices):
        if cell_idx < adata.n_obs:
            adata.layers['predicted_original'][cell_idx, :len(gene_names)] = pred_orig_np[i, :]
            adata.layers['predicted_perturbed'][cell_idx, :len(gene_names)] = pred_pert_np[i, :]
            adata.layers['perturbation_effects'][cell_idx, :len(gene_names)] = perturbation_effects[i, :]
    
    # Add perturbation mask to obs
    adata.obs['perturbation_mask'] = pert_mask_np[:adata.n_obs]


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
    
    # Create visualization
    import matplotlib.pyplot as plt
    
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
    
    # Use the same parameters as the training script
    perturbations = [{'type': 'knockout', 'genes': ['Gm12878'], 'magnitude': 0.0, 'cell_types': ['T cell', 'NSC', 'Pericyte'], 'proportion': 1.0}]
    adata_perturbed = predict_perturbation_effects(
        adata=adata,
        model_path=model_path,
        perturbations=perturbations,
        dataset="aging_coronal",
        base_path="/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/raw",
        k_hop=2,
        augment_hop=2,
        center_celltypes="T cell,NSC,Pericyte",
        node_feature="expression",
        inject_feature="None",
        learning_rate=1e-4,
        loss="weightedl1",
        epochs=10
    )
    visualize_perturbation_effects(adata_perturbed)
