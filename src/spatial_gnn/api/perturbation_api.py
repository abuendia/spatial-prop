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

# Import from the spatial-gnn codebase
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from aging_gnn_model import SpatialAgingCellDataset, GNN
from perturbation import predict, perturb_data, get_center_celltypes


def predict_perturbation_effects(
    adata: ad.AnnData,
    model: torch.nn.Module,
    perturbations: List[Dict[str, Any]],
    k_hop: int = 2,
    node_feature: str = "expression",
    inject_feature: Optional[str] = None,
    celltypes_to_index: Optional[Dict[str, int]] = None,
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
    
    model : torch.nn.Module
        Trained GNN model (should be compatible with the GNN class from aging_gnn_model.py)
    
    perturbations : List[Dict[str, Any]]
        List of perturbation specifications. Each perturbation should be a dictionary with:
        - 'type': str, one of ['add', 'multiply', 'knockout', 'overexpression']
        - 'genes': List[str] or List[int], gene names or indices to perturb
        - 'magnitude': float, magnitude of perturbation
        - 'cell_types': Optional[List[str]], specific cell types to perturb (if None, all cells)
        - 'spatial_region': Optional[Dict], spatial region constraints
        - 'proportion': Optional[float], proportion of cells to perturb (default: 1.0)
    
    k_hop : int, default=2
        Number of hops for spatial neighborhood construction
    
    node_feature : str, default="expression"
        Type of node features to use ("expression", "celltype", "celltype_expression")
    
    inject_feature : Optional[str], default=None
        Feature to inject into the model (e.g., "center_celltype")
    
    celltypes_to_index : Optional[Dict[str, int]], default=None
        Mapping of cell type names to indices. If None, will be inferred from adata.obs
    
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
    ...     model=trained_model,
    ...     perturbations=perturbations,
    ...     k_hop=2
    ... )
    """
    
    # Validate inputs
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an AnnData object")
    
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a PyTorch module")
    
    if not isinstance(perturbations, list) or len(perturbations) == 0:
        raise ValueError("perturbations must be a non-empty list")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Create a copy of the original data
    adata_result = adata.copy()
    
    # Infer celltypes_to_index if not provided
    if celltypes_to_index is None:
        if 'celltype' in adata.obs.columns:
            unique_celltypes = sorted(adata.obs['celltype'].unique())
            celltypes_to_index = {ct: i for i, ct in enumerate(unique_celltypes)}
        else:
            warnings.warn("No celltype column found in adata.obs, using default celltypes_to_index")
            celltypes_to_index = {
                'Neuron-Excitatory': 0, 'Neuron-Inhibitory': 1, 'Neuron-MSN': 2,
                'Astrocyte': 3, 'Microglia': 4, 'Oligodendrocyte': 5, 'OPC': 6,
                'Endothelial': 7, 'Pericyte': 8, 'VSMC': 9, 'VLMC': 10,
                'Ependymal': 11, 'Neuroblast': 12, 'NSC': 13, 'Macrophage': 14,
                'Neutrophil': 15, 'T cell': 16, 'B cell': 17
            }
    
    # Prepare dataset for processing
    dataset = _prepare_dataset(
        adata, k_hop, node_feature, inject_feature, celltypes_to_index, **kwargs
    )
    
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
            batch_data = batch_data.to(device)
            
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


def _prepare_dataset(
    adata: ad.AnnData,
    k_hop: int,
    node_feature: str,
    inject_feature: Optional[str],
    celltypes_to_index: Dict[str, int],
    **kwargs
) -> SpatialAgingCellDataset:
    """Prepare dataset for processing."""
    
    # Create temporary file path for the dataset
    temp_filepath = f"temp_adata_{id(adata)}.h5ad"
    adata.write(temp_filepath)
    
    try:
        # Create dataset
        dataset = SpatialAgingCellDataset(
            root=".",
            raw_filepaths=[temp_filepath],
            k_hop=k_hop,
            node_feature=node_feature,
            inject_feature=inject_feature,
            celltypes_to_index=celltypes_to_index,
            **kwargs
        )
        
        return dataset
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)


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