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
import os
import tqdm 
import matplotlib.pyplot as plt


from spatial_gnn.scripts.train_gnn_model_expression import train_model_from_scratch
from spatial_gnn.utils.dataset_utils import load_model_from_path
from spatial_gnn.utils.dataset_utils import create_graphs_from_adata
from spatial_gnn.models.gnn_model import predict


def create_perturbation_mask(
    adata: ad.AnnData,
    perturbation_dict: Dict[str, Dict[str, float]],
    mask_key: str = 'perturbation_mask',
    save_path: Optional[str] = None
) -> str:
    """
    Create a perturbation mask and save the AnnData with the mask to a file.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Input AnnData object
    perturbation_dict : Dict[str, Dict[str, float]]
        Dictionary specifying perturbations
    mask_key : str, default='perturbation_mask'
        Key to store the perturbation mask in adata.obsm
    save_dir : Optional[str], default=None
        Directory to save the file. If None, uses current working directory.
        
    Returns
    -------
    str
        Path to the saved AnnData file with perturbation mask
    """
    # Make a copy to avoid modifying the original
    adata_result = adata.copy()
    
    # Create perturbation mask
    perturbation_mask = np.ones((adata_result.shape[0], adata_result.shape[1]))
    
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
                perturbation_mask[cell_indices, gene_idx] = multiplier * adata_result.X[cell_indices, gene_idx]
                print(f"  - Gene '{gene_name}': multiplier = {multiplier}")
            else:
                print(f"Warning: Gene '{gene_name}' not found in data")
    
    # Add the perturbation mask to the AnnData
    adata_result.obsm[mask_key] = perturbation_mask
    
    # Save to file if path is provided
    if save_path is not None:
        adata.write(save_path)
        print(f"Saved AnnData with perturbation mask to: {save_path}")
    
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
    
    return save_path


def predict_perturbation_effects(
    anndata_path: str,
    model_path: Optional[str] = None,
    perturbation_mask_key: str = 'perturbation_mask',
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
    dataset: Optional[str] = None,
    base_path: Optional[str] = None,
    k_hop: int = 2,
    augment_hop: int = 2,
    center_celltypes: str = "all",
    node_feature: str = "expression",
    inject_feature: str = "None",
    gene_list: Optional[str] = None,
    normalize_total: bool = True,
    learning_rate: float = 1e-4,
    loss: str = "mse",
    epochs: int = 100,
    debug: bool = False,
    debug_subset_size: int = 100,
    **kwargs
) -> ad.AnnData:
    """
    Predict the effects of perturbations on spatial transcriptomics data using a GNN model.
    
    This function supports two modes:
    1. **Training from scratch**: When model_path is None, train a new model from scratch
    2. **Inference with pretrained model**: When model_path is provided, load and use a pretrained model
    
    The function takes an AnnData object with perturbations specified either through:
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
    
    model_path : Optional[str], default=None
        Path to the saved model state dictionary (.pth file). If None, train a new model from scratch.
    
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
    
    learning_rate : float, default=1e-4
        Learning rate for training (used when model_path is None)
    
    loss : str, default="mse"
        Loss function for training: "mse", "l1", "weightedl1", "balanced_mse", "npcc" (used when model_path is None)
    
    epochs : int, default=100
        Number of training epochs (used when model_path is None)
    
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
    """
    # Validate inputs
    if anndata_path is not None and not isinstance(anndata_path, str):
        raise TypeError("anndata_path must be a string or None")
    if model_path is not None and not isinstance(model_path, str):
        raise TypeError("model_path must be a string or None")
        
    # Handle model loading/training
    if model_path is not None:
        model, model_config = load_model_from_path(model_path, device)
        print(f"Loaded pretrained model from: {model_path}")
    else:
        # Train new model from scratch
        print("No model path provided. Training new model from scratch...")
        model, model_config, trained_model_path = train_model_from_scratch(
            dataset=dataset,
            base_path=base_path,
            k_hop=k_hop,
            augment_hop=augment_hop,
            center_celltypes=center_celltypes,
            node_feature=node_feature,
            inject_feature=inject_feature,
            learning_rate=learning_rate,
            loss=loss,
            epochs=epochs,
            gene_list=gene_list,
            normalize_total=normalize_total,
            debug=debug,
            debug_subset_size=debug_subset_size,
            device=device
        )
        print(f"Training completed. Model saved to: {trained_model_path}")

    # Create graphs from the input AnnData
    print("Creating graphs from input data...")
    inference_dataloader = create_graphs_from_adata(
        anndata_path=anndata_path,
        dataset_name=dataset,
        model_config=model_config,
        use_all_ids=True,
        batch_size=batch_size,
        perturbation_mask_key=perturbation_mask_key
    )    
    print(f"Created {len(inference_dataloader)} graphs from input data")
    # Run model predict function
    adata_result = predict(model, inference_dataloader, adata, device)
    print("Perturbation prediction completed successfully!")
    return adata_result

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
    save_path = "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/perturbed/aging_coronal_perturbed.h5ad"

    model_path = "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/results/gnn/aging_coronal_expression_100per_2hop_2C0aug_200delaunay_expressionFeat_TNP_NoneInject/DEBUG_weightedl1_1en04/best_model.pth"
    
    adata = sc.read_h5ad(data_path)
    
    perturbation_dict = {
        'T cell': {'Igf2': 0.0},  
        'NSC': {'Sox9': 2.0},         
        'Pericyte': {'Ccl4': 0.5}    
    }
    # Create perturbation mask and get file path
    adata_file_path = create_perturbation_mask(adata, perturbation_dict, save_path=save_path)
    
    # Example 1: Using pretrained model (inference mode)
    # Params inferred from model config
    print("=== Example 1: Using pretrained model ===")
    adata_perturbed = predict_perturbation_effects(
        anndata_path=adata_file_path,
        model_path=model_path,
        perturbation_mask_key="perturbation_mask"
    )
    

    training_params = {
        "k_hop": 2,
        "augment_hop": 2,
        "center_celltypes": "T cell,NSC,Pericyte",
        "node_feature": "expression",
        "inject_feature": "None",
        "debug": True,
        "debug_subset_size": 50
    }
    # # Example 2: Training from scratch (training mode)
    # print("\n=== Example 2: Training from scratch ===")
    # adata_perturbed_trained = predict_perturbation_effects(
    #     adata_file,  # Now this is a file path
    #     model_path=None,  # No model path - trains new model
    #     dataset="aging_coronal",
    #     base_path="/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/raw",
    #     k_hop=2,
    #     augment_hop=2,
    #     center_celltypes="T cell,NSC,Pericyte",
    #     node_feature="expression",
    #     inject_feature="None",
    #     debug=True,
    #     debug_subset_size=50,
    #     learning_rate=1e-4,
    #     loss="mse",
    #     epochs=50  # Fewer epochs for demo
    # )
    
    # Visualize results from pretrained model
    # visualize_perturbation_effects(adata_perturbed)
