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
from spatial_gnn.models.gnn_model import predict
from spatial_gnn.datasets.spatial_dataset import SpatialAgingCellDataset


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
    # Create perturbation mask
    perturbation_mask = np.ones((adata.shape[0], adata.shape[1]))
    
    # Check if celltype column exists
    if 'celltype' not in adata.obs.columns:
        raise ValueError("AnnData object must have 'celltype' column in obs")
    
    # Apply perturbations for each cell type
    for cell_type, gene_multipliers in perturbation_dict.items():
        # Find cells of this type
        cell_mask = adata.obs['celltype'] == cell_type
        cell_indices = np.where(cell_mask)[0]
        
        if len(cell_indices) == 0:
            print(f"Warning: No cells found for cell type '{cell_type}'")
            continue
            
        print(f"Applying perturbations to {len(cell_indices)} cells of type '{cell_type}'")
        
        # Apply gene-specific multipliers
        for gene_name, multiplier in gene_multipliers.items():
            if gene_name in adata.var_names:
                gene_idx = adata.var_names.get_loc(gene_name)
                perturbation_mask[cell_indices, gene_idx] = multiplier * adata.X[cell_indices, gene_idx]
                print(f"  - Gene '{gene_name}': multiplier = {multiplier}")
            else:
                print(f"Warning: Gene '{gene_name}' not found in data")
    
    # Add the perturbation mask to the AnnData
    adata.obsm[mask_key] = perturbation_mask
    
    # Save to file if path is provided
    if save_path is not None:
        adata.write(save_path)
        print(f"Saved AnnData with perturbation mask to: {save_path}")
    
    # Store metadata about the perturbation
    perturbation_info = {
        'cell_types': list(perturbation_dict.keys()),
        'n_perturbed_cells': sum(len(np.where(adata.obs['celltype'] == ct)[0]) 
                                for ct in perturbation_dict.keys()),
        'perturbed_genes': list(set(gene for genes in perturbation_dict.values() 
                                   for gene in genes.keys()))
    }
    adata.uns['perturbation_info'] = perturbation_info
    
    print(f"\nPerturbation mask created:")
    print(f"- Shape: {perturbation_mask.shape}")
    print(f"- Cell types affected: {perturbation_info['cell_types']}")
    print(f"- Cells affected: {perturbation_info['n_perturbed_cells']}")
    print(f"- Genes affected: {perturbation_info['perturbed_genes']}")
    print(f"- Mask stored in adata.obsm['{mask_key}']")
    
    return save_path


def train_perturbation_model(
    adata_path: str,
    exp_name: str,
    k_hop: int = 2,
    augment_hop: int = 2,
    center_celltypes: str = "all",
    node_feature: str = "expression",
    inject_feature: str = "None",
    gene_list: Optional[str] = None,
    num_cells_per_ct_id: int = 100,
    normalize_total: bool = True,
    learning_rate: float = 1e-4,
    loss: str = "weightedl1",
    epochs: int = 100,
    debug: bool = False,
    debug_subset_size: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs
) -> Tuple[Any, Dict, str]:
    """
    Train a GNN model for perturbation prediction from scratch.
    
    Parameters
    ----------
    adata_path : str
        Path to the training AnnData file (.h5ad)
    exp_name : str
        Name of the experiment
    k_hop : int, default=2
        k-hop neighborhood size
    augment_hop : int, default=2
        Number of hops for graph augmentation
    center_celltypes : str, default="all"
        Cell types to center graphs on (comma-separated or "all")
    node_feature : str, default="expression"
        Node feature type
    inject_feature : str, default="None"
        Feature injection type
    gene_list : Optional[str], default=None
        Path to gene list file
    num_cells_per_ct_id : int, default=100
        Number of cells per cell type per ID
    normalize_total : bool, default=True
        Whether to normalize total gene expression
    learning_rate : float, default=1e-4
        Learning rate for training
    loss : str, default="weightedl1"
        Loss function type
    epochs : int, default=100
        Number of training epochs
    debug : bool, default=False
        Enable debug mode
    debug_subset_size : int, default=100
        Number of cells for debug mode
    device : str, default="cuda" if available else "cpu"
        Device to run training on
        
    Returns
    -------
    Tuple[Any, Dict, str]
        (trained_model, model_config, model_path)
    """
    print("Training new perturbation model from scratch...")
    model, model_config, trained_model_path = train_model_from_scratch(
        exp_name=exp_name,
        k_hop=k_hop,
        augment_hop=augment_hop,
        center_celltypes=center_celltypes,
        node_feature=node_feature,
        inject_feature=inject_feature,
        learning_rate=learning_rate,
        loss=loss,
        epochs=epochs,
        num_cells_per_ct_id=num_cells_per_ct_id,
        adata_path=adata_path,
        gene_list=gene_list,
        normalize_total=normalize_total,
        debug=debug,
        debug_subset_size=debug_subset_size,
        device=device
    )
    print(f"Training completed. Model saved to: {trained_model_path}")
    return model, model_config, trained_model_path


def predict_perturbation_effects(
    adata_path: str,
    model_path: str,
    exp_name: str,
    perturbation_dict: Dict[str, Dict[str, float]],
    perturbation_mask_key: str = 'perturbation_mask',
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs
) -> ad.AnnData:
    """
    Predict the effects of perturbations on spatial transcriptomics data using a trained GNN model.
    
    This function loads a pretrained model and applies perturbations to predict their effects
    on gene expression across the spatial network.
    
    Parameters
    ----------
    adata_path : str
        Path to the test AnnData file (.h5ad) containing spatial transcriptomics data
    model_path : str
        Path to the saved model state dictionary (.pth file)
    exp_name : str
        Name of the experiment
    perturbation_dict : Dict[str, Dict[str, float]]
        Dictionary specifying perturbations:
        - Keys: cell type names
        - Values: dictionaries with gene names as keys and multipliers as values
        Example: {'T cell': {'Igf2': 0.0}, 'NSC': {'Sox9': 2.0}}
    perturbation_mask_key : str, default='perturbation_mask'
        Key to store the perturbation mask in adata.obsm
    device : str, default="cuda" if available else "cpu"
        Device to run the model on
    **kwargs
        Additional arguments (for compatibility)
    
    Returns
    -------
    anndata.AnnData
        Updated AnnData object with perturbation effects stored in:
        - adata.layers['perturbation_effects']: Predicted perturbation effects
        - adata.obsm[perturbation_mask_key]: Applied perturbation mask
    """
    # Validate inputs
    if not isinstance(adata_path, str):
        raise TypeError("adata_path must be a string")
    if not isinstance(model_path, str):
        raise TypeError("model_path must be a string")
    if not isinstance(perturbation_dict, dict):
        raise TypeError("perturbation_dict must be a dictionary")
        
    # Load the pretrained model
    model, model_config = load_model_from_path(model_path, device)
    print(f"Loaded pretrained model from: {model_path}")
    
    # Load the test data
    adata = sc.read_h5ad(adata_path)
    
    # Create perturbation mask and apply it to the data
    print("Creating perturbation mask...")
    create_perturbation_mask(adata, perturbation_dict, mask_key=perturbation_mask_key)
    
    # Create graphs from the input AnnData
    print("Creating graphs from input data...")
    inference_dataset = SpatialAgingCellDataset(
        subfolder_name="predict",
        dataset_prefix=exp_name,
        target="expression",
        k_hop=model_config["k_hop"],
        augment_hop=model_config["augment_hop"],
        node_feature=model_config["node_feature"],
        inject_feature=model_config["inject_feature"],
        num_cells_per_ct_id=model_config["num_cells_per_ct_id"],
        center_celltypes=model_config["center_celltypes"],
        use_ids=model_config["test_ids"] if model_config["test_ids"] is not None else True,
        raw_filepaths=[adata_path],
        celltypes_to_index=model_config["celltypes_to_index"],
        normalize_total=model_config["normalize_total"],
        perturbation_mask_key=perturbation_mask_key
    )
    inference_dataset.process()
    
    # Load batch files the same way as training
    all_inference_data = []
    for f in inference_dataset.processed_file_names:
        all_inference_data.append(torch.load(os.path.join(inference_dataset.processed_dir, f), weights_only=False))
    
    inference_dataloader = DataLoader(all_inference_data, batch_size=512, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    print(f"Created {len(all_inference_data)} batch files from input data")

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
        'num_perturbed_cells': np.sum(effects != 0)
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
    train_data_path = "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/raw/aging_coronal.h5ad"
    test_data_path = "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/raw/aging_coronal.h5ad"  # Using same data for demo
    
    # Define perturbations
    perturbation_dict = {
        'T cell': {'Igf2': 0.0},  
        'NSC': {'Sox9': 2.0},         
        'Pericyte': {'Ccl4': 0.5}    
    }
    
    print("=== Training a new perturbation model ===")
    model, model_config, model_path = train_perturbation_model(
        adata_path=train_data_path,
        k_hop=2,
        augment_hop=2,
        center_celltypes="T cell,NSC,Pericyte",
        node_feature="expression",
        inject_feature="None",
        num_cells_per_ct_id=100,
        epochs=10,
    )
    
    test_adata = sc.read_h5ad(test_data_path)
    test_data_path_perturbed = create_perturbation_mask(test_adata, perturbation_dict, save_path=test_data_path)

    print("\n=== Predicting perturbation effects ===")
    adata_perturbed = predict_perturbation_effects(
        adata_path=test_data_path_perturbed,
        model_path=model_path,
        perturbation_dict=perturbation_dict,
        perturbation_mask_key="perturbation_mask"
    )
    
    print("=== Visualizing perturbation effects ===")
    visualize_perturbation_effects(adata_perturbed)
    
    print("=== Getting perturbation summary ===")
    perturbation_summary = get_perturbation_summary(adata_perturbed)
    print(perturbation_summary)
    