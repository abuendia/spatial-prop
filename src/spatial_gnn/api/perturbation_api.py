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


from spatial_gnn.scripts.aging_gnn_model import SpatialAgingCellDataset, predict, GNN
from spatial_gnn.scripts.train_gnn_model_expression import train_model_from_scratch
from spatial_gnn.scripts.ageaccel_proximity import build_spatial_graph
from spatial_gnn.scripts.utils import load_model_from_path


def _combine_expression_with_genept(expression_matrix, gene_names, genept_embeddings):
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
    genept_embeddings : dict
        Dictionary mapping gene names to their GenePT embeddings
        
    Returns:
    --------
    np.ndarray
        Combined features matrix (cells x (genes * embedding_dim))
    """
    if genept_embeddings is None:
        return expression_matrix
    
    # Get embedding dimension
    emb_dim = len(next(iter(genept_embeddings.values())))
    
    # Initialize output matrix
    n_cells, n_genes = expression_matrix.shape
    combined_features = np.zeros((n_cells, n_genes * emb_dim), dtype=np.float32)
    
    # For each gene, combine expression with embedding
    for i, gene_name in enumerate(gene_names):
        if gene_name in genept_embeddings:
            # Get the GenePT embedding for this gene
            gene_embedding = genept_embeddings[gene_name]
            
            # Multiply expression values by the embedding
            # This creates a feature vector where each element is expression * embedding_dim
            for j in range(emb_dim):
                combined_features[:, i * emb_dim + j] = expression_matrix[:, i] * gene_embedding[j]
        else:
            # If gene not in embeddings, use zeros for that gene's embedding dimensions
            combined_features[:, i * emb_dim:(i + 1) * emb_dim] = 0.0
    
    return combined_features


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
    adata: ad.AnnData,
    dataset: str,
    base_path: str,
    k_hop: int,
    augment_hop: int,
    center_celltypes: str,
    node_feature: str,
    inject_feature: str,
    model_path: Optional[str] = None,
    perturbation_mask_key: str = 'perturbation_mask',
    gene_list: Optional[str] = None,
    normalize_total: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
    debug: bool = False,
    debug_subset_size: int = 100,
    learning_rate: float = 1e-4,
    loss: str = "mse",
    epochs: int = 100,
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
    
    if model_path is not None and not isinstance(model_path, str):
        raise TypeError("model_path must be a string or None")
        
    # Handle model loading/training
    if model_path is not None:
        model = load_model_from_path(model_path, device)
        print(f"Loaded pretrained model from: {model_path}")
    else:
        # Train new model from scratch
        print("No model path provided. Training new model from scratch...")
        model, trained_model_path = train_model_from_scratch(
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

    # Create graphs from the input AnnData using the existing function
    print("Creating graphs from input data...")
    graphs_data, gene_names, cell_indices_mapping = _create_graphs_from_adata(
        adata=adata,
        k_hop=k_hop,
        node_feature=node_feature,
        inject_feature=inject_feature,
        celltypes_to_index=celltypes_to_index,
        normalize_total=normalize_total,
        debug=debug,
        debug_subset_size=debug_subset_size
    )
    
    if not graphs_data:
        raise ValueError("No valid graphs could be created from the input data")
    
    print(f"Created {len(graphs_data)} graphs from input data")
    
    # Create a copy of the original data for results
    adata_result = adata.copy()
    
    # Create dataloader from graphs
    dataloader = DataLoader(graphs_data, batch_size=batch_size, shuffle=False)

    # Initialize storage for results
    all_predictions_original = []
    all_predictions_perturbed = []
    all_perturbation_masks = []
    all_cell_indices = []
    
    # Process each batch
    print("Running inference...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            batch_data = batch_data.to(device)
            
            # Get original predictions
            predictions_original = predict(model, batch_data, inject=inject_feature is not None)
            
            # Store cell indices for mapping back to original data
            batch_cell_indices = [cell_indices_mapping[i] for i in range(batch_data.num_graphs)]
            
            # Apply perturbation mask to batch
            batch_data_perturbed, perturbation_mask = _apply_perturbation_mask_to_batch(
                batch_data, adata_result, perturbation_mask_key, batch_cell_indices
            )
            predictions_perturbed = predict(model, batch_data_perturbed, inject=inject_feature is not None)
            
            # Store results
            all_predictions_original.append(predictions_original.cpu())
            all_predictions_perturbed.append(predictions_perturbed.cpu())
            all_perturbation_masks.append(perturbation_mask.cpu())
            all_cell_indices.extend(batch_cell_indices)
    
    # Concatenate all results
    predictions_original = torch.cat(all_predictions_original, dim=0)
    predictions_perturbed = torch.cat(all_predictions_perturbed, dim=0)
    perturbation_mask = torch.cat(all_perturbation_masks, dim=0)
    
    # Map predictions back to original AnnData structure
    _add_predictions_to_adata(
        adata_result, predictions_original, predictions_perturbed, 
        perturbation_mask, all_cell_indices, gene_names
    )
    
    print("Perturbation prediction completed successfully!")
    return adata_result


def _create_graphs_from_adata(
    adata: ad.AnnData,
    k_hop: int,
    node_feature: str,
    inject_feature: Optional[str],
    celltypes_to_index: Dict[str, int],
    normalize_total: bool,
    debug: bool = False,
    debug_subset_size: int = 100,
    genept_embeddings_path: Optional[str] = None
) -> Tuple[List[Data], List[str], np.ndarray]:
    """
    Create graph objects by leveraging SpatialAgingCellDataset.process().
    
    Returns
    -------
    Tuple[List[Data], List[str], np.ndarray]
        - List of graph Data objects
        - List of gene names
        - Array of original cell indices (for mapping back to original data)
    """
    # Create a minimal dataset instance
    temp_dataset = SpatialAgingCellDataset(
        root=".",
        dataset_prefix="temp",
        target="expression",
        k_hop=k_hop,
        augment_hop=0,  # No augmentation for inference
        node_feature=node_feature,
        inject_feature=inject_feature,
        num_cells_per_ct_id=1,  # Take all cells
        center_celltypes='all',  # Use all cell types
        use_ids=None,  # Use all data
        raw_filepaths=None,  # We'll set this after saving the file
        gene_list=None,
        celltypes_to_index=celltypes_to_index,
        normalize_total=normalize_total,
        genept_embeddings_path=genept_embeddings_path
    )
    
    # Save the AnnData to a temporary file in the dataset's processed directory
    temp_filepath = os.path.join(temp_dataset.processed_dir, "temp_adata.h5ad")
    os.makedirs(temp_dataset.processed_dir, exist_ok=True)
    adata.write(temp_filepath)
    
    # Set the filepath and process
    temp_dataset.raw_filepaths = [temp_filepath]
    temp_dataset.process()
    
    # Load all the created graphs
    graphs = []
    for i in range(len(temp_dataset)):
        graph_data = temp_dataset.get(i)
        graphs.append(graph_data)
    
    # Get gene names from the dataset
    gene_names = temp_dataset.gene_names
    
    # Create cell indices mapping (since we're using all cells)
    cell_indices = list(range(adata.shape[0]))
    
    # Clean up the temporary file
    if os.path.exists(temp_filepath):
        os.unlink(temp_filepath)
    
    return graphs, gene_names, cell_indices


def _apply_perturbation_mask_to_batch(
    batch_data: Data,
    adata: ad.AnnData,
    perturbation_mask_key: str,
    batch_cell_indices: List[int]
) -> Tuple[Data, torch.Tensor]:
    """
    Applies a perturbation mask to a batch of PyG Data objects.
    
    This function modifies the 'x' attribute of the graph Data objects in the batch
    to reflect the perturbed expression values.
    
    Parameters
    ----------
    batch_data : torch_geometric.data.Data
        Batch of graph Data objects.
    adata : anndata.AnnData
        Original AnnData object containing the perturbation mask.
    perturbation_mask_key : str
        Key in adata.obsm containing the perturbation mask (cells x genes matrix of multipliers).
    batch_cell_indices : List[int]
        List of original cell indices corresponding to the batch.
        
    Returns
    -------
    Tuple[torch_geometric.data.Data, torch.Tensor]
        - Modified batch of graph Data objects.
        - Tensor of perturbation masks for the batch.
    """
    perturbation_mask = torch.zeros((batch_data.num_graphs, adata.X.shape[1]), dtype=torch.float32)
    
    for i, cell_idx in enumerate(batch_cell_indices):
        if cell_idx < len(adata.obsm[perturbation_mask_key]):
            perturbation_mask[i, :] = adata.obsm[perturbation_mask_key][cell_idx, :]
        else:
            # If cell_idx is out of bounds, it means the cell was not in the original adata
            # or the perturbation mask was not large enough.
            # For now, we'll set it to zeros, which will result in no perturbation for this cell.
            # A more robust solution might involve padding or handling this case.
            pass
    
    # Apply perturbation mask to the 'x' attribute of each graph in the batch
    for i in range(batch_data.num_graphs):
        # Get the perturbation multiplier for this cell
        perturbation_multiplier = perturbation_mask[i, :]
        
        # Apply the perturbation to the 'x' attribute of the graph
        # This modifies the 'x' attribute in place
        batch_data.x[i, :] = batch_data.x[i, :] * perturbation_multiplier
    
    return batch_data, perturbation_mask


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
        'T cell': {'Igf2': 0.0},  
        'NSC': {'Sox9': 2.0},         
        'Pericyte': {'Ccl4': 0.5}    
    }
    # Create perturbation mask and get file path
    adata_file_path = create_perturbation_mask(adata, perturbation_dict)
    
    # Example 1: Using pretrained model (inference mode)
    print("=== Example 1: Using pretrained model ===")
    adata_perturbed = predict_perturbation_effects(
        adata_file_path, 
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
    
    # Example 2: Training from scratch (training mode)
    print("\n=== Example 2: Training from scratch ===")
    adata_perturbed_trained = predict_perturbation_effects(
        adata_file,  # Now this is a file path
        model_path=None,  # No model path - trains new model
        dataset="aging_coronal",
        base_path="/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/raw",
        k_hop=2,
        augment_hop=2,
        center_celltypes="T cell,NSC,Pericyte",
        node_feature="expression",
        inject_feature="None",
        debug=True,
        debug_subset_size=50,
        learning_rate=1e-4,
        loss="mse",
        epochs=50  # Fewer epochs for demo
    )
    
    # Visualize results from pretrained model
    visualize_perturbation_effects(adata_perturbed)
