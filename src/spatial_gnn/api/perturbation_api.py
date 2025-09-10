"""
API for spatial GNN perturbation predictions.
"""
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
from torch_geometric.loader import DataLoader
from scipy.sparse import issparse
from typing import List, Dict, Union, Optional, Tuple, Any
import os
import tqdm 


from spatial_gnn.scripts.train_gnn_model_expression import train_model_from_scratch
from spatial_gnn.utils.dataset_utils import load_model_from_path
from spatial_gnn.models.gnn_model import predict
from spatial_gnn.datasets.spatial_dataset import SpatialAgingCellDataset


def create_perturbation_mask(
    adata: ad.AnnData,
    perturbation_dict: Dict[str, Dict[str, float]],
    mask_key: str = 'perturbed_input',
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
    mask_key : str, default='perturbed_input'
        Key to store the perturbation mask in adata.obsm
    save_dir : Optional[str], default=None
        Directory to save the file. If None, uses current working directory.
        
    Returns
    -------
    str
        Path to the saved AnnData file with perturbation mask
    """
    # Create perturbation mask
    perturbed = np.ones((adata.shape[0], adata.shape[1]))
    
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
                vals = adata.X[cell_indices, gene_idx]
                if issparse(vals): 
                    vals = vals.A.ravel()
                else: 
                    vals = np.asarray(vals).ravel()
                perturbed[cell_indices, gene_idx] = multiplier * vals
                print(f"  - Gene '{gene_name}': multiplier = {multiplier}")
            else:
                print(f"Warning: Gene '{gene_name}' not found in data")
    
    # Add the perturbation mask to the AnnData
    adata.obsm[mask_key] = perturbed
    
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
    print(f"- Shape: {perturbed.shape}")
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
    inject_feature: Optional[str] = None,
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
    inject_feature : Optional[str], default=None
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
    perturbation_mask_key: str = 'perturbed_input',
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
    perturbation_mask_key : str, default='perturbed_input'
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
    create_perturbation_mask(adata, perturbation_dict, mask_key=perturbation_mask_key, save_path=adata_path)
    
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
        use_ids=model_config.get("test_ids", None),
        raw_filepaths=[adata_path],
        celltypes_to_index=model_config["celltypes_to_index"],
        normalize_total=model_config["normalize_total"],
        perturbation_mask_key=perturbation_mask_key
    )
    inference_dataset.process()
    
    # Load batch files the same way as training
    all_inference_data = []
    for f in tqdm.tqdm(inference_dataset.processed_file_names):
        batch_list = torch.load(os.path.join(inference_dataset.processed_dir, f), weights_only=False)
        all_inference_data.extend(batch_list)

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


if __name__ == "__main__":
    train_data_path = "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/raw/aging_coronal.h5ad"
    test_data_path = "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/raw/aging_coronal.h5ad"  # Using same data for demo
    
    print("=== Training a new perturbation model ===")
    model, model_config, model_path = train_perturbation_model(
        adata_path=train_data_path,
        exp_name="api_run",
        k_hop=2,
        augment_hop=2,
        center_celltypes="T cell,NSC,Pericyte",
        node_feature="expression",
        inject_feature="None",
        num_cells_per_ct_id=100,
        epochs=50,
    )
    
    # Define perturbations
    perturbation_dict = {
        'T cell': {'Igf2': 0.0},  
        'NSC': {'Sox9': 2.0},         
        'Pericyte': {'Ccl4': 0.5}    
    }
    model_path = "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/results/gnn/api_run_expression_100per_2hop_2C0aug_200delaunay_expressionFeat_TNP_NoneInject/DEBUG_weightedl1_1en04/best_model.pth"


    test_adata = sc.read_h5ad(test_data_path)
    test_data_path_perturbed = create_perturbation_mask(test_adata, perturbation_dict, save_path=test_data_path)

    print("\n=== Predicting perturbation effects ===")
    adata_perturbed = predict_perturbation_effects(
        adata_path=test_data_path_perturbed,
        exp_name="api_run",
        model_path=model_path,
        perturbation_dict=perturbation_dict,
        perturbation_mask_key="perturbed_input"
    )
    adata_perturbed.write("/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/perturbed/aging_coronal_perturbed_result.h5ad")
