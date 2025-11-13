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


from spatial_gnn.scripts.train_gnn_with_celltype import train_model_from_scratch
from spatial_gnn.utils.dataset_utils import load_model_from_path
from spatial_gnn.models.gnn_model import predict
from spatial_gnn.datasets.spatial_dataset import SpatialAgingCellDataset
from spatial_gnn.utils.dataset_utils import get_dataset_config


def create_perturbation_input_matrix(
    adata: ad.AnnData,
    perturbation_dict: Dict[str, Dict[str, float]],
    mask_key: str = 'perturbed_input',
    save_path: Optional[str] = None,
    normalize_total: bool = True,
) -> str:
    """
    Store a full perturbed expression matrix in adata.obsm[mask_key] with the same 
    normalization as the training data.
    """
    perturbed_adata = adata.copy()

    X = perturbed_adata.X
    if issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)

    perturbed = X.copy()   # start from normalized expression

    for cell_type, gene_multipliers in perturbation_dict.items():
        cell_mask = perturbed_adata.obs['celltype'] == cell_type
        cell_indices = np.where(cell_mask)[0]

        if len(cell_indices) == 0:
            print(f"Warning: No cells found for cell type '{cell_type}'")
            continue

        print(f"Applying perturbations to {len(cell_indices)} cells of type '{cell_type}'")

        for gene_name, multiplier in gene_multipliers.items():
            if gene_name in perturbed_adata.var_names:
                gene_idx = perturbed_adata.var_names.get_loc(gene_name)

                # multiply existing expression by the factor
                perturbed[cell_indices, gene_idx] *= multiplier
                print(f"  - Gene '{gene_name}': multiplier = {multiplier}")
            else:
                print(f"Warning: Gene '{gene_name}' not found in data")

    if normalize_total:
        target_sum = X.shape[1]
        row_sums = perturbed.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid /0
        perturbed = perturbed / row_sums * target_sum
        perturbed_adata.obsm[mask_key] = perturbed

    if save_path is not None:
        perturbed_adata.write(save_path)
        print(f"Saved AnnData with perturbation input to: {save_path}")

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
    # Load the pretrained model
    test_adata = sc.read_h5ad(adata_path)

    model, model_config = load_model_from_path(model_path, device)
    print(f"Loaded pretrained model from: {model_path}")
    
    dataset = "aging_coronal"
    base_path = "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/raw"
    _, file_path, _, test_ids, celltypes_to_index = get_dataset_config(dataset, base_path)
    
    # Create graphs from the input AnnData
    print("Creating graphs from input data...")
    test_dataset = SpatialAgingCellDataset(
        subfolder_name="predict",
        dataset_prefix=exp_name,
        target="expression",
        k_hop=2,
        augment_hop=2,
        node_feature="expression",
        inject_feature=None,
        num_cells_per_ct_id=100,
        center_celltypes="all",
        use_ids=test_ids,
        raw_filepaths=[adata_path],
        gene_list=None,
        celltypes_to_index=celltypes_to_index,
        normalize_total=True,
        debug=True,
        overwrite=False,
        use_mp=False,
    )
    test_dataset.process()

    all_test_data = []
    for idx, f in tqdm.tqdm(enumerate(test_dataset.processed_file_names), total=len(test_dataset.processed_file_names)):
        batch_list = torch.load(os.path.join(test_dataset.processed_dir, f), weights_only=False)
        all_test_data.extend(batch_list)
    test_loader = DataLoader(all_test_data, batch_size=512, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    adata_result = predict(model, test_loader, test_adata, device)
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
    test_data_path = "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/raw/aging_coronal.h5ad" 
    model_path = "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/output/base_model/results/gnn/aging_coronal_expression_2hop_2augment_expression_none/weightedl1_1en04_GenePT_all_genes/best_model.pth"
    save_path = "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/perturbed_adata/aging_coronal_perturbed.h5ad"

    # Il6, Tnf, Ifng
    perturbation_dict = {
        'T cell': {'Il6': 10.0, 'Tnf': 10.0, 'Ifng': 10.0},    
        'Microglia': {'Il6': 10.0, 'Tnf': 10.0, 'Ifng': 10.0},          
    }
    test_adata = sc.read_h5ad(test_data_path)
    save_path = create_perturbation_input_matrix(test_adata, perturbation_dict, save_path=save_path, normalize_total=True)

    print("\n=== Predicting perturbation effects ===")
    adata_perturbed = predict_perturbation_effects(
        adata_path=save_path,
        exp_name="aging_coronal_perturbed_debug",
        model_path=model_path,
        perturbation_dict=perturbation_dict,
        perturbation_mask_key="perturbed_input"
    )
    adata_perturbed.write("/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/perturbed_adata/aging_coronal_pred_on_perturbed.h5ad")
