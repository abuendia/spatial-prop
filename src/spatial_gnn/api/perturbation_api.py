"""
Training and inference API for spatial GNN perturbation predictions.
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

from spatial_gnn.scripts.train_gnn import train_model_from_scratch
from spatial_gnn.utils.dataset_utils import load_model_from_path
from spatial_gnn.models.gnn_model import predict
from spatial_gnn.datasets.spatial_dataset import SpatialAgingCellDataset
from spatial_gnn.utils.dataset_utils import get_dataset_config, create_dataloader_from_dataset
from spatial_gnn.scripts.eval_gnn_expression import eval_model


def train_perturbation_model(
    k_hop: int,
    augment_hop: int,
    center_celltypes: Union[str, List[str], None],
    node_feature: str,
    learning_rate: float,
    loss: str,
    epochs: int,
    num_cells_per_ct_id: int,
    inject_feature: Optional[str] = None,
    dataset: Optional[str] = None,
    base_path: Optional[str] = None,
    file_path: Optional[str] = None,
    train_ids: Optional[List[str]] = None,
    test_ids: Optional[List[str]] = None,
    exp_name: Optional[str] = None,
    gene_list: Optional[List[str]] = None,
    normalize_total: bool = True,
    predict_celltype: bool = False,
    pool: Optional[str] = None,
    predict_residuals: bool = False,
    do_eval: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs
) -> Tuple[Any, Dict, str]:
    """
    Train a GNN model for perturbation prediction from scratch.
    
    Parameters
    ----------
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
    # Create save directory structure
    exp_dir_name = f"{dataset}_expression_{k_hop}hop_{augment_hop}augment_{node_feature}_{inject_feature}"
    model_dir_name = loss + f"_{learning_rate:.0e}".replace("-", "n")
    model_save_dir = os.path.join(f"output/{exp_name}", exp_dir_name, model_dir_name)
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"Model will be saved to: ./{model_save_dir}")

    test_loader, gene_names, (model, model_config, trained_model_path) = train_model_from_scratch(
        dataset=dataset,
        base_path=base_path,
        model_save_dir=model_save_dir,
        exp_name=exp_name,
        file_path=file_path,
        train_ids=train_ids,
        test_ids=test_ids,
        k_hop=k_hop,
        augment_hop=augment_hop,
        center_celltypes=center_celltypes,
        node_feature=node_feature,
        inject_feature=inject_feature,
        learning_rate=learning_rate,
        loss=loss,
        epochs=epochs,
        num_cells_per_ct_id=num_cells_per_ct_id,
        gene_list=gene_list,
        normalize_total=normalize_total,
        predict_celltype=predict_celltype,
        pool=pool,
        predict_residuals=predict_residuals,
        device=device
    )
    if do_eval:
        eval_model(
            model=model,
            test_loader=test_loader,
            save_dir=model_save_dir,
            device=device,
            inject=False,
            gene_names=gene_names,
        )
    print(f"Training completed. Model saved to: {trained_model_path}")
    return test_loader, gene_names, (model, model_config, trained_model_path)


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
        # ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        perturbed_adata.write(save_path)
        print(f"Saved AnnData with perturbation input to: {save_path}")

    return save_path


def predict_perturbation_effects(
    adata_path: str,
    model_path: str,
    exp_name: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_ids: Optional[List[str]] = None,
    whole_tissue: bool = True,
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

    if use_ids is None:
        use_ids = test_ids

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
        whole_tissue=whole_tissue,
        use_ids=use_ids,
        raw_filepaths=[adata_path],
        celltypes_to_index=celltypes_to_index,
        normalize_total=True,
    )
    test_dataset.process()

    perturbed_test_dataset = SpatialAgingCellDataset(
        subfolder_name="predict",
        dataset_prefix=exp_name,
        target="expression",
        k_hop=2,
        augment_hop=2,
        node_feature="expression",
        inject_feature=None,
        num_cells_per_ct_id=100,
        center_celltypes="all",
        whole_tissue=True,
        use_ids=use_ids,
        raw_filepaths=[adata_path],
        celltypes_to_index=celltypes_to_index,
        normalize_total=True,
        perturbation_mask_key="perturbed_input",
        use_perturbed_expression=True,
    )
    perturbed_test_dataset.process()

    _, test_loader = create_dataloader_from_dataset(
        dataset=test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    _, perturbed_test_loader = create_dataloader_from_dataset(
        dataset=perturbed_test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # Get predictions for both unperturbed and perturbed data, and apply tempering
    adata_result = predict(
        model, 
        test_adata, 
        test_loader, 
        perturbed_dataloader=perturbed_test_loader,
        use_ids=use_ids,
        temper_method="distribution_renormalize",
        device=device
    )
    print("Perturbation prediction completed successfully!")
    return adata_result
