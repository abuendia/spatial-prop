import numpy as np
import pickle
from typing import Tuple, Union, Optional, List
import argparse
import tqdm
import scanpy as sc
import json
import os

import torch
from torch_geometric.loader import DataLoader

from spatial_gnn.models.gnn_model import GNN, train, test, BMCLoss, Neg_Pearson_Loss, WeightedL1Loss
from spatial_gnn.datasets.spatial_dataset import SpatialAgingCellDataset
from spatial_gnn.utils.dataset_utils import get_dataset_config, split_anndata_train_test
from spatial_gnn.scripts.model_performance import eval_model


def train_model_from_scratch(
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
    exp_name: Optional[str] = None,
    adata_path: Optional[str] = None,
    gene_list: Optional[List[str]] = None,
    normalize_total: bool = True,
    debug: bool = False,
    debug_subset_size: int = 2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_by: Optional[str] = None
) -> Tuple[GNN, str]:
    """
    Train a GNN model from scratch.
    
    It supports two modes:
    1. Dataset mode: Provide dataset name and base_path
    2. AnnData mode: Provide AnnData object directly
    
    Parameters
    ----------
    k_hop : int
        k-hop neighborhood size
    augment_hop : int
        Number of hops for graph augmentation
    center_celltypes : Union[str, List[str], None]
        Cell types to center graphs on
    node_feature : str
        Node feature type
    inject_feature : Optional[str]
        Inject feature type
    learning_rate : float
        Learning rate for training
    loss : str
        Loss function type
    epochs : int
        Number of training epochs
    dataset : Optional[str], default=None
        Dataset name (aging_coronal, aging_sagittal, etc.) - required if not using AnnData
    base_path : Optional[str], default=None
        Base path to the data directory - required if not using AnnData
    exp_name : Optional[str], default=None
        Experiment name
    adata_path : Optional[str], default=None
        Path to AnnData object to use directly - required if not using dataset+base_path
    gene_list : Optional[List[str]], default=None
        List of genes to use
    normalize_total : bool, default=True
        Whether to normalize total gene expression
    debug : bool, default=False
        Enable debug mode
    debug_subset_size : int, default=2
        Number of batches in debug mode
    device : str, default="cuda" if available else "cpu"
        Device to train on
    test_size : float, default=0.2
        Proportion of data to use for testing (only used in AnnData mode)
    random_state : int, default=42
        Random seed for reproducibility (only used in AnnData mode)
    stratify_by : Optional[str], default=None
        Column name in adata.obs to stratify the split by (only used in AnnData mode)
        
    Returns
    -------
    Tuple[GNN, str]
        - Trained model
        - Path to saved model
    """
    if adata_path is not None:    
        adata = sc.read_h5ad(adata_path)  
        train_ids, test_ids = split_anndata_train_test(
            adata, test_size=test_size, random_state=random_state, stratify_by=stratify_by
        )
        train_ids = list(train_ids)
        test_ids = list(test_ids)
        celltypes_to_index = {ct: i for i, ct in enumerate(adata.obs["celltype"].unique())}
        file_path = adata_path
    else:
        _, file_path, train_ids, test_ids, celltypes_to_index = get_dataset_config(dataset, base_path)
    
    if inject_feature is not None and inject_feature.lower() == "none":
        inject_feature = None
        inject = False
    else:
        inject = True

    print(f"Training on device: {device}", flush=True)

    if dataset is not None:
        exp_name = dataset

    train_dataset = SpatialAgingCellDataset(
        subfolder_name="train",
        dataset_prefix=exp_name,
        target="expression",
        k_hop=k_hop,
        augment_hop=augment_hop,
        node_feature=node_feature,
        inject_feature=inject_feature,
        num_cells_per_ct_id=num_cells_per_ct_id,
        center_celltypes=center_celltypes,
        use_ids=train_ids,
        raw_filepaths=[file_path],
        gene_list=gene_list,
        celltypes_to_index=celltypes_to_index,
        normalize_total=normalize_total,
        debug=debug,
        overwrite=False,
        use_mp=False,
    )

    test_dataset = SpatialAgingCellDataset(
        subfolder_name="test",
        dataset_prefix=exp_name,
        target="expression",
        k_hop=k_hop,
        augment_hop=augment_hop,
        node_feature=node_feature,
        inject_feature=inject_feature,
        num_cells_per_ct_id=num_cells_per_ct_id,
        center_celltypes=center_celltypes,
        use_ids=test_ids,
        raw_filepaths=[file_path],
        gene_list=gene_list,
        celltypes_to_index=celltypes_to_index,
        normalize_total=normalize_total,
        debug=debug,
        overwrite=False,
        use_mp=False,
    )
    
    # Process datasets
    test_dataset.process()
    print("Finished processing test dataset", flush=True)
    train_dataset.process()
    print("Finished processing train dataset", flush=True)

    # Load data
    all_train_data = []
    all_test_data = []
    
    # Get file names to load - use subset if in debug mode
    if debug:
        train_files = train_dataset.processed_file_names[:debug_subset_size]
        test_files = test_dataset.processed_file_names[:debug_subset_size]
    else:
        train_files = train_dataset.processed_file_names
        test_files = test_dataset.processed_file_names
    
    all_train_data = []
    for f in tqdm.tqdm(train_files):
        batch_list = torch.load(os.path.join(train_dataset.processed_dir, f), weights_only=False)  # list[Data]
        all_train_data.extend(batch_list)

    all_test_data = []
    for f in tqdm.tqdm(test_files):
        batch_list = torch.load(os.path.join(test_dataset.processed_dir, f), weights_only=False)
        all_test_data.extend(batch_list)
    
    train_loader = DataLoader(all_train_data, batch_size=512, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(all_test_data, batch_size=512, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"Train samples: {len(all_train_data)}", flush=True)
    print(f"Test samples: {len(all_test_data)}", flush=True)

    # Initialize model
    if inject is True:
        model = GNN(
            hidden_channels=64,
            input_dim=int(train_dataset.get(0).x.shape[1]),
            output_dim=len(train_dataset.get(0).y),
            inject_dim=int(train_dataset.get(0).inject.shape[1]),
            method="GIN", 
            pool="add", 
            num_layers=k_hop
        )
    else:
        model = GNN(
            hidden_channels=64,
            input_dim=int(train_dataset.get(0).x.shape[1]),
            output_dim=len(train_dataset.get(0).y),
            method="GIN", 
            pool="add", 
            num_layers=k_hop
        )
    model.to(device)
    print(f"Model initialized on {device}")

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Setup loss function
    if loss == "mse":
        criterion = torch.nn.MSELoss()
    elif loss == "l1":
        criterion = torch.nn.L1Loss()
    elif loss == "weightedl1":
        criterion = WeightedL1Loss(zero_weight=1, nonzero_weight=10)
    elif loss == "balanced_mse":
        criterion = BMCLoss(0.1)
        optimizer.add_param_group({'params': criterion.noise_sigma, 'lr': learning_rate, 'name': 'noise_sigma'})
    elif loss == "npcc":
        criterion = Neg_Pearson_Loss()
    else:
        raise ValueError(f"Loss '{loss}' is not recognized!")

    # Training loop
    print(f"Starting training for {epochs} epochs...")
    best_score = np.inf
    training_results = {"metric": loss, "epoch": [], "train": [], "test": []}

    # Create directory to save results
    exp_dir_name = f"{k_hop}hop_{augment_hop}augment_{node_feature}_{inject_feature}_{learning_rate:.0e}lr_{loss}_{epochs}epochs"
    if debug:
        exp_dir_name = f"DEBUG_{exp_dir_name}"
    model_dir_name = loss+f"_{learning_rate:.0e}".replace("-","n")
    
    save_dir = os.path.join("results/gnn", train_dataset.processed_dir.split("/")[-2], model_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(1, epochs + 1):
        # Training
        train(model, train_loader, criterion, optimizer, inject=inject, device=device)
        
        # Evaluation
        train_score = test(model, train_loader, loss, criterion, inject=inject, device=device)
        test_score = test(model, test_loader, loss, criterion, inject=inject, device=device)
        
        # Save best model
        if test_score < best_score:
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            best_score = test_score
        
        # Log results
        if loss == "mse":
            print(f'Epoch: {epoch:03d}, Train MSE: {train_score:.4f}, Test MSE: {test_score:.4f}', flush=True)
        elif loss == "l1":
            print(f'Epoch: {epoch:03d}, Train L1: {train_score:.4f}, Test L1: {test_score:.4f}', flush=True)
        elif loss == "weightedl1":
            print(f'Epoch: {epoch:03d}, Train WL1: {train_score:.4f}, Test WL1: {test_score:.4f}', flush=True)
        elif loss == "balanced_mse":
            print(f'Epoch: {epoch:03d}, Train BMC: {train_score:.4f}, Test BMC: {test_score:.4f}', flush=True)
        elif loss == "npcc":
            print(f'Epoch: {epoch:03d}, Train NPCC: {train_score:.4f}, Test NPCC: {test_score:.4f}', flush=True)
            
        training_results["epoch"].append(epoch)
        training_results["train"].append(train_score)    
        training_results["test"].append(test_score)

    # Save final model
    final_model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), final_model_path)
    
    # Save model configuration
    model_config = {
        "input_dim": int(train_dataset.get(0).x.shape[1]),
        "output_dim": len(train_dataset.get(0).y),
        "inject_dim": int(train_dataset.get(0).inject.shape[1]) if inject_feature is not None else 0,
        "num_layers": k_hop,
        "hidden_channels": 64,
        "method": "GIN",
        "pool": "add",
        "node_feature": node_feature,
        "inject_feature": inject_feature,
        "k_hop": k_hop,
        "augment_hop": augment_hop,
        "center_celltypes": center_celltypes,
        "normalize_total": normalize_total,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "data_file_path": file_path,
        "celltypes_to_index": celltypes_to_index,
        "num_cells_per_ct_id": num_cells_per_ct_id
    }
    
    config_path = os.path.join(save_dir, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"Training completed. Model saved to {final_model_path}")
    print(f"Model configuration saved to {config_path}")

    # Save training results
    with open(os.path.join(save_dir, "training.pkl"), 'wb') as f:
        pickle.dump(training_results, f)
    print("Training logs saved")

    return model, model_config, final_model_path, test_loader, save_dir


def main():
    parser = argparse.ArgumentParser()
    
    # Add mutually exclusive group for data input
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--dataset", help="Dataset to use (aging_coronal, aging_sagittal, exercise, reprogramming, allen, kukanja, pilot)", type=str)
    data_group.add_argument("--anndata", help="Path to AnnData file (.h5ad) to use directly", type=str)
    
    parser.add_argument("--exp_name", help="Experiment name", type=str, default=None)
    parser.add_argument("--base_path", help="Base path to the data directory (required if using --dataset)", type=str)
    parser.add_argument("--k_hop", help="k-hop neighborhood size", type=int, required=True)
    parser.add_argument("--augment_hop", help="number of hops to take for graph augmentation", type=int, required=True)
    parser.add_argument("--center_celltypes", help="cell type labels to center graphs on, separated by comma. Use 'all' for all cell types or 'none' for no cell type filtering", type=str, required=True)
    parser.add_argument("--node_feature", help="node features key, e.g. 'celltype_age_region'", type=str, required=True)
    parser.add_argument("--inject_feature", help="inject features key, e.g. 'center_celltype'", type=str, required=True)
    parser.add_argument("--learning_rate", help="learning rate", type=float, required=True)
    parser.add_argument("--loss", help="loss: balanced_mse, npcc, mse, l1", type=str, required=True)
    parser.add_argument("--epochs", help="number of epochs", type=int, required=True)
    parser.add_argument("--num_cells_per_ct_id", help="number of cells per cell type to use for training", type=int, default=100)
    parser.add_argument("--gene_list", help="Path to file containing list of genes to use (optional)", type=str, default=None)
    parser.add_argument("--normalize_total", action='store_true')
    parser.add_argument("--no-normalize_total", dest='normalize_total', action='store_false')
    parser.add_argument("--device", help="device to use", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--do_eval", action='store_true', help="Enable evaluation mode")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode with subset of data for quick testing")
    parser.add_argument("--debug_subset_size", type=int, default=10, help="Number of batches to use in debug mode (default: 2)")
    
    # AnnData-specific arguments
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data to use for testing when using AnnData (default: 0.2)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility when using AnnData (default: 42)")
    parser.add_argument("--stratify_by", type=str, default=None, help="Column name in AnnData.obs to stratify the train/test split by (e.g., 'celltype')")
    
    parser.set_defaults(normalize_total=True)
    args = parser.parse_args()

    # Validate arguments
    if args.dataset and not args.base_path:
        parser.error("--base_path is required when using --dataset")
    if args.anndata and args.base_path:
        parser.error("--base_path should not be specified when using --anndata")

    model, _, _, test_loader, save_dir = train_model_from_scratch(
        k_hop=args.k_hop,
        augment_hop=args.augment_hop,
        center_celltypes=args.center_celltypes,
        node_feature=args.node_feature,
        inject_feature=args.inject_feature,
        learning_rate=args.learning_rate,
        loss=args.loss,
        epochs=args.epochs,
        num_cells_per_ct_id=args.num_cells_per_ct_id,
        dataset=args.dataset,
        base_path=args.base_path,
        exp_name=args.exp_name,
        adata_path=args.anndata,
        gene_list=args.gene_list,
        normalize_total=args.normalize_total,
        debug=args.debug,
        debug_subset_size=args.debug_subset_size,
        device=args.device,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify_by=args.stratify_by
    )

    if args.do_eval:
        eval_model(
            model=model,
            test_loader=test_loader,
            save_dir=save_dir,
            device=args.device,
            inject=False
        )


if __name__ == "__main__":
    main()
