use_wandb = False

import numpy as np
import pickle
import os
from typing import Tuple, Union, Optional, List
import argparse
import tqdm

import torch
from torch_geometric.loader import DataLoader

from spatial_gnn.scripts.aging_gnn_model import GNN, train, test, BMCLoss, Neg_Pearson_Loss, WeightedL1Loss, SpatialAgingCellDataset
from spatial_gnn.scripts.utils import load_dataset_config

if use_wandb is True:
    import wandb    


def train_model_from_scratch(
    dataset: str,
    base_path: str,
    k_hop: int,
    augment_hop: int,
    center_celltypes: Union[str, List[str], None],
    node_feature: str,
    inject_feature: Optional[str],
    learning_rate: float,
    loss: str,
    epochs: int,
    gene_list: Optional[List[str]] = None,
    normalize_total: bool = True,
    debug: bool = False,
    debug_subset_size: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[GNN, str]:
    """
    Train a GNN model from scratch using the same approach as the main training script.
    
    This function can be called from other modules to train models programmatically.
    
    Parameters
    ----------
    dataset : str
        Dataset name (aging_coronal, aging_sagittal, etc.)
    base_path : str
        Base path to the data directory
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
    gene_list : Optional[List[str]], default=None
        List of genes to use
    normalize_total : bool, default=True
        Whether to normalize total gene expression
    debug : bool, default=False
        Enable debug mode
    debug_subset_size : int, default=100
        Number of samples in debug mode
    device : str, default="cuda" if available else "cpu"
        Device to train on
        
    Returns
    -------
    Tuple[GNN, str]
        - Trained model
        - Path to saved model
    """
    
    # Load dataset configurations
    DATASET_CONFIGS = load_dataset_config()

    # Validate dataset choice
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Dataset must be one of: {', '.join(DATASET_CONFIGS.keys())}")

    # Load parameters from dataset config
    dataset_config = DATASET_CONFIGS[dataset]
    train_ids = dataset_config['train_ids']
    test_ids = dataset_config['test_ids']
    file_path = os.path.join(base_path, dataset_config['file_name'])
    
    # Build cell type index
    celltypes_to_index = {}
    for ci, cellt in enumerate(dataset_config["celltypes"]):
        celltypes_to_index[cellt] = ci
    
    # Handle center_celltypes
    if isinstance(center_celltypes, str):
        if center_celltypes.lower() == 'none':
            center_celltypes = None
        elif center_celltypes.lower() == 'all':
            center_celltypes = 'all'
        else:
            center_celltypes = center_celltypes.split(",")
    
    # Handle inject_feature
    if inject_feature is not None and inject_feature.lower() == "none":
        inject_feature = None
    inject = inject_feature is not None
    
    print(f"Training on device: {device}", flush=True)

    # Initialize datasets
    train_dataset = SpatialAgingCellDataset(
        subfolder_name="train",
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
        normalize_total=normalize_total
    )

    test_dataset = SpatialAgingCellDataset(
        subfolder_name="test",
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
        normalize_total=normalize_total
    )
    
    # Process datasets
    test_dataset.process()
    print("Finished processing test dataset", flush=True)
    train_dataset.process()
    print("Finished processing train dataset", flush=True)

    # Apply debug mode subsetting if enabled
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

    # Load data
    all_train_data = []
    all_test_data = []
    
    # Get file names to load - use subset if in debug mode
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

    print(f"Train samples: {len(all_train_data)}", flush=True)
    print(f"Test samples: {len(all_test_data)}", flush=True)

    # Initialize model
    if inject:
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
    best_mse = np.inf
    training_results = {"metric": loss, "epoch": [], "train": [], "test": []}

    # Create directory to save results
    exp_name = f"{k_hop}hop_{augment_hop}augment_{node_feature}_{inject_feature}_{learning_rate:.0e}lr_{loss}_{epochs}epochs"
    if debug:
        exp_name = f"DEBUG_{exp_name}"
    
    model_dirname = loss + f"_{learning_rate:.0e}".replace("-", "n")
    if debug:
        model_dirname = f"DEBUG_{model_dirname}"
    
    save_dir = os.path.join("results/gnn", train_dataset.processed_dir.split("/")[-2], model_dirname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(1, epochs + 1):
        # Training
        train(model, train_loader, criterion, optimizer, inject=inject, device=device)
        
        # Evaluation
        train_mse = test(model, train_loader, loss, criterion, inject=inject, device=device)
        test_mse = test(model, test_loader, loss, criterion, inject=inject, device=device)
        
        # Save best model
        if test_mse < best_mse:
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            best_mse = test_mse
        
        # Log results
        if loss == "mse":
            print(f'Epoch: {epoch:03d}, Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}', flush=True)
        elif loss == "l1":
            print(f'Epoch: {epoch:03d}, Train L1: {train_mse:.4f}, Test L1: {test_mse:.4f}', flush=True)
        elif loss == "weightedl1":
            print(f'Epoch: {epoch:03d}, Train WL1: {train_mse:.4f}, Test WL1: {test_mse:.4f}', flush=True)
        elif loss == "balanced_mse":
            print(f'Epoch: {epoch:03d}, Train BMC: {train_mse:.4f}, Test BMC: {test_mse:.4f}', flush=True)
        elif loss == "npcc":
            print(f'Epoch: {epoch:03d}, Train NPCC: {train_mse:.4f}, Test NPCC: {test_mse:.4f}', flush=True)
            
        training_results["epoch"].append(epoch)
        training_results["train"].append(train_mse)    
        training_results["test"].append(test_mse)

    # Save final model
    final_model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training completed. Model saved to {final_model_path}")

    # Save training results
    with open(os.path.join(save_dir, "training.pkl"), 'wb') as f:
        pickle.dump(training_results, f)
    print("Training logs saved")

    return model, final_model_path


def main():
    # set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset to use (aging_coronal, aging_sagittal, exercise, reprogramming, allen, kukanja, pilot)", type=str, required=True)
    parser.add_argument("--base_path", help="Base path to the data directory", type=str, required=True)
    parser.add_argument("--k_hop", help="k-hop neighborhood size", type=int, required=True)
    parser.add_argument("--augment_hop", help="number of hops to take for graph augmentation", type=int, required=True)
    parser.add_argument("--center_celltypes", help="cell type labels to center graphs on, separated by comma. Use 'all' for all cell types or 'none' for no cell type filtering", type=str, required=True)
    parser.add_argument("--node_feature", help="node features key, e.g. 'celltype_age_region'", type=str, required=True)
    parser.add_argument("--inject_feature", help="inject features key, e.g. 'center_celltype'", type=str, required=True)
    parser.add_argument("--learning_rate", help="learning rate", type=float, required=True)
    parser.add_argument("--loss", help="loss: balanced_mse, npcc, mse, l1", type=str, required=True)
    parser.add_argument("--epochs", help="number of epochs", type=int, required=True)
    parser.add_argument("--gene_list", help="Path to file containing list of genes to use (optional)", type=str, default=None)
    parser.add_argument("--normalize_total", action='store_true')
    parser.add_argument("--no-normalize_total", dest='normalize_total', action='store_false')
    parser.add_argument("--device", help="device to use", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode with subset of data for quick testing")
    parser.add_argument("--debug_subset_size", type=int, default=100, help="Number of samples to use in debug mode (default: 100)")
    parser.set_defaults(normalize_total=True)
    args = parser.parse_args()

    train_model_from_scratch(
        dataset=args.dataset,
        base_path=args.base_path,
        k_hop=args.k_hop,
        augment_hop=args.augment_hop,
        center_celltypes=args.center_celltypes,
        node_feature=args.node_feature,
        inject_feature=args.inject_feature,
        learning_rate=args.learning_rate,
        loss=args.loss,
        epochs=args.epochs,
        gene_list=args.gene_list,
        normalize_total=args.normalize_total,
        debug=args.debug,
        debug_subset_size=args.debug_subset_size,
        device=args.device
    )


if __name__ == "__main__":
    main()
