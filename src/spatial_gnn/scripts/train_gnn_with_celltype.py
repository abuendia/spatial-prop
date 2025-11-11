import numpy as np
import pickle
from typing import Tuple, Union, Optional, List
import argparse
import tqdm
import scanpy as sc
import json
import os
import sys
from collections import Counter

import torch
from torch_geometric.loader import DataLoader

from spatial_gnn.models.gnn_model import GNN, train, test, BMCLoss, Neg_Pearson_Loss, WeightedL1Loss, _get_expression_params, _get_celltype_params
from spatial_gnn.datasets.spatial_dataset import SpatialAgingCellDataset
from spatial_gnn.utils.dataset_utils import get_dataset_config, split_anndata_train_test
from spatial_gnn.utils.logging_utils import setup_logging_to_file
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
    model_save_dir: Optional[str] = None,
    stratify_by: Optional[str] = None,
    genept_embeddings: Optional[str] = None,
    genept_strategy: Optional[str] = None,
    predict_celltype: bool = False,
    train_multitask: bool = False,
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
    genept_embeddings : Optional[str], default=None
        Path to file containing GenePT embeddings
    train_multitask : bool, default=False
        Whether to train a multitask model
    Returns
    -------
    Tuple[GNN, str]
        - Trained model
        - Path to saved model
    """
    # Split data if not present in config
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

    if genept_embeddings is not None:
        with open(genept_embeddings, "rb") as f:
            genept_embeddings_raw = pickle.load(f)
        # Convert all keys to uppercase
        genept_embeddings = {k.upper(): v for k, v in genept_embeddings_raw.items()}

    # Load data
    all_train_data = []
    all_test_data = []
    
    for idx, f in tqdm.tqdm(enumerate(train_dataset.processed_file_names), total=len(train_dataset.processed_file_names)):
        if debug and idx > debug_subset_size:
            break
        batch_list = torch.load(os.path.join(train_dataset.processed_dir, f), weights_only=False)  # list[Data]
        all_train_data.extend(batch_list)
    
    for idx, f in tqdm.tqdm(enumerate(test_dataset.processed_file_names), total=len(test_dataset.processed_file_names)):
        if debug and idx > debug_subset_size:
            break
        batch_list = torch.load(os.path.join(test_dataset.processed_dir, f), weights_only=False)
        all_test_data.extend(batch_list)
    
    train_loader = DataLoader(all_train_data, batch_size=512, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(all_test_data, batch_size=512, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"Train samples: {len(all_train_data)}", flush=True)
    print(f"Test samples: {len(all_test_data)}", flush=True)

    # Determine if we're using GenePT embeddings
    use_genept = genept_embeddings is not None
    gene_names = [gene.upper() for gene in train_dataset.gene_names]

    model = GNN(
        hidden_channels=64,
        input_dim=int(train_dataset.get(0).x.shape[1]),
        output_dim=len(train_dataset.get(0).y),
        inject_dim=int(train_dataset.get(0).inject.shape[1]) if inject is True else 0,
        method="GIN", 
        pool="add", 
        num_layers=k_hop,
        genept_embeddings=genept_embeddings,
        genept_strategy=genept_strategy,  
        celltypes_to_index=celltypes_to_index,
        predict_celltype=predict_celltype,
        train_multitask=train_multitask,
    )
    model.to(device)
    print(f"Model initialized on {device}")

    if predict_celltype:
        label_counts = Counter()
        for batch in tqdm.tqdm(train_loader, total=len(train_loader), desc="Counting cell type labels"):
            # map string label -> integer index
            center_celltypes_idx = [model.celltypes_to_index[item[0]] for item in batch.center_celltype]
            label_counts.update(center_celltypes_idx)

        print(f"Label counts (by index): {label_counts}", flush=True)

        total_labels = sum(label_counts.values())
        num_classes = len(model.celltypes_to_index)

        freqs = [label_counts[i] / total_labels for i in range(num_classes)]
        class_weights = [1.0 / (f + 1e-12) for f in freqs]        # inverse frequency with stability
        class_weights = torch.tensor(class_weights, device=device)
        print(f"Freqs: {freqs}", flush=True)
        print(f"Class weights: {class_weights}", flush=True)
    else:
        class_weights = None

    # Setup optimizers
    if predict_celltype and train_multitask:
        expr_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        ct_optimizer = None
    elif predict_celltype and not train_multitask:
        expr_params, expr_param_ids = _get_expression_params(model)
        ct_params, ct_param_ids = _get_celltype_params(model)
        expr_optimizer = torch.optim.Adam(expr_params, lr=learning_rate)
        ct_optimizer = torch.optim.Adam(ct_params, lr=learning_rate)
    else:
        expr_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        ct_optimizer = None

    # Setup loss function
    if loss == "mse":
        criterion = torch.nn.MSELoss()
    elif loss == "l1":
        criterion = torch.nn.L1Loss()
    elif loss == "weightedl1":
        criterion = WeightedL1Loss(zero_weight=1, nonzero_weight=10)
    elif loss == "balanced_mse":
        criterion = BMCLoss(0.1)
        expr_optimizer.add_param_group({'params': criterion.noise_sigma, 'lr': learning_rate, 'name': 'noise_sigma'})
    elif loss == "npcc":
        criterion = Neg_Pearson_Loss()
    else:
        raise ValueError(f"Loss '{loss}' is not recognized!")

    # Training loop
    model_type = "GenePT" if use_genept else "Baseline"
    print(f"Starting {model_type} training for {epochs} epochs...")
    best_score = np.inf
    training_results = {"metric": loss, "epoch": [], "train": [], "test": [], "test_spearman": [], "test_celltype_accuracy": []}

    for epoch in range(1, epochs + 1):
        # Training
        train(model, train_loader, criterion, expr_optimizer, ct_optimizer, gene_names=gene_names, inject=inject, device=device, celltype_weight=1.0, class_weights=class_weights)


        train_score, _, _ = test(model, train_loader, loss, criterion, gene_names=gene_names, inject=inject, device=device)
        test_score, test_spearman, test_celltype_accuracy = test(model, test_loader, loss, criterion, gene_names=gene_names, inject=inject, device=device)

        # Save best model
        if test_score < best_score:
            save_dir = model_save_dir
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            best_score = test_score
        
        # Log results
        prefix = f'[{model_type}]' if use_genept else ''
        if loss == "mse":
            print(f'{prefix} Epoch: {epoch:03d}, Train MSE: {train_score:.4f}, Test MSE: {test_score:.4f}, Test Spearman: {test_spearman:.4f}, Test Celltype Accuracy: {test_celltype_accuracy:.4f}', flush=True)
        elif loss == "l1":
            print(f'{prefix} Epoch: {epoch:03d}, Train L1: {train_score:.4f}, Test L1: {test_score:.4f}, Test Spearman: {test_spearman:.4f}, Test Celltype Accuracy: {test_celltype_accuracy:.4f}', flush=True)
        elif loss == "weightedl1":
            print(f'{prefix} Epoch: {epoch:03d}, Train WL1: {train_score:.4f}, Test WL1: {test_score:.4f}, Test Spearman: {test_spearman:.4f}, Test Celltype Accuracy: {test_celltype_accuracy:.4f}', flush=True)
        elif loss == "balanced_mse":
            print(f'{prefix} Epoch: {epoch:03d}, Train BMC: {train_score:.4f}, Test BMC: {test_score:.4f}, Test Spearman: {test_spearman:.4f}, Test Celltype Accuracy: {test_celltype_accuracy:.4f}', flush=True)
        elif loss == "npcc":
            print(f'{prefix} Epoch: {epoch:03d}, Train NPCC: {train_score:.4f}, Test NPCC: {test_score:.4f}, Test Spearman: {test_spearman:.4f}, Test Celltype Accuracy: {test_celltype_accuracy:.4f}', flush=True)
            
        training_results["epoch"].append(epoch)
        training_results["train"].append(train_score)    
        training_results["test"].append(test_score)
        training_results["test_spearman"].append(test_spearman)
        training_results["test_celltype_accuracy"].append(test_celltype_accuracy)
    
    # Save final model
    save_dir = model_save_dir
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
        "num_cells_per_ct_id": num_cells_per_ct_id,
        "genept_embeddings": use_genept,
        "genept_strategy": genept_strategy,
        "train_multitask": train_multitask,
    }
    
    config_path = os.path.join(model_save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"{model_type} training completed. Model saved to {final_model_path}")
    print(f"Configuration saved to {config_path}")
    
    # Save training results
    with open(os.path.join(model_save_dir, "training.pkl"), 'wb') as f:
        pickle.dump(training_results, f)
    print(f"{model_type} training logs saved")

    return test_loader, gene_names, (model, model_config, final_model_path)


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
    parser.add_argument("--center_celltypes", help="cell type labels to center graphs on, separated by comma. Use 'all' for all cell types or 'none' for no cell type filtering or 'infer' to infer from the data", type=str, required=True)
    parser.add_argument("--node_feature", help="node features key, e.g. 'celltype_age_region'", type=str, required=True)
    parser.add_argument("--inject_feature", help="inject features key, e.g. 'center_celltype'", type=str, required=True)
    parser.add_argument("--learning_rate", help="learning rate", type=float, required=True)
    parser.add_argument("--loss", help="loss: balanced_mse, npcc, mse, l1", type=str, required=True)
    parser.add_argument("--epochs", help="number of epochs", type=int, required=True)
    parser.add_argument("--num_cells_per_ct_id", help="number of cells per cell type to use for training", type=int, default=100)
    parser.add_argument("--gene_list", help="Path to file containing list of genes to use (optional)", type=str, default=None)
    parser.add_argument("--device", help="device to use", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--do_eval", action='store_true', help="Enable evaluation mode")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode with subset of data for quick testing")
    parser.add_argument("--debug_subset_size", type=int, default=10, help="Number of batches to use in debug mode (default: 2)")
    parser.add_argument("--genept_embeddings", help="Path to file containing GenePT embeddings", type=str, default=None)
    parser.add_argument("--genept_strategy", help="Strategy to use for GenePT embeddings", type=str, default=None) # early_fusion, late_fusion, xattn
    parser.add_argument("--log_to_terminal", action='store_true', help="Log to terminal in addition to file")
    parser.add_argument("--predict_celltype", action='store_true', help="Enable cell type prediction")
    parser.add_argument("--train_multitask", action='store_true', help="Enable training a multitask model")

    # AnnData-specific arguments
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data to use for testing when using AnnData (default: 0.2)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility when using AnnData (default: 42)")
    parser.add_argument("--stratify_by", type=str, default=None, help="Column name in AnnData.obs to stratify the train/test split by (e.g., 'celltype')")
    
    args = parser.parse_args()

    if args.dataset in ["allen", "liverperturb"]:
        args.normalize_total = False
    else:
        args.normalize_total = True

    # Validate arguments
    if args.dataset and not args.base_path:
        parser.error("--base_path is required when using --dataset")
    if args.anndata and args.base_path:
        parser.error("--base_path should not be specified when using --anndata")

    # Determine dataset name
    if args.dataset:
        dataset_name = args.dataset
    elif args.anndata:
        dataset_name = args.exp_name if args.exp_name else "unknown"
    else:
        dataset_name = "unknown"
    
    # Create save directory structure
    exp_dir_name = f"{dataset_name}_expression_{args.k_hop}hop_{args.augment_hop}augment_{args.node_feature}_{args.inject_feature}"
    if args.debug:
        exp_dir_name = f"DEBUG_{exp_dir_name}"

    model_dir_name = args.loss + f"_{args.learning_rate:.0e}".replace("-", "n")
    model_save_dir = os.path.join(f"output/{args.exp_name}", exp_dir_name, model_dir_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Set up logging
    if args.log_to_terminal:
        print(f"Logging to terminal in addition to file", flush=True)
    else:
        print(f"Logging to file only", flush=True)
        log_file = setup_logging_to_file(os.path.join(f"output/{args.exp_name}", exp_dir_name)) 

    test_loader, gene_names, (model, model_config, _) = train_model_from_scratch(
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
        stratify_by=args.stratify_by,
        genept_embeddings=args.genept_embeddings,
        model_save_dir=model_save_dir,
        genept_strategy=args.genept_strategy, 
        predict_celltype=args.predict_celltype,
        train_multitask=args.train_multitask,
    )

    if args.do_eval:
        # Baseline eval
        eval_model(
            model=model,
            test_loader=test_loader,
            save_dir=model_save_dir,
            device=args.device,
            inject=False,
            gene_names=gene_names
        )


if __name__ == "__main__":
    main()
