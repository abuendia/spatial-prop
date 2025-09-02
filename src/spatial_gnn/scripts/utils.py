import os
import json
from typing import Union, List, Optional, Tuple
import numpy as np

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph, one_hot
from scipy.sparse import issparse
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from spatial_gnn.scripts.aging_gnn_model import SpatialAgingCellDataset, predict, GNN


def load_dataset_config():
    """Load dataset configurations from JSON file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'datasets.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset configuration file not found at {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in dataset configuration file at {config_path}")


def parse_center_celltypes(center_celltypes: Union[str, List[str], None]) -> Union[str, List[str], None]:
    """
    Parse center_celltypes parameter consistently across the codebase.
    
    Parameters
    ----------
    center_celltypes : Union[str, List[str], None]
        Input center_celltypes parameter
        
    Returns
    -------
    Union[str, List[str], None]
        Parsed center_celltypes parameter
    """
    if isinstance(center_celltypes, str):
        if center_celltypes.lower() == 'none':
            return None
        elif center_celltypes.lower() == 'all':
            return 'all'
        else:
            return [ct.strip() for ct in center_celltypes.split(',')]
    return center_celltypes


def parse_gene_list(gene_list: Optional[str]) -> Optional[List[str]]:
    """
    Parse gene_list parameter consistently across the codebase.
    
    Parameters
    ----------
    gene_list : Optional[str]
        Path to file containing list of genes to use
        
    Returns
    -------
    Optional[List[str]]
        List of gene names, or None if file not found or not provided
    """
    if gene_list is None:
        return None
        
    if os.path.exists(gene_list):
        try:
            with open(gene_list, 'r') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            print(f"Warning: Error reading gene_list file {gene_list}: {e}. Using all genes.")
            return None
    else:
        print(f"Warning: gene_list file {gene_list} not found. Using all genes.")
        return None


def get_dataset_config(dataset: str, base_path: str) -> tuple:
    """
    Get dataset configuration and build common parameters.
    
    Parameters
    ----------
    dataset : str
        Dataset name
    base_path : str
        Base path to data directory
        
    Returns
    -------
    tuple
        (config, file_path, train_ids, test_ids, celltypes_to_index)
    """
    DATASET_CONFIGS = load_dataset_config()
    
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{dataset}' not found in configuration. Available: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset]
    file_path = os.path.join(base_path, config.get('file_path', config.get('file_name', '')))
    train_ids = config.get('train_ids', [])
    test_ids = config.get('test_ids', [])
    
    # Build cell type index
    celltypes_to_index = {}
    for ci, cellt in enumerate(config.get("celltypes", [])):
        celltypes_to_index[cellt] = ci
    
    return config, file_path, train_ids, test_ids, celltypes_to_index


def load_model_from_path(model_path: str, device: str) -> torch.nn.Module:
    """
    Load a trained GNN model using the saved config.json file.

    
    Parameters
    ----------
    model_path : str
        Path to the saved model state dictionary
    device : str
        Device to load the model on
        
    Returns
    -------
    torch.nn.Module
        Loaded GNN model with correct dimensions
    """
    # Get the model directory and config path
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, "model_config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Model config file not found at {config_path}. "
            "Please ensure the model was saved with a config.json file."
        )
    
    # Load the model configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded model configuration:")
    print(f"  - input_dim: {config['input_dim']}")
    print(f"  - output_dim: {config['output_dim']}")
    print(f"  - inject_dim: {config['inject_dim']}")
    print(f"  - num_layers: {config['num_layers']}")
    print(f"  - method: {config['method']}")
    print(f"  - pool: {config['pool']}")
    
    # Create the model with the saved configuration
    model = GNN(
        hidden_channels=config['hidden_channels'],
        input_dim=config['input_dim'],
        output_dim=config['output_dim'],
        inject_dim=config['inject_dim'],
        method=config['method'],
        pool=config['pool'],
        num_layers=config['num_layers']
    )
    
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    # Move to device and set to eval mode
    model.to(device)
    model.eval()
    
    return model


def split_anndata_train_test(
    adata: 'ad.AnnData',
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_by: Optional[str] = None
) -> Tuple['ad.AnnData', 'ad.AnnData', List[str], List[str]]:
    """
    Split an AnnData object into train and test sets.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Input AnnData object to split
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
    stratify_by : Optional[str], default=None
        Column name in adata.obs to stratify the split by (e.g., 'celltype')
        
    Returns
    -------
    Tuple[anndata.AnnData, anndata.AnnData, List[str], List[str]]
        - Training AnnData object
        - Testing AnnData object  
        - List of training cell IDs
        - List of testing cell IDs
    """
    # Set random seed
    np.random.seed(random_state)
    
    # Get all cell IDs
    all_cell_ids = adata.obs_names.tolist()
    n_cells = len(all_cell_ids)
    n_test = int(n_cells * test_size)
    
    if stratify_by is not None and stratify_by in adata.obs.columns:
        # Stratified split
        from sklearn.model_selection import train_test_split
        
        # Get stratification labels
        stratify_labels = adata.obs[stratify_by].values
        
        # Perform stratified split
        train_indices, test_indices = train_test_split(
            range(n_cells),
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels
        )
        
        train_ids = [all_cell_ids[i] for i in train_indices]
        test_ids = [all_cell_ids[i] for i in test_indices]
    else:
        # Random split
        test_indices = np.random.choice(n_cells, size=n_test, replace=False)
        train_indices = np.setdiff1d(range(n_cells), test_indices)
        
        train_ids = [all_cell_ids[i] for i in train_indices]
        test_ids = [all_cell_ids[i] for i in test_indices]
    
    # Create train and test AnnData objects
    train_adata = adata[train_ids].copy()
    test_adata = adata[test_ids].copy()
    
    print(f"Split {n_cells} cells into {len(train_ids)} training and {len(test_ids)} testing cells")
    
    return train_adata, test_adata, train_ids, test_ids


def extract_anndata_info(
    adata: 'ad.AnnData',
    center_celltypes: Union[str, List[str], None] = None,
    inject_feature: Optional[str] = None
) -> Tuple[dict, str, dict]:
    """
    Extract necessary information from AnnData object for training.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Input AnnData object
    center_celltypes : Union[str, List[str], None], default=None
        Cell types to center graphs on
    inject_feature : Optional[str], default=None
        Inject feature type
        
    Returns
    -------
    Tuple[dict, str, dict]
        - Configuration dictionary
        - File path (temporary)
        - Cell types to index mapping
    """
    if not ANNDATA_AVAILABLE:
        raise ImportError("AnnData is required for this function. Install with 'pip install anndata'")
    
    # Create a minimal config
    config = {
        'file_path': 'temp_anndata.h5ad',  # Will be set when saving
        'celltypes': adata.obs['celltype'].unique().tolist() if 'celltype' in adata.obs.columns else []
    }
    
    # Build cell type index mapping
    celltypes_to_index = {}
    if 'celltype' in adata.obs.columns:
        for ci, cellt in enumerate(adata.obs['celltype'].unique()):
            celltypes_to_index[cellt] = ci
    
    # Handle center_celltypes
    center_celltypes_parsed = parse_center_celltypes(center_celltypes)
    
    # Handle inject_feature
    if inject_feature is not None and inject_feature.lower() == "none":
        inject_feature = None
    
    # Create a temporary file path (will be set when actually saving)
    file_path = "temp_anndata.h5ad"
    
    return config, file_path, celltypes_to_index
