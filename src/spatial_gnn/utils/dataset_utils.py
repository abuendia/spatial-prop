import os
import json
from typing import Union, List, Optional, Tuple
import numpy as np
import scanpy as sc

import torch
from sklearn.model_selection import train_test_split
import anndata as ad
from spatial_gnn.models.gnn_model import GNN


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
        elif center_celltypes.lower() == 'infer':
            return 'infer'
        else:
            return [ct.strip() for ct in center_celltypes.split(',')]
    return center_celltypes


def infer_center_celltypes_from_adata(adata_path):
    """
    Infer center cell types by finding the 3 cell types with the lowest cell counts.

    Parameters
    ----------
    adata_path : str
        Path to AnnData object to analyze for cell type counts
        AnnData object to analyze for cell type counts
        
    Returns
    -------
    list
        List of cell type names with the lowest counts (up to 3, or all if less than 3)
    """
    adata = sc.read_h5ad(adata_path)
    celltype_counts = adata.obs['celltype'].value_counts()

    if len(celltype_counts) <= 3:
        inferred_celltypes = celltype_counts.index.tolist()
        print(f"Inferred center cell types (all {len(inferred_celltypes)} available): {inferred_celltypes}")
    else:
        # Get the 3 with lowest counts
        inferred_celltypes = celltype_counts.nsmallest(3).index.tolist()
        print(f"Inferred center cell types (3 lowest counts): {inferred_celltypes}")
        print(f"Cell type counts: {celltype_counts[inferred_celltypes].to_dict()}")

    return inferred_celltypes


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
    config_path = os.path.join(model_dir, "config.json")
    
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
    
    return model, config


def split_anndata_train_test(
    adata: 'ad.AnnData',
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_by: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    """
    Split an AnnData object into train and test sets based on mouse_id.
    
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
    Tuple[List[str], List[str]]
        - List of training cell IDs
        - List of testing cell IDs
    """

    # Get unique mouse IDs
    unique_mouse_ids = adata.obs["mouse_id"].unique()
    
    if stratify_by is not None:
        # For stratification, we need to get the most common value per mouse
        # Group by mouse_id and get the most frequent value for stratify_by
        mouse_stratify_values = []
        for mouse_id in unique_mouse_ids:
            mouse_mask = adata.obs["mouse_id"] == mouse_id
            mouse_stratify = adata.obs.loc[mouse_mask, stratify_by].mode()
            if len(mouse_stratify) > 0:
                mouse_stratify_values.append(mouse_stratify.iloc[0])
            else:
                mouse_stratify_values.append(None)
        stratify_labels = mouse_stratify_values
    else:
        stratify_labels = None
    
    # Split unique mouse IDs
    train_mouse_ids, test_mouse_ids = train_test_split(
        unique_mouse_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels
    )

    return train_mouse_ids, test_mouse_ids


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