import torch
import numpy as np
from typing import List, Dict, Any
from torch_geometric.data import Data, Batch


def khop_mean_baseline(graph_data: Data, gene_idx: int) -> float:
    """
    Baseline 1: In k-hop graph, take the average expression over all cells for the given gene.
    Use that expression as the prediction for the center cell.
    
    Parameters
    ----------
    graph_data : Data
        PyTorch Geometric Data object containing:
        - x: node features (cells x genes expression matrix)
        - center_node: index of the center cell
        - y: target expression for center cell (all genes)
    gene_idx : int
        Index of the gene to predict
        
    Returns
    -------
    float
        Predicted expression value for the center cell and specified gene
    """
    # Extract expression data for all cells in the k-hop neighborhood
    # x contains node features (expression data for all cells)
    expression_matrix = graph_data.x  # Shape: (num_cells, num_genes)
    center_node = graph_data.center_node
    
    # Exclude the center cell from the average (it has zeroed features)
    all_indices = torch.arange(expression_matrix.shape[0])
    neighbor_indices = all_indices[all_indices != center_node]
    
    # Get average expression across all neighbor cells for the specified gene
    mean_expression = torch.mean(expression_matrix[neighbor_indices, gene_idx])
    
    return mean_expression.item()


def center_celltype_mean_baseline(graph_data: Data, gene_idx: int) -> float:
    """
    Baseline 2: In k-hop graph, take the average expression over all cells of the center cell type 
    for the gene. Use that expression as the prediction for the center cell.
    
    Parameters
    ----------
    graph_data : Data
        PyTorch Geometric Data object containing:
        - x: node features (cells x genes expression matrix)
        - center_node: index of the center cell
        - celltypes: cell types for all nodes
        - center_celltype: cell type of the center cell
    gene_idx : int
        Index of the gene to predict
        
    Returns
    -------
    float
        Predicted expression value for the center cell and specified gene
    """
    # Extract expression data and cell types
    expression_matrix = graph_data.x  # Shape: (num_cells, num_genes)
    celltypes = graph_data.celltypes  # Cell types for all nodes
    center_celltype = graph_data.center_celltype[0]  # Center cell type
    center_node = graph_data.center_node
    
    # Find cells of the same type as the center cell, excluding the center cell itself
    same_type_mask = (celltypes == center_celltype) & (torch.arange(len(celltypes)) != center_node)
    
    if not torch.any(same_type_mask):
        # If no other cells of the same type, fall back to k-hop mean
        return khop_mean_baseline(graph_data, gene_idx)
    
    # Get expression values for cells of the same type (excluding center cell)
    same_type_expressions = expression_matrix[same_type_mask, gene_idx]
    
    # Calculate mean expression for cells of the same type
    mean_expression = torch.mean(same_type_expressions)
    
    return mean_expression.item()


def global_mean_baseline(train_dataset: List[Data], gene_idx: int) -> float:
    """
    Baseline 3: Take the average expression over all cells for the gene for all cells 
    in the training dataset (excluding center cells which have zeroed features).
    
    Parameters
    ----------
    train_dataset : List[Data]
        List of PyTorch Geometric Data objects from the training dataset
    gene_idx : int
        Index of the gene to predict
        
    Returns
    -------
    float
        Global mean expression value for the specified gene across all training cells
    """
    all_expressions = []
    
    # Collect expression values from all cells in all training graphs (excluding center cells)
    for graph_data in train_dataset:
        expression_matrix = graph_data.x  # Shape: (num_cells, num_genes)
        center_node = graph_data.center_node
        
        # Exclude the center cell from the average (it has zeroed features)
        all_indices = torch.arange(expression_matrix.shape[0])
        neighbor_indices = all_indices[all_indices != center_node]
        
        gene_expressions = expression_matrix[neighbor_indices, gene_idx]
        all_expressions.append(gene_expressions)
    
    # Concatenate all expression values
    all_expressions = torch.cat(all_expressions, dim=0)
    
    # Calculate global mean
    global_mean = torch.mean(all_expressions)
    
    return global_mean.item()


def evaluate_baselines_on_dataset(
    train_dataset: List[Data], 
    test_dataset: List[Data], 
    gene_idx: int
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all three baselines on a dataset and return performance metrics.
    
    Parameters
    ----------
    train_dataset : List[Data]
        Training dataset
    test_dataset : List[Data]
        Test dataset
    gene_idx : int
        Index of the gene to predict
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary containing MSE and MAE for each baseline method
    """
    # Calculate global mean from training data
    global_mean = global_mean_baseline(train_dataset, gene_idx)
    
    # Initialize results
    results = {
        'khop_mean': {'predictions': [], 'targets': []},
        'center_celltype_mean': {'predictions': [], 'targets': []},
        'global_mean': {'predictions': [], 'targets': []}
    }
    
    # Make predictions on test set
    for graph_data in test_dataset:
        target = graph_data.y[gene_idx].item()
        
        # K-hop mean baseline
        khop_pred = khop_mean_baseline(graph_data, gene_idx)
        results['khop_mean']['predictions'].append(khop_pred)
        results['khop_mean']['targets'].append(target)
        
        # Center cell type mean baseline
        cct_pred = center_celltype_mean_baseline(graph_data, gene_idx)
        results['center_celltype_mean']['predictions'].append(cct_pred)
        results['center_celltype_mean']['targets'].append(target)
        
        # Global mean baseline
        results['global_mean']['predictions'].append(global_mean)
        results['global_mean']['targets'].append(target)
    
    # Calculate MSE and MAE for each method
    final_results = {}
    for method, data in results.items():
        predictions = np.array(data['predictions'])
        targets = np.array(data['targets'])
        
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        
        final_results[method] = {
            'mse': mse,
            'mae': mae,
            'predictions': predictions.tolist(),
            'targets': targets.tolist()
        }
    
    return final_results


def batch_evaluate_baselines(
    train_dataset: List[Data], 
    test_dataset: List[Data], 
    num_genes: int
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all baselines for all genes in the dataset.
    
    Parameters
    ----------
    train_dataset : List[Data]
        Training dataset
    test_dataset : List[Data]
        Test dataset
    num_genes : int
        Number of genes to evaluate
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary containing average MSE and MAE across all genes for each baseline
    """
    all_results = {
        'khop_mean': {'mse': [], 'mae': []},
        'center_celltype_mean': {'mse': [], 'mae': []},
        'global_mean': {'mse': [], 'mae': []}
    }
    
    # Evaluate each gene
    for gene_idx in range(num_genes):
        gene_results = evaluate_baselines_on_dataset(train_dataset, test_dataset, gene_idx)
        
        for method in all_results.keys():
            all_results[method]['mse'].append(gene_results[method]['mse'])
            all_results[method]['mae'].append(gene_results[method]['mae'])
    
    # Calculate average performance across all genes
    final_results = {}
    for method, metrics in all_results.items():
        final_results[method] = {
            'avg_mse': np.mean(metrics['mse']),
            'avg_mae': np.mean(metrics['mae']),
            'std_mse': np.std(metrics['mse']),
            'std_mae': np.std(metrics['mae'])
        }
    
    return final_results
