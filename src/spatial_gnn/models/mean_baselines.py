import torch
import numpy as np
from typing import List

from torch_geometric.data import Data, Batch


def khop_mean_baseline_batch(batch_data: Batch) -> torch.Tensor:
    """
    In k-hop graph, take the average expression over all cells for each gene.
    Use that expression as the prediction for the center cell.
    
    Parameters
    ----------
    batch_data : Batch
        PyTorch Geometric Batch object containing:
        - x: node features (cells x genes expression matrix)
        - center_node: indices of center cells
        - batch: batch assignment for each node
        - y: target expression for center cells (all genes)
        
    Returns
    -------
    torch.Tensor
        Predicted expression values for center cells (batch_size x num_genes)
    """
    x = batch_data.x  # [num_nodes, num_genes]
    center_nodes = batch_data.center_node  # [batch_size,]
    batch_ids = batch_data.batch  # [num_nodes,]
    num_genes = x.shape[1]
    batch_size = len(center_nodes)
    
    predictions = torch.zeros(batch_size, num_genes, device=x.device) # [batch_size, num_genes]
    
    for i in range(batch_size):
        center_node = center_nodes[i]

        # Get nodes belonging to this graph
        graph_mask = batch_ids == i
        exclude_center_node_mask = torch.arange(len(batch_ids), device=x.device) != center_node
        neighbor_mask = graph_mask & exclude_center_node_mask
        neighbor_indices = torch.where(neighbor_mask)[0]

        if len(neighbor_indices) > 0:
            # Get average expression across all neighbor cells for all genes
            predictions[i] = torch.mean(x[neighbor_indices], dim=0)
        else:
            # If no neighbors, use zeros
            predictions[i] = torch.zeros(num_genes, device=x.device)
    
    return predictions


def center_celltype_mean_baseline_batch(batch_data: Batch) -> torch.Tensor:
    """
    In k-hop graph, take the average expression over all cells of the center cell type 
    for each gene. Use that expression as the prediction for the center cell.
    
    Parameters
    ----------
    batch_data : Batch
        PyTorch Geometric Batch object containing:
        - x: node features (cells x genes expression matrix)
        - center_node: indices of center cells
        - batch: batch assignment for each node
        - celltypes: cell types for all nodes
        - center_celltype: cell types of center cells
        - y: target expression for center cells (all genes)
        
    Returns
    -------
    torch.Tensor
        Predicted expression values for center cells (batch_size x num_genes)
    """
    x = batch_data.x  # Shape: (total_nodes, num_genes)
    center_nodes = batch_data.center_node  # Shape: (batch_size,)
    batch_ids = batch_data.batch  # Shape: (total_nodes,)
    celltypes = batch_data.celltypes  # Shape: (total_nodes,)
    center_celltypes = batch_data.center_celltype  # Shape: (batch_size,)
    num_genes = x.shape[1]
    batch_size = len(center_nodes)
    
    predictions = torch.zeros(batch_size, num_genes, device=x.device)
    
    for i in range(batch_size):
        # Get nodes belonging to this graph
        graph_mask = batch_ids == i
        center_node = center_nodes[i]
        center_celltype = center_celltypes[i]
        
        # Find cells of the same type as the center cell, excluding the center cell itself
        same_type_mask = (celltypes == center_celltype) & graph_mask & (torch.arange(len(batch_ids), device=x.device) != center_node)
        same_type_indices = torch.where(same_type_mask)[0]
        
        if len(same_type_indices) > 0:
            # Get average expression for cells of the same type
            predictions[i] = torch.mean(x[same_type_indices], dim=0)
        else:
            # If no other cells of the same type, fall back to k-hop mean
            neighbor_mask = graph_mask & (torch.arange(len(batch_ids), device=x.device) != center_node)
            neighbor_indices = torch.where(neighbor_mask)[0]
            
            if len(neighbor_indices) > 0:
                predictions[i] = torch.mean(x[neighbor_indices], dim=0)
            else:
                predictions[i] = torch.zeros(num_genes, device=x.device)
    
    return predictions


def global_mean_baseline_batch(train_dataset: List[Data]) -> torch.Tensor:
    """
    Compute global mean expression values from training dataset.
    
    Parameters
    ----------
    train_dataset : List[Data]
        List of PyTorch Geometric Data objects from the training dataset
        
    Returns
    -------
    torch.Tensor
        Global mean expression values for each gene (num_genes,)
    """
    all_expressions = []
    
    # Collect expression values from all cells in all training graphs (excluding center cells)
    for graph_data in train_dataset:
        expression_matrix = graph_data.x  # Shape: (num_cells, num_genes)
        center_node = graph_data.center_node
        
        # Exclude the center cell from the average (it has zeroed features)
        all_indices = torch.arange(expression_matrix.shape[0])
        neighbor_indices = all_indices[all_indices != center_node]
        
        if len(neighbor_indices) > 0:
            neighbor_expressions = expression_matrix[neighbor_indices]
            all_expressions.append(neighbor_expressions)
    
    if len(all_expressions) > 0:
        # Concatenate all expression values
        all_expressions = torch.cat(all_expressions, dim=0)
        # Calculate global mean for each gene
        global_means = torch.mean(all_expressions, dim=0)
    else:
        # Fallback if no data
        num_genes = train_dataset[0].x.shape[1] if len(train_dataset) > 0 else 0
        global_means = torch.zeros(num_genes)
    
    return global_means
