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
    x = batch_data.x  # (num_nodes, num_genes)
    center_nodes = batch_data.center_node  # (batch_size,)
    batch_ids = batch_data.batch  # (num_nodes,)
    num_genes = x.shape[1]
    batch_size = len(center_nodes)
    
    predictions = torch.zeros(batch_size, num_genes, device=x.device) # (batch_size, num_genes)
    
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
    x = batch_data.x  # (total_nodes, num_genes)
    center_nodes = batch_data.center_node  # (batch_size,)
    batch_ids = batch_data.batch  # (total_nodes,)
    celltypes = np.array([celltype for subgraph in batch_data.celltypes for celltype in subgraph]) # (total_nodes,)
    center_celltypes = batch_data.center_celltype  # (batch_size,)
    num_genes = x.shape[1]
    batch_size = len(center_nodes)
    
    predictions = torch.zeros(batch_size, num_genes, device=x.device)
    
    for i in range(batch_size):
        # Get nodes belonging to this graph
        graph_mask = batch_ids == i
        center_node = center_nodes[i]
        center_celltype = center_celltypes[i]
        
        # Find cells of the same type as the center cell, excluding the center cell itself
        exclude_center_node_mask = torch.arange(len(batch_ids), device=x.device) != center_node
        same_celltype_type_mask = torch.tensor((celltypes == center_celltype), device=x.device) & graph_mask & exclude_center_node_mask
        same_type_indices = torch.where(same_celltype_type_mask)[0]
        
        if len(same_type_indices) > 0:
            # Get average expression for cells of the same type
            predictions[i] = torch.mean(x[same_type_indices], dim=0)
        else:
            # If no other cells of the same type, fall back to k-hop mean
            neighbor_mask = graph_mask & exclude_center_node_mask
            neighbor_indices = torch.where(neighbor_mask)[0]
            
            if len(neighbor_indices) > 0:
                predictions[i] = torch.mean(x[neighbor_indices], dim=0)
            else:
                predictions[i] = torch.zeros(num_genes, device=x.device)
        
    return predictions


def global_mean_baseline_batch(train_dataset, device="cuda") -> torch.Tensor:
    """
    Global mean expression over ALL non-center cells across all batches/graphs.
    Works whether each item is a single-graph Data or a multi-graph Batch.
    """
    sum_x = None
    total = 0

    for batch in train_dataset:
        batch = batch.to(device)

        x = batch.x # (num_nodes, num_genes)
        centers = batch.center_node # (batch_size,)

        # lazily init accumulator on the right device/dtype
        if sum_x is None:
            sum_x = torch.zeros(x.size(1), dtype=x.dtype, device=x.device) # (num_genes,)

        sum_x += x.sum(dim=0)                    # add all cells
        if centers is not None:
            centers = centers.to(x.device).long()
            # subtract centers' contribution
            sum_x -= x[centers].sum(dim=0)
            total += x.size(0) - int(centers.numel())
        else:
            total += x.size(0)

    if sum_x is None or total == 0:
        # fallback
        num_genes = train_dataset[0].x.size(1) if len(train_dataset) > 0 else 0
        return torch.zeros(num_genes)

    return sum_x / float(total)
