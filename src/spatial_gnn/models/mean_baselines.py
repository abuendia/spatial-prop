import torch
import numpy as np
from typing import List
from collections import defaultdict

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


def global_mean_baseline_batch(train_loader, device="cuda") -> torch.Tensor:
    """
    Global mean expression over all cells across all batches/graphs.
    """
    sum_x = None
    total = 0

    for batch in train_loader:
        x = batch.x.to(device)  # (num_nodes, num_genes)

        if sum_x is None:
            sum_x = torch.zeros(x.size(1), dtype=x.dtype, device=device)

        sum_x += x.sum(dim=0)
        total += x.size(0)

    if sum_x is None or total == 0:
        raise RuntimeError("global_mean_baseline_batch: no cells found in train_loader.")

    return sum_x / float(total)


def center_celltype_global_mean_baseline_batch(train_loader, device="cuda") -> dict[int, torch.Tensor]:
    """
    Per-celltype global mean expression over all cells of that type across all batches/graphs.
    """
    sum_by_type: dict[int, torch.Tensor] = {}
    count_by_type: dict[int, int] = defaultdict(int)

    for batch in train_loader:
        x = batch.x.to(device)  # (num_nodes, num_genes)
        ct = batch.celltypes
        celltypes = np.array([item for sublist in ct for item in sublist]) 

        for ct_id in np.unique(celltypes):
            mask = (celltypes == ct_id)
            if mask.any():
                ct_sum = x[mask].sum(dim=0)
                if ct_id not in sum_by_type:
                    sum_by_type[ct_id] = ct_sum
                else:
                    sum_by_type[ct_id] += ct_sum
                count_by_type[ct_id] += int(mask.sum().item())

    if not sum_by_type:
        raise RuntimeError("center_celltype_global_mean_baseline_batch: no cells / celltypes found.")

    means_by_type = {
        ct_id_str: sum_vec / float(count_by_type[ct_id_str])
        for ct_id_str, sum_vec in sum_by_type.items()
    }
    return means_by_type
