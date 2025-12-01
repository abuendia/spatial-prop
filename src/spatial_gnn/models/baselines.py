import numpy as np
from typing import Optional
from collections import defaultdict

import torch
from torch_geometric.data import Batch


def khop_mean_baseline_batch(
    batch_data: Optional[Batch] = None,
    x: Optional[torch.Tensor] = None,
    batch: Optional[torch.Tensor] = None,
    center_nodes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute k-hop mean baseline for test-time subgraphs.
    """
    if batch_data is not None:
        x = batch_data.x                     
        batch_ids = batch_data.batch      
        center_nodes_local = batch_data.center_node  
    else:
        if x is None or batch is None or center_nodes is None:
            raise ValueError("Must provide either batch_data or (x, batch, center_nodes).")
        batch_ids = batch
        center_nodes_local = center_nodes

    num_genes = x.shape[1]
    num_graphs = center_nodes_local.shape[0]
    preds = torch.zeros(num_graphs, num_genes, device=x.device)

    for g in range(num_graphs):
        graph_nodes = (batch_ids == g).nonzero(as_tuple=False).view(-1)
        if graph_nodes.numel() == 0:
            continue  

        local_center_idx = int(center_nodes_local[g])
        global_center_idx = int(graph_nodes[local_center_idx])
        neighbor_indices = graph_nodes[graph_nodes != global_center_idx]

        if neighbor_indices.numel() > 0:
            preds[g] = x[neighbor_indices].mean(dim=0)
        else:
            preds[g] = 0.0  

    return preds


def global_mean_baseline_batch(train_loader, device="cuda") -> torch.Tensor:
    """
    Global mean expression over all cells across all batches/graphs in training set.
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
    Per-celltype global mean expression over all cells of that type across all batches/graphs in training set.
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
