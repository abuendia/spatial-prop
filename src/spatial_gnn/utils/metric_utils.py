from scipy.stats import spearmanr
import numpy as np
import scipy.sparse as sp

import torch.nn.functional as F
import torch


def compute_spearman(preds, targets):
    """Compute Spearman correlation between two numpy arrays."""
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    if preds_flat.shape[0] == 0:
        return 0.0
    corr, _ = spearmanr(preds_flat, targets_flat)
    return corr if not np.isnan(corr) else 0.0


def compute_celltype_accuracy(celltype_logits, targets):
    """Compute celltype accuracy between two numpy arrays."""
    celltype_preds = torch.argmax(F.softmax(celltype_logits, dim=1), dim=1).detach().cpu().numpy()
    celltype_preds_flat = celltype_preds.flatten()
    targets_flat = targets.flatten()
    if celltype_preds_flat.shape[0] == 0:
        return 0.0
    accuracy = np.sum(celltype_preds_flat == targets_flat) / celltype_preds_flat.shape[0]
    return accuracy if not np.isnan(accuracy) else 0.0


def robust_nanmean(x):
    nmx = np.nanmean(x) if np.count_nonzero(~np.isnan(x))>1 else np.mean(x)
    return (nmx)


def robust_nanmedian(x):
    nmx = np.nanmedian(x) if np.count_nonzero(~np.isnan(x))>1 else np.median(x)
    return (nmx)


def get_gene_set_sum_of_log1p(layer, adata, gene_list):
    """
    For each cell, compute sum_j log1p(expr[cell, gene_j]).
    """
    gene_idx = [adata.var_names.get_loc(g) for g in gene_list]

    if sp.issparse(layer):
        sub = layer[:, gene_idx].toarray()
    else:
        sub = layer[:, gene_idx]

    # sum of log1p over genes
    return np.log1p(sub).sum(axis=1).ravel()
