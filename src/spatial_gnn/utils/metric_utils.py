from scipy.stats import spearmanr
import numpy as np
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
