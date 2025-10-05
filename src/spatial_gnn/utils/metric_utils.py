from scipy.stats import spearmanr
import numpy as np


def compute_spearman(preds, targets):
    """Compute Spearman correlation between two numpy arrays."""
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    if preds_flat.shape[0] == 0:
        return 0.0
    corr, _ = spearmanr(preds_flat, targets_flat)
    return corr if not np.isnan(corr) else 0.0
