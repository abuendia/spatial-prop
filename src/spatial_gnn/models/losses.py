import torch
from torch.nn.modules.loss import _Loss
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def bmc_loss(pred, target, noise_var):
    """
    Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, d].
      target: A float tensor of size [batch, d].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    if target.shape != pred.shape:
        target = torch.reshape(target, pred.shape)
    
    I = torch.eye(pred.shape[-1], device=pred.device, dtype=pred.dtype)
    targets = torch.arange(pred.shape[0], device=pred.device, dtype=torch.long)
    logits = MVN(pred.unsqueeze(1), noise_var * I).log_prob(target.unsqueeze(0))
    loss = F.cross_entropy(logits, targets)
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 
    return loss


class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)


def npcc_loss(pred, target):
    """
    Negative Pearson correlation as loss
    """
    # Alternative formulation
    x = torch.flatten(pred)
    y = torch.flatten(target)
    
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    loss = 1-r_val
    return loss
    

class Neg_Pearson_Loss(_Loss):
    def __init__(self):
        super(Neg_Pearson_Loss, self).__init__()
        return

    def forward(self, pred, target):
        return npcc_loss(pred, target)


def weighted_l1_loss(pred, target, zero_weight, nonzero_weight, residual_penalty=None):
    """
    Weighted L1 loss
    """
    if target.shape != pred.shape:
        target = torch.reshape(target, pred.shape)
    
    abs_diff = torch.abs(pred - target)
    zero_mask = (target == 0).float()
    nonzero_mask = (target != 0).float()
    
    loss = (zero_weight * zero_mask * abs_diff +
            nonzero_weight * nonzero_mask * abs_diff)
    if residual_penalty is not None:
        loss = loss + residual_penalty * torch.mean(pred ** 2)
    
    return loss.mean()


class WeightedL1Loss(_Loss):
    def __init__(self, zero_weight=1.0, nonzero_weight=1.0, residual_penalty=None):
        super(WeightedL1Loss, self).__init__()
        self.zero_weight = zero_weight
        self.nonzero_weight = nonzero_weight
        self.residual_penalty = residual_penalty

    def forward(self, predictions, targets):
        
        loss = weighted_l1_loss(predictions, targets, self.zero_weight, self.nonzero_weight, self.residual_penalty)
        
        return loss
