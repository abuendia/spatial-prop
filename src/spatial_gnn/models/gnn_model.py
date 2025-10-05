import numpy as np
from scipy.sparse import issparse
import time
from tqdm import tqdm

import torch
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool, global_add_pool, global_max_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, ModuleList, LayerNorm, Dropout
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.distributions.multivariate_normal import MultivariateNormal as MVN

from spatial_gnn.utils.metric_utils import compute_spearman


class GNN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        input_dim,
        output_dim=1,
        inject_dim=0,
        num_layers=3,
        method="GCN",
        pool="add",
        genept_embeddings=None,
    ):
        super(GNN, self).__init__()
        torch.manual_seed(444)
        
        self.method = method
        self.pool = pool
        self.inject_dim = inject_dim

        self.genept_embeddings = genept_embeddings
        self.genept_embedding_dim = 0
        self._genept_cache_ready = False
        self._cached_gene_count = None
        self._has_genept = genept_embeddings is not None
        readout_dim = 128

        # Set embedding dimension if GenePT embeddings are provided
        if self.genept_embeddings is not None:
            self.genept_embedding_dim = len(next(iter(self.genept_embeddings.values())))
            print(f"Using {len(self.genept_embeddings)} GenePT embeddings with dimension {self.genept_embedding_dim}")
        
        if self.method == "GCN":
            self.conv1 = GCNConv(input_dim, hidden_channels)
            self.convs = ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)])
        
        elif self.method == "GIN":
            self.conv1 = GINConv(
                            Sequential(
                                Linear(input_dim, hidden_channels),
                                BatchNorm1d(hidden_channels),
                                ReLU(),
                                Linear(hidden_channels, hidden_channels)
                            )
                          )
            self.convs = ModuleList([GINConv(
                            Sequential(
                                Linear(hidden_channels, hidden_channels),
                                BatchNorm1d(hidden_channels),
                                ReLU(),
                                Linear(hidden_channels, hidden_channels)
                            )
                          ) for _ in range(num_layers - 1)])
        
        elif self.method == "SAGE":
            self.conv1 = SAGEConv(input_dim, hidden_channels)
            self.convs = ModuleList([SAGEConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)])
            
        else:
            raise Exception("'method' not recognized.")
        
        self.lin_base = Linear(hidden_channels + inject_dim, output_dim)

        if self._has_genept:
            # cross-attention
            self.q_proj = Linear(hidden_channels, hidden_channels)
            self.k_proj = Linear(hidden_channels, hidden_channels)
            self.genept_to_hidden = Linear(self.genept_embedding_dim, hidden_channels)
            self.xattn = torch.nn.MultiheadAttention(hidden_channels, num_heads=8, batch_first=True)

            self.pre_attn_ln = LayerNorm(hidden_channels)
            self.post_attn_ln = LayerNorm(hidden_channels)

            self.genept_dropout = Dropout(0.15)
            self.fuse_gate = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, 1)
            )

    def _prepare_genept_lookup(self, gene_names, device="cuda"):
        # Map genes -> row in embedding matrix
        valid_rows, valid_idx = [], []
        for j, gene_name in enumerate(gene_names):
            emb = self.genept_embeddings.get(gene_name)
            if emb is not None:
                valid_rows.append(torch.as_tensor(emb, dtype=torch.float32))
                valid_idx.append(j)

        if len(valid_rows) == 0:
            # No overlaps; store empty tensors
            self.register_buffer("_E_valid", torch.zeros(0, self.genept_embedding_dim))
            self.register_buffer("_valid_idx", torch.zeros(0, dtype=torch.long))
        else:
            E_valid = torch.stack(valid_rows, dim=0)             # [G_valid, D]
            valid_idx = torch.tensor(valid_idx, dtype=torch.long) # [G_valid]
            self.register_buffer("_E_valid", E_valid.to(device))
            self.register_buffer("_valid_idx", valid_idx.to(device))

        print("GenePT overlap: ", len(valid_rows), "/", len(gene_names))
        self._genept_cache_ready = True

    def _compute_genept_weighted_embedding(self, x, batch, gene_names, device=None):
        if (not self._genept_cache_ready) or (getattr(self, "_cached_gene_count", None) != len(gene_names)):
            self._prepare_genept_lookup(gene_names, device)
            self._cached_gene_count = len(gene_names)
        
        E_valid = self._E_valid  # [valid_genes, D]
        valid_idx = self._valid_idx  # [valid_genes]
        B = int(batch.max().item()) + 1

        if E_valid.numel() == 0:
            return x.new_zeros((B, 1, self.genept_embedding_dim))
        
        mean_expr = global_mean_pool(x, batch)
        expr_valid = mean_expr.index_select(dim=1, index=valid_idx)
        
        weights = expr_valid / (expr_valid.sum(dim=1, keepdim=True) + 1e-12)
        weighted = weights.unsqueeze(-1) * E_valid.unsqueeze(0)
        cell_genept_embed = weighted.sum(dim=1)

        return cell_genept_embed


    def _genept_topk_tokens(self, x, batch, gene_names, device="cuda"):
        if (not getattr(self, "_genept_cache_ready", False)) or \
        (getattr(self, "_cached_gene_count", None) != len(gene_names)):
            self._prepare_genept_lookup(gene_names, device)
            self._cached_gene_count = len(gene_names)
        
        E_valid = self._E_valid  # [valid_genes, D]
        valid_idx = self._valid_idx  # [valid_genes]
        num_graphs = int(batch.max().item()) + 1

        if E_valid.numel() == 0:
            # No overlap between gene_names and genept embeddings
            return x.new_zeros((num_graphs, 1, self.genept_embedding_dim))
        
        mean_expr = global_mean_pool(x, batch) 
        mean_expr_valid = mean_expr.index_select(dim=1, index=valid_idx)

        vals, idx_in_valid = torch.topk(mean_expr_valid, k=self.genept_embedding_dim, dim=1)  # both [B, k_eff]
        del vals

        flattened_idx = idx_in_valid.reshape(-1)
        tokens = E_valid.index_select(0, flattened_idx).reshape(num_graphs, self.genept_embedding_dim, -1)
        return tokens

    def _pool(self, x, batch):
        if self.pool == "mean":
            return global_mean_pool(x, batch)
        elif self.pool == "add":
            return global_add_pool(x, batch)
        elif self.pool == "max":
            return global_max_pool(x, batch)
        else:
            raise ValueError(f"'pool' not recognized: {self.pool}")

    def forward(self, x, edge_index, batch, inject=None, gene_names=None):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index) if i == len(self.convs)-1 else F.relu(conv(h, edge_index))
        pooled = self._pool(h, batch)   # [B, H]
        B = int(batch.max().item()) + 1

        genept_tokens = None
        if self._has_genept and (gene_names is not None):
            if (not getattr(self, "_genept_cache_ready", False)) or \
            (getattr(self, "_cached_gene_count", None) != len(gene_names)):
                self._prepare_genept_lookup(gene_names, device=x.device)
                self._cached_gene_count = len(gene_names)

            E_valid = self._E_valid 
            if E_valid.numel() == 0:
                genept_tokens = x.new_zeros((B, 1, self.genept_embedding_dim))  # no overlap => dummy token
            else:
                genept_tokens = E_valid.unsqueeze(0).expand(B, -1, -1).to(x.device)  # [B, G_valid, D]

        if (genept_tokens is not None) and (self.xattn is not None):
            tokens_h = self.genept_dropout(self.pre_attn_ln(self.genept_to_hidden(genept_tokens)))  # [B,G,H]
            query    = self.q_proj(pooled).unsqueeze(1)   # [B,1,H]
            keys     = self.k_proj(tokens_h)              # [B,G,H]
            fused, _ = self.xattn(query, keys, tokens_h)  # [B,1,H]
            fused_pooled = self.post_attn_ln(fused.squeeze(1))  # [B,H]

            gate = torch.sigmoid(self.fuse_gate(pooled))        # [B,1]
            fused_pooled = pooled + gate * (fused_pooled - pooled)  # gated residual
        else:
            fused_pooled = pooled

        fused_pooled = F.dropout(fused_pooled, p=0.1, training=self.training)
        base_in = fused_pooled if inject is None else torch.cat([fused_pooled, inject], dim=1)
        out = self.lin_base(base_in)
        return out


def bmc_loss(pred, target, noise_var):
    """Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, d].
      target: A float tensor of size [batch, d].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    # reshape target to pred shape -- added 6/25/2024
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



    
# negative pearson correlation loss
def npcc_loss(pred, target):
    """
    Negative pearson correlation as loss
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




def weighted_l1_loss(pred, target, zero_weight, nonzero_weight):

    if target.shape != pred.shape:
        target = torch.reshape(target, pred.shape)
    
    abs_diff = torch.abs(pred - target)
    zero_mask = (target == 0).float()
    nonzero_mask = (target != 0).float()
    
    loss = (zero_weight * zero_mask * abs_diff +
            nonzero_weight * nonzero_mask * abs_diff)
    
    return loss.mean()

class WeightedL1Loss(_Loss):
    def __init__(self, zero_weight=1.0, nonzero_weight=1.0):
        super(WeightedL1Loss, self).__init__()
        self.zero_weight = zero_weight
        self.nonzero_weight = nonzero_weight

    def forward(self, predictions, targets):
        
        loss = weighted_l1_loss(predictions, targets, self.zero_weight, self.nonzero_weight)
        
        return loss


def train(model, loader, criterion, optimizer, gene_names=None, inject=False, device="cuda"):
    model.train()

    for batch in loader:  
        batch.to(device)
        
        # Prepare arguments for forward pass
        forward_args = {
            'x': batch.x,
            'edge_index': batch.edge_index, 
            'batch': batch.batch,
            'inject': batch.inject if inject else None,
            'gene_names': gene_names,
        }
        
        out = model(**forward_args)  # Perform a single forward pass.
        
        loss = criterion(out, batch.y)  # Compute the loss.
                
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

    

def test(model, loader, loss, criterion, gene_names=None, inject=False, device="cuda"):
    model.eval()
    errors = []
    all_preds = []
    all_targets = []

    for batch in loader: 
        batch.to(device)
        
        # Prepare arguments for forward pass
        forward_args = {
            'x': batch.x,
            'edge_index': batch.edge_index, 
            'batch': batch.batch,
            'inject': batch.inject if inject else None,
            'gene_names': gene_names,
        }
        
        out = model(**forward_args)
        target = batch.y.unsqueeze(1)

        all_preds.append(out.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())
        
        if loss == "mse":
            errors.append(F.mse_loss(out, target).sqrt().item())
        elif loss == "l1":
            errors.append(F.l1_loss(out, target).item())
        elif loss == "weightedl1":
            errors.append(weighted_l1_loss(out, target, criterion.zero_weight, criterion.nonzero_weight).item())
        elif loss == "balanced_mse":
            errors.append(bmc_loss(out, target, criterion.noise_sigma**2).item())
        elif loss == "npcc":
            errors.append(npcc_loss(out, target).item())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    spearman = compute_spearman(all_preds, all_targets)

    return np.mean(errors), spearman

def predict(model, dataloader, adata, gene_names=None, device="cuda"):
    """
    Run GNN model predictions and convert back to AnnData format using the stored mapping info.
    
    Parameters:
    -----------
    model : GNN
        Trained GNN model
    dataloader : DataLoader
        PyTorch Geometric dataloader
    original_adata : AnnData
        Original AnnData object used to create the dataset
    device : str
        Device to run inference on
        
    Returns:
    --------
    AnnData
        Updated AnnData with predictions in .X
    """    
    model.eval()
    
    # Create prediction matrix with same shape as original
    n_cells, n_genes = adata.shape
    prediction_matrix = np.zeros((n_cells, n_genes), dtype=np.float32)
    
    # Track which cells have been predicted
    predicted_cells = set()
    
    batch_count = 0    
    print(f"Starting prediction for {n_cells} cells, {n_genes} genes")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing data groups"):
            batch_count += 1
            
            batch.to(device)
            
            # Prepare arguments for forward pass
            forward_args = {
                'x': batch.x,
                'edge_index': batch.edge_index, 
                'batch': batch.batch,
                'inject': batch.inject if hasattr(batch, 'inject') and batch.inject is not None else None,
                'gene_names': gene_names,
            }
            
            out = model(**forward_args)
            
            batch_predictions = out.detach().cpu().numpy()
            
            for i, pred in enumerate(batch_predictions):
                original_cell_idx = batch.original_cell_idx[i].item()
                
                if len(pred) == n_genes:
                    prediction_matrix[original_cell_idx, :] = pred
                else:
                    raise ValueError(f"Prediction dimension {len(pred)} doesn't match genes {n_genes}")
                predicted_cells.add(original_cell_idx)
        
    # Final summary
    print(f"Processed {batch_count} batches")

    original_expression = adata.X.toarray() if issparse(adata.X) else adata.X
    perturbation_effects = prediction_matrix - original_expression
    
    adata.layers['predicted_perturbed'] = prediction_matrix
    adata.layers['perturbation_effects'] = perturbation_effects
    print(f"Predicted {len(predicted_cells)} cells out of {n_cells} total cells")
    print(f"Perturbation effects calculated as: predicted - original")
    
    return adata
