import numpy as np
from scipy.sparse import issparse
import time
from tqdm import tqdm

import torch
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool, global_add_pool, global_max_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, ModuleList
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.distributions.multivariate_normal import MultivariateNormal as MVN


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, input_dim, output_dim=1, inject_dim=0,
                 num_layers=3, method="GCN", pool="add", genept_embeddings=None):
        super(GNN, self).__init__()
        torch.manual_seed(444)
        
        self.method = method
        self.pool = pool
        self.genept_embeddings = genept_embeddings
        self.genept_embedding_dim = 0
        
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
        
        # Adjust inject_dim if GenePT embeddings are used
        if self.genept_embeddings is not None:
            # Single weighted embedding vector from neighbor context
            inject_dim += self.genept_embedding_dim
            
        self.lin = Linear(hidden_channels+inject_dim, output_dim)

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

        self._genept_cache_ready = True

    def _create_genept_injection_features(self, x, batch, gene_names, device="cuda", k: int = 10):
        # Lazy-prepare cache if needed or if gene_names changed in size
        if not getattr(self, "_genept_cache_ready", False) or \
        getattr(self, "_cached_gene_count", None) != len(gene_names):
            self._prepare_genept_lookup(gene_names, device)
            self._cached_gene_count = len(gene_names)

        genept_embeddings = self._E_valid  # [G_valid, D]
        valid_idx = self._valid_idx  # [G_valid]
        num_graphs = int(batch.max().item()) + 1

        if genept_embeddings.numel() == 0:
            # No overlap between gene_names and genept embeddings
            return x.new_zeros((num_graphs, self.genept_embedding_dim))

        mean_expr = global_mean_pool(x, batch)  # uses all nodes per graph
        mean_expr_valid = mean_expr.index_select(dim=1, index=valid_idx)
        k_eff = min(k, mean_expr_valid.size(1))
        vals, idx_in_valid = torch.topk(mean_expr_valid, k=k_eff, dim=1)  # both [B, k_eff]
        topk_E = genept_embeddings.index_select(0, idx_in_valid.reshape(-1)).reshape(num_graphs, k_eff, -1)

        weights = vals.unsqueeze(-1)                  # [B, k_eff, 1]
        num = (weights * topk_E).sum(dim=1)           # [B, D]
        denom = weights.abs().sum(dim=1).clamp_min(1e-12)  # [B, 1]
        out = num / denom

        return out

    def forward(self, x, edge_index, batch, inject=None, gene_names=None):    
        # Store original x for GenePT feature extraction
        original_x = x
        
        # node embeddings 
        x = F.relu(self.conv1(x, edge_index))
        for layer_idx, conv in enumerate(self.convs):
            if layer_idx < len(self.convs) - 1:
                x = F.relu(conv(x, edge_index))
            else:
                x = conv(x, edge_index)

        # pooling and readout
        if self.pool == "mean":
            x = global_mean_pool(x, batch)
        elif self.pool == "add":
            x = global_add_pool(x, batch)
        elif self.pool == "max":
            x = global_max_pool(x, batch)
        else:
            raise Exception ("'pool' not recognized")

        # final prediction
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Prepare injection features
        injection_features = []
        
        # Add existing inject features if provided
        if inject is not None:
            injection_features.append(inject)
        
        # Add GenePT embedding features if available
        if (self.genept_embeddings is not None and gene_names is not None):
            genept_features = self._create_genept_injection_features(
                original_x, batch, gene_names, x.device, k=10
            )
            injection_features.append(genept_features)
        
        # Concatenate all injection features
        if injection_features:
            all_inject_features = torch.cat(injection_features, dim=1)
            x = self.lin(torch.cat((x, all_inject_features), dim=1))
        else:
            x = self.lin(x)
        
        return x
    

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
        
        if loss == "mse":
            errors.append(F.mse_loss(out, batch.y.unsqueeze(1)).sqrt().item())
        elif loss == "l1":
            errors.append(F.l1_loss(out, batch.y.unsqueeze(1)).item())
        elif loss == "weightedl1":
            errors.append(weighted_l1_loss(out, batch.y.unsqueeze(1), criterion.zero_weight, criterion.nonzero_weight).item())
        elif loss == "balanced_mse":
            errors.append(bmc_loss(out, batch.y.unsqueeze(1), criterion.noise_sigma**2).item())
        elif loss == "npcc":
            errors.append(npcc_loss(out, batch.y.unsqueeze(1)).item())

    return np.mean(errors)  # Derive ratio of correct predictions.

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
            
            batch_predictions = out.cpu().numpy()
            
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
