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

from spatial_gnn.utils.metric_utils import compute_spearman, compute_celltype_accuracy


class GNN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        input_dim,
        output_dim=1,
        inject_dim=0,
        num_layers=4,
        method="GCN",
        pool="add",
        genept_embeddings=None,
        genept_strategy=None,
        celltypes_to_index=None,
        predict_celltype=False,
        train_multitask=False,
    ):
        super(GNN, self).__init__()
        torch.manual_seed(444)
        
        self.method = method
        self.pool = pool
        self.inject_dim = inject_dim
        self.predict_celltype = predict_celltype
        self.train_multitask = train_multitask
        self.celltypes_to_index = celltypes_to_index
        self.num_celltypes = len(celltypes_to_index)

        self.has_genept = genept_embeddings is not None
        self.genept_embeddings = genept_embeddings
        self.genept_embedding_dim = 0
        self._genept_cache_ready = False
        self.genept_strategy = genept_strategy

        # Set embedding dimension if GenePT embeddings are provided
        if self.genept_embeddings is not None:
            self.genept_embedding_dim = len(next(iter(self.genept_embeddings.values())))
            print(f"Using {len(self.genept_embeddings)} GenePT embeddings with dimension {self.genept_embedding_dim}")

        if self.genept_strategy == "early_fusion":
            input_dim = input_dim + self.genept_embedding_dim
            self.lin_base = Linear(hidden_channels + inject_dim, output_dim)
        elif self.genept_strategy == "late_fusion":
            self.lin_base = Linear(hidden_channels + inject_dim + self.genept_embedding_dim, output_dim)
        elif self.genept_strategy == "xattn":
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
            self.lin_base = Linear(hidden_channels + inject_dim, output_dim)
        else:
            self.lin_base = Linear(hidden_channels + inject_dim, output_dim)

        # Expression branch GNN
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

        # Cell-type branch GNN (only if predicting cell type)
        if self.predict_celltype:
            if self.method == "GCN":
                self.ct_conv1 = GCNConv(input_dim, hidden_channels)
                self.ct_convs = ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)])
            elif self.method == "GIN":
                self.ct_conv1 = GINConv(
                                Sequential(
                                    Linear(input_dim, hidden_channels),
                                    BatchNorm1d(hidden_channels),
                                    ReLU(),
                                    Linear(hidden_channels, hidden_channels)
                                )
                              )
                self.ct_convs = ModuleList([GINConv(
                                Sequential(
                                    Linear(hidden_channels, hidden_channels),
                                    BatchNorm1d(hidden_channels),
                                    ReLU(),
                                    Linear(hidden_channels, hidden_channels)
                                )
                              ) for _ in range(num_layers - 1)])
            elif self.method == "SAGE":
                self.ct_conv1 = SAGEConv(input_dim, hidden_channels)
                self.ct_convs = ModuleList([SAGEConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)])

            # Cell-type classifier head
            self.celltype_head = Linear(hidden_channels, self.num_celltypes)

            # Expression MLP head that consumes pooled expr features (+ optional inject) + one-hot cell type
            expr_head_in_dim = hidden_channels + inject_dim + self.num_celltypes
            self.expr_mlp = Sequential(
                Linear(expr_head_in_dim, hidden_channels),
                ReLU(),
                Linear(hidden_channels, output_dim)
            )


    def _prepare_genept_lookup(self, gene_names, device="cuda"):
        # Map genes -> row in embedding matrix
        emb_list = []
        valid_idx_set = set()

        for i, gene_name in enumerate(gene_names):
            if gene_name in self.genept_embeddings:
                emb_list.append(self.genept_embeddings[gene_name])
                valid_idx_set.add(i)
            else:
                emb_list.append(torch.zeros(self.genept_embedding_dim))

        emb_matrix = torch.tensor(emb_list)
        mean_emb = emb_matrix[list(valid_idx_set)].mean(dim=0)  # only compute mean on real genes
        for i in range(len(emb_matrix)):
            if i not in valid_idx_set:
                emb_matrix[i] = mean_emb
   
        emb_matrix = emb_matrix.to(device)
        print(f"GenePT overlap: {len(valid_idx_set)}/{len(gene_names)}")
        self._E_valid = emb_matrix
        self._valid_idx_set = valid_idx_set
        self._genept_cache_ready = True

        return emb_matrix


    def _compute_genept_weighted_embedding(self, x, gene_names, device=None):
        if not self._genept_cache_ready:
            self._prepare_genept_lookup(gene_names, device)
         
        cell_genept_embed = x @ self._E_valid
        cell_genept_embed = cell_genept_embed / (torch.linalg.norm(cell_genept_embed, axis=1, keepdim=True) + 1e-12)
        return cell_genept_embed

    def _pool(self, x, batch):
        if self.pool == "mean":
            return global_mean_pool(x, batch)
        elif self.pool == "add":
            return global_add_pool(x, batch)
        elif self.pool == "max":
            return global_max_pool(x, batch)
        else:
            raise ValueError(f"'pool' not recognized: {self.pool}")

    def _forward_expr_backbone(self, x, edge_index, batch, gene_names=None):
        # Optionally augment node features with GenePT (early fusion)
        if self.genept_strategy == "early_fusion":
            weighted_genept_embeddings = self._compute_genept_weighted_embedding(x, gene_names, device=x.device)
            x = torch.cat([x, weighted_genept_embeddings], dim=1)

        h = self.conv1(x, edge_index)
        h = F.relu(h)
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index) if i == len(self.convs)-1 else F.relu(conv(h, edge_index))

        if self.genept_strategy == "late_fusion":
            weighted_genept_embeddings = self._compute_genept_weighted_embedding(x, gene_names, device=x.device)
            h = torch.cat([h, weighted_genept_embeddings], dim=1)

        expr_pooled = self._pool(h, batch)
        expr_pooled = F.dropout(expr_pooled, p=0.1, training=self.training)
        return expr_pooled

    def forward_celltype(self, x, edge_index, batch):
        """Return only cell-type logits (no expression)."""
        assert self.predict_celltype, "Celltype branch not enabled"

        ct_h = self.ct_conv1(x, edge_index)
        ct_h = F.relu(ct_h)
        for i, conv in enumerate(self.ct_convs):
            ct_h = conv(ct_h, edge_index) if i == len(self.ct_convs)-1 else F.relu(conv(ct_h, edge_index))

        ct_pooled = self._pool(ct_h, batch)
        ct_pooled = F.dropout(ct_pooled, p=0.1, training=self.training)
        celltype_logits = self.celltype_head(ct_pooled)
        return celltype_logits

    def forward_expression(
        self,
        x,
        edge_index,
        batch,
        inject=None,
        gene_names=None,
        center_celltype=None,
        ct_logits=None,
        use_oracle_ct=False,
        allow_grad_through_ct=False,
    ):
        """Expression prediction, optionally conditioned on celltype logits."""
        expr_pooled = self._forward_expr_backbone(x, edge_index, batch, gene_names=gene_names)

        if not self.predict_celltype:
            base_in = expr_pooled if inject is None else torch.cat([expr_pooled, inject], dim=1)
            return self.lin_base(base_in)

        # With celltype prediction 
        if use_oracle_ct:
            # oracle one-hot
            ct_idx = torch.tensor(center_celltype, device=expr_pooled.device, dtype=torch.long)
            celltype_features = F.one_hot(ct_idx, num_classes=self.num_celltypes).float()
        else:
            assert ct_logits is not None, "ct_logits must be provided if not using oracle CT"
            if allow_grad_through_ct:
                # multitask: soft distribution, gradients flow back into ct branch
                celltype_features = F.softmax(ct_logits, dim=1)
            else:
                # decoupled: hard argmax, detached one-hot
                hard_ct_pred = torch.argmax(ct_logits, dim=1)
                celltype_features = F.one_hot(hard_ct_pred, num_classes=self.num_celltypes).float().detach()

        fused = torch.cat([expr_pooled, celltype_features], dim=1)
        if inject is not None:
            fused = torch.cat([fused, inject], dim=1)

        expr_out = self.expr_mlp(fused)
        return expr_out

    def forward(
        self,
        x,
        edge_index,
        batch,
        inject=None,
        gene_names=None,
        use_oracle_ct=False,
        center_celltype=None,
        allow_grad_through_ct=False,
    ):
        """Wrapper that returns (expr_out, ct_logits) when predict_celltype=True, otherwise only expr_out"""
        if not self.predict_celltype:
            expr_out = self.forward_expression(
                x=x,
                edge_index=edge_index,
                batch=batch,
                inject=inject,
                gene_names=gene_names,
                use_oracle_ct=False,
                center_celltype=None,
                ct_logits=None,
                allow_grad_through_ct=False,
            )
            return expr_out

        ct_logits = self.forward_celltype(x, edge_index, batch)
        expr_out = self.forward_expression(
            x=x,
            edge_index=edge_index,
            batch=batch,
            inject=inject,
            gene_names=gene_names,
            use_oracle_ct=use_oracle_ct,
            center_celltype=center_celltype,
            ct_logits=ct_logits,
            allow_grad_through_ct=allow_grad_through_ct,
        )
        return expr_out, ct_logits


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


def _get_celltype_params(model):
    """
    Get celltype network parameters and their IDs.
    
    Returns:
        tuple: (list of parameters, set of parameter IDs)
    """
    celltype_params = []
    if hasattr(model, 'ct_conv1'):
        celltype_params.extend(list(model.ct_conv1.parameters()))
    if hasattr(model, 'ct_convs'):
        for conv in model.ct_convs:
            celltype_params.extend(list(conv.parameters()))
    if hasattr(model, 'celltype_head'):
        celltype_params.extend(list(model.celltype_head.parameters()))
    celltype_param_ids = {id(p) for p in celltype_params}
    return celltype_params, celltype_param_ids


def _get_expression_params(model):
    """
    Get expression network parameters and their IDs.
    
    Returns:
        tuple: (list of parameters, set of parameter IDs)
    """
    expr_params = []
    if hasattr(model, 'conv1'):
        expr_params.extend(list(model.conv1.parameters()))
    if hasattr(model, 'convs'):
        for conv in model.convs:
            expr_params.extend(list(conv.parameters()))
    if hasattr(model, 'expr_mlp'):
        expr_params.extend(list(model.expr_mlp.parameters()))
    elif hasattr(model, 'lin_base'):
        expr_params.extend(list(model.lin_base.parameters()))
    # Also include GenePT-related parameters if they exist
    for attr_name in ['q_proj', 'k_proj', 'genept_to_hidden', 'xattn', 'pre_attn_ln', 'post_attn_ln', 'fuse_gate']:
        if hasattr(model, attr_name):
            expr_params.extend(list(getattr(model, attr_name).parameters()))
    expr_param_ids = {id(p) for p in expr_params}
    return expr_params, expr_param_ids


def train(
    model,
    loader,
    criterion,
    expr_optimizer,
    ct_optimizer=None,
    gene_names=None,
    inject=False,
    device="cuda",
    celltype_weight=1.0,
):
    model.train()

    for batch in loader:
        batch = batch.to(device)
        center_celltypes = [model.celltypes_to_index[item[0]] for item in batch.center_celltype]
        center_ct = torch.tensor(center_celltypes, device=device, dtype=torch.long)

        # Case 1: multitask, shared gradients
        if model.predict_celltype and model.train_multitask:
            expr_optimizer.zero_grad(set_to_none=True)

            # forward with gradient flow through ct branch
            expr_out, ct_logits = model(
                x=batch.x,
                edge_index=batch.edge_index,
                batch=batch.batch,
                inject=(batch.inject if inject else None),
                gene_names=gene_names,
                use_oracle_ct=False,
                center_celltype=None,
                allow_grad_through_ct=True,
            )

            expr_loss = criterion(expr_out, batch.y)
            ct_loss = F.cross_entropy(ct_logits, center_ct)
            total_loss = expr_loss + celltype_weight * ct_loss
            total_loss.backward()
            expr_optimizer.step()
            continue

        # Case 2: two-step decoupled training
        if model.predict_celltype and not model.train_multitask:
            # ct step (only ct optimizer, only ct params)
            ct_optimizer.zero_grad(set_to_none=True)
            ct_logits = model.forward_celltype(batch.x, batch.edge_index, batch.batch)
            ct_loss = F.cross_entropy(ct_logits, center_ct)
            ct_loss.backward()
            ct_optimizer.step()

            # Expression step (only expr optimizer, celltype treated as fixed)
            expr_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                ct_logits_detached = model.forward_celltype(batch.x, batch.edge_index, batch.batch)

            expr_out = model.forward_expression(
                x=batch.x,
                edge_index=batch.edge_index,
                batch=batch.batch,
                inject=(batch.inject if inject else None),
                gene_names=gene_names,
                use_oracle_ct=False,
                center_celltype=None,
                ct_logits=ct_logits_detached,
                allow_grad_through_ct=False,   # ensures no accidental graph links
            )
            expr_loss = criterion(expr_out, batch.y)
            expr_loss.backward()
            expr_optimizer.step()
            continue

        # Case 3: Expression-only model
        expr_optimizer.zero_grad(set_to_none=True)
        out = model.forward_expression(
            x=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch,
            inject=(batch.inject if inject else None),
            gene_names=gene_names,
            ct_logits=None,
            use_oracle_ct=False,
        )
        loss = criterion(out, batch.y)
        loss.backward()
        expr_optimizer.step()


def test(
    model, 
    loader, 
    loss, 
    criterion, 
    gene_names=None, 
    inject=False, 
    device="cuda",
):
    model.eval()
    errors = []
    all_preds = []
    all_targets = []

    all_ct_preds = []
    all_ct_targets = []

    for batch in loader: 
        batch.to(device)
        
        center_celltypes = [model.celltypes_to_index[item[0]] for item in batch.center_celltype]
        # Prepare arguments for forward pass
        forward_args = {
            'x': batch.x,
            'edge_index': batch.edge_index, 
            'batch': batch.batch,
            'inject': batch.inject if inject else None,
            'gene_names': gene_names,
            'use_oracle_ct': False,
        }
        
        forward_output = model(**forward_args)
        
        # Handle both single-task and multi-task outputs
        if model.predict_celltype:
            expr_out, celltype_logits = forward_output  # Unpack tuple
            all_ct_preds.append(celltype_logits)
            all_ct_targets.append(center_celltypes)
        else:
            expr_out = forward_output
        
        target = batch.y.unsqueeze(1)

        all_preds.append(expr_out.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())
        
        if loss == "mse":
            errors.append(F.mse_loss(expr_out, target).sqrt().item())
        elif loss == "l1":
            errors.append(F.l1_loss(expr_out, target).item())
        elif loss == "weightedl1":
            errors.append(weighted_l1_loss(expr_out, target, criterion.zero_weight, criterion.nonzero_weight).item())
        elif loss == "balanced_mse":
            errors.append(bmc_loss(expr_out, target, criterion.noise_sigma**2).item())
        elif loss == "npcc":
            errors.append(npcc_loss(expr_out, target).item())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_ct_preds = torch.cat(all_ct_preds, dim=0)
    all_ct_targets = np.concatenate(all_ct_targets)

    spearman = compute_spearman(all_preds, all_targets)
    celltype_accuracy = compute_celltype_accuracy(all_ct_preds, all_ct_targets)

    return np.mean(errors), spearman, celltype_accuracy

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
            
            forward_output = model(**forward_args)
            
            # Handle both single-task and multi-task outputs
            if model.predict_celltype:
                out, _ = forward_output  # Unpack tuple, use expression output for prediction
            else:
                out = forward_output
            
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
