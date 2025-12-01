import torch
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F

from spatial_gnn.utils.metric_utils import compute_spearman, compute_celltype_accuracy
from spatial_gnn.models.baselines import khop_mean_baseline_batch
from spatial_gnn.models.losses import weighted_l1_loss, bmc_loss, npcc_loss


def train(
    model,
    loader,
    criterion,
    expr_optimizer,
    gene_names=None,
    inject=False,
    device="cuda",
    celltype_weight=1.0,
    class_weights=None,
    use_oracle_ct=False,
):
    model.train()

    for batch in loader:
        batch = batch.to(device)
        center_celltypes = [model.celltypes_to_index[item[0]] for item in batch.center_celltype]
        center_ct = torch.tensor(center_celltypes, device=device, dtype=torch.long)

        # Compute baseline and adjust targets if predicting residuals
        target_expr = batch.y
        if model.predict_residuals:
            k_hop_baseline = khop_mean_baseline_batch(
                x=batch.x,
                batch=batch.batch,
                center_nodes=batch.center_node,
            )
            k_hop_baseline = k_hop_baseline.flatten()
            target_expr = target_expr - k_hop_baseline

        # Case 1: multitask, shared gradients
        if model.predict_celltype and model.train_multitask:
            expr_optimizer.zero_grad(set_to_none=True)

            # forward with gradient flow through ct branch
            expr_out, ct_logits = model(
                x=batch.x,
                edge_index=batch.edge_index,
                batch=batch.batch,
                center_cell_idx=batch.center_node,
                inject=(batch.inject if inject else None),
                gene_names=gene_names,
                use_oracle_ct=use_oracle_ct,
                center_celltype=center_celltypes,
                allow_grad_through_ct=True,
            )

            expr_loss = criterion(expr_out, target_expr)
            ct_loss = F.cross_entropy(ct_logits, center_ct, weight=class_weights)
            total_loss = expr_loss + celltype_weight * ct_loss
            total_loss.backward()
            expr_optimizer.step()
            continue

        # Case 2: decoupled training - expression only (cell type model already trained separately)
        if model.predict_celltype and not model.train_multitask:
            # Cell type predictions come from pre-trained celltype_model
            expr_optimizer.zero_grad(set_to_none=True)
            
            if not use_oracle_ct:
                with torch.no_grad():
                    ct_logits = model.forward_celltype(
                        x=batch.x,
                        edge_index=batch.edge_index,
                        batch=batch.batch,
                        center_cell_idx=batch.center_node,
                    )
            else:
                ct_logits = None

            expr_out = model.forward_expression(
                x=batch.x,
                edge_index=batch.edge_index,
                batch=batch.batch,
                center_cell_idx=batch.center_node,
                inject=(batch.inject if inject else None),
                gene_names=gene_names,
                use_oracle_ct=use_oracle_ct,    
                center_celltype=center_celltypes,
                ct_logits=ct_logits,
                allow_grad_through_ct=False,   # ensures no accidental graph links
            )
            expr_loss = criterion(expr_out, target_expr)
            expr_loss.backward()
            expr_optimizer.step()
            continue

        # Case 3: Expression-only model
        expr_optimizer.zero_grad(set_to_none=True)
        out = model.forward_expression(
            x=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch,
            center_cell_idx=batch.center_node,
            inject=(batch.inject if inject else None),
            gene_names=gene_names,
            ct_logits=None,
            use_oracle_ct=use_oracle_ct,
        )
        loss = criterion(out, target_expr)
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
    use_oracle_ct=False,
    is_last_epoch=False,
    save_dir=None,
    gene_list=None,
):
    model.eval()
    errors = []
    all_preds = []
    all_targets = []
    all_center_ct_strings = []

    all_ct_preds = []
    all_ct_targets = []

    for batch in loader: 
        batch.to(device)
        
        center_celltypes = [model.celltypes_to_index[item[0]] for item in batch.center_celltype]
        center_ct_strings = [item[0] for item in batch.center_celltype]
        
        # Compute baseline if predicting residuals
        if model.predict_residuals:
            k_hop_baseline = khop_mean_baseline_batch(
                x=batch.x,
                batch=batch.batch,
                center_nodes=batch.center_node,
            )
        
        # Prepare arguments for forward pass
        forward_args = {
            'x': batch.x,
            'edge_index': batch.edge_index, 
            'batch': batch.batch,
            'center_cell_idx': batch.center_node,
            'inject': batch.inject if inject else None,
            'gene_names': gene_names,
            'use_oracle_ct': use_oracle_ct,
            'center_celltype': center_celltypes,
        }
        
        forward_output = model(**forward_args)
        
        # Handle both single-task and multi-task outputs
        if model.predict_celltype:
            expr_out, celltype_logits = forward_output  # Unpack tuple
            all_ct_preds.append(celltype_logits)
        else:
            expr_out = forward_output
        
        # If predicting residuals, add baseline back to get final expression prediction
        if model.predict_residuals:
            expr_out = expr_out + k_hop_baseline
        
        target = batch.y.unsqueeze(1)

        all_preds.append(expr_out.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())
        all_ct_targets.append(center_celltypes)
        all_center_ct_strings.extend(center_ct_strings)
        
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
    all_ct_targets = np.concatenate(all_ct_targets)
    all_center_ct_strings = np.array(all_center_ct_strings)
    spearman = compute_spearman(all_preds, all_targets)

    # if last epoch save the predictions and targets to file 
    if is_last_epoch:        
        n_cells = all_preds.shape[0]
        n_genes = all_preds.shape[1]
        cell_idx = np.repeat(np.arange(n_cells), n_genes)
        gene_names = np.tile(gene_list, n_cells)
        pred_expr = all_preds.flatten()
        true_expr = all_targets.flatten()
        cell_type = np.repeat(all_center_ct_strings, n_genes)
        
        df = pd.DataFrame({
            'cell_idx': cell_idx,
            'cell_type': cell_type,
            'gene_name': gene_names,
            'pred_expr': pred_expr,
            'true_expr': true_expr,
        })
        df.to_csv(os.path.join(save_dir, "last_epoch_preds.csv"), index=False)
        
    if model.predict_celltype:
        all_ct_preds = torch.cat(all_ct_preds, dim=0)
        celltype_accuracy = compute_celltype_accuracy(all_ct_preds, all_ct_targets)
        return np.mean(errors), spearman, celltype_accuracy
    else:
        return np.mean(errors), spearman, 0.0


def train_celltype_model(
    celltype_model,
    loader,
    class_weights=None,
    device="cuda",
    epochs=1,
):
    """
    Train a separate cell type model independently.
    Used in decoupled training mode.
    """
    celltype_model.train()
    optimizer = torch.optim.Adam(celltype_model.parameters(), lr=1e-5)
    
    for epoch in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            center_celltypes = [celltype_model.celltypes_to_index[item[0]] for item in batch.center_celltype]
            center_ct = torch.tensor(center_celltypes, device=device, dtype=torch.long)
            
            optimizer.zero_grad(set_to_none=True)
            ct_logits = celltype_model(
                x = batch.x,
                edge_index=batch.edge_index,
                batch=batch.batch,
                center_cell_idx=batch.center_node,
            )
            ct_loss = F.cross_entropy(ct_logits, center_ct, weight=class_weights)
            ct_loss.backward()
            optimizer.step()
    
    return celltype_model
