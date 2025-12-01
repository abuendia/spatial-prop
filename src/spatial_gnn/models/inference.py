import torch
import numpy as np
import tqdm

from spatial_gnn.utils.perturbation_utils import temper


def predict(
    model,
    adata,
    dataloader,
    perturbed_dataloader=None,
    gene_names=None,
    use_ids=None,
    device="cuda", 
    temper_method="distribution_renormalize"
):
    """
    Run GNN model predictions and convert back to AnnData format using the stored mapping info.
    
    Parameters:
    -----------
    model : GNN
        Trained GNN model
    adata : AnnData
        Original AnnData object used to create the dataset
    dataloader : DataLoader
        PyTorch Geometric dataloader for unperturbed predictions
    perturbed_dataloader : DataLoader
        PyTorch Geometric dataloader for perturbed predictions. If provided along with 
        temper_method, tempering will be applied to combine predictions.
    gene_names : list, optional
        List of gene names
    use_ids : list, optional
        List of IDs to use
    device : str
        Device to run inference on
    temper_method : str, optional
        Method for tempering predictions. Options: "none", "prediction_delta", "distribution",
        "distribution_dampen", "distribution_renormalize", "renormalize". Requires perturbed_dataloader to be provided.
    Returns:
    --------
    AnnData
        Updated AnnData with predictions in .layers['predicted_perturbed'] (GNN output) and 
        .layers['predicted_tempered'] (after SparseRenorm)
    """    
    model.eval()
    adata = adata.copy()
    if use_ids is not None:
        adata = adata[adata.obs['mouse_id'].isin(use_ids)]

    n_cells, n_genes = adata.shape
    predicted_cells = set()
    batch_count = 0    

    unperturbed_prediction_matrix = np.full((n_cells, n_genes), np.nan, dtype=np.float32)
    perturbed_prediction_matrix = np.full((n_cells, n_genes), np.nan, dtype=np.float32)
    tempered_prediction_matrix = np.full((n_cells, n_genes), np.nan, dtype=np.float32)
    
    print(f"Starting prediction for {n_cells} cells, {n_genes} genes")
    with torch.no_grad():
        for batch, perturbed_batch in tqdm(
            zip(dataloader, perturbed_dataloader),
            total=len(dataloader),
            desc="Predicting (unperturbed & perturbed) with tempering..."
        ):
            batch.to(device)
            perturbed_batch.to(device)
            batch_count += 1

            # ---------- unperturbed forward ----------
            forward_args_unpert = {
                'x': batch.x,
                'edge_index': batch.edge_index, 
                'batch': batch.batch,
                'center_cell_idx': batch.center_node,
                'inject': batch.inject if hasattr(batch, 'inject') and batch.inject is not None else None,
                'gene_names': gene_names,
            }
            forward_output_unpert = model(**forward_args_unpert)
            if model.predict_celltype:
                out_unpert, _ = forward_output_unpert
            else:
                out_unpert = forward_output_unpert

            # ---------- perturbed forward ----------
            forward_args_pert = {
                'x': perturbed_batch.x,
                'edge_index': perturbed_batch.edge_index, 
                'batch': perturbed_batch.batch,
                'center_cell_idx': perturbed_batch.center_node,
                'inject': perturbed_batch.inject if hasattr(perturbed_batch, 'inject') and perturbed_batch.inject is not None else None,
                'gene_names': gene_names,
            }
            forward_output_pert = model(**forward_args_pert)
            if model.predict_celltype:
                out_pert, _ = forward_output_pert
            else:
                out_pert = forward_output_pert

            # Actual expression
            if batch.y.shape != out_unpert.shape:
                actual = batch.y.float().reshape_as(out_unpert)
            else:
                actual = batch.y.float()
            batch_predictions_unpert = out_unpert.detach().cpu().numpy()
            batch_predictions_pert = out_pert.detach().cpu().numpy()

            # Collect indices for this batch
            batch_indices = []
            for i in range(batch_predictions_unpert.shape[0]):
                original_cell_id = batch.original_cell_id[i]
                original_cell_idx = adata.obs_names.get_loc(original_cell_id)
                perturbed_batch_id = perturbed_batch.original_cell_id[i]
                perturbed_cell_idx = adata.obs_names.get_loc(perturbed_batch_id)
                assert original_cell_idx == perturbed_cell_idx

                unperturbed_prediction_matrix[original_cell_idx, :] = batch_predictions_unpert[i]
                perturbed_prediction_matrix[perturbed_cell_idx, :] = batch_predictions_pert[i]
                predicted_cells.add(original_cell_idx)
                batch_indices.append(original_cell_idx)

            true_expn_batch = torch.tensor(
                actual,
                dtype=torch.float32,
                device=device,
            )
            pred_expn_batch = torch.tensor(
                batch_predictions_unpert,
                dtype=torch.float32,
                device=device,
            )
            pred_perturb_expn_batch = torch.tensor(
                batch_predictions_pert,
                dtype=torch.float32,
                device=device,
            )
            tempered_batch = temper(
                true_expn=true_expn_batch,
                pred_expn=pred_expn_batch,
                pred_perturb_expn=pred_perturb_expn_batch,
                method=temper_method,
            )

            tempered_batch_np = tempered_batch.detach().cpu().numpy()
            tempered_prediction_matrix[batch_indices, :] = tempered_batch_np

    print(f"Predicted on {len(predicted_cells)} cells")

    adata.layers["predicted_unperturbed"] = unperturbed_prediction_matrix
    adata.layers["predicted_perturbed"] = perturbed_prediction_matrix
    adata.layers["predicted_tempered"] = tempered_prediction_matrix
    print(f"Applied tempering with method '{temper_method}' (batch-wise)")
    return adata
