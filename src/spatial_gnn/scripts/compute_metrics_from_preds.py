import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import os
from tqdm import tqdm

from spatial_gnn.scripts.model_performance import robust_nanmean


def fast_compute_metrics_from_preds(preds_df, save_dir):
    # get gene names that have max value of 0
    gene_max = preds_df.groupby("gene_name")["true_expr"].max()
    genes_missing = gene_max[gene_max == 0].index.tolist()
    print(f"Dropping {len(genes_missing)} genes that have max value of 0")
    preds_df = preds_df[~preds_df["gene_name"].isin(genes_missing)]

    # Calculate micro averages over all genes
    print("Micro averages")
    preds_flat = preds_df["pred_expr"].values
    actuals_flat = preds_df["true_expr"].values
    micro_r, _ = pearsonr(preds_flat, actuals_flat)
    micro_s, _ = spearmanr(preds_flat, actuals_flat)
    micro_r2 = r2_score(actuals_flat, preds_flat)
    micro_mae = np.mean(np.abs(preds_flat - actuals_flat))
    micro_rmse = np.sqrt(np.mean((preds_flat - actuals_flat)**2))
    print(f"Pearson: {micro_r}, Spearman: {micro_s}, R2: {micro_r2}, MAE: {micro_mae}, RMSE: {micro_rmse}")

    # Calculate micro averages for non-zero genes
    print("Micro averages (non-zero genes)")
    preds_flat_nonzero = preds_flat[actuals_flat != 0]
    actuals_flat_nonzero = actuals_flat[actuals_flat != 0]
    micro_r_nonzero, _ = pearsonr(preds_flat_nonzero, actuals_flat_nonzero)
    micro_s_nonzero, _ = spearmanr(preds_flat_nonzero, actuals_flat_nonzero)
    micro_r2_nonzero = r2_score(actuals_flat_nonzero, preds_flat_nonzero)
    micro_mae_nonzero = np.mean(np.abs(preds_flat_nonzero - actuals_flat_nonzero))
    micro_rmse_nonzero = np.sqrt(np.mean((preds_flat_nonzero - actuals_flat_nonzero)**2))
    print(f"Pearson: {micro_r_nonzero}, Spearman: {micro_s_nonzero}, R2: {micro_r2_nonzero}, MAE: {micro_mae_nonzero}, RMSE: {micro_rmse_nonzero}")

    metrics_df = pd.DataFrame({
        "Metric": ["Pearson", "Spearman", "R2", "MAE", "RMSE", "Pearson (Nonzero)", "Spearman (Nonzero)", "R2 (Nonzero)", "MAE (Nonzero)", "RMSE (Nonzero)"],
        "Value": [micro_r, micro_s, micro_r2, micro_mae, micro_rmse, micro_r_nonzero, micro_s_nonzero, micro_r2_nonzero, micro_mae_nonzero, micro_rmse_nonzero]
    })
    metrics_df.to_csv(os.path.join(save_dir, "test_evaluation_stats_micro.csv"), index=False)


def compute_metrics_from_preds(preds_df, save_dir):
    # Calculate micro and macro averages
    print("Computing micro and macro averages...", flush=True)

    # Reshape into cell by gene matrix using pivot
    # Sort to ensure consistent ordering
    preds_df = preds_df.sort_values(['cell_idx', 'gene_name'])
    
    cell_idx = np.unique(preds_df["cell_idx"].values)
    gene_names = np.unique(preds_df["gene_name"].values)
    
    # Pivot to create cell x gene matrices
    preds_pivot = preds_df.pivot_table(
        index='cell_idx', 
        columns='gene_name', 
        values='pred_expr',
        aggfunc='first'
    )
    actuals_pivot = preds_df.pivot_table(
        index='cell_idx', 
        columns='gene_name', 
        values='true_expr',
        aggfunc='first'
    )
    celltypes_pivot = preds_df.pivot_table(
        index='cell_idx', 
        columns='gene_name', 
        values='cell_type',
        aggfunc='first'
    )
    
    # Extract matrices (reorder to match cell_idx order)
    preds = preds_pivot.loc[cell_idx, gene_names].values
    actuals = actuals_pivot.loc[cell_idx, gene_names].values
    celltypes = celltypes_pivot.loc[cell_idx, gene_names[0]].values

    preds = preds[:, actuals.max(axis=0)>=0]
    actuals = actuals[:, actuals.max(axis=0)>=0]
        
    # Micro averages: computed over all individual cell-gene pairs
    preds_flat = preds.flatten()
    actuals_flat = actuals.flatten()

    # also compute over non-zero values
    preds_flat_nonzero = preds_flat[actuals_flat != 0]
    actuals_flat_nonzero = actuals_flat[actuals_flat != 0]

    micro_r, _ = pearsonr(preds_flat, actuals_flat)
    micro_s, _ = spearmanr(preds_flat, actuals_flat)
    micro_r2 = r2_score(actuals_flat, preds_flat)
    micro_mae = np.mean(np.abs(preds_flat - actuals_flat))
    micro_rmse = np.sqrt(np.mean((preds_flat - actuals_flat)**2))

    micro_r_nonzero, _ = pearsonr(preds_flat_nonzero, actuals_flat_nonzero)
    micro_s_nonzero, _ = spearmanr(preds_flat_nonzero, actuals_flat_nonzero)
    micro_r2_nonzero = r2_score(actuals_flat_nonzero, preds_flat_nonzero)
    micro_mae_nonzero = np.mean(np.abs(preds_flat_nonzero - actuals_flat_nonzero))
    micro_rmse_nonzero = np.sqrt(np.mean((preds_flat_nonzero - actuals_flat_nonzero)**2))

    # Macro averages: computed over all cell types
    ct_mean_stats_dict = {}
    ct_mean_stats_dict_nonzero = {}

    for ct in np.unique(celltypes):

        ct_mean_stats_dict[ct] = {}
        ct_mean_stats_dict_nonzero[ct] = {}

        cell_r = []
        cell_s = []
        cell_r2 = []
        cell_mae = []
        cell_rmse = []

        cell_r_nonzero = []
        cell_s_nonzero = []
        cell_r2_nonzero = []
        cell_mae_nonzero = []
        cell_rmse_nonzero = []

        for c in np.where(celltypes==ct)[0]:

            if len(preds[c,:]) > 1:
                r, p = pearsonr(preds[c,:], actuals[c,:])
                s, p = spearmanr(preds[c,:], actuals[c,:])
                r2 = r2_score(actuals[c,:], preds[c,:])
                mae = np.mean(np.abs(preds[c,:]-actuals[c,:]))
                rmse = np.sqrt(np.mean((preds[c,:]-actuals[c,:])**2))
            else:
                r = np.nan
                s = np.nan
                r2 = np.nan
                mae = np.nan
                rmse = np.nan
            
            cell_r.append(r)
            cell_s.append(s)
            cell_r2.append(r2)
            cell_mae.append(mae)
            cell_rmse.append(rmse)

            # non-zero values
            if len(preds[c,:][actuals[c,:] != 0]) > 1:
                r_nonzero, p_nonzero = pearsonr(preds[c,:][actuals[c,:] != 0], actuals[c,:][actuals[c,:] != 0])
                s_nonzero, p_nonzero = spearmanr(preds[c,:][actuals[c,:] != 0], actuals[c,:][actuals[c,:] != 0])
                r2_nonzero = r2_score(actuals[c,:][actuals[c,:] != 0], preds[c,:][actuals[c,:] != 0])
                mae_nonzero = np.mean(np.abs(preds[c,:][actuals[c,:] != 0]-actuals[c,:][actuals[c,:] != 0]))
                rmse_nonzero = np.sqrt(np.mean((preds[c,:][actuals[c,:] != 0]-actuals[c,:][actuals[c,:] != 0])**2))
            else:
                r_nonzero = np.nan
                s_nonzero = np.nan
                r2_nonzero = np.nan
                mae_nonzero = np.nan
                rmse_nonzero = np.nan
            
            cell_r_nonzero.append(r_nonzero)
            cell_s_nonzero.append(s_nonzero)
            cell_r2_nonzero.append(r2_nonzero)
            cell_mae_nonzero.append(mae_nonzero)
            cell_rmse_nonzero.append(rmse_nonzero)

        ct_mean_stats_dict[ct]["Cell - Pearson (mean)"] = robust_nanmean(cell_r)
        ct_mean_stats_dict[ct]["Cell - Spearman (mean)"] = robust_nanmean(cell_s)
        ct_mean_stats_dict[ct]["Cell - R2 (mean)"] = robust_nanmean(cell_r2)
        ct_mean_stats_dict[ct]["Cell - MAE (mean)"] = robust_nanmean(cell_mae)
        ct_mean_stats_dict[ct]["Cell - RMSE (mean)"] = robust_nanmean(cell_rmse)

        ct_mean_stats_dict_nonzero[ct]["Cell - Pearson (mean)"] = robust_nanmean(cell_r_nonzero)
        ct_mean_stats_dict_nonzero[ct]["Cell - Spearman (mean)"] = robust_nanmean(cell_s_nonzero)
        ct_mean_stats_dict_nonzero[ct]["Cell - R2 (mean)"] = robust_nanmean(cell_r2_nonzero)
        ct_mean_stats_dict_nonzero[ct]["Cell - MAE (mean)"] = robust_nanmean(cell_mae_nonzero)
        ct_mean_stats_dict_nonzero[ct]["Cell - RMSE (mean)"] = robust_nanmean(cell_rmse_nonzero)

    # get macro average as average over cell type in 
    macro_r = robust_nanmean(np.array([ct_mean_stats_dict[ct]["Cell - Pearson (mean)"] for ct in ct_mean_stats_dict.keys()]))
    macro_s = robust_nanmean(np.array([ct_mean_stats_dict[ct]["Cell - Spearman (mean)"] for ct in ct_mean_stats_dict.keys()]))
    macro_r2 = robust_nanmean(np.array([ct_mean_stats_dict[ct]["Cell - R2 (mean)"] for ct in ct_mean_stats_dict.keys()]))
    macro_mae = robust_nanmean(np.array([ct_mean_stats_dict[ct]["Cell - MAE (mean)"] for ct in ct_mean_stats_dict.keys()]))
    macro_rmse = robust_nanmean(np.array([ct_mean_stats_dict[ct]["Cell - RMSE (mean)"] for ct in ct_mean_stats_dict.keys()]))

    macro_r_nonzero = robust_nanmean(np.array([ct_mean_stats_dict_nonzero[ct]["Cell - Pearson (mean)"] for ct in ct_mean_stats_dict_nonzero.keys()]))
    macro_s_nonzero = robust_nanmean(np.array([ct_mean_stats_dict_nonzero[ct]["Cell - Spearman (mean)"] for ct in ct_mean_stats_dict_nonzero.keys()]))
    macro_r2_nonzero = robust_nanmean(np.array([ct_mean_stats_dict_nonzero[ct]["Cell - R2 (mean)"] for ct in ct_mean_stats_dict_nonzero.keys()]))
    macro_mae_nonzero = robust_nanmean(np.array([ct_mean_stats_dict_nonzero[ct]["Cell - MAE (mean)"] for ct in ct_mean_stats_dict_nonzero.keys()]))
    macro_rmse_nonzero = robust_nanmean(np.array([ct_mean_stats_dict_nonzero[ct]["Cell - RMSE (mean)"] for ct in ct_mean_stats_dict_nonzero.keys()]))

    # save macro and micro averages in one dataframe
    overall_stats_dict = {
        "Macro - Pearson": macro_r,
        "Macro - Spearman": macro_s,
        "Macro - R2": macro_r2,
        "Macro - MAE": macro_mae,
        "Macro - RMSE": macro_rmse,
        "Macro (Nonzero) - Pearson": macro_r_nonzero,
        "Macro (Nonzero) - Spearman": macro_s_nonzero,
        "Macro (Nonzero) - R2": macro_r2_nonzero,
        "Macro (Nonzero) - MAE": macro_mae_nonzero,
        "Macro (Nonzero) - RMSE": macro_rmse_nonzero,
        "Micro - Pearson": micro_r,
        "Micro - Spearman": micro_s,
        "Micro - R2": micro_r2,
        "Micro - MAE": micro_mae,
        "Micro - RMSE": micro_rmse,
        "Micro (Nonzero) - Pearson": micro_r_nonzero,
        "Micro (Nonzero) - Spearman": micro_s_nonzero,
        "Micro (Nonzero) - R2": micro_r2_nonzero,
        "Micro (Nonzero) - MAE": micro_mae_nonzero,
        "Micro (Nonzero) - RMSE": micro_rmse_nonzero,
    }
    overall_stats_dict = pd.DataFrame(
        list(overall_stats_dict.items()),
        columns=["Metric", "Value"]
    )
    overall_stats_dict.to_csv(os.path.join(save_dir, "test_evaluation_stats_macro_micro.csv"), index=False)

    # save the dictionary for each cell type
    with open(os.path.join(save_dir, "test_evaluation_stats_bycelltype.pkl"), 'wb') as f:
        pickle.dump(ct_mean_stats_dict, f)
    with open(os.path.join(save_dir, "test_evaluation_stats_bycelltype_nonzero_genes.pkl"), 'wb') as f:
        pickle.dump(ct_mean_stats_dict_nonzero, f)
    
    print("Finished cell type analysis.", flush=True)

if __name__ == "__main__":
    datasets = [
        "aging_coronal", 
        "aging_sagittal", 
        "exercise",
        "reprogramming", 
        "kukanja",
        "androvic",
        "zeng", 
        "pilot"
    ]
    models = [
        "expression_with_celltype_decoupled_no_genept_oracle_ct_ablate_gene_expression_one_hot_ct",
        "expression_with_celltype_decoupled_no_genept_oracle_ct_softmax_ct",
        "expression_with_celltype_decoupled_no_genept_one_hot_ct",
        "expression_with_celltype_decoupled_no_genept_softmax_ct",
        "expression_only_no_genept_softmax_ct"
    ]

    all_pairs = [(model, dataset) for model in models for dataset in datasets]

    for model, dataset in tqdm(all_pairs):
        pred_file = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/output/final_exps/{model}/{dataset}_expression_2hop_2augment_expression_none/weightedl1_1en04/last_epoch_preds.csv"
        preds_df = pd.read_csv(pred_file)
        save_dir = os.path.dirname(pred_file)
        fast_compute_metrics_from_preds(preds_df, save_dir)
