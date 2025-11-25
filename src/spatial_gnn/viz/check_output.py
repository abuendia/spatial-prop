# %%
import os
import pandas as pd
import numpy as np

# ------------------------
# Config (from your prompt)
# ------------------------

model_dirs = [
    "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/results/expr_model_predict/appendix/expression_only_khop2_no_genept_softmax_ct_center_pool",
    "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/results/expr_model_predict/appendix/expression_with_celltype_decoupled_khop2_no_genept_softmax_ct_center_pool",
    "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/results/expr_model_predict/appendix/expression_with_celltype_decoupled_khop2_no_genept_softmax_ct_center_pool_predict_residuals",
    "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/results/expr_model_predict/appendix/expression_with_celltype_decoupled_khop2_no_genept_softmax_ct_GlobalAttention",
    "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/results/expr_model_predict/appendix/expression_with_celltype_decoupled_khop3_no_genept_softmax_ct_center_pool"
]

baseline_dirs = [
    "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/results/expr_mean_baselines/aging_coronal_expression_100per_2hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/global_mean",
    "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/results/expr_mean_baselines/aging_coronal_expression_100per_2hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/khop_mean"
]

model_name_to_short = {
    "expression_only_khop2_no_genept_softmax_ct_center_pool": "SpatialProp (GNN)",
    "expression_with_celltype_decoupled_khop2_no_genept_softmax_ct_center_pool": "SpatialProp (GNN + CT)",
    "expression_with_celltype_decoupled_khop2_no_genept_softmax_ct_center_pool_predict_residuals": "SpatialProp (GNN + CT + Residuals)",
    "expression_with_celltype_decoupled_khop2_no_genept_softmax_ct_GlobalAttention": "SpatialProp (GNN + CT + Global Attention)",
    "expression_with_celltype_decoupled_khop3_no_genept_softmax_ct_center_pool": "SpatialProp (GNN + CT + 3-hop)",
    "global_mean": "Baseline (Train Global Mean)",
    "khop_mean": "Baseline (K-hop Test Graph Mean)",
}

datasets = [
    "aging_coronal", "aging_sagittal", "exercise", "reprogramming",
    "kukanja", "androvic", "zeng", "pilot", "farah"
]

dataset_name_to_short = {
    "aging_coronal": "AC",
    "aging_sagittal": "AS",
    "exercise": "EX",
    "reprogramming": "RE",
    "kukanja": "EA",
    "androvic": "MS",
    "zeng": "AD",
    "pilot": "AP",
    "farah": "HD",
}

# ------------------------
# Helpers
# ------------------------

def load_micro_nonzero_metrics(csv_path: str):
    """Return (spearman, mae) from the macro/micro CSV."""
    df = pd.read_csv(csv_path)
    try:
        spearman = float(
            df.loc[df["Metric"] == "Micro (Nonzero) - Spearman", "Value"].iloc[0]
        )
    except IndexError:
        raise ValueError(f"'Micro (Nonzero) - Spearman' not found in {csv_path}")

    try:
        mae = float(
            df.loc[df["Metric"] == "Micro (Nonzero) - MAE", "Value"].iloc[0]
        )
    except IndexError:
        raise ValueError(f"'Micro (Nonzero) - MAE' not found in {csv_path}")

    return spearman, mae


def find_model_csv_for_dataset(model_root: str, dataset: str):
    """
    Search recursively under model_root for test_evaluation_stats_macro_micro.csv
    whose path contains the dataset name.
    """
    candidates = []
    for root, dirs, files in os.walk(model_root):
        if "test_evaluation_stats_macro_micro.csv" in files and dataset in root:
            candidates.append(os.path.join(root, "test_evaluation_stats_macro_micro.csv"))

    if not candidates:
        print(f"WARNING: no CSV found for dataset '{dataset}' under {model_root}")
        return None

    candidates = sorted(candidates)
    if len(candidates) > 1:
        print(
            f"NOTE: multiple CSVs for dataset '{dataset}' under {model_root}. "
            f"Using {candidates[0]}"
        )
    return candidates[0]


# ------------------------
# Baseline path template (generalize aging_coronal path)
# ------------------------

# Use the first baseline dir to infer the template
first_baseline_dir = baseline_dirs[0]  # .../expr_mean_baselines/aging_coronal_.../global_mean
baseline_dataset_dir = os.path.dirname(first_baseline_dir)  # .../aging_coronal_expression_...
baseline_root = os.path.dirname(baseline_dataset_dir)       # .../expr_mean_baselines

template_dataset = datasets[0]  # assume "aging_coronal"
baseline_dataset_template = os.path.basename(baseline_dataset_dir)  # "aging_coronal_expression_..."
# everything after the dataset name
suffix = baseline_dataset_template.split(template_dataset, 1)[1]    # "_expression_100per_..."

# ------------------------
# Set up result tables
# ------------------------

dataset_short_names = [dataset_name_to_short[d] for d in datasets]

# Column order: baselines first (in the order of baseline_dirs), then models (in order of model_dirs)
columns_short = []

# Add baselines
for bdir in baseline_dirs:
    key = os.path.basename(bdir)  # "global_mean", "khop_mean"
    short = model_name_to_short[key]
    if short not in columns_short:
        columns_short.append(short)

# Add models
for mdir in model_dirs:
    key = os.path.basename(mdir)  # e.g. expression_only_khop2_...
    short = model_name_to_short[key]
    if short not in columns_short:
        columns_short.append(short)

spearman_df = pd.DataFrame(index=dataset_short_names, columns=columns_short, dtype=float)
mae_df = pd.DataFrame(index=dataset_short_names, columns=columns_short, dtype=float)

# ------------------------
# Fill baselines
# ------------------------

for bdir in baseline_dirs:
    baseline_name = os.path.basename(bdir)  # "global_mean" or "khop_mean"
    col_short = model_name_to_short[baseline_name]

    for ds in datasets:
        ds_short = dataset_name_to_short[ds]
        ds_dirname = f"{ds}{suffix}"
        csv_path = os.path.join(
            baseline_root,
            ds_dirname,
            baseline_name,
            "test_evaluation_stats_macro_micro.csv",
        )

        if not os.path.exists(csv_path):
            print(f"WARNING: baseline CSV missing: {csv_path}")
            continue

        try:
            spearman, mae = load_micro_nonzero_metrics(csv_path)
        except Exception as e:
            print(f"ERROR reading {csv_path}: {e}")
            continue

        spearman_df.loc[ds_short, col_short] = spearman
        mae_df.loc[ds_short, col_short] = mae

# ------------------------
# Fill GNN models
# ------------------------

for mdir in model_dirs:
    model_key = os.path.basename(mdir)
    col_short = model_name_to_short[model_key]

    for ds in datasets:
        ds_short = dataset_name_to_short[ds]
        csv_path = find_model_csv_for_dataset(mdir, ds)
        if csv_path is None:
            continue

        try:
            spearman, mae = load_micro_nonzero_metrics(csv_path)
        except Exception as e:
            print(f"ERROR reading {csv_path}: {e}")
            continue

        spearman_df.loc[ds_short, col_short] = spearman
        mae_df.loc[ds_short, col_short] = mae

# ------------------------
# Done: spearman_df and mae_df have rows = dataset short, columns = model short
# ------------------------

print("Micro (Nonzero) Spearman table:")
print(spearman_df)

print("\nMicro (Nonzero) MAE table:")
print(mae_df)

# Optionally save to CSV
# spearman_df.to_csv("micro_nonzero_spearman_table.csv")
# mae_df.to_csv("micro_nonzero_mae_table.csv")

# %%

# round all values to 3 decimal places
spearman_df = spearman_df.round(3)
mae_df = mae_df.round(3)

# %%

# reorder columns
column_order = [
    "SpatialProp (GNN)",
    "Baseline (Train Global Mean)",
    "Baseline (K-hop Test Graph Mean)",
    "SpatialProp (GNN + CT)",
    "SpatialProp (GNN + CT + Global Attention)",
    "SpatialProp (GNN + CT + Residuals)",
    "SpatialProp (GNN + CT + 3-hop)",
]

spearman_df = spearman_df[column_order]
mae_df = mae_df[column_order]

# %%

# save to csv

spearman_df = spearman_df.applymap(lambda x: f"{x:.3f}")
mae_df = mae_df.applymap(lambda x: f"{x:.3f}")

# %%
spearman_df.to_csv("/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/results/appendix_summary/micro_nonzero_spearman_table.csv", index=True)
mae_df.to_csv("/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/results/appendix_summary/micro_nonzero_mae_table.csv", index=True)

# %%