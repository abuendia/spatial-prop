# %% 
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_palette("deep")

# ------------------------
# 1. Helpers to load metrics
# ------------------------

def load_metrics_macro_micro(path: str) -> dict:
    """Load metrics from a macro/micro CSV (with 'Micro - Pearson', etc.)."""
    df = pd.read_csv(path)
    return {
        "pearson": df.loc[df["Metric"] == "Micro - Pearson", "Value"].iloc[0],
        "spearman": df.loc[df["Metric"] == "Micro - Spearman", "Value"].iloc[0],
        "mae": df.loc[df["Metric"] == "Micro - MAE", "Value"].iloc[0],
        "pearson_nonzero": df.loc[df["Metric"] == "Micro (Nonzero) - Pearson", "Value"].iloc[0],
        "spearman_nonzero": df.loc[df["Metric"] == "Micro (Nonzero) - Spearman", "Value"].iloc[0],
        "mae_nonzero": df.loc[df["Metric"] == "Micro (Nonzero) - MAE", "Value"].iloc[0],
    }


def load_metrics_micro_only(path: str) -> dict:
    """Load metrics from a 'micro only' CSV (Metric,Value with 'Pearson', 'Pearson (Nonzero)', etc.)."""
    df = pd.read_csv(path)
    return {
        "pearson": df.loc[df["Metric"] == "Pearson", "Value"].iloc[0],
        "spearman": df.loc[df["Metric"] == "Spearman", "Value"].iloc[0],
        "mae": df.loc[df["Metric"] == "MAE", "Value"].iloc[0],
        "pearson_nonzero": df.loc[df["Metric"] == "Pearson (Nonzero)", "Value"].iloc[0],
        "spearman_nonzero": df.loc[df["Metric"] == "Spearman (Nonzero)", "Value"].iloc[0],
        "mae_nonzero": df.loc[df["Metric"] == "MAE (Nonzero)", "Value"].iloc[0],
    }


def safe_load_metrics(path: str, loader, label: str = "") -> dict:
    """Wrapper that prints a warning instead of crashing if a file is missing."""
    if not os.path.exists(path):
        print(f"[WARN] File not found for {label}: {path}")
        return None
    try:
        return loader(path)
    except Exception as e:
        print(f"[WARN] Failed to load {label} from {path}: {e}")
        return None


# ------------------------
# 2. Collect all rows: GNN 2/3-hop + baselines
# ------------------------

k_hop_options = [2, 3]
pool_options = ["ASAPooling", "center_pool", "GlobalAttention"]
datasets = ["androvic", "reprogramming"]

rows = []

# --- GNN models (2- and 3-hop, three pooling strategies) ---
for dataset in datasets:
    for k_hop in k_hop_options:
        for pool in pool_options:
            results_file = (
                f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/output/"
                f"expression_with_celltype_decoupled_khop{k_hop}_no_genept_softmax_ct_{pool}/"
                f"{dataset}_expression_{k_hop}hop_2augment_expression_none/weightedl1_1en04/"
                "test_evaluation_stats_macro_micro.csv"
            )
            metrics = safe_load_metrics(
                results_file,
                load_metrics_macro_micro,
                label=f"{dataset}, {k_hop}-hop, {pool}",
            )
            if metrics is None:
                continue

            row = {
                "dataset": dataset,
                "k_hop": k_hop,
                "pool": pool,
            }
            row.update(metrics)
            rows.append(row)

# --- Baseline models: 2-hop and 3-hop for khop_mean/global_mean, 2-hop for no_graph_decoupled ---

# NOTE: now each tuple is (pool_name, k_hop, path, loader)
baseline_paths = {
    "androvic": [
        # 2-hop baselines
        (
            "khop_mean",
            2,
            "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/baselines/"
            "androvic_expression_100per_2hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/"
            "khop_mean/test_evaluation_stats_macro_micro.csv",
            load_metrics_macro_micro,
        ),
        (
            "global_mean",
            2,
            "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/baselines/"
            "androvic_expression_100per_2hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/"
            "global_mean/test_evaluation_stats_macro_micro.csv",
            load_metrics_macro_micro,
        ),
        (
            "no_graph_decoupled",
            2,
            "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/output/final_exps/"
            "expression_with_celltype_decoupled_no_genept_softmax_ct/"
            "androvic_expression_2hop_2augment_expression_none/weightedl1_1en04/"
            "test_evaluation_stats_micro.csv",
            load_metrics_micro_only,
        ),
        # 3-hop baselines (NEW)
        (
            "khop_mean",
            3,
            "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/baselines/"
            "androvic_expression_100per_3hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/"
            "khop_mean/test_evaluation_stats_macro_micro.csv",
            load_metrics_macro_micro,
        ),
        (
            "global_mean",
            3,
            "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/baselines/"
            "androvic_expression_100per_3hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/"
            "global_mean/test_evaluation_stats_macro_micro.csv",
            load_metrics_macro_micro,
        ),
    ],
    "reprogramming": [
        # 2-hop baselines
        (
            "khop_mean",
            2,
            "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/baselines/"
            "reprogramming_expression_100per_2hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/"
            "khop_mean/test_evaluation_stats_macro_micro.csv",
            load_metrics_macro_micro,
        ),
        (
            "global_mean",
            2,
            "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/baselines/"
            "reprogramming_expression_100per_2hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/"
            "global_mean/test_evaluation_stats_macro_micro.csv",
            load_metrics_macro_micro,
        ),
        (
            "no_graph_decoupled",
            2,
            "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/output/final_exps/"
            "expression_with_celltype_decoupled_no_genept_softmax_ct/"
            "reprogramming_expression_2hop_2augment_expression_none/weightedl1_1en04/"
            "test_evaluation_stats_micro.csv",
            load_metrics_micro_only,
        ),
        # 3-hop baselines (NEW)
        (
            "khop_mean",
            3,
            "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/baselines/"
            "reprogramming_expression_100per_3hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/"
            "khop_mean/test_evaluation_stats_macro_micro.csv",
            load_metrics_macro_micro,
        ),
        (
            "global_mean",
            3,
            "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/baselines/"
            "reprogramming_expression_100per_3hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/"
            "global_mean/test_evaluation_stats_macro_micro.csv",
            load_metrics_macro_micro,
        ),
    ],
}

for dataset, specs in baseline_paths.items():
    for pool_name, k_hop, path, loader in specs:
        metrics = safe_load_metrics(
            path,
            loader,
            label=f"{dataset}, baseline {pool_name}, {k_hop}-hop",
        )
        if metrics is None:
            continue

        row = {
            "dataset": dataset,
            "k_hop": k_hop,  # <-- now correctly 2 or 3
            "pool": pool_name,
        }
        row.update(metrics)
        rows.append(row)

# ------------------------
# 3. Build DataFrame
# ------------------------

metrics_df = pd.DataFrame(rows)
print("Data shape:", metrics_df.shape)
print(metrics_df.head())

# Create combined label: "2-hop ASAPooling", "3-hop center_pool", "2-hop khop_mean", etc.
metrics_df["pool_k"] = metrics_df.apply(
    lambda r: f"{int(r['k_hop'])}-hop {r['pool']}",
    axis=1,
)

# ------------------------
# 4. Compute best models per dataset/metric (excluding baselines)
# ------------------------

metrics = [
    "pearson",
    "pearson_nonzero",
    "spearman",
    "spearman_nonzero",
    "mae",
    "mae_nonzero",
]

baseline_pools = ["khop_mean", "global_mean", "no_graph_decoupled"]

# Exclude baselines for "best model" calculation (works for both 2-hop and 3-hop baselines)
gnn_only_df = metrics_df[~metrics_df["pool"].isin(baseline_pools)].copy()

print("\n========== Best GNN Models per Dataset (Excluding Baselines) ==========")

for metric in metrics:
    print(f"\n--- Metric: {metric} ---")
    for dataset in datasets:
        df_sub = gnn_only_df[gnn_only_df["dataset"] == dataset]
        if df_sub.empty:
            print(f"  {dataset}: [no data]")
            continue

        if metric.startswith("mae"):  # lower is better
            idx = df_sub[metric].idxmin()
            best_row = gnn_only_df.loc[idx]
            direction = "LOWEST"
        else:  # higher is better
            idx = df_sub[metric].idxmax()
            best_row = gnn_only_df.loc[idx]
            direction = "HIGHEST"

        print(
            f"  {dataset}: {direction} {metric} = {best_row[metric]:.4f} | "
            f"model = {int(best_row['k_hop'])}-hop {best_row['pool']}"
        )

# ------------------------
# 5. Plotting
# ------------------------

dataset_order = ["androvic", "reprogramming"]

# Define all possible model labels in the order you care about
base_pools = ["ASAPooling", "center_pool", "GlobalAttention",
              "khop_mean", "global_mean", "no_graph_decoupled"]

# Now allow both 2-hop and 3-hop for ALL pools (filter will drop missing combos)
pool_k_order = (
    [f"2-hop {p}" for p in base_pools] +
    [f"3-hop {p}" for p in base_pools]
)

# Filter pool_k_order to only those that actually exist in the data
existing_pool_k = metrics_df["pool_k"].unique().tolist()
pool_k_order = [pk for pk in pool_k_order if pk in existing_pool_k]

for metric in metrics:
    plt.figure(figsize=(9, 4))
    sns.barplot(
        data=metrics_df,
        x="dataset",
        y=metric,
        hue="pool_k",
        order=dataset_order,
        hue_order=pool_k_order,
        palette="deep",
    )
    plt.title(f"{metric.replace('_', ' ').title()} (2-hop vs 3-hop + baselines)")
    plt.xlabel("Dataset")
    plt.ylabel(metric.replace("_", " ").title())
    plt.legend(
        title="Model (k-hop / pooling)",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    plt.tight_layout()
    plt.show()
# %%
