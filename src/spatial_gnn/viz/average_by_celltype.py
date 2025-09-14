import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle


metric_name = "Cell - Spearman (mean)"
datasets = ["aging_coronal", "aging_sagittal", "exercise", "reprogramming", "kukanja", "androvic", "zeng", "pilot", "liverperturb", "lohoff"]
exclude_datasets = ["allen", "lohoff", "liverperturb"]  # excluded from plot


dataset_rename = {
    "aging_coronal": "Sun et al. 2024 (aging coronal)",
    "aging_sagittal": "Sun et al. 2024 (aging sagittal)",
    "exercise": "Sun et al. 2024 (exercise)",
    "reprogramming": "Sun et al. 2024 (reprogrammed)",
    "pilot": "Sun et al. 2024 (pilot)",
    "androvic": "Androvic et al. 2023",
    "kukanja": "Kukanja et al. 2024",
    "zeng": "Zeng et al. 2023",
    "allen": "Allen et al. 2023",
    "lohoff": "Lohoff et al. 2021",
    "liverperturb": "Liverperturb et al. 2023",
}

def macro_avg(metric_dict, metric_key):
    """Average metric across cell types; ignores missing/NaN."""
    vals = []
    for ct, metrics in metric_dict.items():
        if isinstance(metrics, dict) and metric_key in metrics:
            v = metrics[metric_key]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(v)
    return float(np.mean(vals)) if len(vals) else np.nan

def plot_average_by_celltype():
    rows = []
    label_map = {
        "results_without_genept": "GNN (no GenePT)",
        "results_with_genept": "GNN (with GenePT)",
        "global_mean_baseline": "Global mean baseline",
        "khop_mean_baseline": "k-hop mean",
        "khop_celltype_mean_baseline": "k-hop celltype mean",
    }

    # Compute macro averages per dataset/model
    for dataset in datasets:
        if dataset in exclude_datasets:
            continue
        results_without_genept_file = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/results/gnn/{dataset}_expression_100per_2hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/weightedl1_1en04/test_evaluation_stats_bycelltype.pkl"
        results_with_genept_file = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/results/gnn/{dataset}_expression_100per_2hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/weightedl1_1en04_GenePT/test_evaluation_stats_bycelltype.pkl"
        global_mean_baseline_file = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/results/baselines_updated/{dataset}_expression_100per_2hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/global_mean/test_evaluation_stats_bycelltype.pkl"
        khop_mean_baseline_file = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/results/baselines_updated/{dataset}_expression_100per_2hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/khop_mean/test_evaluation_stats_bycelltype.pkl"
        khop_celltype_mean_baseline_file = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/results/baselines_updated/{dataset}_expression_100per_2hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/khop_celltype_mean/test_evaluation_stats_bycelltype.pkl"

        with open(results_without_genept_file, "rb") as f:
            results_without_genept = pickle.load(f)
        with open(results_with_genept_file, "rb") as f:
            results_with_genept = pickle.load(f)
        with open(global_mean_baseline_file, "rb") as f:
            global_mean_baseline = pickle.load(f)
        with open(khop_mean_baseline_file, "rb") as f:
            khop_mean_baseline = pickle.load(f)
        with open(khop_celltype_mean_baseline_file, "rb") as f:
            khop_celltype_mean_baseline = pickle.load(f)

        model_dicts = {
            "results_without_genept": results_without_genept,
            "results_with_genept": results_with_genept,
            "global_mean_baseline": global_mean_baseline,
            # "khop_mean_baseline": khop_mean_baseline,
            # "khop_celltype_mean_baseline": khop_celltype_mean_baseline,
        }

        for key, d in model_dicts.items():
            avg = macro_avg(d, metric_name)
            rows.append({"dataset": dataset, "model": label_map[key], "spearman": avg})

    # Assemble tidy dataframe
    df = pd.DataFrame(rows)

    # Keep order but drop excluded; make pretty names for x-axis
    dataset_order = ["aging_coronal", "aging_sagittal", "exercise", "reprogramming", "pilot", "androvic", "kukanja", "zeng"]
    pretty_order = [dataset_rename[d] for d in dataset_order]

    model_order = [
        "GNN (no GenePT)",
        "GNN (with GenePT)",
        "Global mean baseline",
    ]

    viridis = cm.get_cmap("viridis", 5)
    colors = [viridis(i) for i in range(3)]

    # Plot grouped bars
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(dataset_order))
    n_models = len(model_order)


    bar_width = 0.20
    spacing_factor = 1.0  # >1 means small gaps; guarantees no overlap
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * bar_width * spacing_factor

    for i, model in enumerate(model_order):
        y = (
            df[df["model"] == model]
            .set_index("dataset")
            .reindex(dataset_order)["spearman"]
            .values
        )
        ax.bar(x + offsets[i], y, width=bar_width, label=model, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(pretty_order, rotation=45, ha="right")  # 1) pretty names
    ax.set_ylabel(f"Spearman correlation")
    ax.set_xlabel("Dataset") 
    ax.set_title("Spearman correlation (macro average by cell type)")
    ax.legend(title="Model", frameon=False, loc="upper right")  # legend inside
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(
        "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/results/plots/macro_spearman.jpg",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()

    # Optional: print table with pretty dataset names
    df_print = df.copy()
    df_print["dataset"] = df_print["dataset"].map(dataset_rename)
    print(df_print.pivot(index="dataset", columns="model", values="spearman").round(3))


if __name__ == "__main__":
    plot_average_by_celltype()
