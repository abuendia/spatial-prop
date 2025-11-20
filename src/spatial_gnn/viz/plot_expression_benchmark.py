# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib 
import seaborn as sns

from collections import defaultdict

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# %%
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
baselines = [
    "khop_mean",
    "center_celltype_global_mean",
    "global_mean"
]

rename_map = {
    "expression_with_celltype_decoupled_no_genept_oracle_ct_ablate_gene_expression_one_hot_ct": "GNN (Ablate GEX)",
    "expression_with_celltype_decoupled_no_genept_oracle_ct_softmax_ct": "GNN (Oracle Softmax CT)",
    "expression_with_celltype_decoupled_no_genept_one_hot_ct": "GNN (Predict One-hot CT)",
    "expression_with_celltype_decoupled_no_genept_softmax_ct": "GNN (Predict Softmax CT)",
    "expression_only_no_genept_softmax_ct": "GNN (Predict GEX Only)",
    "khop_mean": "Baseline (K-hop Test Graph Mean)",
    "center_celltype_global_mean": "Baseline (Center Celltype Train Global Mean)",
    "global_mean": "Baseline (Train Global Mean)"
}


def plot_metrics(metric):
    results_dict = defaultdict(dict)

    for dataset in datasets:
        for model in models:
            results_file = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/output/final_exps/{model}/{dataset}_expression_2hop_2augment_expression_none/weightedl1_1en04/test_evaluation_stats_micro.csv"
            results = pd.read_csv(results_file)
            results_dict[model][dataset] = results[results["Metric"] == metric]["Value"].values[0]
        for baseline in baselines:
            results_file = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/baselines/{dataset}_expression_100per_2hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/{baseline}/test_evaluation_stats_macro_micro.csv"
            results = pd.read_csv(results_file)
            results_dict[baseline][dataset] = results[results["Metric"] == f"Micro - {metric}"]["Value"].values[0]

    results_dict_nonzero = defaultdict(dict)
    for dataset in datasets:
        for model in models:
            results_file = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/output/final_exps/{model}/{dataset}_expression_2hop_2augment_expression_none/weightedl1_1en04/test_evaluation_stats_micro.csv"
            results = pd.read_csv(results_file)
            results_dict_nonzero[model][dataset] = results[results["Metric"] == f"{metric} (Nonzero)"]["Value"].values[0]
        for baseline in baselines:
            results_file = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/baselines/{dataset}_expression_100per_2hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/{baseline}/test_evaluation_stats_macro_micro.csv"
            results = pd.read_csv(results_file)
            results_dict_nonzero[baseline][dataset] = results[results["Metric"] == f"Micro (Nonzero) - {metric}"]["Value"].values[0]

    df = pd.DataFrame(results_dict).T 
    df.index.name = "model"
    df = df.reset_index()
    long_df = df.melt(id_vars="model",
                    var_name="dataset",
                    value_name="score")
    long_df["model"] = long_df["model"].map(rename_map)

    plt.figure(figsize=(18, 6))
    sns.barplot(
        data=long_df,
        x="dataset",
        y="score",
        hue="model",
        palette="tab10"
    )
    plt.legend(
        title="Model",
        bbox_to_anchor=(1.02, 0.7),   # (x, y) position; x > 1 shifts outside
        loc="upper left",           
        borderaxespad=0
    )
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Dataset")
    plt.ylabel(metric)  
    plt.title(f"Center cell expression prediction ({metric}, all genes)")
    plt.tight_layout()
    plt.savefig(f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/src/spatial_gnn/viz/output/center_cell_expression_prediction_{metric}_all_genes.png", dpi=300)
    plt.show()

    df = pd.DataFrame(results_dict_nonzero).T 
    df.index.name = "model"
    df = df.reset_index()
    long_df = df.melt(id_vars="model",
                    var_name="dataset",
                    value_name="score")
    long_df["model"] = long_df["model"].map(rename_map)
    
    plt.figure(figsize=(18, 6))
    sns.barplot(
        data=long_df,
        x="dataset",
        y="score",
        hue="model",
        palette="tab10"
    )
    plt.legend(
        title="Model",
        bbox_to_anchor=(1.02, 0.7),   # (x, y) position; x > 1 shifts outside
        loc="upper left",           
        borderaxespad=0
    )
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Dataset")
    plt.ylabel(metric)  
    plt.title(f"Center cell expression prediction ({metric}, nonzero genes)")
    plt.tight_layout()
    plt.savefig(f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/src/spatial_gnn/viz/output/center_cell_expression_prediction_{metric}_nonzero_genes.png", dpi=300)
    plt.show()

# %%
plot_metrics("Pearson")
# %%
plot_metrics("Spearman")
# %%
plot_metrics("MAE")
# %%
