# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle

# %%

results_file = "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/output/base_model/results/gnn/aging_coronal_expression_2hop_2augment_expression_none/weightedl1_1en04_GenePT_all_genes/test_evaluation_summary_micro_macro.csv"
results = pd.read_csv(results_file)
results = results[results["Metric"] == "MAE"]["Micro_Average"].values[0]

# %%
datasets = ["aging_coronal", "aging_sagittal", "exercise", "reprogramming", "kukanja", "androvic", "zeng", "pilot"]

metric = "Pearson"
base_model_results = {}
train_mean_results = {}

for dataset in datasets:
    results_file = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/output/base_model/results/gnn/{dataset}_expression_2hop_2augment_expression_none/weightedl1_1en04_GenePT_all_genes/test_evaluation_summary_micro_macro.csv"
    results = pd.read_csv(results_file)
    results = results[results["Metric"] == metric]["Micro_Average"].values[0]
    base_model_results[dataset] = results

    results_file = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/baselines/{dataset}_expression_100per_2hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/global_mean/test_evaluation_stats_macro_micro.csv"
    results = pd.read_csv(results_file)
    results = results[results["Metric"] == f"Micro - {metric}"]["Value"].values[0]
    train_mean_results[dataset] = results
    
# %%
# Convert to DataFrame for seaborn
df = pd.DataFrame({
    "dataset": list(base_model_results.keys()) * 2,
    metric: list(base_model_results.values()) + list(train_mean_results.values()),
    "model": ["GNN"] * len(base_model_results) + ["Baseline"] * len(train_mean_results)
})

# Define custom color palette: Baseline = grey, GNN = red
palette = {"Baseline": "grey", "GNN": "blue"}

# Plot
import matplotlib 
import seaborn as sns

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.figure(figsize=(12, 5))
sns.barplot(data=df, x="dataset", y=metric, hue="model", palette=palette)

plt.title("Center cell expression prediction", fontsize=14)
plt.xlabel("Dataset", fontsize=12)
plt.ylabel(metric, fontsize=12)
plt.xticks(rotation=30, ha="right")
plt.legend(title="")
plt.tight_layout()
plt.savefig(f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/src/spatial_gnn/viz/output/center_cell_expression_prediction_{metric}.pdf", format="pdf")
# %%
