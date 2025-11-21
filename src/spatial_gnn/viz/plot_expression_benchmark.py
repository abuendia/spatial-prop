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

dataset_order = ["AC", "AS", "EX", "RE", "EA", "MS", "AD", "AP", "HD"]

# Define a dictionary for mapping dataset names
dataset_mapping_short = {'aging_coronal': 'AC',
                    'aging_sagittal': 'AS',
                    'androvic': 'MS',
                    'exercise': 'EX',
                    'kukanja': 'EA',
                    'farah': 'HD', 
                    'pilot': 'AP',
                    'reprogramming': 'RE',
                    'zeng': 'AD',
                    'farah': 'HD',}

dataset_mapping_long = {'aging_coronal': 'MERFISH-AgingCoronal\n(AC)',
                    'aging_sagittal': 'MERFISH-AgingSagittal\n(AS)',
                    'androvic': 'MERFISH-LocalInjury\n(MS)',
                    'exercise': 'MERFISH-Exercise\n(EX)',
                    'kukanja': 'ISS-GlobalInjury\n(EA)',
                    'farah': 'MERFISH-HeartDevelopment\n(HD)',
                    'pilot': 'MERFISH-AgingPilot\n(AP)',
                    'reprogramming': 'MERFISH-Reprogramming\n(RE)',
                    'zeng': 'STARmap-Alzheimers\n(AD)',}

# %%
datasets = [
    "aging_coronal", 
    "aging_sagittal", 
    "exercise",
    "reprogramming", 
    "kukanja",
    "androvic",
    "zeng", 
    "pilot",
    "farah"
]
models = [
    "expression_only_khop2_no_genept_softmax_ct_center_pool"
]
baselines = [
    "global_mean",
    #"khop_mean", ### ONLY if consistently outperforms this baseline, otherwise we can just mention baseline in text)
]

rename_map = {
    "expression_only_khop2_no_genept_softmax_ct_center_pool": "SpatialProp GNN",
    #"khop_mean": "Baseline (K-hop Test Graph Mean)",
    "global_mean": "Baseline (Train Global Mean)"
}


def plot_metrics(metric):
    results_dict = defaultdict(dict)

    for dataset in datasets:
        for model in models:
            results_file = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/output/{model}/{dataset}_expression_2hop_2augment_expression_none/weightedl1_1en04/test_evaluation_stats_macro_micro.csv"
            results = pd.read_csv(results_file)
            results_dict[model][dataset] = results[results["Metric"] == f"Micro - {metric}"]["Value"].values[0]
        for baseline in baselines:
            results_file = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/expr_baselines/{dataset}_expression_100per_2hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/{baseline}/test_evaluation_stats_macro_micro.csv"
            results = pd.read_csv(results_file)
            results_dict[baseline][dataset] = results[results["Metric"] == f"Micro - {metric}"]["Value"].values[0]

    results_dict_nonzero = defaultdict(dict)
    for dataset in datasets:
        for model in models:
            results_file = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/output/{model}/{dataset}_expression_2hop_2augment_expression_none/weightedl1_1en04/test_evaluation_stats_macro_micro.csv"
            results = pd.read_csv(results_file)
            results_dict_nonzero[model][dataset] = results[results["Metric"] == f"Micro (Nonzero) - {metric}"]["Value"].values[0]
        for baseline in baselines:
            results_file = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/expr_baselines/{dataset}_expression_100per_2hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/{baseline}/test_evaluation_stats_macro_micro.csv"
            results = pd.read_csv(results_file)
            results_dict_nonzero[baseline][dataset] = results[results["Metric"] == f"Micro (Nonzero) - {metric}"]["Value"].values[0]

    df = pd.DataFrame(results_dict).T 
    df.index.name = "model"
    df = df.reset_index()
    long_df = df.melt(id_vars="model",
                    var_name="dataset",
                    value_name="score")
    long_df["model"] = long_df["model"].map(rename_map)

    # map to new names
    long_df["dataset_name"] = long_df["dataset"].map(dataset_mapping_short)
    
    long_df["dataset_name"] = pd.Categorical(
        long_df["dataset_name"],
        categories=dataset_order,
        ordered=True,
    )

    plt.figure(figsize=(12, 4))
    sns.barplot(
        data=long_df,
        x="dataset_name",
        y="score",
        hue="model",
        order=dataset_order,
        palette={
                    "SpatialProp GNN": '#AFE1AF',  # celadon
                    "Baseline (Train Global Mean)": '#A9A9A9',  # light gray
                    #"Baseline (K-hop Test Graph Mean)": '#808080'   # gray
                }
    )
    plt.legend(
        title="Model",
        bbox_to_anchor=(1.02, 0.7),   # (x, y) position; x > 1 shifts outside
        loc="upper left",           
        borderaxespad=0,
        fontsize=16
    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Dataset", fontsize=16)
    plt.ylabel(metric, fontsize=16)  
    plt.title(f"Center cell expression prediction ({metric}, all genes)", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/src/spatial_gnn/viz/output/center_cell_expression_prediction_{metric}_all_genes.pdf", bbox_inches='tight')
    plt.show()

    df = pd.DataFrame(results_dict_nonzero).T 
    df.index.name = "model"
    df = df.reset_index()
    long_df = df.melt(id_vars="model",
                    var_name="dataset",
                    value_name="score")
    long_df["model"] = long_df["model"].map(rename_map)
    
    # map to new names
    long_df["dataset_name"] = long_df["dataset"].map(dataset_mapping_short)
    long_df["dataset_name"] = pd.Categorical(
        long_df["dataset_name"],
        categories=dataset_order,
        ordered=True,
    )

    plt.figure(figsize=(12, 4))
    sns.barplot(
        data=long_df,
        x="dataset_name",
        y="score",
        hue="model",
        order=dataset_order,
        palette={
                    "SpatialProp GNN": '#AFE1AF',  # pastel purple
                    "Baseline (Train Global Mean)": '#A9A9A9',  # light gray
                    #"Baseline (K-hop Test Graph Mean)": '#808080'   # gray
                }
    )
    plt.legend(
        title="Model",
        bbox_to_anchor=(1.02, 0.7),   # (x, y) position; x > 1 shifts outside
        loc="upper left",           
        borderaxespad=0,
        fontsize=16
    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Dataset", fontsize=16)
    plt.ylabel(metric, fontsize=16)  
    plt.title(f"Center cell expression prediction ({metric}, nonzero genes)", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/src/spatial_gnn/viz/output/center_cell_expression_prediction_{metric}_nonzero_genes.pdf", bbox_inches='tight')
    plt.show()

# %%
plot_metrics("Pearson")
# %%
plot_metrics("Spearman")
# %%
plot_metrics("MAE")
# %%
