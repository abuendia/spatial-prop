# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Toggle: choose which family to plot (2 or 3) ---
HOP = 2  # change to 3 for the 3-hop variants

# --- Config ---
datasets = [
    "aging_coronal_expression", "aging_sagittal_expression",
    "androvic_expression", "exercise_expression", "kukanja_expression",
    "lohoff_expression", "pilot_expression",
    "reprogramming_expression", "zeng_expression"
]

if HOP == 2:
    MODEL_LABELS = [
        "SpatialProp",
        "SpatialProp (GenePT)",
        "2-hop mean",
    ]
else:
    MODEL_LABELS = [
        "3-hop SpatialProp",
        "3-hop SpatialProp (GenePT)",
        "3-hop mean",
    ]

def _read_score(csv_path):
    """
    Reads a test_evaluation_summary_micro_macro.csv and returns macro_spearman (float).
    Handles both MultiIndex and single-level columns.
    """
    df = pd.read_csv(csv_path, index_col=0)
    try:
        macro = df["Macro_Average"]["Spearman"]
    except Exception:
        try:
            macro = df.loc["Spearman", "Macro_Average"]
        except Exception:
            macro = df.get("Macro_Spearman", np.nan)
    return float(macro)

def build_paths(dataset, hop):
    """
    Returns a dict {label: path} for the three models (given hop) for a dataset.
    """
    if hop == 2:
        base_no_genept = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/base/results/gnn/{dataset}_2hop_2augment_expression_none/weightedl1_1en04_GenePT_all_genes/test_evaluation_summary_micro_macro.csv"
        xattn_genept   = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/xattn/results/gnn/{dataset}_2hop_2augment_expression_none/weightedl1_1en04_GenePT_all_genes/test_evaluation_summary_micro_macro.csv"
        mean_path      = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/baselines/{dataset}_100per_2hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/khop_mean/test_evaluation_summary_micro_macro.csv"
        return {
            "SpatialProp": base_no_genept,
            "SpatialProp (GenePT)": xattn_genept,
            "2-hop mean": mean_path,
        }
    else:
        base_no_genept = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/base/results/gnn/{dataset}_3hop_2augment_expression_none/weightedl1_1en04_GenePT_all_genes/test_evaluation_summary_micro_macro.csv"
        xattn_genept   = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/xattn/results/gnn/{dataset}_3hop_2augment_expression_none/weightedl1_1en04_GenePT_all_genes/test_evaluation_summary_micro_macro.csv"
        mean_path      = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/baselines/{dataset}_100per_3hop_2C0aug_200delaunay_expressionFeat_all_NoneInject/khop_mean/test_evaluation_summary_micro_macro.csv"
        return {
            "3-hop SpatialProp": base_no_genept,
            "3-hop SpatialProp (GenePT)": xattn_genept,
            "3-hop mean": mean_path,
        }

# --- Load macro-only scores ---
results_macro = {ds: [np.nan]*len(MODEL_LABELS) for ds in datasets}

for ds in datasets:
    paths = build_paths(ds, HOP)
    for i, label in enumerate(MODEL_LABELS):
        p = paths[label]
        if not os.path.exists(p):
            print(f"[WARN] Missing file for {ds} - {label}: {p}")
            macro = np.nan
        else:
            try:
                macro = _read_score(p)
            except Exception as e:
                print(f"[WARN] Failed to parse {ds} - {label}: {p}\n  -> {e}")
                macro = np.nan
        results_macro[ds][i] = macro

# --- Single Plot: Macro Average Only ---
dataset_names = datasets
x = np.arange(len(dataset_names))
num_bars = len(MODEL_LABELS)  # 3

# Thinner bars
group_width = 0.8
bar_width = group_width / num_bars * 0.9

# Viridis colors at 1/4, 2/4, 3/4 of the colormap range
cmap = plt.cm.get_cmap('viridis')
fractions = [1/4, 2/4, 3/4]
colors = [cmap(f) for f in fractions]

fig, ax = plt.subplots(figsize=(max(14, len(dataset_names)*0.9), 6))

for i in range(num_bars):
    offsets = (i - (num_bars-1)/2) * bar_width
    heights = [results_macro[ds][i] for ds in dataset_names]
    ax.bar(x + offsets, heights, bar_width, label=MODEL_LABELS[i], color=colors[i], alpha=0.95)

title_hop = f"{HOP}-hop"
ax.set_title(f"Center cell expression prediction (macro-averaged by cell type)")
ax.set_xlabel("Dataset")
ax.set_ylabel("Spearman Correlation")
ax.set_xticks(x)
ax.set_xticklabels(dataset_names, rotation=45, ha='right')
ax.grid(True, alpha=0.3)

# Legend in bottom left but shifted slightly to the right
legend = ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.001, 1),  # shifted right compared to (0.02, 0.02)
    ncol=1,
    frameon=True,
    fontsize=10,
)
legend.get_frame().set_linewidth(0.8)
legend.get_frame().set_edgecolor('gray')

plt.tight_layout()
plt.savefig(f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/results/plots/macro_spearman.jpg", bbox_inches="tight", dpi=300)
# %%
