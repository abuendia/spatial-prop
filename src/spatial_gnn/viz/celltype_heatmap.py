# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# --- Optional: your normalization map (edit as needed) ---

datasets = [
    "aging_coronal", "aging_sagittal",
    "androvic", "exercise", "kukanja",
    "pilot",
    "reprogramming", "zeng"
]


normalization = {
    "Endothelium": "Endothelial",
    "Endo": "Endothelial",
    "DC": "Dendritic cell",
    "Schw": "Schwann cell",
    "VC": "Vascular cell",
    "Hep": "Hepatocyte",
    "Kuppfer": "Kupffer cell",
}

def normalize_spearman_keys(spearman_dict, normalization_map=None):
    """Return a new dict with cell-type keys normalized per dataset."""
    if normalization_map is None:
        normalization_map = {}
    out = {}
    for ds, ct2val in spearman_dict.items():
        merged = {}
        for ct, val in ct2val.items():
            nct = normalization_map.get(ct, ct)
            # If both original and normalized exist, average numeric values (ignore None)
            if nct in merged and merged[nct] is not None and val is not None:
                try:
                    merged[nct] = (merged[nct] + val) / 2
                except TypeError:
                    pass
            else:
                merged[nct] = val if val is not None else merged.get(nct, None)
        out[ds] = merged
    return out

def plot_spearman_heatmap(
    spearman_dict,
    normalization_map=None,
    datasets_to_exclude=None,
    title="",
    figsize=(12, 14),
    value_fmt="{:.2f}",
):
    """
    spearman_dict: dict like {dataset: {celltype: correlation_value}}
    normalization_map: dict mapping specific -> shared names (optional)
    datasets_to_exclude: list of dataset names to drop (optional)
    """
    # Normalize keys
    norm_dict = normalize_spearman_keys(spearman_dict, normalization_map)

    # Optionally exclude datasets
    if datasets_to_exclude:
        norm_dict = {ds: v for ds, v in norm_dict.items() if ds not in datasets_to_exclude}

    # Build DataFrame with rows=cell types, cols=datasets
    all_celltypes = sorted(set(ct for v in norm_dict.values() for ct in v.keys()))
    all_datasets  = sorted(norm_dict.keys())

    mat = pd.DataFrame(index=all_celltypes, columns=all_datasets, dtype=float)
    for ds, ct2val in norm_dict.items():
        for ct, val in ct2val.items():
            mat.at[ct, ds] = val

    # Drop rows that are all NaN
    mat = mat.dropna(how="all", axis=0)
    mat = mat.drop(index=[ct for ct in ["Other", "doublet"] if ct in mat.index], errors="ignore")

    # Prepare heatmap (0..1 scale, viridis)
    data = mat.values.astype(float)
    vmin, vmax = 0.0, 1.0
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Axes ticks/labels
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")  # <-- 45° rotation
    ax.set_yticklabels(mat.index)
    ax.set_xlabel("Datasets")
    ax.set_ylabel("Cell types")
    ax.set_title(title)

    # Grid lines
    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="lightgrey", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate: numbers for present cells; 'x' for NaN
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iat[i, j]
            if pd.isna(val):
                ax.text(j, i, "x", ha="center", va="center", fontsize=8, color="black")
            else:
                ax.text(j, i, value_fmt.format(val), ha="center", va="center", fontsize=7, color="black")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Spearman Correlation", rotation=90)

    fig.tight_layout()
    return fig, ax, mat

def get_spearman_dict(datasets):
    spearman_dict = {}
    for dataset in datasets:
        celltype_results = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/output/base_model/results/gnn/{dataset}_expression_2hop_2augment_expression_none/weightedl1_1en04_GenePT_all_genes/test_evaluation_stats_bycelltype.pkl"
        with open(celltype_results, "rb") as f:
            celltype_results = pickle.load(f)
        celltype_results = {k: v["Cell - Spearman (mean)"] for k, v in celltype_results.items()}
        spearman_dict[dataset] = celltype_results
    return spearman_dict


# %%

spearman_dict = get_spearman_dict(datasets)
plot_spearman_heatmap(spearman_dict)



import matplotlib 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.savefig(f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/src/spatial_gnn/viz/output/celltype_heatmap.pdf", format="pdf")
#plt.close()
# %%