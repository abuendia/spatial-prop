import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.sparse as sp

from spatial_gnn.utils.metric_utils import get_gene_set_sum_of_log1p


def plot_loss_curves(save_dir):
    with open(os.path.join(save_dir, "training.pkl"), 'rb') as handle:
        b = pickle.load(handle)
    
    best_idx = np.argmin(b['test'])
    plt.figure(figsize=(4,4))
    plt.plot(b['epoch'],b['train'],label='Train',color='0.2',zorder=0)
    plt.plot(b['epoch'],b['test'],label='Test',color='green',zorder=1)
    plt.scatter(b['epoch'][best_idx],b['test'][best_idx],s=50,c='green',marker="D",zorder=2,label="Selected Model")
    plt.legend(fontsize=12)
    plt.ylabel("Weighted L1 Loss", fontsize=16)
    plt.xlabel("Training Epochs", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
    if save_dir is not None:   
        plt.savefig(os.path.join(save_dir, "loss_curves.pdf"), bbox_inches='tight')
    plt.close()


def plot_celltype_performance(save_dir):
    with open(os.path.join(save_dir, "test_evaluation_stats_bycelltype_nonzero_genes.pkl"), 'rb') as handle:
        ct_stats_dict = pickle.load(handle)

    columns_to_plot = ["Cell - Pearson (mean)", "Cell - Spearman (mean)"]
    metric_col = []
    ct_col = []
    val_col = []

    for col in columns_to_plot:
        for ct in ct_stats_dict.keys():
            val = ct_stats_dict[ct][col]
            
            metric_col.append(col)
            ct_col.append(ct)
            val_col.append(val)

    plot_df = pd.DataFrame(np.vstack((metric_col, ct_col, val_col)).T, columns=["Metric", "Cell type", "Value"])
    plot_df["Value"] = plot_df["Value"].astype(float)

    # plot
    fig, ax = plt.subplots(figsize=(12,4))
    sns.barplot(plot_df, x="Cell type", y="Value", hue="Metric", palette="Reds", ax=ax)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.7))
    plt.title("Performance by cell type (nonzero genes)", fontsize=14)
    plt.xticks(rotation=30, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Cell type", fontsize=14)
    plt.ylabel("Metric Value", fontsize=14)
    plt.setp(ax.get_legend().get_texts(), fontsize='14')
    plt.setp(ax.get_legend().get_title(), fontsize='16')
    plt.tight_layout()
    plt.show()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "celltype_performance_nonzero_genes.pdf"), bbox_inches='tight')
    plt.close()


def plot_gene_in_section(adata, gene, layer, title, save_dir):
    coordinates = adata.obsm["spatial"]
    g_idx = np.where(adata.var_names == gene)[0][0]

    if sp.issparse(layer):
        pert = layer[:, g_idx].toarray().ravel()
    else:
        pert = layer[:, g_idx]

    expn_log = np.log1p(pert)
    fig_pert, ax_pert = plt.subplots(figsize=(6, 5))

    sc = ax_pert.scatter(
        coordinates[:, 0], coordinates[:, 1],
        c=expn_log,
        cmap="afmhot",
        s=2,
        vmin=0, vmax=1,
        rasterized=True
    )
    ax_pert.axis("off")
    ax_pert.set_title(title, fontsize=14)

    cbar_ax = fig_pert.add_axes([0.90, 0.15, 0.03, 0.7])
    cbar = fig_pert.colorbar(sc, cax=cbar_ax)
    cbar.set_label("Expression (log scale)", fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    title_fig = title.replace(" ", "_")
    plt.show()
    
    if save_dir is not None:
        fig_pert.savefig(
            f"{save_dir}/{gene}_{title_fig}.pdf",
            bbox_inches="tight",
            dpi=300
        )
    plt.close()


def plot_celltypes_in_section(adata, ct_key="celltype", s=0.5, figsize=(6,6), save_path=None):
    """
    Plot all cells in the section colored by their cell type label.
    
    Parameters
    ----------
    adata : AnnData
        Contains adata.obsm["spatial"] and adata.obs[ct_key].
    ct_key : str
        Column in adata.obs with cell type labels.
    s : float
        Scatterpoint size.
    figsize : tuple
        Figure size.
    save_path : str or None
        If provided, save the figure to this path.
    """

    coords = adata.obsm["spatial"]
    celltypes = adata.obs[ct_key].astype(str).values

    # unique categories
    unique_ct = np.unique(celltypes)

    # color palette
    cmap = plt.get_cmap("tab20")
    colors = {ct: cmap(i % 20) for i, ct in enumerate(unique_ct)}

    # map each cell to a color
    cell_colors = [colors[ct] for ct in celltypes]

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=cell_colors,
        s=s,
        rasterized=True
    )
    ax.axis("off")

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=ct, markerfacecolor=colors[ct], markersize=6) for ct in unique_ct]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6)
    plt.show()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_propagation_results_for_gene_set(
    adata,
    gene_list,
    orig_layer,
    pert_layer,
    temp_layer,
    fig_title,
    save_path,
    vmin=0,
    vmax=6,
    point_size=0.1,
):
    """
    Plot 3 side-by-side panels for the *sum of log1p* over gene_list:
      left:   original
      center: perturbed predicted
      right:  tempered predicted
    And draw a colorbar on the right.
    """
    coordinates = adata.obsm["spatial"]
    orig = get_gene_set_sum_of_log1p(orig_layer, adata, gene_list)
    pert = get_gene_set_sum_of_log1p(pert_layer, adata, gene_list)
    temp = get_gene_set_sum_of_log1p(temp_layer, adata, gene_list)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharex=True, sharey=True)

    panel_info = [
        (axes[0], orig, "Original (sum log1p)"),
        (axes[1], pert, "Perturbed Predicted (sum log1p)"),
        (axes[2], temp, "Tempered (sum log1p)"),
    ]

    scatter_mappable = None
    for ax, vals, title in panel_info:
        sc = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=vals,
            cmap="afmhot",
            s=point_size,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        scatter_mappable = sc

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(scatter_mappable, cax=cbar_ax)
    cbar.set_label("sum log1p(expression)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(fig_title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.90, 0.95])
    plt.show()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
