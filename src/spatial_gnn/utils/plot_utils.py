import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.sparse as sp


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
    plt.savefig(os.path.join(save_dir, "loss_curves.pdf"), bbox_inches='tight')
    plt.close()


def plot_celltype_performance(save_dir):
    with open(os.path.join(save_dir, "test_evaluation_stats_bycelltype_nonzero_genes.pkl"), 'rb') as handle:
        ct_stats_dict = pickle.load(handle)

    columns_to_plot = ["Cell - Pearson (mean)", "Cell - Spearman (mean)", "Cell - R2 (mean)"]
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

    # save figure
    title_fig = title.replace(" ", "_")
    fig_pert.savefig(
        f"{save_dir}/{gene}_{title_fig}.pdf",
        bbox_inches="tight",
        dpi=300
    )
