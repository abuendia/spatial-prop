# %%
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.colors as colors

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

mouse_id = "57"
use_ids = [mouse_id]
perturbed_result = sc.read_h5ad(
    f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/perturbed_adata/aging_coronal_perturbed_mouse_{mouse_id}_result.h5ad"
)
mouse_id_to_age = perturbed_result.obs.groupby("mouse_id")["age"].mean().to_dict()
perturbed_result = perturbed_result[perturbed_result.obs["mouse_id"].isin(use_ids)]
sc.pp.normalize_total(perturbed_result, target_sum=perturbed_result.shape[1])

orig_expn = perturbed_result.X
unperturbed_expn = perturbed_result.layers["predicted_unperturbed"]
perturbed_expn = perturbed_result.layers["predicted_perturbed"]
tempered_expn = perturbed_result.layers["tempered"]
outdir = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/src/spatial_gnn/viz/output/{mouse_id}"
os.makedirs(outdir, exist_ok=True)


def _get_gene_set_sum_of_log1p(layer, adata, gene_list):
    """
    For each cell, compute sum_j log1p(expr[cell, gene_j]).
    """
    gene_idx = [adata.var_names.get_loc(g) for g in gene_list]

    if sp.issparse(layer):
        sub = layer[:, gene_idx].toarray()
    else:
        sub = layer[:, gene_idx]

    # sum of log1p over genes
    return np.log1p(sub).sum(axis=1).ravel()


def plot_four_panels_for_gene_set(
    adata,
    gene_list,
    orig_layer,
    unpert_layer,
    pert_layer,
    temp_layer,
    fig_title,
    save_path,
    point_size=0.1,
):
    """
    Plot 2x2 panels for the *sum of log1p* over gene_list:
      top-left: original
      top-right: tempered
      bottom-left: unperturbed predicted
      bottom-right: perturbed predicted
    And draw a colorbar on the right.
    """
    coordinates = adata.obsm["spatial"]

    # --- per-cell sum of log1p(expression) for each panel ---
    orig   = _get_gene_set_sum_of_log1p(orig_layer,   adata, gene_list)
    unpert = _get_gene_set_sum_of_log1p(unpert_layer, adata, gene_list)
    pert   = _get_gene_set_sum_of_log1p(pert_layer,   adata, gene_list)
    temp   = _get_gene_set_sum_of_log1p(temp_layer,   adata, gene_list)

    # collect all values for global sanity
    all_vals = np.concatenate([orig, unpert, pert, temp])
    all_vals = all_vals[np.isfinite(all_vals)]

    # --- color scaling strategy ---
    # 1) let original roughly define the "dark-to-mid" range
    vmin = 0
    vmax = 6

    print(f"vmin: {vmin:.3f}, vmax: {vmax:.3f}")

    # --- Figure layout ---
    fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)

    panel_info = [
        (axes[0, 0], orig,   "Original (sum log1p)"),
        (axes[0, 1], temp,   "Tempered (sum log1p)"),
        (axes[1, 0], unpert, "Unperturbed Predicted (sum log1p)"),
        (axes[1, 1], pert,   "Perturbed Predicted (sum log1p)"),
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
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        scatter_mappable = sc  # capture last mappable

    # --- Add a colorbar on the right ---
    cbar_ax = fig.add_axes([0.90, 0.15, 0.03, 0.70])
    cbar = fig.colorbar(scatter_mappable, cax=cbar_ax)
    cbar.set_label("sum log1p(expression)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(fig_title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.88, 0.98])  # leave room for colorbar
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


genes = ["Stat1", "Bst2", "Jak1", "Ifit1", "Cdkn1a", "Cdkn2a", "C4b", "H2-D1", "H2-K1"]

save_path = f"{outdir}/plot_four_panels_mouse_{mouse_id}_response_geneset_sum.pdf"
fig_title = f"Sum over response geneset\n(mouse_id {mouse_id}, age {mouse_id_to_age[mouse_id]})"
plot_four_panels_for_gene_set(
    perturbed_result,
    genes,            # the list defined above
    orig_expn,
    unperturbed_expn,
    perturbed_expn,
    tempered_expn,
    fig_title,
    save_path,
)

# %%
