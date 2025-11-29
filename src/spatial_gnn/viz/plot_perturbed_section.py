# %%
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

use_ids = ["11"]

perturbed_result = sc.read_h5ad("/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/perturbed_adata/perturbed_adata_resut_final.h5ad")
perturbed_result = perturbed_result[perturbed_result.obs['mouse_id'].isin(use_ids)]
sc.pp.normalize_total(perturbed_result, target_sum=perturbed_result.shape[1])
# %%

def plot_slice(adata, gene, layer, title):
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
        f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/src/spatial_gnn/viz/output/{gene}_{title_fig}.pdf",
        bbox_inches="tight",
        dpi=300
    )


gene = "Stat1"  
orig_expn = perturbed_result.X 
unperturbed_expn = perturbed_result.layers["predicted_unperturbed"]
perturbed_expn = perturbed_result.layers["predicted_perturbed"]
tempered_expn = perturbed_result.layers["tempered"]

plot_slice(perturbed_result, gene, orig_expn, "Original expression")
plot_slice(perturbed_result, gene, unperturbed_expn, "Predicted expression (unperturbed)")
plot_slice(perturbed_result, gene, perturbed_expn, "Predicted expression (perturbed)")
plot_slice(perturbed_result, gene, tempered_expn, "Predicted expression (tempered)")

# %%

# check how many cells are different between perturbed and unperturbed with a tolerance of 1e-4
print(np.sum(np.abs(perturbed_expn - unperturbed_expn) > 1e-4))

# find genes where the difference is greater between perturbed and unperturbed

# %%
