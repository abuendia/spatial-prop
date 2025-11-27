# %%
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# %%
perturbed_result = sc.read_h5ad("/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/perturbed_adata/perturbed_adata_result_debug.h5ad")
sc.pp.normalize_total(perturbed_result, target_sum=perturbed_result.shape[1])

# %%
cell_ids = np.unique(np.where(perturbed_result.layers['predicted_perturbed'] > 0)[0])
cell_types = np.unique(perturbed_result.obs['celltype'][cell_ids])
mouse_ids = np.unique(perturbed_result.obs['mouse_id'][cell_ids])

print(f"Cell types: {cell_types}")
print(f"Mouse ids: {mouse_ids}")
print(f"Number of cells: {len(cell_ids)}")

# %%
gene = "Stat1"  # gene of interest
adata = perturbed_result[cell_ids]
assert gene in adata.var_names, f"{gene} not found in var_names"

g_idx = np.where(adata.var_names == gene)[0][0]

X = adata.X
if sp.issparse(X):
    orig = X[:, g_idx].toarray().ravel()
else:
    orig = X[:, g_idx]

layer = adata.layers["predicted_perturbed"]
if sp.issparse(layer):
    pert = layer[:, g_idx].toarray().ravel()
else:
    pert = layer[:, g_idx]

orig_log = np.log1p(orig)
pert_log = np.log1p(pert)

vmin = min(orig_log.min(), pert_log.min())
vmax = max(orig_log.max(), pert_log.max())
# %%
coordinates = adata.obsm["spatial"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Reserve extra space on the right for the colorbar
plt.subplots_adjust(right=0.88, wspace=0.40) 

# --- ORIGINAL ---
sc1 = axes[0].scatter(
    coordinates[:, 0], coordinates[:, 1],
    c=orig_log,
    cmap="magma",
    s=2,
    vmin=vmin, vmax=1
)
axes[0].axis("off")
axes[0].set_title(f"{gene.upper()} original expression", fontsize=18)

# --- PERTURBED ---
sc2 = axes[1].scatter(
    coordinates[:, 0], coordinates[:, 1],
    c=pert_log,
    cmap="magma",
    s=1.75,
    vmin=vmin, vmax=1
)
axes[1].axis("off")
axes[1].set_title(f"{gene.upper()} perturbed expression", fontsize=18)

cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
cbar = fig.colorbar(sc1, cax=cbar_ax)
cbar.set_label("Expression (log scale)", fontsize=16)
cbar.ax.tick_params(labelsize=12)


plt.savefig(f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/src/spatial_gnn/viz/output/{gene}_perturbed_coronal.pdf", bbox_inches="tight")
# %%

# --- SAVE ORIGINAL PANEL SEPARATELY ---
fig_orig, ax_orig = plt.subplots(figsize=(6, 5))

sc = ax_orig.scatter(
    coordinates[:, 0], coordinates[:, 1],
    c=orig_log,
    cmap="magma",
    s=2,
    vmin=vmin, vmax=1
)
ax_orig.axis("off")
ax_orig.set_title(f"{gene.upper()} original expression", fontsize=18)

fig_orig.savefig(
    f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/src/spatial_gnn/viz/output/{gene}_original_only.pdf",
    bbox_inches="tight"
)


# --- SAVE PERTURBED PANEL SEPARATELY ---
fig_pert, ax_pert = plt.subplots(figsize=(6, 5))

sc = ax_pert.scatter(
    coordinates[:, 0], coordinates[:, 1],
    c=pert_log,
    cmap="magma",
    s=2,
    vmin=vmin, vmax=1
)
ax_pert.axis("off")
ax_pert.set_title(f"{gene.upper()} perturbed expression", fontsize=18)

# -----------------------------------------
# Add colorbar manually without shrinking plot
# -----------------------------------------
# (left, bottom, width, height) in figure fraction units
cbar_ax = fig_pert.add_axes([0.90, 0.15, 0.03, 0.7])
cbar = fig_pert.colorbar(sc, cax=cbar_ax)
cbar.set_label("Expression (log scale)", fontsize=14)
cbar.ax.tick_params(labelsize=10)

fig_pert.savefig(
    f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/src/spatial_gnn/viz/output/{gene}_perturbed_only.pdf",
    bbox_inches="tight"
)

# %%
