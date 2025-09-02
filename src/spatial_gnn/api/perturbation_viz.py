# %%
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np

anndata = "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/raw/aging_coronal.h5ad"

# %%
adata = sc.read_h5ad(anndata)

# %%
# Check available columns
print("Available obs columns:")
print(adata.obs.columns.tolist())

# %%
# Single plot of spatial coordinates colored by cell type
if 'celltype' in adata.obs.columns:
    unique_celltypes = adata.obs['celltype'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_celltypes)))

    plt.figure(figsize=(8, 6))
    for i, celltype in enumerate(unique_celltypes):
        mask = adata.obs['celltype'] == celltype
        plt.scatter(
            adata.obs.loc[mask, 'center_x'],
            adata.obs.loc[mask, 'center_y'],
            s=2,
            alpha=0.7,
            c=[colors[i]],
            label=celltype
        )
    
    plt.xlabel('Center X')
    plt.ylabel('Center Y')
    plt.title('Spatial Coordinates by Cell Type')
    plt.gca().set_aspect('equal')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()

# %%
# Summary statistics
print(f"Data shape: {adata.shape}")
print(f"X coordinate range: {adata.obs['center_x'].min():.2f} to {adata.obs['center_x'].max():.2f}")
print(f"Y coordinate range: {adata.obs['center_y'].min():.2f} to {adata.obs['center_y'].max():.2f}")

if 'celltype' in adata.obs.columns:
    print(f"\nCell types ({len(adata.obs['celltype'].unique())}):")
    print(adata.obs['celltype'].value_counts())

# %%
