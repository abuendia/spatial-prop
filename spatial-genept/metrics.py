import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans


def compute_ari_ami(adata, embedding_key, label_key):
    """
    Compute ARI and AMI on embeddings from an anndata object.

    Parameters:
    - adata: AnnData object containing embeddings and labels.
    - embedding_key: Key to retrieve embeddings from adata.obsm.
    - label_key: Key to retrieve labels from adata.obs.

    Returns:
    - ari: Adjusted Rand Index
    - ami: Adjusted Mutual Information
    """
    # Retrieve embeddings and labels
    embeddings = adata.obsm[embedding_key]
    labels = adata.obs[label_key].values

    # Check for consistency
    if embeddings.shape[0] != len(labels):
        raise ValueError("Number of embeddings and labels do not match.")

    n_clusters = len(np.unique(labels))  # Number of unique labels
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Compute ARI and AMI
    ari = adjusted_rand_score(labels, cluster_labels)
    ami = adjusted_mutual_info_score(labels, cluster_labels)

    return ari, ami
