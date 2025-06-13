import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans


def compute_ari_ami(cell_list, embeddings, labels):
    """
    Compute ARI and AMI for the given embeddings and labels.
    :param cell_list: List of cell ids
    :param embeddings: Dictionary of cell id to embedding
    :param labels: Dictionary of cell id to label
    """
    # Check for consistency
    if len(embeddings) != len(labels):
        raise ValueError("Number of embeddings and labels do not match.")

    # reoder embeddings and labels 
    new_embeddings = []
    new_labels = []
    for cell in cell_list:
        new_embeddings.append(embeddings[cell])
        new_labels.append(labels[cell])

    n_clusters = len(np.unique(new_labels))  # Number of unique labels
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(new_embeddings)
    
    # Compute ARI and AMI
    ari = adjusted_rand_score(new_labels, cluster_labels)
    ami = adjusted_mutual_info_score(new_labels, cluster_labels)

    return ari, ami
