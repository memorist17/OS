"""Clustering Evaluation Metrics."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def evaluate_clustering(
    X: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """
    Evaluate clustering quality using multiple metrics.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster labels (n_samples,)
        
    Returns:
        Dictionary of evaluation metrics
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    if n_clusters < 2:
        # Not enough clusters for meaningful evaluation
        return {
            "silhouette_score": -1.0,
            "davies_bouldin_score": float("inf"),
            "calinski_harabasz_score": 0.0,
            "n_clusters": n_clusters,
            "n_noise": int(np.sum(labels == -1)),
        }
    
    # Filter out noise points for some metrics
    valid_mask = labels != -1
    if np.sum(valid_mask) < 2:
        return {
            "silhouette_score": -1.0,
            "davies_bouldin_score": float("inf"),
            "calinski_harabasz_score": 0.0,
            "n_clusters": n_clusters,
            "n_noise": int(np.sum(labels == -1)),
        }
    
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    
    # Silhouette score (higher is better, range: -1 to 1)
    try:
        silhouette = silhouette_score(X_valid, labels_valid)
    except Exception:
        silhouette = -1.0
    
    # Davies-Bouldin score (lower is better)
    try:
        db_score = davies_bouldin_score(X_valid, labels_valid)
    except Exception:
        db_score = float("inf")
    
    # Calinski-Harabasz score (higher is better)
    try:
        ch_score = calinski_harabasz_score(X_valid, labels_valid)
    except Exception:
        ch_score = 0.0
    
    return {
        "silhouette_score": float(silhouette),
        "davies_bouldin_score": float(db_score),
        "calinski_harabasz_score": float(ch_score),
        "n_clusters": n_clusters,
        "n_noise": int(np.sum(labels == -1)),
        "n_valid_samples": int(np.sum(valid_mask)),
    }

