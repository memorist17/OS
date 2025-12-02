"""Clustering Analysis for Urban Structure Indicators."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from tqdm import tqdm

from .clustering_preprocessing import ClusteringPreprocessor, prepare_clustering_data
from .feature_extraction import FeatureExtractor


@dataclass
class ClusteringAnalyzer:
    """
    Perform clustering analysis on preprocessed features.
    
    Supports:
    - K-means
    - DBSCAN
    - Hierarchical clustering
    """
    
    method: str = "kmeans"  # "kmeans", "dbscan", "hierarchical"
    n_clusters: int | None = None  # For K-means and hierarchical
    eps: float = 0.5  # For DBSCAN
    min_samples: int = 3  # For DBSCAN
    linkage: str = "ward"  # For hierarchical
    random_state: int = 42
    
    def __post_init__(self):
        """Validate configuration."""
        if self.method not in ["kmeans", "dbscan", "hierarchical"]:
            raise ValueError(
                f"method must be one of ['kmeans', 'dbscan', 'hierarchical'], "
                f"got {self.method}"
            )
        
        if self.method in ["kmeans", "hierarchical"] and self.n_clusters is None:
            raise ValueError(
                f"n_clusters must be specified for {self.method}"
            )
    
    def fit_predict(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Fit clustering model and predict labels.
        
        Args:
            X: Preprocessed feature matrix (n_samples, n_features)
            
        Returns:
            Tuple of:
            - labels: Cluster labels (-1 for noise in DBSCAN)
            - metadata: Clustering metadata
        """
        metadata = {}
        
        if self.method == "kmeans":
            clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
            labels = clusterer.fit_predict(X)
            metadata["clusterer"] = clusterer
            metadata["inertia"] = clusterer.inertia_
            metadata["n_clusters"] = len(np.unique(labels))
            metadata["cluster_centers"] = clusterer.cluster_centers_
        
        elif self.method == "dbscan":
            clusterer = DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
            )
            labels = clusterer.fit_predict(X)
            metadata["clusterer"] = clusterer
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            metadata["n_clusters"] = n_clusters
            metadata["n_noise"] = n_noise
            metadata["noise_ratio"] = n_noise / len(labels) if len(labels) > 0 else 0.0
        
        elif self.method == "hierarchical":
            clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage=self.linkage,
            )
            labels = clusterer.fit_predict(X)
            metadata["clusterer"] = clusterer
            metadata["n_clusters"] = len(np.unique(labels))
        
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        metadata["labels"] = labels
        metadata["method"] = self.method
        
        # Compute cluster statistics
        unique_labels = np.unique(labels)
        cluster_stats = {}
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise points in DBSCAN
            mask = labels == label
            cluster_stats[int(label)] = {
                "size": int(np.sum(mask)),
                "ratio": float(np.sum(mask) / len(labels)),
            }
        metadata["cluster_stats"] = cluster_stats
        
        return labels, metadata


def analyze_clusters(
    output_dirs: list,
    preprocessor_config: dict[str, Any] | None = None,
    clustering_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """
    Complete clustering pipeline: extract features, preprocess, cluster.
    
    Args:
        output_dirs: List of output directory paths
        preprocessor_config: Configuration for ClusteringPreprocessor
        clustering_config: Configuration for ClusteringAnalyzer
        
    Returns:
        Tuple of:
        - results_df: DataFrame with features, labels, and metadata
        - X_processed: Preprocessed feature matrix
        - metadata: Complete pipeline metadata
    """
    from pathlib import Path
    
    # Convert to Path objects
    output_dirs = [Path(d) for d in output_dirs]
    
    # Setup preprocessor
    if preprocessor_config is None:
        preprocessor_config = {}
    preprocessor = ClusteringPreprocessor(**preprocessor_config)
    
    # Setup feature extractor
    feature_extractor = FeatureExtractor()
    
    # Prepare data
    features_df, X_processed, prep_metadata = prepare_clustering_data(
        output_dirs,
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
    )
    
    # Setup clusterer
    if clustering_config is None:
        clustering_config = {}
    
    # Auto-select number of clusters if not specified
    if "n_clusters" not in clustering_config:
        # Simple heuristic: sqrt(n_samples / 2)
        n_samples = X_processed.shape[0]
        n_clusters = max(2, int(np.sqrt(n_samples / 2)))
        clustering_config["n_clusters"] = n_clusters
    
    clustering_analyzer = ClusteringAnalyzer(**clustering_config)
    
    # Perform clustering
    print("Performing clustering...")
    labels, cluster_metadata = clustering_analyzer.fit_predict(X_processed)
    
    # Combine results
    results_df = features_df.copy()
    results_df["cluster_label"] = labels
    results_df["sample_id"] = prep_metadata["sample_ids"]
    
    # Combine metadata
    complete_metadata = {
        "preprocessing": prep_metadata,
        "clustering": cluster_metadata,
        "n_samples": len(output_dirs),
        "n_features_original": prep_metadata["n_features_original"],
        "n_features_processed": X_processed.shape[1],
    }
    
    return results_df, X_processed, complete_metadata

