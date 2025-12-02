"""Clustering Optimization: Elbow Method and Parameter Tuning."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from tqdm import tqdm

from .clustering_evaluation import evaluate_clustering


@dataclass
class ClusteringOptimizer:
    """
    Optimize clustering parameters using elbow method and grid search.
    
    Supports:
    - Elbow method for K-means and Hierarchical clustering
    - Parameter grid search for DBSCAN
    - Multiple evaluation metrics
    """
    
    min_clusters: int = 2
    max_clusters: int = 10
    n_cluster_range: list[int] | None = None  # Custom range if provided
    random_state: int = 42
    
    # DBSCAN parameter ranges
    eps_range: list[float] | None = None  # Default: [0.1, 0.3, 0.5, 0.7, 1.0]
    min_samples_range: list[int] | None = None  # Default: [2, 3, 4, 5]
    
    def __post_init__(self):
        """Initialize default parameter ranges."""
        if self.n_cluster_range is None:
            self.n_cluster_range = list(range(self.min_clusters, self.max_clusters + 1))
        
        if self.eps_range is None:
            self.eps_range = [0.1, 0.3, 0.5, 0.7, 1.0]
        
        if self.min_samples_range is None:
            self.min_samples_range = [2, 3, 4, 5]
    
    def elbow_method_kmeans(
        self, X: np.ndarray, metric: str = "inertia"
    ) -> tuple[dict[int, dict[str, Any]], int | None]:
        """
        Find optimal number of clusters using elbow method for K-means.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            metric: Metric to use ("inertia", "silhouette", "davies_bouldin")
            
        Returns:
            Tuple of:
            - results: Dictionary mapping n_clusters to evaluation metrics
            - optimal_k: Optimal number of clusters (None if not found)
        """
        results = {}
        inertias = []
        silhouette_scores = []
        db_scores = []
        
        for k in tqdm(self.n_cluster_range, desc="Elbow method (K-means)"):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # Evaluate clustering
            evaluation = evaluate_clustering(X, labels)
            
            results[k] = {
                "inertia": kmeans.inertia_,
                "silhouette_score": evaluation["silhouette_score"],
                "davies_bouldin_score": evaluation["davies_bouldin_score"],
                "calinski_harabasz_score": evaluation["calinski_harabasz_score"],
                "n_clusters": evaluation["n_clusters"],
            }
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(evaluation["silhouette_score"])
            db_scores.append(evaluation["davies_bouldin_score"])
        
        # Find elbow point
        optimal_k = self._find_elbow_point(
            self.n_cluster_range,
            inertias if metric == "inertia" else silhouette_scores,
            metric=metric,
        )
        
        return results, optimal_k
    
    def optimize_dbscan(
        self, X: np.ndarray
    ) -> tuple[dict[tuple[float, int], dict[str, Any]], tuple[float, int] | None]:
        """
        Find optimal DBSCAN parameters using grid search.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Tuple of:
            - results: Dictionary mapping (eps, min_samples) to evaluation metrics
            - optimal_params: Optimal (eps, min_samples) tuple (None if not found)
        """
        results = {}
        best_score = -1
        optimal_params = None
        
        total_combinations = len(self.eps_range) * len(self.min_samples_range)
        
        with tqdm(total=total_combinations, desc="DBSCAN optimization") as pbar:
            for eps in self.eps_range:
                for min_samples in self.min_samples_range:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(X)
                    
                    # Evaluate clustering
                    evaluation = evaluate_clustering(X, labels)
                    
                    params = (eps, min_samples)
                    results[params] = {
                        "silhouette_score": evaluation["silhouette_score"],
                        "davies_bouldin_score": evaluation["davies_bouldin_score"],
                        "calinski_harabasz_score": evaluation["calinski_harabasz_score"],
                        "n_clusters": evaluation["n_clusters"],
                        "n_noise": evaluation["n_noise"],
                        "noise_ratio": evaluation.get("n_noise", 0) / len(labels) if len(labels) > 0 else 0.0,
                    }
                    
                    # Find best based on silhouette score (if valid clusters found)
                    if evaluation["n_clusters"] >= 2:
                        score = evaluation["silhouette_score"]
                        if score > best_score:
                            best_score = score
                            optimal_params = params
                    
                    pbar.update(1)
        
        return results, optimal_params
    
    def compare_cluster_numbers(
        self, X: np.ndarray, method: str = "kmeans"
    ) -> pd.DataFrame:
        """
        Compare clustering quality across different numbers of clusters.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            method: Clustering method ("kmeans" or "hierarchical")
            
        Returns:
            DataFrame with evaluation metrics for each cluster number
        """
        results = []
        
        for k in tqdm(self.n_cluster_range, desc=f"Comparing {method} cluster numbers"):
            if method == "kmeans":
                clusterer = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            else:
                from sklearn.cluster import AgglomerativeClustering
                clusterer = AgglomerativeClustering(n_clusters=k, linkage="ward")
            
            labels = clusterer.fit_predict(X)
            evaluation = evaluate_clustering(X, labels)
            
            result = {
                "n_clusters": k,
                "silhouette_score": evaluation["silhouette_score"],
                "davies_bouldin_score": evaluation["davies_bouldin_score"],
                "calinski_harabasz_score": evaluation["calinski_harabasz_score"],
            }
            
            if method == "kmeans":
                result["inertia"] = clusterer.inertia_
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _find_elbow_point(
        self, k_values: list[int], scores: list[float], metric: str = "inertia"
    ) -> int | None:
        """
        Find elbow point using rate of change method.
        
        Args:
            k_values: List of cluster numbers
            scores: List of scores (inertia or silhouette)
            metric: "inertia" (decreasing) or "silhouette" (increasing)
            
        Returns:
            Optimal k value or None if not found
        """
        if len(scores) < 3:
            return None
        
        # Calculate rate of change
        rates = []
        for i in range(1, len(scores)):
            if metric == "inertia":
                # For inertia, we want to find where decrease slows down
                rate = (scores[i-1] - scores[i]) / scores[i-1] if scores[i-1] > 0 else 0
            else:
                # For silhouette, we want to find where increase slows down
                rate = (scores[i] - scores[i-1]) / abs(scores[i-1]) if scores[i-1] != 0 else 0
            rates.append(rate)
        
        # Find elbow: where rate of change decreases significantly
        if len(rates) < 2:
            return None
        
        # Find maximum rate of change (elbow point)
        if metric == "inertia":
            # For inertia, elbow is where rate of decrease is maximum
            max_rate_idx = np.argmax(rates)
        else:
            # For silhouette, elbow is where rate of increase is maximum
            max_rate_idx = np.argmax(rates)
        
        # Return k value at elbow point
        return k_values[max_rate_idx + 1] if max_rate_idx + 1 < len(k_values) else k_values[-1]

