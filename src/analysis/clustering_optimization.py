"""Clustering Optimization: Elbow Method, Parameter Grid Search."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from .clustering_evaluation import evaluate_clustering


@dataclass
class ElbowMethod:
    """Find optimal number of clusters using elbow method."""
    
    k_range: list[int] | None = None
    metric: str = "inertia"  # "inertia" or "silhouette"
    n_init: int = 10
    random_state: int = 42
    
    def find_elbow(
        self, X: np.ndarray, k_range: list[int] | None = None
    ) -> tuple[int, dict[str, Any]]:
        """
        Find optimal number of clusters using elbow method.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            k_range: Range of k values to test (if None, uses self.k_range)
            
        Returns:
            Tuple of (optimal_k, metadata)
        """
        if k_range is None:
            k_range = self.k_range or list(range(2, min(11, X.shape[0])))
        
        results = []
        
        for k in tqdm(k_range, desc="Elbow method"):
            if k >= X.shape[0]:
                continue
            
            kmeans = KMeans(
                n_clusters=k,
                n_init=self.n_init,
                random_state=self.random_state,
            )
            labels = kmeans.fit_predict(X)
            
            if self.metric == "inertia":
                score = kmeans.inertia_
            elif self.metric == "silhouette":
                if len(np.unique(labels)) < 2:
                    score = -1.0
                else:
                    score = silhouette_score(X, labels)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            
            results.append({
                "k": k,
                "score": score,
                "inertia": kmeans.inertia_,
            })
        
        results_df = pd.DataFrame(results)
        
        if self.metric == "inertia":
            # Find elbow: point where decrease rate changes most
            optimal_k = self._find_elbow_point(results_df["k"].values, results_df["score"].values)
        else:  # silhouette
            # Find maximum
            optimal_k = results_df.loc[results_df["score"].idxmax(), "k"]
        
        metadata = {
            "results": results_df.to_dict("records"),
            "optimal_k": int(optimal_k),
            "metric": self.metric,
        }
        
        return int(optimal_k), metadata
    
    def _find_elbow_point(self, k_values: np.ndarray, scores: np.ndarray) -> int:
        """Find elbow point using rate of change."""
        if len(scores) < 3:
            return int(k_values[np.argmin(scores)])
        
        # Calculate rate of change
        rates = np.diff(scores)
        # Find point where rate of change decreases most
        rate_changes = np.diff(rates)
        elbow_idx = np.argmax(rate_changes) + 1
        
        if elbow_idx >= len(k_values):
            elbow_idx = len(k_values) - 1
        
        return int(k_values[elbow_idx])


@dataclass
class DBSCANParameterSearch:
    """Grid search for optimal DBSCAN parameters."""
    
    eps_range: list[float] = None
    min_samples_range: list[int] = None
    
    def __post_init__(self):
        if self.eps_range is None:
            self.eps_range = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        if self.min_samples_range is None:
            self.min_samples_range = [2, 3, 4, 5]
    
    def search(
        self, X: np.ndarray
    ) -> tuple[dict[str, float], pd.DataFrame]:
        """
        Search for optimal DBSCAN parameters.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Tuple of (best_params, results_df)
        """
        results = []
        
        for eps in tqdm(self.eps_range, desc="DBSCAN parameter search"):
            for min_samples in self.min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels) if len(labels) > 0 else 1.0
                
                # Evaluate if we have valid clusters
                if n_clusters >= 2:
                    evaluation = evaluate_clustering(X, labels)
                    silhouette = evaluation["silhouette_score"]
                    db_score = evaluation["davies_bouldin_score"]
                else:
                    silhouette = -1.0
                    db_score = float("inf")
                
                results.append({
                    "eps": eps,
                    "min_samples": min_samples,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "noise_ratio": noise_ratio,
                    "silhouette_score": silhouette,
                    "davies_bouldin_score": db_score,
                })
        
        results_df = pd.DataFrame(results)
        
        # Find best parameters (maximize clusters, minimize noise, maximize silhouette)
        valid_results = results_df[
            (results_df["n_clusters"] >= 2) & 
            (results_df["noise_ratio"] < 0.5) &
            (results_df["silhouette_score"] > -1)
        ]
        
        if len(valid_results) > 0:
            best_idx = valid_results["silhouette_score"].idxmax()
            best_params = {
                "eps": float(valid_results.loc[best_idx, "eps"]),
                "min_samples": int(valid_results.loc[best_idx, "min_samples"]),
            }
        else:
            # Fallback: use default
            best_params = {"eps": 0.5, "min_samples": 3}
        
        return best_params, results_df


def optimize_cluster_count(
    X: np.ndarray,
    k_range: list[int] | None = None,
    method: str = "elbow",
    metric: str = "inertia",
) -> tuple[int, dict[str, Any]]:
    """
    Optimize cluster count using various methods.
    
    Args:
        X: Feature matrix
        k_range: Range of k values
        method: "elbow" or "silhouette"
        metric: Metric to use for elbow method
        
    Returns:
        Tuple of (optimal_k, metadata)
    """
    if k_range is None:
        k_range = list(range(2, min(11, X.shape[0])))
    
    if method == "elbow":
        elbow = ElbowMethod(metric=metric, k_range=k_range)
        optimal_k, metadata = elbow.find_elbow(X)
    elif method == "silhouette":
        elbow = ElbowMethod(metric="silhouette", k_range=k_range)
        optimal_k, metadata = elbow.find_elbow(X)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return optimal_k, metadata
