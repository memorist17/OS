"""Clustering Analysis for MFA, Lacunarity, and Percolation indicators.

This module provides methods to:
1. Extract features from the 3 analysis indicators
2. Normalize features for clustering
3. Apply dimensionality reduction (optional)
4. Perform clustering analysis
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.interpolate import interp1d


class NormalizationMethod(str, Enum):
    """Normalization methods for feature scaling."""

    STANDARD = "standard"  # Z-score: (x - mean) / std
    MINMAX = "minmax"  # Min-Max: (x - min) / (max - min)
    ROBUST = "robust"  # Robust: (x - median) / IQR


class ClusteringMethod(str, Enum):
    """Clustering algorithms."""

    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"


class DimensionReductionMethod(str, Enum):
    """Dimensionality reduction methods."""

    PCA = "pca"
    NONE = "none"


@dataclass
class FeatureExtractor:
    """Extract scalar features from MFA, Lacunarity, and Percolation results.

    Each indicator produces multi-dimensional or curve-based outputs.
    This class compresses them into scalar features suitable for clustering.
    """

    # MFA feature extraction parameters
    mfa_q_values: list[float] = field(default_factory=lambda: [0, 1, 2])

    # Lacunarity feature extraction parameters
    lac_scales: list[int] = field(default_factory=lambda: [4, 16, 64])

    # Percolation feature extraction parameters
    perc_fractions: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    def extract_mfa_features(self, spectrum_df: pd.DataFrame) -> dict[str, float]:
        """Extract scalar features from MFA spectrum.

        Args:
            spectrum_df: DataFrame with columns [q, alpha, f_alpha, tau, R2]

        Returns:
            Dictionary of feature names to values:
            - mfa_spectrum_width: Range of α values (Δα = α_max - α_min)
            - mfa_D0: Capacity dimension D(0)
            - mfa_D1: Information dimension D(1)
            - mfa_D2: Correlation dimension D(2)
            - mfa_asymmetry: Asymmetry of f(α) spectrum
            - mfa_mean_R2: Mean R² of linear fits
        """
        features = {}

        q = spectrum_df["q"].values
        alpha = spectrum_df["alpha"].values
        tau = spectrum_df["tau"].values

        # Spectrum width (Δα)
        valid_alpha = alpha[~np.isnan(alpha)]
        if len(valid_alpha) > 0:
            features["mfa_spectrum_width"] = float(np.max(valid_alpha) - np.min(valid_alpha))
        else:
            features["mfa_spectrum_width"] = 0.0

        # Generalized dimensions D(q)
        for q_val in self.mfa_q_values:
            idx = np.argmin(np.abs(q - q_val))
            if abs(q_val - 1) < 0.01:
                # D(1) via derivative
                D_q = np.gradient(tau, q)[idx] if len(q) > 1 else tau[idx]
            else:
                # D(q) = τ(q) / (q - 1) for q ≠ 1
                D_q = tau[idx] / (q_val - 1)
            features[f"mfa_D{int(q_val)}"] = float(D_q)

        # Asymmetry: compare α at q<0 vs q>0
        alpha_neg = alpha[q < 0]
        alpha_pos = alpha[q > 0]
        if len(alpha_neg) > 0 and len(alpha_pos) > 0:
            alpha_neg_mean = np.nanmean(alpha_neg)
            alpha_pos_mean = np.nanmean(alpha_pos)
            features["mfa_asymmetry"] = float(alpha_neg_mean - alpha_pos_mean)
        else:
            features["mfa_asymmetry"] = 0.0

        # Mean R²
        features["mfa_mean_R2"] = float(spectrum_df["R2"].mean())

        return features

    def extract_lacunarity_features(self, lacunarity_df: pd.DataFrame) -> dict[str, float]:
        """Extract scalar features from Lacunarity curve.

        Args:
            lacunarity_df: DataFrame with columns [r, lambda, sigma, mu, cv]

        Returns:
            Dictionary of feature names to values:
            - lac_beta: Power law decay exponent
            - lac_R2: R² of power law fit
            - lac_at_r{scale}: Lacunarity at specific scales
            - lac_mean: Mean lacunarity across all scales
        """
        features = {}

        r = lacunarity_df["r"].values
        lambda_vals = lacunarity_df["lambda"].values

        # Filter valid values for power law fit
        valid_mask = (lambda_vals > 0) & (~np.isnan(lambda_vals))
        if np.sum(valid_mask) >= 2:
            log_r = np.log(r[valid_mask])
            log_lambda = np.log(lambda_vals[valid_mask])
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_r, log_lambda)
            features["lac_beta"] = float(-slope)
            features["lac_R2"] = float(r_value**2)
        else:
            features["lac_beta"] = 0.0
            features["lac_R2"] = 0.0

        # Lacunarity at specific scales (with interpolation)
        valid_r = r[valid_mask]
        valid_lambda = lambda_vals[valid_mask]

        if len(valid_r) >= 2:
            # Create interpolator
            interp_func = interp1d(
                valid_r, valid_lambda, kind="linear", bounds_error=False, fill_value="extrapolate"
            )
            for scale in self.lac_scales:
                features[f"lac_at_r{scale}"] = float(interp_func(scale))
        else:
            for scale in self.lac_scales:
                features[f"lac_at_r{scale}"] = 1.0

        # Mean lacunarity
        if np.any(valid_mask):
            features["lac_mean"] = float(np.nanmean(lambda_vals[valid_mask]))
        else:
            features["lac_mean"] = 1.0

        return features

    def extract_percolation_features(
        self, percolation_df: pd.DataFrame, total_nodes: int | None = None
    ) -> dict[str, float]:
        """Extract scalar features from Percolation curve.

        Args:
            percolation_df: DataFrame with columns [d, max_cluster_size, n_clusters, giant_fraction]
            total_nodes: Total number of nodes (optional, for normalization)

        Returns:
            Dictionary of feature names to values:
            - perc_d_critical_{fraction}: Distance threshold at specified giant fraction
            - perc_transition_width: Width of percolation transition (d_90 - d_10)
            - perc_max_clusters: Maximum number of clusters
            - perc_fragmentation: Average fragmentation (n_clusters / max_cluster_size)
        """
        features = {}

        d = percolation_df["d"].values
        gf = percolation_df["giant_fraction"].values
        n_clusters = percolation_df["n_clusters"].values
        max_cluster_size = percolation_df["max_cluster_size"].values

        # Critical thresholds at various giant fractions
        for fraction in self.perc_fractions:
            d_crit = self._find_threshold(d, gf, fraction)
            # Use percentage for naming (0.5 -> 50)
            features[f"perc_d_critical_{int(fraction * 100)}"] = float(d_crit)

        # Transition width
        d_10 = self._find_threshold(d, gf, 0.1)
        d_90 = self._find_threshold(d, gf, 0.9)
        features["perc_transition_width"] = float(d_90 - d_10)

        # Maximum clusters
        features["perc_max_clusters"] = float(np.max(n_clusters))

        # Average fragmentation (exclude zero cluster sizes)
        valid_mask = max_cluster_size > 0
        if np.any(valid_mask):
            fragmentation = n_clusters[valid_mask] / max_cluster_size[valid_mask]
            features["perc_fragmentation"] = float(np.mean(fragmentation))
        else:
            features["perc_fragmentation"] = 0.0

        return features

    def _find_threshold(self, d: np.ndarray, gf: np.ndarray, target_fraction: float) -> float:
        """Find the distance threshold where giant fraction reaches target.

        Args:
            d: Distance thresholds
            gf: Giant fraction values
            target_fraction: Target fraction

        Returns:
            Interpolated distance threshold
        """
        for i in range(len(gf) - 1):
            if gf[i] < target_fraction <= gf[i + 1]:
                t = (target_fraction - gf[i]) / (gf[i + 1] - gf[i] + 1e-10)
                return d[i] + t * (d[i + 1] - d[i])

        # If not found, return boundary
        if gf[-1] < target_fraction:
            return d[-1]
        return d[0]

    def extract_all_features(
        self,
        mfa_spectrum: pd.DataFrame | None = None,
        lacunarity: pd.DataFrame | None = None,
        percolation: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Extract all features from available indicators.

        Args:
            mfa_spectrum: MFA spectrum DataFrame (optional)
            lacunarity: Lacunarity DataFrame (optional)
            percolation: Percolation DataFrame (optional)

        Returns:
            Combined dictionary of all extracted features
        """
        features = {}

        if mfa_spectrum is not None:
            features.update(self.extract_mfa_features(mfa_spectrum))

        if lacunarity is not None:
            features.update(self.extract_lacunarity_features(lacunarity))

        if percolation is not None:
            features.update(self.extract_percolation_features(percolation))

        return features


@dataclass
class ClusteringAnalyzer:
    """Perform clustering analysis on extracted features from multiple sites.

    This class handles:
    1. Feature normalization
    2. Optional dimensionality reduction
    3. Clustering
    """

    normalization: NormalizationMethod = NormalizationMethod.STANDARD
    clustering: ClusteringMethod = ClusteringMethod.KMEANS
    dimension_reduction: DimensionReductionMethod = DimensionReductionMethod.NONE

    # Clustering parameters
    n_clusters: int = 5
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 3
    hierarchical_linkage: Literal["ward", "complete", "average", "single"] = "ward"

    # PCA parameters
    pca_n_components: int | float = 3  # int for exact, float for variance ratio

    def fit_transform(
        self, features_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        """Fit the clustering pipeline and transform the data.

        Args:
            features_df: DataFrame with features (rows=sites, columns=features)

        Returns:
            result_df: DataFrame with cluster labels and reduced dimensions (if applicable)
            metadata: Dictionary with normalization parameters and cluster info
        """
        # Store feature names for later
        feature_names = features_df.columns.tolist()
        site_indices = features_df.index.tolist()

        # Get numeric data
        X = features_df.values.astype(np.float64)

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Normalize
        X_normalized, norm_params = self._normalize(X)

        # Dimensionality reduction
        if self.dimension_reduction == DimensionReductionMethod.PCA:
            X_reduced, reduction_params = self._reduce_dimensions_pca(X_normalized)
        else:
            X_reduced = X_normalized
            reduction_params = {}

        # Clustering
        labels = self._cluster(X_reduced)

        # Build result DataFrame
        result_data = {"cluster": labels}

        # Add reduced dimensions if applicable
        if self.dimension_reduction == DimensionReductionMethod.PCA:
            for i in range(X_reduced.shape[1]):
                result_data[f"PC{i + 1}"] = X_reduced[:, i]

        # Add normalized features
        for i, name in enumerate(feature_names):
            result_data[f"{name}_norm"] = X_normalized[:, i]

        result_df = pd.DataFrame(result_data, index=site_indices)

        # Build metadata
        metadata = {
            "normalization_method": self.normalization.value,
            "normalization_params": norm_params,
            "dimension_reduction_method": self.dimension_reduction.value,
            "dimension_reduction_params": reduction_params,
            "clustering_method": self.clustering.value,
            "n_clusters_found": len(np.unique(labels[labels >= 0])),
            "feature_names": feature_names,
        }

        return result_df, metadata

    def _normalize(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """Normalize features.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            X_normalized: Normalized feature matrix
            params: Normalization parameters for inverse transform
        """
        params = {}

        if self.normalization == NormalizationMethod.STANDARD:
            # Z-score normalization
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            X_normalized = (X - mean) / std
            params = {"mean": mean.tolist(), "std": std.tolist()}

        elif self.normalization == NormalizationMethod.MINMAX:
            # Min-Max normalization
            min_val = np.min(X, axis=0)
            max_val = np.max(X, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1  # Avoid division by zero
            X_normalized = (X - min_val) / range_val
            params = {"min": min_val.tolist(), "max": max_val.tolist()}

        elif self.normalization == NormalizationMethod.ROBUST:
            # Robust normalization using median and IQR
            median = np.median(X, axis=0)
            q75 = np.percentile(X, 75, axis=0)
            q25 = np.percentile(X, 25, axis=0)
            iqr = q75 - q25
            iqr[iqr == 0] = 1  # Avoid division by zero
            X_normalized = (X - median) / iqr
            params = {"median": median.tolist(), "iqr": iqr.tolist()}

        else:
            X_normalized = X

        return X_normalized, params

    def _reduce_dimensions_pca(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """Reduce dimensions using PCA.

        Args:
            X: Normalized feature matrix (n_samples, n_features)

        Returns:
            X_reduced: Reduced feature matrix
            params: PCA parameters
        """
        # Center the data (already normalized, but ensure mean=0)
        X_centered = X - np.mean(X, axis=0)

        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)

        # Handle 1D case
        if cov_matrix.ndim == 0:
            return X, {"n_components": 1, "explained_variance_ratio": [1.0]}

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalue in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Determine number of components
        if isinstance(self.pca_n_components, float):
            # Use variance ratio
            variance_ratio = eigenvalues / eigenvalues.sum()
            cumsum = np.cumsum(variance_ratio)
            n_components = np.searchsorted(cumsum, self.pca_n_components) + 1
        else:
            n_components = min(self.pca_n_components, X.shape[1], X.shape[0])

        # Project data
        X_reduced = X_centered @ eigenvectors[:, :n_components]

        # Calculate explained variance ratio
        variance_ratio = eigenvalues[:n_components] / eigenvalues.sum()

        params = {
            "n_components": n_components,
            "explained_variance_ratio": variance_ratio.tolist(),
            "components": eigenvectors[:, :n_components].tolist(),
        }

        return X_reduced, params

    def _cluster(self, X: np.ndarray) -> np.ndarray:
        """Perform clustering.

        Args:
            X: Feature matrix (possibly reduced)

        Returns:
            labels: Cluster labels for each sample
        """
        if self.clustering == ClusteringMethod.KMEANS:
            return self._kmeans(X)
        elif self.clustering == ClusteringMethod.DBSCAN:
            return self._dbscan(X)
        elif self.clustering == ClusteringMethod.HIERARCHICAL:
            return self._hierarchical(X)
        else:
            return np.zeros(X.shape[0], dtype=int)

    def _kmeans(self, X: np.ndarray, max_iter: int = 300) -> np.ndarray:
        """K-means clustering implementation.

        Args:
            X: Feature matrix
            max_iter: Maximum iterations

        Returns:
            labels: Cluster labels
        """
        n_samples, n_features = X.shape
        n_clusters = min(self.n_clusters, n_samples)

        if n_samples == 0:
            return np.array([], dtype=int)

        # Initialize centroids using k-means++
        centroids = self._kmeans_plusplus_init(X, n_clusters)

        labels = np.zeros(n_samples, dtype=int)

        for _ in range(max_iter):
            # Assign points to nearest centroid
            diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
            distances = np.sqrt((diff**2).sum(axis=2))
            new_labels = np.argmin(distances, axis=1)

            # Check convergence
            if np.all(labels == new_labels):
                break
            labels = new_labels

            # Update centroids
            for k in range(n_clusters):
                mask = labels == k
                if np.any(mask):
                    centroids[k] = X[mask].mean(axis=0)

        return labels

    def _kmeans_plusplus_init(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """K-means++ initialization.

        Args:
            X: Feature matrix
            n_clusters: Number of clusters

        Returns:
            centroids: Initial centroid positions
        """
        n_samples = X.shape[0]
        centroids = []

        # Choose first centroid randomly
        idx = np.random.randint(n_samples)
        centroids.append(X[idx])

        for _ in range(1, n_clusters):
            # Compute distances to nearest centroid
            distances = np.min(
                [np.sum((X - c) ** 2, axis=1) for c in centroids],
                axis=0
            )
            # Choose next centroid with probability proportional to distance squared
            total_dist = distances.sum()
            if total_dist == 0:
                # All points are identical to existing centroids, choose randomly
                idx = np.random.randint(n_samples)
            else:
                probs = distances / total_dist
                idx = np.random.choice(n_samples, p=probs)
            centroids.append(X[idx])

        return np.array(centroids)

    def _dbscan(self, X: np.ndarray) -> np.ndarray:
        """DBSCAN clustering implementation.

        Note: This implementation precomputes the full pairwise distance matrix,
        which has O(n²) memory complexity. For very large datasets (>10000 samples),
        consider using scikit-learn's DBSCAN with a ball tree or KD-tree.

        Args:
            X: Feature matrix

        Returns:
            labels: Cluster labels (-1 for noise)
        """
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)

        # Compute pairwise distances (O(n²) memory)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        distances = np.sqrt((diff**2).sum(axis=2))

        cluster_id = 0

        for i in range(n_samples):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = np.where(distances[i] <= self.dbscan_eps)[0]

            if len(neighbors) < self.dbscan_min_samples:
                # Mark as noise
                continue

            # Start new cluster
            labels[i] = cluster_id

            # Expand cluster using a set to track which points to process
            seed_set = set(neighbors.tolist())
            processed = {i}

            while seed_set - processed:
                q = (seed_set - processed).pop()
                processed.add(q)

                if not visited[q]:
                    visited[q] = True
                    q_neighbors = np.where(distances[q] <= self.dbscan_eps)[0]
                    if len(q_neighbors) >= self.dbscan_min_samples:
                        seed_set.update(q_neighbors.tolist())

                if labels[q] == -1:
                    labels[q] = cluster_id

            cluster_id += 1

        return labels

    def _hierarchical(self, X: np.ndarray) -> np.ndarray:
        """Hierarchical clustering using scipy.

        Args:
            X: Feature matrix

        Returns:
            labels: Cluster labels
        """
        if X.shape[0] < 2:
            return np.zeros(X.shape[0], dtype=int)

        # Compute linkage matrix
        Z = linkage(X, method=self.hierarchical_linkage)

        # Cut tree to get clusters
        labels = fcluster(Z, t=self.n_clusters, criterion="maxclust") - 1

        return labels


def create_feature_matrix(
    site_results: dict[str, dict[str, pd.DataFrame]],
    extractor: FeatureExtractor | None = None
) -> pd.DataFrame:
    """Create feature matrix from multiple site analysis results.

    Args:
        site_results: Dictionary mapping site_id to dict of DataFrames
            e.g., {"tokyo": {"mfa": mfa_df, "lacunarity": lac_df, "percolation": perc_df}}
        extractor: FeatureExtractor instance (created with defaults if None)

    Returns:
        DataFrame with sites as rows and features as columns
    """
    if extractor is None:
        extractor = FeatureExtractor()

    all_features = []

    for site_id, results in site_results.items():
        features = extractor.extract_all_features(
            mfa_spectrum=results.get("mfa"),
            lacunarity=results.get("lacunarity"),
            percolation=results.get("percolation"),
        )
        features["site_id"] = site_id
        all_features.append(features)

    df = pd.DataFrame(all_features)
    df = df.set_index("site_id")

    return df
