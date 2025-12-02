"""Clustering Preprocessing: Normalization and Dimensionality Reduction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tqdm import tqdm

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False

from .feature_extraction import FeatureExtractor


@dataclass
class ClusteringPreprocessor:
    """
    Preprocess features for clustering analysis.
    
    Handles:
    - Normalization (Min-Max, Z-score, Robust scaling)
    - Dimensionality reduction (PCA, UMAP, t-SNE)
    - Feature selection
    """
    
    normalization_method: str = "robust"  # "minmax", "standard", "robust"
    dimensionality_reduction: str | None = "pca"  # "pca", "umap", "tsne", None
    n_components: int | None = None  # None = auto-select
    random_state: int = 42
    
    def __post_init__(self):
        """Validate configuration."""
        if self.normalization_method not in ["minmax", "standard", "robust"]:
            raise ValueError(
                f"normalization_method must be one of ['minmax', 'standard', 'robust'], "
                f"got {self.normalization_method}"
            )
        
        if self.dimensionality_reduction not in [None, "pca", "umap", "tsne"]:
            raise ValueError(
                f"dimensionality_reduction must be one of [None, 'pca', 'umap', 'tsne'], "
                f"got {self.dimensionality_reduction}"
            )
        
        if self.dimensionality_reduction == "umap" and not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        
        if self.dimensionality_reduction == "tsne" and not TSNE_AVAILABLE:
            raise ImportError("t-SNE not available. Install scikit-learn")
    
    def normalize(self, X: np.ndarray, fit: bool = True) -> tuple[np.ndarray, Any]:
        """
        Normalize feature matrix.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            fit: Whether to fit the scaler (True for training, False for transform only)
            
        Returns:
            Normalized matrix and scaler object
        """
        if self.normalization_method == "minmax":
            scaler = MinMaxScaler()
        elif self.normalization_method == "standard":
            scaler = StandardScaler()
        elif self.normalization_method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
        
        if fit:
            X_normalized = scaler.fit_transform(X)
        else:
            X_normalized = scaler.transform(X)
        
        return X_normalized, scaler
    
    def reduce_dimensions(
        self, X: np.ndarray, fit: bool = True
    ) -> tuple[np.ndarray, Any]:
        """
        Apply dimensionality reduction.
        
        Args:
            X: Normalized feature matrix (n_samples, n_features)
            fit: Whether to fit the reducer (True for training, False for transform only)
            
        Returns:
            Reduced matrix and reducer object
        """
        if self.dimensionality_reduction is None:
            return X, None
        
        n_samples, n_features = X.shape
        
        # Auto-select number of components
        if self.n_components is None:
            if self.dimensionality_reduction == "pca":
                # Keep 95% variance
                n_components = min(n_features, n_samples - 1)
            else:
                # Default to 2 for visualization
                n_components = min(2, n_features)
        else:
            n_components = min(self.n_components, n_features, n_samples - 1)
        
        if self.dimensionality_reduction == "pca":
            reducer = PCA(n_components=n_components, random_state=self.random_state)
            if fit:
                X_reduced = reducer.fit_transform(X)
            else:
                X_reduced = reducer.transform(X)
        
        elif self.dimensionality_reduction == "umap":
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=self.random_state,
                n_neighbors=min(15, n_samples - 1),
                min_dist=0.1,
            )
            if fit:
                X_reduced = reducer.fit_transform(X)
            else:
                X_reduced = reducer.transform(X)
        
        elif self.dimensionality_reduction == "tsne":
            reducer = TSNE(
                n_components=n_components,
                random_state=self.random_state,
                perplexity=min(30, n_samples - 1),
            )
            if fit:
                X_reduced = reducer.fit_transform(X)
            else:
                # t-SNE doesn't support transform, refit
                X_reduced = reducer.fit_transform(X)
        
        else:
            raise ValueError(
                f"Unknown dimensionality reduction method: {self.dimensionality_reduction}"
            )
        
        return X_reduced, reducer
    
    def preprocess(
        self, X: np.ndarray, fit: bool = True
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Complete preprocessing pipeline: normalize + reduce dimensions.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            fit: Whether to fit transformers (True for training, False for transform only)
            
        Returns:
            Preprocessed matrix and metadata dictionary
        """
        metadata = {}
        
        # Step 1: Normalize
        X_normalized, scaler = self.normalize(X, fit=fit)
        metadata["scaler"] = scaler
        metadata["normalization_method"] = self.normalization_method
        
        # Step 2: Reduce dimensions
        X_reduced, reducer = self.reduce_dimensions(X_normalized, fit=fit)
        metadata["reducer"] = reducer
        metadata["dimensionality_reduction"] = self.dimensionality_reduction
        
        if reducer is not None:
            if hasattr(reducer, "explained_variance_ratio_"):
                metadata["explained_variance_ratio"] = reducer.explained_variance_ratio_.tolist()
                metadata["cumulative_variance"] = np.cumsum(
                    reducer.explained_variance_ratio_
                ).tolist()
            metadata["n_components"] = X_reduced.shape[1]
        else:
            metadata["n_components"] = X_normalized.shape[1]
        
        return X_reduced, metadata


def prepare_clustering_data(
    output_dirs: list[Path],
    preprocessor: ClusteringPreprocessor | None = None,
    feature_extractor: FeatureExtractor | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """
    Prepare data for clustering from multiple output directories.
    
    Args:
        output_dirs: List of output directory paths
        preprocessor: Optional preprocessor (uses default if None)
        feature_extractor: Optional feature extractor (uses default if None)
        
    Returns:
        Tuple of:
        - features_df: DataFrame with features for each sample
        - X_processed: Preprocessed feature matrix ready for clustering
        - metadata: Preprocessing metadata
    """
    if feature_extractor is None:
        feature_extractor = FeatureExtractor()
    
    if preprocessor is None:
        preprocessor = ClusteringPreprocessor()
    
    # Extract features from all directories
    all_features = []
    sample_ids = []
    
    print(f"Extracting features from {len(output_dirs)} output directories...")
    for output_dir in tqdm(output_dirs, desc="Extracting features"):
        try:
            features = feature_extractor.extract_from_output_dir(output_dir)
            if features:
                all_features.append(features)
                sample_ids.append(output_dir.name)
        except Exception as e:
            print(f"Warning: Failed to extract features from {output_dir}: {e}")
            continue
    
    if not all_features:
        raise ValueError("No features extracted from any output directory")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features, index=sample_ids)
    
    # Fill NaN values with 0 (for missing indicators)
    features_df = features_df.fillna(0)
    
    # Convert to numpy array
    X = features_df.values
    
    # Preprocess
    print("Preprocessing features...")
    X_processed, metadata = preprocessor.preprocess(X, fit=True)
    
    # Add sample IDs to metadata
    metadata["sample_ids"] = sample_ids
    metadata["feature_names"] = features_df.columns.tolist()
    metadata["n_samples"] = len(sample_ids)
    metadata["n_features_original"] = X.shape[1]
    
    return features_df, X_processed, metadata

