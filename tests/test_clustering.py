"""Tests for clustering module."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.clustering import (
    ClusteringAnalyzer,
    ClusteringMethod,
    DimensionReductionMethod,
    FeatureExtractor,
    NormalizationMethod,
    create_feature_matrix,
)


class TestFeatureExtractor:
    """Test cases for FeatureExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create default feature extractor."""
        return FeatureExtractor()

    @pytest.fixture
    def sample_mfa_spectrum(self):
        """Create sample MFA spectrum DataFrame."""
        q = np.linspace(-5, 5, 21)
        # Simulate typical multifractal spectrum
        tau = 2 * q - 0.5 * q**2 / 10  # τ(q) curve
        alpha = np.gradient(tau, q)  # α = dτ/dq
        f_alpha = q * alpha - tau  # f(α) = qα - τ

        return pd.DataFrame({
            "q": q,
            "alpha": alpha,
            "f_alpha": f_alpha,
            "tau": tau,
            "R2": np.ones_like(q) * 0.95,
        })

    @pytest.fixture
    def sample_lacunarity(self):
        """Create sample Lacunarity DataFrame."""
        r = np.array([2, 4, 8, 16, 32, 64, 128])
        # Simulate power law decay Λ(r) ∝ r^(-β)
        lambda_vals = 2.0 * r ** (-0.5)

        return pd.DataFrame({
            "r": r,
            "lambda": lambda_vals,
            "sigma": np.ones_like(r) * 0.1,
            "mu": np.ones_like(r) * 50,
            "cv": np.ones_like(r) * 0.1,
        })

    @pytest.fixture
    def sample_percolation(self):
        """Create sample Percolation DataFrame."""
        d = np.linspace(1, 100, 50)
        # Simulate S-shaped percolation curve
        d_crit = 50
        steepness = 0.1
        giant_fraction = 1 / (1 + np.exp(-steepness * (d - d_crit)))
        n_clusters = 100 * (1 - giant_fraction) + 1
        max_cluster_size = giant_fraction * 1000

        return pd.DataFrame({
            "d": d,
            "max_cluster_size": max_cluster_size,
            "n_clusters": n_clusters,
            "giant_fraction": giant_fraction,
        })

    def test_extract_mfa_features(self, extractor, sample_mfa_spectrum):
        """Test MFA feature extraction."""
        features = extractor.extract_mfa_features(sample_mfa_spectrum)

        # Check all expected features are present
        assert "mfa_spectrum_width" in features
        assert "mfa_D0" in features
        assert "mfa_D1" in features
        assert "mfa_D2" in features
        assert "mfa_asymmetry" in features
        assert "mfa_mean_R2" in features

        # Check values are reasonable
        assert features["mfa_spectrum_width"] >= 0
        assert features["mfa_mean_R2"] == pytest.approx(0.95)

    def test_extract_lacunarity_features(self, extractor, sample_lacunarity):
        """Test Lacunarity feature extraction."""
        features = extractor.extract_lacunarity_features(sample_lacunarity)

        # Check all expected features are present
        assert "lac_beta" in features
        assert "lac_R2" in features
        assert "lac_at_r4" in features
        assert "lac_at_r16" in features
        assert "lac_at_r64" in features
        assert "lac_mean" in features

        # Check power law exponent is reasonable (should be ~0.5 for our simulation)
        assert features["lac_beta"] > 0
        assert features["lac_R2"] > 0.9

    def test_extract_percolation_features(self, extractor, sample_percolation):
        """Test Percolation feature extraction."""
        features = extractor.extract_percolation_features(sample_percolation)

        # Check all expected features are present
        assert "perc_d_critical_10" in features
        assert "perc_d_critical_50" in features
        assert "perc_d_critical_90" in features
        assert "perc_transition_width" in features
        assert "perc_max_clusters" in features
        assert "perc_fragmentation" in features

        # Critical threshold at 50% should be near d_crit=50
        assert 30 < features["perc_d_critical_50"] < 70

        # Transition width should be positive
        assert features["perc_transition_width"] > 0

    def test_extract_all_features(
        self, extractor, sample_mfa_spectrum, sample_lacunarity, sample_percolation
    ):
        """Test combined feature extraction."""
        features = extractor.extract_all_features(
            mfa_spectrum=sample_mfa_spectrum,
            lacunarity=sample_lacunarity,
            percolation=sample_percolation,
        )

        # Should have features from all three indicators
        mfa_features = [k for k in features if k.startswith("mfa_")]
        lac_features = [k for k in features if k.startswith("lac_")]
        perc_features = [k for k in features if k.startswith("perc_")]

        assert len(mfa_features) >= 6
        assert len(lac_features) >= 5
        assert len(perc_features) >= 5

    def test_partial_features(self, extractor, sample_mfa_spectrum):
        """Test extraction with only some indicators available."""
        features = extractor.extract_all_features(
            mfa_spectrum=sample_mfa_spectrum,
            lacunarity=None,
            percolation=None,
        )

        # Should only have MFA features
        assert any(k.startswith("mfa_") for k in features)
        assert not any(k.startswith("lac_") for k in features)
        assert not any(k.startswith("perc_") for k in features)


class TestClusteringAnalyzer:
    """Test cases for ClusteringAnalyzer."""

    @pytest.fixture
    def sample_features_df(self):
        """Create sample feature matrix for clustering."""
        np.random.seed(42)
        n_samples = 30

        # Create 3 clusters of data
        cluster1 = np.random.randn(10, 5) + np.array([0, 0, 0, 0, 0])
        cluster2 = np.random.randn(10, 5) + np.array([3, 3, 0, 0, 0])
        cluster3 = np.random.randn(10, 5) + np.array([0, 0, 3, 3, 0])

        data = np.vstack([cluster1, cluster2, cluster3])

        return pd.DataFrame(
            data,
            columns=["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
            index=[f"site_{i}" for i in range(n_samples)],
        )

    def test_standard_normalization(self, sample_features_df):
        """Test standard (Z-score) normalization."""
        analyzer = ClusteringAnalyzer(
            normalization=NormalizationMethod.STANDARD,
            clustering=ClusteringMethod.KMEANS,
            n_clusters=3,
        )

        result_df, metadata = analyzer.fit_transform(sample_features_df)

        # Check normalized values have mean ~0 and std ~1
        norm_cols = [c for c in result_df.columns if c.endswith("_norm")]
        for col in norm_cols:
            assert abs(result_df[col].mean()) < 0.1
            assert abs(result_df[col].std() - 1.0) < 0.2

    def test_minmax_normalization(self, sample_features_df):
        """Test Min-Max normalization."""
        analyzer = ClusteringAnalyzer(
            normalization=NormalizationMethod.MINMAX,
            clustering=ClusteringMethod.KMEANS,
            n_clusters=3,
        )

        result_df, metadata = analyzer.fit_transform(sample_features_df)

        # Check normalized values are in [0, 1] range
        norm_cols = [c for c in result_df.columns if c.endswith("_norm")]
        for col in norm_cols:
            assert result_df[col].min() >= -0.01
            assert result_df[col].max() <= 1.01

    def test_robust_normalization(self, sample_features_df):
        """Test Robust normalization."""
        analyzer = ClusteringAnalyzer(
            normalization=NormalizationMethod.ROBUST,
            clustering=ClusteringMethod.KMEANS,
            n_clusters=3,
        )

        result_df, metadata = analyzer.fit_transform(sample_features_df)

        # Should complete without error
        assert "cluster" in result_df.columns
        assert len(result_df) == len(sample_features_df)

    def test_kmeans_clustering(self, sample_features_df):
        """Test K-means clustering."""
        analyzer = ClusteringAnalyzer(
            normalization=NormalizationMethod.STANDARD,
            clustering=ClusteringMethod.KMEANS,
            n_clusters=3,
        )

        result_df, metadata = analyzer.fit_transform(sample_features_df)

        # Check cluster labels
        assert "cluster" in result_df.columns
        assert result_df["cluster"].nunique() <= 3
        assert metadata["n_clusters_found"] <= 3

    def test_dbscan_clustering(self, sample_features_df):
        """Test DBSCAN clustering."""
        analyzer = ClusteringAnalyzer(
            normalization=NormalizationMethod.STANDARD,
            clustering=ClusteringMethod.DBSCAN,
            dbscan_eps=1.5,
            dbscan_min_samples=3,
        )

        result_df, metadata = analyzer.fit_transform(sample_features_df)

        # Check cluster labels exist
        assert "cluster" in result_df.columns
        # DBSCAN may label some points as noise (-1)
        assert result_df["cluster"].min() >= -1

    def test_hierarchical_clustering(self, sample_features_df):
        """Test Hierarchical clustering."""
        analyzer = ClusteringAnalyzer(
            normalization=NormalizationMethod.STANDARD,
            clustering=ClusteringMethod.HIERARCHICAL,
            n_clusters=3,
            hierarchical_linkage="ward",
        )

        result_df, metadata = analyzer.fit_transform(sample_features_df)

        # Check cluster labels
        assert "cluster" in result_df.columns
        assert result_df["cluster"].nunique() <= 3

    def test_pca_reduction(self, sample_features_df):
        """Test PCA dimensionality reduction."""
        analyzer = ClusteringAnalyzer(
            normalization=NormalizationMethod.STANDARD,
            clustering=ClusteringMethod.KMEANS,
            dimension_reduction=DimensionReductionMethod.PCA,
            pca_n_components=2,
            n_clusters=3,
        )

        result_df, metadata = analyzer.fit_transform(sample_features_df)

        # Check PCA columns are present
        assert "PC1" in result_df.columns
        assert "PC2" in result_df.columns
        assert "explained_variance_ratio" in metadata["dimension_reduction_params"]

    def test_pca_variance_ratio(self, sample_features_df):
        """Test PCA with variance ratio threshold."""
        analyzer = ClusteringAnalyzer(
            normalization=NormalizationMethod.STANDARD,
            clustering=ClusteringMethod.KMEANS,
            dimension_reduction=DimensionReductionMethod.PCA,
            pca_n_components=0.9,  # 90% variance
            n_clusters=3,
        )

        result_df, metadata = analyzer.fit_transform(sample_features_df)

        # Should have some PC columns
        pc_cols = [c for c in result_df.columns if c.startswith("PC")]
        assert len(pc_cols) >= 1

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        analyzer = ClusteringAnalyzer(n_clusters=3)
        empty_df = pd.DataFrame(columns=["feature_1", "feature_2"])

        result_df, metadata = analyzer.fit_transform(empty_df)

        assert len(result_df) == 0

    def test_single_sample(self):
        """Test handling of single sample."""
        analyzer = ClusteringAnalyzer(n_clusters=3)
        single_df = pd.DataFrame(
            {"feature_1": [1.0], "feature_2": [2.0]},
            index=["site_0"],
        )

        result_df, metadata = analyzer.fit_transform(single_df)

        assert len(result_df) == 1
        assert "cluster" in result_df.columns

    def test_metadata_contents(self, sample_features_df):
        """Test metadata dictionary contents."""
        analyzer = ClusteringAnalyzer(
            normalization=NormalizationMethod.STANDARD,
            clustering=ClusteringMethod.KMEANS,
            n_clusters=3,
        )

        result_df, metadata = analyzer.fit_transform(sample_features_df)

        assert "normalization_method" in metadata
        assert "normalization_params" in metadata
        assert "clustering_method" in metadata
        assert "n_clusters_found" in metadata
        assert "feature_names" in metadata


class TestCreateFeatureMatrix:
    """Test cases for create_feature_matrix function."""

    def test_create_from_site_results(self):
        """Test creating feature matrix from site results."""
        # Create mock site results
        np.random.seed(42)

        def make_mfa_spectrum():
            q = np.linspace(-5, 5, 11)
            tau = 2 * q - 0.1 * q**2
            alpha = np.gradient(tau, q)
            f_alpha = q * alpha - tau
            return pd.DataFrame({
                "q": q, "alpha": alpha, "f_alpha": f_alpha,
                "tau": tau, "R2": np.ones_like(q) * 0.9
            })

        def make_lacunarity():
            r = np.array([2, 4, 8, 16, 32])
            return pd.DataFrame({
                "r": r, "lambda": 1.5 * r**(-0.3),
                "sigma": np.ones_like(r) * 0.1,
                "mu": np.ones_like(r) * 50,
                "cv": np.ones_like(r) * 0.1,
            })

        site_results = {
            "tokyo": {
                "mfa": make_mfa_spectrum(),
                "lacunarity": make_lacunarity(),
            },
            "osaka": {
                "mfa": make_mfa_spectrum(),
                "lacunarity": make_lacunarity(),
            },
        }

        features_df = create_feature_matrix(site_results)

        # Check structure
        assert len(features_df) == 2
        assert "tokyo" in features_df.index
        assert "osaka" in features_df.index

        # Check features exist
        assert any(c.startswith("mfa_") for c in features_df.columns)
        assert any(c.startswith("lac_") for c in features_df.columns)

    def test_partial_site_data(self):
        """Test handling sites with partial data."""
        site_results = {
            "site_a": {
                "mfa": pd.DataFrame({
                    "q": [0], "alpha": [1.5], "f_alpha": [1.0],
                    "tau": [0], "R2": [0.9]
                }),
            },
            "site_b": {
                "lacunarity": pd.DataFrame({
                    "r": [4, 8], "lambda": [1.5, 1.3],
                    "sigma": [0.1, 0.1], "mu": [50, 50], "cv": [0.1, 0.1]
                }),
            },
        }

        features_df = create_feature_matrix(site_results)

        assert len(features_df) == 2
        # site_a should have MFA features but not lac features
        # site_b should have lac features but not MFA features


class TestIntegrationClustering:
    """Integration tests for the clustering pipeline."""

    def test_full_pipeline(self):
        """Test full feature extraction and clustering pipeline."""
        np.random.seed(42)

        # Generate synthetic site data
        n_sites = 20

        def generate_site_data(cluster_id):
            q = np.linspace(-5, 5, 11)
            base_tau = 2 * q - 0.1 * q**2

            if cluster_id == 0:
                tau = base_tau + np.random.randn() * 0.1
            elif cluster_id == 1:
                tau = base_tau * 1.2 + np.random.randn() * 0.1
            else:
                tau = base_tau * 0.8 + np.random.randn() * 0.1

            alpha = np.gradient(tau, q)
            f_alpha = q * alpha - tau

            mfa = pd.DataFrame({
                "q": q, "alpha": alpha, "f_alpha": f_alpha,
                "tau": tau, "R2": np.ones_like(q) * 0.9
            })

            r = np.array([2, 4, 8, 16, 32])
            beta = 0.3 + cluster_id * 0.1 + np.random.randn() * 0.02
            lac = pd.DataFrame({
                "r": r, "lambda": 1.5 * r**(-beta),
                "sigma": np.ones_like(r) * 0.1,
                "mu": np.ones_like(r) * 50,
                "cv": np.ones_like(r) * 0.1,
            })

            return {"mfa": mfa, "lacunarity": lac}

        # Generate sites in 3 clusters
        site_results = {}
        true_labels = []
        for i in range(n_sites):
            cluster_id = i % 3
            site_results[f"site_{i}"] = generate_site_data(cluster_id)
            true_labels.append(cluster_id)

        # Extract features
        features_df = create_feature_matrix(site_results)

        # Run clustering
        analyzer = ClusteringAnalyzer(
            normalization=NormalizationMethod.STANDARD,
            clustering=ClusteringMethod.KMEANS,
            n_clusters=3,
        )

        result_df, metadata = analyzer.fit_transform(features_df)

        # Verify results
        assert len(result_df) == n_sites
        assert "cluster" in result_df.columns
        assert metadata["n_clusters_found"] == 3
