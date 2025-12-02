"""Feature Extraction from MFA, Lacunarity, and Percolation Results."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass
class FeatureExtractor:
    """
    Extract clustering-ready features from three indicators.
    
    Extracts statistical features from:
    - MFA: Multifractal spectrum (q, alpha, f_alpha, tau)
    - Lacunarity: Lacunarity curve (r, lambda)
    - Percolation: Percolation curve (d, max_cluster_size, n_clusters, giant_fraction)
    """

    def extract_mfa_features(self, mfa_spectrum: pd.DataFrame) -> dict[str, float]:
        """
        Extract features from MFA spectrum.
        
        Args:
            mfa_spectrum: DataFrame with columns [q, alpha, f_alpha, tau, R2]
            
        Returns:
            Dictionary of feature names and values
        """
        features = {}
        
        # Basic statistics from spectrum
        features["mfa_alpha_mean"] = mfa_spectrum["alpha"].mean()
        features["mfa_alpha_std"] = mfa_spectrum["alpha"].std()
        features["mfa_alpha_min"] = mfa_spectrum["alpha"].min()
        features["mfa_alpha_max"] = mfa_spectrum["alpha"].max()
        features["mfa_alpha_range"] = features["mfa_alpha_max"] - features["mfa_alpha_min"]
        
        features["mfa_f_alpha_mean"] = mfa_spectrum["f_alpha"].mean()
        features["mfa_f_alpha_max"] = mfa_spectrum["f_alpha"].max()
        features["mfa_f_alpha_std"] = mfa_spectrum["f_alpha"].std()
        
        features["mfa_tau_mean"] = mfa_spectrum["tau"].mean()
        features["mfa_tau_std"] = mfa_spectrum["tau"].std()
        features["mfa_tau_min"] = mfa_spectrum["tau"].min()
        features["mfa_tau_max"] = mfa_spectrum["tau"].max()
        
        # R2 (quality of fit)
        features["mfa_R2_mean"] = mfa_spectrum["R2"].mean()
        features["mfa_R2_min"] = mfa_spectrum["R2"].min()
        
        # Spectrum width (multifractality measure)
        # Width of f(alpha) spectrum
        valid_f = mfa_spectrum[mfa_spectrum["f_alpha"] > 0]
        if len(valid_f) > 0:
            features["mfa_spectrum_width"] = valid_f["alpha"].max() - valid_f["alpha"].min()
        else:
            features["mfa_spectrum_width"] = 0.0
        
        # Asymmetry: difference between left and right sides of spectrum
        alpha_median = mfa_spectrum["alpha"].median()
        left_side = mfa_spectrum[mfa_spectrum["alpha"] < alpha_median]
        right_side = mfa_spectrum[mfa_spectrum["alpha"] > alpha_median]
        if len(left_side) > 0 and len(right_side) > 0:
            features["mfa_asymmetry"] = (
                right_side["f_alpha"].max() - left_side["f_alpha"].max()
            ) / (features["mfa_f_alpha_max"] + 1e-10)
        else:
            features["mfa_asymmetry"] = 0.0
        
        return features

    def extract_lacunarity_features(self, lacunarity_df: pd.DataFrame) -> dict[str, float]:
        """
        Extract features from Lacunarity curve.
        
        Args:
            lacunarity_df: DataFrame with columns [r, lambda, sigma, mu, cv]
            
        Returns:
            Dictionary of feature names and values
        """
        features = {}
        
        # Basic statistics
        features["lac_lambda_mean"] = lacunarity_df["lambda"].mean()
        features["lac_lambda_std"] = lacunarity_df["lambda"].std()
        features["lac_lambda_min"] = lacunarity_df["lambda"].min()
        features["lac_lambda_max"] = lacunarity_df["lambda"].max()
        features["lac_lambda_range"] = features["lac_lambda_max"] - features["lac_lambda_min"]
        
        # Scale-dependent features
        # Small scale (first 25% of r values)
        n_small = max(1, len(lacunarity_df) // 4)
        small_scale = lacunarity_df.head(n_small)
        features["lac_lambda_small_scale_mean"] = small_scale["lambda"].mean()
        
        # Large scale (last 25% of r values)
        n_large = max(1, len(lacunarity_df) // 4)
        large_scale = lacunarity_df.tail(n_large)
        features["lac_lambda_large_scale_mean"] = large_scale["lambda"].mean()
        
        # Scale dependency (ratio)
        if features["lac_lambda_small_scale_mean"] > 0:
            features["lac_scale_dependency"] = (
                features["lac_lambda_small_scale_mean"] / 
                (features["lac_lambda_large_scale_mean"] + 1e-10)
            )
        else:
            features["lac_scale_dependency"] = 0.0
        
        # Coefficient of variation statistics
        features["lac_cv_mean"] = lacunarity_df["cv"].mean()
        features["lac_cv_std"] = lacunarity_df["cv"].std()
        
        # Power law decay (approximate from log-log slope)
        if len(lacunarity_df) > 2:
            log_r = np.log(lacunarity_df["r"].values + 1e-10)
            log_lambda = np.log(lacunarity_df["lambda"].values + 1e-10)
            # Simple linear regression for slope
            if len(log_r) > 1 and np.std(log_r) > 0:
                slope = np.polyfit(log_r, log_lambda, 1)[0]
                features["lac_power_law_slope"] = slope
            else:
                features["lac_power_law_slope"] = 0.0
        else:
            features["lac_power_law_slope"] = 0.0
        
        return features

    def extract_percolation_features(
        self, percolation_df: pd.DataFrame, percolation_stats: dict[str, Any] | None = None
    ) -> dict[str, float]:
        """
        Extract features from Percolation curve.
        
        Args:
            percolation_df: DataFrame with columns [d, max_cluster_size, n_clusters, giant_fraction]
            percolation_stats: Optional dictionary with precomputed statistics
            
        Returns:
            Dictionary of feature names and values
        """
        features = {}
        
        # Basic statistics
        features["perc_max_cluster_size_mean"] = percolation_df["max_cluster_size"].mean()
        features["perc_max_cluster_size_max"] = percolation_df["max_cluster_size"].max()
        features["perc_max_cluster_size_std"] = percolation_df["max_cluster_size"].std()
        
        features["perc_n_clusters_mean"] = percolation_df["n_clusters"].mean()
        features["perc_n_clusters_max"] = percolation_df["n_clusters"].max()
        features["perc_n_clusters_min"] = percolation_df["n_clusters"].min()
        
        features["perc_giant_fraction_mean"] = percolation_df["giant_fraction"].mean()
        features["perc_giant_fraction_max"] = percolation_df["giant_fraction"].max()
        features["perc_giant_fraction_std"] = percolation_df["giant_fraction"].std()
        
        # Transition features
        # Find where giant fraction crosses 0.5
        d_values = percolation_df["d"].values
        gf_values = percolation_df["giant_fraction"].values
        
        # Critical distance (interpolated)
        critical_d = None
        for i in range(len(gf_values) - 1):
            if gf_values[i] < 0.5 <= gf_values[i + 1]:
                t = (0.5 - gf_values[i]) / (gf_values[i + 1] - gf_values[i])
                critical_d = d_values[i] + t * (d_values[i + 1] - d_values[i])
                break
        
        if critical_d is None:
            if gf_values[-1] < 0.5:
                critical_d = d_values[-1]
            else:
                critical_d = d_values[0]
        
        features["perc_critical_d_50"] = critical_d
        
        # Transition width (10% to 90%)
        d_10 = None
        d_90 = None
        
        for i in range(len(gf_values) - 1):
            if gf_values[i] < 0.1 <= gf_values[i + 1] and d_10 is None:
                t = (0.1 - gf_values[i]) / (gf_values[i + 1] - gf_values[i])
                d_10 = d_values[i] + t * (d_values[i + 1] - d_values[i])
            if gf_values[i] < 0.9 <= gf_values[i + 1] and d_90 is None:
                t = (0.9 - gf_values[i]) / (gf_values[i + 1] - gf_values[i])
                d_90 = d_values[i] + t * (d_values[i + 1] - d_values[i])
        
        if d_10 is None:
            d_10 = d_values[0]
        if d_90 is None:
            d_90 = d_values[-1]
        
        features["perc_transition_width"] = d_90 - d_10
        
        # Use precomputed stats if available
        if percolation_stats:
            features["perc_d_critical_50"] = percolation_stats.get("d_critical_50", critical_d)
            features["perc_d_critical_10"] = percolation_stats.get("d_critical_10", d_10)
            features["perc_d_critical_90"] = percolation_stats.get("d_critical_90", d_90)
            features["perc_transition_width"] = percolation_stats.get("transition_width", features["perc_transition_width"])
            features["perc_max_clusters"] = percolation_stats.get("max_clusters", features["perc_n_clusters_max"])
        
        # Percolation strength (how quickly it transitions)
        # Slope of giant_fraction vs d near critical point
        critical_idx = np.argmin(np.abs(d_values - critical_d))
        window = max(3, len(d_values) // 10)
        start_idx = max(0, critical_idx - window)
        end_idx = min(len(d_values), critical_idx + window)
        
        if end_idx > start_idx + 1:
            d_window = d_values[start_idx:end_idx]
            gf_window = gf_values[start_idx:end_idx]
            if np.std(d_window) > 0:
                slope = np.polyfit(d_window, gf_window, 1)[0]
                features["perc_transition_slope"] = slope
            else:
                features["perc_transition_slope"] = 0.0
        else:
            features["perc_transition_slope"] = 0.0
        
        return features

    def extract_all_features(
        self,
        mfa_spectrum: pd.DataFrame | None = None,
        lacunarity_df: pd.DataFrame | None = None,
        percolation_df: pd.DataFrame | None = None,
        percolation_stats: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """
        Extract all features from available indicators.
        
        Args:
            mfa_spectrum: MFA spectrum DataFrame
            lacunarity_df: Lacunarity curve DataFrame
            percolation_df: Percolation curve DataFrame
            percolation_stats: Percolation statistics dictionary
            
        Returns:
            Combined dictionary of all features
        """
        all_features = {}
        
        if mfa_spectrum is not None:
            mfa_features = self.extract_mfa_features(mfa_spectrum)
            all_features.update(mfa_features)
        
        if lacunarity_df is not None:
            lac_features = self.extract_lacunarity_features(lacunarity_df)
            all_features.update(lac_features)
        
        if percolation_df is not None:
            perc_features = self.extract_percolation_features(percolation_df, percolation_stats)
            all_features.update(perc_features)
        
        return all_features

    def extract_from_output_dir(self, output_dir: Path) -> dict[str, float]:
        """
        Extract features from a run output directory.
        
        Args:
            output_dir: Path to run output directory
            
        Returns:
            Dictionary of features
        """
        mfa_spectrum = None
        lacunarity_df = None
        percolation_df = None
        percolation_stats = None
        
        # Load MFA
        mfa_path = output_dir / "mfa_spectrum.csv"
        if mfa_path.exists():
            mfa_spectrum = pd.read_csv(mfa_path)
        
        # Load Lacunarity
        lac_path = output_dir / "lacunarity.csv"
        if lac_path.exists():
            lacunarity_df = pd.read_csv(lac_path)
        
        # Load Percolation
        perc_path = output_dir / "percolation.csv"
        if perc_path.exists():
            percolation_df = pd.read_csv(perc_path)
        
        # Load Percolation stats
        perc_stats_path = output_dir / "percolation_stats.yaml"
        if perc_stats_path.exists():
            with open(perc_stats_path) as f:
                percolation_stats = yaml.safe_load(f)
        
        return self.extract_all_features(
            mfa_spectrum=mfa_spectrum,
            lacunarity_df=lacunarity_df,
            percolation_df=percolation_df,
            percolation_stats=percolation_stats,
        )

