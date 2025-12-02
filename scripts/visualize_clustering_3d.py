#!/usr/bin/env python
"""Visualize Clustering Results in 3D Space."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.clustering import analyze_clusters


def create_3d_clustering_visualization(
    results_df: pd.DataFrame,
    X_processed: np.ndarray,
    metadata: dict,
    output_path: Path,
):
    """
    Create 3D visualization of clustering results.
    
    Args:
        results_df: DataFrame with clustering results
        X_processed: Preprocessed feature matrix
        metadata: Clustering metadata
    """
    labels = results_df["cluster_label"].values
    sample_ids = results_df["sample_id"].values if "sample_id" in results_df.columns else results_df.index
    
    # Use first 3 dimensions for 3D visualization
    if X_processed.shape[1] >= 3:
        X_3d = X_processed[:, :3]
        # Get dimension names from metadata if available
        prep_meta = metadata.get("preprocessing", {})
        if "dimensionality_reduction" in prep_meta and prep_meta["dimensionality_reduction"] == "pca":
            dim_names = ["PC1", "PC2", "PC3"]
        else:
            dim_names = ["Dim1", "Dim2", "Dim3"]
    else:
        # If less than 3 dimensions, pad with zeros
        X_3d = np.zeros((X_processed.shape[0], 3))
        X_3d[:, :X_processed.shape[1]] = X_processed
        dim_names = [f"Dim{i+1}" for i in range(3)]
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Color map for clusters
    unique_labels = sorted([l for l in np.unique(labels) if l != -1])
    n_clusters = len(unique_labels)
    
    # Use distinct colors
    if n_clusters <= 10:
        colors = px.colors.qualitative.Set3[:n_clusters]
    else:
        colors = px.colors.qualitative.Set3 * ((n_clusters // 10) + 1)
    
    # Plot each cluster
    for i, label in enumerate(unique_labels):
        mask = labels == label
        cluster_samples = [sample_ids[j] for j in range(len(sample_ids)) if mask[j]]
        
        fig.add_trace(go.Scatter3d(
            x=X_3d[mask, 0],
            y=X_3d[mask, 1],
            z=X_3d[mask, 2],
            mode="markers",
            name=f"Cluster {int(label)} ({np.sum(mask)} samples)",
            marker=dict(
                size=12,
                color=colors[i % len(colors)],
                opacity=0.8,
                line=dict(width=1, color="white"),
            ),
            text=cluster_samples,
            hovertemplate="<b>%{text}</b><br>" +
                        f"{dim_names[0]}: %{{x:.3f}}<br>" +
                        f"{dim_names[1]}: %{{y:.3f}}<br>" +
                        f"{dim_names[2]}: %{{z:.3f}}<br>" +
                        f"Cluster: {int(label)}<extra></extra>",
        ))
    
    # Plot noise points if any
    if -1 in labels:
        mask = labels == -1
        noise_samples = [sample_ids[j] for j in range(len(sample_ids)) if mask[j]]
        fig.add_trace(go.Scatter3d(
            x=X_3d[mask, 0],
            y=X_3d[mask, 1],
            z=X_3d[mask, 2],
            mode="markers",
            name=f"Noise ({np.sum(mask)} samples)",
            marker=dict(
                size=10,
                color="gray",
                opacity=0.5,
                symbol="x",
                line=dict(width=1, color="white"),
            ),
            text=noise_samples,
            hovertemplate="<b>%{text}</b><br>" +
                        f"{dim_names[0]}: %{{x:.3f}}<br>" +
                        f"{dim_names[1]}: %{{y:.3f}}<br>" +
                        f"{dim_names[2]}: %{{z:.3f}}<br>" +
                        "Cluster: Noise<extra></extra>",
        ))
    
    # Get explained variance if PCA was used
    prep_meta = metadata.get("preprocessing", {})
    variance_info = ""
    if "explained_variance_ratio" in prep_meta:
        var_ratio = prep_meta["explained_variance_ratio"]
        if var_ratio and len(var_ratio) >= 3:
            cum_var = sum(var_ratio[:3])
            variance_info = f" (Explained Variance: {cum_var:.1%})"
    
    # Update layout
    fig.update_layout(
        title=f"Clustering Results - 3D Visualization{variance_info}",
        scene=dict(
            xaxis_title=dim_names[0],
            yaxis_title=dim_names[1],
            zaxis_title=dim_names[2],
            bgcolor="#1a1a2e",
            xaxis=dict(backgroundcolor="#16213e", gridcolor="#2a2a4e"),
            yaxis=dict(backgroundcolor="#16213e", gridcolor="#2a2a4e"),
            zaxis=dict(backgroundcolor="#16213e", gridcolor="#2a2a4e"),
        ),
        template="plotly_dark",
        height=800,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="white",
            borderwidth=1,
        ),
    )
    
    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "clustering_3d.html"
    fig.write_html(str(output_file))
    
    print(f"3D visualization saved to: {output_file}")
    print(f"  - {n_clusters} clusters visualized in 3D space")
    if -1 in labels:
        print(f"  - {np.sum(labels == -1)} noise points")


def main():
    """Create 3D visualization of clustering results."""
    parser = argparse.ArgumentParser(
        description="Visualize clustering results in 3D space"
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing run output directories",
    )
    parser.add_argument(
        "--clustering-results-dir",
        type=Path,
        default=Path("outputs/clustering_results"),
        help="Directory containing existing clustering results",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/clustering.yaml"),
        help="Clustering configuration file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/clustering_results"),
        help="Output directory for visualization",
    )
    parser.add_argument(
        "--use-best-config",
        action="store_true",
        help="Use best configuration from comparison results",
    )

    args = parser.parse_args()

    # Check if clustering results already exist
    results_csv = args.clustering_results_dir / "clustering_results.csv"
    processed_features_npy = args.clustering_results_dir / "processed_features.npy"
    metadata_yaml = args.clustering_results_dir / "clustering_metadata.yaml"
    
    if results_csv.exists() and processed_features_npy.exists() and metadata_yaml.exists():
        # Load existing results
        print("Loading existing clustering results...")
        results_df = pd.read_csv(results_csv)
        X_processed = np.load(processed_features_npy)
        
        with open(metadata_yaml) as f:
            metadata = yaml.safe_load(f)
        
        print(f"Loaded: {len(results_df)} samples, {X_processed.shape[1]} dimensions")
    else:
        # Run clustering first
        print("Running clustering analysis...")
        
        # Find output directories
        output_dirs = [
            d
            for d in args.outputs_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ]
        
        if len(output_dirs) == 0:
            raise ValueError(f"No output directories found in {args.outputs_dir}")
        
        # Load config
        if args.use_best_config:
            # Load best config from comparison results
            comparison_summary = Path("outputs/clustering_comparison/comparison_summary.yaml")
            if comparison_summary.exists():
                with open(comparison_summary) as f:
                    summary = yaml.safe_load(f)
                best = summary.get("best_silhouette", {})
                if best:
                    preprocessing_config = {
                        "normalization_method": best.get("normalization", "robust"),
                        "dimensionality_reduction": best.get("dimensionality_reduction", "pca"),
                        "n_components": None,
                        "random_state": 42,
                    }
                    clustering_config = {
                        "method": best.get("clustering_method", "kmeans"),
                        "n_clusters": int(best.get("n_clusters", 2)),
                        "random_state": 42,
                    }
                    if best.get("clustering_method") == "hierarchical":
                        clustering_config["linkage"] = best.get("linkage", "ward")
                    print(f"Using best configuration: {best.get('config_id')}")
                else:
                    raise ValueError("Best configuration not found in comparison results")
            else:
                raise ValueError("Comparison results not found. Run compare_clustering_methods.py first.")
        else:
            # Load from config file
            with open(args.config) as f:
                config = yaml.safe_load(f)
            
            preprocessing_config = config.get("preprocessing", {})
            clustering_config = config.get("clustering", {})
        
        results_df, X_processed, metadata = analyze_clusters(
            output_dirs,
            preprocessor_config=preprocessing_config,
            clustering_config=clustering_config,
        )
        
        # Save results
        args.output.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.output / "clustering_results.csv", index=False)
        np.save(args.output / "processed_features.npy", X_processed)
        with open(args.output / "clustering_metadata.yaml", "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
    
    # Create 3D visualization
    create_3d_clustering_visualization(
        results_df,
        X_processed,
        metadata,
        args.output,
    )


if __name__ == "__main__":
    main()

