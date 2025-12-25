#!/usr/bin/env python
"""Launch Cluster and Image Correspondence View Dashboard."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.clustering import ClusteringAnalyzer, FeatureExtractor, create_feature_matrix
from src.visualization.cluster_image_view import create_cluster_image_correspondence_view
from src.visualization.dashboard import load_results


def load_clustering_results(outputs_dir: Path) -> dict[str, dict]:
    """Load all analysis results for clustering."""
    from src.visualization.dashboard import load_results
    
    site_results = {}
    
    run_dirs = [
        d
        for d in outputs_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]
    
    for run_dir in run_dirs:
        try:
            results = load_results(run_dir)
            
            # Extract site ID
            site_id = run_dir.name
            if "config" in results:
                config = results["config"]
                site_metadata = config.get("site_metadata", {})
                site_id = site_metadata.get("meta_info", {}).get("site_id", run_dir.name)
            
            # Extract DataFrames
            site_data = {}
            if "mfa_spectrum" in results:
                site_data["mfa"] = results["mfa_spectrum"]
            if "lacunarity" in results:
                site_data["lacunarity"] = results["lacunarity"]
            if "percolation" in results:
                site_data["percolation"] = results["percolation"]
            
            if site_data:
                site_results[site_id] = site_data
        except Exception as e:
            print(f"Warning: Could not load {run_dir}: {e}")
            continue
    
    return site_results


def main():
    """Launch the Cluster and Image Correspondence View."""
    parser = argparse.ArgumentParser(
        description="Launch Cluster & Image Correspondence Dashboard"
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to outputs directory containing run directories",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8052,
        help="Port to run dashboard on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (use 0.0.0.0 for external access)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--clustering-config",
        type=Path,
        default=None,
        help="Path to saved clustering results JSON (optional)",
    )

    args = parser.parse_args()

    # Load config
    config = {}
    if args.config.exists():
        with open(args.config) as f:
            full_config = yaml.safe_load(f)
            clustering_config = full_config.get("analysis", {}).get("clustering", {})
            config = full_config.get("visualization", {}).get("cluster_image_view", {})

    # Load or compute clustering results
    if args.clustering_config and args.clustering_config.exists():
        # Load saved clustering results
        print(f"Loading clustering results from {args.clustering_config}")
        with open(args.clustering_config) as f:
            clustering_data = json.load(f)
        
        cluster_results = {
            "coordinates": np.array(clustering_data["coordinates"]),
            "labels": np.array(clustering_data["labels"]),
        }
        if "cluster_centers" in clustering_data:
            cluster_results["cluster_centers"] = np.array(clustering_data["cluster_centers"])
        
        # Get results directories in same order
        run_dirs = [
            Path(args.outputs_dir) / run_id
            for run_id in clustering_data.get("run_ids", [])
        ]
    else:
        # Compute clustering from results
        print("Loading analysis results...")
        site_results = load_clustering_results(args.outputs_dir)
        
        if not site_results:
            print(f"Error: No analysis results found in {args.outputs_dir}")
            return 1
        
        print(f"Found {len(site_results)} sites")
        
        # Extract features
        print("Extracting features...")
        extractor = FeatureExtractor(
            mfa_q_values=clustering_config.get("feature_extraction", {}).get("mfa_q_values", [0, 1, 2]),
            lac_scales=clustering_config.get("feature_extraction", {}).get("lacunarity_scales", [4, 16, 64]),
            perc_fractions=clustering_config.get("feature_extraction", {}).get("percolation_fractions", [0.1, 0.5, 0.9]),
        )
        
        feature_df = create_feature_matrix(site_results, extractor)
        print(f"Feature matrix shape: {feature_df.shape}")
        
        # Perform clustering
        print("Performing clustering...")
        analyzer = ClusteringAnalyzer(
            method=clustering_config.get("method", "kmeans"),
            n_clusters=clustering_config.get("n_clusters", 5),
            normalization=clustering_config.get("normalization", "standard"),
            dimension_reduction=clustering_config.get("dimension_reduction", "none"),
            pca_n_components=clustering_config.get("pca_n_components", 3),
        )
        
        labels = analyzer.fit_predict(feature_df.values)
        
        # Get coordinates (use first 2 PCA components or first 2 features)
        if analyzer.dimension_reduction == "pca" and hasattr(analyzer, "pca_"):
            coordinates = analyzer.pca_.transform(feature_df.values)[:, :2]
        else:
            # Use first 2 features
            coordinates = feature_df.values[:, :2]
        
        # Calculate cluster centers
        n_clusters = len(set(labels))
        cluster_centers = []
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_centers.append(coordinates[mask].mean(axis=0))
        
        cluster_results = {
            "coordinates": coordinates,
            "labels": labels,
            "cluster_centers": np.array(cluster_centers),
        }
        
        # Get results directories in same order as site_results
        run_dirs = []
        for site_id in feature_df.index:
            # Find matching run directory
            for run_dir in args.outputs_dir.iterdir():
                if run_dir.is_dir() and run_dir.name.startswith("run_"):
                    results = load_results(run_dir)
                    if "config" in results:
                        config_data = results["config"]
                        site_metadata = config_data.get("site_metadata", {})
                        result_site_id = site_metadata.get("meta_info", {}).get("site_id", run_dir.name)
                        if result_site_id == site_id:
                            run_dirs.append(run_dir)
                            break
            else:
                # Fallback: use first run directory
                run_dirs.append(list(args.outputs_dir.iterdir())[0])

    print("=" * 60)
    print("Cluster & Image Correspondence Dashboard")
    print("=" * 60)
    print(f"Outputs directory: {args.outputs_dir}")
    print(f"Number of points: {len(cluster_results['coordinates'])}")
    print(f"Number of clusters: {len(set(cluster_results['labels']))}")
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 60)
    print()

    # Create and run dashboard
    app = create_cluster_image_correspondence_view(
        cluster_results=cluster_results,
        results_dirs=run_dirs,
        config=config,
    )

    print(f"Starting dashboard at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

