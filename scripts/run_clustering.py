#!/usr/bin/env python
"""Clustering Analysis Script for Multiple Places."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.clustering import analyze_clusters


def main():
    """Run clustering analysis on multiple output directories."""
    parser = argparse.ArgumentParser(
        description="Cluster urban structure indicators from multiple places"
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing run output directories",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Main config file path (default.yaml)",
    )
    parser.add_argument(
        "--clustering-config",
        type=Path,
        default=Path("configs/clustering.yaml"),
        help="Clustering-specific config file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/clustering_results"),
        help="Output directory for clustering results",
    )
    parser.add_argument(
        "--run-ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific run IDs to include (if not provided, uses all)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum number of samples required for clustering",
    )

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Try to load clustering-specific config
    clustering_config_path = Path("configs/clustering.yaml")
    if clustering_config_path.exists():
        with open(clustering_config_path) as f:
            clustering_full_config = yaml.safe_load(f)
        
        # Use clustering.yaml if available, otherwise fall back to default.yaml
        preprocessing_config = clustering_full_config.get("preprocessing", {})
        clustering_method_config = clustering_full_config.get("clustering", {})
        
        # Override with command line or default.yaml if needed
        if not preprocessing_config:
            clustering_config = config.get("clustering", {})
            preprocessing_config = clustering_config.get("preprocessing", {})
            clustering_method_config = {
                k: v
                for k, v in clustering_config.items()
                if k != "preprocessing"
            }
    else:
        # Fall back to default.yaml
        clustering_config = config.get("clustering", {})
        preprocessor_config = clustering_config.get("preprocessing", {})
        clustering_method_config = {
            k: v
            for k, v in clustering_config.items()
            if k != "preprocessing"
        }

    # Find output directories
    if not args.outputs_dir.exists():
        raise ValueError(f"Outputs directory not found: {args.outputs_dir}")

    if args.run_ids:
        # Use specified run IDs
        output_dirs = [
            args.outputs_dir / run_id
            for run_id in args.run_ids
            if (args.outputs_dir / run_id).exists()
        ]
    else:
        # Find all run directories
        output_dirs = [
            d
            for d in args.outputs_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ]

    if len(output_dirs) < args.min_samples:
        raise ValueError(
            f"Not enough samples: found {len(output_dirs)}, "
            f"required at least {args.min_samples}"
        )

    print("=" * 60)
    print("Clustering Analysis")
    print("=" * 60)
    print(f"Output directories: {len(output_dirs)}")
    print(f"Preprocessing: {preprocessor_config.get('normalization_method', 'robust')} + "
          f"{preprocessor_config.get('dimensionality_reduction', 'pca')}")
    print(f"Clustering method: {clustering_method_config.get('method', 'kmeans')}")
    print("=" * 60)

    # Run clustering
    try:
        results_df, X_processed, metadata = analyze_clusters(
            output_dirs,
            preprocessor_config=preprocessor_config,
            clustering_config=clustering_method_config,
        )
    except Exception as e:
        print(f"Error during clustering: {e}")
        raise

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Save results
    results_df.to_csv(args.output / "clustering_results.csv", index=False)
    np.save(args.output / "processed_features.npy", X_processed)

    # Save metadata
    with open(args.output / "clustering_metadata.yaml", "w") as f:
        # Convert numpy types to native Python types for YAML
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj

        metadata_clean = convert_to_native(metadata)
        yaml.dump(metadata_clean, f, default_flow_style=False, allow_unicode=True)

    # Print summary
    print("\n" + "=" * 60)
    print("Clustering Complete!")
    print("=" * 60)
    print(f"Output directory: {args.output}")
    print(f"\nCluster distribution:")
    cluster_counts = results_df["cluster_label"].value_counts().sort_index()
    for label, count in cluster_counts.items():
        if label == -1:
            print(f"  Noise: {count} samples")
        else:
            print(f"  Cluster {label}: {count} samples")

    if "clustering" in metadata and "cluster_stats" in metadata["clustering"]:
        print(f"\nCluster statistics:")
        for label, stats in metadata["clustering"]["cluster_stats"].items():
            print(f"  Cluster {label}: {stats['size']} samples ({stats['ratio']*100:.1f}%)")

    if "preprocessing" in metadata:
        prep_meta = metadata["preprocessing"]
        if "explained_variance_ratio" in prep_meta:
            var_ratio = prep_meta["explained_variance_ratio"]
            if var_ratio:
                print(f"\nDimensionality reduction:")
                print(f"  Components: {prep_meta['n_components']}")
                print(f"  Explained variance: {sum(var_ratio):.2%}")

    print("=" * 60)


if __name__ == "__main__":
    main()

