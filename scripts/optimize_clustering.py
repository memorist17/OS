#!/usr/bin/env python
"""Optimize Clustering Parameters using Elbow Method and Grid Search."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.clustering import analyze_clusters
from src.analysis.clustering_optimization import ClusteringOptimizer
from src.analysis.clustering_preprocessing import ClusteringPreprocessor, prepare_clustering_data
from src.analysis.feature_extraction import FeatureExtractor


def interpret_clusters(
    results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Interpret clustering results by computing statistics for each cluster.
    
    Args:
        results_df: DataFrame with features and cluster labels
        output_path: Path to save interpretation results
    """
    # Separate feature columns from metadata
    feature_cols = [
        col for col in results_df.columns
        if col not in ["cluster_label", "sample_id"]
    ]
    
    # Compute cluster statistics
    cluster_stats = []
    
    for cluster_id in sorted(results_df["cluster_label"].unique()):
        if cluster_id == -1:
            continue  # Skip noise points
        
        cluster_data = results_df[results_df["cluster_label"] == cluster_id]
        
        stats = {
            "cluster_id": int(cluster_id),
            "n_samples": len(cluster_data),
            "sample_ids": cluster_data["sample_id"].tolist(),
        }
        
        # Compute statistics for each feature
        for col in feature_cols:
            if col in cluster_data.columns:
                values = cluster_data[col].values
                stats[f"{col}_mean"] = float(np.mean(values))
                stats[f"{col}_std"] = float(np.std(values))
                stats[f"{col}_min"] = float(np.min(values))
                stats[f"{col}_max"] = float(np.max(values))
        
        cluster_stats.append(stats)
    
    # Save interpretation
    interpretation_df = pd.DataFrame(cluster_stats)
    interpretation_df.to_csv(output_path / "cluster_interpretation.csv", index=False)
    
    # Save summary
    summary = {
        "n_clusters": len(cluster_stats),
        "total_samples": len(results_df),
        "cluster_sizes": {
            int(row["cluster_id"]): int(row["n_samples"])
            for _, row in interpretation_df.iterrows()
        },
    }
    
    with open(output_path / "cluster_interpretation_summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\nCluster Interpretation:")
    print(f"  Number of clusters: {len(cluster_stats)}")
    for _, row in interpretation_df.iterrows():
        print(f"  Cluster {int(row['cluster_id'])}: {int(row['n_samples'])} samples")


def main():
    """Optimize clustering parameters."""
    parser = argparse.ArgumentParser(
        description="Optimize clustering parameters using elbow method"
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
        help="Config file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/clustering_optimization"),
        help="Output directory for optimization results",
    )
    parser.add_argument(
        "--run-ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific run IDs to include",
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Skip optimization and use config values directly",
    )

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    clustering_config = config.get("clustering", {})
    preprocessor_config = clustering_config.get("preprocessing", {})
    optimization_config = clustering_config.get("optimization", {})
    clustering_method_config = {
        k: v
        for k, v in clustering_config.items()
        if k not in ["preprocessing", "optimization"]
    }

    # Find output directories
    if not args.outputs_dir.exists():
        raise ValueError(f"Outputs directory not found: {args.outputs_dir}")

    if args.run_ids:
        output_dirs = [
            args.outputs_dir / run_id
            for run_id in args.run_ids
            if (args.outputs_dir / run_id).exists()
        ]
    else:
        output_dirs = [
            d
            for d in args.outputs_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ]

    if len(output_dirs) < 3:
        raise ValueError(f"Not enough samples: found {len(output_dirs)}, required at least 3")

    print("=" * 80)
    print("Clustering Parameter Optimization")
    print("=" * 80)
    print(f"Total samples: {len(output_dirs)}")
    print(f"Config file: {args.config}")
    print("=" * 80)

    # Prepare data
    feature_extractor = FeatureExtractor()
    preprocessor = ClusteringPreprocessor(**preprocessor_config)
    
    from src.analysis.clustering_preprocessing import prepare_clustering_data
    features_df, X_processed, prep_metadata = prepare_clustering_data(
        output_dirs,
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
    )

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Optimization
    if not args.skip_optimization and optimization_config.get("use_elbow_method", True):
        print("\n" + "=" * 80)
        print("Optimizing Clustering Parameters")
        print("=" * 80)
        
        optimizer = ClusteringOptimizer(
            min_clusters=optimization_config.get("min_clusters", 2),
            max_clusters=optimization_config.get("max_clusters", 10),
            random_state=preprocessor_config.get("random_state", 42),
        )
        
        method = clustering_method_config.get("method", "kmeans")
        
        if method == "kmeans":
            # Elbow method for K-means
            elbow_metric = optimization_config.get("elbow_metric", "silhouette")
            print(f"Running elbow method for K-means (metric: {elbow_metric})...")
            elbow_results, optimal_k = optimizer.elbow_method_kmeans(X_processed, metric=elbow_metric)
            
            # Save elbow results
            elbow_df = pd.DataFrame([
                {"n_clusters": k, **metrics}
                for k, metrics in elbow_results.items()
            ])
            elbow_df.to_csv(args.output / "elbow_method_results.csv", index=False)
            
            print(f"\nElbow Method Results:")
            print(elbow_df.to_string(index=False))
            
            if optimal_k:
                print(f"\nOptimal number of clusters: {optimal_k}")
                clustering_method_config["n_clusters"] = optimal_k
            else:
                # Use best silhouette score
                best_row = elbow_df.loc[elbow_df["silhouette_score"].idxmax()]
                optimal_k = int(best_row["n_clusters"])
                print(f"\nUsing best silhouette score: {optimal_k} clusters")
                clustering_method_config["n_clusters"] = optimal_k
        
        elif method == "dbscan":
            # Grid search for DBSCAN
            print("Running grid search for DBSCAN...")
            optimizer.eps_range = optimization_config.get("dbscan_eps_range", optimizer.eps_range)
            optimizer.min_samples_range = optimization_config.get("dbscan_min_samples_range", optimizer.min_samples_range)
            
            dbscan_results, optimal_params = optimizer.optimize_dbscan(X_processed)
            
            # Save DBSCAN results
            dbscan_df = pd.DataFrame([
                {"eps": eps, "min_samples": min_s, **metrics}
                for (eps, min_s), metrics in dbscan_results.items()
            ])
            dbscan_df.to_csv(args.output / "dbscan_optimization_results.csv", index=False)
            
            print(f"\nDBSCAN Optimization Results:")
            print(dbscan_df[dbscan_df["n_clusters"] >= 2].to_string(index=False))
            
            if optimal_params:
                eps, min_samples = optimal_params
                print(f"\nOptimal parameters: eps={eps}, min_samples={min_samples}")
                clustering_method_config["eps"] = eps
                clustering_method_config["min_samples"] = min_samples
            else:
                print("\nWarning: No valid clusters found with any parameter combination")
        
        # Compare multiple cluster numbers if requested
        if optimization_config.get("compare_cluster_numbers", True) and method in ["kmeans", "hierarchical"]:
            print(f"\nComparing multiple cluster numbers for {method}...")
            comparison_df = optimizer.compare_cluster_numbers(X_processed, method=method)
            comparison_df.to_csv(args.output / "cluster_number_comparison.csv", index=False)
            
            print(f"\nCluster Number Comparison:")
            print(comparison_df.to_string(index=False))
    
    # Run final clustering with optimized parameters
    print("\n" + "=" * 80)
    print("Running Final Clustering")
    print("=" * 80)
    
    results_df, X_processed_final, metadata = analyze_clusters(
        output_dirs,
        preprocessor_config=preprocessor_config,
        clustering_config=clustering_method_config,
    )
    
    # Save results
    results_df.to_csv(args.output / "optimized_clustering_results.csv", index=False)
    np.save(args.output / "processed_features.npy", X_processed_final)
    
    # Save metadata
    with open(args.output / "optimization_metadata.yaml", "w") as f:
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
    
    # Interpret results if requested
    if optimization_config.get("interpret_results", True):
        print("\n" + "=" * 80)
        print("Interpreting Clustering Results")
        print("=" * 80)
        interpret_clusters(results_df, args.output)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Optimization Complete!")
    print("=" * 80)
    print(f"Output directory: {args.output}")
    print(f"\nFinal configuration:")
    print(f"  Method: {clustering_method_config.get('method', 'kmeans')}")
    print(f"  N clusters: {clustering_method_config.get('n_clusters', 'auto')}")
    if clustering_method_config.get("method") == "dbscan":
        print(f"  Eps: {clustering_method_config.get('eps', 0.5)}")
        print(f"  Min samples: {clustering_method_config.get('min_samples', 3)}")
    
    cluster_counts = results_df["cluster_label"].value_counts().sort_index()
    print(f"\nCluster distribution:")
    for label, count in cluster_counts.items():
        if label == -1:
            print(f"  Noise: {count} samples")
        else:
            print(f"  Cluster {label}: {count} samples")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

