#!/usr/bin/env python
"""Compare Multiple Clustering Methods and Preprocessing Approaches."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.clustering import ClusteringAnalyzer, analyze_clusters
from src.analysis.clustering_evaluation import evaluate_clustering


def compare_methods(
    output_dirs: list[Path],
    output_path: Path,
    min_samples: int = 3,
) -> pd.DataFrame:
    """
    Compare multiple clustering methods and preprocessing approaches.
    
    Args:
        output_dirs: List of output directory paths
        output_path: Path to save comparison results
        min_samples: Minimum number of samples required
        
    Returns:
        DataFrame with comparison results
    """
    if len(output_dirs) < min_samples:
        raise ValueError(
            f"Not enough samples: found {len(output_dirs)}, "
            f"required at least {min_samples}"
        )
    
    # Define configurations to test
    normalization_methods = ["robust", "standard", "minmax"]
    dimensionality_reductions = ["pca", None]  # Add "umap" if available
    clustering_methods = ["kmeans", "dbscan", "hierarchical"]
    
    # Auto-select cluster count
    n_samples = len(output_dirs)
    auto_n_clusters = max(2, int(np.sqrt(n_samples / 2)))
    
    results = []
    
    print("=" * 80)
    print("Clustering Methods Comparison")
    print("=" * 80)
    print(f"Total samples: {n_samples}")
    print(f"Auto-selected n_clusters: {auto_n_clusters}")
    print(f"Testing {len(normalization_methods)} normalization methods")
    print(f"Testing {len(dimensionality_reductions)} dimensionality reduction methods")
    print(f"Testing {len(clustering_methods)} clustering methods")
    print(f"Total combinations: {len(normalization_methods) * len(dimensionality_reductions) * len(clustering_methods)}")
    print("=" * 80)
    
    total_combinations = (
        len(normalization_methods)
        * len(dimensionality_reductions)
        * len(clustering_methods)
    )
    
    with tqdm(total=total_combinations, desc="Comparing methods") as pbar:
        for norm_method in normalization_methods:
            for dim_reduction in dimensionality_reductions:
                # Skip UMAP if not available
                if dim_reduction == "umap":
                    try:
                        import umap
                    except ImportError:
                        pbar.update(1)
                        continue
                
                for cluster_method in clustering_methods:
                    config_id = f"{norm_method}_{dim_reduction or 'none'}_{cluster_method}"
                    pbar.set_description(f"Testing {config_id}")
                    
                    try:
                        # Setup preprocessing config
                        preprocessor_config = {
                            "normalization_method": norm_method,
                            "dimensionality_reduction": dim_reduction,
                            "n_components": None,  # Auto-select
                            "random_state": 42,
                        }
                        
                        # Setup clustering config
                        clustering_config = {
                            "method": cluster_method,
                            "random_state": 42,
                        }
                        
                        # Set method-specific parameters
                        if cluster_method == "kmeans":
                            clustering_config["n_clusters"] = auto_n_clusters
                        elif cluster_method == "dbscan":
                            clustering_config["eps"] = 0.5
                            clustering_config["min_samples"] = 3
                        elif cluster_method == "hierarchical":
                            clustering_config["n_clusters"] = auto_n_clusters
                            clustering_config["linkage"] = "ward"
                        
                        # Run clustering
                        results_df, X_processed, metadata = analyze_clusters(
                            output_dirs,
                            preprocessor_config=preprocessor_config,
                            clustering_config=clustering_config,
                        )
                        
                        # Evaluate clustering
                        labels = results_df["cluster_label"].values
                        evaluation = evaluate_clustering(X_processed, labels)
                        
                        # Extract preprocessing metadata
                        prep_meta = metadata.get("preprocessing", {})
                        cluster_meta = metadata.get("clustering", {})
                        
                        # Calculate explained variance if PCA was used
                        explained_variance = None
                        if dim_reduction == "pca" and "explained_variance_ratio" in prep_meta:
                            var_ratio = prep_meta["explained_variance_ratio"]
                            if var_ratio:
                                explained_variance = sum(var_ratio)
                        
                        # Store results
                        result = {
                            "config_id": config_id,
                            "normalization": norm_method,
                            "dimensionality_reduction": dim_reduction or "none",
                            "clustering_method": cluster_method,
                            "n_clusters": evaluation["n_clusters"],
                            "n_noise": evaluation["n_noise"],
                            "silhouette_score": evaluation["silhouette_score"],
                            "davies_bouldin_score": evaluation["davies_bouldin_score"],
                            "calinski_harabasz_score": evaluation["calinski_harabasz_score"],
                            "n_features_original": metadata.get("n_features_original", 0),
                            "n_features_processed": metadata.get("n_features_processed", 0),
                            "explained_variance": explained_variance,
                            "status": "success",
                        }
                        
                        # Add method-specific metrics
                        if cluster_method == "kmeans" and "inertia" in cluster_meta:
                            result["inertia"] = cluster_meta["inertia"]
                        
                        results.append(result)
                        
                    except Exception as e:
                        # Store failure
                        results.append({
                            "config_id": config_id,
                            "normalization": norm_method,
                            "dimensionality_reduction": dim_reduction or "none",
                            "clustering_method": cluster_method,
                            "status": "failed",
                            "error": str(e)[:200],  # Truncate error message
                        })
                    
                    pbar.update(1)
    
    # Create results DataFrame
    comparison_df = pd.DataFrame(results)
    
    # Save results
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path / "comparison_results.csv", index=False)
    
    # Save summary
    successful = comparison_df[comparison_df["status"] == "success"]
    if len(successful) > 0:
        # Sort by silhouette score (higher is better)
        best_silhouette = successful.nlargest(1, "silhouette_score")
        
        # Sort by Davies-Bouldin score (lower is better)
        best_db = successful.nsmallest(1, "davies_bouldin_score")
        
        summary = {
            "total_combinations": len(comparison_df),
            "successful": len(successful),
            "failed": len(comparison_df) - len(successful),
            "best_silhouette": best_silhouette.iloc[0].to_dict() if len(best_silhouette) > 0 else None,
            "best_davies_bouldin": best_db.iloc[0].to_dict() if len(best_db) > 0 else None,
        }
        
        with open(output_path / "comparison_summary.yaml", "w") as f:
            yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)
        
        # Print summary
        print("\n" + "=" * 80)
        print("Comparison Summary")
        print("=" * 80)
        print(f"Total combinations tested: {len(comparison_df)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(comparison_df) - len(successful)}")
        
        if len(best_silhouette) > 0:
            best = best_silhouette.iloc[0]
            print(f"\nBest configuration (by Silhouette Score):")
            print(f"  Config ID: {best['config_id']}")
            print(f"  Normalization: {best['normalization']}")
            print(f"  Dimensionality Reduction: {best['dimensionality_reduction']}")
            print(f"  Clustering Method: {best['clustering_method']}")
            print(f"  Silhouette Score: {best['silhouette_score']:.4f}")
            print(f"  Davies-Bouldin Score: {best['davies_bouldin_score']:.4f}")
            print(f"  Calinski-Harabasz Score: {best['calinski_harabasz_score']:.2f}")
            print(f"  Number of Clusters: {best['n_clusters']}")
        
        print("=" * 80)
    
    return comparison_df


def main():
    """Run clustering methods comparison."""
    parser = argparse.ArgumentParser(
        description="Compare multiple clustering methods and preprocessing approaches"
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing run output directories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/clustering_comparison"),
        help="Output directory for comparison results",
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

    # Run comparison
    comparison_df = compare_methods(
        output_dirs,
        args.output,
        min_samples=args.min_samples,
    )

    print(f"\nResults saved to: {args.output}")
    print(f"  - comparison_results.csv: Full comparison results")
    print(f"  - comparison_summary.yaml: Summary with best configurations")


if __name__ == "__main__":
    main()

