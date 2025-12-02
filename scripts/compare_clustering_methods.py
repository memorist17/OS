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
from src.analysis.clustering_optimization import (
    DBSCANParameterSearch,
    ElbowMethod,
    optimize_cluster_count,
)


def load_clustering_config(config_path: Path) -> dict[str, Any]:
    """Load clustering configuration from YAML file."""
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config
    else:
        # Return default config
        return {
            "preprocessing": {
                "normalization_options": ["robust", "standard", "minmax"],
                "dimensionality_reduction_options": ["pca", None],
            },
            "clustering": {
                "methods_to_compare": ["kmeans", "dbscan", "hierarchical"],
                "dbscan": {
                    "eps_range": [0.5],
                    "min_samples_range": [3],
                },
            },
            "cluster_optimization": {
                "use_elbow_method": False,
                "compare_multiple_clusters": False,
                "cluster_range": None,
            },
        }


def compare_methods(
    output_dirs: list[Path],
    output_path: Path,
    config: dict[str, Any],
    min_samples: int = 3,
) -> pd.DataFrame:
    """
    Compare multiple clustering methods and preprocessing approaches.
    
    Args:
        output_dirs: List of output directory paths
        output_path: Path to save comparison results
        config: Clustering configuration dictionary
        min_samples: Minimum number of samples required
        
    Returns:
        DataFrame with comparison results
    """
    if len(output_dirs) < min_samples:
        raise ValueError(
            f"Not enough samples: found {len(output_dirs)}, "
            f"required at least {min_samples}"
        )
    
    # Load configuration
    preprocessing_config = config.get("preprocessing", {})
    clustering_config = config.get("clustering", {})
    optimization_config = config.get("cluster_optimization", {})
    
    normalization_methods = preprocessing_config.get(
        "normalization_options", ["robust", "standard", "minmax"]
    )
    dimensionality_reductions = preprocessing_config.get(
        "dimensionality_reduction_options", ["pca", None]
    )
    clustering_methods = clustering_config.get(
        "methods_to_compare", ["kmeans", "dbscan", "hierarchical"]
    )
    
    n_samples = len(output_dirs)
    
    # Determine cluster count range
    use_elbow = optimization_config.get("use_elbow_method", False)
    compare_multiple = optimization_config.get("compare_multiple_clusters", False)
    cluster_range = optimization_config.get("cluster_range", None)
    
    if cluster_range is None:
        # Auto-select based on sample size
        max_clusters = max(2, int(n_samples * 0.3))
        cluster_range = list(range(2, min(max_clusters + 1, n_samples)))
    
    # For initial preprocessing to find optimal k
    print("=" * 80)
    print("Clustering Methods Comparison")
    print("=" * 80)
    print(f"Total samples: {n_samples}")
    print(f"Normalization methods: {normalization_methods}")
    print(f"Dimensionality reductions: {dimensionality_reductions}")
    print(f"Clustering methods: {clustering_methods}")
    print(f"Cluster optimization: elbow={use_elbow}, compare_multiple={compare_multiple}")
    if compare_multiple:
        print(f"Cluster range: {cluster_range}")
    print("=" * 80)
    
    results = []
    
    # First, prepare data once for elbow method if needed
    optimal_k_per_config = {}
    
    if use_elbow or compare_multiple:
        print("\nPreparing data for cluster optimization...")
        from src.analysis.clustering_preprocessing import ClusteringPreprocessor, prepare_clustering_data
        from src.analysis.feature_extraction import FeatureExtractor
        
        feature_extractor = FeatureExtractor()
        # Use default preprocessing for elbow method
        default_preprocessor = ClusteringPreprocessor(
            normalization_method=preprocessing_config.get("normalization_method", "minmax"),
            dimensionality_reduction=preprocessing_config.get("dimensionality_reduction", "pca"),
        )
        
        _, X_processed, _ = prepare_clustering_data(
            output_dirs,
            preprocessor=default_preprocessor,
            feature_extractor=feature_extractor,
        )
        
        if use_elbow:
            print("Finding optimal cluster count using elbow method...")
            elbow = ElbowMethod(
                k_range=cluster_range,
                metric=optimization_config.get("elbow_metric", "inertia"),
                n_init=optimization_config.get("elbow_n_init", 10),
            )
            optimal_k, elbow_metadata = elbow.find_elbow(X_processed)
            print(f"Optimal k (elbow method): {optimal_k}")
            optimal_k_per_config["default"] = optimal_k
    
    # Calculate total combinations
    n_cluster_values = cluster_range if compare_multiple else [optimal_k_per_config.get("default", max(2, int(np.sqrt(n_samples / 2))))]
    
    total_combinations = (
        len(normalization_methods)
        * len(dimensionality_reductions)
        * len(clustering_methods)
        * len(n_cluster_values)
    )
    
    print(f"\nTotal combinations to test: {total_combinations}")
    print("=" * 80)
    
    with tqdm(total=total_combinations, desc="Comparing methods") as pbar:
        for norm_method in normalization_methods:
            for dim_reduction in dimensionality_reductions:
                # Skip UMAP if not available
                if dim_reduction == "umap":
                    try:
                        import umap
                    except ImportError:
                        pbar.update(len(clustering_methods) * len(n_cluster_values))
                        continue
                
                for cluster_method in clustering_methods:
                    for n_clusters in n_cluster_values:
                        config_id = f"{norm_method}_{dim_reduction or 'none'}_{cluster_method}_k{n_clusters}"
                        pbar.set_description(f"Testing {config_id}")
                        
                        try:
                            # Setup preprocessing config
                            preprocessor_config_dict = {
                                "normalization_method": norm_method,
                                "dimensionality_reduction": dim_reduction,
                                "n_components": preprocessing_config.get("n_components"),
                                "random_state": preprocessing_config.get("random_state", 42),
                            }
                            
                            # Setup clustering config
                            clustering_config_dict = {
                                "method": cluster_method,
                                "random_state": clustering_config.get("kmeans", {}).get("random_state", 42),
                            }
                            
                            # Set method-specific parameters
                            if cluster_method == "kmeans":
                                clustering_config_dict["n_clusters"] = n_clusters
                                clustering_config_dict["n_init"] = clustering_config.get("kmeans", {}).get("n_init", 10)
                            elif cluster_method == "dbscan":
                                # Use parameter search if configured
                                dbscan_config = clustering_config.get("dbscan", {})
                                eps_range = dbscan_config.get("eps_range", [0.5])
                                min_samples_range = dbscan_config.get("min_samples_range", [3])
                                
                                # For comparison, test first combination only (to avoid explosion)
                                # Use parameter search for optimal params
                                eps = eps_range[0] if len(eps_range) == 1 else eps_range[len(eps_range)//2]
                                min_samples = min_samples_range[0] if len(min_samples_range) == 1 else min_samples_range[len(min_samples_range)//2]
                                
                                clustering_config_dict["eps"] = eps
                                clustering_config_dict["min_samples"] = min_samples
                                
                                config_id_db = f"{config_id}_eps{eps}_min{min_samples}"
                                
                                # Run clustering
                                results_df, X_processed, metadata = analyze_clusters(
                                    output_dirs,
                                    preprocessor_config=preprocessor_config_dict,
                                    clustering_config=clustering_config_dict,
                                )
                                
                                # Evaluate clustering
                                labels = results_df["cluster_label"].values
                                evaluation = evaluate_clustering(X_processed, labels)
                                
                                # Store result
                                result = _create_result_dict(
                                    config_id_db,
                                    norm_method,
                                    dim_reduction,
                                    cluster_method,
                                    evaluation,
                                    metadata,
                                    {"eps": eps, "min_samples": min_samples},
                                )
                                results.append(result)
                                
                                pbar.update(1)
                                continue  # Skip the rest for DBSCAN
                                
                            elif cluster_method == "hierarchical":
                                clustering_config_dict["n_clusters"] = n_clusters
                                hierarchical_config = clustering_config.get("hierarchical", {})
                                linkage_options = hierarchical_config.get("linkage_options", ["ward"])
                                
                                # Use default linkage (first option)
                                linkage = linkage_options[0]
                                clustering_config_dict["linkage"] = linkage
                                
                                config_id_hier = f"{config_id}_link{linkage}"
                                
                                # Run clustering
                                results_df, X_processed, metadata = analyze_clusters(
                                    output_dirs,
                                    preprocessor_config=preprocessor_config_dict,
                                    clustering_config=clustering_config_dict,
                                )
                                
                                # Evaluate clustering
                                labels = results_df["cluster_label"].values
                                evaluation = evaluate_clustering(X_processed, labels)
                                
                                # Store result
                                result = _create_result_dict(
                                    config_id_hier,
                                    norm_method,
                                    dim_reduction,
                                    cluster_method,
                                    evaluation,
                                    metadata,
                                    {"linkage": linkage},
                                )
                                results.append(result)
                                
                                pbar.update(1)
                                continue  # Skip the rest for hierarchical
                            
                            # Run clustering for K-means
                            results_df, X_processed, metadata = analyze_clusters(
                                output_dirs,
                                preprocessor_config=preprocessor_config_dict,
                                clustering_config=clustering_config_dict,
                            )
                            
                            # Evaluate clustering
                            labels = results_df["cluster_label"].values
                            evaluation = evaluate_clustering(X_processed, labels)
                            
                            # Store result
                            result = _create_result_dict(
                                config_id,
                                norm_method,
                                dim_reduction,
                                cluster_method,
                                evaluation,
                                metadata,
                                {"n_clusters": n_clusters},
                            )
                            results.append(result)
                            
                        except Exception as e:
                            # Store failure
                            results.append({
                                "config_id": config_id,
                                "normalization": norm_method,
                                "dimensionality_reduction": dim_reduction or "none",
                                "clustering_method": cluster_method,
                                "status": "failed",
                                "error": str(e)[:200],
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
            if "n_clusters" in best and pd.notna(best["n_clusters"]):
                print(f"  Number of Clusters: {int(best['n_clusters'])}")
        
        print("=" * 80)
    
    return comparison_df


def _create_result_dict(
    config_id: str,
    norm_method: str,
    dim_reduction: str | None,
    cluster_method: str,
    evaluation: dict[str, Any],
    metadata: dict[str, Any],
    extra_params: dict[str, Any],
) -> dict[str, Any]:
    """Create result dictionary from evaluation and metadata."""
    prep_meta = metadata.get("preprocessing", {})
    cluster_meta = metadata.get("clustering", {})
    
    # Calculate explained variance if PCA was used
    explained_variance = None
    if dim_reduction == "pca" and "explained_variance_ratio" in prep_meta:
        var_ratio = prep_meta["explained_variance_ratio"]
        if var_ratio:
            explained_variance = sum(var_ratio)
    
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
    
    # Add extra parameters
    result.update(extra_params)
    
    # Add method-specific metrics
    if cluster_method == "kmeans" and "inertia" in cluster_meta:
        result["inertia"] = cluster_meta["inertia"]
    
    return result


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
        "--config",
        type=Path,
        default=Path("configs/clustering.yaml"),
        help="Clustering configuration file",
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

    # Load configuration
    config = load_clustering_config(args.config)
    
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
        config,
        min_samples=args.min_samples,
    )

    print(f"\nResults saved to: {args.output}")
    print(f"  - comparison_results.csv: Full comparison results")
    print(f"  - comparison_summary.yaml: Summary with best configurations")


if __name__ == "__main__":
    main()
