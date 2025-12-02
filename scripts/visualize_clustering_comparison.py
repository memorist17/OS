#!/usr/bin/env python
"""Visualize Clustering Comparison Results."""

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_comparison_visualizations(comparison_path: Path, output_path: Path):
    """Create visualizations for clustering comparison results."""
    # Load comparison results
    comparison_df = pd.read_csv(comparison_path / "comparison_results.csv")
    
    # Filter successful results
    successful = comparison_df[comparison_df["status"] == "success"].copy()
    
    if len(successful) == 0:
        print("No successful results to visualize")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Silhouette Score Comparison
    fig1 = px.bar(
        successful.sort_values("silhouette_score", ascending=False).head(20),
        x="config_id",
        y="silhouette_score",
        color="clustering_method",
        title="Top 20 Configurations by Silhouette Score",
        labels={"silhouette_score": "Silhouette Score (higher is better)", "config_id": "Configuration"},
    )
    fig1.update_xaxes(tickangle=45)
    fig1.write_html(str(output_path / "silhouette_comparison.html"))
    
    # 2. Davies-Bouldin Score Comparison
    fig2 = px.bar(
        successful.sort_values("davies_bouldin_score", ascending=True).head(20),
        x="config_id",
        y="davies_bouldin_score",
        color="clustering_method",
        title="Top 20 Configurations by Davies-Bouldin Score (lower is better)",
        labels={"davies_bouldin_score": "Davies-Bouldin Score (lower is better)", "config_id": "Configuration"},
    )
    fig2.update_xaxes(tickangle=45)
    fig2.write_html(str(output_path / "davies_bouldin_comparison.html"))
    
    # 3. Method Comparison Heatmap
    method_comparison = successful.groupby(["normalization", "dimensionality_reduction", "clustering_method"]).agg({
        "silhouette_score": "mean",
        "davies_bouldin_score": "mean",
        "calinski_harabasz_score": "mean",
    }).reset_index()
    
    # Create pivot table for heatmap
    pivot_silhouette = method_comparison.pivot_table(
        index=["normalization", "dimensionality_reduction"],
        columns="clustering_method",
        values="silhouette_score",
    )
    
    fig3 = px.imshow(
        pivot_silhouette.values,
        x=pivot_silhouette.columns,
        y=[f"{idx[0]}_{idx[1]}" for idx in pivot_silhouette.index],
        labels=dict(x="Clustering Method", y="Preprocessing", color="Silhouette Score"),
        title="Average Silhouette Score by Method Combination",
        color_continuous_scale="Viridis",
    )
    fig3.write_html(str(output_path / "method_heatmap_silhouette.html"))
    
    # 4. Scatter Plot: Silhouette vs Davies-Bouldin
    fig4 = px.scatter(
        successful,
        x="silhouette_score",
        y="davies_bouldin_score",
        color="clustering_method",
        size="n_clusters",
        hover_data=["config_id", "normalization", "dimensionality_reduction"],
        title="Silhouette Score vs Davies-Bouldin Score",
        labels={
            "silhouette_score": "Silhouette Score (higher is better)",
            "davies_bouldin_score": "Davies-Bouldin Score (lower is better)",
        },
    )
    fig4.write_html(str(output_path / "score_scatter.html"))
    
    # 5. Multi-metric Comparison
    fig5 = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Silhouette Score", "Davies-Bouldin Score", "Calinski-Harabasz Score", "Number of Clusters"),
        specs=[[{"type": "box"}, {"type": "box"}], [{"type": "box"}, {"type": "box"}]],
    )
    
    for method in successful["clustering_method"].unique():
        method_data = successful[successful["clustering_method"] == method]
        fig5.add_trace(
            go.Box(y=method_data["silhouette_score"], name=f"{method}_silhouette", showlegend=False),
            row=1, col=1,
        )
        fig5.add_trace(
            go.Box(y=method_data["davies_bouldin_score"], name=f"{method}_db", showlegend=False),
            row=1, col=2,
        )
        fig5.add_trace(
            go.Box(y=method_data["calinski_harabasz_score"], name=f"{method}_ch", showlegend=False),
            row=2, col=1,
        )
        fig5.add_trace(
            go.Box(y=method_data["n_clusters"], name=f"{method}_clusters", showlegend=False),
            row=2, col=2,
        )
    
    fig5.update_layout(title="Metric Distribution by Clustering Method", height=800)
    fig5.write_html(str(output_path / "metric_distribution.html"))
    
    print(f"Visualizations saved to: {output_path}")
    print("  - silhouette_comparison.html")
    print("  - davies_bouldin_comparison.html")
    print("  - method_heatmap_silhouette.html")
    print("  - score_scatter.html")
    print("  - metric_distribution.html")


def main():
    """Create visualizations for clustering comparison."""
    parser = argparse.ArgumentParser(
        description="Visualize clustering comparison results"
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=Path("outputs/clustering_comparison"),
        help="Directory containing comparison results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/clustering_comparison/visualizations"),
        help="Output directory for visualizations",
    )

    args = parser.parse_args()

    comparison_path = args.comparison_dir / "comparison_results.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"Comparison results not found: {comparison_path}")

    create_comparison_visualizations(args.comparison_dir, args.output)


if __name__ == "__main__":
    main()

