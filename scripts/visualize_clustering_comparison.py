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
    
    # Create a single HTML file with all visualizations using subplots
    # Layout: 3 rows x 2 cols
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Top Configurations by Silhouette Score",
            "Top Configurations by Davies-Bouldin Score",
            "Method Heatmap (Silhouette Score)",
            "Silhouette vs Davies-Bouldin Score",
            "Metric Distribution: Silhouette & Davies-Bouldin",
            "Metric Distribution: Calinski-Harabasz & Clusters"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "heatmap"}, {"type": "scatter"}],
            [{"type": "box", "colspan": 2}, None],
        ],
        row_heights=[0.3, 0.35, 0.35],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    
    # 1. Top Configurations by Silhouette Score (Row 1, Col 1)
    top_silhouette = successful.sort_values("silhouette_score", ascending=False).head(10)
    fig.add_trace(
        go.Bar(
            x=top_silhouette["config_id"],
            y=top_silhouette["silhouette_score"],
            marker_color=top_silhouette["clustering_method"].map({
                "kmeans": "#1f77b4",
                "dbscan": "#ff7f0e",
                "hierarchical": "#2ca02c"
            }),
            name="Silhouette Score",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=1)
    
    # 2. Top Configurations by Davies-Bouldin Score (Row 1, Col 2)
    top_db = successful.sort_values("davies_bouldin_score", ascending=True).head(10)
    fig.add_trace(
        go.Bar(
            x=top_db["config_id"],
            y=top_db["davies_bouldin_score"],
            marker_color=top_db["clustering_method"].map({
                "kmeans": "#1f77b4",
                "dbscan": "#ff7f0e",
                "hierarchical": "#2ca02c"
            }),
            name="Davies-Bouldin Score",
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.update_xaxes(tickangle=45, row=1, col=2)
    fig.update_yaxes(title_text="Davies-Bouldin Score", row=1, col=2)
    
    # 3. Method Heatmap (Row 2, Col 1)
    method_comparison = successful.groupby(["normalization", "dimensionality_reduction", "clustering_method"]).agg({
        "silhouette_score": "mean",
    }).reset_index()
    
    pivot_silhouette = method_comparison.pivot_table(
        index=["normalization", "dimensionality_reduction"],
        columns="clustering_method",
        values="silhouette_score",
    )
    
    fig.add_trace(
        go.Heatmap(
            z=pivot_silhouette.values,
            x=pivot_silhouette.columns.tolist(),
            y=[f"{idx[0]}_{idx[1]}" for idx in pivot_silhouette.index],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Silhouette Score", len=0.3, y=0.5),
        ),
        row=2, col=1,
    )
    fig.update_xaxes(title_text="Clustering Method", row=2, col=1)
    fig.update_yaxes(title_text="Preprocessing", row=2, col=1)
    
    # 4. Scatter Plot: Silhouette vs Davies-Bouldin (Row 2, Col 2)
    for method in successful["clustering_method"].unique():
        method_data = successful[successful["clustering_method"] == method]
        fig.add_trace(
            go.Scatter(
                x=method_data["silhouette_score"],
                y=method_data["davies_bouldin_score"],
                mode="markers",
                name=method,
                marker=dict(
                    size=method_data["n_clusters"] * 3,
                    opacity=0.7,
                ),
                text=method_data["config_id"],
                hovertemplate="<b>%{text}</b><br>" +
                            "Silhouette: %{x:.3f}<br>" +
                            "Davies-Bouldin: %{y:.3f}<br>" +
                            "<extra></extra>",
            ),
            row=2, col=2,
        )
    fig.update_xaxes(title_text="Silhouette Score", row=2, col=2)
    fig.update_yaxes(title_text="Davies-Bouldin Score", row=2, col=2)
    
    # 5. Metric Distribution Box Plots (Row 3)
    # Add box plots directly to row 3
    for method in successful["clustering_method"].unique():
        method_data = successful[successful["clustering_method"] == method]
        fig.add_trace(
            go.Box(y=method_data["silhouette_score"], name=f"{method} (Silhouette)", legendgroup=method, showlegend=True),
            row=3, col=1,
        )
        fig.add_trace(
            go.Box(y=method_data["davies_bouldin_score"], name=f"{method} (DB)", legendgroup=method, showlegend=False),
            row=3, col=1,
        )
        fig.add_trace(
            go.Box(y=method_data["calinski_harabasz_score"], name=f"{method} (CH)", legendgroup=method, showlegend=False),
            row=3, col=1,
        )
        fig.add_trace(
            go.Box(y=method_data["n_clusters"], name=f"{method} (Clusters)", legendgroup=method, showlegend=False),
            row=3, col=1,
        )
    
    # Update layout
    fig.update_layout(
        title_text="Clustering Methods Comparison - All Visualizations",
        height=1400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    # Save as single HTML file
    output_file = output_path / "clustering_comparison_all.html"
    fig.write_html(str(output_file))
    
    print(f"Combined visualization saved to: {output_file}")
    print(f"  - All 6 visualizations in one HTML file")


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
        default=Path("outputs/clustering_comparison"),
        help="Output directory for visualization file",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Create separate HTML files instead of one combined file",
    )

    args = parser.parse_args()

    comparison_path = args.comparison_dir / "comparison_results.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"Comparison results not found: {comparison_path}")

    # Create single combined file (default)
    create_comparison_visualizations(args.comparison_dir, args.output)


if __name__ == "__main__":
    main()

