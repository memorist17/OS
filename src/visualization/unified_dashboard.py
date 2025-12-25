"""Unified Dashboard combining Indicator & Image View and Cluster & Image View."""

import json
from pathlib import Path
from typing import Any, Literal

import dash
import numpy as np
import plotly.graph_objects as go
import yaml
from dash import dcc, html
from dash.dependencies import Input, Output, State

from .cluster_image_view import (
    _calculate_zoom_level,
    _get_cluster_color,
    create_interactive_cluster_figure,
    create_point_detail_view,
    create_static_cluster_gallery,
    update_cluster_figure_display,
)
from .dashboard import load_results
from .indicator_image_view import (
    create_horizontal_scrollable_chart,
    create_image_gallery,
    create_value_popup,
    extract_indicator_data,
    extract_indicator_value,
    update_chart_highlight,
)
import json


def _create_clustering_chart(
    cluster_results: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> go.Figure:
    """Create a simple 2D clustering scatter plot.
    
    Args:
        cluster_results: Cluster analysis results
        config: Configuration dictionary
    
    Returns:
        Plotly Figure
    """
    if config is None:
        config = {}
    
    coordinates = np.array(cluster_results["coordinates"])
    labels = np.array(cluster_results["labels"])
    n_clusters = len(set(labels))
    
    chart_height = config.get("square_chart_size", 400)
    
    fig = go.Figure()
    
    # Plot each cluster
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_coords = coordinates[mask]
        
        fig.add_trace(go.Scatter(
            x=cluster_coords[:, 0],
            y=cluster_coords[:, 1],
            mode="markers",
            name=f"Cluster {cluster_id}",
            marker=dict(
                size=8,
                color=_get_cluster_color(cluster_id),
                opacity=0.7,
            ),
        ))
    
    fig.update_layout(
        title="",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        height=chart_height,
        width=chart_height,  # Square
        hovermode="closest",
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif", color="#c9d1d9", size=10),
        autosize=False,
        margin=dict(l=50, r=20, t=20, b=50),
        showlegend=True,
        xaxis=dict(
            gridcolor="#21262d",
            linecolor="#30363d",
        ),
        yaxis=dict(
            gridcolor="#21262d",
            linecolor="#30363d",
        ),
    )
    
    return fig


def load_clustering_results_for_unified(
    outputs_dir: Path,
    clustering_config: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[Path]]:
    """Load or compute clustering results for unified dashboard."""
    try:
        from src.analysis.clustering import ClusteringAnalyzer, FeatureExtractor, create_feature_matrix
    except ImportError:
        return None, []
    
    # Load all analysis results
    site_results = {}
    run_dirs = []
    
    run_dir_list = [
        d
        for d in outputs_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]
    
    for run_dir in sorted(run_dir_list, key=lambda x: x.stat().st_mtime, reverse=True):
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
                run_dirs.append(run_dir)
        except Exception as e:
            print(f"Warning: Could not load {run_dir}: {e}")
            continue
    
    if not site_results or len(site_results) < 2:
        return None, run_dirs
    
    # Extract features
    extractor = FeatureExtractor(
        mfa_q_values=clustering_config.get("feature_extraction", {}).get("mfa_q_values", [0, 1, 2]),
        lac_scales=clustering_config.get("feature_extraction", {}).get("lacunarity_scales", [4, 16, 64]),
        perc_fractions=clustering_config.get("feature_extraction", {}).get("percolation_fractions", [0.1, 0.5, 0.9]),
    )
    
    feature_df = create_feature_matrix(site_results, extractor)
    
    # Perform clustering
    from src.analysis.clustering import NormalizationMethod, ClusteringMethod, DimensionReductionMethod
    
    analyzer = ClusteringAnalyzer()
    
    # Set normalization method
    norm_method = clustering_config.get("normalization", "standard")
    if norm_method == "standard":
        analyzer.normalization = NormalizationMethod.STANDARD
    elif norm_method == "minmax":
        analyzer.normalization = NormalizationMethod.MINMAX
    elif norm_method == "robust":
        analyzer.normalization = NormalizationMethod.ROBUST
    
    # Set clustering method
    cluster_method = clustering_config.get("method", "kmeans")
    if cluster_method == "kmeans":
        analyzer.clustering = ClusteringMethod.KMEANS
    elif cluster_method == "dbscan":
        analyzer.clustering = ClusteringMethod.DBSCAN
    elif cluster_method == "hierarchical":
        analyzer.clustering = ClusteringMethod.HIERARCHICAL
    
    # Set dimension reduction
    dim_reduction = clustering_config.get("dimension_reduction", "none")
    if dim_reduction == "pca":
        analyzer.dimension_reduction = DimensionReductionMethod.PCA
    else:
        analyzer.dimension_reduction = DimensionReductionMethod.NONE
    
    # Set parameters
    analyzer.n_clusters = clustering_config.get("n_clusters", 5)
    analyzer.dbscan_eps = clustering_config.get("dbscan_eps", 0.5)
    analyzer.dbscan_min_samples = clustering_config.get("dbscan_min_samples", 3)
    analyzer.hierarchical_linkage = clustering_config.get("hierarchical_linkage", "ward")
    analyzer.pca_n_components = clustering_config.get("pca_n_components", 3)
    
    # Fit and transform
    result_df, metadata = analyzer.fit_transform(feature_df)
    labels = result_df["cluster"].values
    
    # Get coordinates
    if analyzer.dimension_reduction == DimensionReductionMethod.PCA:
        # Extract PCA coordinates from result_df
        pc_cols = [col for col in result_df.columns if col.startswith("PC")]
        if len(pc_cols) >= 2:
            coordinates = result_df[pc_cols[:2]].values
        else:
            coordinates = feature_df.values[:, :2]
    else:
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
    
    return cluster_results, run_dirs


def create_unified_dashboard(
    outputs_dir: Path | str,
    config: dict[str, Any] | None = None,
) -> dash.Dash:
    """Create unified dashboard with both Indicator & Image View and Cluster & Image View.
    
    Args:
        outputs_dir: Path to outputs directory containing run directories
        config: Configuration dictionary
    
    Returns:
        Dash application with tabs for both views
    """
    outputs_dir = Path(outputs_dir)
    
    if config is None:
        config = {}
        config_path = Path("configs/default.yaml")
        if config_path.exists():
            with open(config_path) as f:
                full_config = yaml.safe_load(f)
                config = full_config.get("visualization", {})
    
    # Find all run directories
    run_dirs = [
        d
        for d in outputs_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]
    
    if not run_dirs:
        raise ValueError(f"No run directories found in {outputs_dir}")
    
    run_dirs = sorted(run_dirs, key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Load all results
    all_results = [load_results(d) for d in run_dirs]
    
    # Load clustering config
    clustering_config = {}
    config_path = Path("configs/default.yaml")
    if config_path.exists():
        with open(config_path) as f:
            full_config = yaml.safe_load(f)
            clustering_config = full_config.get("analysis", {}).get("clustering", {})
    
    # Try to compute clustering results
    cluster_results, cluster_run_dirs = load_clustering_results_for_unified(
        outputs_dir, clustering_config
    )
    
    # Indicator view config
    indicator_config = config.get("indicator_image_view", {})
    
    # Cluster view config
    cluster_config = config.get("cluster_image_view", {})
    
    # Create app
    app = dash.Dash(
        __name__,
        title="Urban Structure Analysis - Unified View",
        suppress_callback_exceptions=True,
    )
    
    # Custom CSS - GitHub-like UI
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <link href="https://fonts.googleapis.com/css2?family=-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif&display=swap" rel="stylesheet">
            <style>
                body {
                    background: #0d1117;
                    min-height: 100vh;
                    margin: 0;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
                    color: #c9d1d9;
                }
                .container {
                    max-width: 1600px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1 {
                    text-align: center;
                    color: #f0f6fc;
                    font-size: 2em;
                    font-weight: 600;
                    margin-bottom: 10px;
                    border-bottom: 1px solid #21262d;
                    padding-bottom: 10px;
                }
                .subtitle {
                    text-align: center;
                    color: #8b949e;
                    margin-bottom: 30px;
                    font-size: 0.9em;
                }
                .sticky-header {
                    position: sticky;
                    top: 0;
                    background: #0d1117;
                    z-index: 100;
                    padding: 10px 0;
                    border-bottom: 1px solid #21262d;
                    margin-bottom: 0;
                }
                .sticky-header .chart-container {
                    margin-bottom: 0;
                }
                .chart-container .chart-title {
                    position: sticky;
                    top: 0;
                    background: #161b22;
                    z-index: 10;
                    padding: 8px 16px;
                    margin: 0;
                }
                .chart-container {
                    margin-bottom: 20px;
                    border: 1px solid #21262d;
                    border-radius: 6px;
                    background: #161b22;
                    padding: 16px;
                }
                .chart-title {
                    color: #f0f6fc;
                    font-size: 1.1em;
                    font-weight: 600;
                    margin-bottom: 10px;
                    padding-bottom: 8px;
                    border-bottom: 1px solid #21262d;
                }
                /* Slider Customization */
                .rc-slider-track {
                    background-color: #238636 !important;
                }
                .rc-slider-handle {
                    border-color: #238636 !important;
                    background-color: #238636 !important;
                }
                .rc-slider-rail {
                    background-color: #21262d !important;
                }
                .rc-slider-mark-text {
                    color: #8b949e !important;
                    font-size: 0.8em;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Initial values - create 3 charts for MFA, Lacunarity, and Percolation
    # Note: indicator_value is not used for lacunarity and percolation (they show full curves)
    # Set square chart size
    square_chart_size = indicator_config.get("square_chart_size", 400)
    indicator_config["square_chart_size"] = square_chart_size
    indicator_config["chart_height"] = square_chart_size
    
    mfa_chart = create_horizontal_scrollable_chart(
        all_results, "mfa", "D(0)", indicator_config
    )
    lacunarity_chart = create_horizontal_scrollable_chart(
        all_results, "lacunarity", "", indicator_config
    )
    percolation_chart = create_horizontal_scrollable_chart(
        all_results, "percolation", "", indicator_config
    )
    
    # Create clustering chart if available
    clustering_chart = None
    if cluster_results is not None:
        clustering_chart = _create_clustering_chart(cluster_results, indicator_config)
    
    gallery_items = create_image_gallery(all_results, run_dirs, indicator_config)
    
    highlight_color = indicator_config.get("highlight_color", "rgba(255, 0, 0, 1.0)")
    
    # Create cluster figure if available
    cluster_fig = None
    if cluster_results is not None:
        cluster_fig = create_interactive_cluster_figure(
            cluster_results, cluster_run_dirs, cluster_config
        )
    
    zoom_threshold = cluster_config.get("zoom_threshold", 0.5)
    
    # Build layout with tabs - GitHub-like UI
    app.layout = html.Div([
        html.Div([
            # Sticky header section
            html.Div([
                html.H1("Urban Structure Analysis - Unified View"),
                html.P(f"Number of runs: {len(run_dirs)}", className="subtitle"),
                
                # Resizable Layout Controls
                html.Div([
                    html.Div([
                        html.Label("Gallery Width", style={"color": "#8b949e", "margin-right": "15px", "font-size": "0.9em"}),
                        dcc.Slider(
                            id="gallery-width-slider",
                            min=10,
                            max=70,
                            step=5,
                            value=25,
                            marks={i: {'label': f'{i}%', 'style': {'color': '#8b949e'}} for i in range(10, 71, 10)},
                        ),
                    ], style={"flex": "1", "padding": "0 20px"}),
                ], style={
                    "display": "flex", 
                    "align-items": "center", 
                    "background": "#161b22", 
                    "padding": "10px", 
                    "border-radius": "6px",
                    "margin": "0 20px 20px 20px",
                    "border": "1px solid #21262d"
                }),
            
            dcc.Tabs(
                id="main-tabs",
                value="indicator-tab",
                children=[
                    dcc.Tab(
                            label="Indicator & Image",
                        value="indicator-tab",
                        style={
                                "background": "#161b22",
                                "color": "#c9d1d9",
                                "border": "1px solid #21262d",
                                "border-bottom": "none",
                        },
                        selected_style={
                                "background": "#0d1117",
                                "color": "#f0f6fc",
                                "border": "1px solid #21262d",
                                "border-bottom": "1px solid #0d1117",
                        },
                    ),
                    dcc.Tab(
                            label="Cluster & Image",
                        value="cluster-tab",
                        style={
                                "background": "#161b22",
                                "color": "#c9d1d9",
                                "border": "1px solid #21262d",
                                "border-bottom": "none",
                        },
                        selected_style={
                                "background": "#0d1117",
                                "color": "#f0f6fc",
                                "border": "1px solid #21262d",
                                "border-bottom": "1px solid #0d1117",
                        },
                    ),
                ],
                style={
                        "margin-bottom": "0",
                        "border-bottom": "1px solid #21262d",
                },
            ),
                
            ], className="sticky-header"),
            
            # Tab content
            html.Div(id="tab-content"),
        ], className="container"),
        
        # Shared popup for indicator view
        html.Div(
            id="value-popup",
            style={
                "position": "fixed",
                "top": "50%",
                "left": "50%",
                "transform": "translate(-50%, -50%)",
                "background": "#161b22",
                "padding": "20px",
                "border": "1px solid #21262d",
                "border-radius": "6px",
                "z-index": "1000",
                "display": "none",
                "max-width": "500px",
                "color": "#c9d1d9",
            },
        ),
    ], style={
        "background": "#0d1117",
        "min-height": "100vh",
    })
    
    # Callback to update tab content
    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
        Input("gallery-width-slider", "value"),
    )
    def update_tab_content(active_tab, gallery_width):
        """Update content based on active tab."""
        if active_tab == "indicator-tab":
            # Indicator & Image View - Resizable side-by-side layout
            chart_height = indicator_config.get("square_chart_size", 400)
            
            return html.Div([
                # Left side: Gallery
                html.Div([
                    html.H3("Measurement Points Gallery", style={
                        "color": "#f0f6fc",
                        "font-size": "1.1em",
                        "font-weight": "600",
                        "margin-bottom": "16px",
                        "padding-bottom": "8px",
                        "border-bottom": "1px solid #21262d",
                }),
                    html.Div(
                        id="image-gallery",
                        children=gallery_items,
                        style={
                            "display": "grid",
                            "grid-template-columns": "repeat(auto-fill, minmax(200px, 1fr))",
                            "gap": "16px",
                            "padding": "20px 0",
                            "overflow-y": "auto",
                            "max-height": "calc(100vh - 200px)",
                        },
                    ),
                    # Hidden div for scroll target
                    html.Div(id="scroll-target", style={"display": "none"}),
                ], style={
                    "width": f"{gallery_width}%",
                    "float": "left",
                    "padding": "20px",
                    "box-sizing": "border-box",
                }),
                
                # Right side: Charts in vertical column
                html.Div([
                    # MFA Chart - Square and static
                    html.Div([
                        html.Div("MFA - D(q)", className="chart-title", style={"font-size": "0.9em"}),
                        html.Div([
                            dcc.Graph(
                                id="mfa-chart-sticky",
                                figure=mfa_chart,
                                style={
                                    "height": "100%",
                                    "width": "100%",
                                    "margin": "0 auto",
                                },
                                config={
                                    "staticPlot": False,  # Enable interaction for clicking
                                    "displayModeBar": False,
                                },
                    ),
                        ], style={
                            "width": "100%",
                            "height": f"{chart_height}px",
                            "min-height": f"{chart_height}px",
                            "margin": "0 auto",
                        }),
                    ], className="chart-container", style={
                        "margin-bottom": "20px",
                    }),
                    
                    # Lacunarity Chart - Square and static
                    html.Div([
                        html.Div("Lacunarity - Λ(r)", className="chart-title", style={"font-size": "0.9em"}),
                        html.Div([
                            dcc.Graph(
                                id="lacunarity-chart-sticky",
                                figure=lacunarity_chart,
                                style={
                                    "height": "100%",
                                    "width": "100%",
                                    "margin": "0 auto",
                                },
                                config={
                                    "staticPlot": False,  # Enable interaction for clicking
                                    "displayModeBar": False,
                                },
                            ),
                        ], style={
                            "width": "100%",
                            "height": f"{chart_height}px",
                            "min-height": f"{chart_height}px",
                            "margin": "0 auto",
                        }),
                    ], className="chart-container", style={
                        "margin-bottom": "20px",
                    }),
                    
                    # Percolation Chart - Square and static
                    html.Div([
                        html.Div("Percolation - Giant Component Fraction", className="chart-title", style={"font-size": "0.9em"}),
                        html.Div([
                            dcc.Graph(
                                id="percolation-chart-sticky",
                                figure=percolation_chart,
                                style={
                                    "height": "100%",
                                    "width": "100%",
                                    "margin": "0 auto",
                                },
                                config={
                                    "staticPlot": False,  # Enable interaction for clicking
                                    "displayModeBar": False,
                                },
                            ),
                        ], style={
                            "width": "100%",
                            "height": f"{chart_height}px",
                            "min-height": f"{chart_height}px",
                            "margin": "0 auto",
                        }),
                    ], className="chart-container", style={
                        "margin-bottom": "20px",
                    }),
                    
                    # Clustering Chart (if available) - Square and static
                    (html.Div([
                        html.Div("Clustering - 2D Projection", className="chart-title", style={"font-size": "0.9em"}),
                        html.Div([
                            dcc.Graph(
                                id="clustering-chart-sticky",
                                figure=clustering_chart,
                                style={
                                    "height": "100%",
                                    "width": "100%",
                                    "margin": "0 auto",
                                },
                                config={
                                    "staticPlot": False,  # Enable interaction for clicking
                                    "displayModeBar": False,
                                },
                            ),
                        ], style={
                            "width": "100%",
                            "height": f"{chart_height}px",
                            "min-height": f"{chart_height}px",
                            "margin": "0 auto",
                        }),
                    ], className="chart-container") if clustering_chart is not None else html.Div()),
                ], style={
                    "width": f"{100 - gallery_width}%",
                    "float": "right",
                    "padding": "20px",
                    "box-sizing": "border-box",
                    "position": "sticky",
                    "top": "100px",
                    "max-height": "calc(100vh - 100px)",
                    "overflow-y": "auto",
                }),
            ], style={
                "clear": "both",
                "display": "flex",
            })
        else:
            # Cluster & Image View - Resizable side-by-side layout
            if cluster_results is None:
                return html.Div([
                    html.H3("Cluster & Image Correspondence", style={
                        "color": "#f0f6fc",
                        "font-size": "1.1em",
                        "font-weight": "600",
                        "margin-bottom": "16px",
                    }),
                    html.P(
                        "クラスターと画像対応ビューを使用するには、最低2つ以上のrunディレクトリが必要です。",
                        style={"color": "#8b949e", "padding": "20px"},
                    ),
                ], className="chart-container")
            else:
                # Create static cluster gallery
                static_gallery = create_static_cluster_gallery(
                    cluster_results, cluster_run_dirs, cluster_config
                )
                
                # Create gallery for cluster view (same as indicator view)
                cluster_gallery_items = create_image_gallery(all_results, run_dirs, indicator_config)
                
                return html.Div([
                    # Left side: Gallery
                    html.Div([
                        html.H3("Measurement Points Gallery", style={
                            "color": "#f0f6fc",
                            "font-size": "1.1em",
                            "font-weight": "600",
                            "margin-bottom": "16px",
                            "padding-bottom": "8px",
                            "border-bottom": "1px solid #21262d",
                        }),
                        html.Div(
                            id="cluster-image-gallery",
                            children=cluster_gallery_items,
                            style={
                                "display": "grid",
                                "grid-template-columns": "repeat(auto-fill, minmax(200px, 1fr))",
                                "gap": "16px",
                                "padding": "20px 0",
                                "overflow-y": "auto",
                                "max-height": "calc(100vh - 200px)",
                            },
                        ),
                    ], style={
                        "width": f"{gallery_width}%",
                        "float": "left",
                        "padding": "20px",
                        "box-sizing": "border-box",
                    }),
                    
                    # Right side: Cluster Visualization
                    html.Div([
                        html.Div("Cluster Visualization", className="chart-title", style={
                            "font-size": "1.1em",
                            "margin-bottom": "10px",
                        }),
                        html.Div(
                            static_gallery,
                            style={
                                "overflow": "auto",
                                "background": "#ffffff",
                                "border-radius": "6px",
                                "border": "1px solid #21262d",
                            }
                        ),
                    ], style={
                        "width": f"{100 - gallery_width}%",
                        "float": "right",
                        "padding": "20px",
                        "box-sizing": "border-box",
                        "position": "sticky",
                        "top": "100px",
                        "max-height": "calc(100vh - 100px)",
                        "overflow-y": "auto",
                    }),
                ], style={
                    "clear": "both",
                    "display": "flex",
                })
    
    # Callback: Image click -> Chart highlight (indicator view)
    @app.callback(
        Output("mfa-chart-sticky", "figure", allow_duplicate=True),
        Output("lacunarity-chart-sticky", "figure", allow_duplicate=True),
        Output("percolation-chart-sticky", "figure", allow_duplicate=True),
        Input({"type": "gallery-image", "index": dash.dependencies.ALL, "img_type": dash.dependencies.ALL}, "n_clicks"),
        State("mfa-chart-sticky", "figure"),
        State("lacunarity-chart-sticky", "figure"),
        State("percolation-chart-sticky", "figure"),
        prevent_initial_call=True,
    )
    def highlight_charts_on_image_click(n_clicks_list, mfa_fig, lacunarity_fig, percolation_fig):
        """Highlight charts when image is clicked."""
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update
        
        triggered_id = ctx.triggered[0]["prop_id"]
        if not triggered_id or "gallery-image" not in triggered_id:
            return dash.no_update, dash.no_update, dash.no_update
        
        try:
            prop_id = json.loads(triggered_id.split(".")[0])
            point_idx = prop_id.get("index", -1)
            if point_idx < 0 or point_idx >= len(all_results):
                return dash.no_update, dash.no_update, dash.no_update
        except Exception:
            return dash.no_update, dash.no_update, dash.no_update
        
        # Update all charts with highlight
        highlight_color = indicator_config.get("highlight_color", "rgba(255, 0, 0, 1.0)")
        
        updated_mfa = update_chart_highlight(
            go.Figure(mfa_fig), point_idx, highlight_color
        )
        updated_lacunarity = update_chart_highlight(
            go.Figure(lacunarity_fig), point_idx, highlight_color
        )
        updated_percolation = update_chart_highlight(
            go.Figure(percolation_fig), point_idx, highlight_color
        )
        
        return updated_mfa, updated_lacunarity, updated_percolation
    
    # Callback: Chart line click -> Image highlight and scroll (indicator view)
    @app.callback(
        Output("image-gallery", "children", allow_duplicate=True),
        Output("scroll-target", "children", allow_duplicate=True),
        Input("mfa-chart-sticky", "clickData"),
        Input("lacunarity-chart-sticky", "clickData"),
        Input("percolation-chart-sticky", "clickData"),
        State("image-gallery", "children"),
        prevent_initial_call=True,
    )
    def highlight_image_on_chart_click(mfa_click, lacunarity_click, percolation_click, current_gallery):
        """Highlight image and scroll to it when chart line is clicked."""
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update
        
        # Get click data from any chart
        click_data = mfa_click or lacunarity_click or percolation_click
        if not click_data or "points" not in click_data:
            return dash.no_update, dash.no_update
        
        try:
            # Get point index from clicked trace
            point_data = click_data["points"][0]
            # Try to get customdata first (point index), then fallback to curveNumber (trace index)
            point_idx = -1
            if "customdata" in point_data:
                customdata = point_data["customdata"]
                if isinstance(customdata, list) and len(customdata) > 0:
                    point_idx = customdata[0]
                elif isinstance(customdata, (int, float)):
                    point_idx = int(customdata)
            elif "curveNumber" in point_data:
                # Fallback to curveNumber (trace index)
                point_idx = point_data["curveNumber"]
            
            if point_idx < 0 or point_idx >= len(all_results):
                return dash.no_update, dash.no_update
            
            # Recreate gallery with highlight and scroll target
            highlight_color = indicator_config.get("highlight_color", "rgba(255, 0, 0, 1.0)")
            gallery_items = create_image_gallery(all_results, run_dirs, indicator_config)
            
            # Add highlight style and ID to the clicked image for scrolling
            if point_idx < len(gallery_items):
                # Find the image element in the gallery item
                gallery_item = gallery_items[point_idx]
                if isinstance(gallery_item, html.Div):
                    # Add ID for scrolling
                    gallery_item.id = f"gallery-item-{point_idx}"
                    if len(gallery_item.children) > 0:
                        img_element = gallery_item.children[0]
                        if isinstance(img_element, html.Img):
                            # Update image style with highlight
                            current_style = dict(img_element.style) if img_element.style else {}
                            current_style["border"] = f"3px solid {highlight_color}"
                            current_style["box-shadow"] = f"0 0 10px {highlight_color}"
                            current_style["transform"] = "scale(1.05)"
                            img_element.style = current_style
            
            # Return gallery items and scroll target ID to trigger scroll
            return gallery_items, point_idx
        except Exception as e:
            print(f"Error highlighting image: {e}")
            return dash.no_update, dash.no_update
    
    # Clientside callback to scroll to highlighted image
    app.clientside_callback(
        """
        function(scroll_target) {
            if (scroll_target !== null && scroll_target !== undefined) {
                const element = document.getElementById('gallery-item-' + scroll_target);
                if (element) {
                    setTimeout(function() {
                        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }, 100);
                }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("image-gallery", "style", allow_duplicate=True),
        Input("scroll-target", "children"),
        prevent_initial_call=True,
    )
    
    return app
