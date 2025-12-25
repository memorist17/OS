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
    create_interactive_cluster_figure,
    create_point_detail_view,
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
    analyzer = ClusteringAnalyzer(
        method=clustering_config.get("method", "kmeans"),
        n_clusters=clustering_config.get("n_clusters", 5),
        normalization=clustering_config.get("normalization", "standard"),
        dimension_reduction=clustering_config.get("dimension_reduction", "none"),
        pca_n_components=clustering_config.get("pca_n_components", 3),
    )
    
    labels = analyzer.fit_predict(feature_df.values)
    
    # Get coordinates
    if analyzer.dimension_reduction == "pca" and hasattr(analyzer, "pca_"):
        coordinates = analyzer.pca_.transform(feature_df.values)[:, :2]
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
    
    # Custom CSS
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
            <style>
                body {
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                    min-height: 100vh;
                    margin: 0;
                    font-family: 'Noto Sans JP', sans-serif;
                    color: #eee;
                }
                .container {
                    max-width: 1600px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1 {
                    text-align: center;
                    background: linear-gradient(120deg, #FF6B6B, #4ECDC4, #45B7D1);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-size: 2.5em;
                    margin-bottom: 10px;
                }
                .subtitle {
                    text-align: center;
                    color: #888;
                    margin-bottom: 30px;
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
    
    # Initial values
    indicator_type_default = "mfa"
    indicator_value_default = "D(0)"
    
    # Create initial chart and gallery
    initial_chart = create_horizontal_scrollable_chart(
        all_results, indicator_type_default, indicator_value_default, indicator_config
    )
    gallery_items = create_image_gallery(all_results, run_dirs, indicator_config)
    
    highlight_color = indicator_config.get("highlight_color", "rgba(255, 0, 0, 1.0)")
    
    # Create cluster figure if available
    cluster_fig = None
    if cluster_results is not None:
        cluster_fig = create_interactive_cluster_figure(
            cluster_results, cluster_run_dirs, cluster_config
        )
    
    zoom_threshold = cluster_config.get("zoom_threshold", 0.5)
    
    # Build layout with tabs
    app.layout = html.Div([
        html.Div([
            html.H1("🏙️ Urban Structure Analysis - Unified View"),
            html.P(f"Number of runs: {len(run_dirs)}", className="subtitle"),
            
            dcc.Tabs(
                id="main-tabs",
                value="indicator-tab",
                children=[
                    dcc.Tab(
                        label="📊 Indicator & Image",
                        value="indicator-tab",
                        style={
                            "background": "#16213e",
                            "color": "#eee",
                            "border": "1px solid #4ECDC4",
                        },
                        selected_style={
                            "background": "#1a1a2e",
                            "color": "#4ECDC4",
                            "border": "2px solid #4ECDC4",
                        },
                    ),
                    dcc.Tab(
                        label="🔗 Cluster & Image",
                        value="cluster-tab",
                        style={
                            "background": "#16213e",
                            "color": "#eee",
                            "border": "1px solid #4ECDC4",
                        },
                        selected_style={
                            "background": "#1a1a2e",
                            "color": "#4ECDC4",
                            "border": "2px solid #4ECDC4",
                        },
                    ),
                ],
                style={
                    "margin-bottom": "20px",
                },
            ),
            
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
                "background": "#1a1a2e",
                "padding": "20px",
                "border": "2px solid #4ECDC4",
                "border-radius": "10px",
                "z-index": "1000",
                "display": "none",
                "max-width": "500px",
                "color": "#eee",
            },
        ),
    ], style={
        "background": "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)",
        "min-height": "100vh",
    })
    
    # Callback to update tab content
    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
    )
    def update_tab_content(active_tab):
        """Update content based on active tab."""
        if active_tab == "indicator-tab":
            # Indicator & Image View
            return html.Div([
                html.Div([
                    html.Label("Indicator Type:", style={"color": "#eee", "margin-right": "10px"}),
                    dcc.Dropdown(
                        id="indicator-type-dropdown",
                        options=[
                            {"label": "MFA", "value": "mfa"},
                            {"label": "Lacunarity", "value": "lacunarity"},
                            {"label": "Percolation", "value": "percolation"},
                        ],
                        value=indicator_type_default,
                        style={"width": "200px", "display": "inline-block"},
                    ),
                    html.Label("Indicator Value:", style={"color": "#eee", "margin-left": "20px", "margin-right": "10px"}),
                    dcc.Dropdown(
                        id="indicator-value-dropdown",
                        options=[
                            {"label": "D(0)", "value": "D(0)"},
                            {"label": "D(1)", "value": "D(1)"},
                            {"label": "D(2)", "value": "D(2)"},
                        ],
                        value=indicator_value_default,
                        style={"width": "200px", "display": "inline-block"},
                    ),
                ], style={"margin-bottom": "20px", "padding": "20px", "background": "rgba(255,255,255,0.05)", "border-radius": "10px"}),
                
                # Chart area
                html.Div([
                    dcc.Graph(
                        id="scrollable-chart",
                        figure=initial_chart,
                        style={"height": f"{indicator_config.get('chart_height', 400)}px"},
                        config={"scrollZoom": True},
                    ),
                ], style={
                    "width": "100%",
                    "overflow-x": "auto",
                    "border": "1px solid #ccc",
                    "padding": "10px",
                    "border-radius": "10px",
                    "background": "#16213e",
                }),
                
                # Image gallery
                html.Div([
                    html.H3("Measurement Points Gallery", style={"color": "#4ECDC4"}),
                    html.Div(
                        id="image-gallery",
                        children=gallery_items,
                        style={
                            "display": "grid",
                            "grid-template-columns": "repeat(auto-fill, minmax(200px, 1fr))",
                            "gap": "10px",
                            "padding": "20px",
                        },
                    ),
                ]),
            ])
        else:
            # Cluster & Image View
            if cluster_results is None or cluster_fig is None:
                return html.Div([
                    html.H3("Cluster & Image Correspondence", style={"color": "#4ECDC4"}),
                    html.P(
                        "クラスターと画像対応ビューを使用するには、最低2つ以上のrunディレクトリが必要です。",
                        style={"color": "#888", "padding": "20px"},
                    ),
                ])
            else:
                return html.Div([
                    # Cluster plot
                    dcc.Graph(
                        id="cluster-plot",
                        figure=cluster_fig,
                        style={"height": f"{cluster_config.get('plot_height', 800)}px"},
                        config={"scrollZoom": True, "doubleClick": "reset"},
                    ),
                    
                    # Zoom level display
                    html.Div(
                        id="zoom-level-display",
                        style={"margin": "10px", "color": "#eee", "text-align": "center"},
                    ),
                    
                    # Selected point details
                    html.Div(
                        id="selected-point-details",
                        style={"margin": "20px", "color": "#eee"},
                    ),
                ])
    
    # Callback for indicator chart update
    @app.callback(
        Output("scrollable-chart", "figure"),
        Input("indicator-type-dropdown", "value"),
        Input("indicator-value-dropdown", "value"),
    )
    def update_indicator_chart(indicator_type, indicator_value):
        """Update indicator chart when dropdowns change."""
        if indicator_type is None:
            indicator_type = indicator_type_default
        if indicator_value is None:
            indicator_value = indicator_value_default
        
        return create_horizontal_scrollable_chart(
            all_results, indicator_type, indicator_value, indicator_config
        )
    
    # Callback: Image click -> Chart highlight (indicator view)
    @app.callback(
        Output("scrollable-chart", "figure", allow_duplicate=True),
        Output("value-popup", "children"),
        Output("value-popup", "style"),
        Input({"type": "gallery-image", "index": dash.dependencies.ALL, "img_type": dash.dependencies.ALL}, "n_clicks"),
        State("scrollable-chart", "figure"),
        State("indicator-type-dropdown", "value"),
        State("indicator-value-dropdown", "value"),
        prevent_initial_call=True,
    )
    def highlight_chart_on_image_click(n_clicks_list, current_fig, indicator_type, indicator_value):
        """Highlight chart when image is clicked."""
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, {"display": "none"}
        
        triggered_id = ctx.triggered[0]["prop_id"]
        if not triggered_id or "gallery-image" not in triggered_id:
            return dash.no_update, dash.no_update, {"display": "none"}
        
        try:
            prop_id = json.loads(triggered_id.split(".")[0])
            point_idx = prop_id.get("index", -1)
            if point_idx < 0 or point_idx >= len(all_results):
                return dash.no_update, dash.no_update, {"display": "none"}
        except Exception:
            return dash.no_update, dash.no_update, {"display": "none"}
        
        updated_fig = update_chart_highlight(
            go.Figure(current_fig), point_idx, highlight_color
        )
        
        point_data = all_results[point_idx]
        popup_content = create_value_popup(point_data, indicator_type or indicator_type_default, indicator_value or indicator_value_default)
        popup_style = {
            "position": "fixed",
            "top": "50%",
            "left": "50%",
            "transform": "translate(-50%, -50%)",
            "background": "#1a1a2e",
            "padding": "20px",
            "border": "2px solid #4ECDC4",
            "border-radius": "10px",
            "z-index": "1000",
            "display": "block",
            "max-width": "500px",
            "color": "#eee",
        }
        
        return updated_fig, popup_content, popup_style
    
    # Callback: Close popup
    @app.callback(
        Output("value-popup", "style", allow_duplicate=True),
        Input("value-popup", "n_clicks"),
        prevent_initial_call=True,
    )
    def close_popup(n_clicks):
        """Close popup when clicked."""
        return {"display": "none"}
    
    # Callback: Update cluster display by zoom (cluster view)
    if cluster_results is not None:
        @app.callback(
            Output("cluster-plot", "figure"),
            Output("zoom-level-display", "children"),
            Input("cluster-plot", "relayoutData"),
            State("cluster-plot", "figure"),
            prevent_initial_call=True,
        )
        def update_display_by_zoom(relayout_data, current_fig):
            """Update display based on zoom level."""
            if relayout_data is None:
                return dash.no_update, "Zoom level: Default"
            
            zoom_level = _calculate_zoom_level(relayout_data)
            display_mode = "images" if zoom_level > zoom_threshold else "points"
            
            updated_fig = update_cluster_figure_display(
                go.Figure(current_fig),
                cluster_results,
                cluster_run_dirs,
                display_mode,
                zoom_level,
                cluster_config,
            )
            
            zoom_info = f"Zoom level: {zoom_level:.2f} | Display: {display_mode}"
            return updated_fig, zoom_info
        
        # Callback: Display point details on click (cluster view)
        @app.callback(
            Output("selected-point-details", "children"),
            Input("cluster-plot", "clickData"),
        )
        def display_point_details(click_data):
            """Display details of clicked point."""
            if click_data is None:
                return html.Div("Click on a point to see details", style={"color": "#888"})
            
            try:
                point_idx = click_data["points"][0]["customdata"]
                if isinstance(point_idx, list):
                    point_idx = point_idx[0]
                
                if point_idx >= len(cluster_run_dirs):
                    return html.Div("Invalid point index", style={"color": "#888"})
                
                results_dir = cluster_run_dirs[point_idx]
                results = load_results(results_dir)
                
                return create_point_detail_view(results, results_dir, point_idx)
            except Exception as e:
                return html.Div(f"Error: {str(e)}", style={"color": "#888"})
    
    return app
