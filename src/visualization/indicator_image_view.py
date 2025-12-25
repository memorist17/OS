"""Indicator and Image Correspondence View for Dashboard."""

import base64
import json
from pathlib import Path
from typing import Any, Literal

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State

from .dashboard import load_results


def extract_indicator_data(
    results: dict[str, Any],
    indicator_type: Literal["mfa", "lacunarity", "percolation"],
    indicator_value: str = "D(0)",
) -> dict[str, Any]:
    """Extract indicator data for plotting.
    
    Args:
        results: Analysis results dictionary
        indicator_type: Type of indicator
        indicator_value: Name of indicator value to extract
    
    Returns:
        Dictionary with x, y data for plotting
    """
    if indicator_type == "mfa":
        if "mfa_dimensions" in results:
            df = results["mfa_dimensions"]
            return {"x": df["q"].values, "y": df["D_q"].values}
        elif "mfa_spectrum" in results:
            df = results["mfa_spectrum"]
            if indicator_value == "D(0)":
                # Extract D(0) from spectrum
                q0_idx = df[df["q"] == 0].index
                if len(q0_idx) > 0:
                    alpha = df.loc[q0_idx[0], "alpha"]
                    return {"x": [0], "y": [alpha]}
        return {"x": [], "y": []}
    
    elif indicator_type == "lacunarity":
        if "lacunarity" in results:
            df = results["lacunarity"]
            # Check column names (may be "lambda" or "lambda_r")
            if "lambda_r" in df.columns:
                return {"r": df["r"].values, "lambda_r": df["lambda_r"].values}
            elif "lambda" in df.columns:
                return {"r": df["r"].values, "lambda_r": df["lambda"].values}
        return {"r": [], "lambda_r": []}
    
    elif indicator_type == "percolation":
        if "percolation" in results:
            df = results["percolation"]
            return {"d": df["d"].values, "giant_fraction": df["giant_fraction"].values}
        return {"d": [], "giant_fraction": []}
    
    return {}


def extract_indicator_value(
    results: dict[str, Any],
    indicator_type: Literal["mfa", "lacunarity", "percolation"],
    indicator_value: str,
) -> float:
    """Extract a specific indicator value.
    
    Args:
        results: Analysis results dictionary
        indicator_type: Type of indicator
        indicator_value: Name of indicator value
    
    Returns:
        Indicator value
    """
    if indicator_type == "mfa":
        if "mfa_dimensions" in results:
            df = results["mfa_dimensions"]
            if indicator_value == "D(0)":
                q0 = df[df["q"] == 0]
                if len(q0) > 0:
                    return float(q0["D_q"].values[0])
        return 0.0
    
    elif indicator_type == "lacunarity":
        if "lacunarity_fit" in results:
            fit = results["lacunarity_fit"]
            if indicator_value == "beta":
                return float(fit.get("beta", 0.0))
        return 0.0
    
    elif indicator_type == "percolation":
        if "percolation_stats" in results:
            stats = results["percolation_stats"]
            if indicator_value == "d_critical":
                return float(stats.get("d_critical_50", 0.0))
        return 0.0
    
    return 0.0


def extract_all_indicator_values(results: dict[str, Any]) -> dict[str, float]:
    """Extract all indicator values from results."""
    values = {}
    
    # MFA values
    if "mfa_dimensions" in results:
        df = results["mfa_dimensions"]
        for q in [0, 1, 2]:
            q_data = df[df["q"] == q]
            if len(q_data) > 0:
                values[f"D({q})"] = float(q_data["D_q"].values[0])
    
    # Lacunarity values
    if "lacunarity_fit" in results:
        fit = results["lacunarity_fit"]
        values["beta"] = float(fit.get("beta", 0.0))
        values["R2"] = float(fit.get("R2", 0.0))
    
    # Percolation values
    if "percolation_stats" in results:
        stats = results["percolation_stats"]
        values["d_critical"] = float(stats.get("d_critical_50", 0.0))
        values["transition_width"] = float(stats.get("transition_width", 0.0))
    
    return values


def _get_xaxis_label(indicator_type: str) -> str:
    """Get x-axis label for indicator type."""
    labels = {
        "mfa": "q",
        "lacunarity": "Box size r [px]",
        "percolation": "Distance threshold d [m]",
    }
    return labels.get(indicator_type, "X")


def _get_yaxis_label(indicator_type: str, indicator_value: str) -> str:
    """Get y-axis label for indicator type."""
    if indicator_type == "mfa":
        return "D(q)"
    elif indicator_type == "lacunarity":
        return "Λ(r)"
    elif indicator_type == "percolation":
        return "Giant Component Fraction"
    return "Y"


def _encode_image(image_path: Path) -> str:
    """Encode image to Base64 string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def _get_location_images(
    results: dict[str, Any],
    results_dir: Path,
) -> dict[str, Path]:
    """Get location image paths.
    
    Args:
        results: Analysis results
        results_dir: Results directory path
    
    Returns:
        Dictionary mapping image type to path
    """
    images = {}
    
    # Get data directory from config
    if "config" in results:
        config = results["config"]
        data_dir = Path(config.get("data_dir", ""))
        if data_dir.exists():
            # Look for PNG images first (preferred)
            combined_path = data_dir / "combined_visualization.png"
            if combined_path.exists():
                images["combined"] = combined_path
            
            # Buildings visualization
            buildings_img_path = data_dir / "buildings_visualization.png"
            if buildings_img_path.exists():
                images["buildings"] = buildings_img_path
            
            # Roads visualization
            roads_img_path = data_dir / "roads_raster_comparison.png"
            if roads_img_path.exists():
                images["roads"] = roads_img_path
            
            # Network visualization (look for PNG files with network in name)
            for png_file in data_dir.glob("*network*.png"):
                images["network"] = png_file
                break
            
            # Fallback to raster files if no PNG found
            if "combined" not in images:
                buildings_path = data_dir / "buildings_binary.npy"
                if buildings_path.exists():
                    images["buildings"] = buildings_path
                
                roads_path = data_dir / "roads_weighted.npy"
                if roads_path.exists():
                    images["roads"] = roads_path
    
    # Thumbnail image (if exists)
    thumbnail_path = results_dir / "thumbnail.png"
    if thumbnail_path.exists():
        images["thumbnail"] = thumbnail_path
    
    # Also check data directory for thumbnail
    if "config" in results:
        config = results["config"]
        data_dir = Path(config.get("data_dir", ""))
        if data_dir.exists():
            data_thumbnail = data_dir / "thumbnail.png"
            if data_thumbnail.exists() and "thumbnail" not in images:
                images["thumbnail"] = data_thumbnail
    
    return images


def create_horizontal_scrollable_chart(
    all_results: list[dict[str, Any]],
    indicator_type: str,
    indicator_value: str,
    config: dict[str, Any] | None = None,
) -> go.Figure:
    """Create horizontally scrollable chart.
    
    Args:
        all_results: List of analysis results for all points
        indicator_type: Type of indicator
        indicator_value: Indicator value name
        config: Configuration dictionary
    
    Returns:
        Plotly Figure with horizontal scroll
    """
    if config is None:
        config = {}
    
    # Use viewport height for full screen charts
    chart_height = config.get("chart_height", "100vh")
    chart_min_width = config.get("chart_min_width", 1200)
    
    fig = go.Figure()
    
    # Helper function to get point name from results
    def get_point_name(results: dict[str, Any], idx: int) -> str:
        """Get display name or site_id from results."""
        default_name = f"Point {idx}"
        if "config" in results:
            config = results["config"]
            site_metadata = config.get("site_metadata", {})
            meta_info = site_metadata.get("meta_info", {})
            # Try display_name first, then site_id (lat_lon format)
            display_name = meta_info.get("display_name")
            if display_name:
                return display_name
            site_id = meta_info.get("site_id") or site_metadata.get("site_id")
            if site_id:
                return site_id
        return default_name
    
    # Plot data for each point
    for idx, results in enumerate(all_results):
        data = extract_indicator_data(results, indicator_type, indicator_value)
        point_name = get_point_name(results, idx)
        
        if indicator_type == "mfa":
            if "x" in data and len(data["x"]) > 0:
                fig.add_trace(go.Scatter(
                    x=data["x"],
                    y=data["y"],
                    mode="lines+markers",
                    name=point_name,
                    line=dict(width=2),
                    marker=dict(size=6),
                    customdata=[idx] * len(data["x"]),
                ))
        
        elif indicator_type == "lacunarity":
            if "r" in data and len(data["r"]) > 0:
                fig.add_trace(go.Scatter(
                    x=data["r"],
                    y=data["lambda_r"],
                    mode="lines+markers",
                    name=point_name,
                    line=dict(width=2),
                    marker=dict(size=6),
                    customdata=[idx] * len(data["r"]),
                ))
        
        elif indicator_type == "percolation":
            if "d" in data and len(data["d"]) > 0:
                fig.add_trace(go.Scatter(
                    x=data["d"],
                    y=data["giant_fraction"],
                    mode="lines+markers",
                    name=point_name,
                    line=dict(width=2),
                    marker=dict(size=6),
                    customdata=[idx] * len(data["d"]),
                ))
    
    # Square chart: use chart_height for both width and height
    if isinstance(chart_height, str) and chart_height.endswith("vh"):
        # If it's viewport height, use a fixed pixel size for square
        square_size = config.get("square_chart_size", 400)
    else:
        square_size = chart_height
    
    # Ensure square dimensions
    width = square_size
    height = square_size
    
    # Minimize margins and show legend while maintaining square
    # Use autosize to fill container width, but maintain square aspect ratio
    fig.update_layout(
        title="",
        xaxis_title=_get_xaxis_label(indicator_type),
        yaxis_title=_get_yaxis_label(indicator_type, indicator_value),
        height=height,
        width=None,  # Let it autosize to container width
        hovermode="closest",
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif", color="#c9d1d9", size=9),
        autosize=True,  # Enable autosize to fill container
        showlegend=True,  # Show legend
        legend=dict(
            x=1.02,  # Position legend to the right
            y=1,
            xanchor="left",
            yanchor="top",
            font=dict(size=8),
        ),
        margin=dict(l=30, r=10, t=10, b=30),  # Minimize margins
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


def create_image_gallery(
    all_results: list[dict[str, Any]],
    results_dirs: list[Path],
    config: dict[str, Any] | None = None,
) -> list[html.Div]:
    """Create image gallery.
    
    Args:
        all_results: List of analysis results
        results_dirs: List of results directories
        config: Configuration dictionary
    
    Returns:
        List of gallery item HTML divs
    """
    if config is None:
        config = {}
    
    gallery_items = []
    
    for idx, (results, results_dir) in enumerate(zip(all_results, results_dirs)):
        image_paths = _get_location_images(results, results_dir)
        
        # Get location name from config
        location_name = results_dir.name  # Default to run_id
        if "config" in results:
            config = results["config"]
            site_metadata = config.get("site_metadata", {})
            meta_info = site_metadata.get("meta_info", {})
            # Try to get display_name or site_id
            location_name = meta_info.get("display_name") or meta_info.get("site_id") or location_name
        
        # Prioritize combined visualization, then thumbnail, then others
        priority_order = ["combined", "thumbnail", "buildings", "roads", "network"]
        displayed_images = []
        
        for img_type in priority_order:
            if img_type in image_paths:
                img_path = image_paths[img_type]
                # Skip .npy files (raster data)
                if img_path.suffix == ".npy":
                    continue
                
                image_base64 = _encode_image(img_path)
                if image_base64:
                    displayed_images.append((img_type, img_path, image_base64))
                    break  # Only show one image per location
        
        # If no priority image found, try any other image
        if not displayed_images:
            for img_type, img_path in image_paths.items():
                if img_path.suffix == ".npy":
                    continue
                
                image_base64 = _encode_image(img_path)
                if image_base64:
                    displayed_images.append((img_type, img_path, image_base64))
                    break
        
        for img_type, img_path, image_base64 in displayed_images:
            
            gallery_items.append(
                html.Div([
                    html.Img(
                        src=f"data:image/png;base64,{image_base64}",
                        style={
                            "width": "100%",
                            "height": "auto",
                            "border": "2px solid #ccc",
                            "border-radius": "5px",
                            "cursor": "pointer",
                        },
                        id={"type": "gallery-image", "index": idx, "img_type": img_type},
                    ),
                    html.P(
                        location_name,
                        style={"text-align": "center", "margin-top": "5px", "color": "#c9d1d9", "font-size": "0.9em"},
                    ),
                ], style={"text-align": "center"}),
            )
    
    return gallery_items


def update_chart_highlight(
    fig: go.Figure,
    point_idx: int,
    highlight_color: str = "rgba(255, 0, 0, 1.0)",
) -> go.Figure:
    """Highlight specific trace in chart.
    
    Args:
        fig: Plotly Figure
        point_idx: Index of point to highlight
        highlight_color: Color for highlight
    
    Returns:
        Updated Figure
    """
    # Make all traces semi-transparent
    for trace in fig.data:
        if hasattr(trace, "line") and trace.line is not None:
            trace.line.color = "rgba(128, 128, 128, 0.3)"
        if hasattr(trace, "marker") and trace.marker is not None:
            trace.marker.color = "rgba(128, 128, 128, 0.3)"
    
    # Highlight selected trace
    if point_idx < len(fig.data):
        selected_trace = fig.data[point_idx]
        if hasattr(selected_trace, "line") and selected_trace.line is not None:
            selected_trace.line.color = highlight_color
            selected_trace.line.width = 4
        if hasattr(selected_trace, "marker") and selected_trace.marker is not None:
            selected_trace.marker.color = highlight_color
            selected_trace.marker.size = 10
    
    return fig


def create_value_popup(
    point_data: dict[str, Any],
    indicator_type: str,
    indicator_value: str,
) -> html.Div:
    """Create value popup content.
    
    Args:
        point_data: Point analysis results
        indicator_type: Type of indicator
        indicator_value: Indicator value name
    
    Returns:
        Popup HTML div
    """
    value = extract_indicator_value(point_data, indicator_type, indicator_value)
    all_values = extract_all_indicator_values(point_data)
    
    return html.Div([
        html.H4("Indicator Value", style={"color": "#4ECDC4"}),
        html.P(f"Type: {indicator_type.upper()}"),
        html.P(f"Value: {indicator_value} = {value:.4f}"),
        html.Hr(),
        html.H5("All Values", style={"color": "#4ECDC4"}),
        html.Pre(
            json.dumps(all_values, indent=2),
            style={"background": "#16213e", "padding": "10px", "border-radius": "5px"},
        ),
        html.Button(
            "Close",
            id="close-popup",
            n_clicks=0,
            style={
                "margin-top": "10px",
                "padding": "10px 20px",
                "background": "#4ECDC4",
                "color": "#1a1a2e",
                "border": "none",
                "border-radius": "5px",
                "cursor": "pointer",
            },
        ),
    ])


def create_indicator_image_correspondence_view(
    results_dirs: list[Path],
    indicator_type: Literal["mfa", "lacunarity", "percolation"] = "mfa",
    indicator_value: str = "D(0)",
    config: dict[str, Any] | None = None,
) -> dash.Dash:
    """Create indicator and image correspondence view.
    
    Args:
        results_dirs: List of results directories
        indicator_type: Type of indicator
        indicator_value: Indicator value name
        config: Configuration dictionary
    
    Returns:
        Dash application
    """
    if config is None:
        config = {}
    
    # Load all results
    all_results = [load_results(d) for d in results_dirs]
    
    # Create chart
    chart_fig = create_horizontal_scrollable_chart(
        all_results, indicator_type, indicator_value, config
    )
    
    # Create gallery
    gallery_items = create_image_gallery(all_results, results_dirs, config)
    
    # Create app
    app = dash.Dash(
        __name__,
        title="Indicator & Image Correspondence",
        suppress_callback_exceptions=True,
    )
    
    highlight_color = config.get("highlight_color", "rgba(255, 0, 0, 1.0)")
    
    app.layout = html.Div([
        html.H1(
            f"Indicator & Image Correspondence: {indicator_type.upper()}",
            style={
                "text-align": "center",
                "color": "#4ECDC4",
                "margin-bottom": "20px",
            },
        ),
        
        # Chart area (horizontally scrollable)
        html.Div([
            dcc.Graph(
                id="scrollable-chart",
                figure=chart_fig,
                style={"height": f"{config.get('chart_height', 400)}px"},
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
        
        # Popup area
        dcc.Store(id="selected-point-store", data=None),
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
        "padding": "20px",
    })
    
    # Callback: Image click -> Chart highlight
    @app.callback(
        Output("scrollable-chart", "figure"),
        Output("value-popup", "children"),
        Output("value-popup", "style"),
        Input({"type": "gallery-image", "index": dash.dependencies.ALL, "img_type": dash.dependencies.ALL}, "n_clicks"),
        State("scrollable-chart", "figure"),
        prevent_initial_call=True,
    )
    def highlight_chart_on_image_click(n_clicks_list, current_fig):
        """Highlight chart when image is clicked."""
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, {"display": "none"}
        
        # Find which image was clicked
        triggered_id = ctx.triggered[0]["prop_id"]
        if not triggered_id or "gallery-image" not in triggered_id:
            return dash.no_update, dash.no_update, {"display": "none"}
        
        # Parse the ID to get point index
        try:
            import json
            prop_id = json.loads(triggered_id.split(".")[0])
            point_idx = prop_id.get("index", -1)
            if point_idx < 0 or point_idx >= len(all_results):
                return dash.no_update, dash.no_update, {"display": "none"}
        except Exception:
            return dash.no_update, dash.no_update, {"display": "none"}
        
        # Update chart highlight
        updated_fig = update_chart_highlight(
            go.Figure(current_fig), point_idx, highlight_color
        )
        
        # Create popup content
        point_data = all_results[point_idx]
        popup_content = create_value_popup(point_data, indicator_type, indicator_value)
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
        Output("value-popup", "style"),
        Input("close-popup", "n_clicks"),
        prevent_initial_call=True,
    )
    def close_popup(n_clicks):
        """Close popup when close button is clicked."""
        return {"display": "none"}
    
    return app

