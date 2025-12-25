"""Cluster and Image Correspondence View for Dashboard."""

import base64
import json
from pathlib import Path
from typing import Any, Literal

import dash
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State

from .dashboard import load_results


def _encode_image(image_path: Path) -> str:
    """Encode image to Base64 string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def _get_representative_image_path(results_dir: Path) -> Path | None:
    """Get representative image path.
    
    Priority: combined_visualization.png > thumbnail.png > other PNG files
    """
    # Try to get data directory from config
    config_path = results_dir / "config_snapshot.yaml"
    data_dir = None
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
                data_dir = Path(config.get("data_dir", ""))
        except Exception:
            pass
    
    # Check data directory first
    if data_dir and data_dir.exists():
        # Priority: combined_visualization.png
        combined_path = data_dir / "combined_visualization.png"
        if combined_path.exists():
            return combined_path
        
        # Check for other PNG files
        for png_file in data_dir.glob("*.png"):
            if "combined" in png_file.name.lower() or "visualization" in png_file.name.lower():
                return png_file
        
        # Any PNG file
        png_files = list(data_dir.glob("*.png"))
        if png_files:
            return png_files[0]
    
    # Check results directory
    thumbnail_path = results_dir / "thumbnail.png"
    if thumbnail_path.exists():
        return thumbnail_path
    
    # Check for any PNG in results directory
    png_files = list(results_dir.glob("*.png"))
    if png_files:
        return png_files[0]
    
    return None


def _get_cluster_color(cluster_id: int) -> str:
    """Get color for cluster ID."""
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    return colors[cluster_id % len(colors)]


def _calculate_zoom_level(relayout_data: dict[str, Any]) -> float:
    """Calculate zoom level from relayout data.
    
    Args:
        relayout_data: Plotly relayoutData dictionary
    
    Returns:
        Zoom level (0.0-1.0)
    """
    if "xaxis.range[0]" not in relayout_data:
        return 0.0
    
    try:
        x_range = relayout_data["xaxis.range[1]"] - relayout_data["xaxis.range[0]"]
        y_range = relayout_data["yaxis.range[1]"] - relayout_data["yaxis.range[0]"]
        
        # Default range (adjust based on actual data range)
        default_range = 10.0
        zoom_level = min(1.0, default_range / max(x_range, y_range))
        
        return zoom_level
    except Exception:
        return 0.0


def _add_representative_image(
    fig: go.Figure,
    results_dir: Path,
    position: np.ndarray,
    cluster_id: int,
    image_size: float = 50.0,
) -> None:
    """Add representative image to figure.
    
    Args:
        fig: Plotly Figure
        results_dir: Results directory
        position: Image position (x, y)
        cluster_id: Cluster ID
        image_size: Image size in coordinate units
    """
    image_path = _get_representative_image_path(results_dir)
    if image_path is None:
        return
    
    image_base64 = _encode_image(image_path)
    if not image_base64:
        return
    
    # Add image as layout image
    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{image_base64}",
            xref="x",
            yref="y",
            x=position[0],
            y=position[1],
            sizex=image_size,
            sizey=image_size,
            xanchor="center",
            yanchor="middle",
            opacity=0.8,
            layer="above",
        )
    )


def _add_point_image(
    fig: go.Figure,
    results_dir: Path,
    position: np.ndarray,
    point_idx: int,
    image_size: float = 20.0,
) -> None:
    """Add point image to figure.
    
    Args:
        fig: Plotly Figure
        results_dir: Results directory
        position: Image position (x, y)
        point_idx: Point index
        image_size: Image size in coordinate units
    """
    image_path = _get_representative_image_path(results_dir)
    if image_path is None:
        return
    
    image_base64 = _encode_image(image_path)
    if not image_base64:
        return
    
    # Add image as layout image
    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{image_base64}",
            xref="x",
            yref="y",
            x=position[0],
            y=position[1],
            sizex=image_size,
            sizey=image_size,
            xanchor="center",
            yanchor="middle",
            opacity=1.0,  # Full opacity for better visibility
            layer="above",
        )
    )


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


def create_interactive_cluster_figure(
    cluster_results: dict[str, Any],
    results_dirs: list[Path],
    config: dict[str, Any] | None = None,
) -> go.Figure:
    """Create interactive cluster figure.
    
    Args:
        cluster_results: Cluster analysis results with keys:
            - coordinates: 2D array (n_points, 2)
            - labels: Cluster labels array (n_points,)
            - cluster_centers: Cluster center coordinates (n_clusters, 2) [optional]
        results_dirs: List of results directories
        config: Configuration dictionary
    
    Returns:
        Plotly Figure
    """
    if config is None:
        config = {}
    
    coordinates = np.array(cluster_results["coordinates"])
    labels = np.array(cluster_results["labels"])
    n_clusters = len(set(labels))
    
    plot_height = config.get("plot_height", 800)
    point_size = config.get("point_size", 8)
    representative_image_size = config.get("representative_image_size", 50.0)
    
    fig = go.Figure()
    
    # Image size for all points
    point_image_size = config.get("point_image_size", 15.0)
    
    # Plot each cluster with images
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_coords = coordinates[mask]
        cluster_indices = np.where(mask)[0]
        
        # Add images for all points in this cluster
        for idx, coord in zip(cluster_indices, cluster_coords):
            _add_point_image(
                fig,
                results_dirs[idx],
                coord,
                int(idx),
                point_image_size,
            )
        
        # Add invisible points for hover interaction (behind images)
        fig.add_trace(go.Scatter(
            x=cluster_coords[:, 0],
            y=cluster_coords[:, 1],
            mode="markers",
            name=f"Cluster {cluster_id}",
            marker=dict(
                size=point_size,
                color=_get_cluster_color(cluster_id),
                opacity=0.0,  # Invisible, only for hover interaction
            ),
            customdata=cluster_indices.tolist(),
            hovertemplate="<b>Point %{customdata}</b><br>" +
                         "Cluster: %{fullData.name}<br>" +
                         "Position: (%{x:.2f}, %{y:.2f})<extra></extra>",
        ))
    
    fig.update_layout(
        title="",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        height=plot_height,
        hovermode="closest",
        clickmode="event+select",
        template="plotly_white",  # White background like the image
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif", color="#000000"),
        xaxis=dict(
            gridcolor="#e0e0e0",
            linecolor="#cccccc",
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor="#e0e0e0",
            linecolor="#cccccc",
            showgrid=True,
        ),
        showlegend=False,  # Hide legend for cleaner look
    )
    
    return fig


def update_cluster_figure_display(
    fig: go.Figure,
    cluster_results: dict[str, Any],
    results_dirs: list[Path],
    display_mode: Literal["points", "images"],
    zoom_level: float,
    config: dict[str, Any] | None = None,
) -> go.Figure:
    """Update cluster figure display based on zoom level.
    
    Args:
        fig: Original Figure
        cluster_results: Cluster analysis results
        results_dirs: List of results directories
        display_mode: Display mode ("points" or "images")
        zoom_level: Zoom level (0.0-1.0)
        config: Configuration dictionary
    
    Returns:
        Updated Figure
    """
    if config is None:
        config = {}
    
    coordinates = np.array(cluster_results["coordinates"])
    labels = np.array(cluster_results["labels"])
    n_clusters = len(set(labels))
    
    point_size = config.get("point_size", 8)
    point_image_size = config.get("point_image_size", 20.0)
    max_images_per_zoom = config.get("max_images_per_zoom", 50)
    
    # Remove existing traces (except representative images which are layout images)
    traces_to_remove = []
    for i, trace in enumerate(fig.data):
        if trace.name and not trace.name.startswith("Representative"):
            traces_to_remove.append(i)
    
    for i in reversed(traces_to_remove):
        fig.data = tuple(list(fig.data[:i]) + list(fig.data[i+1:]))
    
    # Update display based on mode
    if display_mode == "images":
        # Image mode: Add images for points (limited by zoom level)
        max_images = int(max_images_per_zoom * zoom_level)
        
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_coords = coordinates[mask]
            cluster_indices = np.where(mask)[0]
            
            # Limit number of images
            display_indices = cluster_indices[:max_images]
            display_coords = cluster_coords[:max_images]
            
            for idx, coord in zip(display_indices, display_coords):
                _add_point_image(
                    fig, results_dirs[idx], coord, int(idx), point_image_size
                )
    else:
        # Point mode: Normal point display
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_coords = coordinates[mask]
            cluster_indices = np.where(mask)[0]
            
            fig.add_trace(go.Scatter(
                x=cluster_coords[:, 0],
                y=cluster_coords[:, 1],
                mode="markers",
                name=f"Cluster {cluster_id}",
                marker=dict(
                    size=point_size,
                    color=_get_cluster_color(cluster_id),
                    opacity=0.7,
                ),
                customdata=cluster_indices.tolist(),
                hovertemplate="<b>Point %{customdata}</b><br>" +
                             "Cluster: %{fullData.name}<br>" +
                             "Position: (%{x:.2f}, %{y:.2f})<extra></extra>",
            ))
    
    return fig


def create_static_cluster_gallery(
    cluster_results: dict[str, Any],
    results_dirs: list[Path],
    config: dict[str, Any] | None = None,
) -> html.Div:
    """Create static cluster gallery with images positioned based on coordinates.
    
    Args:
        cluster_results: Cluster analysis results with keys:
            - coordinates: 2D array (n_points, 2)
            - labels: Cluster labels array (n_points,)
        results_dirs: List of results directories
        config: Configuration dictionary
    
    Returns:
        HTML Div with absolutely positioned images
    """
    if config is None:
        config = {}
    
    coordinates = np.array(cluster_results["coordinates"])
    labels = np.array(cluster_results["labels"])
    
    # Normalize coordinates to pixel positions
    x_coords = coordinates[:, 0]
    y_coords = coordinates[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0
    
    # Image size in pixels
    image_size = config.get("static_image_size", 60)
    
    # Calculate container size (add padding for images)
    container_width = max(1200, int(x_range * 10) + image_size * 2)
    container_height = max(800, int(y_range * 10) + image_size * 2)
    
    # Normalize to pixel positions
    x_pixels = ((x_coords - x_min) / x_range) * (container_width - image_size * 2) + image_size
    y_pixels = ((y_coords - y_min) / y_range) * (container_height - image_size * 2) + image_size
    
    # Create image elements
    image_elements = []
    for idx, (x_px, y_px) in enumerate(zip(x_pixels, y_pixels)):
        image_path = _get_representative_image_path(results_dirs[idx])
        if image_path is None:
            continue
        
        image_base64 = _encode_image(image_path)
        if not image_base64:
            continue
        
        image_elements.append(
            html.Img(
                src=f"data:image/png;base64,{image_base64}",
                style={
                    "position": "absolute",
                    "left": f"{x_px}px",
                    "top": f"{y_px}px",
                    "width": f"{image_size}px",
                    "height": f"{image_size}px",
                    "object-fit": "cover",
                    "border": "1px solid #e0e0e0",
                    "border-radius": "2px",
                },
            )
        )
    
    return html.Div(
        image_elements,
        style={
            "position": "relative",
            "width": f"{container_width}px",
            "height": f"{container_height}px",
            "min-width": "100%",
            "min-height": "100vh",
            "background": "#ffffff",
            "overflow": "auto",
            "margin": "0 auto",
        },
    )


def create_point_detail_view(
    results: dict[str, Any],
    results_dir: Path,
    point_idx: int,
) -> html.Div:
    """Create point detail view.
    
    Args:
        results: Analysis results
        results_dir: Results directory
        point_idx: Point index
    
    Returns:
        Detail view HTML div
    """
    image_path = _get_representative_image_path(results_dir)
    image_html = ""
    if image_path:
        image_base64 = _encode_image(image_path)
        if image_base64:
            image_html = html.Img(
                src=f"data:image/png;base64,{image_base64}",
                style={"width": "300px", "height": "auto", "margin-bottom": "10px"},
            )
    
    all_values = extract_all_indicator_values(results)
    
    return html.Div([
        html.H3(f"Point {point_idx}: {results_dir.name}", style={"color": "#4ECDC4"}),
        image_html,
        html.Pre(
            json.dumps(all_values, indent=2),
            style={
                "background": "#16213e",
                "padding": "10px",
                "border-radius": "5px",
                "color": "#eee",
            },
        ),
    ])


def create_cluster_image_correspondence_view(
    cluster_results: dict[str, Any],
    results_dirs: list[Path],
    config: dict[str, Any] | None = None,
) -> dash.Dash:
    """Create cluster and image correspondence view.
    
    Args:
        cluster_results: Cluster analysis results with keys:
            - coordinates: 2D array (n_points, 2)
            - labels: Cluster labels array (n_points,)
            - cluster_centers: Cluster center coordinates (n_clusters, 2) [optional]
        results_dirs: List of results directories
        config: Configuration dictionary
    
    Returns:
        Dash application
    """
    if config is None:
        config = {}
    
    # Create cluster figure
    cluster_fig = create_interactive_cluster_figure(
        cluster_results, results_dirs, config
    )
    
    # Create app
    app = dash.Dash(
        __name__,
        title="Cluster & Image Correspondence",
        suppress_callback_exceptions=True,
    )
    
    zoom_threshold = config.get("zoom_threshold", 0.5)
    
    app.layout = html.Div([
        html.H1(
            "Cluster & Image Correspondence",
            style={
                "text-align": "center",
                "color": "#4ECDC4",
                "margin-bottom": "20px",
            },
        ),
        
        # Cluster plot (zoomable)
        dcc.Graph(
            id="cluster-plot",
            figure=cluster_fig,
            style={"height": f"{config.get('plot_height', 800)}px"},
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
    ], style={
        "background": "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)",
        "min-height": "100vh",
        "padding": "20px",
    })
    
    # Callback: Update display by zoom level
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
        
        # Calculate zoom level
        zoom_level = _calculate_zoom_level(relayout_data)
        
        # Determine display mode
        display_mode = "images" if zoom_level > zoom_threshold else "points"
        
        # Update figure
        updated_fig = update_cluster_figure_display(
            go.Figure(current_fig),
            cluster_results,
            results_dirs,
            display_mode,
            zoom_level,
            config,
        )
        
        zoom_info = f"Zoom level: {zoom_level:.2f} | Display: {display_mode}"
        
        return updated_fig, zoom_info
    
    # Callback: Display point details on click
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
            
            if point_idx >= len(results_dirs):
                return html.Div("Invalid point index", style={"color": "#888"})
            
            results_dir = results_dirs[point_idx]
            results = load_results(results_dir)
            
            return create_point_detail_view(results, results_dir, point_idx)
        except Exception as e:
            return html.Div(f"Error: {str(e)}", style={"color": "#888"})
    
    return app

