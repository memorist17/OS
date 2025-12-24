"""Gallery Dashboard for comparing multiple places with structure visualizations.

This dashboard provides a gallery view where:
- Overall charts (MFA spectrum, lacunarity, percolation) are always visible
- Each location's raster images (buildings, roads) and network graphs
  are displayed in a gallery format for comparison
- Interactive selection allows detailed exploration of each location
"""

from pathlib import Path
from typing import Any

import dash
import networkx as nx
import plotly.graph_objects as go
from dash import dcc, html
from plotly.subplots import make_subplots

from .dashboard import load_results


def load_all_places_for_gallery(outputs_dir: Path) -> dict[str, dict[str, Any]]:
    """
    Load all place results from outputs directory for gallery view.

    Returns:
        Dictionary mapping run_id to results dict with display_name and site_id
    """
    all_results = {}

    if not outputs_dir.exists():
        return all_results

    # Find all run directories
    run_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]

    for run_dir in sorted(run_dirs, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            results = load_results(run_dir)
            if "config" in results:
                # Extract site info
                config = results["config"]
                site_metadata = config.get("site_metadata", {})
                site_id = site_metadata.get("meta_info", {}).get("site_id", "unknown")
                if site_id == "unknown":
                    site_id = site_metadata.get("site_id", run_dir.name)

                # Create display name from resolved_places.json if available
                display_name = site_id
                places_file = Path("data/resolved_places.json")
                if places_file.exists():
                    import json
                    with open(places_file) as f:
                        places = json.load(f)
                    for place in places:
                        place_site_id = f"{place['latitude']}_{place['longitude']}"
                        if place_site_id == site_id:
                            display_name = place.get("display_name", site_id)
                            break

                results["display_name"] = display_name
                results["site_id"] = site_id
                results["run_id"] = run_dir.name
                all_results[run_dir.name] = results
        except Exception as e:
            print(f"Error loading {run_dir}: {e}")
            continue

    return all_results


def create_overview_mfa_figure(all_results: dict[str, dict[str, Any]]) -> go.Figure:
    """Create overview MFA comparison figure for all places."""
    fig = go.Figure()

    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#F39C12", "#9B59B6",
        "#E74C3C", "#3498DB", "#1ABC9C", "#F1C40F", "#E67E22",
        "#95A5A6", "#34495E", "#16A085", "#27AE60", "#2980B9",
        "#8E44AD", "#C0392B", "#D35400"
    ]

    for idx, (run_id, results) in enumerate(all_results.items()):
        if "mfa_spectrum" not in results:
            continue

        display_name = results.get("display_name", run_id)
        color = colors[idx % len(colors)]

        df = results["mfa_spectrum"]

        # f(α) vs α
        fig.add_trace(
            go.Scatter(
                x=df["alpha"],
                y=df["f_alpha"],
                mode="lines+markers",
                name=display_name,
                line=dict(color=color, width=2),
                marker=dict(size=4),
                hovertemplate=f"{display_name}<br>α: %{{x:.3f}}<br>f(α): %{{y:.3f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Multifractal Spectrum f(α) - All Locations",
        height=350,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(family="Noto Sans JP, sans-serif", color="#eee"),
        legend=dict(
            x=1.02, y=1, xanchor="left",
            font=dict(size=10),
            itemsizing="constant",
        ),
        margin=dict(l=60, r=150, t=50, b=50),
        xaxis_title="α (Hölder exponent)",
        yaxis_title="f(α)",
        xaxis=dict(gridcolor="#2a2a4e"),
        yaxis=dict(gridcolor="#2a2a4e"),
    )

    return fig


def create_overview_lacunarity_figure(all_results: dict[str, dict[str, Any]]) -> go.Figure:
    """Create overview lacunarity comparison figure for all places."""
    fig = go.Figure()

    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#F39C12", "#9B59B6",
        "#E74C3C", "#3498DB", "#1ABC9C", "#F1C40F", "#E67E22",
        "#95A5A6", "#34495E", "#16A085", "#27AE60", "#2980B9",
        "#8E44AD", "#C0392B", "#D35400"
    ]

    for idx, (run_id, results) in enumerate(all_results.items()):
        if "lacunarity" not in results:
            continue

        display_name = results.get("display_name", run_id)
        color = colors[idx % len(colors)]

        df = results["lacunarity"]

        fig.add_trace(
            go.Scatter(
                x=df["r"],
                y=df["lambda"],
                mode="lines+markers",
                name=display_name,
                line=dict(color=color, width=2),
                marker=dict(size=4),
                hovertemplate=f"{display_name}<br>r: %{{x}}<br>Λ(r): %{{y:.3f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Lacunarity Λ(r) - All Locations",
        height=350,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(family="Noto Sans JP, sans-serif", color="#eee"),
        legend=dict(
            x=1.02, y=1, xanchor="left",
            font=dict(size=10),
            itemsizing="constant",
        ),
        margin=dict(l=60, r=150, t=50, b=50),
        xaxis_title="Box size r [px]",
        yaxis_title="Λ(r)",
        xaxis=dict(gridcolor="#2a2a4e", type="log"),
        yaxis=dict(gridcolor="#2a2a4e", type="log"),
    )

    return fig


def create_overview_percolation_figure(all_results: dict[str, dict[str, Any]]) -> go.Figure:
    """Create overview percolation comparison figure for all places."""
    fig = go.Figure()

    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#F39C12", "#9B59B6",
        "#E74C3C", "#3498DB", "#1ABC9C", "#F1C40F", "#E67E22",
        "#95A5A6", "#34495E", "#16A085", "#27AE60", "#2980B9",
        "#8E44AD", "#C0392B", "#D35400"
    ]

    for idx, (run_id, results) in enumerate(all_results.items()):
        if "percolation" not in results:
            continue

        display_name = results.get("display_name", run_id)
        color = colors[idx % len(colors)]

        df = results["percolation"]

        fig.add_trace(
            go.Scatter(
                x=df["d"],
                y=df["giant_fraction"],
                mode="lines+markers",
                name=display_name,
                line=dict(color=color, width=2),
                marker=dict(size=4),
                hovertemplate=(
                    f"{display_name}<br>d: %{{x:.1f}}<br>"
                    "Giant Fraction: %{y:.3f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Giant Component Fraction - All Locations",
        height=350,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(family="Noto Sans JP, sans-serif", color="#eee"),
        legend=dict(
            x=1.02, y=1, xanchor="left",
            font=dict(size=10),
            itemsizing="constant",
        ),
        margin=dict(l=60, r=150, t=50, b=50),
        xaxis_title="Distance threshold d [m]",
        yaxis_title="Giant fraction",
        xaxis=dict(gridcolor="#2a2a4e"),
        yaxis=dict(gridcolor="#2a2a4e"),
    )

    return fig


def create_clustering_summary_figure(all_results: dict[str, dict[str, Any]]) -> go.Figure:
    """Create clustering/dimension comparison bar chart."""
    place_names = []
    d0_values = []
    d1_values = []
    d2_values = []
    beta_values = []
    d_critical_values = []

    for run_id, results in all_results.items():
        display_name = results.get("display_name", run_id)

        # Get MFA dimensions
        if "mfa_dimensions" in results:
            dq = results["mfa_dimensions"]
            d0 = dq[dq["q"] == 0]["D_q"].values[0] if 0 in dq["q"].values else None
            d1_mask = abs(dq["q"] - 1) < 0.5
            d1 = dq[d1_mask]["D_q"].values[0] if len(dq[d1_mask]) > 0 else None
            d2 = dq[dq["q"] == 2]["D_q"].values[0] if 2 in dq["q"].values else None
        else:
            d0, d1, d2 = None, None, None

        # Get lacunarity beta
        if "lacunarity_fit" in results:
            beta = results["lacunarity_fit"].get("beta", None)
        else:
            beta = None

        # Get percolation critical distance
        if "percolation_stats" in results:
            d_critical = results["percolation_stats"].get("d_critical_50", None)
        else:
            d_critical = None

        if d0 is not None:
            place_names.append(display_name)
            d0_values.append(d0)
            d1_values.append(d1)
            d2_values.append(d2)
            beta_values.append(beta if beta else 0)
            d_critical_values.append(d_critical if d_critical else 0)

    if not place_names:
        fig = go.Figure()
        fig.add_annotation(
            text="No clustering data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Fractal Dimensions D(q)", "Lacunarity β / Critical Distance"),
        horizontal_spacing=0.15,
    )

    # D(q) bar chart
    fig.add_trace(
        go.Bar(name="D(0)", x=place_names, y=d0_values, marker_color="#FF6B6B"),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name="D(1)", x=place_names, y=d1_values, marker_color="#4ECDC4"),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name="D(2)", x=place_names, y=d2_values, marker_color="#45B7D1"),
        row=1, col=1
    )

    # Beta and d_critical bar chart
    fig.add_trace(
        go.Bar(name="β (Lacunarity)", x=place_names, y=beta_values, marker_color="#F39C12"),
        row=1, col=2
    )

    fig.update_layout(
        height=350,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(family="Noto Sans JP, sans-serif", color="#eee"),
        legend=dict(x=1.02, y=1, xanchor="left"),
        margin=dict(l=60, r=150, t=50, b=100),
        barmode="group",
    )

    fig.update_xaxes(tickangle=45, gridcolor="#2a2a4e")
    fig.update_yaxes(gridcolor="#2a2a4e")

    return fig


def create_location_building_figure(results: dict[str, Any]) -> go.Figure | None:
    """Create building raster figure for a single location."""
    if "buildings_raster" not in results:
        return None

    buildings = results["buildings_raster"]
    display_name = results.get("display_name", "Unknown")

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=buildings,
            colorscale="gray",
            showscale=False,
            hovertemplate="X: %{x}<br>Y: %{y}<br>Value: %{z}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=f"Buildings - {display_name}", font=dict(size=12)),
        height=250,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(family="Noto Sans JP, sans-serif", color="#eee"),
        showlegend=False,
        margin=dict(l=30, r=30, t=40, b=30),
        xaxis=dict(showticklabels=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(showticklabels=False),
    )

    return fig


def create_location_roads_figure(results: dict[str, Any]) -> go.Figure | None:
    """Create roads raster figure for a single location."""
    if "roads_raster" not in results:
        return None

    roads = results["roads_raster"]
    display_name = results.get("display_name", "Unknown")

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=roads,
            colorscale="viridis",
            showscale=True,
            colorbar=dict(thickness=10, len=0.5),
            hovertemplate="X: %{x}<br>Y: %{y}<br>Weight: %{z:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=f"Roads - {display_name}", font=dict(size=12)),
        height=250,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(family="Noto Sans JP, sans-serif", color="#eee"),
        showlegend=False,
        margin=dict(l=30, r=60, t=40, b=30),
        xaxis=dict(showticklabels=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(showticklabels=False),
    )

    return fig


def create_location_network_figure(
    results: dict[str, Any], max_nodes: int = 2000
) -> go.Figure | None:
    """Create network graph figure for a single location."""
    if "network_path" not in results:
        return None

    try:
        network_path = Path(results["network_path"])
        if not network_path.exists():
            return None

        G = nx.read_graphml(str(network_path))

        if G.number_of_nodes() == 0:
            return None

        display_name = results.get("display_name", "Unknown")

        # Sample nodes for visualization if too large
        if G.number_of_nodes() > max_nodes:
            import random
            nodes_to_keep = random.sample(list(G.nodes()), max_nodes)
            G = G.subgraph(nodes_to_keep).copy()

        # Use spring layout
        pos = nx.spring_layout(G, k=0.1, iterations=20, seed=42)

        # Extract coordinates
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig = go.Figure()

        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.3, color="#888"),
            hoverinfo='none',
            mode='lines',
        ))

        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(size=2, color="#4ECDC4"),
            hoverinfo='skip',
        ))

        fig.update_layout(
            title=dict(
                text=(
                    f"Network - {display_name}<br>"
                    f"<sub>N={G.number_of_nodes()}, E={G.number_of_edges()}</sub>"
                ),
                font=dict(size=12)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=30, l=30, r=30, t=50),
            height=250,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
            font=dict(family="Noto Sans JP, sans-serif", color="#eee"),
        )

        return fig

    except Exception as e:
        print(f"Error creating network figure: {e}")
        return None


def create_location_metrics_card(results: dict[str, Any]) -> html.Div:
    """Create metrics summary card for a single location."""
    display_name = results.get("display_name", "Unknown")

    # Extract key metrics
    metrics = []

    # MFA dimensions
    if "mfa_dimensions" in results:
        dq = results["mfa_dimensions"]
        d0 = dq[dq["q"] == 0]["D_q"].values[0] if 0 in dq["q"].values else None
        d2 = dq[dq["q"] == 2]["D_q"].values[0] if 2 in dq["q"].values else None
        if d0 is not None:
            metrics.append(f"D(0)={d0:.3f}")
        if d2 is not None:
            metrics.append(f"D(2)={d2:.3f}")

    # Lacunarity
    if "lacunarity_fit" in results:
        beta = results["lacunarity_fit"].get("beta", None)
        if beta is not None:
            metrics.append(f"β={beta:.3f}")

    # Percolation
    if "percolation_stats" in results:
        d_c = results["percolation_stats"].get("d_critical_50", None)
        if d_c is not None:
            metrics.append(f"d_c={d_c:.1f}m")

    header_style = {
        "margin": "0 0 5px 0",
        "color": "#4ECDC4",
        "fontSize": "14px"
    }
    return html.Div([
        html.H4(display_name, style=header_style),
        html.P(" | ".join(metrics) if metrics else "No metrics", style={
            "margin": 0,
            "fontSize": "11px",
            "fontFamily": "'JetBrains Mono', monospace",
            "color": "#aaa"
        })
    ], style={
        "padding": "8px",
        "backgroundColor": "rgba(255,255,255,0.03)",
        "borderRadius": "8px",
        "marginBottom": "8px",
    })


def create_gallery_dashboard(outputs_dir: Path | str) -> dash.Dash:
    """
    Create gallery dashboard for comparing multiple places with structure visualizations.

    The dashboard displays:
    - Top section: Overview charts (MFA, Lacunarity, Percolation comparisons)
    - Bottom section: Gallery of location images (Buildings, Roads, Network for each location)

    Args:
        outputs_dir: Path to outputs directory containing run directories

    Returns:
        Configured Dash application
    """
    outputs_dir = Path(outputs_dir)

    # Load all places
    all_results = load_all_places_for_gallery(outputs_dir)

    if not all_results:
        # Fallback to single place dashboard
        from .dashboard import create_dashboard
        return create_dashboard(outputs_dir)

    app = dash.Dash(
        __name__,
        title="Urban Structure Gallery - 構造可視化",
        suppress_callback_exceptions=True,
    )

    # Custom CSS with gallery-specific styles
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
                    max-width: 1800px;
                    margin: 0 auto;
                    padding: 15px;
                }
                h1 {
                    text-align: center;
                    background: linear-gradient(120deg, #FF6B6B, #4ECDC4, #45B7D1);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-size: 2em;
                    margin-bottom: 5px;
                }
                .subtitle {
                    text-align: center;
                    color: #888;
                    margin-bottom: 15px;
                    font-size: 0.9em;
                }
                .overview-section {
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 12px;
                    padding: 15px;
                    margin-bottom: 15px;
                    border: 1px solid rgba(78, 205, 196, 0.3);
                }
                .overview-section h2 {
                    color: #4ECDC4;
                    margin: 0 0 10px 0;
                    padding-bottom: 8px;
                    border-bottom: 2px solid #4ECDC4;
                    font-size: 1.2em;
                }
                .gallery-section {
                    background: rgba(255, 255, 255, 0.02);
                    border-radius: 12px;
                    padding: 15px;
                    margin-bottom: 15px;
                }
                .gallery-section h2 {
                    color: #FF6B6B;
                    margin: 0 0 15px 0;
                    padding-bottom: 8px;
                    border-bottom: 2px solid #FF6B6B;
                    font-size: 1.2em;
                }
                .gallery-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                    gap: 15px;
                }
                .location-card {
                    background: rgba(255, 255, 255, 0.03);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 10px;
                    padding: 10px;
                    transition: all 0.3s ease;
                }
                .location-card:hover {
                    border-color: rgba(78, 205, 196, 0.5);
                    background: rgba(78, 205, 196, 0.05);
                }
                .chart-row {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 15px;
                }
                @media (max-width: 1200px) {
                    .chart-row {
                        grid-template-columns: 1fr;
                    }
                }
                .overview-charts {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 10px;
                }
                @media (max-width: 1000px) {
                    .overview-charts {
                        grid-template-columns: 1fr;
                    }
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

    # Create overview figures
    mfa_fig = create_overview_mfa_figure(all_results)
    lac_fig = create_overview_lacunarity_figure(all_results)
    perc_fig = create_overview_percolation_figure(all_results)
    clustering_fig = create_clustering_summary_figure(all_results)

    # Create gallery items for each location
    gallery_items = []
    for run_id, results in all_results.items():
        # Create figures for this location
        building_fig = create_location_building_figure(results)
        roads_fig = create_location_roads_figure(results)
        network_fig = create_location_network_figure(results)

        # Create card content
        card_content = [create_location_metrics_card(results)]

        # Add image rows (2 columns: buildings + roads, then network)
        image_row1 = []
        if building_fig:
            image_row1.append(
                html.Div([
                    dcc.Graph(
                        figure=building_fig,
                        config={"displayModeBar": False, "scrollZoom": True},
                        style={"height": "220px"}
                    )
                ], style={"flex": "1", "minWidth": "0"})
            )
        if roads_fig:
            image_row1.append(
                html.Div([
                    dcc.Graph(
                        figure=roads_fig,
                        config={"displayModeBar": False, "scrollZoom": True},
                        style={"height": "220px"}
                    )
                ], style={"flex": "1", "minWidth": "0"})
            )

        if image_row1:
            card_content.append(
                html.Div(image_row1, style={
                    "display": "flex",
                    "gap": "5px",
                    "marginBottom": "5px"
                })
            )

        if network_fig:
            card_content.append(
                html.Div([
                    dcc.Graph(
                        figure=network_fig,
                        config={"displayModeBar": False, "scrollZoom": True},
                        style={"height": "220px"}
                    )
                ])
            )

        # Add card to gallery if there's any visualization
        if building_fig or roads_fig or network_fig:
            gallery_items.append(
                html.Div(card_content, className="location-card")
            )

    # Build layout
    app.layout = html.Div([
        html.Div([
            html.H1("🏙️ Urban Structure Gallery"),
            html.P(f"構造可視化 - {len(all_results)} Locations", className="subtitle"),

            # Overview section (always visible)
            html.Div([
                html.H2("📊 Overview Charts - 全体比較"),
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id="overview-mfa",
                            figure=mfa_fig,
                            config={"displayModeBar": True, "scrollZoom": True},
                        ),
                    ]),
                    html.Div([
                        dcc.Graph(
                            id="overview-lacunarity",
                            figure=lac_fig,
                            config={"displayModeBar": True, "scrollZoom": True},
                        ),
                    ]),
                    html.Div([
                        dcc.Graph(
                            id="overview-percolation",
                            figure=perc_fig,
                            config={"displayModeBar": True, "scrollZoom": True},
                        ),
                    ]),
                    html.Div([
                        dcc.Graph(
                            id="overview-clustering",
                            figure=clustering_fig,
                            config={"displayModeBar": True, "scrollZoom": True},
                        ),
                    ]),
                ], className="overview-charts"),
            ], className="overview-section"),

            # Gallery section
            html.Div([
                html.H2("🖼️ Structure Gallery - 構造画像ギャラリー"),
                html.P(
                    "各地点の建物構造（灰色）、道路構造（カラー）、ネットワーク図を比較表示",
                    style={"color": "#888", "fontSize": "0.9em", "marginBottom": "15px"}
                ),
                html.Div(gallery_items, className="gallery-grid"),
            ], className="gallery-section"),

        ], className="container"),
    ])

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch Urban Structure Gallery Dashboard")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to outputs directory containing run directories",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run dashboard on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Urban Structure Gallery Dashboard")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 60)

    app = create_gallery_dashboard(args.results_dir)
    app.run(host=args.host, port=args.port, debug=args.debug)
