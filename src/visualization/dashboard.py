"""Dash Dashboard for interactive visualization of urban structure analysis."""

from pathlib import Path
from typing import Any

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
from dash import Input, Output, State, callback, dcc, html
from plotly.subplots import make_subplots


def load_results(results_dir: Path) -> dict[str, Any]:
    """
    Load analysis results from directory.

    Args:
        results_dir: Path to results directory

    Returns:
        Dictionary with loaded data
    """
    results: dict[str, Any] = {}

    # Load config snapshot
    config_path = results_dir / "config_snapshot.yaml"
    if config_path.exists():
        with open(config_path) as f:
            results["config"] = yaml.safe_load(f)

    # Load MFA results
    mfa_path = results_dir / "mfa_spectrum.csv"
    if mfa_path.exists():
        results["mfa_spectrum"] = pd.read_csv(mfa_path)

    mfa_dim_path = results_dir / "mfa_dimensions.csv"
    if mfa_dim_path.exists():
        results["mfa_dimensions"] = pd.read_csv(mfa_dim_path)

    # Load Lacunarity results
    lac_path = results_dir / "lacunarity.csv"
    if lac_path.exists():
        results["lacunarity"] = pd.read_csv(lac_path)

    lac_fit_path = results_dir / "lacunarity_fit.yaml"
    if lac_fit_path.exists():
        with open(lac_fit_path) as f:
            results["lacunarity_fit"] = yaml.safe_load(f)

    # Load Percolation results
    perc_path = results_dir / "percolation.csv"
    if perc_path.exists():
        results["percolation"] = pd.read_csv(perc_path)

    perc_stats_path = results_dir / "percolation_stats.yaml"
    if perc_stats_path.exists():
        with open(perc_stats_path) as f:
            results["percolation_stats"] = yaml.safe_load(f)

    return results


def create_mfa_figure(results: dict[str, Any]) -> go.Figure:
    """Create MFA spectrum figure."""
    if "mfa_spectrum" not in results:
        return go.Figure().add_annotation(
            text="MFA data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    df = results["mfa_spectrum"]

    # Create subplots: f(α) vs α, τ(q) vs q, D(q) vs q
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Multifractal Spectrum f(α)",
            "Mass Exponent τ(q)",
            "Generalized Dimensions D(q)",
            "R² by q"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # f(α) vs α
    fig.add_trace(
        go.Scatter(
            x=df["alpha"],
            y=df["f_alpha"],
            mode="lines+markers",
            name="f(α)",
            line=dict(color="#FF6B6B", width=2),
            marker=dict(size=6),
        ),
        row=1, col=1
    )

    # τ(q) vs q
    fig.add_trace(
        go.Scatter(
            x=df["q"],
            y=df["tau"],
            mode="lines+markers",
            name="τ(q)",
            line=dict(color="#4ECDC4", width=2),
            marker=dict(size=6),
        ),
        row=1, col=2
    )

    # D(q) vs q
    if "mfa_dimensions" in results:
        dq = results["mfa_dimensions"]
        fig.add_trace(
            go.Scatter(
                x=dq["q"],
                y=dq["D_q"],
                mode="lines+markers",
                name="D(q)",
                line=dict(color="#45B7D1", width=2),
                marker=dict(size=6),
            ),
            row=2, col=1
        )

    # R² vs q
    fig.add_trace(
        go.Scatter(
            x=df["q"],
            y=df["R2"],
            mode="lines+markers",
            name="R²",
            line=dict(color="#96CEB4", width=2),
            marker=dict(size=6),
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=700,
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(family="Noto Sans JP, sans-serif", color="#eee"),
    )

    fig.update_xaxes(title_text="α (Hölder exponent)", row=1, col=1, gridcolor="#2a2a4e")
    fig.update_yaxes(title_text="f(α)", row=1, col=1, gridcolor="#2a2a4e")
    fig.update_xaxes(title_text="q (moment order)", row=1, col=2, gridcolor="#2a2a4e")
    fig.update_yaxes(title_text="τ(q)", row=1, col=2, gridcolor="#2a2a4e")
    fig.update_xaxes(title_text="q", row=2, col=1, gridcolor="#2a2a4e")
    fig.update_yaxes(title_text="D(q)", row=2, col=1, gridcolor="#2a2a4e")
    fig.update_xaxes(title_text="q", row=2, col=2, gridcolor="#2a2a4e")
    fig.update_yaxes(title_text="R²", row=2, col=2, gridcolor="#2a2a4e")

    return fig


def create_lacunarity_figure(results: dict[str, Any]) -> go.Figure:
    """Create Lacunarity curve figure."""
    if "lacunarity" not in results:
        return go.Figure().add_annotation(
            text="Lacunarity data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    df = results["lacunarity"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Lacunarity Curve Λ(r)", "Log-Log Plot"),
        horizontal_spacing=0.12,
    )

    # Linear scale
    fig.add_trace(
        go.Scatter(
            x=df["r"],
            y=df["lambda"],
            mode="lines+markers",
            name="Λ(r)",
            line=dict(color="#F39C12", width=2),
            marker=dict(size=8),
        ),
        row=1, col=1
    )

    # Log-log scale with power law fit
    fig.add_trace(
        go.Scatter(
            x=df["r"],
            y=df["lambda"],
            mode="markers",
            name="Λ(r)",
            marker=dict(color="#F39C12", size=8),
        ),
        row=1, col=2
    )

    # Add power law fit line
    if "lacunarity_fit" in results:
        fit = results["lacunarity_fit"]
        r_fit = np.logspace(np.log10(df["r"].min()), np.log10(df["r"].max()), 100)
        lambda_fit = fit["intercept"] * r_fit ** (-fit["beta"])
        fig.add_trace(
            go.Scatter(
                x=r_fit,
                y=lambda_fit,
                mode="lines",
                name=f"Fit: β={fit['beta']:.3f}",
                line=dict(color="#E74C3C", width=2, dash="dash"),
            ),
            row=1, col=2
        )

    fig.update_layout(
        height=400,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(family="Noto Sans JP, sans-serif", color="#eee"),
        legend=dict(x=0.7, y=0.95),
    )

    fig.update_xaxes(title_text="Box size r [px]", row=1, col=1, gridcolor="#2a2a4e")
    fig.update_yaxes(title_text="Λ(r)", row=1, col=1, gridcolor="#2a2a4e")
    fig.update_xaxes(title_text="Box size r [px]", type="log", row=1, col=2, gridcolor="#2a2a4e")
    fig.update_yaxes(title_text="Λ(r)", type="log", row=1, col=2, gridcolor="#2a2a4e")

    return fig


def create_percolation_figure(results: dict[str, Any]) -> go.Figure:
    """Create Percolation analysis figure."""
    if "percolation" not in results:
        return go.Figure().add_annotation(
            text="Percolation data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    df = results["percolation"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Giant Component Fraction", "Number of Clusters"),
        horizontal_spacing=0.12,
    )

    # Giant component fraction
    fig.add_trace(
        go.Scatter(
            x=df["d"],
            y=df["giant_fraction"],
            mode="lines+markers",
            name="Giant fraction",
            line=dict(color="#9B59B6", width=2),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(155, 89, 182, 0.2)",
        ),
        row=1, col=1
    )

    # Add critical threshold line
    if "percolation_stats" in results:
        stats = results["percolation_stats"]
        d_critical = stats.get("d_critical_50", None)
        if d_critical:
            fig.add_vline(
                x=d_critical,
                line=dict(color="#E74C3C", width=2, dash="dash"),
                annotation_text=f"d_c={d_critical:.1f}",
                annotation_position="top right",
                row=1, col=1
            )

    # Number of clusters
    fig.add_trace(
        go.Scatter(
            x=df["d"],
            y=df["n_clusters"],
            mode="lines+markers",
            name="# Clusters",
            line=dict(color="#3498DB", width=2),
            marker=dict(size=6),
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=400,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(family="Noto Sans JP, sans-serif", color="#eee"),
        showlegend=False,
    )

    fig.update_xaxes(title_text="Distance threshold d [m]", row=1, col=1, gridcolor="#2a2a4e")
    fig.update_yaxes(title_text="Giant fraction", row=1, col=1, gridcolor="#2a2a4e")
    fig.update_xaxes(title_text="Distance threshold d [m]", row=1, col=2, gridcolor="#2a2a4e")
    fig.update_yaxes(title_text="Number of clusters", row=1, col=2, gridcolor="#2a2a4e")

    return fig


def create_summary_card(results: dict[str, Any]) -> html.Div:
    """Create summary statistics card."""
    cards = []

    # Config info
    if "config" in results:
        config = results["config"]
        cards.append(
            html.Div([
                html.H4("📍 Analysis Info", style={"color": "#4ECDC4"}),
                html.P(f"Run ID: {config.get('run_id', 'N/A')}"),
                html.P(f"Site: {config.get('site_metadata', {}).get('site_id', 'N/A')}"),
                html.P(f"Timestamp: {config.get('timestamp', 'N/A')[:19]}"),
            ], className="summary-card")
        )

    # MFA summary
    if "mfa_dimensions" in results:
        dq = results["mfa_dimensions"]
        d0 = dq[dq["q"] == 0]["D_q"].values[0] if 0 in dq["q"].values else None
        d1 = dq[abs(dq["q"] - 1) < 0.5]["D_q"].values[0] if len(dq[abs(dq["q"] - 1) < 0.5]) > 0 else None
        d2 = dq[dq["q"] == 2]["D_q"].values[0] if 2 in dq["q"].values else None

        cards.append(
            html.Div([
                html.H4("📐 Multifractal Dimensions", style={"color": "#FF6B6B"}),
                html.P(f"D(0) = {d0:.4f}" if d0 else "D(0) = N/A"),
                html.P(f"D(1) ≈ {d1:.4f}" if d1 else "D(1) = N/A"),
                html.P(f"D(2) = {d2:.4f}" if d2 else "D(2) = N/A"),
            ], className="summary-card")
        )

    # Lacunarity summary
    if "lacunarity_fit" in results:
        fit = results["lacunarity_fit"]
        cards.append(
            html.Div([
                html.H4("🔲 Lacunarity", style={"color": "#F39C12"}),
                html.P(f"β = {fit['beta']:.4f}"),
                html.P(f"R² = {fit['R2']:.4f}"),
            ], className="summary-card")
        )

    # Percolation summary
    if "percolation_stats" in results:
        stats = results["percolation_stats"]
        cards.append(
            html.Div([
                html.H4("🔗 Percolation", style={"color": "#9B59B6"}),
                html.P(f"d_c (50%) = {stats['d_critical_50']:.2f} m"),
                html.P(f"Transition width = {stats['transition_width']:.2f} m"),
                html.P(f"Max clusters = {stats['max_clusters']}"),
            ], className="summary-card")
        )

    return html.Div(cards, className="summary-container")


def create_dashboard(results_dir: Path | str) -> dash.Dash:
    """
    Create Dash application for visualizing analysis results.

    Args:
        results_dir: Path to results directory containing CSV and NPY files

    Returns:
        Configured Dash application
    """
    results_dir = Path(results_dir)
    results = load_results(results_dir)

    app = dash.Dash(
        __name__,
        title="Urban Structure Analysis",
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
                    max-width: 1400px;
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
                .summary-container {
                    display: flex;
                    gap: 20px;
                    flex-wrap: wrap;
                    justify-content: center;
                    margin-bottom: 30px;
                }
                .summary-card {
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 12px;
                    padding: 20px;
                    min-width: 200px;
                    backdrop-filter: blur(10px);
                }
                .summary-card h4 {
                    margin-top: 0;
                    margin-bottom: 15px;
                }
                .summary-card p {
                    margin: 5px 0;
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 0.9em;
                }
                .section {
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 16px;
                    padding: 20px;
                    margin-bottom: 30px;
                }
                .section h2 {
                    color: #4ECDC4;
                    border-bottom: 2px solid #4ECDC4;
                    padding-bottom: 10px;
                    margin-top: 0;
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

    run_id = results.get("config", {}).get("run_id", results_dir.name)

    app.layout = html.Div([
        html.Div([
            html.H1("🏙️ Urban Structure Analysis"),
            html.P(f"Run: {run_id}", className="subtitle"),

            # Summary cards
            create_summary_card(results),

            # MFA Section
            html.Div([
                html.H2("Multifractal Analysis"),
                dcc.Graph(
                    id="mfa-spectrum",
                    figure=create_mfa_figure(results),
                    config={"displayModeBar": True, "scrollZoom": True},
                ),
            ], className="section"),

            # Lacunarity Section
            html.Div([
                html.H2("Lacunarity Analysis"),
                dcc.Graph(
                    id="lacunarity-curve",
                    figure=create_lacunarity_figure(results),
                    config={"displayModeBar": True, "scrollZoom": True},
                ),
            ], className="section"),

            # Percolation Section
            html.Div([
                html.H2("Percolation Analysis"),
                dcc.Graph(
                    id="percolation-curve",
                    figure=create_percolation_figure(results),
                    config={"displayModeBar": True, "scrollZoom": True},
                ),
            ], className="section"),

        ], className="container"),
    ])

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch Urban Structure Analysis Dashboard")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to results directory",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run dashboard on",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )

    args = parser.parse_args()

    # Find latest results directory
    if args.results_dir.is_dir():
        subdirs = [d for d in args.results_dir.iterdir() if d.is_dir()]
        if subdirs:
            # Use most recently modified
            latest = max(subdirs, key=lambda x: x.stat().st_mtime)
            print(f"Using results from: {latest}")
            app = create_dashboard(latest)
        else:
            print(f"No results found in {args.results_dir}")
            app = create_dashboard(args.results_dir)
    else:
        app = create_dashboard(args.results_dir)

    app.run_server(debug=args.debug, port=args.port)
