#!/usr/bin/env python
"""Visualize Clustering Results in 3D Space."""

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.clustering import analyze_clusters


def load_place_names(outputs_dir: Path) -> dict[str, str]:
    """Load place names from resolved_places.json and config snapshots.
    
    Returns:
        Dictionary mapping run_id -> display_name
    """
    place_names = {}
    site_id_to_name = {}
    
    # Load resolved_places.json
    places_file = Path("data/resolved_places.json")
    if places_file.exists():
        with open(places_file) as f:
            places = json.load(f)
        for place in places:
            site_id = f"{place['latitude']}_{place['longitude']}"
            display_name = place.get("display_name", site_id)
            site_id_to_name[site_id] = display_name
    
    # Load from config snapshots and map run_id to display_name
    for run_dir in outputs_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        
        run_id = run_dir.name
        config_snapshot = run_dir / "config_snapshot.yaml"
        if config_snapshot.exists():
            try:
                with open(config_snapshot) as f:
                    config = yaml.safe_load(f)
                
                # Get site_id from metadata
                site_metadata = config.get("site_metadata", {})
                site_id = site_metadata.get("meta_info", {}).get("site_id", "")
                if not site_id:
                    site_id = site_metadata.get("site_id", "")
                
                # If site_id not found, try to extract from data_dir
                if not site_id:
                    data_dir = config.get("data_dir", "")
                    if data_dir:
                        site_id = Path(data_dir).name
                
                # Get display name
                if site_id in site_id_to_name:
                    place_names[run_id] = site_id_to_name[site_id]
                else:
                    # Fallback to site_id
                    place_names[run_id] = site_id
            except Exception as e:
                # Fallback to run_id
                place_names[run_id] = run_id
                continue
    
    return place_names


def load_building_footprint_image(data_dir: Path, max_size: int = 200) -> str | None:
    """
    Load building footprint image and convert to base64.
    Image is embedded directly in HTML for standalone viewing.
    
    Args:
        data_dir: Data directory containing buildings_binary.npy
        max_size: Maximum size for image (will be resized)
        
    Returns:
        Base64 encoded image string (data URI) or None
    """
    buildings_path = data_dir / "buildings_binary.npy"
    if not buildings_path.exists():
        return None
    
    try:
        buildings = np.load(buildings_path)
        
        # Convert to image
        # Normalize to 0-255
        if buildings.max() > 0:
            img_array = (buildings / buildings.max() * 255).astype(np.uint8)
        else:
            img_array = buildings.astype(np.uint8)
        
        # Resize if too large (to keep HTML file size manageable)
        if img_array.shape[0] > max_size or img_array.shape[1] > max_size:
            img = Image.fromarray(img_array)
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            img_array = np.array(img)
        
        # Convert to PIL Image
        img = Image.fromarray(img_array, mode='L')
        
        # Convert to RGB (for better display)
        img_rgb = Image.new('RGB', img.size)
        img_rgb.paste(img)
        
        # Convert to base64 - embed directly in HTML
        buffer = BytesIO()
        # Use optimize=True to reduce file size
        img_rgb.save(buffer, format='PNG', optimize=True)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Return data URI for embedding in HTML
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Warning: Failed to load building image from {data_dir}: {e}")
        return None


def get_data_dir_from_run_id(run_id: str, outputs_dir: Path) -> Path | None:
    """Get data directory path from run_id."""
    run_dir = outputs_dir / run_id
    if not run_dir.exists():
        return None
    
    config_snapshot = run_dir / "config_snapshot.yaml"
    if config_snapshot.exists():
        try:
            with open(config_snapshot) as f:
                config = yaml.safe_load(f)
            data_dir = config.get("data_dir", "")
            if data_dir:
                return Path(data_dir)
        except Exception:
            pass
    
    return None


def create_3d_clustering_visualization(
    results_df: pd.DataFrame,
    X_processed: np.ndarray,
    metadata: dict,
    output_path: Path,
    outputs_dir: Path,
):
    """
    Create 3D visualization of clustering results.
    
    Args:
        results_df: DataFrame with clustering results
        X_processed: Preprocessed feature matrix
        metadata: Clustering metadata
        outputs_dir: Directory containing run output directories
    """
    labels = results_df["cluster_label"].values
    sample_ids = results_df["sample_id"].values if "sample_id" in results_df.columns else results_df.index
    
    # Load place names
    place_names = load_place_names(outputs_dir)
    
    # Get dimension names with explained variance
    prep_meta = metadata.get("preprocessing", {}) if metadata else {}
    dim_names = ["PC1", "PC2", "PC3"]  # Default
    
    if prep_meta and "dimensionality_reduction" in prep_meta:
        if prep_meta["dimensionality_reduction"] == "pca":
            var_ratio = prep_meta.get("explained_variance_ratio", [])
            # Handle numpy arrays or lists
            if var_ratio:
                try:
                    if hasattr(var_ratio, '__len__') and len(var_ratio) >= 3:
                        # Convert to list if numpy array
                        if hasattr(var_ratio, 'tolist'):
                            var_ratio = var_ratio.tolist()
                        dim_names = [
                            f"PC1 ({var_ratio[0]:.1%})",
                            f"PC2 ({var_ratio[1]:.1%})",
                            f"PC3 ({var_ratio[2]:.1%})",
                        ]
                except Exception:
                    dim_names = ["PC1", "PC2", "PC3"]
        else:
            dim_names = ["Dim1", "Dim2", "Dim3"]
    
    # Use first 3 dimensions for 3D visualization
    if X_processed.shape[1] >= 3:
        X_3d = X_processed[:, :3]
    else:
        # If less than 3 dimensions, pad with zeros
        X_3d = np.zeros((X_processed.shape[0], 3))
        X_3d[:, :X_processed.shape[1]] = X_processed
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Color map for clusters
    unique_labels = sorted([l for l in np.unique(labels) if l != -1])
    n_clusters = len(unique_labels)
    
    # Use distinct colors
    if n_clusters <= 10:
        colors = px.colors.qualitative.Set3[:n_clusters]
    else:
        colors = px.colors.qualitative.Set3 * ((n_clusters // 10) + 1)
    
    # Plot each cluster
    for i, label in enumerate(unique_labels):
        mask = labels == label
        cluster_samples = [sample_ids[j] for j in range(len(sample_ids)) if mask[j]]
        
        # Prepare hover text with place names and images
        hover_texts = []
        for j, sample_id in enumerate(cluster_samples):
            # Get place name
            place_name = place_names.get(sample_id, sample_id)
            
            # Get building footprint image (embedded as base64 in HTML)
            data_dir = get_data_dir_from_run_id(sample_id, outputs_dir)
            img_html = ""
            if data_dir:
                img_base64 = load_building_footprint_image(data_dir, max_size=150)
                if img_base64:
                    # Embed image directly in HTML using base64 data URI
                    # This makes the HTML file standalone (no external dependencies)
                    img_html = f'<img src="{img_base64}" style="max-width:150px;max-height:150px;margin-top:10px;border:1px solid #666;"><br>'
            
            hover_texts.append(
                f"<b>{place_name}</b><br>"
                f"Run ID: {sample_id}<br>"
                f"{dim_names[0]}: {X_3d[mask][j, 0]:.3f}<br>"
                f"{dim_names[1]}: {X_3d[mask][j, 1]:.3f}<br>"
                f"{dim_names[2]}: {X_3d[mask][j, 2]:.3f}<br>"
                f"Cluster: {int(label)}<br>"
                f"{img_html}"
            )
        
        fig.add_trace(go.Scatter3d(
            x=X_3d[mask, 0],
            y=X_3d[mask, 1],
            z=X_3d[mask, 2],
            mode="markers",
            name=f"Cluster {int(label)} ({np.sum(mask)} samples)",
            marker=dict(
                size=12,
                color=colors[i % len(colors)],
                opacity=0.8,
                line=dict(width=1, color="white"),
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        ))
    
    # Plot noise points if any
    if -1 in labels:
        mask = labels == -1
        noise_samples = [sample_ids[j] for j in range(len(sample_ids)) if mask[j]]
        
        # Prepare hover text with place names and images
        hover_texts = []
        for j, sample_id in enumerate(noise_samples):
            place_name = place_names.get(sample_id, sample_id)
            data_dir = get_data_dir_from_run_id(sample_id, outputs_dir)
            img_html = ""
            if data_dir:
                img_base64 = load_building_footprint_image(data_dir, max_size=150)
                if img_base64:
                    # Embed image directly in HTML using base64 data URI
                    # This makes the HTML file standalone (no external dependencies)
                    img_html = f'<img src="{img_base64}" style="max-width:150px;max-height:150px;margin-top:10px;border:1px solid #666;"><br>'
            
            hover_texts.append(
                f"<b>{place_name}</b><br>"
                f"Run ID: {sample_id}<br>"
                f"{dim_names[0]}: {X_3d[mask][j, 0]:.3f}<br>"
                f"{dim_names[1]}: {X_3d[mask][j, 1]:.3f}<br>"
                f"{dim_names[2]}: {X_3d[mask][j, 2]:.3f}<br>"
                f"Cluster: Noise<br>"
                f"{img_html}"
            )
        
        fig.add_trace(go.Scatter3d(
            x=X_3d[mask, 0],
            y=X_3d[mask, 1],
            z=X_3d[mask, 2],
            mode="markers",
            name=f"Noise ({np.sum(mask)} samples)",
            marker=dict(
                size=10,
                color="gray",
                opacity=0.5,
                symbol="x",
                line=dict(width=1, color="white"),
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        ))
    
    # Get explained variance if PCA was used
    prep_meta = metadata.get("preprocessing", {}) if metadata else {}
    variance_info = ""
    if prep_meta and "explained_variance_ratio" in prep_meta:
        var_ratio = prep_meta["explained_variance_ratio"]
        try:
            if hasattr(var_ratio, '__len__') and len(var_ratio) >= 3:
                # Convert to list if numpy array
                if hasattr(var_ratio, 'tolist'):
                    var_ratio = var_ratio.tolist()
                cum_var = sum(var_ratio[:3])
                variance_info = f" (累積分散: {cum_var:.1%})"
        except Exception:
            pass
    
    # Update layout
    fig.update_layout(
        title=f"クラスタリング結果 - 3D可視化{variance_info}",
        scene=dict(
            xaxis_title=dim_names[0],
            yaxis_title=dim_names[1],
            zaxis_title=dim_names[2],
            bgcolor="#1a1a2e",
            xaxis=dict(backgroundcolor="#16213e", gridcolor="#2a2a4e"),
            yaxis=dict(backgroundcolor="#16213e", gridcolor="#2a2a4e"),
            zaxis=dict(backgroundcolor="#16213e", gridcolor="#2a2a4e"),
        ),
        template="plotly_dark",
        height=800,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="white",
            borderwidth=1,
        ),
        font=dict(family="Noto Sans JP, sans-serif", color="#eee"),
    )
    
    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "clustering_3d.html"
    fig.write_html(str(output_file))
    
    print(f"3D visualization saved to: {output_file}")
    print(f"  - {n_clusters} clusters visualized in 3D space")
    if -1 in labels:
        print(f"  - {np.sum(labels == -1)} noise points")


def main():
    """Create 3D visualization of clustering results."""
    parser = argparse.ArgumentParser(
        description="Visualize clustering results in 3D space"
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing run output directories",
    )
    parser.add_argument(
        "--clustering-results-dir",
        type=Path,
        default=Path("outputs/clustering_results"),
        help="Directory containing existing clustering results",
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
        default=Path("outputs/clustering_results"),
        help="Output directory for visualization",
    )
    parser.add_argument(
        "--use-best-config",
        action="store_true",
        help="Use best configuration from comparison results",
    )

    args = parser.parse_args()

    # Check if clustering results already exist
    results_csv = args.clustering_results_dir / "clustering_results.csv"
    processed_features_npy = args.clustering_results_dir / "processed_features.npy"
    metadata_yaml = args.clustering_results_dir / "clustering_metadata.yaml"
    
    if results_csv.exists() and processed_features_npy.exists():
        # Load existing results
        print("Loading existing clustering results...")
        results_df = pd.read_csv(results_csv)
        X_processed = np.load(processed_features_npy)
        
        # Try to load metadata, but use empty dict if it fails (e.g., numpy objects)
        metadata = {}
        if metadata_yaml.exists():
            try:
                with open(metadata_yaml) as f:
                    metadata = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Could not load metadata from {metadata_yaml}: {e}")
                print("Continuing with empty metadata...")
                metadata = {}
        
        print(f"Loaded: {len(results_df)} samples, {X_processed.shape[1]} dimensions")
    else:
        # Run clustering first
        print("Running clustering analysis...")
        
        # Find output directories
        output_dirs = [
            d
            for d in args.outputs_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ]
        
        if len(output_dirs) == 0:
            raise ValueError(f"No output directories found in {args.outputs_dir}")
        
        # Load config
        if args.use_best_config:
            # Load best config from comparison results
            comparison_summary = Path("outputs/clustering_comparison/comparison_summary.yaml")
            if comparison_summary.exists():
                with open(comparison_summary) as f:
                    summary = yaml.safe_load(f)
                best = summary.get("best_silhouette", {})
                if best:
                    preprocessing_config = {
                        "normalization_method": best.get("normalization", "robust"),
                        "dimensionality_reduction": best.get("dimensionality_reduction", "pca"),
                        "n_components": None,
                        "random_state": 42,
                    }
                    clustering_config = {
                        "method": best.get("clustering_method", "kmeans"),
                        "n_clusters": int(best.get("n_clusters", 2)),
                        "random_state": 42,
                    }
                    if best.get("clustering_method") == "hierarchical":
                        clustering_config["linkage"] = best.get("linkage", "ward")
                    print(f"Using best configuration: {best.get('config_id')}")
                else:
                    raise ValueError("Best configuration not found in comparison results")
            else:
                raise ValueError("Comparison results not found. Run compare_clustering_methods.py first.")
        else:
            # Load from config file
            with open(args.config) as f:
                config = yaml.safe_load(f)
            
            preprocessing_config = config.get("preprocessing", {})
            clustering_config = config.get("clustering", {})
        
        results_df, X_processed, metadata = analyze_clusters(
            output_dirs,
            preprocessor_config=preprocessing_config,
            clustering_config=clustering_config,
        )
        
        # Save results
        args.output.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.output / "clustering_results.csv", index=False)
        np.save(args.output / "processed_features.npy", X_processed)
        with open(args.output / "clustering_metadata.yaml", "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
    
    # Create 3D visualization
    create_3d_clustering_visualization(
        results_df,
        X_processed,
        metadata,
        args.output,
        args.outputs_dir,
    )


if __name__ == "__main__":
    main()

