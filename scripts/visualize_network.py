#!/usr/bin/env python
"""Visualize network graph with roads and buildings.

ネットワークグラフを可視化し、道路と建物を色分けして表示する。
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx
import numpy as np
from matplotlib.patches import Circle, Rectangle
from tqdm import tqdm

# Set Japanese font
try:
    # Try to use Noto Sans CJK JP
    jp_font = fm.FontProperties(fname='/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family'] = jp_font.get_name()
except:
    try:
        # Fallback to DejaVu Sans (will show warnings for Japanese)
        plt.rcParams['font.family'] = 'DejaVu Sans'
    except:
        pass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.network_builder import NetworkBuilder


def load_data(data_dir: Path):
    """Load roads and buildings from GeoJSON files."""
    roads_path = data_dir / "roads.geojson"
    buildings_path = data_dir / "buildings.geojson"
    
    roads = gpd.read_file(roads_path) if roads_path.exists() else gpd.GeoDataFrame()
    buildings = gpd.read_file(buildings_path) if buildings_path.exists() else gpd.GeoDataFrame()
    
    return roads, buildings


def visualize_network(
    graph: nx.Graph,
    roads: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    output_path: Path,
    title: str = "Network Visualization",
    show_buildings: bool = True,
    show_roads: bool = True,
):
    """Visualize network graph with roads and buildings."""
    fig, ax = plt.subplots(figsize=(16, 16), facecolor='white')
    
    # Set up coordinate limits
    if graph.number_of_nodes() == 0:
        print("Warning: Graph has no nodes")
        ax.text(0.5, 0.5, "No network data", ha='center', va='center', transform=ax.transAxes)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    # Extract node coordinates
    node_x = []
    node_y = []
    node_types = []
    
    for node_id in graph.nodes():
        node_data = graph.nodes[node_id]
        x = node_data.get('x', 0)
        y = node_data.get('y', 0)
        node_type = node_data.get('type', 'unknown')
        
        node_x.append(x)
        node_y.append(y)
        node_types.append(node_type)
    
    node_x = np.array(node_x)
    node_y = np.array(node_y)
    
    # Calculate bounds
    x_min, x_max = node_x.min(), node_x.max()
    y_min, y_max = node_y.min(), node_y.max()
    
    # Add margin
    margin = max(x_max - x_min, y_max - y_min) * 0.05
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    
    # Draw roads from GeoDataFrame (background)
    if show_roads and len(roads) > 0:
        for idx, road in tqdm(roads.iterrows(), total=len(roads), desc="Drawing roads"):
            geom = road.geometry
            if geom is None or geom.is_empty or not geom.is_valid:
                continue
            
            # Handle different geometry types
            if geom.geom_type == "LineString":
                coords = list(geom.coords)
                if len(coords) >= 2:
                    x_coords = [c[0] for c in coords]
                    y_coords = [c[1] for c in coords]
                    ax.plot(x_coords, y_coords, color='#888888', linewidth=0.5, alpha=0.6, zorder=1)
            elif geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    if line.is_empty or not line.is_valid:
                        continue
                    coords = list(line.coords)
                    if len(coords) >= 2:
                        x_coords = [c[0] for c in coords]
                        y_coords = [c[1] for c in coords]
                        ax.plot(x_coords, y_coords, color='#888888', linewidth=0.5, alpha=0.6, zorder=1)
            elif geom.geom_type == "GeometryCollection":
                # Handle GeometryCollection (can occur after clipping)
                for sub_geom in geom.geoms:
                    if sub_geom.geom_type == "LineString" and not sub_geom.is_empty and sub_geom.is_valid:
                        coords = list(sub_geom.coords)
                        if len(coords) >= 2:
                            x_coords = [c[0] for c in coords]
                            y_coords = [c[1] for c in coords]
                            ax.plot(x_coords, y_coords, color='#888888', linewidth=0.5, alpha=0.6, zorder=1)
                    elif sub_geom.geom_type == "MultiLineString":
                        for line in sub_geom.geoms:
                            if line.is_empty or not line.is_valid:
                                continue
                            coords = list(line.coords)
                            if len(coords) >= 2:
                                x_coords = [c[0] for c in coords]
                                y_coords = [c[1] for c in coords]
                                ax.plot(x_coords, y_coords, color='#888888', linewidth=0.5, alpha=0.6, zorder=1)
    
    # Draw network edges
    edge_colors = []
    for u, v, data in graph.edges(data=True):
        edge_type = data.get('type', 'road')
        if edge_type == 'virtual':
            edge_colors.append('#FF6B6B')  # Red for virtual edges
        else:
            edge_colors.append('#4ECDC4')  # Cyan for road edges
        
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        ux, uy = u_data.get('x', 0), u_data.get('y', 0)
        vx, vy = v_data.get('x', 0), v_data.get('y', 0)
        
        ax.plot([ux, vx], [uy, vy], color=edge_colors[-1], linewidth=0.3, alpha=0.7, zorder=2)
    
    # Draw buildings from GeoDataFrame (background)
    if show_buildings and len(buildings) > 0:
        for idx, building in tqdm(buildings.iterrows(), total=len(buildings), desc="Drawing buildings"):
            geom = building.geometry
            if geom is None or geom.is_empty:
                continue
            
            if geom.geom_type == "Polygon":
                coords = list(geom.exterior.coords)
                x_coords = [c[0] for c in coords]
                y_coords = [c[1] for c in coords]
                ax.fill(x_coords, y_coords, color='#E8E8E8', edgecolor='#CCCCCC', linewidth=0.2, alpha=0.5, zorder=0)
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    coords = list(poly.exterior.coords)
                    x_coords = [c[0] for c in coords]
                    y_coords = [c[1] for c in coords]
                    ax.fill(x_coords, y_coords, color='#E8E8E8', edgecolor='#CCCCCC', linewidth=0.2, alpha=0.5, zorder=0)
    
    # Draw network nodes with color coding
    road_nodes = [i for i, t in enumerate(node_types) if t == 'road']
    building_nodes = [i for i, t in enumerate(node_types) if t == 'building']
    
    if road_nodes:
        ax.scatter(
            node_x[road_nodes], node_y[road_nodes],
            c='#3498DB', s=2, alpha=0.6, zorder=3, label=f'Road nodes ({len(road_nodes)})'
        )
    
    if building_nodes:
        ax.scatter(
            node_x[building_nodes], node_y[building_nodes],
            c='#E74C3C', s=1, alpha=0.4, zorder=3, label=f'Building nodes ({len(building_nodes)})'
        )
    
    # Add title and labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    
    # Add statistics text
    stats_text = (
        f"Nodes: {graph.number_of_nodes()}\n"
        f"Edges: {graph.number_of_edges()}\n"
        f"Road nodes: {len(road_nodes)}\n"
        f"Building nodes: {len(building_nodes)}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved network visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize network graph")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Data directory containing roads.geojson, buildings.geojson, and network.graphml"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for visualization images (default: data_dir)"
    )
    parser.add_argument(
        "--rebuild-network",
        action="store_true",
        help="Rebuild network from roads and buildings (even if network.graphml exists)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Title for visualization (default: site_id from metadata)"
    )
    parser.add_argument(
        "--no-buildings",
        action="store_true",
        help="Don't show buildings in visualization"
    )
    parser.add_argument(
        "--no-roads",
        action="store_true",
        help="Don't show roads in visualization"
    )
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    output_dir = args.output_dir if args.output_dir else data_dir
    
    # Load metadata for title
    metadata_path = data_dir / "metadata.yaml"
    title = args.title
    if title is None and metadata_path.exists():
        import yaml
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
            site_id = metadata.get("meta_info", {}).get("site_id", "Unknown")
            title = site_id
    
    if title is None:
        title = data_dir.name
    
    # Check if network.graphml exists
    network_path = data_dir / "network.graphml"
    rebuild = args.rebuild_network or not network_path.exists()
    
    if rebuild:
        print("Building network from roads and buildings...")
        roads, buildings = load_data(data_dir)
        
        if len(roads) == 0:
            print("Error: No roads found")
            return
        
        builder = NetworkBuilder()
        graph = builder.build_network(roads, buildings if not args.no_buildings else None)
        builder.save(graph, network_path)
        print(f"Network saved to {network_path}")
    else:
        print(f"Loading network from {network_path}...")
        graph = nx.read_graphml(str(network_path))
        print(f"Loaded network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Load roads and buildings for visualization
    roads, buildings = load_data(data_dir)
    
    # Generate visualization
    output_path = output_dir / f"{data_dir.name}_network_all_roads_colored.png"
    visualize_network(
        graph=graph,
        roads=roads,
        buildings=buildings,
        output_path=output_path,
        title=title,
        show_buildings=not args.no_buildings,
        show_roads=not args.no_roads,
    )
    
    print(f"\nVisualization complete!")
    print(f"  Network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()

