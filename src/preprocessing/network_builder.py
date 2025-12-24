"""Build NetworkX network from vector data."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely import LineString, Point
from shapely.ops import linemerge, unary_union
from tqdm import tqdm

# NetworkX is always available (graph-tool not used in this version)
GRAPH_TOOL_AVAILABLE = True


@dataclass
class NetworkBuilder:
    """Build spatial network graph from roads and buildings using NetworkX."""

    snap_tolerance: float = 1.0  # メートル

    def build_network(
        self,
        roads: gpd.GeoDataFrame,
        buildings: gpd.GeoDataFrame | None = None,
        verbose: bool = True,
    ) -> nx.Graph:
        """
        Build network graph with:
        - Road intersection nodes
        - Building centroid nodes (optional)
        - Road edges
        - Virtual edges (building -> nearest road) (optional)

        Args:
            roads: Road line geometries in local coordinates
            buildings: Building polygon geometries in local coordinates (optional)
            verbose: Show progress

        Returns:
            NetworkX Graph with 'length' edge attribute
        """
        G = nx.Graph()

        if len(roads) == 0:
            if verbose:
                print("No roads to process")
            return G

        if verbose:
            print(f"Building network from {len(roads)} road segments...")

        # Step 1: Extract nodes from road endpoints and intersections
        node_coords: dict[tuple[float, float], int] = {}
        node_counter = 0

        def get_or_create_node(x: float, y: float) -> int:
            """Get existing node or create new one at coordinates."""
            nonlocal node_counter
            # Snap to tolerance
            key = (round(x / self.snap_tolerance) * self.snap_tolerance,
                   round(y / self.snap_tolerance) * self.snap_tolerance)

            if key not in node_coords:
                node_coords[key] = node_counter
                G.add_node(node_counter, x=key[0], y=key[1], type="road")
                node_counter += 1
            return node_coords[key]

        # Step 2: Add road edges
        for idx, row in tqdm(roads.iterrows(), total=len(roads), desc="Processing roads", disable=not verbose):
            geom = row.geometry
            if geom is None or geom.is_empty or not geom.is_valid:
                continue

            if geom.geom_type == "MultiLineString":
                lines = list(geom.geoms)
            elif geom.geom_type == "LineString":
                lines = [geom]
            else:
                continue

            for line in lines:
                if len(line.coords) < 2:
                    continue

                coords = list(line.coords)

                # Create nodes at all vertices
                for i in range(len(coords) - 1):
                    x1, y1 = coords[i][0], coords[i][1]
                    x2, y2 = coords[i + 1][0], coords[i + 1][1]

                    node1 = get_or_create_node(x1, y1)
                    node2 = get_or_create_node(x2, y2)

                    if node1 != node2:
                        segment_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        # Add or update edge (keep shortest if duplicate)
                        if G.has_edge(node1, node2):
                            existing_length = G[node1][node2]["length"]
                            if segment_length < existing_length:
                                G[node1][node2]["length"] = segment_length
                        else:
                            G.add_edge(node1, node2, length=segment_length)

        if verbose:
            print(f"Road network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Step 3: Add building centroids as nodes (optional)
        if buildings is not None and len(buildings) > 0:
            if verbose:
                print(f"Adding {len(buildings)} building nodes...")

            building_nodes = []
            for idx, row in tqdm(buildings.iterrows(), total=len(buildings),
                                 desc="Adding buildings", disable=not verbose):
                geom = row.geometry
                if geom is None or geom.is_empty or not geom.is_valid:
                    continue

                centroid = geom.centroid
                x, y = centroid.x, centroid.y

                # Add as new node
                G.add_node(node_counter, x=x, y=y, type="building")
                building_nodes.append((node_counter, x, y))
                node_counter += 1

            # Step 4: Connect buildings to nearest road node
            if building_nodes and len(node_coords) > 0:
                if verbose:
                    print("Connecting buildings to road network...")

                road_node_array = np.array([(x, y) for (x, y), _ in node_coords.items()])
                road_node_ids = [nid for _, nid in node_coords.items()]

                for bnode, bx, by in tqdm(building_nodes, desc="Connecting buildings", disable=not verbose):
                    # Find nearest road node
                    distances = np.sqrt((road_node_array[:, 0] - bx) ** 2 +
                                        (road_node_array[:, 1] - by) ** 2)
                    nearest_idx = np.argmin(distances)
                    nearest_road_node = road_node_ids[nearest_idx]
                    nearest_distance = distances[nearest_idx]

                    # Add virtual edge
                    G.add_edge(bnode, nearest_road_node, length=nearest_distance, type="virtual")

            if verbose:
                print(f"Final network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        return G

    def save(self, graph: nx.Graph, output_path: Path) -> None:
        """Save graph to .graphml file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure path has correct extension
        if output_path.suffix != ".graphml":
            output_path = output_path.with_suffix(".graphml")

        nx.write_graphml(graph, str(output_path))

    def load(self, input_path: Path) -> nx.Graph:
        """Load graph from .graphml file."""
        return nx.read_graphml(str(input_path))

    def get_edge_lengths(self, graph: nx.Graph) -> np.ndarray:
        """Extract all edge lengths as numpy array."""
        return np.array([data.get("length", 0) for _, _, data in graph.edges(data=True)])

    def compute_statistics(self, graph: nx.Graph) -> dict[str, Any]:
        """Compute basic network statistics."""
        if graph.number_of_nodes() == 0:
            return {
                "n_nodes": 0,
                "n_edges": 0,
                "n_components": 0,
                "avg_degree": 0,
                "total_length": 0,
            }

        lengths = self.get_edge_lengths(graph)
        degrees = [d for _, d in graph.degree()]

        return {
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "n_components": nx.number_connected_components(graph),
            "avg_degree": np.mean(degrees) if degrees else 0,
            "total_length": np.sum(lengths),
            "mean_edge_length": np.mean(lengths) if len(lengths) > 0 else 0,
            "max_edge_length": np.max(lengths) if len(lengths) > 0 else 0,
        }

    # Alias for backwards compatibility
    get_network_stats = compute_statistics
