"""Continuum Percolation Analysis using NetworkX connected components.

This module implements percolation analysis for urban network graphs,
supporting both edge-based filtering and shortest path distance approaches.

Distance Calculation Methods:
- "edge": Uses direct edge lengths for filtering (default, fast)
- "shortest_path": Uses shortest path distances between nodes (more accurate, slower)

Disconnected Component Handling:
- When distance_type="edge": Disconnected nodes remain isolated at all thresholds
- When distance_type="shortest_path": Disconnected pairs have infinite distance

Reference:
    Standard percolation analysis uses shortest path distances (network distance)
    as recommended in urban network literature. Edge-based filtering is faster
    but may not capture true network connectivity characteristics.
"""

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class PercolationAnalyzer:
    """
    Compute percolation metrics using NetworkX graph filtering.

    Algorithm:
    1. Load network graph with edge lengths
    2. For each distance threshold d:
       - Keep only edges/node pairs where distance <= d
       - Count connected components and max component size
    3. Track percolation transition (emergence of giant component)

    Args:
        d_min: Minimum distance threshold
        d_max: Maximum distance threshold
        d_steps: Number of threshold steps
        distance_type: "edge" for edge lengths, "shortest_path" for network distances
        node_filter: Filter nodes by type (e.g., "building" for building-only analysis)

    Note:
        For building-to-building percolation (node_filter="building"), nodes are
        considered connected if the shortest path distance between them is <= d.
        Disconnected building pairs have infinite distance and are never connected.
    """

    d_min: float = 1
    d_max: float = 100
    d_steps: int = 50
    distance_type: str = "edge"  # "edge" or "shortest_path"
    node_filter: str | None = None  # e.g., "building" to analyze only building nodes

    def analyze(
        self, graph: nx.Graph | str | Path
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Compute percolation curve.

        Args:
            graph: NetworkX Graph or path to .graphml file

        Returns:
            percolation_df: DataFrame with columns [d, max_cluster_size, n_clusters, giant_fraction]
            mesh: 2D array of cluster labels (d_steps, N_nodes)

        Note:
            When distance_type="shortest_path", disconnected node pairs have
            infinite distance and are never considered connected at any threshold.
            This correctly models real-world network connectivity where paths
            don't exist between all building pairs.
        """
        # Load graph if path provided
        if isinstance(graph, (str, Path)):
            graph = nx.read_graphml(str(graph))

        if graph.number_of_nodes() == 0:
            raise ValueError("Graph has no nodes")

        # Filter nodes if specified
        if self.node_filter is not None:
            filtered_nodes = [
                n for n, data in graph.nodes(data=True)
                if data.get("type") == self.node_filter
            ]
            if len(filtered_nodes) == 0:
                raise ValueError(f"No nodes with type '{self.node_filter}' found")
            analysis_nodes = filtered_nodes
        else:
            analysis_nodes = list(graph.nodes())

        n_nodes = len(analysis_nodes)
        thresholds = self._get_thresholds()

        print(f"Computing Percolation: {len(thresholds)} thresholds, {n_nodes} nodes")
        print(f"Distance range: {self.d_min:.1f} to {self.d_max:.1f}")
        print(f"Distance type: {self.distance_type}")
        if self.node_filter:
            print(f"Node filter: {self.node_filter}")

        # Map node labels to indices
        node_to_idx = {node: idx for idx, node in enumerate(analysis_nodes)}

        # Choose analysis method based on distance_type
        if self.distance_type == "shortest_path":
            return self._analyze_shortest_path(
                graph, analysis_nodes, node_to_idx, thresholds
            )
        else:
            return self._analyze_edge_based(
                graph, analysis_nodes, node_to_idx, thresholds
            )

    def _analyze_edge_based(
        self,
        graph: nx.Graph,
        analysis_nodes: list,
        node_to_idx: dict,
        thresholds: np.ndarray,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Analyze percolation using edge length filtering.

        This is the original (fast) method that filters edges by their direct length.
        Disconnected nodes remain isolated regardless of threshold.
        """
        n_nodes = len(analysis_nodes)

        # Get edge lengths
        edge_lengths = {}
        for u, v, data in graph.edges(data=True):
            length = float(data.get("length", 1.0))
            edge_lengths[(u, v)] = length

        # Compute percolation metrics for each threshold
        results = []
        mesh = np.zeros((len(thresholds), n_nodes), dtype=np.int32)

        for d_idx, d in enumerate(tqdm(thresholds, desc="Computing percolation (edge)")):
            # Create filtered graph view
            filtered = self._filter_graph(graph, edge_lengths, d)

            # If node filter is set, work with subgraph
            if self.node_filter is not None:
                filtered = filtered.subgraph(analysis_nodes).copy()

            # Compute connected components
            components = list(nx.connected_components(filtered))
            n_clusters = len(components)

            # Find largest component
            if components:
                largest = max(components, key=len)
                max_size = len(largest)
            else:
                max_size = 0

            giant_fraction = max_size / n_nodes if n_nodes > 0 else 0

            results.append({
                "d": d,
                "max_cluster_size": max_size,
                "n_clusters": n_clusters,
                "giant_fraction": giant_fraction,
            })

            # Store cluster labels
            for cluster_idx, component in enumerate(components):
                for node in component:
                    if node in node_to_idx:
                        mesh[d_idx, node_to_idx[node]] = cluster_idx + 1

        percolation_df = pd.DataFrame(results)
        return percolation_df, mesh

    def _analyze_shortest_path(
        self,
        graph: nx.Graph,
        analysis_nodes: list,
        node_to_idx: dict,
        thresholds: np.ndarray,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Analyze percolation using shortest path distances.

        This method computes network distances (shortest path) between all node pairs
        and connects them if distance <= threshold. More accurate for urban networks
        but computationally expensive for large graphs.

        Disconnected node pairs have infinite distance and are never connected.
        """
        n_nodes = len(analysis_nodes)

        print("Computing pairwise shortest path distances...")
        print("(This may take a while for large networks)")

        # Compute shortest path distances for analysis nodes
        # Use weighted shortest path with edge 'length' attribute
        distance_matrix = np.full((n_nodes, n_nodes), np.inf)
        np.fill_diagonal(distance_matrix, 0)

        # Compute single-source shortest paths from each analysis node
        for i, source in enumerate(tqdm(analysis_nodes, desc="Computing distances")):
            try:
                lengths = nx.single_source_dijkstra_path_length(
                    graph, source, weight="length"
                )
                for target, dist in lengths.items():
                    if target in node_to_idx:
                        j = node_to_idx[target]
                        distance_matrix[i, j] = dist
            except nx.NetworkXNoPath:
                # Source is disconnected from some nodes - distances remain inf
                pass

        # Report disconnected pairs
        n_disconnected = np.sum(np.isinf(distance_matrix)) - n_nodes  # exclude diagonal
        n_total_pairs = n_nodes * (n_nodes - 1)
        if n_disconnected > 0:
            print(f"Disconnected pairs: {n_disconnected // 2} / {n_total_pairs // 2} "
                  f"({n_disconnected / n_total_pairs * 100:.1f}%)")
            print("Note: Disconnected pairs have infinite distance and remain isolated.")

        # Compute percolation metrics for each threshold
        results = []
        mesh = np.zeros((len(thresholds), n_nodes), dtype=np.int32)

        for d_idx, d in enumerate(tqdm(thresholds, desc="Computing percolation (shortest_path)")):
            # Create graph from distance matrix
            filtered = nx.Graph()
            filtered.add_nodes_from(range(n_nodes))

            # Add edges where distance <= threshold
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if distance_matrix[i, j] <= d:
                        filtered.add_edge(i, j, distance=distance_matrix[i, j])

            # Compute connected components
            components = list(nx.connected_components(filtered))
            n_clusters = len(components)

            # Find largest component
            if components:
                largest = max(components, key=len)
                max_size = len(largest)
            else:
                max_size = 0

            giant_fraction = max_size / n_nodes if n_nodes > 0 else 0

            results.append({
                "d": d,
                "max_cluster_size": max_size,
                "n_clusters": n_clusters,
                "giant_fraction": giant_fraction,
            })

            # Store cluster labels
            for cluster_idx, component in enumerate(components):
                for node_idx in component:
                    mesh[d_idx, node_idx] = cluster_idx + 1

        percolation_df = pd.DataFrame(results)
        return percolation_df, mesh

    def _get_thresholds(self) -> np.ndarray:
        """Generate distance thresholds."""
        return np.linspace(self.d_min, self.d_max, self.d_steps)

    def _filter_graph(
        self,
        graph: nx.Graph,
        edge_lengths: dict[tuple, float],
        d: float
    ) -> nx.Graph:
        """
        Create filtered graph with edges <= d.

        Args:
            graph: Original graph
            edge_lengths: Dictionary of edge lengths
            d: Distance threshold

        Returns:
            Filtered graph (new graph, not view)
        """
        filtered = nx.Graph()
        filtered.add_nodes_from(graph.nodes(data=True))

        for (u, v), length in edge_lengths.items():
            if length <= d:
                filtered.add_edge(u, v, length=length)

        return filtered

    def find_percolation_threshold(
        self, percolation_df: pd.DataFrame, target_fraction: float = 0.5
    ) -> float:
        """
        Find the distance threshold where giant component reaches target fraction.

        Args:
            percolation_df: Percolation results
            target_fraction: Target fraction of nodes in giant component

        Returns:
            Critical distance threshold (interpolated)
        """
        d = percolation_df["d"].values
        gf = percolation_df["giant_fraction"].values

        # Find crossing point
        for i in range(len(gf) - 1):
            if gf[i] < target_fraction <= gf[i + 1]:
                # Linear interpolation
                t = (target_fraction - gf[i]) / (gf[i + 1] - gf[i])
                return d[i] + t * (d[i + 1] - d[i])

        # If not found, return boundary
        if gf[-1] < target_fraction:
            return d[-1]
        return d[0]

    def compute_susceptibility(self, percolation_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute percolation susceptibility χ(d).

        χ = <s²> / <s> where s is component size
        (excluding the largest component)

        Returns:
            DataFrame with columns [d, susceptibility]
        """
        # Simplified susceptibility approximation based on cluster statistics
        d = percolation_df["d"].values
        n = percolation_df["n_clusters"].values
        gf = percolation_df["giant_fraction"].values

        # Approximate susceptibility as variance-like measure
        # Higher near transition, lower far from it
        susceptibility = n * (1 - gf) * gf

        return pd.DataFrame({
            "d": d,
            "susceptibility": susceptibility,
        })

    def analyze_with_statistics(
        self, graph: nx.Graph | str | Path
    ) -> tuple[pd.DataFrame, dict]:
        """
        Run percolation analysis with additional statistics.

        Returns:
            percolation_df: Main percolation results
            stats: Dictionary with transition statistics
        """
        percolation_df, mesh = self.analyze(graph)

        # Find critical thresholds
        d_05 = self.find_percolation_threshold(percolation_df, 0.5)
        d_01 = self.find_percolation_threshold(percolation_df, 0.1)
        d_09 = self.find_percolation_threshold(percolation_df, 0.9)

        # Compute susceptibility peak
        susceptibility = self.compute_susceptibility(percolation_df)
        peak_idx = np.argmax(susceptibility["susceptibility"].values)
        d_peak = susceptibility["d"].values[peak_idx]

        stats = {
            "d_critical_50": d_05,
            "d_critical_10": d_01,
            "d_critical_90": d_09,
            "d_susceptibility_peak": d_peak,
            "transition_width": d_09 - d_01,
            "max_clusters": percolation_df["n_clusters"].max(),
        }

        return percolation_df, stats


@dataclass
class PathDiversityAnalyzer:
    """
    Analyze path diversity as a separate metric from percolation.

    This class computes path diversity metrics between node pairs,
    treating it independently from percolation analysis as recommended
    in urban network literature.

    Path diversity considers:
    - Number of alternative paths between node pairs
    - Distribution of path lengths
    - Redundancy in network connections

    Note:
        This analysis is computationally expensive for large networks.
        Consider using sampling (sample_pairs parameter) for large graphs.
    """

    max_paths: int = 5  # Maximum number of paths to consider per pair
    length_tolerance: float = 1.5  # Paths within this factor of shortest are "diverse"
    sample_pairs: int | None = None  # Sample this many pairs (None = all pairs)
    node_filter: str | None = None  # Filter nodes by type
    max_path_hops: int = 50  # Maximum number of hops in path search
    random_seed: int | None = 42  # Random seed for reproducible sampling (None = random)

    def analyze(
        self, graph: nx.Graph | str | Path
    ) -> tuple[pd.DataFrame, dict]:
        """
        Compute path diversity metrics.

        Args:
            graph: NetworkX Graph or path to .graphml file

        Returns:
            diversity_df: DataFrame with per-pair diversity metrics
            stats: Summary statistics
        """
        # Load graph if path provided
        if isinstance(graph, (str, Path)):
            graph = nx.read_graphml(str(graph))

        if graph.number_of_nodes() == 0:
            raise ValueError("Graph has no nodes")

        # Filter nodes if specified
        if self.node_filter is not None:
            analysis_nodes = [
                n for n, data in graph.nodes(data=True)
                if data.get("type") == self.node_filter
            ]
            if len(analysis_nodes) == 0:
                raise ValueError(f"No nodes with type '{self.node_filter}' found")
        else:
            analysis_nodes = list(graph.nodes())

        n_nodes = len(analysis_nodes)
        print(f"Computing Path Diversity: {n_nodes} nodes")
        if self.node_filter:
            print(f"Node filter: {self.node_filter}")

        # Generate node pairs to analyze
        pairs = []
        for i in range(len(analysis_nodes)):
            for j in range(i + 1, len(analysis_nodes)):
                pairs.append((analysis_nodes[i], analysis_nodes[j]))

        # Sample pairs if requested
        if self.sample_pairs is not None and len(pairs) > self.sample_pairs:
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            indices = np.random.choice(len(pairs), self.sample_pairs, replace=False)
            pairs = [pairs[i] for i in indices]
            print(f"Sampling {self.sample_pairs} pairs from {n_nodes * (n_nodes - 1) // 2}")

        # Compute diversity metrics for each pair
        results = []
        connected_pairs = 0
        total_diverse_paths = 0

        for source, target in tqdm(pairs, desc="Computing path diversity"):
            try:
                # Get shortest path length
                shortest_length = nx.shortest_path_length(
                    graph, source, target, weight="length"
                )

                # Count paths within tolerance
                max_length = shortest_length * self.length_tolerance
                n_paths = self._count_paths_within_length(
                    graph, source, target, max_length, self.max_paths
                )

                results.append({
                    "source": source,
                    "target": target,
                    "shortest_distance": shortest_length,
                    "n_diverse_paths": n_paths,
                    "connected": True,
                })
                connected_pairs += 1
                total_diverse_paths += n_paths

            except nx.NetworkXNoPath:
                # Disconnected pair
                results.append({
                    "source": source,
                    "target": target,
                    "shortest_distance": np.inf,
                    "n_diverse_paths": 0,
                    "connected": False,
                })

        diversity_df = pd.DataFrame(results)

        # Compute summary statistics
        if connected_pairs > 0:
            avg_diverse_paths = total_diverse_paths / connected_pairs
            mean_distance = diversity_df[
                diversity_df["connected"]
            ]["shortest_distance"].mean()
        else:
            avg_diverse_paths = 0
            mean_distance = np.inf

        stats = {
            "total_pairs": len(pairs),
            "connected_pairs": connected_pairs,
            "disconnected_pairs": len(pairs) - connected_pairs,
            "connectivity_ratio": connected_pairs / len(pairs) if pairs else 0,
            "avg_diverse_paths": avg_diverse_paths,
            "mean_shortest_distance": mean_distance,
        }

        return diversity_df, stats

    def _count_paths_within_length(
        self,
        graph: nx.Graph,
        source,
        target,
        max_length: float,
        max_paths: int,
    ) -> int:
        """
        Count number of simple paths within a maximum length.

        Uses a bounded search to limit computation.
        """
        count = 0
        try:
            # Use simple paths generator with cutoff
            # Note: This can be expensive; we limit by max_paths
            for path in nx.all_simple_paths(
                graph, source, target, cutoff=self.max_path_hops
            ):
                path_length = sum(
                    graph[path[i]][path[i + 1]].get("length", 1.0)
                    for i in range(len(path) - 1)
                )
                if path_length <= max_length:
                    count += 1
                    if count >= max_paths:
                        break
        except nx.NetworkXNoPath:
            pass
        return count
