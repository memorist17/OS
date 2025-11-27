"""Continuum Percolation Analysis using NetworkX connected components."""

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
       - Keep only edges where length <= d
       - Count connected components and max component size
    3. Track percolation transition (emergence of giant component)
    """

    d_min: float = 1
    d_max: float = 100
    d_steps: int = 50

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
        """
        # Load graph if path provided
        if isinstance(graph, (str, Path)):
            graph = nx.read_graphml(str(graph))

        if graph.number_of_nodes() == 0:
            raise ValueError("Graph has no nodes")

        n_nodes = graph.number_of_nodes()
        thresholds = self._get_thresholds()

        print(f"Computing Percolation: {len(thresholds)} thresholds, {n_nodes} nodes")
        print(f"Distance range: {self.d_min:.1f} to {self.d_max:.1f}")

        # Get edge lengths
        edge_lengths = {}
        for u, v, data in graph.edges(data=True):
            length = float(data.get("length", 1.0))
            edge_lengths[(u, v)] = length

        # Map node labels to indices
        node_list = list(graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        # Compute percolation metrics for each threshold
        results = []
        mesh = np.zeros((len(thresholds), n_nodes), dtype=np.int32)

        for d_idx, d in enumerate(tqdm(thresholds, desc="Computing percolation")):
            # Create filtered graph view
            filtered = self._filter_graph(graph, edge_lengths, d)

            # Compute connected components
            components = list(nx.connected_components(filtered))
            n_clusters = len(components)

            # Find largest component
            if components:
                largest = max(components, key=len)
                max_size = len(largest)
            else:
                max_size = 0
                largest = set()

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
        # This requires the mesh data; for now return simplified version
        # based on n_clusters and max_cluster_size
        d = percolation_df["d"].values
        n = percolation_df["n_clusters"].values
        max_s = percolation_df["max_cluster_size"].values
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
