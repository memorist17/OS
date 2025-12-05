"""Tests for analysis modules."""

import numpy as np
import pytest

from src.analysis.lacunarity import LacunarityAnalyzer
from src.analysis.multifractal import MultifractalAnalyzer


class TestMultifractalAnalyzer:
    """Test cases for MultifractalAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with small parameters for testing."""
        return MultifractalAnalyzer(
            r_min=2,
            r_max=64,
            r_steps=5,
            q_min=-5,
            q_max=5,
            q_steps=11,
            grid_shift_count=4,
        )

    def test_box_sizes_generation(self, analyzer):
        """Test that box sizes are generated correctly."""
        sizes = analyzer._get_box_sizes()

        assert sizes[0] >= 2
        assert sizes[-1] <= 64
        assert len(sizes) <= 5
        # Should be monotonically increasing
        assert all(sizes[i] < sizes[i + 1] for i in range(len(sizes) - 1))

    def test_q_values_generation(self, analyzer):
        """Test that q values are generated correctly."""
        q_values = analyzer._get_q_values()

        assert q_values[0] == -5
        assert q_values[-1] == 5
        assert len(q_values) == 11

    def test_uniform_image_analysis(self, analyzer):
        """Test analysis of uniform image."""
        # Uniform image should have narrow spectrum
        image = np.ones((128, 128), dtype=np.float64) * 100
        spectrum_df, mesh = analyzer.analyze(image, verbose=False)

        assert len(spectrum_df) == 11  # q_steps
        assert "alpha" in spectrum_df.columns
        assert "f_alpha" in spectrum_df.columns

    def test_random_image_analysis(self, analyzer):
        """Test analysis of random image."""
        np.random.seed(42)
        image = np.random.rand(128, 128) * 255

        spectrum_df, mesh = analyzer.analyze(image, verbose=False)

        # Should have valid spectrum
        assert not spectrum_df["alpha"].isna().all()
        assert spectrum_df["R2"].mean() > 0.5  # Reasonable fit

    def test_spectrum_width(self, analyzer):
        """Test spectrum width calculation."""
        np.random.seed(42)
        image = np.random.rand(128, 128) * 255
        spectrum_df, _ = analyzer.analyze(image, verbose=False)

        width = analyzer.get_spectrum_width(spectrum_df)
        assert width >= 0


class TestLacunarityAnalyzer:
    """Test cases for LacunarityAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with small parameters for testing."""
        return LacunarityAnalyzer(
            r_min=2,
            r_max=64,
            r_steps=5,
            full_scan=True,
        )

    def test_box_sizes_generation(self, analyzer):
        """Test that box sizes are generated correctly."""
        sizes = analyzer._get_box_sizes()

        assert sizes[0] >= 2
        assert sizes[-1] <= 64

    def test_empty_image_lacunarity(self, analyzer):
        """Test lacunarity of empty image."""
        image = np.zeros((128, 128), dtype=np.uint8)
        lacunarity_df, mesh = analyzer.analyze(image, verbose=False)

        # Empty image should have NaN lacunarity (division by zero)
        assert lacunarity_df["lambda"].isna().all() or (lacunarity_df["mu"] == 0).all()

    def test_full_image_lacunarity(self, analyzer):
        """Test lacunarity of fully filled image."""
        image = np.ones((128, 128), dtype=np.uint8)
        lacunarity_df, mesh = analyzer.analyze(image, verbose=False)

        # Full image should have lacunarity = 1 (no variance)
        assert all(lacunarity_df["lambda"] == 1.0)

    def test_random_binary_image(self, analyzer):
        """Test lacunarity of random binary image."""
        np.random.seed(42)
        image = (np.random.rand(128, 128) > 0.5).astype(np.uint8)

        lacunarity_df, mesh = analyzer.analyze(image, verbose=False)

        # Should have valid lacunarity values > 1
        valid = lacunarity_df[lacunarity_df["lambda"].notna()]
        assert len(valid) > 0
        assert valid["lambda"].min() >= 1.0

    def test_decay_exponent(self, analyzer):
        """Test decay exponent calculation."""
        np.random.seed(42)
        image = (np.random.rand(128, 128) > 0.3).astype(np.uint8)
        lacunarity_df, _ = analyzer.analyze(image, verbose=False)

        beta, r2 = analyzer.get_decay_exponent(lacunarity_df)

        # Should have valid decay exponent
        assert not np.isnan(beta)
        assert 0 <= r2 <= 1

    def test_lacunarity_at_scale(self, analyzer):
        """Test interpolation at specific scale."""
        np.random.seed(42)
        image = (np.random.rand(128, 128) > 0.5).astype(np.uint8)
        lacunarity_df, _ = analyzer.analyze(image, verbose=False)

        # Get lacunarity at a scale that exists
        r_existing = lacunarity_df["r"].iloc[0]
        lam = analyzer.get_lacunarity_at_scale(lacunarity_df, int(r_existing))

        assert lam == lacunarity_df.iloc[0]["lambda"]


class TestPercolationAnalyzer:
    """Test cases for PercolationAnalyzer."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple test graph."""
        import networkx as nx
        G = nx.Graph()
        # Create a small network with 5 nodes
        # 0 -- 1 -- 2
        # |    |
        # 3 -- 4
        G.add_node(0, x=0, y=0, type="building")
        G.add_node(1, x=10, y=0, type="road")
        G.add_node(2, x=20, y=0, type="building")
        G.add_node(3, x=0, y=10, type="building")
        G.add_node(4, x=10, y=10, type="road")

        G.add_edge(0, 1, length=10.0)
        G.add_edge(1, 2, length=10.0)
        G.add_edge(0, 3, length=10.0)
        G.add_edge(1, 4, length=10.0)
        G.add_edge(3, 4, length=10.0)

        return G

    @pytest.fixture
    def disconnected_graph(self):
        """Create a graph with disconnected components."""
        import networkx as nx
        G = nx.Graph()
        # Component 1: nodes 0, 1
        G.add_node(0, x=0, y=0, type="building")
        G.add_node(1, x=10, y=0, type="building")
        G.add_edge(0, 1, length=10.0)

        # Component 2: nodes 2, 3 (disconnected from component 1)
        G.add_node(2, x=100, y=100, type="building")
        G.add_node(3, x=110, y=100, type="building")
        G.add_edge(2, 3, length=10.0)

        return G

    @pytest.fixture
    def analyzer_edge(self):
        """Create analyzer with edge-based distance."""
        from src.analysis.percolation import PercolationAnalyzer
        return PercolationAnalyzer(
            d_min=1,
            d_max=50,
            d_steps=10,
            distance_type="edge",
        )

    @pytest.fixture
    def analyzer_shortest_path(self):
        """Create analyzer with shortest path distance."""
        from src.analysis.percolation import PercolationAnalyzer
        return PercolationAnalyzer(
            d_min=1,
            d_max=50,
            d_steps=10,
            distance_type="shortest_path",
        )

    def test_edge_based_analysis(self, analyzer_edge, simple_graph):
        """Test edge-based percolation analysis."""
        percolation_df, mesh = analyzer_edge.analyze(simple_graph)

        assert len(percolation_df) == 10  # d_steps
        assert "d" in percolation_df.columns
        assert "max_cluster_size" in percolation_df.columns
        assert "n_clusters" in percolation_df.columns
        assert "giant_fraction" in percolation_df.columns

        # At small threshold, all nodes should be disconnected (5 clusters)
        assert percolation_df.iloc[0]["n_clusters"] == 5
        # At large threshold (>10), all nodes should be connected (1 cluster)
        assert percolation_df.iloc[-1]["n_clusters"] == 1

    def test_shortest_path_analysis(self, analyzer_shortest_path, simple_graph):
        """Test shortest path distance percolation analysis."""
        percolation_df, mesh = analyzer_shortest_path.analyze(simple_graph)

        assert len(percolation_df) == 10
        # At large threshold, all nodes should be connected
        assert percolation_df.iloc[-1]["giant_fraction"] == 1.0

    def test_disconnected_components_edge(self, analyzer_edge, disconnected_graph):
        """Test handling of disconnected components with edge-based analysis."""
        percolation_df, mesh = analyzer_edge.analyze(disconnected_graph)

        # Even at maximum threshold, should have 2 components (disconnected)
        # because edge-based only considers direct edges
        final_clusters = percolation_df.iloc[-1]["n_clusters"]
        assert final_clusters >= 2

    def test_disconnected_components_shortest_path(
        self, analyzer_shortest_path, disconnected_graph
    ):
        """Test handling of disconnected components with shortest path analysis."""
        percolation_df, mesh = analyzer_shortest_path.analyze(disconnected_graph)

        # Even at maximum threshold, disconnected pairs remain disconnected
        final_giant_fraction = percolation_df.iloc[-1]["giant_fraction"]
        # Giant component should be 2/4 = 0.5 at best
        assert final_giant_fraction <= 0.5

    def test_node_filter(self, simple_graph):
        """Test node filtering for building-only analysis."""
        from src.analysis.percolation import PercolationAnalyzer
        analyzer = PercolationAnalyzer(
            d_min=1,
            d_max=50,
            d_steps=5,
            distance_type="shortest_path",
            node_filter="building",
        )
        percolation_df, mesh = analyzer.analyze(simple_graph)

        # Should only analyze building nodes (3 buildings)
        assert mesh.shape[1] == 3

    def test_find_percolation_threshold(self, analyzer_edge, simple_graph):
        """Test finding critical percolation threshold."""
        percolation_df, mesh = analyzer_edge.analyze(simple_graph)
        d_critical = analyzer_edge.find_percolation_threshold(percolation_df, 0.5)

        assert d_critical > 0
        assert d_critical <= analyzer_edge.d_max

    def test_compute_susceptibility(self, analyzer_edge, simple_graph):
        """Test susceptibility computation."""
        percolation_df, mesh = analyzer_edge.analyze(simple_graph)
        susceptibility_df = analyzer_edge.compute_susceptibility(percolation_df)

        assert len(susceptibility_df) == len(percolation_df)
        assert "d" in susceptibility_df.columns
        assert "susceptibility" in susceptibility_df.columns

    def test_analyze_with_statistics(self, analyzer_edge, simple_graph):
        """Test analysis with statistics."""
        percolation_df, stats = analyzer_edge.analyze_with_statistics(simple_graph)

        assert "d_critical_50" in stats
        assert "d_critical_10" in stats
        assert "d_critical_90" in stats
        assert "transition_width" in stats
        assert "max_clusters" in stats

    def test_empty_graph_raises_error(self, analyzer_edge):
        """Test that empty graph raises ValueError."""
        import networkx as nx
        G = nx.Graph()

        with pytest.raises(ValueError, match="Graph has no nodes"):
            analyzer_edge.analyze(G)


class TestPathDiversityAnalyzer:
    """Test cases for PathDiversityAnalyzer."""

    @pytest.fixture
    def simple_graph(self):
        """Create a test graph with multiple paths."""
        import networkx as nx
        G = nx.Graph()
        # Create a graph with alternative paths
        # 0 -- 1 -- 3
        # |    |    |
        # 2 ------- 4
        G.add_node(0, x=0, y=0, type="building")
        G.add_node(1, x=10, y=0, type="road")
        G.add_node(2, x=0, y=10, type="building")
        G.add_node(3, x=20, y=0, type="building")
        G.add_node(4, x=20, y=10, type="building")

        G.add_edge(0, 1, length=10.0)
        G.add_edge(1, 3, length=10.0)
        G.add_edge(0, 2, length=10.0)
        G.add_edge(2, 4, length=20.0)
        G.add_edge(3, 4, length=10.0)
        G.add_edge(1, 4, length=14.14)  # Diagonal connection

        return G

    @pytest.fixture
    def analyzer(self):
        """Create path diversity analyzer."""
        from src.analysis.percolation import PathDiversityAnalyzer
        return PathDiversityAnalyzer(
            max_paths=3,
            length_tolerance=1.5,
            sample_pairs=None,
        )

    def test_basic_analysis(self, analyzer, simple_graph):
        """Test basic path diversity analysis."""
        diversity_df, stats = analyzer.analyze(simple_graph)

        assert len(diversity_df) > 0
        assert "source" in diversity_df.columns
        assert "target" in diversity_df.columns
        assert "shortest_distance" in diversity_df.columns
        assert "n_diverse_paths" in diversity_df.columns
        assert "connected" in diversity_df.columns

    def test_stats_computed(self, analyzer, simple_graph):
        """Test that statistics are properly computed."""
        diversity_df, stats = analyzer.analyze(simple_graph)

        assert "total_pairs" in stats
        assert "connected_pairs" in stats
        assert "disconnected_pairs" in stats
        assert "connectivity_ratio" in stats
        assert "avg_diverse_paths" in stats

    def test_node_filter(self, simple_graph):
        """Test node filtering for building-only analysis."""
        from src.analysis.percolation import PathDiversityAnalyzer
        analyzer = PathDiversityAnalyzer(
            max_paths=3,
            length_tolerance=1.5,
            node_filter="building",
        )
        diversity_df, stats = analyzer.analyze(simple_graph)

        # Should only analyze pairs between building nodes
        # Buildings: 0, 2, 3, 4 = 4 nodes = 6 pairs
        assert stats["total_pairs"] == 6

    def test_sample_pairs(self, simple_graph):
        """Test sampling of node pairs."""
        from src.analysis.percolation import PathDiversityAnalyzer
        analyzer = PathDiversityAnalyzer(
            max_paths=3,
            length_tolerance=1.5,
            sample_pairs=3,  # Sample only 3 pairs
        )
        diversity_df, stats = analyzer.analyze(simple_graph)

        assert stats["total_pairs"] == 3


class TestIntegration:
    """Integration tests for analysis pipeline."""

    def test_consistent_box_sizes(self):
        """Test that MFA and Lacunarity use consistent box sizes."""
        mfa = MultifractalAnalyzer(r_min=4, r_max=128, r_steps=10)
        lac = LacunarityAnalyzer(r_min=4, r_max=128, r_steps=10)

        mfa_sizes = mfa._get_box_sizes()
        lac_sizes = lac._get_box_sizes()

        # Should have same box sizes
        np.testing.assert_array_equal(mfa_sizes, lac_sizes)
