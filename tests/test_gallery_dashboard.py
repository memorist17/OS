"""Tests for gallery dashboard visualization."""

import numpy as np
import pandas as pd
import pytest

from src.visualization.gallery_dashboard import (
    create_clustering_summary_figure,
    create_gallery_dashboard,
    create_location_building_figure,
    create_location_metrics_card,
    create_location_network_figure,
    create_location_roads_figure,
    create_overview_lacunarity_figure,
    create_overview_mfa_figure,
    create_overview_percolation_figure,
)


class TestOverviewFigures:
    """Test cases for overview comparison figures."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results data for testing."""
        return {
            "run_001": {
                "display_name": "Tokyo",
                "site_id": "tokyo_jp",
                "mfa_spectrum": pd.DataFrame({
                    "q": np.linspace(-5, 5, 11),
                    "alpha": np.linspace(1.5, 2.5, 11),
                    "f_alpha": np.linspace(1.0, 2.0, 11),
                    "tau": np.linspace(-5, 5, 11),
                    "R2": [0.95] * 11,
                }),
                "mfa_dimensions": pd.DataFrame({
                    "q": np.linspace(-5, 5, 11),
                    "D_q": np.linspace(1.8, 2.2, 11),
                }),
                "lacunarity": pd.DataFrame({
                    "r": [2, 4, 8, 16, 32, 64],
                    "lambda": [2.5, 2.0, 1.7, 1.5, 1.3, 1.2],
                }),
                "lacunarity_fit": {"beta": 0.35, "R2": 0.92},
                "percolation": pd.DataFrame({
                    "d": np.linspace(1, 100, 20),
                    "giant_fraction": np.linspace(0.0, 1.0, 20),
                    "n_clusters": list(range(20, 0, -1)),
                }),
                "percolation_stats": {
                    "d_critical_50": 45.5,
                    "transition_width": 15.0,
                    "max_clusters": 20,
                },
            },
            "run_002": {
                "display_name": "Krakow",
                "site_id": "krakow_pl",
                "mfa_spectrum": pd.DataFrame({
                    "q": np.linspace(-5, 5, 11),
                    "alpha": np.linspace(1.6, 2.6, 11),
                    "f_alpha": np.linspace(1.1, 2.1, 11),
                    "tau": np.linspace(-4, 6, 11),
                    "R2": [0.93] * 11,
                }),
                "mfa_dimensions": pd.DataFrame({
                    "q": np.linspace(-5, 5, 11),
                    "D_q": np.linspace(1.7, 2.1, 11),
                }),
                "lacunarity": pd.DataFrame({
                    "r": [2, 4, 8, 16, 32, 64],
                    "lambda": [2.8, 2.2, 1.9, 1.6, 1.4, 1.25],
                }),
                "lacunarity_fit": {"beta": 0.40, "R2": 0.90},
                "percolation": pd.DataFrame({
                    "d": np.linspace(1, 100, 20),
                    "giant_fraction": np.linspace(0.0, 1.0, 20),
                    "n_clusters": list(range(25, 5, -1)),
                }),
                "percolation_stats": {
                    "d_critical_50": 50.0,
                    "transition_width": 12.0,
                    "max_clusters": 25,
                },
            },
        }

    def test_overview_mfa_figure_creation(self, sample_results):
        """Test MFA overview figure is created correctly."""
        fig = create_overview_mfa_figure(sample_results)

        # Should have traces for each location
        assert len(fig.data) == 2
        assert fig.layout.title.text == "Multifractal Spectrum f(α) - All Locations"

    def test_overview_lacunarity_figure_creation(self, sample_results):
        """Test lacunarity overview figure is created correctly."""
        fig = create_overview_lacunarity_figure(sample_results)

        # Should have traces for each location
        assert len(fig.data) == 2
        assert fig.layout.title.text == "Lacunarity Λ(r) - All Locations"

    def test_overview_percolation_figure_creation(self, sample_results):
        """Test percolation overview figure is created correctly."""
        fig = create_overview_percolation_figure(sample_results)

        # Should have traces for each location
        assert len(fig.data) == 2
        assert fig.layout.title.text == "Giant Component Fraction - All Locations"

    def test_clustering_summary_figure_creation(self, sample_results):
        """Test clustering summary figure is created correctly."""
        fig = create_clustering_summary_figure(sample_results)

        # Should have bar traces
        assert len(fig.data) > 0

    def test_empty_results_handling(self):
        """Test that empty results are handled gracefully."""
        empty_results = {}

        mfa_fig = create_overview_mfa_figure(empty_results)
        lac_fig = create_overview_lacunarity_figure(empty_results)
        perc_fig = create_overview_percolation_figure(empty_results)

        # Should not raise errors and create empty figures
        assert mfa_fig is not None
        assert lac_fig is not None
        assert perc_fig is not None


class TestLocationFigures:
    """Test cases for individual location figure creation."""

    @pytest.fixture
    def sample_location_results(self):
        """Create sample results for a single location."""
        return {
            "display_name": "Tokyo",
            "site_id": "tokyo_jp",
            "buildings_raster": np.random.rand(100, 100),
            "roads_raster": np.random.rand(100, 100),
            "mfa_dimensions": pd.DataFrame({
                "q": [0, 1, 2],
                "D_q": [1.85, 1.90, 1.95],
            }),
            "lacunarity_fit": {"beta": 0.35, "R2": 0.92},
            "percolation_stats": {
                "d_critical_50": 45.5,
                "transition_width": 15.0,
                "max_clusters": 20,
            },
        }

    def test_building_figure_creation(self, sample_location_results):
        """Test building raster figure is created correctly."""
        fig = create_location_building_figure(sample_location_results)

        assert fig is not None
        assert len(fig.data) == 1  # One heatmap trace
        assert "Buildings" in fig.layout.title.text

    def test_roads_figure_creation(self, sample_location_results):
        """Test roads raster figure is created correctly."""
        fig = create_location_roads_figure(sample_location_results)

        assert fig is not None
        assert len(fig.data) == 1  # One heatmap trace
        assert "Roads" in fig.layout.title.text

    def test_building_figure_without_data(self):
        """Test building figure returns None without raster data."""
        results_no_buildings = {"display_name": "Test"}
        fig = create_location_building_figure(results_no_buildings)
        assert fig is None

    def test_roads_figure_without_data(self):
        """Test roads figure returns None without raster data."""
        results_no_roads = {"display_name": "Test"}
        fig = create_location_roads_figure(results_no_roads)
        assert fig is None

    def test_network_figure_without_data(self):
        """Test network figure returns None without network path."""
        results_no_network = {"display_name": "Test"}
        fig = create_location_network_figure(results_no_network)
        assert fig is None


class TestMetricsCard:
    """Test cases for location metrics card creation."""

    def test_metrics_card_with_all_data(self):
        """Test metrics card with complete data."""
        results = {
            "display_name": "Tokyo",
            "mfa_dimensions": pd.DataFrame({
                "q": [0, 1, 2],
                "D_q": [1.85, 1.90, 1.95],
            }),
            "lacunarity_fit": {"beta": 0.35},
            "percolation_stats": {"d_critical_50": 45.5},
        }

        card = create_location_metrics_card(results)

        assert card is not None
        # Check that it's an html.Div
        assert hasattr(card, "children")

    def test_metrics_card_with_minimal_data(self):
        """Test metrics card with minimal data."""
        results = {"display_name": "Test Location"}

        card = create_location_metrics_card(results)

        assert card is not None
        assert hasattr(card, "children")

    def test_metrics_card_missing_display_name(self):
        """Test metrics card with missing display name uses default."""
        results = {}

        card = create_location_metrics_card(results)

        assert card is not None


class TestGalleryDashboard:
    """Test cases for the main gallery dashboard creation."""

    def test_dashboard_with_nonexistent_directory(self, tmp_path):
        """Test dashboard creation with non-existent directory."""
        nonexistent_dir = tmp_path / "nonexistent"

        # Should not raise error, but fall back to single dashboard
        app = create_gallery_dashboard(nonexistent_dir)
        assert app is not None

    def test_dashboard_with_empty_directory(self, tmp_path):
        """Test dashboard creation with empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        app = create_gallery_dashboard(empty_dir)
        assert app is not None
