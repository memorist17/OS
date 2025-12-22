"""Tests for rasterization."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from src.preprocessing.rasterizer import Rasterizer


class TestRasterizer:
    """Test cases for Rasterizer."""

    def test_rasterizer_initialization(self):
        """Test that Rasterizer initializes correctly."""
        rasterizer = Rasterizer(canvas_size=100, resolution_m=1.0, half_size_m=50.0)
        assert rasterizer.canvas_size == 100
        assert rasterizer.resolution_m == 1.0
        assert rasterizer.half_size_m == 50.0
        assert rasterizer.supersample_factor == 1
        assert rasterizer.interpolation == "bilinear"

    def test_rasterizer_with_supersample(self):
        """Test that Rasterizer initializes with supersampling."""
        rasterizer = Rasterizer(
            canvas_size=100,
            resolution_m=1.0,
            half_size_m=50.0,
            supersample_factor=2,
            interpolation="bicubic",
        )
        assert rasterizer.supersample_factor == 2
        assert rasterizer.interpolation == "bicubic"
        assert rasterizer._internal_size == 200

    def test_rasterizer_invalid_supersample(self):
        """Test that invalid supersample factor raises error."""
        with pytest.raises(ValueError, match="supersample_factor must be >= 1"):
            Rasterizer(canvas_size=100, supersample_factor=0)

    def test_building_rasterization_empty(self):
        """Test that empty GeoDataFrame returns zero array."""
        rasterizer = Rasterizer(canvas_size=100, resolution_m=1.0, half_size_m=50.0)
        empty_gdf = gpd.GeoDataFrame(geometry=[])
        result = rasterizer.rasterize_buildings(empty_gdf, verbose=False)

        assert result.shape == (100, 100)
        assert result.dtype == np.uint8
        assert np.sum(result) == 0

    def test_building_rasterization_single_polygon(self):
        """Test that a single building is rasterized correctly."""
        rasterizer = Rasterizer(canvas_size=100, resolution_m=1.0, half_size_m=50.0)

        # Create a 10x10m square building at the center
        building = Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
        gdf = gpd.GeoDataFrame(geometry=[building])

        result = rasterizer.rasterize_buildings(gdf, verbose=False)

        assert result.shape == (100, 100)
        assert result.dtype == np.uint8
        assert np.sum(result) > 0  # Should have non-zero pixels

    def test_building_rasterization_with_supersample(self):
        """Test that supersampling produces smoother edges."""
        # Create a building
        building = Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
        gdf = gpd.GeoDataFrame(geometry=[building])

        # Rasterize without supersampling
        rasterizer_1x = Rasterizer(
            canvas_size=100,
            resolution_m=1.0,
            half_size_m=50.0,
            supersample_factor=1,
        )
        result_1x = rasterizer_1x.rasterize_buildings(gdf, verbose=False)

        # Rasterize with 2x supersampling
        rasterizer_2x = Rasterizer(
            canvas_size=100,
            resolution_m=1.0,
            half_size_m=50.0,
            supersample_factor=2,
        )
        result_2x = rasterizer_2x.rasterize_buildings(gdf, verbose=False)

        # Both should have the same shape
        assert result_1x.shape == result_2x.shape == (100, 100)
        # Both should have building pixels
        assert np.sum(result_1x) > 0
        assert np.sum(result_2x) > 0

    def test_road_rasterization_empty(self):
        """Test that empty roads GeoDataFrame returns zero array."""
        rasterizer = Rasterizer(canvas_size=100, resolution_m=1.0, half_size_m=50.0)
        empty_gdf = gpd.GeoDataFrame(geometry=[])
        result = rasterizer.rasterize_roads(empty_gdf, verbose=False)

        assert result.shape == (100, 100)
        assert result.dtype == np.uint8
        assert np.sum(result) == 0

    def test_road_rasterization_single_line(self):
        """Test that a single road is rasterized correctly."""
        rasterizer = Rasterizer(canvas_size=100, resolution_m=1.0, half_size_m=50.0)

        # Create a road line
        road = LineString([(-20, 0), (20, 0)])
        gdf = gpd.GeoDataFrame(geometry=[road])
        gdf["width"] = 5.0

        result = rasterizer.rasterize_roads(gdf, verbose=False)

        assert result.shape == (100, 100)
        assert result.dtype == np.uint8
        assert np.sum(result) > 0  # Should have non-zero pixels

    def test_road_rasterization_with_supersample(self):
        """Test that supersampling works for roads."""
        road = LineString([(-20, 0), (20, 0)])
        gdf = gpd.GeoDataFrame(geometry=[road])
        gdf["width"] = 5.0

        # Rasterize without supersampling
        rasterizer_1x = Rasterizer(
            canvas_size=100,
            resolution_m=1.0,
            half_size_m=50.0,
            supersample_factor=1,
        )
        result_1x = rasterizer_1x.rasterize_roads(gdf, verbose=False)

        # Rasterize with 2x supersampling
        rasterizer_2x = Rasterizer(
            canvas_size=100,
            resolution_m=1.0,
            half_size_m=50.0,
            supersample_factor=2,
        )
        result_2x = rasterizer_2x.rasterize_roads(gdf, verbose=False)

        # Both should have the same shape
        assert result_1x.shape == result_2x.shape == (100, 100)
        # Both should have road pixels
        assert np.sum(result_1x) > 0
        assert np.sum(result_2x) > 0

    def test_interpolation_methods(self):
        """Test that all interpolation methods work."""
        building = Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
        gdf = gpd.GeoDataFrame(geometry=[building])

        for method in ["nearest", "bilinear", "bicubic", "lanczos"]:
            rasterizer = Rasterizer(
                canvas_size=100,
                resolution_m=1.0,
                half_size_m=50.0,
                supersample_factor=2,
                interpolation=method,
            )
            result = rasterizer.rasterize_buildings(gdf, verbose=False)
            assert result.shape == (100, 100)
            assert np.sum(result) > 0

    def test_output_shape(self):
        """Test that output has correct shape."""
        for size in [50, 100, 200]:
            rasterizer = Rasterizer(
                canvas_size=size, resolution_m=1.0, half_size_m=size / 2
            )
            building = Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
            gdf = gpd.GeoDataFrame(geometry=[building])
            result = rasterizer.rasterize_buildings(gdf, verbose=False)
            assert result.shape == (size, size)

    def test_combined_rasterization(self):
        """Test combined building and road rasterization."""
        rasterizer = Rasterizer(canvas_size=100, resolution_m=1.0, half_size_m=50.0)

        # Create test geometries
        building = Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
        buildings_gdf = gpd.GeoDataFrame(geometry=[building])

        road = LineString([(-20, 0), (20, 0)])
        roads_gdf = gpd.GeoDataFrame(geometry=[road])
        roads_gdf["width"] = 5.0

        result = rasterizer.rasterize_combined(buildings_gdf, roads_gdf, verbose=False)

        assert result.shape == (100, 100)
        assert result.dtype == np.uint8
        assert np.sum(result) > 0
        # Combined should be binary (0 or 1)
        assert np.all((result == 0) | (result == 1))

