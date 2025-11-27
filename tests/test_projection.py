"""Tests for AEQD projection transformer."""

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

import geopandas as gpd

from src.projection.aeqd_transformer import AEQDTransformer


class TestAEQDTransformer:
    """Test cases for AEQDTransformer."""

    @pytest.fixture
    def transformer(self):
        """Create transformer centered at Tokyo."""
        return AEQDTransformer(
            center_lat=35.6762,
            center_lon=139.6503,
            half_size_m=1000.0,
        )

    def test_crs_creation(self, transformer):
        """Test that CRS is created correctly."""
        crs = transformer.crs
        assert crs is not None
        assert "aeqd" in crs.to_proj4().lower()

    def test_canvas_bounds(self, transformer):
        """Test canvas bounds are symmetric around origin."""
        bounds = transformer.get_canvas_bounds()
        assert bounds == (-1000, -1000, 1000, 1000)

    def test_canvas_bounds_wgs84(self, transformer):
        """Test WGS84 bounding box is reasonable."""
        bbox = transformer.get_canvas_bounds_wgs84()
        minx, miny, maxx, maxy = bbox

        # Center should be approximately at the center
        center_lon = (minx + maxx) / 2
        center_lat = (miny + maxy) / 2

        assert abs(center_lon - 139.6503) < 0.1
        assert abs(center_lat - 35.6762) < 0.1

    def test_canvas_shape(self, transformer):
        """Test canvas shape calculation."""
        shape = transformer.get_canvas_shape(resolution_m=1.0)
        assert shape == (2000, 2000)

        shape = transformer.get_canvas_shape(resolution_m=2.0)
        assert shape == (1000, 1000)

    def test_transform_gdf(self, transformer):
        """Test GeoDataFrame transformation."""
        # Create a point at the center
        gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[Point(139.6503, 35.6762)],
            crs="EPSG:4326",
        )

        transformed = transformer.transform_gdf(gdf)

        # Center should be at (0, 0) in AEQD
        point = transformed.geometry.iloc[0]
        assert abs(point.x) < 10  # Within 10m of center
        assert abs(point.y) < 10

    def test_clip_to_canvas(self, transformer):
        """Test clipping works correctly."""
        # Create a polygon that extends outside canvas
        gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[Polygon([
                (-500, -500), (-500, 1500), (500, 1500), (500, -500)
            ])],
            crs=transformer.crs,
        )

        clipped = transformer.clip_to_canvas(gdf)

        # Should be clipped to canvas bounds
        bounds = clipped.total_bounds
        assert bounds[0] >= -1000
        assert bounds[1] >= -1000
        assert bounds[2] <= 1000
        assert bounds[3] <= 1000

    def test_coords_to_pixel(self, transformer):
        """Test coordinate to pixel conversion."""
        # Center should map to middle of image
        row, col = transformer.coords_to_pixel(0, 0, resolution_m=1.0)
        assert row == 1000
        assert col == 1000

        # Top-left corner
        row, col = transformer.coords_to_pixel(-1000, 1000, resolution_m=1.0)
        assert row == 0
        assert col == 0

    def test_empty_gdf_handling(self, transformer):
        """Test handling of empty GeoDataFrame."""
        empty_gdf = gpd.GeoDataFrame(columns=["geometry"])
        result = transformer.transform_gdf(empty_gdf)
        assert len(result) == 0
