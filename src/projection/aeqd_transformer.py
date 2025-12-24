"""Dynamic Azimuthal Equidistant (AEQD) projection transformer."""

from dataclasses import dataclass

import geopandas as gpd
from pyproj import CRS, Transformer
try:
    from shapely import box
except ImportError:
    from shapely.geometry import box


@dataclass
class AEQDTransformer:
    """Transform coordinates using dynamic AEQD projection centered at target location."""

    center_lat: float
    center_lon: float
    half_size_m: float = 1000.0

    def __post_init__(self) -> None:
        """Initialize CRS and transformer."""
        # Create dynamic AEQD CRS centered at target location
        self._crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={self.center_lat} +lon_0={self.center_lon} "
            f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )

        # Create transformers
        self._to_local = Transformer.from_crs("EPSG:4326", self._crs, always_xy=True)
        self._to_wgs84 = Transformer.from_crs(self._crs, "EPSG:4326", always_xy=True)

    @property
    def crs(self) -> CRS:
        """Get the AEQD CRS."""
        return self._crs

    @property
    def crs_wkt(self) -> str:
        """Get the CRS as WKT string."""
        return self._crs.to_wkt()

    def transform_gdf(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Transform GeoDataFrame from WGS84 to local AEQD coordinates.

        Args:
            gdf: GeoDataFrame in EPSG:4326

        Returns:
            GeoDataFrame in local AEQD coordinates
        """
        if len(gdf) == 0:
            return gdf.set_crs(self._crs, allow_override=True)

        return gdf.to_crs(self._crs)

    def clip_to_canvas(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Clip geometries to canvas extent.

        Args:
            gdf: GeoDataFrame in local AEQD coordinates

        Returns:
            Clipped GeoDataFrame
        """
        if len(gdf) == 0:
            return gdf

        canvas_bounds = self.get_canvas_bounds()
        canvas_box = box(*canvas_bounds)

        clipped = gdf.clip(canvas_box)
        return clipped

    def get_canvas_bounds(self) -> tuple[float, float, float, float]:
        """
        Get canvas bounds in local coordinates.

        Returns:
            (minx, miny, maxx, maxy) in meters
        """
        return (
            -self.half_size_m,
            -self.half_size_m,
            self.half_size_m,
            self.half_size_m,
        )

    def get_canvas_bounds_wgs84(self) -> tuple[float, float, float, float]:
        """
        Get canvas bounds in WGS84 coordinates.

        Returns:
            (min_lon, min_lat, max_lon, max_lat)
        """
        # Transform corner points from local to WGS84
        min_lon, min_lat = self._to_wgs84.transform(-self.half_size_m, -self.half_size_m)
        max_lon, max_lat = self._to_wgs84.transform(self.half_size_m, self.half_size_m)

        return (min_lon, min_lat, max_lon, max_lat)

    def transform_and_clip(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Transform GeoDataFrame from WGS84 to local AEQD and clip to canvas.

        Args:
            gdf: GeoDataFrame in EPSG:4326

        Returns:
            Transformed and clipped GeoDataFrame
        """
        if len(gdf) == 0:
            return gdf.set_crs(self._crs, allow_override=True)

        transformed = self.transform_gdf(gdf)
        return self.clip_to_canvas(transformed)

    def get_canvas_size_px(self, resolution_m: float = 1.0) -> int:
        """
        Get canvas size in pixels.

        Args:
            resolution_m: Meters per pixel

        Returns:
            Canvas size in pixels (square)
        """
        return int(2 * self.half_size_m / resolution_m)

    def transform_point_to_local(self, lon: float, lat: float) -> tuple[float, float]:
        """
        Transform a single point from WGS84 to local coordinates.

        Args:
            lon: Longitude
            lat: Latitude

        Returns:
            (x, y) in local meters
        """
        return self._to_local.transform(lon, lat)

    def transform_point_to_wgs84(self, x: float, y: float) -> tuple[float, float]:
        """
        Transform a single point from local to WGS84 coordinates.

        Args:
            x: Local x coordinate (meters)
            y: Local y coordinate (meters)

        Returns:
            (lon, lat) in WGS84
        """
        return self._to_wgs84.transform(x, y)
