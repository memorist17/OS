"""Overture Maps data fetcher using DuckDB + S3.

Based on official docs: https://docs.overturemaps.org
Uses anonymous S3 access to overturemaps-us-west-2 bucket.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import geopandas as gpd
from shapely import wkb
from tqdm import tqdm

# Latest release version (update as needed)
OVERTURE_RELEASE = "2025-11-19.0"
OVERTURE_S3_BASE = f"s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}"


@dataclass
class OvertureFetcher:
    """Fetch buildings and roads from Overture Maps via DuckDB S3 access."""

    bbox_wgs84: tuple[float, float, float, float] | None = None  # (min_lon, min_lat, max_lon, max_lat)
    lat: float | None = None
    lon: float | None = None
    half_size_m: float = 1000.0
    road_width_fallback: dict[str, float] = field(default_factory=lambda: {
        "motorway": 20,
        "trunk": 15,
        "primary": 12,
        "secondary": 10,
        "tertiary": 8,
        "residential": 6,
        "service": 4,
        "default": 5,
    })

    def __post_init__(self) -> None:
        """Initialize DuckDB connection with spatial extensions and S3 access."""
        self.conn = duckdb.connect()
        # Install and load required extensions
        self.conn.execute("INSTALL spatial; LOAD spatial;")
        self.conn.execute("INSTALL httpfs; LOAD httpfs;")
        # Configure anonymous S3 access (no-sign-request equivalent)
        self.conn.execute("SET s3_region='us-west-2';")
        self.conn.execute("SET s3_access_key_id='';")
        self.conn.execute("SET s3_secret_access_key='';")

    def __enter__(self) -> "OvertureFetcher":
        """Context manager enter."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def fetch_buildings(self, verbose: bool = True) -> gpd.GeoDataFrame:
        """Fetch building polygons within the bounding box."""
        bbox = self._get_bbox_wgs84()
        min_lon, min_lat, max_lon, max_lat = bbox

        if verbose:
            print(f"Fetching buildings in bbox: {bbox}")

        query = f"""
        SELECT
            id,
            names.primary AS name,
            height,
            num_floors,
            ST_AsWKB(geometry) AS geometry
        FROM read_parquet(
            '{OVERTURE_S3_BASE}/theme=buildings/type=building/*',
            filename=true,
            hive_partitioning=1
        )
        WHERE bbox.xmin >= {min_lon}
          AND bbox.xmax <= {max_lon}
          AND bbox.ymin >= {min_lat}
          AND bbox.ymax <= {max_lat}
        """

        result = self.conn.execute(query).fetchdf()

        if len(result) == 0:
            return gpd.GeoDataFrame(
                columns=["id", "name", "height", "num_floors", "geometry"],
                geometry="geometry",
                crs="EPSG:4326",
            )

        # Convert WKB to geometry (handle bytearray from DuckDB)
        geometries = [
            wkb.loads(bytes(g)) if g is not None else None 
            for g in tqdm(result["geometry"], desc="Parsing buildings", disable=not verbose)
        ]
        gdf = gpd.GeoDataFrame(
            result.drop(columns=["geometry"]),
            geometry=geometries,
            crs="EPSG:4326",
        )

        if verbose:
            print(f"Fetched {len(gdf)} buildings")

        return gdf

    def fetch_roads(self, verbose: bool = True) -> gpd.GeoDataFrame:
        """Fetch road segments within the bounding box."""
        bbox = self._get_bbox_wgs84()
        min_lon, min_lat, max_lon, max_lat = bbox

        if verbose:
            print(f"Fetching roads in bbox: {bbox}")

        query = f"""
        SELECT
            id,
            names.primary AS name,
            class,
            subclass,
            ST_AsWKB(geometry) AS geometry
        FROM read_parquet(
            '{OVERTURE_S3_BASE}/theme=transportation/type=segment/*',
            filename=true,
            hive_partitioning=1
        )
        WHERE bbox.xmin >= {min_lon}
          AND bbox.xmax <= {max_lon}
          AND bbox.ymin >= {min_lat}
          AND bbox.ymax <= {max_lat}
          AND subtype = 'road'
        """

        result = self.conn.execute(query).fetchdf()

        if len(result) == 0:
            return gpd.GeoDataFrame(
                columns=["id", "name", "class", "subclass", "width", "geometry"],
                geometry="geometry",
                crs="EPSG:4326",
            )

        # Convert WKB to geometry (handle bytearray from DuckDB)
        geometries = [
            wkb.loads(bytes(g)) if g is not None else None
            for g in tqdm(result["geometry"], desc="Parsing roads", disable=not verbose)
        ]
        gdf = gpd.GeoDataFrame(
            result.drop(columns=["geometry"]),
            geometry=geometries,
            crs="EPSG:4326",
        )

        # Add road width based on class
        gdf["width"] = gdf["class"].apply(
            lambda x: self.road_width_fallback.get(x, self.road_width_fallback["default"])
        )

        if verbose:
            print(f"Fetched {len(gdf)} road segments")

        return gdf

    def _get_bbox_wgs84(self) -> tuple[float, float, float, float]:
        """
        Get approximate WGS84 bounding box.

        Returns:
            (min_lon, min_lat, max_lon, max_lat)
        """
        # If bbox_wgs84 is provided directly, use it
        if self.bbox_wgs84 is not None:
            return self.bbox_wgs84

        # Otherwise compute from lat/lon
        if self.lat is None or self.lon is None:
            raise ValueError("Either bbox_wgs84 or lat/lon must be provided")

        # Approximate degrees per meter at given latitude
        import math
        lat_rad = math.radians(self.lat)
        m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
        m_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)

        delta_lat = self.half_size_m / m_per_deg_lat
        delta_lon = self.half_size_m / m_per_deg_lon

        return (
            self.lon - delta_lon,  # min_lon
            self.lat - delta_lat,  # min_lat
            self.lon + delta_lon,  # max_lon
            self.lat + delta_lat,  # max_lat
        )

    def get_width_fallback_stats(self, roads_gdf: gpd.GeoDataFrame) -> dict:
        """
        Get statistics about road width fallback usage.

        Args:
            roads_gdf: Road GeoDataFrame with 'class' and 'width' columns

        Returns:
            Dictionary with road count and class distribution
        """
        if len(roads_gdf) == 0:
            return {"count": 0, "class_distribution": {}}

        class_counts = roads_gdf["class"].value_counts().to_dict()
        return {
            "count": len(roads_gdf),
            "class_distribution": class_counts,
        }

    def close(self) -> None:
        """Close DuckDB connection."""
        self.conn.close()
