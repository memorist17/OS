"""Overture Maps data fetcher using DuckDB + S3."""

from dataclasses import dataclass, field

import duckdb
import geopandas as gpd
import pandas as pd
from shapely import wkb
from tqdm import tqdm


@dataclass
class OvertureFetcher:
    """Fetch buildings and roads from Overture Maps via DuckDB."""

    lat: float
    lon: float
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
        """Initialize DuckDB connection with spatial extensions."""
        self.conn = duckdb.connect()
        self.conn.execute("INSTALL spatial; LOAD spatial;")
        self.conn.execute("INSTALL httpfs; LOAD httpfs;")
        self.conn.execute("SET s3_region='us-west-2';")

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
            's3://overturemaps-us-west-2/release/2024-11-13.0/theme=buildings/type=building/*',
            filename=true,
            hive_partitioning=true
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

        # Convert WKB to geometry
        geometries = [wkb.loads(g) for g in tqdm(result["geometry"], desc="Parsing buildings", disable=not verbose)]
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
            's3://overturemaps-us-west-2/release/2024-11-13.0/theme=transportation/type=segment/*',
            filename=true,
            hive_partitioning=true
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

        # Convert WKB to geometry
        geometries = [wkb.loads(g) for g in tqdm(result["geometry"], desc="Parsing roads", disable=not verbose)]
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

    def close(self) -> None:
        """Close DuckDB connection."""
        self.conn.close()
