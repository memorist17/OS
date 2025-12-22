"""Rasterize vector geometries to NumPy arrays."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import geopandas as gpd
import numpy as np
from rasterio import features
from rasterio.transform import from_bounds
from shapely.geometry import mapping
from tqdm import tqdm

# Type aliases for quality options
InterpolationMethod = Literal["nearest", "bilinear", "bicubic", "lanczos"]


@dataclass
class Rasterizer:
    """Convert vector geometries to raster images with quality options.

    Attributes:
        canvas_size: Output size in pixels (square)
        resolution_m: Meters per pixel resolution
        half_size_m: Canvas half-size in meters (±half_size_m)
        supersample_factor: Rendering multiplier for anti-aliasing (1 = no supersampling)
        interpolation: Downsampling method when supersample_factor > 1
            - "nearest": No interpolation (fastest, lowest quality)
            - "bilinear": Bilinear interpolation (good balance)
            - "bicubic": Bicubic interpolation (better quality, slower)
            - "lanczos": Lanczos interpolation (best quality, slowest)
    """

    canvas_size: int  # pixels (square)
    resolution_m: float = 1.0
    half_size_m: float = 1000.0  # canvas半径
    supersample_factor: int = 1  # Anti-aliasing via supersampling
    interpolation: InterpolationMethod = "bilinear"

    def __post_init__(self) -> None:
        """Initialize raster transform."""
        if self.supersample_factor < 1:
            raise ValueError("supersample_factor must be >= 1")

        # キャンバス範囲: -half_size_m to +half_size_m
        self.bounds = (
            -self.half_size_m,
            -self.half_size_m,
            self.half_size_m,
            self.half_size_m,
        )

        # 内部レンダリングサイズ（スーパーサンプリング適用）
        self._internal_size = self.canvas_size * self.supersample_factor
        self.transform = from_bounds(*self.bounds, self._internal_size, self._internal_size)

        # OpenCV interpolation mapping
        self._interpolation_map = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }

    def _downsample(self, array: np.ndarray) -> np.ndarray:
        """Downsample array from internal size to output size.

        Args:
            array: High-resolution array (internal_size, internal_size)

        Returns:
            Downsampled array (canvas_size, canvas_size)
        """
        if self.supersample_factor == 1:
            return array

        interp_method = self._interpolation_map[self.interpolation]
        downsampled = cv2.resize(
            array,
            (self.canvas_size, self.canvas_size),
            interpolation=interp_method,
        )
        return downsampled

    def rasterize_buildings(
        self, buildings: gpd.GeoDataFrame, verbose: bool = True
    ) -> np.ndarray:
        """
        Rasterize buildings to binary image (0/1) with anti-aliasing.

        Args:
            buildings: GeoDataFrame with building polygons in local coordinates
            verbose: Show progress

        Returns:
            Binary numpy array (H, W) with 1 for building pixels
        """
        if len(buildings) == 0:
            return np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)

        if verbose:
            ss_msg = (
                f" (supersample {self.supersample_factor}x)"
                if self.supersample_factor > 1
                else ""
            )
            print(f"Rasterizing {len(buildings)} buildings{ss_msg}...")

        # Convert to shapes for rasterio
        shapes = [
            (mapping(geom), 1)
            for geom in buildings.geometry
            if geom is not None and geom.is_valid
        ]

        if not shapes:
            return np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)

        # Rasterize at internal (supersampled) resolution
        raster = features.rasterize(
            shapes=shapes,
            out_shape=(self._internal_size, self._internal_size),
            transform=self.transform,
            fill=0,
            dtype=np.uint8,
        )

        # Downsample to target size with quality interpolation
        raster = self._downsample(raster)

        if verbose:
            coverage = np.sum(raster > 0) / raster.size * 100
            print(f"Building coverage: {coverage:.2f}%")

        return raster

    def rasterize_roads(
        self,
        roads: gpd.GeoDataFrame,
        width_column: str = "width",
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Rasterize roads to weighted grayscale image (0-255) with anti-aliasing.

        Roads are buffered by their width and rendered with intensity
        proportional to road importance.

        Args:
            roads: GeoDataFrame with road lines in local coordinates
            width_column: Column name containing road width
            verbose: Show progress

        Returns:
            Grayscale numpy array (H, W) with road intensities (0-255)
        """
        if len(roads) == 0:
            return np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)

        if verbose:
            ss_msg = (
                f" (supersample {self.supersample_factor}x)"
                if self.supersample_factor > 1
                else ""
            )
            print(f"Rasterizing {len(roads)} road segments{ss_msg}...")

        # Buffer roads by half their width to get polygons
        shapes = []
        max_width = roads[width_column].max() if width_column in roads.columns else 20

        for _, row in tqdm(
            roads.iterrows(),
            total=len(roads),
            desc="Buffering roads",
            disable=not verbose,
        ):
            geom = row.geometry
            if geom is None or geom.is_empty or not geom.is_valid:
                continue

            width = row.get(width_column, 5)
            buffered = geom.buffer(width / 2, cap_style=2)  # flat cap

            if buffered.is_valid and not buffered.is_empty:
                # Intensity proportional to width (thicker = more important)
                intensity = int(min(255, (width / max_width) * 255))
                intensity = max(50, intensity)  # minimum visibility
                shapes.append((mapping(buffered), intensity))

        if not shapes:
            return np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)

        # Rasterize at internal (supersampled) resolution
        raster = features.rasterize(
            shapes=shapes,
            out_shape=(self._internal_size, self._internal_size),
            transform=self.transform,
            fill=0,
            dtype=np.uint8,
            merge_alg=features.MergeAlg.add,  # Overlap areas get summed
        )

        # Clip to 255 before downsampling
        raster = np.clip(raster, 0, 255).astype(np.uint8)

        # Downsample to target size with quality interpolation
        raster = self._downsample(raster)

        if verbose:
            coverage = np.sum(raster > 0) / raster.size * 100
            print(f"Road coverage: {coverage:.2f}%")

        return raster

    def rasterize_combined(
        self,
        buildings: gpd.GeoDataFrame,
        roads: gpd.GeoDataFrame,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Create combined binary raster (buildings + roads).

        Args:
            buildings: Building polygons in local coordinates
            roads: Road lines in local coordinates
            verbose: Show progress

        Returns:
            Binary numpy array (H, W) with 1 for any urban structure
        """
        building_raster = self.rasterize_buildings(buildings, verbose=verbose)
        road_raster = self.rasterize_roads(roads, verbose=verbose)

        # Combine: any non-zero value becomes 1
        combined = ((building_raster > 0) | (road_raster > 0)).astype(np.uint8)

        if verbose:
            coverage = np.sum(combined > 0) / combined.size * 100
            print(f"Combined coverage: {coverage:.2f}%")

        return combined

    def save(self, array: np.ndarray, output_path: Path) -> None:
        """Save raster array to .npy file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, array)
