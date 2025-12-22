#!/usr/bin/env python
"""Phase 1: Data acquisition and preprocessing pipeline.

Overture Mapsから都市データを取得し、AEQD投影で前処理する。
建物ラスター、道路ラスター、ネットワークグラフを生成。
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.acquisition.overture_fetcher import OvertureFetcher
from src.preprocessing.network_builder import GRAPH_TOOL_AVAILABLE, NetworkBuilder
from src.preprocessing.rasterizer import Rasterizer
from src.projection.aeqd_transformer import AEQDTransformer


def convert_to_native(obj):
    """Convert numpy types to Python native types for YAML serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_metadata(
    output_dir: Path,
    site_id: str,
    center_lat: float,
    center_lon: float,
    transformer: AEQDTransformer,
    config: dict,
    stats: dict,
) -> None:
    """Save metadata.yaml with processing details."""
    # Convert numpy types to native Python types
    stats = convert_to_native(stats)

    metadata = {
        "meta_info": {
            "site_id": site_id,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "crs_projection": f"AEQD centered at [{center_lat}, {center_lon}]",
            "crs_wkt": transformer.crs_wkt,
            "resolution_m": config["canvas"]["resolution_m"],
            "canvas_size_m": config["canvas"]["half_size_m"] * 2,
            "created_at": datetime.now().isoformat(),
        },
        "data_processing_log": {
            "buildings": stats.get("buildings", {}),
            "roads": stats.get("roads", {}),
            "network": stats.get("network", {}),
        },
        "config_used": config,
    }

    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)


def main():
    """Run data acquisition pipeline."""
    parser = argparse.ArgumentParser(
        description="Acquire and preprocess urban data from Overture Maps"
    )
    parser.add_argument("--lat", type=float, required=True, help="Center latitude")
    parser.add_argument("--lon", type=float, required=True, help="Center longitude")
    parser.add_argument("--site-id", type=str, required=True, help="Site identifier")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Config file path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory",
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip network building (if graph-tool not available)",
    )

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("Urban Structure Acquisition Pipeline")
    print("=" * 60)
    print(f"Site: {args.site_id}")
    print(f"Center: ({args.lat}, {args.lon})")
    canvas_size_m = config["canvas"]["half_size_m"] * 2
    print(f"Canvas: ±{config['canvas']['half_size_m']}m ({canvas_size_m}m square)")
    print(f"Resolution: {config['canvas']['resolution_m']}m/px")
    print("=" * 60)

    # Create output directory
    output_dir = args.output_dir / args.site_id
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {}

    # Step 1: Create AEQD transformer
    print("\n[1/5] Creating AEQD projection...")
    transformer = AEQDTransformer(
        center_lat=args.lat,
        center_lon=args.lon,
        half_size_m=config["canvas"]["half_size_m"],
    )
    bbox_wgs84 = transformer.get_canvas_bounds_wgs84()
    print(f"  WGS84 bbox: {bbox_wgs84}")

    # Step 2: Fetch data from Overture Maps
    print("\n[2/5] Fetching data from Overture Maps...")
    with OvertureFetcher(
        bbox_wgs84=bbox_wgs84,
        road_width_fallback=config["road_width_fallback"],
    ) as fetcher:
        buildings_wgs84 = fetcher.fetch_buildings()
        roads_wgs84 = fetcher.fetch_roads()

        stats["buildings"] = {"count": len(buildings_wgs84)}
        stats["roads"] = fetcher.get_width_fallback_stats(roads_wgs84)

    # Step 3: Transform and clip
    print("\n[3/5] Transforming to AEQD and clipping...")
    buildings = transformer.transform_and_clip(buildings_wgs84)
    roads = transformer.transform_and_clip(roads_wgs84)
    print(f"  Buildings after clip: {len(buildings)}")
    print(f"  Roads after clip: {len(roads)}")

    # Save GeoJSON
    if not buildings.empty:
        buildings.to_file(output_dir / "buildings.geojson", driver="GeoJSON")
    if not roads.empty:
        roads.to_file(output_dir / "roads.geojson", driver="GeoJSON")

    # Step 4: Rasterize
    print("\n[4/5] Rasterizing...")
    canvas_size = int(config["canvas"]["half_size_m"] * 2 / config["canvas"]["resolution_m"])

    # Get rasterization quality settings with defaults
    raster_config = config.get("rasterization", {})
    supersample_factor = raster_config.get("supersample_factor", 1)
    interpolation = raster_config.get("interpolation", "bilinear")

    rasterizer = Rasterizer(
        canvas_size=canvas_size,
        resolution_m=config["canvas"]["resolution_m"],
        half_size_m=config["canvas"]["half_size_m"],
        supersample_factor=supersample_factor,
        interpolation=interpolation,
    )

    buildings_raster = rasterizer.rasterize_buildings(buildings)
    rasterizer.save(buildings_raster, output_dir / "buildings_binary.npy")

    roads_raster = rasterizer.rasterize_roads(roads)
    rasterizer.save(roads_raster, output_dir / "roads_weighted.npy")

    # Step 5: Build network
    if not args.skip_network and GRAPH_TOOL_AVAILABLE:
        print("\n[5/5] Building network graph...")
        builder = NetworkBuilder()
        graph = builder.build_network(roads, buildings)
        builder.save(graph, output_dir / "network.graphml")
        stats["network"] = builder.get_network_stats(graph)
    else:
        print("\n[5/5] Skipping network building")
        stats["network"] = {"status": "skipped"}

    # Save metadata
    print("\nSaving metadata...")
    save_metadata(
        output_dir=output_dir,
        site_id=args.site_id,
        center_lat=args.lat,
        center_lon=args.lon,
        transformer=transformer,
        config=config,
        stats=stats,
    )

    print("\n" + "=" * 60)
    print("Acquisition complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
