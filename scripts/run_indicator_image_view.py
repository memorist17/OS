#!/usr/bin/env python
"""Launch Indicator and Image Correspondence View Dashboard."""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.indicator_image_view import create_indicator_image_correspondence_view


def main():
    """Launch the Indicator and Image Correspondence View."""
    parser = argparse.ArgumentParser(
        description="Launch Indicator & Image Correspondence Dashboard"
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to outputs directory containing run directories",
    )
    parser.add_argument(
        "--indicator-type",
        type=str,
        choices=["mfa", "lacunarity", "percolation"],
        default="mfa",
        help="Type of indicator to display",
    )
    parser.add_argument(
        "--indicator-value",
        type=str,
        default="D(0)",
        help="Indicator value name (e.g., D(0) for MFA, beta for Lacunarity)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8051,
        help="Port to run dashboard on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (use 0.0.0.0 for external access)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to config file",
    )

    args = parser.parse_args()

    # Load config
    config = {}
    if args.config.exists():
        with open(args.config) as f:
            full_config = yaml.safe_load(f)
            config = full_config.get("visualization", {}).get("indicator_image_view", {})

    # Find all run directories
    if not args.outputs_dir.exists():
        print(f"Error: Outputs directory not found: {args.outputs_dir}")
        return 1

    run_dirs = [
        d
        for d in args.outputs_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]

    if not run_dirs:
        print(f"Error: No run directories found in {args.outputs_dir}")
        return 1

    # Sort by modification time (newest first)
    run_dirs = sorted(run_dirs, key=lambda x: x.stat().st_mtime, reverse=True)

    print("=" * 60)
    print("Indicator & Image Correspondence Dashboard")
    print("=" * 60)
    print(f"Outputs directory: {args.outputs_dir}")
    print(f"Indicator type: {args.indicator_type}")
    print(f"Indicator value: {args.indicator_value}")
    print(f"Number of runs: {len(run_dirs)}")
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 60)
    print("\nRun directories:")
    for i, run_dir in enumerate(run_dirs):
        print(f"  {i}: {run_dir.name}")
    print()

    # Create and run dashboard
    app = create_indicator_image_correspondence_view(
        results_dirs=run_dirs,
        indicator_type=args.indicator_type,
        indicator_value=args.indicator_value,
        config=config,
    )

    print(f"Starting dashboard at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

