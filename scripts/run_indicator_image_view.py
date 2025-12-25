#!/usr/bin/env python
"""Launch Indicator and Image Correspondence View Dashboard."""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.unified_dashboard import create_unified_dashboard


def main():
    """Launch the Unified Dashboard (replaces individual view)."""
    parser = argparse.ArgumentParser(
        description="Launch Unified Dashboard (Indicator & Image + Cluster & Image)"
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to outputs directory containing run directories",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
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
    config = None
    if args.config.exists():
        with open(args.config) as f:
            full_config = yaml.safe_load(f)
            config = full_config.get("visualization", {})

    print("=" * 60)
    print("Urban Structure Analysis - Unified Dashboard")
    print("=" * 60)
    print(f"Outputs directory: {args.outputs_dir}")
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 60)
    print("\nThis dashboard includes:")
    print("  📊 Indicator & Image Correspondence")
    print("  🔗 Cluster & Image Correspondence")
    print()

    # Create and run dashboard
    app = create_unified_dashboard(
        outputs_dir=args.outputs_dir,
        config=config,
    )

    print(f"Starting unified dashboard at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

