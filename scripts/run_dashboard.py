#!/usr/bin/env python
"""Phase 4: Launch visualization dashboard."""

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.unified_dashboard import create_unified_dashboard


def main():
    """Launch the Urban Structure Analysis dashboard."""
    parser = argparse.ArgumentParser(description="Launch Urban Structure Analysis Dashboard")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to results directory (or parent containing run directories)",
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
        help="Run in debug mode (auto-reload on code changes)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Urban Structure Analysis - Unified Dashboard")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 60)
    print("\nThis dashboard includes:")
    print("  📊 Indicator & Image Correspondence")
    print("  🔗 Cluster & Image Correspondence")
    print()

    # Create and run dashboard
    app = create_unified_dashboard(
        outputs_dir=args.results_dir,
    )

    print(f"Starting unified dashboard at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print()

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
