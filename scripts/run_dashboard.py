#!/usr/bin/env python
"""Phase 4: Launch visualization dashboard."""

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.dashboard import create_dashboard
from src.visualization.comparison_dashboard import create_comparison_dashboard

# Try to import gallery dashboard if available
try:
    from src.visualization.gallery_dashboard import create_gallery_dashboard
    GALLERY_AVAILABLE = True
except ImportError:
    GALLERY_AVAILABLE = False


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
        "--run-id",
        type=str,
        default=None,
        help="Specific run ID to display (uses latest if not specified)",
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
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Show comparison view for all places",
    )
    parser.add_argument(
        "--gallery",
        action="store_true",
        help="Show gallery view with structure visualizations (if available)",
    )

    args = parser.parse_args()

    # Determine results directory
    results_dir = args.results_dir

    # Check for gallery mode first
    if args.gallery and GALLERY_AVAILABLE:
        print("=" * 60)
        print("Urban Structure Analysis Dashboard - Gallery Mode")
        print("=" * 60)
        print(f"Results directory: {results_dir}")
        print(f"Server: http://{args.host}:{args.port}")
        print("=" * 60)
        
        app = create_gallery_dashboard(results_dir)
        print(f"Starting gallery dashboard at http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop")
        print()
        
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
        )
        return

    # Check if comparison mode or multiple runs available
    use_comparison = args.comparison
    if not use_comparison and results_dir.is_dir():
        run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        if len(run_dirs) > 1:
            use_comparison = True
            print(f"Multiple runs detected ({len(run_dirs)}), using comparison mode")
            print("Use --comparison to explicitly enable, or specify --run-id for single view")
            if GALLERY_AVAILABLE:
                print("Use --gallery for gallery view with structure visualizations")

    if use_comparison:
        # Comparison mode: show all places
        print("=" * 60)
        print("Urban Structure Analysis Dashboard - Comparison Mode")
        print("=" * 60)
        print(f"Results directory: {results_dir}")
        print(f"Server: http://{args.host}:{args.port}")
        print("=" * 60)
        
        app = create_comparison_dashboard(results_dir)
    else:
        # Single place mode
        if args.run_id:
            # Use specific run
            target_dir = results_dir / args.run_id
            if not target_dir.exists():
                print(f"Error: Run directory not found: {target_dir}")
                return 1
        elif results_dir.is_dir():
            # Find latest run directory
            subdirs = [d for d in results_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if subdirs:
                target_dir = max(subdirs, key=lambda x: x.stat().st_mtime)
            else:
                target_dir = results_dir
        else:
            target_dir = results_dir

        print("=" * 60)
        print("Urban Structure Analysis Dashboard")
        print("=" * 60)
        print(f"Results directory: {target_dir}")
        print(f"Server: http://{args.host}:{args.port}")
        print("=" * 60)

        # List available files
        if target_dir.exists():
            print("Available files:")
            for f in sorted(target_dir.iterdir()):
                if f.is_file():
                    size_kb = f.stat().st_size / 1024
                    print(f"  - {f.name} ({size_kb:.1f} KB)")
            print()

        # Create and run dashboard
        app = create_dashboard(target_dir)

    print(f"Starting dashboard at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print()

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
