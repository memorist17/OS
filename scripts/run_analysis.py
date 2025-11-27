#!/usr/bin/env python
"""Phase 3: Urban structure analysis pipeline."""

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.multifractal import MultifractalAnalyzer
from src.analysis.lacunarity import LacunarityAnalyzer
from src.analysis.percolation import PercolationAnalyzer


def main():
    """Run urban structure analysis pipeline."""
    parser = argparse.ArgumentParser(description="Analyze urban structure data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to site data directory (containing .npy files)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Config file path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (auto-generated if not provided)",
    )
    parser.add_argument(
        "--skip-mfa",
        action="store_true",
        help="Skip Multifractal Analysis",
    )
    parser.add_argument(
        "--skip-lacunarity",
        action="store_true",
        help="Skip Lacunarity Analysis",
    )
    parser.add_argument(
        "--skip-percolation",
        action="store_true",
        help="Skip Percolation Analysis",
    )

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    analysis_config = config["analysis"]
    execution_config = config["execution"]

    # Generate run ID
    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

    # Create output directory
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load site metadata
    metadata_path = args.data_dir / "metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path) as f:
            site_metadata = yaml.safe_load(f)
    else:
        site_metadata = {"site_id": args.data_dir.name}

    print("=" * 60)
    print("Urban Structure Analysis Pipeline")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Run ID: {run_id}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Save config snapshot
    config_snapshot = {
        "run_id": run_id,
        "data_dir": str(args.data_dir),
        "site_metadata": site_metadata,
        "analysis_config": analysis_config,
        "execution_config": execution_config,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(config_snapshot, f, default_flow_style=False, allow_unicode=True)

    # Load raster data
    buildings_path = args.data_dir / "buildings_binary.npy"
    roads_path = args.data_dir / "roads_weighted.npy"

    if buildings_path.exists():
        buildings_raster = np.load(buildings_path)
        print(f"Loaded buildings raster: {buildings_raster.shape}")
    else:
        buildings_raster = None
        print("Buildings raster not found")

    if roads_path.exists():
        roads_raster = np.load(roads_path)
        print(f"Loaded roads raster: {roads_raster.shape}")
    else:
        roads_raster = None
        print("Roads raster not found")

    # Use combined raster for analysis
    if buildings_raster is not None and roads_raster is not None:
        # Combine buildings (binary) and roads (weighted)
        analysis_raster = np.maximum(buildings_raster, (roads_raster > 0).astype(np.uint8))
    elif buildings_raster is not None:
        analysis_raster = buildings_raster
    elif roads_raster is not None:
        analysis_raster = (roads_raster > 0).astype(np.uint8)
    else:
        raise ValueError("No raster data found")

    print(f"Analysis raster shape: {analysis_raster.shape}")
    print(f"Coverage: {(analysis_raster > 0).sum() / analysis_raster.size * 100:.2f}%")

    # =========================================================================
    # Multifractal Analysis
    # =========================================================================
    if not args.skip_mfa:
        print("\n" + "=" * 60)
        print("[1/3] Multifractal Analysis (MFA)")
        print("=" * 60)

        mfa_analyzer = MultifractalAnalyzer(
            r_min=analysis_config["r_min"],
            r_max=analysis_config["r_max"],
            r_steps=analysis_config["r_steps"],
            q_min=analysis_config["mfa"]["q_min"],
            q_max=analysis_config["mfa"]["q_max"],
            q_steps=analysis_config["mfa"]["q_steps"],
            grid_shift_count=analysis_config["mfa"]["grid_shift_count"],
            n_jobs=execution_config["n_jobs"],
        )

        try:
            mfa_spectrum, mfa_mesh = mfa_analyzer.analyze(analysis_raster)

            # Save results
            mfa_spectrum.to_csv(output_dir / "mfa_spectrum.csv", index=False)
            np.save(output_dir / "mfa_mesh.npy", mfa_mesh)

            # Compute generalized dimensions
            D_q = mfa_analyzer.compute_generalized_dimensions(mfa_spectrum)
            D_q.to_csv(output_dir / "mfa_dimensions.csv", index=False)

            print(f"MFA complete: {len(mfa_spectrum)} q-values")
            print(f"  D(0) = {D_q[D_q['q'] == 0]['D_q'].values[0]:.4f} (capacity dimension)")
            print(f"  D(1) ≈ {D_q[abs(D_q['q'] - 1) < 0.5]['D_q'].values[0]:.4f} (information dimension)")
            print(f"  D(2) = {D_q[D_q['q'] == 2]['D_q'].values[0]:.4f} (correlation dimension)")

        except Exception as e:
            print(f"MFA failed: {e}")

    # =========================================================================
    # Lacunarity Analysis
    # =========================================================================
    if not args.skip_lacunarity:
        print("\n" + "=" * 60)
        print("[2/3] Lacunarity Analysis")
        print("=" * 60)

        lac_analyzer = LacunarityAnalyzer(
            r_min=analysis_config["r_min"],
            r_max=analysis_config["r_max"],
            r_steps=analysis_config["r_steps"],
            full_scan=analysis_config["lacunarity"]["full_scan"],
            cache_integral=execution_config["cache_integral"],
            n_jobs=execution_config["n_jobs"],
        )

        try:
            lac_curve, lac_mesh = lac_analyzer.analyze(analysis_raster)

            # Save results
            lac_curve.to_csv(output_dir / "lacunarity.csv", index=False)
            if lac_mesh is not None:
                np.save(output_dir / "lacunarity_mesh.npy", lac_mesh)

            # Fit power law
            power_law = lac_analyzer.fit_power_law(lac_curve)

            print(f"Lacunarity complete: {len(lac_curve)} box sizes")
            print(f"  Λ(r_min) = {lac_curve['lambda'].iloc[0]:.4f}")
            print(f"  Λ(r_max) = {lac_curve['lambda'].iloc[-1]:.4f}")
            print(f"  Power law β = {power_law['beta']:.4f} (R² = {power_law['R2']:.4f})")

            # Save power law fit
            with open(output_dir / "lacunarity_fit.yaml", "w") as f:
                yaml.dump(power_law, f)

        except Exception as e:
            print(f"Lacunarity failed: {e}")

    # =========================================================================
    # Percolation Analysis
    # =========================================================================
    if not args.skip_percolation:
        print("\n" + "=" * 60)
        print("[3/3] Percolation Analysis")
        print("=" * 60)

        network_path = args.data_dir / "network.graphml"
        if not network_path.exists():
            print("Network graph not found, skipping percolation analysis")
        else:
            perc_analyzer = PercolationAnalyzer(
                d_min=analysis_config["percolation"]["d_min"],
                d_max=analysis_config["percolation"]["d_max"],
                d_steps=analysis_config["percolation"]["d_steps"],
            )

            try:
                perc_curve, perc_stats = perc_analyzer.analyze_with_statistics(network_path)
                perc_df, perc_mesh = perc_analyzer.analyze(network_path)

                # Save results
                perc_df.to_csv(output_dir / "percolation.csv", index=False)
                np.save(output_dir / "percolation_mesh.npy", perc_mesh)

                with open(output_dir / "percolation_stats.yaml", "w") as f:
                    yaml.dump(perc_stats, f)

                print(f"Percolation complete: {len(perc_df)} thresholds")
                print(f"  Critical d (50%): {perc_stats['d_critical_50']:.2f}")
                print(f"  Transition width: {perc_stats['transition_width']:.2f}")
                print(f"  Max clusters: {perc_stats['max_clusters']}")

            except Exception as e:
                print(f"Percolation failed: {e}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print("Files created:")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
