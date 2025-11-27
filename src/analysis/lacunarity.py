"""Lacunarity Analysis using Integral Image (Summed Area Table)."""

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


@dataclass
class LacunarityAnalyzer:
    """
    Compute lacunarity using integral image for O(1) box queries.

    Algorithm:
    1. Compute integral image (cv2.integral)
    2. For each box size r, compute mass at all positions in O(1)
    3. Calculate Λ(r) = (σ/μ)² + 1

    Lacunarity measures the "gappiness" or heterogeneity of a pattern.
    Higher lacunarity = more gaps/heterogeneous structure.
    """

    r_min: int = 2
    r_max: int = 512
    r_steps: int = 20
    full_scan: bool = True  # Scan all positions vs. sampling
    sample_fraction: float = 0.1  # If not full_scan
    cache_integral: bool = True
    n_jobs: int = -1

    def analyze(self, image: np.ndarray) -> tuple[pd.DataFrame, np.ndarray | None]:
        """
        Compute lacunarity curve.

        Args:
            image: 2D binary raster array (0/1 or any non-negative values)

        Returns:
            lacunarity_df: DataFrame with columns [r, lambda, sigma, mu, cv]
            mesh: 3D array of local lacunarity (r_steps, H, W) or None if not computed
        """
        image = image.astype(np.float64)
        H, W = image.shape

        if image.sum() == 0:
            raise ValueError("Image is empty (all zeros)")

        # Compute integral image once
        integral = self._compute_integral_image(image)

        # Get box sizes
        box_sizes = self._get_box_sizes()
        valid_sizes = [r for r in box_sizes if r <= min(H, W)]
        if not valid_sizes:
            raise ValueError(f"No valid box sizes for image of shape {image.shape}")

        box_sizes = np.array(valid_sizes)

        print(f"Computing Lacunarity: {len(box_sizes)} box sizes")
        print(f"Box sizes: {box_sizes[0]} to {box_sizes[-1]}")

        # Compute lacunarity for each box size
        results = []

        for r in tqdm(box_sizes, desc="Computing Λ(r)"):
            stats = self._compute_lacunarity_for_size(integral, r, H, W)
            results.append({
                "r": r,
                "lambda": stats["lambda"],
                "sigma": stats["sigma"],
                "mu": stats["mu"],
                "cv": stats["cv"],  # Coefficient of variation
            })

        lacunarity_df = pd.DataFrame(results)

        # Note: Local lacunarity mesh is memory-intensive for large images
        # Return None for now; can be computed on demand
        mesh = None

        return lacunarity_df, mesh

    def _compute_integral_image(self, image: np.ndarray) -> np.ndarray:
        """
        Compute summed area table (integral image).

        Returns:
            Integral image with shape (H+1, W+1)
        """
        return cv2.integral(image)

    def _box_sum(
        self, integral: np.ndarray, x: int, y: int, r: int
    ) -> float:
        """
        Compute box sum using integral image in O(1).

        Args:
            integral: Integral image (H+1, W+1)
            x, y: Top-left corner of box
            r: Box size

        Returns:
            Sum of values in box
        """
        # Integral image has offset of 1
        return (
            integral[y + r, x + r]
            - integral[y + r, x]
            - integral[y, x + r]
            + integral[y, x]
        )

    def _compute_lacunarity_for_size(
        self, integral: np.ndarray, r: int, H: int, W: int
    ) -> dict:
        """
        Compute lacunarity statistics for a given box size.

        Args:
            integral: Integral image
            r: Box size
            H, W: Image dimensions

        Returns:
            Dictionary with lambda, sigma, mu, cv
        """
        # Number of valid box positions
        n_positions_y = H - r + 1
        n_positions_x = W - r + 1

        if n_positions_x <= 0 or n_positions_y <= 0:
            return {"lambda": 1.0, "sigma": 0.0, "mu": 0.0, "cv": 0.0}

        if self.full_scan:
            # Scan all positions
            box_sums = np.zeros((n_positions_y, n_positions_x))
            for y in range(n_positions_y):
                for x in range(n_positions_x):
                    box_sums[y, x] = self._box_sum(integral, x, y, r)
        else:
            # Sample positions
            n_samples = max(100, int(n_positions_x * n_positions_y * self.sample_fraction))
            xs = np.random.randint(0, n_positions_x, n_samples)
            ys = np.random.randint(0, n_positions_y, n_samples)
            box_sums = np.array([
                self._box_sum(integral, x, y, r)
                for x, y in zip(xs, ys)
            ])

        # Compute statistics
        mu = np.mean(box_sums)
        sigma = np.std(box_sums)

        if mu == 0:
            return {"lambda": 1.0, "sigma": sigma, "mu": mu, "cv": 0.0}

        cv = sigma / mu  # Coefficient of variation
        lacunarity = cv ** 2 + 1  # Λ = (σ/μ)² + 1

        return {
            "lambda": lacunarity,
            "sigma": sigma,
            "mu": mu,
            "cv": cv,
        }

    def _get_box_sizes(self) -> np.ndarray:
        """Generate log-spaced box sizes."""
        return np.unique(
            np.logspace(
                np.log2(self.r_min),
                np.log2(self.r_max),
                self.r_steps,
                base=2
            ).astype(int)
        )

    def compute_local_lacunarity(
        self, image: np.ndarray, r: int
    ) -> np.ndarray:
        """
        Compute local lacunarity map for a specific box size.

        Args:
            image: 2D binary image
            r: Box size

        Returns:
            2D array of local lacunarity values
        """
        image = image.astype(np.float64)
        H, W = image.shape

        integral = self._compute_integral_image(image)

        n_positions_y = H - r + 1
        n_positions_x = W - r + 1

        local_lac = np.zeros((n_positions_y, n_positions_x))

        # Use sliding window statistics
        # For each position, compute local variance in a neighborhood
        neighborhood_size = max(3, r // 4)

        for y in range(n_positions_y):
            for x in range(n_positions_x):
                # Get box sums in neighborhood
                y_start = max(0, y - neighborhood_size)
                y_end = min(n_positions_y, y + neighborhood_size + 1)
                x_start = max(0, x - neighborhood_size)
                x_end = min(n_positions_x, x + neighborhood_size + 1)

                local_sums = []
                for yy in range(y_start, y_end):
                    for xx in range(x_start, x_end):
                        local_sums.append(self._box_sum(integral, xx, yy, r))

                local_sums = np.array(local_sums)
                mu = np.mean(local_sums)
                if mu > 0:
                    cv = np.std(local_sums) / mu
                    local_lac[y, x] = cv ** 2 + 1
                else:
                    local_lac[y, x] = 1.0

        return local_lac

    def fit_power_law(self, lacunarity_df: pd.DataFrame) -> dict:
        """
        Fit power law Λ(r) ∝ r^(-β) to lacunarity curve.

        Returns:
            Dictionary with beta (slope), intercept, R2
        """
        from scipy import stats

        log_r = np.log(lacunarity_df["r"].values)
        log_lambda = np.log(lacunarity_df["lambda"].values)

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_r, log_lambda)

        return {
            "beta": -slope,  # Negative because Λ decreases with r
            "intercept": np.exp(intercept),
            "R2": r_value ** 2,
        }
