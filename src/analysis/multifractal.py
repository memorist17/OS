"""Multifractal Analysis (MFA) using Reshape & Sum with Grid Shifting."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from tqdm import tqdm


@dataclass
class MultifractalAnalyzer:
    """
    Compute multifractal spectrum using 4D reshape & sum.

    Algorithm:
    1. Reshape image (H, W) to (H//r, r, W//r, r)
    2. Sum over axis=(1,3) to get box masses
    3. Apply grid shifting for robustness
    4. Linear regression of log(Z(q,r)) vs log(r) to get τ(q)
    5. Legendre transform: α = dτ/dq, f(α) = qα - τ
    """

    r_min: int = 2
    r_max: int = 512
    r_steps: int = 20
    q_min: float = -10
    q_max: float = 10
    q_steps: int = 41
    grid_shift_count: int = 16
    n_jobs: int = -1

    def analyze(self, image: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Compute multifractal spectrum.

        Args:
            image: 2D weighted raster array (non-negative values)

        Returns:
            spectrum_df: DataFrame with columns [q, alpha, f_alpha, tau, R2]
            mesh: 2D array of partition function Z(q, r) shape (q_steps, r_steps)
        """
        # Normalize image to probabilities
        image = image.astype(np.float64)
        total = image.sum()
        if total == 0:
            raise ValueError("Image is empty (all zeros)")
        image = image / total

        # Get box sizes and q values
        box_sizes = self._get_box_sizes()
        q_values = np.linspace(self.q_min, self.q_max, self.q_steps)

        # Filter box sizes that fit in image
        H, W = image.shape
        valid_sizes = [r for r in box_sizes if r <= min(H, W) // 2]
        if not valid_sizes:
            raise ValueError(f"No valid box sizes for image of shape {image.shape}")

        box_sizes = np.array(valid_sizes)
        r_steps_actual = len(box_sizes)

        print(f"Computing MFA: {len(q_values)} q-values, {r_steps_actual} box sizes")
        print(f"Box sizes: {box_sizes[0]} to {box_sizes[-1]}")

        # Compute partition function for all (q, r) combinations
        mesh = np.zeros((len(q_values), r_steps_actual))

        # Parallel computation over q values
        def compute_q_row(q_idx: int) -> np.ndarray:
            q = q_values[q_idx]
            row = np.zeros(r_steps_actual)
            for r_idx, r in enumerate(box_sizes):
                row[r_idx] = self._compute_partition_function_shifted(image, r, q)
            return row

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(compute_q_row)(q_idx)
            for q_idx in tqdm(range(len(q_values)), desc="Computing Z(q,r)")
        )

        for q_idx, row in enumerate(results):
            mesh[q_idx, :] = row

        # Compute τ(q) via linear regression of log(Z) vs log(r)
        log_r = np.log(box_sizes)
        tau_values = np.zeros(len(q_values))
        r2_values = np.zeros(len(q_values))

        for q_idx in range(len(q_values)):
            log_Z = np.log(mesh[q_idx, :] + 1e-300)  # Avoid log(0)
            # τ(q) is the slope of log(Z) vs log(r)
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_r, log_Z)
            tau_values[q_idx] = slope
            r2_values[q_idx] = r_value ** 2

        # Compute α(q) = dτ/dq via numerical differentiation
        alpha_values = np.gradient(tau_values, q_values)

        # Compute f(α) = qα - τ via Legendre transform
        f_alpha_values = q_values * alpha_values - tau_values

        # Create result DataFrame
        spectrum_df = pd.DataFrame({
            "q": q_values,
            "alpha": alpha_values,
            "f_alpha": f_alpha_values,
            "tau": tau_values,
            "R2": r2_values,
        })

        return spectrum_df, mesh

    def _compute_partition_function_shifted(
        self, image: np.ndarray, r: int, q: float
    ) -> float:
        """Compute partition function with grid shift averaging."""
        H, W = image.shape
        max_shift = min(r, self.grid_shift_count)
        shifts = np.linspace(0, r - 1, max_shift, dtype=int)

        Z_sum = 0.0
        count = 0

        for dx in shifts:
            for dy in shifts:
                Z = self._compute_partition_function(image, r, q, dx, dy)
                if Z > 0:
                    Z_sum += Z
                    count += 1

        return Z_sum / count if count > 0 else 0.0

    def _compute_partition_function(
        self, image: np.ndarray, r: int, q: float, dx: int = 0, dy: int = 0
    ) -> float:
        """
        Compute partition function Z(q, r) for given box size and moment.

        Uses 4D reshape trick for efficient box counting.
        """
        H, W = image.shape

        # Apply offset and crop to fit boxes exactly
        h_boxes = (H - dy) // r
        w_boxes = (W - dx) // r

        if h_boxes == 0 or w_boxes == 0:
            return 0.0

        # Crop image
        cropped = image[dy:dy + h_boxes * r, dx:dx + w_boxes * r]

        # Reshape to (h_boxes, r, w_boxes, r) and sum over box dimensions
        reshaped = cropped.reshape(h_boxes, r, w_boxes, r)
        box_masses = reshaped.sum(axis=(1, 3))

        # Filter out zero-mass boxes
        nonzero_masses = box_masses[box_masses > 0]

        if len(nonzero_masses) == 0:
            return 0.0

        # Compute Z(q, r) = sum(p^q)
        if q == 1:
            # Special case: use entropy formula to avoid numerical issues
            Z = np.exp(-np.sum(nonzero_masses * np.log(nonzero_masses)))
        else:
            Z = np.sum(nonzero_masses ** q)

        return Z

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

    def compute_generalized_dimensions(
        self, spectrum_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute generalized dimensions D(q) from τ(q).

        D(q) = τ(q) / (q - 1) for q ≠ 1
        D(1) = limit as q→1 (information dimension)

        Returns:
            DataFrame with columns [q, D_q]
        """
        q = spectrum_df["q"].values
        tau = spectrum_df["tau"].values

        D_q = np.zeros_like(q)

        for i, (qi, ti) in enumerate(zip(q, tau)):
            if abs(qi - 1) < 0.01:
                # Use derivative at q=1
                idx = np.argmin(np.abs(q - 1))
                D_q[i] = np.gradient(tau, q)[idx]
            else:
                D_q[i] = ti / (qi - 1)

        return pd.DataFrame({"q": q, "D_q": D_q})
