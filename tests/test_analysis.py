"""Tests for analysis modules."""

import numpy as np
import pytest

from src.analysis.lacunarity import LacunarityAnalyzer
from src.analysis.multifractal import MultifractalAnalyzer


class TestMultifractalAnalyzer:
    """Test cases for MultifractalAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with small parameters for testing."""
        return MultifractalAnalyzer(
            r_min=2,
            r_max=64,
            r_steps=5,
            q_min=-5,
            q_max=5,
            q_steps=11,
            grid_shift_count=4,
        )

    def test_box_sizes_generation(self, analyzer):
        """Test that box sizes are generated correctly."""
        sizes = analyzer._get_box_sizes()

        assert sizes[0] >= 2
        assert sizes[-1] <= 64
        assert len(sizes) <= 5
        # Should be monotonically increasing
        assert all(sizes[i] < sizes[i + 1] for i in range(len(sizes) - 1))

    def test_q_values_generation(self, analyzer):
        """Test that q values are generated correctly."""
        q_values = analyzer._get_q_values()

        assert q_values[0] == -5
        assert q_values[-1] == 5
        assert len(q_values) == 11

    def test_uniform_image_analysis(self, analyzer):
        """Test analysis of uniform image."""
        # Uniform image should have narrow spectrum
        image = np.ones((128, 128), dtype=np.float64) * 100
        spectrum_df, mesh = analyzer.analyze(image, verbose=False)

        assert len(spectrum_df) == 11  # q_steps
        assert "alpha" in spectrum_df.columns
        assert "f_alpha" in spectrum_df.columns

    def test_random_image_analysis(self, analyzer):
        """Test analysis of random image."""
        np.random.seed(42)
        image = np.random.rand(128, 128) * 255

        spectrum_df, mesh = analyzer.analyze(image, verbose=False)

        # Should have valid spectrum
        assert not spectrum_df["alpha"].isna().all()
        assert spectrum_df["R2"].mean() > 0.5  # Reasonable fit

    def test_spectrum_width(self, analyzer):
        """Test spectrum width calculation."""
        np.random.seed(42)
        image = np.random.rand(128, 128) * 255
        spectrum_df, _ = analyzer.analyze(image, verbose=False)

        width = analyzer.get_spectrum_width(spectrum_df)
        assert width >= 0


class TestLacunarityAnalyzer:
    """Test cases for LacunarityAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with small parameters for testing."""
        return LacunarityAnalyzer(
            r_min=2,
            r_max=64,
            r_steps=5,
            full_scan=True,
        )

    def test_box_sizes_generation(self, analyzer):
        """Test that box sizes are generated correctly."""
        sizes = analyzer._get_box_sizes()

        assert sizes[0] >= 2
        assert sizes[-1] <= 64

    def test_empty_image_lacunarity(self, analyzer):
        """Test lacunarity of empty image."""
        image = np.zeros((128, 128), dtype=np.uint8)
        lacunarity_df, mesh = analyzer.analyze(image, verbose=False)

        # Empty image should have NaN lacunarity (division by zero)
        assert lacunarity_df["lambda"].isna().all() or (lacunarity_df["mu"] == 0).all()

    def test_full_image_lacunarity(self, analyzer):
        """Test lacunarity of fully filled image."""
        image = np.ones((128, 128), dtype=np.uint8)
        lacunarity_df, mesh = analyzer.analyze(image, verbose=False)

        # Full image should have lacunarity = 1 (no variance)
        assert all(lacunarity_df["lambda"] == 1.0)

    def test_random_binary_image(self, analyzer):
        """Test lacunarity of random binary image."""
        np.random.seed(42)
        image = (np.random.rand(128, 128) > 0.5).astype(np.uint8)

        lacunarity_df, mesh = analyzer.analyze(image, verbose=False)

        # Should have valid lacunarity values > 1
        valid = lacunarity_df[lacunarity_df["lambda"].notna()]
        assert len(valid) > 0
        assert valid["lambda"].min() >= 1.0

    def test_decay_exponent(self, analyzer):
        """Test decay exponent calculation."""
        np.random.seed(42)
        image = (np.random.rand(128, 128) > 0.3).astype(np.uint8)
        lacunarity_df, _ = analyzer.analyze(image, verbose=False)

        beta, r2 = analyzer.get_decay_exponent(lacunarity_df)

        # Should have valid decay exponent
        assert not np.isnan(beta)
        assert 0 <= r2 <= 1

    def test_lacunarity_at_scale(self, analyzer):
        """Test interpolation at specific scale."""
        np.random.seed(42)
        image = (np.random.rand(128, 128) > 0.5).astype(np.uint8)
        lacunarity_df, _ = analyzer.analyze(image, verbose=False)

        # Get lacunarity at a scale that exists
        r_existing = lacunarity_df["r"].iloc[0]
        lam = analyzer.get_lacunarity_at_scale(lacunarity_df, int(r_existing))

        assert lam == lacunarity_df.iloc[0]["lambda"]


class TestIntegration:
    """Integration tests for analysis pipeline."""

    def test_consistent_box_sizes(self):
        """Test that MFA and Lacunarity use consistent box sizes."""
        mfa = MultifractalAnalyzer(r_min=4, r_max=128, r_steps=10)
        lac = LacunarityAnalyzer(r_min=4, r_max=128, r_steps=10)

        mfa_sizes = mfa._get_box_sizes()
        lac_sizes = lac._get_box_sizes()

        # Should have same box sizes
        np.testing.assert_array_equal(mfa_sizes, lac_sizes)
