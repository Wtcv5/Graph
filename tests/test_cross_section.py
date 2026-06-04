"""Tests for cross-section extraction."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import unittest
import numpy as np
from tunnel_geology_model import GeologicalModel
from tunnel_geology_model.cross_section import (
    extract_yz_section,
    extract_xz_section,
    extract_xy_section,
    extract_arbitrary_section,
)


class TestCrossSections(unittest.TestCase):
    def setUp(self):
        nx, ny, nz = 40, 25, 15
        self.model = GeologicalModel(
            x=np.linspace(-100, -80, nx, dtype=np.float64),
            y=np.linspace(-10, 10, ny, dtype=np.float64),
            z=np.linspace(2900, 2910, nz, dtype=np.float64),
            fields={
                "Vp": 5000.0 * np.ones((ny, nx, nz), dtype=np.float32),
                "Vs": 2900.0 * np.ones((ny, nx, nz), dtype=np.float32),
            },
            metadata={"grid_step_m": 0.5},
        )

    def test_yz_section(self):
        result = extract_yz_section(self.model, -90.0)
        self.assertIn("Vp", result)
        self.assertIn("Vs", result)
        self.assertEqual(result["Vp"].shape, (25, 15))  # (ny, nz)

    def test_xz_section(self):
        result = extract_xz_section(self.model, 0.0)
        self.assertEqual(result["Vp"].shape, (40, 15))  # (nx, nz)

    def test_xy_section(self):
        result = extract_xy_section(self.model, 2905.0)
        self.assertEqual(result["Vp"].shape, (40, 25))  # (nx, ny) — transposed

    def test_section_preserves_values(self):
        result = extract_xz_section(self.model, 0.0)
        self.assertTrue(np.allclose(result["Vp"], 5000.0, atol=1))

    def test_selective_fields(self):
        result = extract_xz_section(self.model, 0.0, field_names=["Vp"])
        self.assertIn("Vp", result)
        self.assertNotIn("Vs", result)

    def test_arbitrary_section_shape(self):
        result = extract_arbitrary_section(
            self.model,
            origin=(-90.0, 0.0, 2905.0),
            normal=(0.0, 1.0, 0.0),  # Y-normal → X-Z plane
        )
        self.assertIn("Vp", result)
        self.assertIn("plane_u", result)
        self.assertGreater(result["Vp"].size, 0)


if __name__ == "__main__":
    unittest.main()
