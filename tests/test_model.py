"""Tests for GeologicalModel data structure."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import unittest
import numpy as np
from tunnel_geology_model import GeologicalModel


def make_synthetic_model(nx=50, ny=30, nz=20, step=0.5):
    x = np.linspace(-100, -80, nx, dtype=np.float64)
    y = np.linspace(-10, 10, ny, dtype=np.float64)
    z = np.linspace(2900, 2910, nz, dtype=np.float64)
    vp = 5000 + 200 * np.random.default_rng(42).normal(size=(ny, nx, nz))
    vs = vp / 1.73
    return GeologicalModel(
        x=x, y=y, z=z,
        fields={"Vp": vp.astype(np.float32), "Vs": vs.astype(np.float32)},
        metadata={"grid_step_m": step, "source": "test"},
    )


class TestGeologicalModel(unittest.TestCase):
    def setUp(self):
        self.model = make_synthetic_model()

    def test_shape_and_bounds(self):
        self.assertEqual(self.model.nx, 50)
        self.assertEqual(self.model.ny, 30)
        self.assertEqual(self.model.nz, 20)
        self.assertEqual(self.model.shape, (30, 50, 20))
        self.assertAlmostEqual(self.model.grid_step, 0.5)
        self.assertAlmostEqual(self.model.x_range[0], -100.0)
        self.assertAlmostEqual(self.model.y_range[0], -10.0)

    def test_field_access(self):
        self.assertIn("Vp", self.model)
        self.assertIn("Vs", self.model)
        self.assertNotIn("Nonexistent", self.model)

        vp = self.model.field("Vp")
        self.assertEqual(vp.shape, (30, 50, 20))
        self.assertTrue(np.all(np.isfinite(vp)))

        with self.assertRaises(KeyError):
            self.model.field("FakeField")

    def test_getitem(self):
        vp = self.model["Vp"]
        self.assertEqual(vp.shape, (30, 50, 20))

    def test_stats(self):
        s = self.model.stats("Vp")
        self.assertIn("min", s)
        self.assertIn("max", s)
        self.assertIn("mean", s)
        self.assertIn("std", s)
        self.assertTrue(s["mean"] > 4000)

    def test_summary(self):
        text = self.model.summary()
        self.assertIn("GeologicalModel", text)
        self.assertIn("50 x 30 x 20", text)

    def test_subset(self):
        sub = self.model.subset_x(-95, -85)
        self.assertLess(sub.nx, self.model.nx)
        self.assertEqual(sub.ny, self.model.ny)
        self.assertEqual(sub.nz, self.model.nz)

    def test_subset_y(self):
        sub = self.model.subset_y(-5, 5)
        self.assertLess(sub.ny, self.model.ny)

    def test_subset_z(self):
        sub = self.model.subset_z(2902, 2907)
        self.assertLess(sub.nz, self.model.nz)

    def test_validation_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            GeologicalModel(
                x=np.linspace(0, 10, 10),
                y=np.linspace(0, 5, 5),
                z=np.linspace(0, 3, 3),
                fields={"Bad": np.zeros((4, 4, 4), dtype=np.float32)},
            )

    def test_validation_accepts_correct_shape(self):
        # Should not raise
        GeologicalModel(
            x=np.linspace(0, 10, 10),
            y=np.linspace(0, 5, 5),
            z=np.linspace(0, 3, 3),
            fields={"Ok": np.zeros((5, 10, 3), dtype=np.float32)},
        )

    def test_field_raises_keyerror_for_missing(self):
        with self.assertRaises(KeyError):
            self.model.field("NonexistentField")


if __name__ == "__main__":
    unittest.main()
