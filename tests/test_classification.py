"""Tests for rock mass classification."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import unittest
import numpy as np
from tunnel_geology_model.classification import (
    classify_bq,
    classify_rmr,
    geohazard_index,
    bq_class_label,
)


class TestBQClassification(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        # Vp from 2000 to 6500 m/s
        self.vp = 3000 + 3500 * rng.random((20, 30, 15)).astype(np.float32)

    def test_bq_returns_correct_shapes(self):
        bq, cls = classify_bq(self.vp)
        self.assertEqual(bq.shape, self.vp.shape)
        self.assertEqual(cls.shape, self.vp.shape)

    def test_bq_class_in_valid_range(self):
        _, cls = classify_bq(self.vp)
        self.assertTrue(np.all((cls >= 1) & (cls <= 5)))

    def test_high_vp_yields_good_rock(self):
        high_vp = np.full((5, 5, 5), 6000.0, dtype=np.float32)
        bq, cls = classify_bq(high_vp)
        self.assertTrue(np.all(cls <= 3))
        self.assertTrue(np.all(bq > 300))

    def test_low_vp_yields_poor_rock(self):
        low_vp = np.full((5, 5, 5), 1500.0, dtype=np.float32)
        bq, cls = classify_bq(low_vp)
        self.assertTrue(np.all(cls >= 4))
        self.assertTrue(np.all(bq < 400))

    def test_class_labels(self):
        self.assertEqual(bq_class_label(1), "I - Very Good")
        self.assertEqual(bq_class_label(5), "V - Very Poor")
        self.assertEqual(bq_class_label(99), "Unknown")


class TestRMRClassification(unittest.TestCase):
    def test_rmr_shape_and_range(self):
        vp = np.full((10, 10, 10), 5000.0, dtype=np.float32)
        rmr = classify_rmr(vp)
        self.assertEqual(rmr.shape, vp.shape)
        self.assertTrue(np.all((rmr >= 0) & (rmr <= 100)))


class TestGeohazardIndex(unittest.TestCase):
    def test_geohazard_range(self):
        rng = np.random.default_rng(42)
        poisson = 0.25 + 0.1 * rng.random((8, 8, 8)).astype(np.float32)
        young = 60 + 20 * rng.random((8, 8, 8)).astype(np.float32)
        density = 2700 + 100 * rng.random((8, 8, 8)).astype(np.float32)

        idx = geohazard_index(poisson, young, density)
        self.assertEqual(idx.shape, poisson.shape)
        self.assertTrue(np.all((idx >= 0) & (idx <= 1)))


if __name__ == "__main__":
    unittest.main()
