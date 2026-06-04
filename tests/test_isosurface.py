"""Tests for isosurface extraction."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import unittest
import numpy as np
from tunnel_geology_model.isosurface import marching_cubes_isosurface, multi_isosurface


class TestIsosurface(unittest.TestCase):
    def setUp(self):
        # Create a synthetic 3D scalar field — a sphere of value 1 inside, 0 outside
        ny, nx, nz = 40, 40, 40
        yy, xx, zz = np.meshgrid(
            np.linspace(-2, 2, ny),
            np.linspace(-2, 2, nx),
            np.linspace(-2, 2, nz),
            indexing="ij",
        )
        r = np.sqrt(xx**2 + yy**2 + zz**2)
        self.field = np.where(r < 1.0, 1.0, 0.0).astype(np.float32)

    def test_marching_cubes_returns_valid_mesh(self):
        verts, faces, normals, values = marching_cubes_isosurface(
            self.field, 0.5, spatial_step=0.1, origin=(-2.0, -2.0, -2.0)
        )
        self.assertGreater(len(verts), 0)
        self.assertGreater(len(faces), 0)
        self.assertEqual(verts.shape[1], 3)
        self.assertEqual(faces.shape[1], 3)
        self.assertEqual(normals.shape[1], 3)

    def test_multi_isosurface(self):
        results = multi_isosurface(self.field, [0.5], spatial_step=0.1)
        self.assertEqual(len(results), 1)
        self.assertGreater(len(results[0][0]), 0)


if __name__ == "__main__":
    unittest.main()
