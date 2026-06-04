"""Tests for export functions."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import unittest
import tempfile
import numpy as np
from tunnel_geology_model import GeologicalModel
from tunnel_geology_model.export import (
    to_vtk_structured_grid,
    to_numpy_dict,
    to_xyz_point_cloud,
)


class TestExport(unittest.TestCase):
    def setUp(self):
        nx, ny, nz = 10, 8, 6
        self.model = GeologicalModel(
            x=np.linspace(-50, -45, nx, dtype=np.float64),
            y=np.linspace(-3, 3, ny, dtype=np.float64),
            z=np.linspace(2975, 2980, nz, dtype=np.float64),
            fields={
                "Vp": 5400.0 * np.ones((ny, nx, nz), dtype=np.float32),
                "Vs": 3100.0 * np.ones((ny, nx, nz), dtype=np.float32),
            },
            metadata={"grid_step_m": 0.5},
        )

    def test_vtk_export(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "test.vts"
            path = to_vtk_structured_grid(self.model, out)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)

    def test_vtk_export_selective_fields(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "test.vts"
            path = to_vtk_structured_grid(self.model, out, fields=["Vp"])
            self.assertTrue(path.exists())

    def test_numpy_dict_export(self):
        with tempfile.TemporaryDirectory() as td:
            path = to_numpy_dict(self.model, td)
            self.assertTrue(path.exists())
            data = np.load(str(path))
            try:
                self.assertIn("x", data.files)
                self.assertIn("field_Vp", data.files)
            finally:
                data.close()

    def test_xyz_export(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "points.csv"
            path = to_xyz_point_cloud(self.model, out, sample=0.5)
            self.assertTrue(path.exists())
            lines = path.read_text().splitlines()
            self.assertGreater(len(lines), 1)  # header + data


if __name__ == "__main__":
    unittest.main()
