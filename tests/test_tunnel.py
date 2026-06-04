"""Tests for parametric tunnel geometry and coupling."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import unittest
import numpy as np
from tunnel_geology_model.tunnel import (
    ParametricTunnel, TunnelAlignment, TunnelSection, SectionShape,
)
from tunnel_geology_model import GeologicalModel


class TestTunnelSection(unittest.TestCase):
    def test_horseshoe_profile(self):
        sect = TunnelSection(shape=SectionShape.HORSESHOE, width=12, height=10, n_vertices=64)
        profile = sect.generate_profile()[0]
        self.assertEqual(profile.shape, (65, 2))  # 64 + closing vertex
        self.assertTrue(np.allclose(profile[0], profile[-1]))  # closed

    def test_circular_profile(self):
        sect = TunnelSection(shape=SectionShape.CIRCULAR, width=10, height=10, n_vertices=48)
        profile = sect.generate_profile()[0]
        self.assertEqual(profile.shape, (49, 2))

    def test_rectangular_profile(self):
        sect = TunnelSection(shape=SectionShape.RECTANGULAR, width=8, height=6)
        profile = sect.generate_profile()[0]
        self.assertGreater(profile[:, 1].max(), 5)  # height ~6

    def test_layers(self):
        sect = TunnelSection(shape=SectionShape.CIRCULAR, width=10, height=10,
                             shotcrete_thickness=0.15, lining_thickness=0.35)
        profiles = sect.generate_profile(include_layers=True)
        self.assertEqual(len(profiles), 3)


class TestTunnelAlignment(unittest.TestCase):
    def test_from_endpoints(self):
        al = TunnelAlignment.from_endpoints(
            x_start=-100, y_start=0, z_start=3000,
            x_end=-20, y_end=0, z_end=3005,
            gradient_pct=1.0, n_samples=50,
        )
        ch, cx, cy, cz = al.sample()
        self.assertEqual(len(ch), 50)
        self.assertAlmostEqual(al.total_length, 80.0, delta=1)

    def test_tangent(self):
        al = TunnelAlignment.from_endpoints(0, 0, 0, 100, 0, 0)
        t = al.tangent_at(50)
        self.assertAlmostEqual(float(t[0]), 1.0, delta=0.1)


class TestParametricTunnel(unittest.TestCase):
    def setUp(self):
        self.alignment = TunnelAlignment.from_endpoints(
            -100, 0, 3000, -20, 0, 3005, n_samples=20,
        )
        self.section = TunnelSection(
            shape=SectionShape.CIRCULAR, width=10, height=10, n_vertices=32,
        )

    def test_build_mesh(self):
        tunnel = ParametricTunnel(alignment=self.alignment, section=self.section)
        tunnel.build_mesh()
        self.assertIsNotNone(tunnel.vertices)
        self.assertIsNotNone(tunnel.faces)
        self.assertGreater(tunnel.n_vertices, 100)
        self.assertGreater(tunnel.n_faces, 100)

    def test_summary(self):
        tunnel = ParametricTunnel(alignment=self.alignment, section=self.section)
        tunnel.build_mesh()
        s = tunnel.summary()
        self.assertIn("ParametricTunnel", s)


class TestTunnelCoupling(unittest.TestCase):
    def setUp(self):
        # Create a simple geological model
        nx, ny, nz = 40, 20, 15
        self.model = GeologicalModel(
            x=np.linspace(-100, -20, nx, dtype=np.float64),
            y=np.linspace(-15, 15, ny, dtype=np.float64),
            z=np.linspace(2990, 3010, nz, dtype=np.float64),
            fields={
                "Vp": 5500.0 * np.ones((ny, nx, nz), dtype=np.float32),
                "Vs": 3100.0 * np.ones((ny, nx, nz), dtype=np.float32),
                "Density": 2700.0 * np.ones((ny, nx, nz), dtype=np.float32),
                "Young_Modulus": 65.0 * np.ones((ny, nx, nz), dtype=np.float32),
            },
            metadata={"grid_step_m": 2.0},
        )

        alignment = TunnelAlignment.from_endpoints(
            -95, 0, 3000, -25, 0, 3003, n_samples=15,
        )
        section = TunnelSection(
            shape=SectionShape.CIRCULAR, width=8, height=8, n_vertices=24,
        )
        self.tunnel = ParametricTunnel(alignment=alignment, section=section)
        self.tunnel.build_mesh()

    def test_vertex_properties(self):
        from tunnel_geology_model.coupling import TunnelGeologyCoupling
        coupling = TunnelGeologyCoupling(self.tunnel, self.model, field_names=["Vp", "Vs"])
        props = coupling.compute_vertex_properties()
        self.assertIn("Vp", props)
        self.assertEqual(len(props["Vp"]), self.tunnel.n_vertices)
        self.assertTrue(np.allclose(props["Vp"], 5500.0, atol=1))

    def test_centerline_profile(self):
        from tunnel_geology_model.coupling import TunnelGeologyCoupling
        coupling = TunnelGeologyCoupling(self.tunnel, self.model, field_names=["Vp"])
        profile = coupling.compute_centerline_profile()
        self.assertIn("chainage", profile)
        self.assertIn("Vp", profile)
        self.assertEqual(len(profile["chainage"]), 15)

    def test_segments(self):
        from tunnel_geology_model.coupling import TunnelGeologyCoupling
        coupling = TunnelGeologyCoupling(self.tunnel, self.model, field_names=["Vp"])
        coupling.compute_centerline_profile()
        segments = coupling.classify_chainage_segments(segment_length=30)
        self.assertGreater(len(segments), 1)

    def test_excavation_zone(self):
        from tunnel_geology_model.coupling import TunnelGeologyCoupling
        coupling = TunnelGeologyCoupling(self.tunnel, self.model)
        zone = coupling.extract_excavation_zone(margin=2.0)
        self.assertIsInstance(zone, GeologicalModel)
        self.assertLess(zone.nx, self.model.nx)  # sub-volume

    def test_statistics(self):
        from tunnel_geology_model.coupling import TunnelGeologyCoupling
        coupling = TunnelGeologyCoupling(self.tunnel, self.model)
        stats = coupling.vertex_statistics()
        self.assertIn("Vp", stats)
        self.assertAlmostEqual(stats["Vp"]["mean"], 5500, delta=10)


if __name__ == "__main__":
    unittest.main()
