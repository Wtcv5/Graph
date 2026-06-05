from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tunnel_geology_model import GeologicalModel
from tunnel_geology_model.coupling import TunnelGeologyCoupling
from tunnel_geology_model.tunnel import ParametricTunnel, SectionShape, TunnelAlignment, TunnelSection


class CouplingEdgeCaseTests(unittest.TestCase):
    def test_segments_mark_no_data_when_profile_is_outside_model(self):
        model = GeologicalModel(
            x=np.linspace(0, 10, 6, dtype=np.float64),
            y=np.linspace(-2, 2, 5, dtype=np.float64),
            z=np.linspace(0, 10, 6, dtype=np.float64),
            fields={"Vp": np.ones((5, 6, 6), dtype=np.float32) * 5000},
            metadata={"grid_step_m": 2.0},
            classification={"BQ": np.ones((5, 6, 6), dtype=np.float32) * 300},
        )

        alignment = TunnelAlignment.from_endpoints(
            x_start=2,
            y_start=0,
            z_start=5,
            x_end=14,
            y_end=0,
            z_end=5,
            n_samples=8,
        )
        section = TunnelSection(shape=SectionShape.CIRCULAR, width=4, height=4, n_vertices=16)
        tunnel = ParametricTunnel(alignment=alignment, section=section)
        tunnel.build_mesh()

        coupling = TunnelGeologyCoupling(tunnel=tunnel, model=model, field_names=["Vp"])
        coupling.compute_centerline_profile()
        segments = coupling.classify_chainage_segments(segment_length=4.0)

        self.assertTrue(any(not seg["has_data"] for seg in segments))
        self.assertTrue(any(seg.get("BQ_class") is None for seg in segments if not seg["has_data"]))


if __name__ == "__main__":
    unittest.main()
