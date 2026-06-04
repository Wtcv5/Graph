from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tunnel_velocity_model.config import load_dataset_config
from tunnel_velocity_model.pipeline import grid_align


class ConfigTests(unittest.TestCase):
    def test_default_dataset_config_resolves_paths(self) -> None:
        config = load_dataset_config(
            PROJECT_ROOT / "configs" / "datasets" / "right_tunnel_exit_vpvs.json",
            project_root=PROJECT_ROOT,
        )

        self.assertEqual(config.name, "right_tunnel_exit_vpvs")
        self.assertEqual(config.grid_step_m, 0.5)
        self.assertEqual(config.vp_path.name, "出口右线Dyk1252+411-Vp.csv")
        self.assertEqual(config.vs_path.name, "出口右线Dyk1252+411-Vs.csv")
        self.assertEqual(config.output_nc_path.name, "geological_model.nc")

    def test_grid_align_rounds_to_expected_step(self) -> None:
        self.assertEqual(grid_align(-12359.64, 0.5), -12359.5)
        self.assertEqual(grid_align(2975.818, 0.5), 2976.0)


if __name__ == "__main__":
    unittest.main()
