from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tunnel_velocity_model.cli import build_parser, main


class CliTests(unittest.TestCase):
    def test_parser_accepts_gui_subcommand(self) -> None:
        args = build_parser().parse_args(["gui", "outputs/right_tunnel_exit_vpvs/geological_model.nc"])

        self.assertEqual(args.command, "gui")
        self.assertEqual(Path(args.data), Path("outputs/right_tunnel_exit_vpvs/geological_model.nc"))

    def test_gui_command_invokes_runner_with_resolved_path(self) -> None:
        captured: dict[str, Path | None] = {}

        fake_module = types.ModuleType("tunnel_geology_ui")

        def fake_run_gui(data_path=None):
            captured["data_path"] = data_path
            return 0

        fake_module.run_gui = fake_run_gui

        with patch.dict(sys.modules, {"tunnel_geology_ui": fake_module}):
            exit_code = main(
                [
                    "gui",
                    "--project-root",
                    str(PROJECT_ROOT),
                    "outputs/right_tunnel_exit_vpvs/geological_model.nc",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            captured["data_path"],
            (PROJECT_ROOT / "outputs" / "right_tunnel_exit_vpvs" / "geological_model.nc").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
