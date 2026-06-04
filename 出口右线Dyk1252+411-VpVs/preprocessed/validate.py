from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_NC_PATH = Path(__file__).resolve().parent / "geological_model.nc"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tunnel_velocity_model.cli import main


if __name__ == "__main__":
    forwarded_args = list(sys.argv[1:])
    if "--nc-path" not in forwarded_args:
        forwarded_args = ["--nc-path", str(DEFAULT_NC_PATH), *forwarded_args]

    raise SystemExit(
        main(
            [
                "validate",
                "--project-root",
                str(PROJECT_ROOT),
                *forwarded_args,
            ]
        )
    )
