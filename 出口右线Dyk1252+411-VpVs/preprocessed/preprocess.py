from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
LEGACY_OUTPUT_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tunnel_velocity_model.cli import main


if __name__ == "__main__":
    forwarded_args = list(sys.argv[1:])
    if "--output-dir" not in forwarded_args:
        forwarded_args = ["--output-dir", str(LEGACY_OUTPUT_DIR), *forwarded_args]
    if "--force" not in forwarded_args:
        forwarded_args = ["--force", *forwarded_args]

    raise SystemExit(
        main(
            [
                "preprocess",
                "--project-root",
                str(PROJECT_ROOT),
                *forwarded_args,
            ]
        )
    )
