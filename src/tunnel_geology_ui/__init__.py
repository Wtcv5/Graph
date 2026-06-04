"""Tunnel Geology GUI — Qt + VTK 3D geological modeling application."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


def run_gui(data_path: str | Path | None = None) -> int:
    """Launch the GUI application.

    Parameters
    ----------
    data_path : str, Path, or None
        Optional path to a NetCDF file to open on startup.

    Returns
    -------
    int — application exit code.
    """
    app = QApplication(sys.argv)
    app.setApplicationName("Tunnel Geology Modeler")
    app.setOrganizationName("TunnelGeology")
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)

    font = app.font()
    font.setPointSize(9)
    app.setFont(font)

    from .main_window import MainWindow

    window = MainWindow()
    window.resize(1400, 900)
    window.show()

    if data_path:
        window.load_dataset(Path(data_path))

    return app.exec()


def main() -> int:
    """CLI-friendly entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Tunnel Geology Modeler GUI")
    parser.add_argument("data", nargs="?", type=Path, help="Path to NetCDF dataset to open")
    args = parser.parse_args()

    return run_gui(args.data)
