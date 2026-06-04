"""Main window — dock-based layout with central VTK view."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow,
    QDockWidget,
    QMenuBar,
    QMenu,
    QStatusBar,
    QFileDialog,
    QMessageBox,
)
from PySide6.QtCore import Qt

from .panels.data_panel import DataPanel
from .panels.property_panel import PropertyPanel
from .panels.slice_panel import SlicePanel
from .panels.tunnel_panel import TunnelPanel
from .panels.info_panel import InfoPanel
from .dialogs.import_dialog import ImportDialog
from .dialogs.export_dialog import ExportDialog
from .widgets.render_widget import RenderWidget


class MainWindow(QMainWindow):
    """The main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tunnel Geology Modeler — 隧道地质建模工具")
        self.setDockNestingEnabled(True)

        self._model = None
        self._current_field = "Vp"

        # ── Central widget (3D render) ──
        self._render = RenderWidget()
        self.setCentralWidget(self._render)

        # ── Panels ──
        self._data_panel = DataPanel()
        self._data_panel.dataset_loaded.connect(self._on_dataset_loaded)
        self._data_panel.field_selected.connect(self._on_field_selected)

        self._property_panel = PropertyPanel()
        self._slice_panel = SlicePanel()
        self._tunnel_panel = TunnelPanel()
        self._info_panel = InfoPanel()

        # ── Signal wiring ──
        self._render.probe_changed.connect(self._on_probe)
        self._slice_panel.slice_moved.connect(self._on_slice_moved)
        self._tunnel_panel.tunnel_built.connect(self._on_tunnel_built)
        self._tunnel_panel.coupling_computed.connect(self._on_coupling_computed)

        # ── Dock widgets ──
        self._add_dock(self._data_panel, "📁 Data", Qt.DockWidgetArea.LeftDockWidgetArea)
        self._add_dock(self._tunnel_panel, "🚇 Tunnel", Qt.DockWidgetArea.LeftDockWidgetArea)
        self._add_dock(self._property_panel, "📊 Properties", Qt.DockWidgetArea.RightDockWidgetArea)
        self._add_dock(self._slice_panel, "✂️ Slices", Qt.DockWidgetArea.RightDockWidgetArea)
        self._add_dock(self._info_panel, "📋 Log", Qt.DockWidgetArea.BottomDockWidgetArea)

        # ── Menu bar ──
        self._build_menus()

        # ── Status bar ──
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — 准备就绪")

    # ── menus ────────────────────────────────────────────

    def _build_menus(self):
        mb = self.menuBar()

        # File
        file_menu = mb.addMenu("&File")
        file_menu.addAction("&Open NetCDF...", self._open_file, "Ctrl+O")
        file_menu.addAction("&Import CSV...", self._import_csv)
        file_menu.addSeparator()
        file_menu.addAction("&Export VTK...", self._export_vtk)
        file_menu.addAction("E&xport XYZ...", self._export_xyz)
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.close, "Ctrl+Q")

        # View
        view_menu = mb.addMenu("&View")
        view_menu.addAction("Reset &Camera", self._render.reset_camera, "R")
        view_menu.addAction("&Top View", lambda: self._render.set_view("+Z"), "T")
        view_menu.addAction("&Front View", lambda: self._render.set_view("+Y"), "F")
        view_menu.addAction("&Side View", lambda: self._render.set_view("-X"), "S")

        # Help
        help_menu = mb.addMenu("&Help")
        help_menu.addAction("&About", self._about)

    # ── slots ────────────────────────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open NetCDF Dataset", "",
            "NetCDF Files (*.nc);;All Files (*)"
        )
        if path:
            self.load_dataset(Path(path))

    def load_dataset(self, path: Path):
        self._info_panel.log(f"Loading {path.name} ...")
        self._status.showMessage(f"Loading {path.name} ...")
        try:
            from tunnel_geology_model import load_from_netcdf, classify_bq

            self._model = load_from_netcdf(path)
            # Add BQ classification
            bq, bq_cls = classify_bq(self._model["Vp"])
            self._model.classification["BQ"] = bq
            self._model.classification["BQ_Class"] = bq_cls.astype("float32")

            self._render.set_model(self._model, self._current_field)
            self._data_panel.set_model(self._model)
            self._property_panel.set_model(self._model, self._current_field)
            self._slice_panel.set_model(self._model)
            self._tunnel_panel.set_model(self._model)
            self._info_panel.log(f"Loaded: {self._model.summary()}")
            self._status.showMessage(f"Loaded {path.name} — {self._model.nx}x{self._model.ny}x{self._model.nz}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self._info_panel.log(f"ERROR: {e}")

    def _on_dataset_loaded(self, path: Path):
        self.load_dataset(path)

    def _on_field_selected(self, field_name: str):
        self._current_field = field_name
        if self._model:
            self._render.set_model(self._model, field_name)
            self._property_panel.set_model(self._model, field_name)

    def _on_probe(self, x: float, y: float, z: float, values: dict):
        self._property_panel.set_probe(x, y, z, values)

    def _on_slice_moved(self, axis: str, index: int):
        self._render.set_slice(axis, index)

    def _on_tunnel_built(self, tunnel):
        self._render.set_tunnel(tunnel)
        self._info_panel.log(f"Tunnel built: {tunnel.summary()}")

    def _on_coupling_computed(self, coupling):
        self._info_panel.log(f"Coupling computed: {coupling.summary().split(chr(10))[0]}")

    def _import_csv(self):
        dialog = ImportDialog(self)
        if dialog.exec():
            self._info_panel.log(f"Import configuration saved: {dialog.config_path}")

    def _export_vtk(self):
        if not self._model:
            QMessageBox.warning(self, "Warning", "No data loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export VTK", "", "VTK Files (*.vts *.vtr)")
        if path:
            from tunnel_geology_model.export import to_vtk_rectilinear_grid
            to_vtk_rectilinear_grid(self._model, path)
            self._info_panel.log(f"Exported VTK: {path}")

    def _export_xyz(self):
        if not self._model:
            QMessageBox.warning(self, "Warning", "No data loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export XYZ", "", "CSV Files (*.csv)")
        if path:
            from tunnel_geology_model.export import to_xyz_point_cloud
            to_xyz_point_cloud(self._model, path, sample=0.1)  # 10% sample
            self._info_panel.log(f"Exported XYZ (10% sample): {path}")

    def _about(self):
        QMessageBox.about(
            self, "About",
            "Tunnel Geology Modeler v0.1\n\n"
            "隧道超前地质预报 3D 建模工具\n"
            "Built on: PySide6 + VTK + PyVista"
        )

    # ── helpers ──────────────────────────────────────────

    def _add_dock(self, widget, title: str, area: Qt.DockWidgetArea):
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                         QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(area, dock)
