"""Main window - dock-based layout with central VTK view."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QThreadPool, Qt
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QStatusBar,
)

from tunnel_geology_model.coupling import TunnelGeologyCoupling
from tunnel_geology_model.isosurface import marching_cubes_isosurface

from .dialogs.import_dialog import ImportDialog
from .panels.data_panel import DataPanel
from .panels.info_panel import InfoPanel
from .panels.property_panel import PropertyPanel
from .panels.slice_panel import SlicePanel
from .panels.tunnel_panel import TunnelPanel
from .tasks import FunctionWorker
from .widgets.render_widget import RenderWidget


class MainWindow(QMainWindow):
    """The main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tunnel Geology Modeler - 闅ч亾鍦拌川寤烘ā宸ュ叿")
        self.setDockNestingEnabled(True)

        self._model = None
        self._current_field = "Vp"
        self._busy_operation: str | None = None
        self._active_workers: set[FunctionWorker] = set()
        self._thread_pool = QThreadPool.globalInstance()

        self._render = RenderWidget()
        self.setCentralWidget(self._render)

        self._data_panel = DataPanel()
        self._data_panel.dataset_loaded.connect(self._on_dataset_loaded)
        self._data_panel.field_selected.connect(self._on_field_selected)

        self._property_panel = PropertyPanel()
        self._slice_panel = SlicePanel()
        self._tunnel_panel = TunnelPanel()
        self._info_panel = InfoPanel()

        self._render.probe_changed.connect(self._on_probe)
        self._slice_panel.slice_moved.connect(self._on_slice_moved)
        self._slice_panel.isosurface_requested.connect(self._on_isosurface_requested)
        self._slice_panel.isosurface_clear.connect(self._render.clear_isosurface)
        self._tunnel_panel.tunnel_built.connect(self._on_tunnel_built)
        self._tunnel_panel.coupling_requested.connect(self._start_coupling_task)
        self._tunnel_panel.coupling_computed.connect(self._on_coupling_computed)

        self._add_dock(self._data_panel, "馃搧 Data", Qt.DockWidgetArea.LeftDockWidgetArea)
        self._add_dock(self._tunnel_panel, "馃殗 Tunnel", Qt.DockWidgetArea.LeftDockWidgetArea)
        self._add_dock(self._property_panel, "馃搳 Properties", Qt.DockWidgetArea.RightDockWidgetArea)
        self._add_dock(self._slice_panel, "鉁傦笍 Slices", Qt.DockWidgetArea.RightDockWidgetArea)
        self._add_dock(self._info_panel, "馃搵 Log", Qt.DockWidgetArea.BottomDockWidgetArea)

        self._build_menus()

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setMaximumWidth(220)
        self._progress.setVisible(False)
        self._status.addPermanentWidget(self._progress)
        self._status.showMessage("Ready - 鍑嗗灏辩华")

    def _build_menus(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        file_menu.addAction("&Open NetCDF...", self._open_file, "Ctrl+O")
        file_menu.addAction("&Import CSV...", self._import_csv)
        file_menu.addSeparator()
        file_menu.addAction("&Export VTK...", self._export_vtk)
        file_menu.addAction("E&xport XYZ...", self._export_xyz)
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.close, "Ctrl+Q")

        view_menu = mb.addMenu("&View")
        view_menu.addAction("Reset &Camera", self._render.reset_camera, "R")
        view_menu.addAction("&Top View", lambda: self._render.set_view("+Z"), "T")
        view_menu.addAction("&Front View", lambda: self._render.set_view("+Y"), "F")
        view_menu.addAction("&Side View", lambda: self._render.set_view("-X"), "S")

        help_menu = mb.addMenu("&Help")
        help_menu.addAction("&About", self._about)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open NetCDF Dataset",
            "",
            "NetCDF Files (*.nc);;All Files (*)",
        )
        if path:
            self.load_dataset(Path(path))

    def load_dataset(self, path: Path):
        if self._busy_operation:
            self._warn_busy()
            return
        if not path.exists():
            QMessageBox.warning(self, "Missing File", f"Dataset not found:\n{path}")
            return
        if path.suffix.lower() != ".nc":
            QMessageBox.warning(self, "Unsupported File", "Please choose a NetCDF `.nc` dataset.")
            return

        self._run_task(
            operation="load",
            title=f"Loading {path.name}",
            work_fn=self._load_dataset_worker,
            on_success=lambda model: self._apply_loaded_model(path, model),
            args=(path,),
        )

    def _on_dataset_loaded(self, path: Path):
        self.load_dataset(path)

    def _on_field_selected(self, field_name: str):
        if self._busy_operation:
            self._warn_busy()
            return
        if not self._model:
            QMessageBox.information(self, "No Dataset", "Load a dataset before switching fields.")
            return
        if field_name not in self._model:
            QMessageBox.warning(self, "Unknown Field", f"'{field_name}' is not available in the current model.")
            return

        self._current_field = field_name
        self._render.set_field(field_name)
        self._property_panel.set_model(self._model, field_name)
        self._slice_panel.set_field_range(field_name, np.asarray(self._model[field_name]))
        self._info_panel.log(f"Viewing field: {field_name}")

    def _on_probe(self, x: float, y: float, z: float, values: dict):
        self._property_panel.set_probe(x, y, z, values)

    def _on_slice_moved(self, axis: str, index: int):
        if self._busy_operation:
            return
        self._render.set_slice(axis, index)

    def _on_tunnel_built(self, tunnel):
        self._render.set_tunnel(tunnel)
        self._info_panel.log(f"Tunnel built: {tunnel.summary()}")

    def _on_coupling_computed(self, coupling):
        self._info_panel.log(f"Coupling computed: {coupling.summary().split(chr(10))[0]}")

    def _on_isosurface_requested(self, iso_value: float):
        if self._busy_operation:
            self._warn_busy()
            return
        if self._model is None:
            QMessageBox.information(self, "No Dataset", "Load a dataset before building an isosurface.")
            return

        data = np.asarray(self._model[self._current_field], dtype=np.float32)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            QMessageBox.warning(self, "No Valid Data", f"The field '{self._current_field}' contains no finite values.")
            return

        vmin = float(finite.min())
        vmax = float(finite.max())
        if not (vmin <= iso_value <= vmax):
            QMessageBox.warning(
                self,
                "Threshold Out Of Range",
                f"Threshold {iso_value:.3f} is outside the current field range [{vmin:.3f}, {vmax:.3f}].",
            )
            return

        self._run_task(
            operation="isosurface",
            title=f"Building {self._current_field} isosurface",
            work_fn=self._isosurface_worker,
            on_success=self._apply_isosurface_result,
            args=(self._model, self._current_field, float(iso_value)),
        )

    def _start_coupling_task(self, tunnel):
        if self._busy_operation:
            self._warn_busy()
            return
        if self._model is None:
            QMessageBox.information(self, "No Dataset", "Load a dataset before coupling the tunnel.")
            return

        self._run_task(
            operation="coupling",
            title="Coupling tunnel with geology",
            work_fn=self._coupling_worker,
            on_success=self._apply_coupling_result,
            args=(self._model, tunnel),
        )

    def _import_csv(self):
        if self._busy_operation:
            self._warn_busy()
            return
        dialog = ImportDialog(self)
        if dialog.exec():
            self._info_panel.log(f"Import configuration saved: {dialog.config_path}")

    def _export_vtk(self):
        if self._busy_operation:
            self._warn_busy()
            return
        if not self._model:
            QMessageBox.warning(self, "Warning", "No data loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export VTK", "", "VTK Files (*.vts *.vtr)")
        if path:
            from tunnel_geology_model.export import to_vtk_rectilinear_grid

            to_vtk_rectilinear_grid(self._model, path)
            self._info_panel.log(f"Exported VTK: {path}")

    def _export_xyz(self):
        if self._busy_operation:
            self._warn_busy()
            return
        if not self._model:
            QMessageBox.warning(self, "Warning", "No data loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export XYZ", "", "CSV Files (*.csv)")
        if path:
            from tunnel_geology_model.export import to_xyz_point_cloud

            to_xyz_point_cloud(self._model, path, sample=0.1)
            self._info_panel.log(f"Exported XYZ (10% sample): {path}")

    def _about(self):
        QMessageBox.about(
            self,
            "About",
            "Tunnel Geology Modeler v0.1\n\n"
            "闅ч亾瓒呭墠鍦拌川棰勬姤 3D 寤烘ā宸ュ叿\n"
            "Built on: PySide6 + VTK + PyVista",
        )

    def _add_dock(self, widget, title: str, area: Qt.DockWidgetArea):
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(area, dock)

    def _warn_busy(self):
        QMessageBox.information(self, "Operation In Progress", "Please wait for the current task to finish.")

    def _set_busy_state(self, operation: str | None, progress: int = 0, message: str | None = None):
        self._busy_operation = operation
        is_busy = operation is not None
        self._data_panel.set_busy(is_busy)
        self._slice_panel.set_busy(is_busy)
        self._tunnel_panel.set_busy(is_busy, message if is_busy else None)
        self._progress.setVisible(is_busy)
        if is_busy:
            self._progress.setValue(progress)
            if message:
                self._status.showMessage(message)
                self._info_panel.log(message)
        else:
            self._progress.setValue(0)
            self._status.showMessage("Ready - 鍑嗗灏辩华")

    def _run_task(self, operation: str, title: str, work_fn, on_success, args: tuple = ()):
        self._set_busy_state(operation, progress=5, message=title)
        worker = FunctionWorker(work_fn, *args)
        self._active_workers.add(worker)
        worker.signals.progress.connect(self._on_task_progress)
        worker.signals.result.connect(on_success)
        worker.signals.error.connect(self._on_task_error)
        worker.signals.finished.connect(lambda w=worker: self._on_task_finished(w))
        self._thread_pool.start(worker)

    def _on_task_progress(self, value: int, message: str):
        self._progress.setValue(max(0, min(100, value)))
        if message:
            self._status.showMessage(message)

    def _on_task_error(self, message: str, trace: str):
        QMessageBox.critical(self, "Operation Failed", message)
        self._info_panel.log(f"ERROR: {message}")
        if trace.strip():
            self._info_panel.log(trace.strip().splitlines()[-1])

    def _on_task_finished(self, worker: FunctionWorker):
        self._active_workers.discard(worker)
        self._set_busy_state(None)

    def _apply_loaded_model(self, path: Path, model):
        self._model = model
        if self._current_field not in self._model:
            self._current_field = self._model.field_names[0]

        self._render.set_model(self._model, self._current_field)
        self._data_panel.set_model(self._model)
        self._property_panel.set_model(self._model, self._current_field)
        self._slice_panel.set_model(self._model)
        self._slice_panel.set_field_range(self._current_field, np.asarray(self._model[self._current_field]))
        self._tunnel_panel.set_model(self._model)
        self._info_panel.log(f"Loaded: {self._model.summary()}")
        self._status.showMessage(f"Loaded {path.name} - {self._model.nx}x{self._model.ny}x{self._model.nz}")

    def _apply_isosurface_result(self, result: dict):
        self._render.set_isosurface_mesh(result["vertices"], result["faces"])
        self._info_panel.log(
            f"Isosurface ready: field={result['field_name']} threshold={result['iso_value']:.3f} "
            f"triangles={len(result['faces'])}"
        )

    def _apply_coupling_result(self, result: dict):
        self._tunnel_panel.set_coupling_result(
            result["coupling"],
            result["segments"],
            warnings=result["warnings"],
        )
        for warning in result["warnings"]:
            self._info_panel.log(f"WARNING: {warning}")

    @staticmethod
    def _load_dataset_worker(progress, path: Path):
        from tunnel_geology_model import classify_bq, load_from_netcdf

        progress(10, f"Reading NetCDF: {path.name}")
        model = load_from_netcdf(path)
        progress(65, "Computing BQ classification...")
        bq, bq_cls = classify_bq(model["Vp"])
        model.classification["BQ"] = bq
        model.classification["BQ_Class"] = bq_cls.astype("float32")
        progress(90, "Preparing model for display...")
        return model

    @staticmethod
    def _isosurface_worker(progress, model, field_name: str, iso_value: float):
        progress(10, f"Preparing field {field_name}...")
        data = np.asarray(model[field_name], dtype=np.float32)
        progress(35, "Running marching cubes...")
        vertices, faces, _, _ = marching_cubes_isosurface(
            data,
            iso_value=iso_value,
            spatial_step=model.grid_step,
            origin=(model.x_range[0], model.y_range[0], model.z_range[0]),
        )
        progress(95, "Finalizing isosurface...")
        return {
            "field_name": field_name,
            "iso_value": iso_value,
            "vertices": vertices,
            "faces": faces,
        }

    @staticmethod
    def _coupling_worker(progress, model, tunnel):
        progress(10, "Preparing tunnel-geology coupling...")
        coupling = TunnelGeologyCoupling(
            tunnel=tunnel,
            model=model,
            field_names=["Vp", "Vs", "Density", "Young_Modulus"],
        )
        progress(35, "Sampling tunnel mesh properties...")
        coupling.compute_vertex_properties()
        progress(65, "Sampling centerline profile...")
        coupling.compute_centerline_profile()
        progress(85, "Summarizing chainage segments...")
        segments = coupling.classify_chainage_segments(segment_length=20.0)
        warnings = []
        if any(not seg.get("has_data", False) for seg in segments):
            warnings.append("Some chainage segments fall outside the sampled geology range and are marked as no data.")
        return {
            "coupling": coupling,
            "segments": segments,
            "warnings": warnings,
        }
