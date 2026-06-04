"""VTK 3D render widget — ortho-slices, isosurface, probe."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Signal

import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

if TYPE_CHECKING:
    from tunnel_geology_model import GeologicalModel
    from tunnel_geology_model.tunnel import ParametricTunnel


class RenderWidget(QWidget):
    """Central 3D view with VTK renderer."""

    probe_changed = Signal(float, float, float, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model: GeologicalModel | None = None
        self._current_field = "Vp"
        self._slice_x_idx = 0
        self._slice_y_idx = 0
        self._slice_z_idx = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # VTK widget
        self._vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self._vtk_widget)

        # Renderer
        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(0.15, 0.15, 0.18)
        self._renderer.SetBackground2(0.08, 0.08, 0.10)
        self._renderer.GradientBackgroundOn()
        self._vtk_widget.GetRenderWindow().AddRenderer(self._renderer)

        # Interactor
        self._interactor = self._vtk_widget.GetRenderWindow().GetInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self._interactor.SetInteractorStyle(style)
        self._interactor.Initialize()

        # Color LUT
        self._lut = vtk.vtkLookupTable()
        self._build_lut()

        # Actors
        self._outline_actor: vtk.vtkActor | None = None
        self._slice_x_actor: vtk.vtkImageActor | None = None
        self._slice_y_actor: vtk.vtkImageActor | None = None
        self._slice_z_actor: vtk.vtkImageActor | None = None
        self._iso_actor: vtk.vtkActor | None = None
        self._tunnel_actor: vtk.vtkActor | None = None

        # Probe
        self._picker = vtk.vtkPointPicker()
        self._picker.AddObserver("EndPickEvent", self._on_pick)
        self._interactor.SetPicker(self._picker)

        # Initial camera
        self._renderer.GetActiveCamera().SetPosition(0, -300, 3200)
        self._renderer.GetActiveCamera().SetFocalPoint(-12260, 0, 3028)
        self._renderer.GetActiveCamera().SetViewUp(0, 0, 1)

    # ── public API ──────────────────────────────────────

    def set_model(self, model: GeologicalModel, field_name: str | None = None):
        self._model = model
        if field_name:
            self._current_field = field_name
        self._slice_x_idx = model.nx // 2
        self._slice_y_idx = model.ny // 2
        self._slice_z_idx = model.nz // 2
        self._rebuild()

    def set_field(self, field_name: str):
        self._current_field = field_name
        if self._model:
            self._update_slices()

    def set_slice(self, axis: str, index: int):
        if axis == "x":
            self._slice_x_idx = index
            self._rebuild_slice_actor("x")
        elif axis == "y":
            self._slice_y_idx = index
            self._rebuild_slice_actor("y")
        elif axis == "z":
            self._slice_z_idx = index
            self._rebuild_slice_actor("z")
        self._render()

    def set_tunnel(self, tunnel: ParametricTunnel | None):
        if self._tunnel_actor:
            self._renderer.RemoveActor(self._tunnel_actor)
            self._tunnel_actor = None
        if tunnel is None or tunnel.vertices is None or tunnel.faces is None:
            self._render()
            return
        self._tunnel_actor = self._build_poly_actor(
            tunnel.vertices, tunnel.faces,
            color=(0.85, 0.65, 0.4), opacity=0.7, edge_visible=True,
        )
        self._renderer.AddActor(self._tunnel_actor)
        self._render()

    def build_isosurface(self, iso_value: float | None = None):
        """Build and display an isosurface. Slow on large grids — runs on main thread."""
        if self._model is None:
            return
        if self._iso_actor:
            self._renderer.RemoveActor(self._iso_actor)
            self._iso_actor = None

        arr = self._model.field(self._current_field)
        if iso_value is None:
            vmin, vmax = self._field_range()
            iso_value = vmin + 0.7 * (vmax - vmin)

        # Convert to VTK image (nz, ny, nx) order
        data = np.ascontiguousarray(
            arr.transpose(2, 0, 1).astype(np.float32)
        )
        importer = vtk.vtkImageImport()
        importer.CopyImportVoidPointer(data.tobytes(), data.nbytes)
        importer.SetDataScalarTypeToFloat()
        importer.SetNumberOfScalarComponents(1)
        importer.SetWholeExtent(0, self._model.nx - 1, 0, self._model.ny - 1,
                                0, self._model.nz - 1)
        importer.SetDataExtent(0, self._model.nx - 1, 0, self._model.ny - 1,
                               0, self._model.nz - 1)
        importer.Update()

        img = importer.GetOutput()
        img.SetOrigin(self._model.x_range[0], self._model.y_range[0], self._model.z_range[0])
        img.SetSpacing(self._model.grid_step, self._model.grid_step, self._model.grid_step)

        mc = vtk.vtkMarchingCubes()
        mc.SetInputData(img)
        mc.SetValue(0, iso_value)
        mc.ComputeNormalsOn()
        mc.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(mc.GetOutputPort())
        mapper.ScalarVisibilityOff()

        self._iso_actor = vtk.vtkActor()
        self._iso_actor.SetMapper(mapper)
        self._iso_actor.GetProperty().SetColor(1.0, 0.8, 0.3)
        self._iso_actor.GetProperty().SetOpacity(0.6)
        self._renderer.AddActor(self._iso_actor)
        self._render()

    def clear_isosurface(self):
        if self._iso_actor:
            self._renderer.RemoveActor(self._iso_actor)
            self._iso_actor = None
            self._render()

    def reset_camera(self):
        self._renderer.ResetCamera()
        self._render()

    def set_view(self, direction: str):
        cam = self._renderer.GetActiveCamera()
        center = np.array(cam.GetFocalPoint())
        dist = cam.GetDistance()
        if direction == "+Z":
            cam.SetPosition(center[0], center[1], center[2] + dist)
            cam.SetViewUp(0, 1, 0)
        elif direction == "+Y":
            cam.SetPosition(center[0], center[1] + dist, center[2])
            cam.SetViewUp(0, 0, 1)
        elif direction == "-X":
            cam.SetPosition(center[0] - dist, center[1], center[2])
            cam.SetViewUp(0, 0, 1)
        self._renderer.ResetCameraClippingRange()
        self._render()

    # ── rebuild ─────────────────────────────────────────

    def _rebuild(self):
        for a in [self._outline_actor, self._slice_x_actor, self._slice_y_actor,
                   self._slice_z_actor, self._iso_actor, self._tunnel_actor]:
            if a:
                self._renderer.RemoveActor(a)
        self._outline_actor = self._build_outline()
        self._slice_x_actor = self._build_slice("x", self._slice_x_idx)
        self._slice_y_actor = self._build_slice("y", self._slice_y_idx)
        self._slice_z_actor = self._build_slice("z", self._slice_z_idx)
        # Skip isosurface on initial load to avoid hang on large data
        self._iso_actor = None
        self._renderer.ResetCamera()
        self._render()

    def _update_slices(self):
        for a in [self._slice_x_actor, self._slice_y_actor, self._slice_z_actor]:
            if a:
                self._renderer.RemoveActor(a)
        self._slice_x_actor = self._build_slice("x", self._slice_x_idx)
        self._slice_y_actor = self._build_slice("y", self._slice_y_idx)
        self._slice_z_actor = self._build_slice("z", self._slice_z_idx)
        self._render()

    def _rebuild_slice_actor(self, axis: str):
        attr = f"_slice_{axis}_actor"
        idx_attr = f"_slice_{axis}_idx"
        old = getattr(self, attr)
        if old:
            self._renderer.RemoveActor(old)
        new_actor = self._build_slice(axis, getattr(self, idx_attr))
        setattr(self, attr, new_actor)

    # ── actors ──────────────────────────────────────────

    def _build_outline(self) -> vtk.vtkActor:
        m = self._model
        cube = vtk.vtkCubeSource()
        cube.SetBounds(m.x_range[0], m.x_range[1], m.y_range[0], m.y_range[1],
                       m.z_range[0], m.z_range[1])
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetColor(0.4, 0.4, 0.4)
        self._renderer.AddActor(actor)
        return actor

    def _build_slice(self, axis: str, idx: int) -> vtk.vtkImageActor:
        m = self._model
        arr = m.field(self._current_field)

        if axis == "x":
            idx = min(idx, m.nx - 1)
            data_2d = arr[:, idx, :].copy()   # (ny, nz)
            ox, oy, oz = float(m.x[idx]), m.y_range[0], m.z_range[0]
            sx, sy, sz = 1.0, m.grid_step, m.grid_step
        elif axis == "y":
            idx = min(idx, m.ny - 1)
            data_2d = arr[idx, :, :].copy()   # (nx, nz) — needs transpose to XZ
            ox, oy, oz = m.x_range[0], float(m.y[idx]), m.z_range[0]
            sx, sy, sz = m.grid_step, 1.0, m.grid_step
        else:  # z
            idx = min(idx, m.nz - 1)
            data_2d = arr[:, :, idx].copy()   # (ny, nx) — needs transpose to XY
            ox, oy, oz = m.x_range[0], m.y_range[0], float(m.z[idx])
            sx, sy, sz = m.grid_step, m.grid_step, 1.0

        img = self._array_to_vtk_image(data_2d)
        img.SetOrigin(ox, oy, oz)
        img.SetSpacing(sx, sy, sz)
        actor = self._color_map_actor(img)
        self._renderer.AddActor(actor)
        return actor

    def _build_poly_actor(self, vertices, faces, color, opacity, edge_visible):
        vtk_pts = vtk.vtkPoints()
        for v in vertices:
            vtk_pts.InsertNextPoint(v[0], v[1], v[2])
        vtk_cells = vtk.vtkCellArray()
        for f in faces:
            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0, int(f[0]))
            tri.GetPointIds().SetId(1, int(f[1]))
            tri.GetPointIds().SetId(2, int(f[2]))
            vtk_cells.InsertNextCell(tri)
        pd = vtk.vtkPolyData()
        pd.SetPoints(vtk_pts)
        pd.SetPolys(vtk_cells)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        if edge_visible:
            actor.GetProperty().SetEdgeVisibility(True)
            actor.GetProperty().SetEdgeColor(0.3, 0.3, 0.3)
        return actor

    # ── helpers ─────────────────────────────────────────

    def _array_to_vtk_image(self, arr_2d: np.ndarray) -> vtk.vtkImageData:
        """Safely convert a 2D numpy array (cols, rows) to vtkImageData."""
        ny, nx = arr_2d.shape
        data = np.ascontiguousarray(arr_2d.astype(np.float32))
        # Use vtkImageImport for safe conversion
        importer = vtk.vtkImageImport()
        importer.CopyImportVoidPointer(data.tobytes(), data.nbytes)
        importer.SetDataScalarTypeToFloat()
        importer.SetNumberOfScalarComponents(1)
        importer.SetWholeExtent(0, nx - 1, 0, ny - 1, 0, 0)
        importer.SetDataExtent(0, nx - 1, 0, ny - 1, 0, 0)
        importer.Update()
        return importer.GetOutput()

    def _color_map_actor(self, img: vtk.vtkImageData) -> vtk.vtkImageActor:
        cmap = vtk.vtkImageMapToColors()
        cmap.SetInputData(img)
        cmap.SetLookupTable(self._lut)
        cmap.SetOutputFormatToRGBA()
        cmap.Update()
        actor = vtk.vtkImageActor()
        actor.GetMapper().SetInputConnection(cmap.GetOutputPort())
        return actor

    def _build_lut(self):
        self._lut.SetNumberOfTableValues(256)
        self._lut.Build()
        for i in range(256):
            t = i / 255.0
            # Jet: blue→cyan→green→yellow→red
            if t < 0.25:
                r, g, b = 0.0, 4.0 * t, 1.0
            elif t < 0.5:
                r, g, b = 0.0, 1.0, 2.0 - 4.0 * t
            elif t < 0.75:
                r, g, b = 4.0 * t - 2.0, 1.0, 0.0
            else:
                r, g, b = 1.0, 4.0 - 4.0 * t, 0.0
            self._lut.SetTableValue(i, r, g, b, 1.0)

    def _update_lut_range(self):
        vmin, vmax = self._field_range()
        self._lut.SetRange(vmin, vmax)
        self._lut.Build()

    def _field_range(self):
        if self._model is None:
            return 0.0, 1.0
        arr = self._model[self._current_field]
        return float(np.nanmin(arr)), float(np.nanmax(arr))

    # ── probe ───────────────────────────────────────────

    def _on_pick(self, obj, event):
        picker = obj
        if picker.GetPointId() < 0:
            return
        pt = picker.GetPickPosition()
        x, y, z = pt
        m = self._model
        if m:
            ix = int(np.argmin(np.abs(m.x - x)))
            iy = int(np.argmin(np.abs(m.y - y)))
            iz = int(np.argmin(np.abs(m.z - z)))
            vals = {}
            for name in m.field_names:
                vals[name] = float(m.fields[name][iy, ix, iz])
            for name, arr in m.classification.items():
                vals[name] = float(arr[iy, ix, iz])
            self.probe_changed.emit(float(m.x[ix]), float(m.y[iy]), float(m.z[iz]), vals)

    def _render(self):
        self._update_lut_range()
        self._vtk_widget.GetRenderWindow().Render()
