"""VTK 3D render widget — ortho-slices, isosurface, probe."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Signal, Qt

import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

if TYPE_CHECKING:
    from tunnel_geology_model import GeologicalModel


class RenderWidget(QWidget):
    """Central 3D view with VTK renderer."""

    probe_changed = Signal(float, float, float, dict)  # x, y, z, field_values

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model: GeologicalModel | None = None
        self._current_field = "Vp"
        self._slice_x_idx = 0
        self._slice_y_idx = 0
        self._slice_z_idx = 0

        # ── Layout ──
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # ── VTK widget ──
        self._vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self._vtk_widget)

        # ── VTK pipeline ──
        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(0.15, 0.15, 0.18)
        self._renderer.SetBackground2(0.08, 0.08, 0.10)
        self._renderer.GradientBackgroundOn()
        self._vtk_widget.GetRenderWindow().AddRenderer(self._renderer)

        self._interactor = self._vtk_widget.GetRenderWindow().GetInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self._interactor.SetInteractorStyle(style)

        # Actors (created lazily)
        self._outline_actor: vtk.vtkActor | None = None
        self._slice_x_actor: vtk.vtkImageActor | None = None
        self._slice_y_actor: vtk.vtkImageActor | None = None
        self._slice_z_actor: vtk.vtkImageActor | None = None
        self._iso_actor: vtk.vtkActor | None = None
        self._tunnel_actor: vtk.vtkActor | None = None
        self._axes_actor: vtk.vtkAxesActor | None = None

        # Color transfer function
        self._ctf = vtk.vtkColorTransferFunction()
        self._build_jet_ctf()

        # ── Probe picker ──
        self._picker = vtk.vtkPointPicker()
        self._picker.AddObserver("EndPickEvent", self._on_pick)
        self._interactor.SetPicker(self._picker)

        # Initial camera
        self._renderer.GetActiveCamera().SetPosition(0, -300, 3200)
        self._renderer.GetActiveCamera().SetFocalPoint(-12260, 0, 3028)
        self._renderer.GetActiveCamera().SetViewUp(0, 0, 1)

    # ── public API ──────────────────────────────────────

    def set_model(self, model: "GeologicalModel", field_name: str | None = None):
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
            self._update_isosurface()

    def set_slice(self, axis: str, index: int):
        if axis == "x":
            self._slice_x_idx = index
            self._update_slice_x()
        elif axis == "y":
            self._slice_y_idx = index
            self._update_slice_y()
        elif axis == "z":
            self._slice_z_idx = index
            self._update_slice_z()

    def reset_camera(self):
        self._renderer.ResetCamera()
        self._vtk_widget.GetRenderWindow().Render()

    def set_tunnel(self, tunnel: "ParametricTunnel | None"):
        """Display or hide a tunnel 3D mesh overlay."""
        if self._tunnel_actor:
            self._renderer.RemoveActor(self._tunnel_actor)
            self._tunnel_actor = None

        if tunnel is None or tunnel.vertices is None or tunnel.faces is None:
            self._vtk_widget.GetRenderWindow().Render()
            return

        # Convert numpy arrays to VTK polydata
        verts = tunnel.vertices  # (N, 3)
        faces = tunnel.faces      # (M, 3)

        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(vtk.vtkFloatArray())
        for i, v in enumerate(verts):
            vtk_points.InsertNextPoint(v[0], v[1], v[2])

        vtk_cells = vtk.vtkCellArray()
        for f in faces:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, int(f[0]))
            triangle.GetPointIds().SetId(1, int(f[1]))
            triangle.GetPointIds().SetId(2, int(f[2]))
            vtk_cells.InsertNextCell(triangle)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetPolys(vtk_cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        self._tunnel_actor = vtk.vtkActor()
        self._tunnel_actor.SetMapper(mapper)
        self._tunnel_actor.GetProperty().SetColor(0.85, 0.65, 0.4)
        self._tunnel_actor.GetProperty().SetOpacity(0.7)
        self._tunnel_actor.GetProperty().SetEdgeVisibility(True)
        self._tunnel_actor.GetProperty().SetEdgeColor(0.3, 0.3, 0.3)
        self._tunnel_actor.GetProperty().SetLineWidth(0.5)

        self._renderer.AddActor(self._tunnel_actor)
        self._vtk_widget.GetRenderWindow().Render()

    def set_view(self, direction: str):
        cam = self._renderer.GetActiveCamera()
        center = np.array(self._renderer.GetActiveCamera().GetFocalPoint())
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
        self._vtk_widget.GetRenderWindow().Render()

    # ── internal build ──────────────────────────────────

    def _rebuild(self):
        self._clear_actors()
        self._build_outline()
        self._build_slices()
        self._build_isosurface()
        self._renderer.ResetCamera()
        self._vtk_widget.GetRenderWindow().Render()

    def _clear_actors(self):
        for actor in [self._outline_actor, self._slice_x_actor,
                       self._slice_y_actor, self._slice_z_actor, self._iso_actor]:
            if actor:
                self._renderer.RemoveActor(actor)

    def _build_jet_ctf(self):
        """Build a jet-like color transfer function."""
        # Build a jet-like color map: blue → cyan → green → yellow → red
        points = [
            (0.00, 0.0, 0.0, 0.5625),  # blue
            (0.25, 0.0, 1.0, 1.0),      # cyan
            (0.50, 0.0, 1.0, 0.0),      # green
            (0.75, 1.0, 1.0, 0.0),      # yellow
            (1.00, 1.0, 0.0, 0.0),      # red
        ]
        self._ctf.RemoveAllPoints()
        for v, r, g, b in points:
            self._ctf.AddRGBPoint(v, r, g, b)

    def _field_range(self):
        if self._model is None:
            return 0.0, 1.0
        arr = self._model[self._current_field]
        return float(np.nanmin(arr)), float(np.nanmax(arr))

    # ── outline ─────────────────────────────────────────

    def _build_outline(self):
        if self._model is None:
            return
        x0, x1 = self._model.x_range
        y0, y1 = self._model.y_range
        z0, z1 = self._model.z_range

        cube = vtk.vtkCubeSource()
        cube.SetBounds(x0, x1, y0, y1, z0, z1)
        cube.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())

        self._outline_actor = vtk.vtkActor()
        self._outline_actor.SetMapper(mapper)
        self._outline_actor.GetProperty().SetRepresentationToWireframe()
        self._outline_actor.GetProperty().SetColor(0.4, 0.4, 0.4)
        self._outline_actor.GetProperty().SetLineWidth(1)
        self._renderer.AddActor(self._outline_actor)

    # ── slices ──────────────────────────────────────────

    def _build_slices(self):
        if self._model is None:
            return
        self._rebuild_slice_x()
        self._rebuild_slice_y()
        self._rebuild_slice_z()

    def _rebuild_slice_x(self):
        if self._model is None:
            return
        arr = self._model.field(self._current_field)  # (ny, nx, nz)
        nx = self._model.nx
        idx = min(self._slice_x_idx, nx - 1)

        # Extract slice: (ny, nz) at this X
        img = self._numpy_to_vtk_image(arr[:, idx, :])  # (ny, nz)

        # Position
        x_pos = float(self._model.x[idx])
        extent = img.GetExtent()
        img.SetExtent(
            0, 0,
            int(self._model.y_range[0]), int(self._model.y_range[1]),
            extent[4], extent[5],
        )
        # Use SetOrigin/SetSpacing instead
        img.SetOrigin(x_pos, self._model.y_range[0], self._model.z_range[0])
        img.SetSpacing(1.0, self._model.grid_step, self._model.grid_step)

        # Reslice to XZ plane
        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(img)
        reslice.SetOutputDimensionality(2)
        reslice.SetResliceAxesOrigin(x_pos, 0, 0)
        axes = vtk.vtkMatrix4x4()
        axes.Identity()
        axes.SetElement(0, 0, 0); axes.SetElement(0, 2, 1)
        axes.SetElement(2, 0, -1); axes.SetElement(2, 2, 0)
        reslice.SetResliceAxes(axes)
        reslice.Update()

        self._slice_x_actor = self._create_image_actor(reslice.GetOutput())
        self._renderer.AddActor(self._slice_x_actor)

    def _update_slice_x(self):
        self._rebuild_slice_x()
        self._vtk_widget.GetRenderWindow().Render()

    def _update_slice_y(self):
        self._rebuild_slice_y()
        self._vtk_widget.GetRenderWindow().Render()

    def _update_slice_z(self):
        self._rebuild_slice_z()
        self._vtk_widget.GetRenderWindow().Render()

    def _rebuild_slice_y(self):
        if self._model is None:
            return
        arr = self._model.field(self._current_field)  # (ny, nx, nz)
        ny = self._model.ny
        idx = min(self._slice_y_idx, ny - 1)

        img = self._numpy_to_vtk_image(arr[idx, :, :])  # (nx, nz)
        y_pos = float(self._model.y[idx])
        img.SetOrigin(self._model.x_range[0], y_pos, self._model.z_range[0])
        img.SetSpacing(self._model.grid_step, 1.0, self._model.grid_step)

        self._slice_y_actor = self._create_image_actor(img)
        self._renderer.AddActor(self._slice_y_actor)

    def _rebuild_slice_z(self):
        if self._model is None:
            return
        arr = self._model.field(self._current_field)  # (ny, nx, nz)
        nz = self._model.nz
        idx = min(self._slice_z_idx, nz - 1)

        # Extract Z slice: (ny, nx)
        slice_data = arr[:, :, idx]  # (ny, nx)
        z_pos = float(self._model.z[idx])

        img = self._numpy_to_vtk_image(slice_data)
        img.SetOrigin(self._model.x_range[0], self._model.y_range[0], z_pos)
        img.SetSpacing(self._model.grid_step, self._model.grid_step, 1.0)

        self._slice_z_actor = self._create_image_actor(img)
        self._renderer.AddActor(self._slice_z_actor)

    def _update_slices(self):
        """Refresh all slices for new field."""
        if self._slice_x_actor:
            self._renderer.RemoveActor(self._slice_x_actor)
        if self._slice_y_actor:
            self._renderer.RemoveActor(self._slice_y_actor)
        if self._slice_z_actor:
            self._renderer.RemoveActor(self._slice_z_actor)
        self._build_slices()
        self._vtk_widget.GetRenderWindow().Render()

    # ── isosurface ──────────────────────────────────────

    def _build_isosurface(self):
        if self._model is None:
            return
        arr = self._model.field(self._current_field)
        vmin, vmax = self._field_range()
        iso_value = vmin + 0.7 * (vmax - vmin)  # default threshold

        # Convert numpy to VTK image data
        vtk_img = self._numpy_to_vtk_image(arr.transpose(2, 0, 1))  # (nz, ny, nx)
        vtk_img.SetOrigin(self._model.x_range[0], self._model.y_range[0], self._model.z_range[0])
        vtk_img.SetSpacing(self._model.grid_step, self._model.grid_step, self._model.grid_step)

        # Marching cubes
        mc = vtk.vtkMarchingCubes()
        mc.SetInputData(vtk_img)
        mc.SetValue(0, iso_value)
        mc.ComputeNormalsOn()
        mc.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(mc.GetOutputPort())
        mapper.ScalarVisibilityOff()

        self._iso_actor = vtk.vtkActor()
        self._iso_actor.SetMapper(mapper)
        self._iso_actor.GetProperty().SetColor(0.9, 0.6, 0.2)
        self._iso_actor.GetProperty().SetOpacity(0.5)
        self._renderer.AddActor(self._iso_actor)

    def _update_isosurface(self):
        if self._iso_actor:
            self._renderer.RemoveActor(self._iso_actor)
        self._build_isosurface()
        self._vtk_widget.GetRenderWindow().Render()

    # ── probe / pick ────────────────────────────────────

    def _on_pick(self, obj, event):
        picker = obj
        if picker.GetPointId() < 0:
            return
        pt = picker.GetPickPosition()
        x, y, z = pt
        if self._model:
            ix = int(np.argmin(np.abs(self._model.x - x)))
            iy = int(np.argmin(np.abs(self._model.y - y)))
            iz = int(np.argmin(np.abs(self._model.z - z)))
            values = {}
            for name in self._model.field_names:
                arr = self._model[name]
                values[name] = float(arr[iy, ix, iz])
            for name, arr in self._model.classification.items():
                values[name] = float(arr[iy, ix, iz])
            self.probe_changed.emit(
                float(self._model.x[ix]),
                float(self._model.y[iy]),
                float(self._model.z[iz]),
                values,
            )

    # ── helpers ─────────────────────────────────────────

    def _numpy_to_vtk_image(self, arr: np.ndarray) -> vtk.vtkImageData:
        """Convert 2D or 3D numpy array to vtkImageData."""
        arr_flat = np.ascontiguousarray(arr.ravel(order="F"), dtype=np.float32)
        vtk_data = vtk.vtkFloatArray()
        vtk_data.SetVoidArray(arr_flat, len(arr_flat), 1)
        vtk_data.SetNumberOfComponents(1)

        img = vtk.vtkImageData()
        if arr.ndim == 2:
            img.SetDimensions(arr.shape[1], arr.shape[0], 1)
        else:
            img.SetDimensions(arr.shape[2], arr.shape[1], arr.shape[0])
        img.GetPointData().SetScalars(vtk_data)
        return img

    def _create_image_actor(self, img: vtk.vtkImageData) -> vtk.vtkImageActor:
        """Create a color-mapped vtkImageActor."""
        vmin, vmax = self._field_range()

        # Lookup table
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.SetRange(vmin, vmax)
        lut.Build()

        # Color from CTF → LUT
        for i in range(256):
            t = i / 255.0
            r, g, b = self._ctf.GetColor(t)
            lut.SetTableValue(i, r, g, b, 1.0)

        # Map colors through lookup table
        cmap = vtk.vtkImageMapToColors()
        cmap.SetInputData(img)
        cmap.SetLookupTable(lut)
        cmap.SetOutputFormatToRGBA()
        cmap.Update()

        actor = vtk.vtkImageActor()
        actor.GetMapper().SetInputConnection(cmap.GetOutputPort())
        return actor
