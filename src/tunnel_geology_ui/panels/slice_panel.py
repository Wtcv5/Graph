"""Slice + Isosurface control panel."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSlider, QLabel, QGroupBox, QHBoxLayout,
    QPushButton, QDoubleSpinBox,
)
from PySide6.QtCore import Signal, Qt


class SlicePanel(QWidget):
    slice_moved = Signal(str, int)
    isosurface_requested = Signal(float)  # iso_value
    isosurface_clear = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = None

        layout = QVBoxLayout(self)

        # ── Slices ──
        slices_group = QGroupBox("Ortho Slices")
        slices_layout = QVBoxLayout(slices_group)
        self._x_slider = self._make_slider_box("X", "x")
        self._y_slider = self._make_slider_box("Y", "y")
        self._z_slider = self._make_slider_box("Z", "z")
        slices_layout.addWidget(self._x_slider["group"])
        slices_layout.addWidget(self._y_slider["group"])
        slices_layout.addWidget(self._z_slider["group"])
        layout.addWidget(slices_group)

        # ── Isosurface ──
        iso_group = QGroupBox("Isosurface")
        iso_layout = QVBoxLayout(iso_group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Threshold:"))
        self._iso_value = QDoubleSpinBox()
        self._iso_value.setRange(0, 20000)
        self._iso_value.setValue(5800)
        self._iso_value.setDecimals(0)
        self._iso_value.setSingleStep(100)
        row.addWidget(self._iso_value)
        iso_layout.addLayout(row)

        btn_row = QHBoxLayout()
        self._iso_build_btn = QPushButton("Build")
        self._iso_build_btn.clicked.connect(
            lambda: self.isosurface_requested.emit(self._iso_value.value())
        )
        self._iso_clear_btn = QPushButton("Clear")
        self._iso_clear_btn.clicked.connect(self.isosurface_clear.emit)
        btn_row.addWidget(self._iso_build_btn)
        btn_row.addWidget(self._iso_clear_btn)
        iso_layout.addLayout(btn_row)

        layout.addWidget(iso_group)
        layout.addStretch()

    def set_model(self, model):
        self._model = model
        self._config_slider(self._x_slider, model.nx, model.x)
        self._config_slider(self._y_slider, model.ny, model.y)
        self._config_slider(self._z_slider, model.nz, model.z)
        arr = model.field("Vp")
        vmin, vmax = float(arr.min()), float(arr.max())
        self._iso_value.setRange(vmin, vmax)
        self._iso_value.setValue(vmin + 0.7 * (vmax - vmin))

    def _make_slider_box(self, title: str, axis: str) -> dict:
        group = QGroupBox(title)
        vbox = QVBoxLayout(group)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(50)
        slider.valueChanged.connect(
            lambda v, a=axis: self.slice_moved.emit(a, v)
        )
        label_row = QHBoxLayout()
        self_label = QLabel("0.0 m")
        coord_label = QLabel("")
        label_row.addWidget(self_label)
        label_row.addStretch()
        label_row.addWidget(coord_label)
        vbox.addWidget(slider)
        vbox.addLayout(label_row)
        return {
            "group": group, "slider": slider,
            "self_label": self_label, "coord_label": coord_label,
            "axis": axis,
        }

    def _config_slider(self, config: dict, count: int, coords):
        config["slider"].setMaximum(max(0, count - 1))
        config["slider"].setValue(count // 2)
        config["coord_label"].setText(f"[{coords[0]:.1f} ... {coords[-1]:.1f}]")
