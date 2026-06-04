"""Slice control panel — sliders for X, Y, Z ortho-slices."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSlider, QLabel, QGroupBox, QHBoxLayout,
)
from PySide6.QtCore import Signal, Qt


class SlicePanel(QWidget):
    slice_moved = Signal(str, int)  # axis ('x','y','z'), index

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = None

        layout = QVBoxLayout(self)

        self._x_slider = self._make_slider_box("X (tunnel long.)", "x")
        self._y_slider = self._make_slider_box("Y (tunnel trans.)", "y")
        self._z_slider = self._make_slider_box("Z (elevation)", "z")

        layout.addWidget(self._x_slider["group"])
        layout.addWidget(self._y_slider["group"])
        layout.addWidget(self._z_slider["group"])
        layout.addStretch()

    def set_model(self, model):
        self._model = model
        self._config_slider(self._x_slider, model.nx, model.x)
        self._config_slider(self._y_slider, model.ny, model.y)
        self._config_slider(self._z_slider, model.nz, model.z)

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
        config["coord_label"].setText(
            f"[{coords[0]:.1f} ... {coords[-1]:.1f}]"
        )

    def _on_slider(self, axis: str, value: int, config: dict):
        if self._model is None:
            return
        if axis == "x":
            coord = self._model.x[value]
        elif axis == "y":
            coord = self._model.y[value]
        else:
            coord = self._model.z[value]
        config["self_label"].setText(f"{coord:.1f} m")
        self.slice_moved.emit(axis, value)
