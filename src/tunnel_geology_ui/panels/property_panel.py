"""Property panel — probe values and statistics for current field."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QFormLayout,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class PropertyPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = None

        layout = QVBoxLayout(self)

        # ── Field stats ──
        stats_group = QGroupBox("Field Statistics")
        self._stats_form = QFormLayout(stats_group)
        self._stat_labels: dict[str, QLabel] = {}
        for key in ["min", "max", "mean", "std"]:
            label = QLabel("—")
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._stat_labels[key] = label
            self._stats_form.addRow(f"{key}:", label)
        layout.addWidget(stats_group)

        # ── Probe values ──
        probe_group = QGroupBox("Probe Point")
        probe_form = QFormLayout(probe_group)
        self._probe_coord = QLabel("Click in 3D view")
        self._probe_coord.setWordWrap(True)
        probe_form.addRow("XYZ:", self._probe_coord)

        self._probe_values = QLabel("—")
        self._probe_values.setWordWrap(True)
        self._probe_values.setTextInteractionFlags(Qt.TextSelectableByMouse)
        probe_form.addRow("Values:", self._probe_values)
        layout.addWidget(probe_group)

        # ── Rock class ──
        class_group = QGroupBox("Rock Mass Class")
        class_form = QFormLayout(class_group)
        self._class_label = QLabel("—")
        self._class_label.setFont(QFont("", 14, QFont.Bold))
        class_form.addRow("BQ:", self._class_label)
        layout.addWidget(class_group)

        layout.addStretch()

    def set_model(self, model, field_name: str):
        self._model = model
        self._update_stats(field_name)

    def set_probe(self, x: float, y: float, z: float, values: dict):
        self._probe_coord.setText(f"({x:.1f}, {y:.1f}, {z:.1f})")
        lines = []
        for name, val in values.items():
            if name in ("BQ_Class",):
                continue
            if abs(val) > 100:
                lines.append(f"  {name}: {val:.1f}")
            elif abs(val) < 0.01:
                lines.append(f"  {name}: {val:.4f}")
            else:
                lines.append(f"  {name}: {val:.3f}")
        self._probe_values.setText("\n".join(lines))

        # Rock class display
        if "BQ_Class" in values:
            cls_map = {1: "I — Very Good", 2: "II — Good", 3: "III — Fair",
                       4: "IV — Poor", 5: "V — Very Poor"}
            cls = int(values["BQ_Class"])
            label = cls_map.get(cls, f"Class {cls}")
            colors = {1: "#4CAF50", 2: "#8BC34A", 3: "#FFEB3B",
                      4: "#FF9800", 5: "#F44336"}
            self._class_label.setText(label)
            self._class_label.setStyleSheet(f"color: {colors.get(cls, '#999')};")

    def _update_stats(self, field_name: str):
        if self._model is None:
            return
        try:
            stats = self._model.stats(field_name)
        except KeyError:
            return
        for key, label in self._stat_labels.items():
            val = stats.get(key, float("nan"))
            if abs(val) > 100:
                label.setText(f"{val:.1f}")
            elif abs(val) < 0.01:
                label.setText(f"{val:.4f}")
            else:
                label.setText(f"{val:.3f}")
