"""Export dialog — choose format and options."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox,
    QDoubleSpinBox, QSpinBox, QDialogButtonBox, QGroupBox,
)


class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Options")

        layout = QVBoxLayout(self)

        group = QGroupBox("Export Settings")
        form = QFormLayout(group)

        self._format = QComboBox()
        self._format.addItems(["VTK Structured (.vts)", "VTK Rectilinear (.vtr)", "XYZ Point Cloud (.csv)"])
        form.addRow("Format:", self._format)

        self._sample = QDoubleSpinBox()
        self._sample.setRange(0.01, 1.0)
        self._sample.setValue(0.1)
        self._sample.setSingleStep(0.05)
        self._sample.setDecimals(2)
        form.addRow("Sample Fraction:", self._sample)

        layout.addWidget(group)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
