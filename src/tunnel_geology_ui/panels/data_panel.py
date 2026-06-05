"""Data panel — dataset loading and field selection."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QFileDialog, QGroupBox,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont


class DataPanel(QWidget):
    dataset_loaded = Signal(Path)
    field_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = None

        layout = QVBoxLayout(self)

        # ── Load button ──
        self._load_btn = QPushButton("📂 Open NetCDF...")
        self._load_btn.clicked.connect(self._open_file)
        layout.addWidget(self._load_btn)

        # ── Dataset info ──
        self._info_label = QLabel("No dataset loaded")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #888; padding: 4px;")
        layout.addWidget(self._info_label)

        # ── Field selector ──
        fields_group = QGroupBox("Fields")
        fields_layout = QVBoxLayout(fields_group)
        self._field_list = QListWidget()
        self._field_list.currentTextChanged.connect(self.field_selected.emit)
        fields_layout.addWidget(self._field_list)
        layout.addWidget(fields_group)

        # ── Classification ──
        class_group = QGroupBox("Classification")
        class_layout = QVBoxLayout(class_group)
        self._class_list = QListWidget()
        self._class_list.currentTextChanged.connect(self.field_selected.emit)
        class_layout.addWidget(self._class_list)
        layout.addWidget(class_group)

        layout.addStretch()

    def set_model(self, model):
        self._model = model
        self._info_label.setText(
            f"{model.nx} × {model.ny} × {model.nz}\n"
            f"dx={model.grid_step:.1f}m\n"
            f"Z: [{model.z_range[0]:.0f}, {model.z_range[1]:.0f}]"
        )

        self._field_list.clear()
        for name in model.field_names:
            item = QListWidgetItem(name)
            font = QFont()
            font.setBold(True)
            item.setFont(font)
            self._field_list.addItem(item)

        self._class_list.clear()
        for name in model.classification:
            self._class_list.addItem(name)

        if model.field_names:
            self._field_list.setCurrentRow(0)

    def set_busy(self, busy: bool):
        self._load_btn.setEnabled(not busy)
        self._field_list.setEnabled(not busy)
        self._class_list.setEnabled(not busy)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open NetCDF Dataset", "",
            "NetCDF Files (*.nc);;All Files (*)"
        )
        if path:
            self.dataset_loaded.emit(Path(path))
