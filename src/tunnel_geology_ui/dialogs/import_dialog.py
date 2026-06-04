"""Import dialog — configure a new CSV dataset for preprocessing."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QFileDialog, QDialogButtonBox,
    QSpinBox, QDoubleSpinBox, QLabel, QGroupBox,
)


class ImportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import CSV Dataset")
        self.setMinimumWidth(500)
        self.config_path = None

        layout = QVBoxLayout(self)

        # ── CSV inputs ──
        csv_group = QGroupBox("CSV Files")
        csv_form = QFormLayout(csv_group)

        self._vp_path = QLineEdit()
        self._vp_path.setPlaceholderText("Select Vp CSV...")
        vp_btn = QPushButton("...")
        vp_btn.clicked.connect(lambda: self._browse_file(self._vp_path))
        vp_row = QHBoxLayout()
        vp_row.addWidget(self._vp_path)
        vp_row.addWidget(vp_btn)
        csv_form.addRow("Vp CSV:", vp_row)

        self._vs_path = QLineEdit()
        self._vs_path.setPlaceholderText("Select Vs CSV...")
        vs_btn = QPushButton("...")
        vs_btn.clicked.connect(lambda: self._browse_file(self._vs_path))
        vs_row = QHBoxLayout()
        vs_row.addWidget(self._vs_path)
        vs_row.addWidget(vs_btn)
        csv_form.addRow("Vs CSV:", vs_row)
        layout.addWidget(csv_group)

        # ── Grid params ──
        grid_group = QGroupBox("Grid Parameters")
        grid_form = QFormLayout(grid_group)

        self._grid_step = QDoubleSpinBox()
        self._grid_step.setValue(0.5)
        self._grid_step.setDecimals(2)
        self._grid_step.setSingleStep(0.1)
        grid_form.addRow("Grid Step (m):", self._grid_step)

        self._chunk_size = QSpinBox()
        self._chunk_size.setRange(100_000, 5_000_000)
        self._chunk_size.setValue(500_000)
        self._chunk_size.setSingleStep(100_000)
        grid_form.addRow("Chunk Size:", self._chunk_size)

        self._fill_kernel = QSpinBox()
        self._fill_kernel.setRange(1, 7)
        self._fill_kernel.setValue(3)
        self._fill_kernel.setSingleStep(2)
        grid_form.addRow("Fill Kernel:", self._fill_kernel)

        self._fill_iter = QSpinBox()
        self._fill_iter.setRange(1, 20)
        self._fill_iter.setValue(5)
        grid_form.addRow("Fill Iterations:", self._fill_iter)
        layout.addWidget(grid_group)

        # ── Output ──
        output_group = QGroupBox("Output")
        output_form = QFormLayout(output_group)

        self._output_dir = QLineEdit()
        self._output_dir.setPlaceholderText("Output directory...")
        out_btn = QPushButton("...")
        out_btn.clicked.connect(self._browse_dir)
        out_row = QHBoxLayout()
        out_row.addWidget(self._output_dir)
        out_row.addWidget(out_btn)
        output_form.addRow("Output Dir:", out_row)
        layout.addWidget(output_group)

        # ── Buttons ──
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_file(self, line_edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if path:
            line_edit.setText(path)

    def _browse_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self._output_dir.setText(path)

    def _accept(self):
        # Write a minimal dataset config
        import json

        config = {
            "name": "imported_dataset",
            "description": "Imported CSV dataset",
            "data_dir": str(Path(self._vp_path.text()).parent),
            "vp_file": Path(self._vp_path.text()).name,
            "vs_file": Path(self._vs_path.text()).name,
            "output_dir": self._output_dir.text() or "outputs/imported",
            "grid_step_m": self._grid_step.value(),
            "chunk_size": self._chunk_size.value(),
            "fill_kernel_size": self._fill_kernel.value(),
            "fill_max_iter": self._fill_iter.value(),
            "title": "Tunnel Seismic Velocity Model",
            "source": "Tunnel Advance Prediction",
            "coordinate_system": "Local engineering coordinates (meters)",
            "density_formula": "Gardner: rho = 0.31 * Vp^0.25 (g/cm3)",
        }

        out = Path(self._output_dir.text() or "outputs/imported")
        out.mkdir(parents=True, exist_ok=True)
        config_path = out / "dataset_config.json"
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)

        self.config_path = config_path
        self.accept()
