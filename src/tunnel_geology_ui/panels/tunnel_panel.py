"""Tunnel panel — parametric tunnel builder and geology coupling."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLabel, QGroupBox, QComboBox,
    QDoubleSpinBox, QTextEdit, QMessageBox,
)
from PySide6.QtCore import Signal


class TunnelPanel(QWidget):
    tunnel_built = Signal(object)   # ParametricTunnel
    coupling_computed = Signal(object)  # TunnelGeologyCoupling
    coupling_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = None
        self._tunnel = None
        self._coupling = None

        layout = QVBoxLayout(self)

        # ── Alignment ──
        align_group = QGroupBox("Tunnel Alignment")
        align_form = QFormLayout(align_group)

        self._x_start = QDoubleSpinBox()
        self._x_start.setRange(-20000, 0)
        self._x_start.setValue(-12310)
        self._x_start.setDecimals(1)
        align_form.addRow("X start (m):", self._x_start)

        self._y_start = QDoubleSpinBox()
        self._y_start.setRange(-100, 100)
        self._y_start.setValue(0)
        self._y_start.setDecimals(1)
        align_form.addRow("Y start (m):", self._y_start)

        self._z_start = QDoubleSpinBox()
        self._z_start.setRange(2000, 4000)
        self._z_start.setValue(3000)
        self._z_start.setDecimals(1)
        align_form.addRow("Z start (m):", self._z_start)

        self._length = QDoubleSpinBox()
        self._length.setRange(10, 500)
        self._length.setValue(180)
        self._length.setDecimals(1)
        align_form.addRow("Length (m):", self._length)

        self._gradient = QDoubleSpinBox()
        self._gradient.setRange(-10, 10)
        self._gradient.setValue(0.5)
        self._gradient.setDecimals(1)
        self._gradient.setSingleStep(0.1)
        align_form.addRow("Gradient (%):", self._gradient)

        layout.addWidget(align_group)

        # ── Cross-section ──
        section_group = QGroupBox("Cross Section")
        section_form = QFormLayout(section_group)

        self._shape = QComboBox()
        self._shape.addItems(["horseshoe", "circular", "rectangular"])
        section_form.addRow("Shape:", self._shape)

        self._width = QDoubleSpinBox()
        self._width.setRange(2, 30)
        self._width.setValue(12)
        self._width.setDecimals(1)
        section_form.addRow("Width (m):", self._width)

        self._height = QDoubleSpinBox()
        self._height.setRange(2, 30)
        self._height.setValue(10)
        self._height.setDecimals(1)
        section_form.addRow("Height (m):", self._height)

        self._shotcrete = QDoubleSpinBox()
        self._shotcrete.setRange(0, 1)
        self._shotcrete.setValue(0.15)
        self._shotcrete.setDecimals(3)
        self._shotcrete.setSingleStep(0.05)
        section_form.addRow("Shotcrete (m):", self._shotcrete)

        self._lining = QDoubleSpinBox()
        self._lining.setRange(0, 2)
        self._lining.setValue(0.35)
        self._lining.setDecimals(3)
        self._lining.setSingleStep(0.05)
        section_form.addRow("Lining (m):", self._lining)

        layout.addWidget(section_group)

        # ── Actions ──
        btn_layout = QHBoxLayout()
        self._build_btn = QPushButton("🔨 Build Tunnel")
        self._build_btn.clicked.connect(self._build_tunnel)
        self._build_btn.setEnabled(False)
        btn_layout.addWidget(self._build_btn)

        self._couple_btn = QPushButton("🔗 Couple with Geology")
        self._couple_btn.clicked.connect(self._compute_coupling)
        self._couple_btn.setEnabled(False)
        btn_layout.addWidget(self._couple_btn)
        layout.addLayout(btn_layout)

        # ── Results ──
        result_group = QGroupBox("Results")
        result_layout = QVBoxLayout(result_group)
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setMaximumHeight(150)
        self._result_text.setStyleSheet("font-family: Consolas, monospace; font-size: 10px;")
        result_layout.addWidget(self._result_text)
        layout.addWidget(result_group)

    def set_model(self, model):
        self._model = model
        self._build_btn.setEnabled(model is not None)
        self._couple_btn.setEnabled(False)
        self._tunnel = None
        self._coupling = None
        self._result_text.clear()

        # Auto-configure defaults from model
        if model:
            self._x_start.setValue(model.x_range[0] + 50)
            self._z_start.setValue(model.z_range[0] +
                                   0.5 * (model.z_range[1] - model.z_range[0]))

    def set_busy(self, busy: bool, message: str | None = None):
        parameter_controls = [
            self._x_start,
            self._y_start,
            self._z_start,
            self._length,
            self._gradient,
            self._shape,
            self._width,
            self._height,
            self._shotcrete,
            self._lining,
        ]
        for control in parameter_controls:
            control.setEnabled(not busy)

        if busy:
            self._build_btn.setEnabled(False)
            self._couple_btn.setEnabled(False)
        else:
            self._build_btn.setEnabled(self._model is not None)
            self._couple_btn.setEnabled(self._tunnel is not None)

        if busy and message:
            self._result_text.setText(message)

    def set_coupling_result(self, coupling, segments: list[dict], warnings: list[str] | None = None):
        self._coupling = coupling
        lines = [self._coupling.summary(), "", "--- Chainage Segments ---"]
        for seg in segments:
            if not seg.get("has_data", False):
                lines.append(
                    f"  [{seg['chainage_start']:.0f}-{seg['chainage_end']:.0f}m] "
                    f"No geological data sampled"
                )
                continue

            cls_value = seg.get("BQ_class")
            cls_label = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}.get(cls_value, "?")
            lines.append(
                f"  [{seg['chainage_start']:.0f}-{seg['chainage_end']:.0f}m] "
                f"BQ={seg.get('mean_BQ', float('nan')):.0f} ({cls_label})  "
                f"Vp={seg.get('mean_Vp', float('nan')):.0f}"
            )

        if warnings:
            lines.extend(["", "--- Warnings ---", *[f"  - {warning}" for warning in warnings]])

        self._result_text.setText("\n".join(lines))
        self.coupling_computed.emit(self._coupling)

    def _validate_alignment_inputs(self) -> str | None:
        if self._model is None:
            return "Load a geological model before building a tunnel."

        x_end = self._x_start.value() + self._length.value()
        if not (self._model.x_range[0] <= self._x_start.value() <= self._model.x_range[1]):
            return "Tunnel start X is outside the loaded model extent."
        if not (self._model.x_range[0] <= x_end <= self._model.x_range[1]):
            return "Tunnel end X is outside the loaded model extent. Reduce length or move the start point."
        if not (self._model.y_range[0] <= self._y_start.value() <= self._model.y_range[1]):
            return "Tunnel Y is outside the loaded model extent."
        if not (self._model.z_range[0] <= self._z_start.value() <= self._model.z_range[1]):
            return "Tunnel Z is outside the loaded model extent."
        return None

    def _build_tunnel(self):
        if self._model is None:
            return

        validation_error = self._validate_alignment_inputs()
        if validation_error:
            QMessageBox.warning(self, "Invalid Tunnel Parameters", validation_error)
            return

        try:
            from tunnel_geology_model.tunnel import (
                ParametricTunnel, TunnelAlignment, TunnelSection, SectionShape,
            )

            x_end = self._x_start.value() + self._length.value()

            alignment = TunnelAlignment.from_endpoints(
                x_start=self._x_start.value(),
                y_start=self._y_start.value(),
                z_start=self._z_start.value(),
                x_end=x_end,
                y_end=self._y_start.value(),
                z_end=self._z_start.value(),
                gradient_pct=self._gradient.value(),
                n_samples=100,
            )

            section = TunnelSection(
                shape=SectionShape(self._shape.currentText()),
                width=self._width.value(),
                height=self._height.value(),
                shotcrete_thickness=self._shotcrete.value(),
                lining_thickness=self._lining.value(),
                n_vertices=48,
            )

            self._tunnel = ParametricTunnel(
                alignment=alignment,
                section=section,
                name=f"Tunnel_{self._x_start.value():.0f}",
            )
            self._tunnel.build_mesh(include_layers=False)

            self._result_text.setText(self._tunnel.summary())
            self._couple_btn.setEnabled(True)
            self.tunnel_built.emit(self._tunnel)

        except Exception as e:
            QMessageBox.critical(self, "Build Error", str(e))

    def _compute_coupling(self):
        if self._tunnel is None or self._model is None:
            return
        self.coupling_requested.emit(self._tunnel)
