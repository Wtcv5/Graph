"""Info/Log panel — scrolling text log."""

from __future__ import annotations

import time

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit
from PySide6.QtCore import Qt


class InfoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumHeight(120)
        self._text.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        layout.addWidget(self._text)

    def log(self, message: str):
        ts = time.strftime("%H:%M:%S")
        self._text.append(f"[{ts}] {message}")
