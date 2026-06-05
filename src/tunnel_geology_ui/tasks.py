"""Small background task helpers for GUI responsiveness."""

from __future__ import annotations

import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Signal


class WorkerSignals(QObject):
    progress = Signal(int, str)
    result = Signal(object)
    error = Signal(str, str)
    finished = Signal()


class ProgressReporter:
    def __init__(self, signals: WorkerSignals):
        self._signals = signals

    def __call__(self, value: int, message: str) -> None:
        self._signals.progress.emit(int(value), message)


class FunctionWorker(QRunnable):
    """Run a Python callable in Qt's thread pool."""

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:
        reporter = ProgressReporter(self.signals)
        try:
            result = self._fn(reporter, *self._args, **self._kwargs)
        except Exception as exc:  # pragma: no cover - GUI runtime path
            self.signals.error.emit(str(exc), traceback.format_exc())
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
