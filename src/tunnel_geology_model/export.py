"""Export GeologicalModel to standard 3D formats — VTK, NumPy, XYZ.

All write functions use atomic temp-file-then-rename to avoid partial writes.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .model import GeologicalModel


def _validate_export_path(path: Path) -> Path:
    """Ensure output directory exists. Returns resolved path."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_atomic(path: Path, writer) -> Path:
    """Write to a temp file then atomically rename to `path`.

    The `writer` callable receives the temp Path and writes content there.
    On success the temp file replaces the target. On failure the temp is removed.
    """
    path = _validate_export_path(path)
    tmp_dir = path.parent
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(tmp_dir), prefix=".tmp_", suffix=path.suffix)
    os.close(tmp_fd)
    tmp = Path(tmp_path)
    try:
        writer(tmp)
        if path.exists():
            path.unlink()
        tmp.rename(path)
        return path
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _check_pyvista():
    try:
        import pyvista  # noqa: F401
    except ImportError:
        raise ImportError(
            "pyvista is required for VTK export. Install with: pip install pyvista"
        )


# ── VTK exports ────────────────────────────────────────────

def to_vtk_structured_grid(
    model: "GeologicalModel",
    output_path: str | Path,
    fields: list[str] | None = None,
) -> Path:
    """Export as VTK StructuredGrid (.vts)."""
    _check_pyvista()
    import pyvista as pv

    field_names = fields if fields is not None else model.field_names

    def _write(dest: Path):
        xx, yy, zz = np.meshgrid(model.x, model.y, model.z, indexing="xy")
        grid = pv.StructuredGrid(xx, yy, zz)
        for name in field_names:
            grid.point_data[name] = model.field(name).ravel(order="F")
        for name, arr in model.classification.items():
            grid.point_data[name] = arr.ravel(order="F")
        grid.save(str(dest), binary=True)

    return _write_atomic(Path(output_path), _write)


def to_vtk_rectilinear_grid(
    model: "GeologicalModel",
    output_path: str | Path,
    fields: list[str] | None = None,
) -> Path:
    """Export as VTK RectilinearGrid (.vtr) — more compact for regular grids."""
    _check_pyvista()
    import pyvista as pv

    field_names = fields if fields is not None else model.field_names

    def _write(dest: Path):
        grid = pv.RectilinearGrid(model.x, model.y, model.z)
        for name in field_names:
            grid.point_data[name] = model.field(name).ravel(order="F")
        for name, arr in model.classification.items():
            grid.point_data[name] = arr.ravel(order="F")
        grid.save(str(dest), binary=True)

    return _write_atomic(Path(output_path), _write)


# ── NumPy export ───────────────────────────────────────────

def to_numpy_dict(
    model: "GeologicalModel",
    output_dir: str | Path,
) -> Path:
    """Save all fields to a .npz compressed archive."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "geological_model.npz"

    save_dict = {
        "x": model.x.astype(np.float32),
        "y": model.y.astype(np.float32),
        "z": model.z.astype(np.float32),
        **{f"field_{k}": v.astype(np.float32) for k, v in model.fields.items()},
        **{f"class_{k}": v.astype(np.float32) for k, v in model.classification.items()},
    }

    def _write(dest: Path):
        np.savez_compressed(str(dest), **save_dict)

    return _write_atomic(output_path, _write)


# ── XYZ point cloud ────────────────────────────────────────

def to_xyz_point_cloud(
    model: "GeologicalModel",
    output_path: str | Path,
    fields: list[str] | None = None,
    sample: int | float | None = None,
) -> Path:
    """Export sub-sampled XYZ point cloud CSV."""
    output_path = Path(output_path)
    field_names = fields if fields is not None else model.field_names

    xx, yy, zz = np.meshgrid(model.x, model.y, model.z, indexing="xy")
    columns = [xx.ravel(), yy.ravel(), zz.ravel()]
    header = "X,Y,Z"
    for name in field_names:
        columns.append(model.field(name).ravel())
        header += f",{name}"

    data = np.column_stack(columns)

    if sample is not None:
        if isinstance(sample, float) and 0 < sample < 1:
            n = int(len(data) * sample)
        elif isinstance(sample, (int, float)) and sample >= 1:
            n = int(sample)
        else:
            n = len(data)
        n = max(1, min(n, len(data)))
        indices = np.random.default_rng(42).choice(len(data), size=n, replace=False)
        data = data[indices]

    fmt = "%.3f,%.3f,%.3f" + ",%.4f" * len(field_names)

    def _write(dest: Path):
        np.savetxt(str(dest), data, header=header, delimiter=",", fmt=fmt, comments="")

    return _write_atomic(output_path, _write)
