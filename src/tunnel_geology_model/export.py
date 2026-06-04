"""Export GeologicalModel to standard 3D formats — VTK, OBJ, NumPy."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .model import GeologicalModel


def to_vtk_structured_grid(
    model: "GeologicalModel",
    output_path: str | Path,
    fields: list[str] | None = None,
) -> Path:
    """Export the full structured grid to a VTK .vts file.

    Uses pyvista to create a StructuredGrid with all requested fields as
    point data arrays.

    Parameters
    ----------
    model : GeologicalModel
    output_path : str or Path
        Output .vts or .vtk file path.
    fields : list[str] or None
        Field names to export. If None, exports all fields.

    Returns
    -------
    Path to the written file.
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista is required for VTK export. pip install pyvista")

    output_path = Path(output_path)

    # Create structured grid (X, Y, Z as 3D coordinates)
    xx, yy, zz = np.meshgrid(model.x, model.y, model.z, indexing="xy")
    # meshgrid(x, y) outputs (ny, nx, nz) — correct for VTK structured grid

    grid = pv.StructuredGrid(xx, yy, zz)

    field_names = fields if fields is not None else model.field_names
    for name in field_names:
        arr = model.fields[name].ravel(order="F")  # VTK expects Fortran order
        grid.point_data[name] = arr

    # Also export classification fields
    for name, arr in model.classification.items():
        grid.point_data[name] = arr.ravel(order="F")

    if output_path.suffix == ".vts":
        grid.save(str(output_path), binary=True)
    else:
        grid.save(str(output_path))

    return output_path


def to_vtk_rectilinear_grid(
    model: "GeologicalModel",
    output_path: str | Path,
    fields: list[str] | None = None,
) -> Path:
    """Export as a VTK RectilinearGrid (.vtr) — more compact for regular grids.

    Parameters
    ----------
    model : GeologicalModel
    output_path : str or Path
        Output .vtr file path.
    fields : list[str] or None

    Returns
    -------
    Path
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista is required for VTK export. pip install pyvista")

    output_path = Path(output_path)

    grid = pv.RectilinearGrid(model.x, model.y, model.z)

    field_names = fields if fields is not None else model.field_names
    for name in field_names:
        arr = model.fields[name].ravel(order="F")
        grid.point_data[name] = arr

    for name, arr in model.classification.items():
        grid.point_data[name] = arr.ravel(order="F")

    grid.save(str(output_path), binary=True)
    return output_path


def to_numpy_dict(
    model: "GeologicalModel",
    output_dir: str | Path,
) -> Path:
    """Save all fields to a .npz compressed archive.

    Parameters
    ----------
    model : GeologicalModel
    output_dir : str or Path
        Directory to write the .npz file.

    Returns
    -------
    Path to the .npz file.
    """
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
    np.savez_compressed(str(output_path), **save_dict)
    return output_path


def to_xyz_point_cloud(
    model: "GeologicalModel",
    output_path: str | Path,
    fields: list[str] | None = None,
    sample: int | float | None = None,
) -> Path:
    """Export sub-sampled XYZ point cloud CSV.

    Useful for loading into general-purpose 3D tools.

    Parameters
    ----------
    model : GeologicalModel
    output_path : str or Path
    fields : list[str] or None
        Fields to include as columns.
    sample : int, float, or None
        If int: sample every N grid points.
        If float (0-1): fraction of points to keep.
        If None: export all points.

    Returns
    -------
    Path
    """
    output_path = Path(output_path)
    xx, yy, zz = np.meshgrid(model.x, model.y, model.z, indexing="xy")

    field_names = fields if fields is not None else model.field_names
    columns = [xx.ravel(), yy.ravel(), zz.ravel()]
    header = "X,Y,Z"
    for name in field_names:
        columns.append(model.fields[name].ravel())
        header += f",{name}"

    data = np.column_stack(columns)

    if sample is not None:
        if isinstance(sample, float) and 0 < sample < 1:
            n = int(len(data) * sample)
        else:
            n = int(sample)
        n = max(1, min(n, len(data)))
        indices = np.random.default_rng(42).choice(len(data), size=n, replace=False)
        data = data[indices]

    fmt = "%.3f,%.3f,%.3f" + ",%.4f" * len(field_names)
    np.savetxt(str(output_path), data, header=header, delimiter=",", fmt=fmt, comments="")
    return output_path
