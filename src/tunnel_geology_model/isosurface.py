"""Isosurface extraction via marching cubes."""

from __future__ import annotations

import numpy as np
from skimage.measure import marching_cubes


def marching_cubes_isosurface(
    field_3d: np.ndarray,
    iso_value: float,
    spatial_step: float | tuple[float, float, float] = 1.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    spacing: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract isosurface from a 3D scalar field using marching cubes.

    Parameters
    ----------
    field_3d : np.ndarray
        3D array of shape (ny, nx, nz).
    iso_value : float
        Isosurface value threshold.
    spatial_step : float or (dx, dy, dz)
        Voxel spacing in physical units.
    origin : (x0, y0, z0)
        Coordinate origin of the field.
    spacing : (dx, dy, dz), optional
        Deprecated alias for spatial_step.

    Returns
    -------
    vertices : np.ndarray (N, 3)
        Vertex coordinates in physical space.
    faces : np.ndarray (M, 3)
        Triangle indices.
    normals : np.ndarray (N, 3)
        Vertex normals.
    values : np.ndarray (N,)
        Interpolated field values at vertices.
    """
    if spacing is not None:
        spatial_step = spacing

    if isinstance(spatial_step, (int, float)):
        sx = sy = sz = float(spatial_step)
    else:
        sx, sy, sz = spatial_step

    # marching_cubes expects array indexed as (z, y, x) or (rows, cols, slices)?
    # skimage: expects (z, y, x) for 3D medical imaging convention
    # Our convention: (y, x, z) → need to transpose to (z, y, x)
    volume = np.asarray(field_3d, dtype=np.float32)
    volume = np.transpose(volume, (2, 0, 1))  # (ny, nx, nz) → (nz, ny, nx)

    # skimage >= 0.20 returns (verts, faces, normals, values) — 4-tuple
    # spacing must be applied after extraction
    result = marching_cubes(volume, level=iso_value, allow_degenerate=False)
    if len(result) == 4:
        verts_z, verts_y, verts_x, faces, normals, values = (
            result[0][:, 0], result[0][:, 1], result[0][:, 2],
            result[1], result[2], result[3],
        )
    else:
        verts_z, verts_y, verts_x, faces, normals, values = result

    # Apply spacing and origin
    verts_x = verts_x * sx + origin[0]
    verts_y = verts_y * sy + origin[1]
    verts_z = verts_z * sz + origin[2]

    vertices = np.column_stack([verts_x, verts_y, verts_z]).astype(np.float32)

    return vertices, faces, normals.astype(np.float32), values.astype(np.float32)


def multi_isosurface(
    field_3d: np.ndarray,
    iso_values: list[float],
    spatial_step: float | tuple[float, float, float] = 1.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Extract multiple isosurfaces at different thresholds.

    Returns a list of (vertices, faces, normals, values) per isovalue.
    """
    results = []
    for v in iso_values:
        results.append(
            marching_cubes_isosurface(field_3d, v, spatial_step, origin)
        )
    return results
