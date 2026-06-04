"""Cross-section extraction from 3D regular-grid GeologicalModel."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def extract_yz_section(
    model,
    x_value: float,
    field_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Extract a Y-Z slice at a given X coordinate (tunnel-longitudinal section).

    Parameters
    ----------
    model : GeologicalModel
    x_value : float
        X coordinate at which to slice.
    field_names : list[str] or None
        Fields to extract. If None, extracts all fields.

    Returns
    -------
    dict[str, np.ndarray]
        {'field_name': 2D array (ny, nz)}
    """
    names = _resolve_names(model, field_names)
    ix = _nearest_index(model.x, x_value)
    return {n: model.fields[n][:, ix, :].copy() for n in names}


def extract_xz_section(
    model,
    y_value: float,
    field_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Extract a X-Z slice at a given Y coordinate (tunnel cross-section).

    This is the most commonly used section for tunnel engineering — it shows
    the geology perpendicular to the tunnel alignment.

    Parameters
    ----------
    model : GeologicalModel
    y_value : float
        Y coordinate at which to slice.
    field_names : list[str] or None
        Fields to extract.

    Returns
    -------
    dict[str, np.ndarray]
        {'field_name': 2D array (nx, nz)}
    """
    names = _resolve_names(model, field_names)
    iy = _nearest_index(model.y, y_value)
    return {n: model.fields[n][iy, :, :].copy() for n in names}


def extract_xy_section(
    model,
    z_value: float,
    field_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Extract a X-Y slice at a given Z coordinate (horizontal plan view).

    Parameters
    ----------
    model : GeologicalModel
    z_value : float
        Z coordinate at which to slice.
    field_names : list[str] or None
        Fields to extract.

    Returns
    -------
    dict[str, np.ndarray]
        {'field_name': 2D array (nx, ny)}  — note transposed from grid order
    """
    names = _resolve_names(model, field_names)
    iz = _nearest_index(model.z, z_value)
    # fields are (ny, nx, nz) → slice at z → (ny, nx) → transpose for (nx, ny)
    return {n: model.fields[n][:, :, iz].T.copy() for n in names}


def extract_arbitrary_section(
    model,
    origin: tuple[float, float, float],
    normal: tuple[float, float, float],
    resolution: float | None = None,
    field_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Extract an arbitrary plane section using linear interpolation.

    Parameters
    ----------
    model : GeologicalModel
    origin : (x0, y0, z0)
        A point on the cutting plane.
    normal : (nx, ny, nz)
        Normal vector of the cutting plane.
    resolution : float, optional
        Grid resolution for the output plane (default: model.grid_step).
    field_names : list[str] or None

    Returns
    -------
    dict[str, np.ndarray]
        Each field as a 2D array on the plane. Also includes 'plane_x', 'plane_y',
        'plane_z' coordinate arrays for the sampling grid.
    """
    step = resolution or model.grid_step or 0.5

    # Build a local 2D grid on the plane
    n = np.asarray(normal, dtype=np.float64)
    n_norm = n / np.linalg.norm(n)
    origin = np.asarray(origin, dtype=np.float64)

    # Pick two orthogonal in-plane axes
    if np.abs(n_norm[2]) < 0.9:
        u = np.cross(n_norm, [0, 0, 1])
    else:
        u = np.cross(n_norm, [1, 0, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(n_norm, u)

    # Bounding box in-plane distance
    corners = np.array([
        [model.x[0], model.y[0], model.z[0]],
        [model.x[0], model.y[0], model.z[-1]],
        [model.x[0], model.y[-1], model.z[0]],
        [model.x[0], model.y[-1], model.z[-1]],
        [model.x[-1], model.y[0], model.z[0]],
        [model.x[-1], model.y[0], model.z[-1]],
        [model.x[-1], model.y[-1], model.z[0]],
        [model.x[-1], model.y[-1], model.z[-1]],
    ])
    proj_u = (corners - origin) @ u
    proj_v = (corners - origin) @ v

    u_vals = np.arange(proj_u.min(), proj_u.max() + step, step, dtype=np.float64)
    v_vals = np.arange(proj_v.min(), proj_v.max() + step, step, dtype=np.float64)
    uu, vv = np.meshgrid(u_vals, v_vals)

    # Convert plane coords to world coords
    pts = origin + uu[..., None] * u + vv[..., None] * v  # (nv, nu, 3)
    xq = pts[..., 0].ravel()
    yq = pts[..., 1].ravel()
    zq = pts[..., 2].ravel()

    names = _resolve_names(model, field_names)
    result: dict[str, np.ndarray] = {
        "plane_u": uu.astype(np.float32),
        "plane_v": vv.astype(np.float32),
        "plane_x": pts[..., 0].astype(np.float32),
        "plane_y": pts[..., 1].astype(np.float32),
        "plane_z": pts[..., 2].astype(np.float32),
    }

    for name in names:
        interp = RegularGridInterpolator(
            (model.y, model.x, model.z),
            model.fields[name].astype(np.float64),
            bounds_error=False,
            fill_value=np.nan,
        )
        # query expects (ny, nx, nz) → reorder: our coords are (x, y, z)
        sampled = interp(np.column_stack([yq, xq, zq]))
        result[name] = sampled.reshape(uu.shape).astype(np.float32)

    return result


# ── helpers ────────────────────────────────────────────────

def _resolve_names(model, field_names: list[str] | None) -> list[str]:
    if field_names is None:
        return model.field_names
    for n in field_names:
        if n not in model.field_names:
            raise KeyError(f"Field '{n}' not found in model. Available: {model.field_names}")
    return field_names


def _nearest_index(coords: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(coords - value)))
