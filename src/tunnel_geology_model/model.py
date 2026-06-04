"""GeologicalModel — 3D regular-grid data structure for tunnel geology."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr


@dataclass
class GeologicalModel:
    """Core data structure holding a 3D regular grid with multiple property fields.

    Dimensions are stored in (Y, X, Z) order to match NetCDF convention.
    Each field is a 3D numpy array of shape (ny, nx, nz).

    Attributes
    ----------
    x, y, z : np.ndarray
        Coordinate vectors for each axis (1D).
    fields : dict[str, np.ndarray]
        Named 3D property arrays, e.g. {'Vp': ..., 'Vs': ..., 'Poisson_Ratio': ...}.
    metadata : dict
        Arbitrary key-value metadata (source, grid_step, coordinate_system, etc.).
    masks : dict[str, np.ndarray]
        Optional boolean masks (e.g. tunnel_excavation, fault_zone).
    classification : dict[str, np.ndarray]
        Rock-mass classification result arrays (e.g. 'BQ_class', 'RMR').
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    classification: dict[str, np.ndarray] = field(default_factory=dict)

    # ── properties ──────────────────────────────────────────

    @property
    def nx(self) -> int:
        return len(self.x)

    @property
    def ny(self) -> int:
        return len(self.y)

    @property
    def nz(self) -> int:
        return len(self.z)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.ny, self.nx, self.nz)

    @property
    def grid_step(self) -> float:
        return float(self.metadata.get("grid_step_m", 0.0))

    @property
    def field_names(self) -> list[str]:
        return list(self.fields.keys())

    # ── bounds ──────────────────────────────────────────────

    @property
    def x_range(self) -> tuple[float, float]:
        return (float(self.x[0]), float(self.x[-1]))

    @property
    def y_range(self) -> tuple[float, float]:
        return (float(self.y[0]), float(self.y[-1]))

    @property
    def z_range(self) -> tuple[float, float]:
        return (float(self.z[0]), float(self.z[-1]))

    # ── access ──────────────────────────────────────────────

    def __getitem__(self, name: str) -> np.ndarray:
        """Get a field, mask, or classification by name."""
        if name in self.fields:
            return self.fields[name]
        if name in self.masks:
            return self.masks[name]
        if name in self.classification:
            return self.classification[name]
        raise KeyError(f"'{name}' not found in fields, masks, or classification")

    def __contains__(self, name: str) -> bool:
        return name in self.fields or name in self.masks or name in self.classification

    def field(self, name: str) -> np.ndarray:
        """Return field array, guaranteed."""
        arr = self.fields.get(name)
        if arr is None:
            raise KeyError(f"Field '{name}' not found. Available: {list(self.fields)}")
        return arr

    # ── basic stats ─────────────────────────────────────────

    def stats(self, name: str) -> dict[str, float]:
        """Return min, max, mean, std for a named field/mask/class."""
        arr = self[name]
        return {
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "nan_count": int(np.isnan(arr).sum()),
        }  # type: ignore[assignment]

    def summary(self) -> str:
        lines = [
            f"GeologicalModel  {self.nx} x {self.ny} x {self.nz}  ({self.shape})",
            f"  X: [{self.x_range[0]:.1f}, {self.x_range[1]:.1f}] m  dx={self.grid_step:.1f}m",
            f"  Y: [{self.y_range[0]:.1f}, {self.y_range[1]:.1f}] m  dy={self.grid_step:.1f}m",
            f"  Z: [{self.z_range[0]:.1f}, {self.z_range[1]:.1f}] m  dz={self.grid_step:.1f}m",
        ]
        if self.fields:
            lines.append(f"  Fields: {', '.join(self.fields)}")
        if self.masks:
            lines.append(f"  Masks:  {', '.join(self.masks)}")
        if self.classification:
            lines.append(f"  Class:  {', '.join(self.classification)}")
        return "\n".join(lines)

    # ── subsetting ──────────────────────────────────────────

    def subset_x(self, x_min: float, x_max: float) -> "GeologicalModel":
        return self._subset("x", x_min, x_max)

    def subset_y(self, y_min: float, y_max: float) -> "GeologicalModel":
        return self._subset("y", y_min, y_max)

    def subset_z(self, z_min: float, z_max: float) -> "GeologicalModel":
        return self._subset("z", z_min, z_max)

    def _subset(self, axis: str, lo: float, hi: float) -> "GeologicalModel":
        coords = getattr(self, axis)
        mask = (coords >= lo) & (coords <= hi)
        indices = np.where(mask)[0]
        sl = slice(int(indices[0]), int(indices[-1]) + 1)

        def _slice(arr):
            if axis == "x":
                return arr[:, sl, :]
            elif axis == "y":
                return arr[sl, :, :]
            else:
                return arr[:, :, sl]

        return GeologicalModel(
            x=self.x if axis != "x" else self.x[sl],
            y=self.y if axis != "y" else self.y[sl],
            z=self.z if axis != "z" else self.z[sl],
            fields={k: _slice(v) for k, v in self.fields.items()},
            metadata=dict(self.metadata),
            masks={k: _slice(v) for k, v in self.masks.items()},
            classification={k: _slice(v) for k, v in self.classification.items()},
        )


# ── I/O ────────────────────────────────────────────────────

def load_from_netcdf(path: str | Path, engine: str = "h5netcdf") -> GeologicalModel:
    """Load a GeologicalModel from a NetCDF file produced by the preprocessing pipeline.

    Parameters
    ----------
    path : str or Path
        Path to the NetCDF file.
    engine : str
        xarray backend engine (default 'h5netcdf' for Unicode path support on Windows).

    Returns
    -------
    GeologicalModel
    """
    ds = xr.open_dataset(str(path), engine=engine)

    # ── coords ──
    x = ds["X"].values.astype(np.float64)
    y = ds["Y"].values.astype(np.float64)
    z = ds["Z"].values.astype(np.float64)

    # ── classify variables ──
    count_vars = {"Vp_count", "Vs_count"}
    classification_keys = {
        "Poisson_Ratio", "Young_Modulus", "Shear_Modulus",
        "Bulk_Modulus", "Lame_Lambda",
    }
    mask_keys = set()

    fields: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    classification: dict[str, np.ndarray] = {}

    for var_name in ds.data_vars:
        arr = ds[var_name].values
        if var_name in count_vars:
            continue  # skip raw counts
        if var_name in classification_keys:
            classification[var_name] = arr
        elif var_name in mask_keys:
            masks[var_name] = arr
        else:
            fields[var_name] = arr

    metadata = {str(k): _serialize_attr(v) for k, v in ds.attrs.items()}
    ds.close()

    return GeologicalModel(
        x=x, y=y, z=z,
        fields=fields,
        metadata=metadata,
        masks=masks,
        classification=classification,
    )


def _serialize_attr(value) -> str | float | int:
    if isinstance(value, (str, float, int, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
