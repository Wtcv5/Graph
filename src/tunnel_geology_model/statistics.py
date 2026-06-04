"""Geological statistics — histograms, transverse profiles, anomaly detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .model import GeologicalModel


def field_histogram(
    model: "GeologicalModel",
    field_name: str,
    bins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute histogram of a scalar field.

    Returns (counts, bin_edges).
    """
    arr = model.field(field_name)
    counts, edges = np.histogram(arr[~np.isnan(arr)], bins=bins)
    return counts, edges


def depth_profile(
    model: "GeologicalModel",
    field_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean ± std vs depth (Z).

    Returns (z_centers, means, stds).
    """
    arr = model.field(field_name)
    means = np.zeros(model.nz, dtype=np.float64)
    stds = np.zeros(model.nz, dtype=np.float64)

    for iz in range(model.nz):
        slab = arr[:, :, iz]
        valid = slab[~np.isnan(slab)]
        means[iz] = np.mean(valid) if len(valid) > 0 else np.nan
        stds[iz] = np.std(valid) if len(valid) > 0 else np.nan

    return model.z, means, stds


def anomaly_detection(
    model: "GeologicalModel",
    field_name: str,
    n_sigma: float = 2.0,
) -> np.ndarray:
    """Detect statistical anomalies in a field (> n * sigma from mean).

    Returns boolean mask of shape (ny, nx, nz).
    """
    arr = model.field(field_name)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    return np.abs(arr - mean) > n_sigma * std


def correlation_map(
    model: "GeologicalModel",
    field_a: str,
    field_b: str,
) -> float:
    """Pearson correlation coefficient between two fields."""
    a = model.field(field_a).ravel()
    b = model.field(field_b).ravel()
    valid = ~np.isnan(a) & ~np.isnan(b)
    return float(np.corrcoef(a[valid], b[valid])[0, 1])
