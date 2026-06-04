from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import convolve

from .config import DatasetConfig


COLUMNS = ["X", "Y", "Z", "Velocity"]
PROGRESS_EVERY_ROWS = 5_000_000


def print_section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def scan_bounds(filepath: Path, chunk_size: int) -> tuple[float, float, float, float, float, float]:
    x_min, x_max = float("inf"), float("-inf")
    y_min, y_max = float("inf"), float("-inf")
    z_min, z_max = float("inf"), float("-inf")

    for chunk in pd.read_csv(
        filepath,
        header=None,
        names=COLUMNS,
        chunksize=chunk_size,
        dtype=np.float32,
    ):
        x_min = min(x_min, chunk["X"].min())
        x_max = max(x_max, chunk["X"].max())
        y_min = min(y_min, chunk["Y"].min())
        y_max = max(y_max, chunk["Y"].max())
        z_min = min(z_min, chunk["Z"].min())
        z_max = max(z_max, chunk["Z"].max())

    return float(x_min), float(x_max), float(y_min), float(y_max), float(z_min), float(z_max)


def grid_align(value: float, step: float) -> float:
    return float(np.round(value / step) * step)


def build_grid_bounds(config: DatasetConfig) -> dict[str, float | int]:
    print_section("Step 0: Scanning data bounds")

    t0 = time.time()
    print(f"Scanning {config.vp_path.name} ...")
    vp_bounds = scan_bounds(config.vp_path, config.chunk_size)
    print(f"  Scan time: {time.time() - t0:.1f}s")

    t1 = time.time()
    print(f"Scanning {config.vs_path.name} ...")
    vs_bounds = scan_bounds(config.vs_path, config.chunk_size)
    print(f"  Scan time: {time.time() - t1:.1f}s")

    x_min = min(vp_bounds[0], vs_bounds[0])
    x_max = max(vp_bounds[1], vs_bounds[1])
    y_min = min(vp_bounds[2], vs_bounds[2])
    y_max = max(vp_bounds[3], vs_bounds[3])
    z_min = min(vp_bounds[4], vs_bounds[4])
    z_max = max(vp_bounds[5], vs_bounds[5])

    print(f"  Raw bounds: X=[{x_min:.2f}, {x_max:.2f}]")
    print(f"              Y=[{y_min:.2f}, {y_max:.2f}]")
    print(f"              Z=[{z_min:.2f}, {z_max:.2f}]")

    x_min_g = grid_align(x_min, config.grid_step_m)
    x_max_g = grid_align(x_max, config.grid_step_m)
    y_min_g = grid_align(y_min, config.grid_step_m)
    y_max_g = grid_align(y_max, config.grid_step_m)
    z_min_g = grid_align(z_min, config.grid_step_m)
    z_max_g = grid_align(z_max, config.grid_step_m)

    nx = int(round((x_max_g - x_min_g) / config.grid_step_m)) + 1
    ny = int(round((y_max_g - y_min_g) / config.grid_step_m)) + 1
    nz = int(round((z_max_g - z_min_g) / config.grid_step_m)) + 1

    print(f"  Regular grid: X=[{x_min_g}, {x_max_g}] nx={nx}")
    print(f"                Y=[{y_min_g}, {y_max_g}] ny={ny}")
    print(f"                Z=[{z_min_g}, {z_max_g}] nz={nz}")
    print(f"  Total cells: {nx} x {ny} x {nz} = {nx * ny * nz:,}")

    return {
        "x_min": x_min_g,
        "x_max": x_max_g,
        "nx": nx,
        "y_min": y_min_g,
        "y_max": y_max_g,
        "ny": ny,
        "z_min": z_min_g,
        "z_max": z_max_g,
        "nz": nz,
        "step": config.grid_step_m,
    }


def coord_to_index(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    bounds: dict[str, float | int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ix = np.round((x_values - bounds["x_min"]) / bounds["step"]).astype(np.int32)
    iy = np.round((y_values - bounds["y_min"]) / bounds["step"]).astype(np.int32)
    iz = np.round((z_values - bounds["z_min"]) / bounds["step"]).astype(np.int32)
    return iy, ix, iz


def process_file(
    filepath: Path,
    bounds: dict[str, float | int],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    shape = (int(bounds["ny"]), int(bounds["nx"]), int(bounds["nz"]))
    velocity_sum = np.zeros(shape, dtype=np.float64)
    velocity_count = np.zeros(shape, dtype=np.int32)
    total_rows = 0
    clipped_rows = 0

    for chunk in pd.read_csv(
        filepath,
        header=None,
        names=COLUMNS,
        chunksize=chunk_size,
        dtype=np.float32,
    ):
        iy, ix, iz = coord_to_index(
            chunk["X"].values,
            chunk["Y"].values,
            chunk["Z"].values,
            bounds,
        )

        valid = (
            (iy >= 0)
            & (iy < bounds["ny"])
            & (ix >= 0)
            & (ix < bounds["nx"])
            & (iz >= 0)
            & (iz < bounds["nz"])
        )

        invalid_count = int((~valid).sum())
        if invalid_count:
            clipped_rows += invalid_count
            iy = iy[valid]
            ix = ix[valid]
            iz = iz[valid]
            velocity = chunk["Velocity"].values[valid]
        else:
            velocity = chunk["Velocity"].values

        np.add.at(velocity_sum, (iy, ix, iz), velocity)
        np.add.at(velocity_count, (iy, ix, iz), 1)

        total_rows += len(chunk)
        if total_rows % PROGRESS_EVERY_ROWS == 0:
            filled = int((velocity_count > 0).sum())
            total_cells = velocity_count.size
            print(
                f"  Processed {total_rows / 1e6:.0f}M rows | filled {filled:,} / {total_cells:,} "
                f"({100 * filled / total_cells:.1f}%)"
            )

    return velocity_sum, velocity_count, total_rows, clipped_rows


def fill_nan_mean(data_3d: np.ndarray, kernel_size: int, max_iter: int) -> np.ndarray:
    filled = data_3d.copy()
    kernel = np.ones((kernel_size, kernel_size, kernel_size), dtype=np.float32)

    for iteration in range(max_iter):
        nan_mask = np.isnan(filled)
        if not nan_mask.any():
            print(f"    Iter {iteration + 1}: all NaN filled")
            break

        before = int(nan_mask.sum())
        zeroed = np.where(nan_mask, 0.0, filled)
        neighbor_sum = convolve(
            zeroed.astype(np.float64),
            kernel.astype(np.float64),
            mode="constant",
            cval=0.0,
        )
        valid_mask = ~np.isnan(filled)
        neighbor_count = convolve(
            valid_mask.astype(np.float64),
            kernel.astype(np.float64),
            mode="constant",
            cval=0.0,
        )

        fill_mask = nan_mask & (neighbor_count > 0)
        filled[fill_mask] = (neighbor_sum[fill_mask] / neighbor_count[fill_mask]).astype(np.float32)

        after = int(np.isnan(filled).sum())
        print(f"    Iter {iteration + 1}: NaN {before:,} -> {after:,}")
        if after == before:
            break

    return filled


def compute_means_and_fill(
    velocity_sum: np.ndarray,
    velocity_count: np.ndarray,
    kernel_size: int,
    max_iter: int,
    label: str,
) -> tuple[np.ndarray, int]:
    with np.errstate(divide="ignore", invalid="ignore"):
        grid = np.where(velocity_count > 0, velocity_sum / velocity_count, np.nan).astype(np.float32)

    filled_cells = int(np.isfinite(grid).sum())
    total_cells = grid.size
    print(f"  {label} filled: {filled_cells:,} / {total_cells:,} ({100 * filled_cells / total_cells:.1f}%)")
    print(f"  Filling {label} NaN cells (iterative 3x3x3 neighbor average)...")

    grid_no_nan = fill_nan_mean(grid, kernel_size=kernel_size, max_iter=max_iter)
    global_median = float(np.nanmedian(grid_no_nan))
    final_grid = np.where(np.isnan(grid_no_nan), global_median, grid_no_nan)

    print(f"  {label} global median: {global_median:.1f} m/s")
    print(f"  Final {label} NaN count: {int(np.isnan(final_grid).sum())}")
    return final_grid.astype(np.float32), filled_cells


def compute_derived_fields(vp_final: np.ndarray, vs_final: np.ndarray) -> dict[str, np.ndarray]:
    eps = 1e-10
    vs_safe = np.maximum(vs_final, eps)
    vp_safe = np.maximum(vp_final, eps)

    vp_vs_ratio = vp_safe / vs_safe
    vp2 = vp_safe**2
    vs2 = vs_safe**2

    with np.errstate(divide="ignore", invalid="ignore"):
        poisson = (vp2 - 2 * vs2) / (2 * (vp2 - vs2))
        poisson = np.clip(poisson, 0.0, 0.5)
        poisson = np.where(np.abs(vp_safe - vs_safe) < eps, 0.5, poisson)

    density = 0.31 * np.power(vp_final, 0.25)
    density_kgm3 = density * 1000.0
    shear_modulus = density_kgm3 * vs2 / 1e9
    bulk_modulus = density_kgm3 * (vp2 - (4.0 / 3.0) * vs2) / 1e9
    young_modulus = 2.0 * shear_modulus * (1.0 + poisson)
    lame_lambda = bulk_modulus - 2.0 * shear_modulus / 3.0
    p_impedance = density_kgm3 * vp_final / 1e6
    s_impedance = density_kgm3 * vs_final / 1e6

    print_section("Step 3: Computing rock mechanics parameters")
    print(f"  Poisson ratio:   [{np.nanmin(poisson):.3f}, {np.nanmax(poisson):.3f}] (valid 0.0~0.5)")
    print(f"  Density:         [{np.nanmin(density_kgm3):.0f}, {np.nanmax(density_kgm3):.0f}] kg/m3")
    print(f"  Young modulus:   [{np.nanmin(young_modulus):.1f}, {np.nanmax(young_modulus):.1f}] GPa")
    print(f"  Shear modulus:   [{np.nanmin(shear_modulus):.1f}, {np.nanmax(shear_modulus):.1f}] GPa")
    print(f"  Bulk modulus:    [{np.nanmin(bulk_modulus):.1f}, {np.nanmax(bulk_modulus):.1f}] GPa")

    return {
        "Vp_Vs_ratio": vp_vs_ratio.astype(np.float32),
        "Poisson_Ratio": poisson.astype(np.float32),
        "Young_Modulus": young_modulus.astype(np.float32),
        "Shear_Modulus": shear_modulus.astype(np.float32),
        "Bulk_Modulus": bulk_modulus.astype(np.float32),
        "Lame_Lambda": lame_lambda.astype(np.float32),
        "Density": density_kgm3.astype(np.float32),
        "P_Impedance": p_impedance.astype(np.float32),
        "S_Impedance": s_impedance.astype(np.float32),
    }


def create_dataset(
    config: DatasetConfig,
    bounds: dict[str, float | int],
    vp_final: np.ndarray,
    vs_final: np.ndarray,
    derived: dict[str, np.ndarray],
    vp_count: np.ndarray,
    vs_count: np.ndarray,
) -> xr.Dataset:
    step = config.grid_step_m
    x_coords = np.arange(bounds["x_min"], bounds["x_max"] + step / 2, step, dtype=np.float32)
    y_coords = np.arange(bounds["y_min"], bounds["y_max"] + step / 2, step, dtype=np.float32)
    z_coords = np.arange(bounds["z_min"], bounds["z_max"] + step / 2, step, dtype=np.float32)

    print_section("Step 4: Exporting NetCDF")
    print(f"  Coords: X={len(x_coords)}, Y={len(y_coords)}, Z={len(z_coords)}")

    return xr.Dataset(
        data_vars={
            "Vp": (["Y", "X", "Z"], vp_final, {"long_name": "P-wave Velocity", "units": "m/s"}),
            "Vs": (["Y", "X", "Z"], vs_final, {"long_name": "S-wave Velocity", "units": "m/s"}),
            "Vp_Vs_ratio": (["Y", "X", "Z"], derived["Vp_Vs_ratio"], {"long_name": "Vp/Vs Ratio", "units": "1"}),
            "Poisson_Ratio": (
                ["Y", "X", "Z"],
                derived["Poisson_Ratio"],
                {"long_name": "Poisson's Ratio", "units": "1", "valid_range": [0.0, 0.5]},
            ),
            "Young_Modulus": (
                ["Y", "X", "Z"],
                derived["Young_Modulus"],
                {"long_name": "Young's Modulus", "units": "GPa"},
            ),
            "Shear_Modulus": (
                ["Y", "X", "Z"],
                derived["Shear_Modulus"],
                {"long_name": "Shear Modulus", "units": "GPa"},
            ),
            "Bulk_Modulus": (
                ["Y", "X", "Z"],
                derived["Bulk_Modulus"],
                {"long_name": "Bulk Modulus", "units": "GPa"},
            ),
            "Lame_Lambda": (
                ["Y", "X", "Z"],
                derived["Lame_Lambda"],
                {"long_name": "Lame's First Parameter", "units": "GPa"},
            ),
            "Density": (
                ["Y", "X", "Z"],
                derived["Density"],
                {"long_name": "Density (Gardner)", "units": "kg/m3"},
            ),
            "P_Impedance": (
                ["Y", "X", "Z"],
                derived["P_Impedance"],
                {"long_name": "P-wave Impedance", "units": "MPa*s/m"},
            ),
            "S_Impedance": (
                ["Y", "X", "Z"],
                derived["S_Impedance"],
                {"long_name": "S-wave Impedance", "units": "MPa*s/m"},
            ),
            "Vp_count": (
                ["Y", "X", "Z"],
                vp_count.astype(np.int32),
                {"long_name": "Vp measurement count per cell", "units": "count"},
            ),
            "Vs_count": (
                ["Y", "X", "Z"],
                vs_count.astype(np.int32),
                {"long_name": "Vs measurement count per cell", "units": "count"},
            ),
        },
        coords={
            "Y": (["Y"], y_coords, {"long_name": "Y coordinate (tunnel transverse)", "units": "m"}),
            "X": (["X"], x_coords, {"long_name": "X coordinate (tunnel longitudinal)", "units": "m"}),
            "Z": (["Z"], z_coords, {"long_name": "Z coordinate (elevation/depth)", "units": "m"}),
        },
        attrs={
            "title": config.title,
            "description": config.description,
            "source": config.source,
            "grid_step_m": step,
            "coordinate_system": config.coordinate_system,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "preprocessing": "Coordinate rounding to regular grid, mean aggregation, iterative neighbor-mean gap filling",
            "density_formula": config.density_formula,
            "software": "Python xarray/h5netcdf",
        },
    )


def build_encoding(dataset: xr.Dataset) -> dict[str, dict[str, object]]:
    encoding: dict[str, dict[str, object]] = {}

    for variable_name in dataset.data_vars:
        encoding[variable_name] = {
            "zlib": True,
            "complevel": 4,
            "dtype": "int32" if variable_name in {"Vp_count", "Vs_count"} else "float32",
        }

    for coord_name in dataset.coords:
        encoding[coord_name] = {"zlib": True, "complevel": 4}

    return encoding


def write_info_file(
    config: DatasetConfig,
    bounds: dict[str, float | int],
    vp_rows: int,
    vs_rows: int,
    vp_filled_cells: int,
    vs_filled_cells: int,
    dataset: xr.Dataset,
    output_nc_path: Path,
    output_info_path: Path,
) -> None:
    total_cells = int(dataset["Vp"].values.size)
    info = {
        "input": {
            "vp_file": str(config.vp_path),
            "vs_file": str(config.vs_path),
            "vp_rows": vp_rows,
            "vs_rows": vs_rows,
        },
        "grid": bounds,
        "coverage": {
            "vp_filled_cells": vp_filled_cells,
            "vs_filled_cells": vs_filled_cells,
            "total_cells": total_cells,
            "vp_coverage_pct": round(100 * vp_filled_cells / total_cells, 2),
            "vs_coverage_pct": round(100 * vs_filled_cells / total_cells, 2),
        },
        "statistics": {
            "Vp_mean": float(np.nanmean(dataset["Vp"].values)),
            "Vp_std": float(np.nanstd(dataset["Vp"].values)),
            "Vs_mean": float(np.nanmean(dataset["Vs"].values)),
            "Vs_std": float(np.nanstd(dataset["Vs"].values)),
            "Poisson_mean": float(np.nanmean(dataset["Poisson_Ratio"].values)),
            "Density_mean": float(np.nanmean(dataset["Density"].values)),
            "Young_Modulus_mean": float(np.nanmean(dataset["Young_Modulus"].values)),
        },
        "output": {
            "file": str(output_nc_path),
            "size_mb": round(output_nc_path.stat().st_size / (1024 * 1024), 1),
            "format": "NetCDF4",
            "dimensions": ["Y", "X", "Z"],
            "variables": list(dataset.data_vars.keys()),
        },
    }

    with output_info_path.open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, ensure_ascii=False)


def preprocess_dataset(config: DatasetConfig, force: bool = False) -> tuple[Path, Path]:
    if not config.vp_path.exists():
        raise FileNotFoundError(f"Vp file not found: {config.vp_path}")
    if not config.vs_path.exists():
        raise FileNotFoundError(f"Vs file not found: {config.vs_path}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    if not force and (config.output_nc_path.exists() or config.output_info_path.exists()):
        raise FileExistsError(
            f"Output already exists under {config.output_dir}. Re-run with --force to overwrite."
        )

    print(f"Dataset: {config.name}")
    print(f"Config:  {config.config_path}")
    print(f"Vp CSV:  {config.vp_path}")
    print(f"Vs CSV:  {config.vs_path}")
    print(f"Output:  {config.output_dir}")

    started_at = time.time()
    bounds = build_grid_bounds(config)

    print_section("Step 1: Reading and accumulating to grid")
    print(f"Processing {config.vp_path.name} ...")
    t_vp = time.time()
    vp_sum, vp_count, vp_rows, vp_clipped = process_file(config.vp_path, bounds, config.chunk_size)
    vp_cells = int((vp_count > 0).sum())
    print(
        f"  Vp: {vp_rows:,} rows, {vp_clipped} clipped, {vp_cells:,} cells filled "
        f"({100 * vp_cells / vp_count.size:.1f}%)"
    )
    print(f"  Time: {time.time() - t_vp:.1f}s")

    print()
    print(f"Processing {config.vs_path.name} ...")
    t_vs = time.time()
    vs_sum, vs_count, vs_rows, vs_clipped = process_file(config.vs_path, bounds, config.chunk_size)
    vs_cells = int((vs_count > 0).sum())
    print(
        f"  Vs: {vs_rows:,} rows, {vs_clipped} clipped, {vs_cells:,} cells filled "
        f"({100 * vs_cells / vs_count.size:.1f}%)"
    )
    print(f"  Time: {time.time() - t_vs:.1f}s")

    print_section("Step 2: Computing grid means and filling gaps")
    vp_final, vp_filled_cells = compute_means_and_fill(
        vp_sum,
        vp_count,
        kernel_size=config.fill_kernel_size,
        max_iter=config.fill_max_iter,
        label="Vp",
    )
    print()
    vs_final, vs_filled_cells = compute_means_and_fill(
        vs_sum,
        vs_count,
        kernel_size=config.fill_kernel_size,
        max_iter=config.fill_max_iter,
        label="Vs",
    )

    derived = compute_derived_fields(vp_final, vs_final)
    dataset = create_dataset(config, bounds, vp_final, vs_final, derived, vp_count, vs_count)
    encoding = build_encoding(dataset)

    if config.output_nc_path.exists():
        config.output_nc_path.unlink()
    if config.output_info_path.exists():
        config.output_info_path.unlink()

    print(f"  Writing {config.output_nc_path} ...")
    export_started_at = time.time()
    dataset.to_netcdf(
        config.output_nc_path,
        encoding=encoding,
        format="NETCDF4",
        engine="h5netcdf",
    )
    print(f"  Done! File size: {config.output_nc_path.stat().st_size / (1024 * 1024):.1f} MB")
    print(f"  Export time: {time.time() - export_started_at:.1f}s")

    write_info_file(
        config=config,
        bounds=bounds,
        vp_rows=vp_rows,
        vs_rows=vs_rows,
        vp_filled_cells=vp_filled_cells,
        vs_filled_cells=vs_filled_cells,
        dataset=dataset,
        output_nc_path=config.output_nc_path,
        output_info_path=config.output_info_path,
    )

    dataset.close()

    print()
    print("=" * 60)
    print("Preprocessing complete!")
    print(f"  Output: {config.output_nc_path}")
    print(f"  Info:   {config.output_info_path}")
    print(f"  Total time: {time.time() - started_at:.0f}s")
    print("=" * 60)

    return config.output_nc_path, config.output_info_path
