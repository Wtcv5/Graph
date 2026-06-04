from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def validate_dataset(nc_path: Path) -> None:
    dataset = xr.open_dataset(str(nc_path), engine="h5netcdf")

    print("=== Dataset Summary ===")
    print(dataset)
    print()

    print("=== Dimensions ===")
    print(f"  X: {dataset.sizes['X']}")
    print(f"  Y: {dataset.sizes['Y']}")
    print(f"  Z: {dataset.sizes['Z']}")
    print()

    print("=== Variable Statistics ===")
    for name in dataset.data_vars:
        values = dataset[name].values
        print(
            f"  {name:20s}: min={np.nanmin(values):10.3f}  "
            f"max={np.nanmax(values):10.3f}  mean={np.nanmean(values):10.3f}  std={np.nanstd(values):10.3f}"
        )

    print()
    print("=== Spot Check: Vp/Vs ratio sanity ===")
    vp = dataset["Vp"].values
    vs = dataset["Vs"].values
    ratio = vp / np.maximum(vs, 1e-10)
    print(
        f"  Vp/Vs ratio: min={np.nanmin(ratio):.3f}, "
        f"max={np.nanmax(ratio):.3f}, mean={np.nanmean(ratio):.3f}"
    )

    print()
    print("=== Check NaN-free ===")
    all_clean = True
    for name in dataset.data_vars:
        nan_count = int(np.isnan(dataset[name].values).sum())
        if nan_count:
            print(f"  WARNING: {name} has {nan_count} NaN values")
            all_clean = False
        else:
            print(f"  OK: {name}")

    if all_clean:
        print("  => All variables NaN-free!")

    print()
    print("=== Cross-section sample ===")
    mid_y = dataset.sizes["Y"] // 2
    z_index = min(100, dataset.sizes["Z"] - 1)
    print(f"  Y-index {mid_y} (Y = {float(dataset.Y[mid_y].values):.1f}m):")
    vp_slice = dataset["Vp"].values[mid_y, :, z_index]
    print(f"    Vp along X at Z={z_index}: {vp_slice[:5]}... (mean={np.nanmean(vp_slice):.1f})")

    print()
    print("Validation complete!")
    dataset.close()
