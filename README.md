# Tunnel Velocity Model

This repository packages a tunnel seismic velocity modeling workflow into a reproducible Python project.

The raw inputs are two large CSV point clouds:

- `Vp`: P-wave velocity
- `Vs`: S-wave velocity

The pipeline aggregates those scan-line points onto a regular 3D grid, fills sparse gaps, derives rock-mechanics parameters, and exports a NetCDF geological model.

## Repository Layout

```text
.
├── configs/
│   └── datasets/
│       └── right_tunnel_exit_vpvs.json
├── outputs/
│   └── .gitkeep
├── scripts/
│   ├── preprocess.py
│   └── validate.py
├── src/
│   └── tunnel_velocity_model/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py
│       ├── pipeline.py
│       └── validation.py
├── tests/
│   └── test_config.py
└── 出口右线Dyk1252+411-VpVs/
    ├── 出口右线Dyk1252+411-Vp.csv
    ├── 出口右线Dyk1252+411-Vs.csv
    └── preprocessed/
        ├── preprocess.py
        └── validate.py
```

## Why The Raw Data Folder Stays In Place

The current CSV files are already tracked as large files and are several hundred megabytes each. This cleanup keeps them in their original acquisition folder to avoid a risky repository-wide rename, while moving the executable logic to a standard project layout.

## Environment Setup

Create a fresh environment at the repository root instead of using the legacy nested `venv` under `preprocessed/`.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Run The Pipeline

Default dataset config:

```powershell
.\.venv\Scripts\python scripts\preprocess.py --force
```

Outputs are written to:

```text
outputs/right_tunnel_exit_vpvs/
├── geological_model.nc
└── preprocessing_info.json
```

Validate the generated NetCDF:

```powershell
.\.venv\Scripts\python scripts\validate.py
```

## Run The GUI

The default dependency install now includes the GUI stack.
See [docs/GUI_QUICKSTART.md](/d:/BaiduNetdiskDownload/Graph/docs/GUI_QUICKSTART.md:1) for the tested launch paths.

```powershell
.\.venv\Scripts\python scripts\run_gui.py
.\.venv\Scripts\python scripts\run_gui.py outputs/right_tunnel_exit_vpvs/geological_model.nc
```

## Alternative Entrypoints

If you install the package in editable mode:

```powershell
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\tunnel-velocity-model preprocess --force
.\.venv\Scripts\tunnel-velocity-model validate
.\.venv\Scripts\tunnel-velocity-model gui
.\.venv\Scripts\tunnel-velocity-model gui outputs/right_tunnel_exit_vpvs/geological_model.nc
```

Legacy wrappers remain available for compatibility:

- `出口右线Dyk1252+411-VpVs/preprocessed/preprocess.py`
- `出口右线Dyk1252+411-VpVs/preprocessed/validate.py`

They now delegate to the new project code.

## Dataset Config

The workflow is parameterized by JSON files in `configs/datasets/`. A dataset config defines:

- where the raw `Vp` and `Vs` CSV files live
- where outputs should be written
- grid spacing
- chunk size
- gap-filling settings
- metadata for the NetCDF export

That makes the pipeline reusable for additional tunnel sections without copying scripts.

## Test

The lightweight tests do not touch the large CSV data:

```powershell
.\.venv\Scripts\python -m unittest discover -s tests
```
