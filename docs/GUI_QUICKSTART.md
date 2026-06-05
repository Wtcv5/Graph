# GUI Quickstart

## Install

The default project dependencies now include the GUI runtime.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -e .
```

## Launch

Start the GUI without opening a dataset:

```powershell
.\.venv\Scripts\python scripts\run_gui.py
```

Start the GUI and open the sample NetCDF on startup:

```powershell
.\.venv\Scripts\python scripts\run_gui.py outputs/right_tunnel_exit_vpvs/geological_model.nc
```

Launch the same GUI through the package CLI:

```powershell
.\.venv\Scripts\tunnel-velocity-model gui
.\.venv\Scripts\tunnel-velocity-model gui outputs/right_tunnel_exit_vpvs/geological_model.nc
```

## Smoke Test Result

The current sample dataset at `outputs/right_tunnel_exit_vpvs/geological_model.nc` was loaded successfully through the GUI startup path during regression testing.
