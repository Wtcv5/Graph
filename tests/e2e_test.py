from pathlib import Path
import sys
sys.path.insert(0, 'src')

from tunnel_geology_model import (
    load_from_netcdf, classify_bq, geohazard_index,
    extract_xz_section, marching_cubes_isosurface, to_vtk_rectilinear_grid,
)
import numpy as np
import tempfile

nc = Path("D:/BaiduNetdiskDownload/Graph/outputs/right_tunnel_exit_vpvs/geological_model.nc")
model = load_from_netcdf(nc)
print(model.summary())

# Classification
bq, bq_class = classify_bq(model["Vp"])
model.classification["BQ"] = bq
model.classification["BQ_Class"] = bq_class.astype(np.float32)
print(f"BQ range: [{bq.min():.0f}, {bq.max():.0f}]")
dist = np.bincount(bq_class.ravel())
print(f"BQ class distribution (I-V): {list(dist[1:])}")

# Cross-section
section = extract_xz_section(model, 0.0, field_names=["Vp"])
print(f"XZ section Vp shape: {section['Vp'].shape}")

# Geohazard
gh = geohazard_index(
    model["Poisson_Ratio"], model["Young_Modulus"], model["Density"]
)
print(f"Geohazard index: [{gh.min():.3f}, {gh.max():.3f}] mean={gh.mean():.3f}")

# Isosurface
verts, faces, normals, values = marching_cubes_isosurface(
    model["Vp"], 5800.0, spatial_step=model.grid_step,
    origin=(model.x_range[0], model.y_range[0], model.z_range[0]),
)
print(f"Isosurface (Vp=5800): {len(verts)} vertices, {len(faces)} triangles")

# VTK export
td = tempfile.mkdtemp()
out = Path(td) / "model.vtr"
to_vtk_rectilinear_grid(model, out)
print(f"VTK export: {out.stat().st_size / 1024:.0f} KB")

print()
print("End-to-end test PASSED")
