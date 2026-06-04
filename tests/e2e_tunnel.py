import sys; sys.path.insert(0, "src")
from pathlib import Path
from tunnel_geology_model import (
    load_from_netcdf, classify_bq,
    ParametricTunnel, TunnelAlignment, TunnelSection, SectionShape,
    TunnelGeologyCoupling,
)
import numpy as np

nc = Path("D:/BaiduNetdiskDownload/Graph/outputs/right_tunnel_exit_vpvs/geological_model.nc")
model = load_from_netcdf(nc)
bq, bq_cls = classify_bq(model["Vp"])
model.classification["BQ"] = bq
model.classification["BQ_Class"] = bq_cls.astype("float32")

x_mid = (model.x_range[0] + model.x_range[1]) / 2
z_mid = (model.z_range[0] + model.z_range[1]) / 2

alignment = TunnelAlignment.from_endpoints(
    x_start=model.x_range[0] + 5, y_start=0, z_start=z_mid - 10,
    x_end=model.x_range[1] - 5, y_end=0, z_end=z_mid + 10,
    gradient_pct=1.0, n_samples=100,
)
section = TunnelSection(
    shape=SectionShape.HORSESHOE, width=12, height=10,
    shotcrete_thickness=0.15, lining_thickness=0.35, n_vertices=64,
)
tunnel = ParametricTunnel(alignment=alignment, section=section, name="Dyk1252+411_Main")
tunnel.build_mesh()
print(tunnel.summary())

coupling = TunnelGeologyCoupling(tunnel, model, field_names=["Vp", "Vs", "Density", "Young_Modulus"])
props = coupling.compute_vertex_properties()
profile = coupling.compute_centerline_profile()
segments = coupling.classify_chainage_segments(segment_length=20)

print()
print("=== Tunnel Vertex Properties ===")
for name, st in coupling.vertex_statistics().items():
    print(f"  {name:20s}: mean={st['mean']:8.1f} +- {st['std']:6.1f}")

print()
print("=== Chainage Segments ===")
cls_map = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}
for seg in segments:
    bq_c = cls_map.get(seg.get("BQ_class", 0), "?")
    print(f"  [{seg['chainage_start']:6.1f} - {seg['chainage_end']:6.1f}]m  "
          f"Vp={seg.get('mean_Vp', 0):6.0f}  BQ={seg.get('mean_BQ', 0):5.0f} ({bq_c})")

zone = coupling.extract_excavation_zone(margin=3)
print(f"\nExcavation zone: {zone.nx}x{zone.ny}x{zone.nz}")
print("Tunnel-Geology coupling E2E PASSED")
