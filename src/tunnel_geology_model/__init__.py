"""Tunnel geological modeling core — 3D grid model, classification, cross-sections, export."""

from .model import GeologicalModel, load_from_netcdf
from .classification import classify_bq, classify_rmr, geohazard_index
from .cross_section import (
    extract_yz_section,
    extract_xz_section,
    extract_xy_section,
    extract_arbitrary_section,
)
from .isosurface import marching_cubes_isosurface, multi_isosurface
from .export import (
    to_vtk_structured_grid,
    to_vtk_rectilinear_grid,
    to_numpy_dict,
    to_xyz_point_cloud,
)
from .statistics import (
    field_histogram,
    depth_profile,
    anomaly_detection,
    correlation_map,
)

__all__ = [
    "GeologicalModel",
    "load_from_netcdf",
    "classify_bq",
    "classify_rmr",
    "geohazard_index",
    "extract_yz_section",
    "extract_xz_section",
    "extract_xy_section",
    "extract_arbitrary_section",
    "marching_cubes_isosurface",
    "multi_isosurface",
    "to_vtk_structured_grid",
    "to_vtk_rectilinear_grid",
    "to_numpy_dict",
    "to_xyz_point_cloud",
    "field_histogram",
    "depth_profile",
    "anomaly_detection",
    "correlation_map",
]
