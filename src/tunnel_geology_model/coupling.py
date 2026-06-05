"""Tunnel-Geology coupling — intersection, property extraction, analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import RegularGridInterpolator

if TYPE_CHECKING:
    from .model import GeologicalModel
    from .tunnel import ParametricTunnel


@dataclass
class TunnelGeologyCoupling:
    """Bridges a ParametricTunnel with a GeologicalModel.

    Provides:
    - Property lookup at tunnel mesh vertices
    - Geological profile along tunnel centerline
    - Rock mass statistics along the tunnel
    - Excavation zone volume extraction
    """

    tunnel: "ParametricTunnel"
    model: "GeologicalModel"
    field_names: list[str] = field(default_factory=list)

    # Cached results
    _vertex_properties: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _centerline_profile: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _tunnel_bbox: tuple[float, float, float, float, float, float] | None = field(
        default=None, repr=False
    )

    def __post_init__(self):
        if not self.field_names:
            self.field_names = self.model.field_names[:6]  # default: first 6 fields

    # ── vertex properties ───────────────────────────────

    def compute_vertex_properties(self) -> dict[str, np.ndarray]:
        """Look up geological properties at every tunnel mesh vertex.

        Uses trilinear interpolation from the 3D geological grid.

        Returns
        -------
        dict[str, np.ndarray]
            {'Vp': (N,), 'Vs': (N,), ...} — one value per vertex.
        """
        if self.tunnel.vertices is None:
            raise ValueError("Tunnel mesh not built. Call tunnel.build_mesh() first.")

        verts = self.tunnel.vertices  # (N, 3) — x, y, z
        result = {}

        # Build interpolators for each field
        for name in self.field_names:
            if name not in self.model.field_names:
                continue
            field_3d = self.model.field(name).astype(np.float64)

            interp = RegularGridInterpolator(
                (self.model.y, self.model.x, self.model.z),
                field_3d,
                bounds_error=False,
                fill_value=np.nan,
            )
            # Query expects (y, x, z) order — our vertices are (x, y, z)
            values = interp(np.column_stack([verts[:, 1], verts[:, 0], verts[:, 2]]))
            result[name] = values.astype(np.float32)

        # Also include classification fields
        for name, arr in self.model.classification.items():
            if arr.ndim == 3:
                interp = RegularGridInterpolator(
                    (self.model.y, self.model.x, self.model.z),
                    arr.astype(np.float64),
                    bounds_error=False,
                    fill_value=np.nan,
                )
                result[name] = interp(
                    np.column_stack([verts[:, 1], verts[:, 0], verts[:, 2]])
                ).astype(np.float32)

        self._vertex_properties = result
        return result

    # ── centerline profile ──────────────────────────────

    def compute_centerline_profile(self) -> dict[str, np.ndarray]:
        """Extract geological properties along the tunnel centerline.

        Returns
        -------
        dict[str, np.ndarray]
            {'chainage': (M,), 'Vp': (M,), 'Vs': (M,), ...}
        """
        ch, cx, cy, cz = self.tunnel.alignment.sample()
        result = {"chainage": ch.astype(np.float32)}

        for name in self.field_names:
            if name not in self.model.field_names:
                continue
            field_3d = self.model.field(name).astype(np.float64)
            interp = RegularGridInterpolator(
                (self.model.y, self.model.x, self.model.z),
                field_3d,
                bounds_error=False,
                fill_value=np.nan,
            )
            result[name] = interp(np.column_stack([cy, cx, cz])).astype(np.float32)

        # BQ classification along centerline
        if "BQ" in self.model.classification:
            bq_arr = self.model.classification["BQ"].astype(np.float64)
            bq_interp = RegularGridInterpolator(
                (self.model.y, self.model.x, self.model.z),
                bq_arr,
                bounds_error=False,
                fill_value=np.nan,
            )
            result["BQ"] = bq_interp(np.column_stack([cy, cx, cz])).astype(np.float32)

        self._centerline_profile = result
        return result

    # ── rock mass statistics along tunnel ───────────────

    def classify_chainage_segments(
        self,
        segment_length: float = 10.0,
    ) -> list[dict]:
        """Divide tunnel into segments and classify each.

        Parameters
        ----------
        segment_length : float
            Segment length in meters.

        Returns
        -------
        list of dict
            [{chainage_start, chainage_end, mean_Vp, mean_Vs, mean_BQ, BQ_class, ...}]
        """
        profile = self._centerline_profile
        if not profile:
            profile = self.compute_centerline_profile()

        ch = profile["chainage"]
        segments = []
        ch_start = float(ch[0])
        ch_end = float(ch[-1])

        seg_start = ch_start
        while seg_start < ch_end:
            seg_end = min(seg_start + segment_length, ch_end)
            mask = (ch >= seg_start) & (ch <= seg_end)

            seg = {
                "chainage_start": round(seg_start, 1),
                "chainage_end": round(seg_end, 1),
                "has_data": False,
            }
            for name in self.field_names:
                if name in profile:
                    values = profile[name][mask]
                    if np.isfinite(values).any():
                        seg[f"mean_{name}"] = float(np.nanmean(values))
                        seg["has_data"] = True
                    else:
                        seg[f"mean_{name}"] = np.nan
            if "BQ" in profile:
                bq_values = profile["BQ"][mask]
                if np.isfinite(bq_values).any():
                    bq_mean = float(np.nanmean(bq_values))
                    seg["mean_BQ"] = bq_mean
                    seg["BQ_class"] = _bq_to_class(bq_mean)
                    seg["has_data"] = True
                else:
                    seg["mean_BQ"] = np.nan
                    seg["BQ_class"] = None
            segments.append(seg)
            seg_start = seg_end

        return segments

    # ── excavation zone ────────────────────────────────

    def extract_excavation_zone(
        self,
        margin: float = 2.0,
    ) -> "GeologicalModel":
        """Extract the geological sub-volume around the tunnel.

        Parameters
        ----------
        margin : float
            Extra margin (m) beyond the tunnel bounding box.

        Returns
        -------
        GeologicalModel — subset of the full model around the tunnel.
        """
        bbox = self._compute_bbox()
        if bbox is None:
            raise ValueError("Tunnel mesh not built.")

        x_min, x_max, y_min, y_max, z_min, z_max = bbox

        # Clip to model bounds
        x_min = max(x_min - margin, self.model.x_range[0])
        x_max = min(x_max + margin, self.model.x_range[1])
        y_min = max(y_min - margin, self.model.y_range[0])
        y_max = min(y_max + margin, self.model.y_range[1])
        z_min = max(z_min - margin, self.model.z_range[0])
        z_max = min(z_max + margin, self.model.z_range[1])

        sub = self.model.subset_x(x_min, x_max)
        sub = sub.subset_y(y_min, y_max)
        sub = sub.subset_z(z_min, z_max)
        return sub

    def _compute_bbox(self) -> tuple[float, float, float, float, float, float] | None:
        if self._tunnel_bbox is not None:
            return self._tunnel_bbox
        if self.tunnel.vertices is None:
            return None
        v = self.tunnel.vertices
        self._tunnel_bbox = (
            float(v[:, 0].min()), float(v[:, 0].max()),
            float(v[:, 1].min()), float(v[:, 1].max()),
            float(v[:, 2].min()), float(v[:, 2].max()),
        )
        return self._tunnel_bbox

    # ── statistics ──────────────────────────────────────

    def vertex_statistics(self) -> dict[str, dict[str, float]]:
        """Compute statistics of geological properties at tunnel vertices.

        Returns
        -------
        dict[str, dict] — {field_name: {min, max, mean, std}}
        """
        if not self._vertex_properties:
            self.compute_vertex_properties()

        stats = {}
        for name, values in self._vertex_properties.items():
            valid = values[~np.isnan(values)]
            if len(valid) == 0:
                stats[name] = {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
            else:
                stats[name] = {
                    "min": float(np.min(valid)),
                    "max": float(np.max(valid)),
                    "mean": float(np.mean(valid)),
                    "std": float(np.std(valid)),
                }
        return stats

    def summary(self) -> str:
        lines = [
            "TunnelGeologyCoupling",
            f"  Tunnel: {self.tunnel.name}",
            f"  Fields: {', '.join(self.field_names)}",
        ]
        if self._vertex_properties:
            s = self.vertex_statistics()
            lines.append("  Vertex properties:")
            for name, st in s.items():
                lines.append(f"    {name}: mean={st['mean']:.1f} ± {st['std']:.1f}")
        return "\n".join(lines)


def _bq_to_class(bq: float) -> int:
    if bq > 550:
        return 1
    elif bq > 450:
        return 2
    elif bq > 350:
        return 3
    elif bq > 250:
        return 4
    else:
        return 5

def _bq_class_label(cls: int) -> str:
    return {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}.get(cls, "?")
