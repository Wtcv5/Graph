"""Parametric tunnel geometry — centerline, cross-section, 3D mesh generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


class SectionShape(str, Enum):
    HORSESHOE = "horseshoe"
    CIRCULAR = "circular"
    RECTANGULAR = "rectangular"
    CUSTOM = "custom"


@dataclass
class TunnelSection:
    """A 2D cross-section shape defined in local (Y, Z) coordinates at a chainage.

    All dimensions in meters. The section is centered at (y=0, z=0).
    """

    shape: SectionShape = SectionShape.HORSESHOE

    # ── dimensions ──
    width: float = 12.0          # excavation width (m)
    height: float = 10.0         # excavation height (m)
    invert_radius: float = 6.0   # radius of invert arch (horseshoe)
    arch_radius: float = 6.0     # radius of upper arch (horseshoe)

    # ── support layers (concentric offsets from excavation) ──
    shotcrete_thickness: float = 0.15   # primary lining (m)
    lining_thickness: float = 0.35      # secondary lining (m)

    # ── discretization ──
    n_vertices: int = 64         # vertices around the perimeter

    def generate_profile(self, include_layers: bool = False) -> list[np.ndarray]:
        """Generate 2D perimeter vertices (ny, nz) for this section.

        Returns
        -------
        profiles : list of np.ndarray
            [excavation_profile] or [excavation, shotcrete, lining].
            Each is (n_vertices+1, 2) — last point == first (closed loop).
        """
        profiles = [self._excavation_profile()]
        if include_layers:
            profiles.append(self._offset_profile(profiles[0], self.shotcrete_thickness))
            profiles.append(self._offset_profile(profiles[0],
                                                  self.shotcrete_thickness + self.lining_thickness))
        return profiles

    def _excavation_profile(self) -> np.ndarray:
        """Generate excavation perimeter vertices."""
        n = self.n_vertices
        if self.shape == SectionShape.CIRCULAR:
            r = self.width / 2
            theta = np.linspace(0, 2 * np.pi, n + 1)
            y = r * np.cos(theta)
            z = r * np.sin(theta) + r  # bottom at z=0
            return np.column_stack([y, z])

        elif self.shape == SectionShape.RECTANGULAR:
            w, h = self.width, self.height
            # perimeter: bottom → right → top → left → back to bottom
            pts = np.array([
                [-w/2, 0], [w/2, 0], [w/2, h], [-w/2, h], [-w/2, 0]
            ])
            # interpolate to n vertices
            return self._resample(pts, n)

        elif self.shape == SectionShape.HORSESHOE:
            return self._horseshoe_profile(n)

        else:  # CUSTOM — default to horseshoe
            return self._horseshoe_profile(n)

    def _horseshoe_profile(self, n: int) -> np.ndarray:
        """Generate standard horseshoe tunnel section.

        Upper portion: circular arch (radius = arch_radius)
        Lower portion: curved invert (radius = invert_radius)
        Side walls connect arch to invert tangentially.
        """
        w, h = self.width, self.height
        r_arch = self.arch_radius
        r_inv = self.invert_radius

        # Build profile in segments
        segments = []

        # Top arch: from left springline to right springline
        # Springline at y = ±(w/2), z ≈ h - r_arch (for a circular arch)
        spring_z = max(h - r_arch, r_arch)
        arch_center_z = spring_z

        # Arc angle for arch
        half_width = w / 2
        if half_width < r_arch:
            arch_half_angle = np.arcsin(half_width / r_arch)
        else:
            arch_half_angle = np.pi / 2

        # Arch: right to left (clockwise viewed from +X)
        arch_theta = np.linspace(-arch_half_angle, np.pi + arch_half_angle, n // 2)
        arch_y = r_arch * np.cos(arch_theta)
        arch_z = arch_center_z + r_arch * np.sin(arch_theta)
        segments.append(np.column_stack([arch_y, arch_z]))

        # Invert: bottom arc
        inv_center_z = r_inv
        inv_half_angle = np.arcsin(min(half_width / r_inv, 1.0))
        inv_theta = np.linspace(np.pi + inv_half_angle, 2 * np.pi - inv_half_angle, n // 2)
        inv_y = r_inv * np.cos(inv_theta)
        inv_z = inv_center_z + r_inv * np.sin(inv_theta)
        segments.append(np.column_stack([inv_y, inv_z]))

        # Combine and close
        pts = np.vstack(segments)
        pts = self._sort_radial(pts)
        pts = np.vstack([pts, pts[0:1]])  # close loop
        return self._resample(pts, n)

    def _sort_radial(self, pts: np.ndarray) -> np.ndarray:
        """Sort points counter-clockwise around centroid."""
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        idx = np.argsort(angles)
        return pts[idx]

    def _resample(self, pts: np.ndarray, n: int) -> np.ndarray:
        """Resample a polyline to n evenly-spaced vertices."""
        # Cumulative arc length
        diffs = np.diff(pts, axis=0)
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        cumdist = np.concatenate([[0], np.cumsum(dists)])
        total = cumdist[-1]

        if total < 1e-6:
            return pts

        new_dist = np.linspace(0, total, n + 1)
        new_y = np.interp(new_dist, cumdist, pts[:, 0])
        new_z = np.interp(new_dist, cumdist, pts[:, 1])
        return np.column_stack([new_y, new_z])

    def _offset_profile(self, base: np.ndarray, thickness: float) -> np.ndarray:
        """Offset a closed profile outward by `thickness`."""
        # Compute normals at each vertex
        offset = base.copy()
        n_pts = len(base) - 1  # exclude closing duplicate
        for i in range(n_pts):
            prev = base[(i - 1) % n_pts]
            next_ = base[(i + 1) % n_pts]
            tangent = next_ - prev
            normal = np.array([-tangent[1], tangent[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                normal = normal / norm
            offset[i] = base[i] + thickness * normal
        offset[-1] = offset[0]  # close
        return offset


@dataclass
class TunnelAlignment:
    """3D tunnel centerline defined by control points.

    The centerline is parametrized by chainage (meters along tunnel).
    X is the primary longitudinal axis of the geological grid.
    """

    # Control points: (chainage, x, y, z)
    control_points: np.ndarray  # (N, 4)

    # Chainage array for fine sampling
    n_samples: int = 200

    @classmethod
    def from_endpoints(
        cls,
        x_start: float, y_start: float, z_start: float,
        x_end: float, y_end: float, z_end: float,
        chainage_start: float = 0.0,
        gradient_pct: float = 0.0,
        n_samples: int = 200,
    ) -> "TunnelAlignment":
        """Create a straight alignment from two endpoints."""
        dx = x_end - x_start
        dy = y_end - y_start
        length = np.sqrt(dx**2 + dy**2)
        chainage_end = chainage_start + length
        dz = length * gradient_pct / 100.0
        z_end_actual = z_start + dz

        cps = np.array([
            [chainage_start, x_start, y_start, z_start],
            [chainage_end, x_end, y_end, z_end_actual],
        ])
        return cls(control_points=cps, n_samples=n_samples)

    @classmethod
    def from_control_points(
        cls,
        points: list[tuple[float, float, float, float]],
        n_samples: int = 200,
    ) -> "TunnelAlignment":
        """Create alignment from (chainage, x, y, z) control points."""
        return cls(control_points=np.array(points), n_samples=n_samples)

    def sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample the centerline at fine resolution.

        Returns
        -------
        chainage : (M,)
        x, y, z : (M,) — world coordinates along centerline
        """
        cps = self.control_points
        ch = cps[:, 0]
        x_arr = cps[:, 1]
        y_arr = cps[:, 2]
        z_arr = cps[:, 3]

        ch_new = np.linspace(ch[0], ch[-1], self.n_samples)
        x_new = np.interp(ch_new, ch, x_arr)
        y_new = np.interp(ch_new, ch, y_arr)
        z_new = np.interp(ch_new, ch, z_arr)

        return ch_new, x_new, y_new, z_new

    @property
    def total_length(self) -> float:
        ch = self.control_points[:, 0]
        return float(ch[-1] - ch[0])

    def tangent_at(self, chainage: float) -> np.ndarray:
        """Return unit tangent vector (dx, dy, dz) at a given chainage."""
        ch = self.control_points[:, 0]
        x_arr = self.control_points[:, 1]
        y_arr = self.control_points[:, 2]
        z_arr = self.control_points[:, 3]

        eps = 1.0
        ch_lo = max(ch[0], chainage - eps)
        ch_hi = min(ch[-1], chainage + eps)

        x_lo = np.interp(ch_lo, ch, x_arr)
        x_hi = np.interp(ch_hi, ch, x_arr)
        y_lo = np.interp(ch_lo, ch, y_arr)
        y_hi = np.interp(ch_hi, ch, y_arr)
        z_lo = np.interp(ch_lo, ch, z_arr)
        z_hi = np.interp(ch_hi, ch, z_arr)

        tangent = np.array([x_hi - x_lo, y_hi - y_lo, z_hi - z_lo])
        norm = np.linalg.norm(tangent)
        if norm > 0:
            tangent = tangent / norm
        return tangent


@dataclass
class ParametricTunnel:
    """A complete 3D tunnel defined by alignment + varying cross-sections.

    The tunnel is built by sweeping a 2D cross-section profile along a 3D
    centerline, generating a watertight 3D mesh.
    """

    alignment: TunnelAlignment
    section: TunnelSection = field(default_factory=TunnelSection)
    name: str = "Tunnel"

    # Internal mesh data
    vertices: np.ndarray | None = field(default=None, repr=False)  # (N, 3) — x,y,z
    faces: np.ndarray | None = field(default=None, repr=False)     # (M, 3) — triangle indices
    vertex_labels: np.ndarray | None = field(default=None, repr=False)  # surface labels

    def build_mesh(self, include_layers: bool = False) -> "ParametricTunnel":
        """Sweep the cross-section along the centerline to generate a 3D mesh.

        Parameters
        ----------
        include_layers : bool
            If True, also generate support layer meshes.

        Returns
        -------
        self
        """
        ch, cx, cy, cz = self.alignment.sample()
        profiles = self.section.generate_profile(include_layers=include_layers)
        n_verts_per_section = self.section.n_vertices + 1  # includes closing point
        n_sections = len(ch)

        all_verts = []
        all_faces = []
        all_labels = []

        for layer_idx, profile_2d in enumerate(profiles):
            label = ["excavation", "shotcrete", "lining"][layer_idx] if include_layers else "excavation"
            verts, faces = self._sweep_profile(
                profile_2d, ch, cx, cy, cz,
                vert_offset=len(all_verts) if all_verts else 0,
            )
            all_verts.append(verts)
            all_faces.append(faces)
            all_labels.append(label)

        self.vertices = np.vstack(all_verts).astype(np.float32)
        self.faces = np.vstack(all_faces).astype(np.int32)
        self.vertex_labels = np.array([
            label for label, v in zip(all_labels, all_verts)
            for _ in range(len(v))
        ])

        return self

    def _sweep_profile(
        self,
        profile_2d: np.ndarray,  # (n_ring, 2) — (y, z) in section coords
        ch: np.ndarray,
        cx: np.ndarray,
        cy: np.ndarray,
        cz: np.ndarray,
        vert_offset: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sweep a 2D profile along centerline, generating vertices and triangles.

        The profile is always oriented perpendicular to the centerline tangent,
        with local Y pointing right and local Z pointing upward.
        """
        n_ring = len(profile_2d)
        n_sections = len(ch)

        # Allocate
        vertices = np.zeros((n_sections * n_ring, 3), dtype=np.float64)

        # Precompute tangent frames
        for i in range(n_sections):
            if i == 0:
                tangent = self.alignment.tangent_at(ch[0])
            elif i == n_sections - 1:
                tangent = self.alignment.tangent_at(ch[-1])
            else:
                tangent = np.array([
                    cx[i+1] - cx[i-1],
                    cy[i+1] - cy[i-1],
                    cz[i+1] - cz[i-1],
                ])
                norm = np.linalg.norm(tangent)
                if norm > 0:
                    tangent = tangent / norm
                else:
                    tangent = np.array([1.0, 0.0, 0.0])

            # Local frame: Y_local (horizontal perpendicular), Z_local (up perpendicular)
            # Assume world Z is up
            world_up = np.array([0.0, 0.0, 1.0])
            y_local = np.cross(tangent, world_up)
            yn = np.linalg.norm(y_local)
            if yn < 1e-6:
                y_local = np.array([0.0, 1.0, 0.0])
            else:
                y_local = y_local / yn
            z_local = np.cross(y_local, tangent)
            zn = np.linalg.norm(z_local)
            if zn > 0:
                z_local = z_local / zn

            # Transform profile points to world coordinates
            for j in range(n_ring):
                pt = profile_2d[j]
                world_pt = np.array([cx[i], cy[i], cz[i]]) + pt[0] * y_local + pt[1] * z_local
                vertices[i * n_ring + j] = world_pt

        # Generate triangle faces (quads between adjacent sections)
        faces = []
        for i in range(n_sections - 1):
            for j in range(n_ring - 1):
                v0 = vert_offset + i * n_ring + j
                v1 = vert_offset + i * n_ring + j + 1
                v2 = vert_offset + (i + 1) * n_ring + j + 1
                v3 = vert_offset + (i + 1) * n_ring + j
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

        faces_arr = np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)
        return vertices, faces_arr

    # ── properties ──────────────────────────────────────

    @property
    def mesh(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.vertices is None or self.faces is None:
            return None
        return self.vertices, self.faces

    @property
    def n_vertices(self) -> int:
        return len(self.vertices) if self.vertices is not None else 0

    @property
    def n_faces(self) -> int:
        return len(self.faces) if self.faces is not None else 0

    def summary(self) -> str:
        lines = [
            f"ParametricTunnel '{self.name}'",
            f"  Shape: {self.section.shape.value} {self.section.width:.1f}×{self.section.height:.1f}m",
            f"  Length: {self.alignment.total_length:.1f}m",
            f"  Support: shotcrete={self.section.shotcrete_thickness*1000:.0f}mm, "
            f"lining={self.section.lining_thickness*1000:.0f}mm",
        ]
        if self.vertices is not None:
            lines.append(f"  Mesh: {self.n_vertices:,} vertices, {self.n_faces:,} triangles")
        return "\n".join(lines)
