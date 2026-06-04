"""Rock mass classification — BQ (GB/T 50218), RMR, and geohazard indices."""

from __future__ import annotations

import numpy as np


# ── BQ Classification (China National Standard GB/T 50218) ──

def _kv_from_vp(vp: np.ndarray) -> np.ndarray:
    """Integrity coefficient Kv estimated from Vp.

    Kv = (Vp_rock / Vp_intact) ^ 2
    Default Vp_intact ≈ 5000 m/s for hard rock (adjustable).
    """
    vp_intact = 5600.0  # typical intact rock P-wave velocity (m/s)
    with np.errstate(invalid="ignore"):
        kv = (vp / vp_intact) ** 2
    return np.clip(kv, 0.0, 1.0)


def classify_bq(
    vp: np.ndarray,
    vp_intact: float = 5600.0,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute BQ index and rock-mass class per GB/T 50218-2014.

    BQ = 90 + 3 * Vp_km/s + 250 * Kv

    Correction:
      [BQ] = BQ - 100 * (k1 + k2 + k3)

    Classes:
      I:   [BQ] > 550   (very good)
      II:  550 >= [BQ] > 450
      III: 450 >= [BQ] > 350
      IV:  350 >= [BQ] > 250
      V:   [BQ] <= 250  (very poor)

    Parameters
    ----------
    vp : np.ndarray
        P-wave velocity (m/s). Shape (ny, nx, nz) or any.
    vp_intact : float
        Intact rock P-wave velocity for Kv calculation.
    k1, k2, k3 : float
        Correction factors (joint orientation, groundwater, in-situ stress).
        Typical range 0.0–0.6 each.

    Returns
    -------
    bq_index : np.ndarray (float)
    bq_class : np.ndarray (int) 1-5
    """
    vp_kms = vp / 1000.0
    kv = _kv_from_vp(vp)
    bq = 90.0 + 3.0 * vp_kms + 250.0 * kv
    bq_corrected = bq - 100.0 * (k1 + k2 + k3)

    bq_class = np.full_like(bq_corrected, 0, dtype=np.int32)
    bq_class = np.where(bq_corrected > 550, 1, bq_class)
    bq_class = np.where((bq_corrected <= 550) & (bq_corrected > 450), 2, bq_class)
    bq_class = np.where((bq_corrected <= 450) & (bq_corrected > 350), 3, bq_class)
    bq_class = np.where((bq_corrected <= 350) & (bq_corrected > 250), 4, bq_class)
    bq_class = np.where(bq_corrected <= 250, 5, bq_class)

    return bq_corrected.astype(np.float32), bq_class


_BQ_CLASS_LABELS: dict[int, str] = {
    1: "I - Very Good",
    2: "II - Good",
    3: "III - Fair",
    4: "IV - Poor",
    5: "V - Very Poor",
}


def bq_class_label(cls: int) -> str:
    return _BQ_CLASS_LABELS.get(cls, "Unknown")


# ── RMR (Bieniawski 1989) ─────────────────────────────────

def classify_rmr(
    vp: np.ndarray,
    ucs: float | np.ndarray = 100.0,
    rqd: float | np.ndarray = 90.0,
    joint_spacing: float | np.ndarray = 1.0,
    joint_condition: float = 20.0,
    groundwater: float = 10.0,
    joint_orientation: float = -5.0,
) -> np.ndarray:
    """Estimate RMR from Vp plus optional geomechanical inputs.

    Simplified: rating1 (UCS) + rating2 (RQD) + rating3 (joint spacing)
               + rating4 (joint cond) + rating5 (groundwater) + adjustment.

    Uses Vp-to-UCS empirical relation when ucs is not provided:
      UCS ≈ 0.03 * Vp^1.5   (MPa)   for granitic/hard rocks.

    Returns
    -------
    rmr : np.ndarray (float) 0-100
    """
    # Vp → UCS rating (simplified)
    # Rating table for UCS (MPa): >250→15, 100-250→12, 50-100→7, 25-50→4, 5-25→2, <5→1
    ucs_arr = np.asarray(ucs, dtype=np.float64)
    if ucs_arr.ndim == 0:
        ucs_arr = np.full_like(vp, ucs_arr)

    # Estimate UCS from Vp where not provided explicitly
    ucs_from_vp = 0.03 * np.power(vp, 1.5)  # MPa
    ucs_arr = np.where(ucs_arr < 1.0, ucs_from_vp, ucs_arr)

    r1 = np.zeros_like(ucs_arr)
    r1 = np.where(ucs_arr > 250, 15.0, r1)
    r1 = np.where((ucs_arr <= 250) & (ucs_arr > 100), 12.0, r1)
    r1 = np.where((ucs_arr <= 100) & (ucs_arr > 50), 7.0, r1)
    r1 = np.where((ucs_arr <= 50) & (ucs_arr > 25), 4.0, r1)
    r1 = np.where((ucs_arr <= 25) & (ucs_arr > 5), 2.0, r1)
    r1 = np.where(ucs_arr <= 5, 1.0, r1)
    r1 = np.where(ucs_arr < 1.0, 0.0, r1)

    # RQD rating (simplified: 90-100→20, 75-90→17, 50-75→13, 25-50→8, <25→3)
    rqd_arr = np.asarray(rqd, dtype=np.float64)
    r2 = np.where(rqd_arr >= 90, 20.0, 17.0)
    r2 = np.where(rqd_arr < 75, 13.0, r2)
    r2 = np.where(rqd_arr < 50, 8.0, r2)
    r2 = np.where(rqd_arr < 25, 3.0, r2)

    # Joint spacing rating (m) (>2→20, 0.6-2→15, 0.2-0.6→10, 0.06-0.2→8, <0.06→5)
    js = np.asarray(joint_spacing, dtype=np.float64)
    r3 = np.where(js > 2.0, 20.0, 15.0)
    r3 = np.where(js <= 0.6, 10.0, r3)
    r3 = np.where(js <= 0.2, 8.0, r3)
    r3 = np.where(js <= 0.06, 5.0, r3)

    rmr = r1 + r2 + r3 + float(joint_condition) + float(groundwater) + float(joint_orientation)
    return np.clip(rmr, 0.0, 100.0).astype(np.float32)


# ── Composite Geohazard Index ──────────────────────────────

def geohazard_index(
    poisson: np.ndarray,
    young_modulus: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:
    """Composite geohazard risk index (dimensionless, 0-1).

    Based on three indicators:
      - Poisson ratio anomaly (higher → more fractured/weathered)
      - Low Young's modulus (softer rock → higher risk)
      - Density deficit (lower density → cavity/weathered zone)

    Parameters
    ----------
    poisson : np.ndarray
        Poisson's ratio field.
    young_modulus : np.ndarray
        Young's modulus (GPa).
    density : np.ndarray
        Density (kg/m³).

    Returns
    -------
    index : np.ndarray
        Geohazard index 0 (low risk) to 1 (high risk).
    """
    # Normalize each indicator to [0, 1] then combine
    poisson_norm = _minmax_norm(poisson)
    young_inv_norm = _minmax_norm(-young_modulus)  # invert: lower E → higher risk
    density_inv_norm = _minmax_norm(-density)

    # Weighted combination
    index = 0.35 * poisson_norm + 0.35 * young_inv_norm + 0.30 * density_inv_norm
    return np.clip(index, 0.0, 1.0).astype(np.float32)


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    amin, amax = np.nanmin(arr), np.nanmax(arr)
    if np.isclose(amax, amin):
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - amin) / (amax - amin)).astype(np.float32)
