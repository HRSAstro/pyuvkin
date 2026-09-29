"""Asymmetric drift correction (Iorio et al. 2017 §4.3; BBarolo ADRIFT).

Post-processes a fitted tilted-ring rotation curve. The observed ``vrot`` is
the mean azimuthal streaming; pressure support makes it lag the circular
speed. BBarolo / Iorio+17:

    ADC² = −R σ² ( d ln σ² / dR + d ln Σ / dR )
    V_circ = √( V_rot² + ADC² )

where ``σ`` is the velocity dispersion and ``Σ`` is the face-on surface
density (BBarolo: elliptical-annulus median × cos i). Derivatives are taken
from polynomial fits (``adrift_pol1`` for σ², ``adrift_pol2`` for Σ;
``pol1=-1`` keeps the raw σ² profile and uses a finite-difference slope).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .conventions import disc_coordinates
from .models.parameters import ring_mean, ring_values, ring_vrot_kms

logger = logging.getLogger("pyuvkin")

DEFAULT_ADRIFT = False
DEFAULT_ADRIFT_POL1 = 3
DEFAULT_ADRIFT_POL2 = 3


def elliptical_ring_density(
    sb_map: np.ndarray,
    yy: np.ndarray,
    xx: np.ndarray,
    radii: np.ndarray,
    *,
    centre_ra: float,
    centre_dec: float,
    phi_deg: float,
    inclination_deg: float,
) -> np.ndarray:
    """Median surface brightness in elliptical annuli around each ring radius.

    Annulus half-width is half the local ring spacing (edge rings use the
    neighbouring gap). Matches BBarolo Ellprof's per-ring median used as
    ``densprof`` before the cos i face-on correction.
    """
    sb = np.asarray(sb_map, float)
    r = np.asarray(radii, float).ravel()
    if sb.ndim != 2 or r.size == 0:
        return np.zeros_like(r)
    r_gal, _, _ = disc_coordinates(
        yy, xx, centre_ra, centre_dec, phi_deg, inclination_deg,
    )
    # ring half-widths
    if r.size == 1:
        half = np.array([max(float(r[0]), 1e-6) * 0.5])
    else:
        gaps = np.diff(r)
        half = np.empty(r.size)
        half[0] = gaps[0] / 2.0
        half[-1] = gaps[-1] / 2.0
        if r.size > 2:
            half[1:-1] = 0.5 * np.minimum(gaps[:-1], gaps[1:])
    out = np.zeros(r.size, dtype=float)
    lit = np.isfinite(sb) & (sb > 0)
    for i, (ri, hi) in enumerate(zip(r, half)):
        hi = max(float(hi), 1e-9)
        sel = lit & (np.abs(r_gal - ri) <= hi)
        if sel.any():
            out[i] = float(np.median(sb[sel]))
        elif lit.any():
            # fallback: nearest lit pixels by deprojected radius
            d = np.abs(r_gal[lit] - ri)
            k = min(16, int(d.size))
            out[i] = float(np.median(sb[lit].ravel()[np.argpartition(d.ravel(), k - 1)[:k]]))
    return out


def analytic_ring_density(radii: np.ndarray, scale_radius: float) -> np.ndarray:
    """Relative exponential dens profile (absolute scale cancels in dln Σ / dR)."""
    h = max(float(scale_radius), 1e-4)
    return np.exp(-np.asarray(radii, float) / h)


def _poly_values_and_deriv(r: np.ndarray, y: np.ndarray, degree: int):
    """Fit ``y(r)`` with a polynomial of ``degree``; return (y_fit, dy/dr)."""
    r = np.asarray(r, float).ravel()
    y = np.asarray(y, float).ravel()
    n = r.size
    degree = int(degree)
    if degree < 0 or n < 2:
        # finite difference of the raw profile
        return y.copy(), np.gradient(y, r)
    degree = min(degree, n - 1)
    # numpy polyfit: highest power first
    coeffs = np.polyfit(r, y, degree)
    y_fit = np.polyval(coeffs, r)
    dcoeffs = np.polyder(coeffs)
    dy = np.polyval(dcoeffs, r) if dcoeffs.size else np.zeros_like(r)
    return y_fit, dy


def asymmetric_drift_correction(
    radii,
    dens,
    vdisp,
    vrot,
    inclination,
    *,
    pol1: int = DEFAULT_ADRIFT_POL1,
    pol2: int = DEFAULT_ADRIFT_POL2,
) -> dict:
    """Iorio+17 / BBarolo asymmetric drift at each ring.

    Parameters
    ----------
    radii : arcsec
    dens : projected surface-density tracer (Ellprof median or relative dens)
    vdisp, vrot : km/s
    inclination : deg (per ring or scalar)
    pol1, pol2 : polynomial degrees for σ² and Σ_FO (BBarolo ADRIFTPOL1/2);
        ``pol1=-1`` skips smoothing σ².

    Returns a dict of equal-length arrays plus metadata.
    """
    r = np.asarray(radii, float).ravel()
    dens = np.asarray(dens, float).ravel()
    sig = np.asarray(vdisp, float).ravel()
    vrot = np.asarray(vrot, float).ravel()
    inc = np.asarray(inclination, float).ravel()
    n = r.size
    if not (dens.size == sig.size == vrot.size == n):
        raise ValueError("radii, dens, vdisp, vrot must have the same length")
    if inc.size == 1:
        inc = np.full(n, float(inc[0]))
    elif inc.size != n:
        raise ValueError("inclination must be scalar or length n_rings")
    if n < 2:
        raise ValueError("asymmetric drift needs at least 2 rings")

    # face-on surface density (BBarolo: dens × cos i)
    cosi = np.cos(np.radians(inc))
    cosi = np.clip(cosi, 1e-3, None)  # avoid zero at edge-on
    sigma_fo = dens * cosi
    if not np.any(sigma_fo > 0):
        raise ValueError("asymmetric drift: surface-density profile is empty")

    disp2 = sig * sig
    pol1, pol2 = int(pol1), int(pol2)
    if pol1 < 0:
        disp2_reg, disp2_der = disp2.copy(), np.gradient(disp2, r)
    else:
        disp2_reg, disp2_der = _poly_values_and_deriv(r, disp2, pol1)
    # keep σ² positive for logs / ratios
    disp2_reg = np.maximum(disp2_reg, 1e-12)

    # Σ_FO always polynomial-smoothed (BBarolo requires ADRIFTPOL2 >= 0)
    deg2 = max(0, pol2)
    sigma_reg, sigma_der = _poly_values_and_deriv(r, sigma_fo, deg2)
    sigma_reg = np.maximum(sigma_reg, 1e-30)

    adc2 = -r * disp2_reg * (disp2_der / disp2_reg + sigma_der / sigma_reg)
    vcirc2 = vrot * vrot + adc2
    bad = vcirc2 < 0
    if np.any(bad):
        logger.warning(
            "asymmetric drift: V_circ² < 0 at %d ring(s); those V_circ set to NaN",
            int(bad.sum()),
        )
    vcirc = np.where(bad, np.nan, np.sqrt(np.maximum(vcirc2, 0.0)))

    return {
        "radii_arcsec": r,
        "vrot_kms": vrot,
        "vcirc_kms": vcirc,
        "adc2_kms2": adc2,
        "disp2_kms2": disp2,
        "disp2_reg_kms2": disp2_reg,
        "dens": dens,
        "sigma_fo": sigma_fo,
        "sigma_fo_reg": sigma_reg,
        "inclination_deg": inc,
        "adrift_pol1": pol1,
        "adrift_pol2": pol2,
    }


def dens_profile_for_fit(renderer, best, radii: np.ndarray) -> np.ndarray:
    """Surface-density tracer for ADRIFT from the renderer / best-fit geometry."""
    n = int(radii.size)
    cra = ring_mean(best, "centre_ra", n)
    cdec = ring_mean(best, "centre_dec", n)
    phi = ring_mean(best, "phi", n)
    # use mean inclination for the elliptical annuli (BBarolo Ellprof uses mean geometry)
    inc = ring_mean(best, "inclination", n)
    sb_map = getattr(renderer, "_sb_map", None)
    if sb_map is not None and getattr(renderer, "_yy", None) is not None:
        return elliptical_ring_density(
            sb_map, renderer._yy, renderer._xx, radii,
            centre_ra=cra, centre_dec=cdec, phi_deg=phi, inclination_deg=inc,
        )
    scale = float(getattr(best, "scale_radius", 0.3) or 0.3)
    return analytic_ring_density(radii, scale)


def compute_for_rings(
    renderer, best, *, pol1: int = DEFAULT_ADRIFT_POL1, pol2: int = DEFAULT_ADRIFT_POL2,
) -> dict:
    """Run ADRIFT on a free-ring best-fit instance."""
    if not getattr(renderer, "free_rings", False):
        raise ValueError("asymmetric drift requires rotation_curve 'rings'")
    radii = np.asarray(renderer._radii(best), float)
    dens = dens_profile_for_fit(renderer, best, radii)
    vrot = ring_vrot_kms(best, radii.size)
    vdisp = ring_values(best, "velocity_dispersion", radii.size)
    inc = ring_values(best, "inclination", radii.size)
    return asymmetric_drift_correction(
        radii, dens, vdisp, vrot, inc, pol1=pol1, pol2=pol2,
    )


def write_asymdrift_txt(result: dict, path: Path) -> Path:
    """BBarolo-style ``asymdrift.txt``."""
    path = Path(path)
    cols = (
        "RAD(arcs)", "VCIRC(km/s)", "ASYMDRIFT_SQ(km/s)2", "DISP2(km/s)2",
        "DISP2_REG(km/s)2", "DENSPROF", "DENSPROF_REG",
    )
    rows = zip(
        result["radii_arcsec"], result["vcirc_kms"], result["adc2_kms2"],
        result["disp2_kms2"], result["disp2_reg_kms2"],
        result["sigma_fo"], result["sigma_fo_reg"],
    )
    with path.open("w") as fh:
        fh.write("#" + "".join(f"{c:>19}" for c in cols) + "\n")
        for row in rows:
            fh.write("".join(f"{float(v):20.4f}" for v in row) + "\n")
    return path


def result_as_jsonable(result: dict) -> dict:
    out = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            out[k] = [None if not np.isfinite(x) else float(x) for x in v]
        else:
            out[k] = v
    return out
