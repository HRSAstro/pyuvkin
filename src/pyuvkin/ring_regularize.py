"""BBarolo-style radial regularisation of free tilted-ring geometry.

After a free ring-by-ring fit (stage 1), geometric parameters that vary with
radius -- inclination, PA, centres, v_sys -- are replaced by a smooth
function of radius (``REGTYPE``), then stage 2 re-fits only VROT / VDISP with
that geometry held fixed. Matches GALFIT's ``TWOSTAGE`` / ``SecondStage``.

VROT and velocity dispersion are never regularised.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from .models.parameters import (
    parse_ring_param_name,
    ring_param_name,
    ring_values,
)

logger = logging.getLogger("pyuvkin")

#: Bases regularised between stage 1 and stage 2 (BBarolo: INC PA VSYS XPOS YPOS Z0).
GEOMETRY_BASES = ("inclination", "phi", "v_sys", "centre_ra", "centre_dec")

#: Bases that stay free in stage 2.
KINEMATIC_BASES = ("vrot", "velocity_dispersion", "vrad")

DEFAULT_REGTYPE = "auto"
DEFAULT_TWOSTAGE = True


def stage2_free(free: tuple[str, ...]) -> tuple[str, ...]:
    """Bases still free after geometry regularisation."""
    out = tuple(b for b in free if b in KINEMATIC_BASES)
    if "vrot" not in out:
        out = ("vrot",) + out
    return out


def needs_twostage(free: tuple[str, ...]) -> bool:
    """True if any free base is geometry that TWOSTAGE would regularise."""
    return any(b in GEOMETRY_BASES for b in free)


def parse_regtype(regtype: str | dict | None) -> dict[str, str]:
    """Map each geometry base to a method: ``auto``, ``median``, ``bezier``,
    ``constant``, or ``polyN`` (N = polynomial degree, 0 = constant).

    Accepts BBarolo strings: ``\"auto\"``, ``\"bezier\"``, ``\"median\"``,
    ``\"2\"`` (linear for INC+PA), or ``\"inc=bezier pa=median vsys=0\"``.
    """
    if regtype is None:
        regtype = DEFAULT_REGTYPE
    if isinstance(regtype, dict):
        return {b: str(regtype.get(b, "auto")).lower() for b in GEOMETRY_BASES}

    s = str(regtype).strip().lower()
    out = {b: "auto" for b in GEOMETRY_BASES}
    if not s or s == "auto":
        return out
    if "=" not in s and " " not in s.strip():
        # single token → INC and PA (BBarolo); others stay auto→median
        tok = s
        out["inclination"] = out["phi"] = _norm_method(tok)
        return out
    parts = s.replace(",", " ").split()
    if len(parts) == 1 and "=" not in parts[0]:
        out["inclination"] = out["phi"] = _norm_method(parts[0])
        return out
    for part in parts:
        if "=" not in part:
            continue
        key, _, val = part.partition("=")
        key = key.strip()
        aliases = {
            "inc": "inclination", "inclination": "inclination",
            "pa": "phi", "phi": "phi",
            "vsys": "v_sys", "v_sys": "v_sys",
            "xpos": "centre_ra", "centre_ra": "centre_ra",
            "ypos": "centre_dec", "centre_dec": "centre_dec",
        }
        base = aliases.get(key)
        if base is None or base not in GEOMETRY_BASES:
            raise ValueError(f"unknown REGTYPE key {key!r}")
        out[base] = _norm_method(val.strip())
    return out


def _norm_method(tok: str) -> str:
    tok = tok.lower().strip()
    if tok in ("auto", "median", "bezier", "constant"):
        return tok
    if tok.isdigit():
        # BBarolo: number N → polynomial of degree N (their rtype = N+1 coeffs…;
        # "0" / constant ↔ degree 0; "1" ↔ linear). Their getval does 1+atoi,
        # so input "0" → rtype 1 (constant), "1" → rtype 2 (linear).
        return f"poly{int(tok)}"
    if tok.startswith("poly") and tok[4:].isdigit():
        return tok
    raise ValueError(
        f"unknown regularisation method {tok!r}; use auto, median, bezier, "
        "constant, or an integer polynomial degree"
    )


def resolve_method(method: str, radii: np.ndarray, values: np.ndarray, base: str) -> str:
    """Expand ``auto`` the way BBarolo does for INC/PA vs everything else."""
    if method != "auto":
        return method
    n = len(radii)
    if base in ("inclination", "phi"):
        if n <= 4:
            return "constant"
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        sigma = mad * 1.4826  # madfm → sigma
        if sigma > 3.0:
            return "poly1" if n < 10 else "bezier"
        return "median"
    return "median"


def _unwrap_deg(values: np.ndarray) -> np.ndarray:
    """Unwrap a degree series so neighbouring rings are continuous."""
    rad = np.deg2rad(values)
    return np.rad2deg(np.unwrap(rad))


def regularize_profile(
    radii: np.ndarray,
    values: np.ndarray,
    method: str,
    *,
    angle: bool = False,
) -> np.ndarray:
    """Smooth ``values`` vs ``radii``; return one value per input radius.

    ``method`` must already be resolved (not ``auto``): ``median``,
    ``constant``, ``bezier``, or ``polyN``.
    """
    r = np.asarray(radii, float).ravel()
    y = np.asarray(values, float).ravel()
    if r.size != y.size:
        raise ValueError("radii and values must have the same length")
    if r.size == 0:
        return y.copy()
    if angle:
        y = _unwrap_deg(y)
        y0 = float(y[0])
    method = str(method).lower()
    if method == "auto":
        method = "median"

    if method in ("median", "constant"):
        out = np.full_like(y, float(np.median(y)))
    elif method == "bezier":
        out = _bezier_through(r, y)
    elif method.startswith("poly"):
        deg = int(method[4:])
        out = _poly_fit(r, y, deg)
    else:
        raise ValueError(f"unknown regularisation method {method!r}")
    if angle:
        # fold back near the unwrapped stage-1 branch (avoid 200° → -160°)
        out = out - 360.0 * np.round((out - y0) / 360.0)
    return out


def _poly_fit(r: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    n = r.size
    degree = max(0, int(degree))
    if n <= degree:
        degree = 0
    rr, yy = r.copy(), y.copy()
    if n - 2 > degree and n >= 3:
        # BBarolo drops the max and min before fitting
        keep = np.ones(n, dtype=bool)
        keep[int(np.argmax(yy))] = False
        keep[int(np.argmin(yy))] = False
        if keep.sum() > degree:
            rr, yy = rr[keep], yy[keep]
    coeff = np.polyfit(rr, yy, degree)
    return np.polyval(coeff, r)


def _bezier_through(r: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Degree-(n-1) Bezier through the points in ring order (BBarolo).

    Evaluated at equal parameter spacing; y[i] replaces the i-th ring value.
    """
    n = y.size
    if n == 1:
        return y.copy()
    bc = _log_binomial(n)
    out = np.empty(n)
    for i in range(n):
        sr = i / (n - 1)
        if sr == 0.0:
            out[i] = y[0]
        elif sr == 1.0:
            out[i] = y[-1]
        else:
            log_dsr_n = (n - 1) * math.log(1.0 - sr)
            log_sr_dsr = math.log(sr) - math.log(1.0 - sr)
            py = 0.0
            for j in range(n):
                u = math.exp(bc[j] + log_dsr_n + j * log_sr_dsr)
                py += y[j] * u
            out[i] = py
    return out


def _log_binomial(points: int) -> list[float]:
    n = points - 1
    coeff = [0.0] * points
    e = n // 2
    for k in range(e):
        coeff[k + 1] = coeff[k] + math.log((n - k) / (k + 1))
    for k in range(n, e - 1, -1):
        coeff[k] = coeff[n - k]
    return coeff


def regularize_geometry(
    instance,
    radii: np.ndarray,
    free: tuple[str, ...],
    regtype: str | dict | None = "auto",
) -> dict[str, np.ndarray]:
    """Regularise every free geometry base on ``instance``.

    Returns ``{base: array(n_rings)}`` for bases that were free per ring.
    """
    methods = parse_regtype(regtype)
    n = len(np.asarray(radii).ravel())
    out: dict[str, np.ndarray] = {}
    for base in GEOMETRY_BASES:
        if base not in free:
            continue
        if not hasattr(instance, ring_param_name(base, 0)):
            continue
        vals = ring_values(instance, base, n)
        method = resolve_method(methods[base], np.asarray(radii, float), vals, base)
        smoothed = regularize_profile(
            radii, vals, method, angle=(base == "phi"),
        )
        logger.info(
            "  regularise %s (%s): %s → %s",
            base, method,
            np.array2string(vals, precision=2),
            np.array2string(smoothed, precision=2),
        )
        out[base] = smoothed
    return out


def geometry_fixed_from_regularized(
    regularized: dict[str, np.ndarray],
) -> dict[str, float]:
    """Flat ``{inclination_0: …, phi_1: …}`` dict for ``build_model(..., fixed=)``."""
    fixed = {}
    for base, arr in regularized.items():
        for i, v in enumerate(np.asarray(arr, float).ravel()):
            fixed[ring_param_name(base, i)] = float(v)
    return fixed


def stage1_geometry_record(instance, free: tuple[str, ...], n_rings: int) -> dict:
    """Stage-1 free geometry before regularisation, for the outputs JSON."""
    out = {}
    for base in GEOMETRY_BASES:
        if base not in free:
            continue
        if hasattr(instance, ring_param_name(base, 0)):
            out[base] = ring_values(instance, base, n_rings).tolist()
    return out
