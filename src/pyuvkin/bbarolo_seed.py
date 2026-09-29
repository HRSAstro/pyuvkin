"""Seed pyuvkin free-ring starts from BBarolo's built-in FitMod3D search.

Runs 3DFIT on a dirty cube (image-plane) and maps the best rings onto
pyuvkin parameter names for ``search.start``. The uv-plane L-BFGS / sampler
then refines that seed. Not a replacement for the likelihood — geometry and
VROT from FitMod3D are often good enough to start, while pathological inner
rings are pulled back by the visibility fit.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

from . import conventions
from .models.bbarolo import _silence_stdio
from .models.parameters import parse_free_ring_params, parse_ring_param_name, ring_param_name

logger = logging.getLogger("pyuvkin")

SEED_START_NAMES = ("bbarolo", "fitmod3d")


def is_bbarolo_seed(start) -> bool:
    return isinstance(start, str) and start.strip().lower() in SEED_START_NAMES


def _estimate_beam_deg(uvd, geometry) -> tuple[float, float, float]:
    """Natural-weight beam guess (deg): λ / B_95 as circular FWHM."""
    try:
        # max baseline in wavelengths at reference → θ ≈ 1/B rad
        b_lam = float(getattr(uvd, "max_baseline_wavelengths", 0) or 0)
        if b_lam <= 0 and hasattr(uvd, "uvw"):
            uv = np.asarray(uvd.uvw, float)
            b_lam = float(np.nanmax(np.hypot(uv[..., 0], uv[..., 1])))
        if b_lam > 0:
            fwhm_rad = 1.0 / b_lam
            fwhm_deg = float(np.degrees(fwhm_rad))
            return fwhm_deg, fwhm_deg, 0.0
    except Exception:
        pass
    # fallback: ~2.5 image pixels
    ps = float(geometry.pixel_scale) / 3600.0
    return 2.5 * ps, 2.5 * ps, 0.0


def _dirty_cube_to_vrad_fits(
    dirty_native: np.ndarray,
    uvd,
    geometry,
    spectral,
    path: Path,
    *,
    rest_frequency_hz: float | None = None,
) -> Path:
    """Write a FitMod3D-ready dirty cube (FITS orientation, VRAD, beam).

    Velocity axis uses pyuvkin's residual ``spectral.velocities_kms`` so
    FitMod3D ``VSYS`` is in the same frame as ``v_sys`` (not absolute radio
    velocity from rest frequency).
    """
    from pyuvimage.products import build_header, to_fits_orientation

    cube = np.stack([to_fits_orientation(c) for c in dirty_native]).astype(np.float32)
    freqs = np.asarray(uvd.frequencies, float)
    h = build_header(
        n_pix=geometry.n_pixels, pixel_scale_arcsec=geometry.pixel_scale,
        meta=uvd.meta, bunit="Jy/beam", frequencies_hz=freqs,
    )
    bmaj, bmin, bpa = _estimate_beam_deg(uvd, geometry)
    h["BMAJ"], h["BMIN"], h["BPA"] = bmaj, bmin, bpa
    v = np.asarray(spectral.velocities_kms, float)
    h["CTYPE3"] = "VRAD"
    h["CUNIT3"] = "km/s"
    h["CRVAL3"] = float(v[0])
    h["CDELT3"] = float(v[1] - v[0]) if v.size > 1 else float(spectral.dv_kms)
    h["CRPIX3"] = 1.0
    rest = rest_frequency_hz or uvd.meta.get("rest_frequency_hz")
    if rest:
        h["RESTFRQ"] = float(rest)
    path = Path(path)
    fits.writeto(path, cube, h, overwrite=True)
    return path


def _free_to_bbarolo(free: tuple[str, ...]) -> str:
    aliases = {
        "vrot": "VROT", "velocity_dispersion": "VDISP", "vrad": "VRAD",
        "inclination": "INC", "phi": "PA",
        "centre_ra": "XPOS", "centre_dec": "YPOS", "v_sys": "VSYS",
    }
    return " ".join(aliases[b] for b in free if b in aliases)


def _angle_into_prior(angle: float, lo: float, hi: float) -> float:
    """Map an E-of-N angle into ``[lo, hi]`` via 360° wrapping when possible."""
    a = float(angle) % 360.0
    candidates = (a, a - 360.0, a + 360.0)
    in_range = [c for c in candidates if lo <= c <= hi]
    if in_range:
        return float(in_range[0])
    mid = 0.5 * (lo + hi)
    return float(min(candidates, key=lambda c: abs(c - mid)))


def _clip_to_model(start: dict, model) -> dict:
    """Clip seed values into each free parameter's prior support."""
    import autofit as af

    out = {}
    for name, value in start.items():
        prior = getattr(model, name, None)
        if not isinstance(prior, af.Prior):
            continue
        v = float(value)
        lo = getattr(prior, "lower_limit", None)
        hi = getattr(prior, "upper_limit", None)
        base = parse_ring_param_name(name)
        base_name = base[0] if base is not None else name
        if base_name == "phi" and lo is not None and hi is not None:
            v = _angle_into_prior(v, float(lo), float(hi))
        else:
            if lo is not None:
                v = max(v, float(lo))
            if hi is not None:
                v = min(v, float(hi))
        out[name] = v
    return out


def rings_from_fitmod3d(
    *,
    dirty_native: np.ndarray,
    uvd,
    geometry,
    spectral,
    radii_arcsec: np.ndarray,
    free,
    out_dir: Path | None = None,
    rest_frequency_hz: float | None = None,
    scale_height_arcsec: float = 0.0,
    twostage: bool = True,
    mask: str = "SMOOTH",
    norm: str = "LOCAL",
    init_centre_ra: float = 0.0,
    init_centre_dec: float = 0.0,
    init_v_sys: float = 0.0,
    init_inclination: float = 45.0,
    init_phi: float = 0.0,
    init_vrot: float = 200.0,
    init_vdisp: float = 30.0,
) -> dict:
    """Run FitMod3D and return ``{start, rings, paths, ...}``."""
    try:
        from pyBBarolo import FitMod3D
    except ImportError as e:  # pragma: no cover
        raise ImportError("FitMod3D seed needs pyBBarolo") from e

    free = parse_free_ring_params(free)
    radii = np.asarray(radii_arcsec, float).ravel()
    if radii.size < 1:
        raise ValueError("FitMod3D seed needs at least one ring radius")

    if out_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="pkbb_seed_")
        out_dir = Path(tmp.name)
        _tmpdir = tmp
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _tmpdir = None

    try:
        cube_path = _dirty_cube_to_vrad_fits(
            dirty_native, uvd, geometry, spectral,
            out_dir / "seed_dirty_vrad.fits",
            rest_frequency_hz=rest_frequency_hz,
        )
        ny, nx = int(geometry.shape[0]), int(geometry.shape[1])
        ps = float(geometry.pixel_scale)
        y_g, x_g = conventions.sky_to_grid(init_centre_ra, init_centre_dec)
        xpos = (nx - 1) / 2.0 + x_g / ps
        ypos = (ny - 1) / 2.0 + y_g / ps

        logger.info(
            "FitMod3D seed: %d rings, free=%s, writing under %s",
            radii.size, list(free), out_dir,
        )
        fit = FitMod3D(str(cube_path))
        fit.init(
            radii=radii, xpos=xpos, ypos=ypos, vsys=float(init_v_sys),
            vrot=float(init_vrot), vdisp=float(init_vdisp), vrad=0.0,
            z0=float(scale_height_arcsec),
            inc=float(init_inclination), phi=float(init_phi),
        )
        fit.set_options(
            free=_free_to_bbarolo(free),
            norm=norm, mask=mask, twostage=bool(twostage),
            polyn="bezier", smooth=True, ltype=1, wfunc=2, ftype=2,
            outfolder=str(out_dir) + "/",
        )
        with _silence_stdio():
            fit.compute(threads=1)

        rings = _read_rings_table(out_dir)
        start = _rings_to_start(rings, free, geometry)
        return {
            "start": start,
            "rings": rings,
            "radii_arcsec": radii.tolist(),
            "free": list(free),
            "cube_fits": str(cube_path),
            "out_dir": str(out_dir),
        }
    finally:
        if _tmpdir is not None:
            # keep outputs only when caller passed out_dir; temp cleaned up
            _tmpdir.cleanup()


def _read_rings_table(out_dir: Path) -> dict:
    """Parse rings_final2.txt (preferred) or rings_final1.txt."""
    path = out_dir / "rings_final2.txt"
    if not path.exists():
        path = out_dir / "rings_final1.txt"
    if not path.exists():
        raise RuntimeError(f"FitMod3D produced no ring file under {out_dir}")
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(x) for x in line.split()])
    if not rows:
        raise RuntimeError(f"empty ring file {path}")
    arr = np.asarray(rows, float)
    # columns: RAD(Kpc) RAD(arcs) VROT DISP INC PA Z0(pc) Z0(arcs) SIG XPOS YPOS VSYS VRAD
    return {
        "path": str(path),
        "radii_arcsec": arr[:, 1],
        "vrot": arr[:, 2],
        "velocity_dispersion": arr[:, 3],
        "inclination": arr[:, 4],
        "phi": arr[:, 5],
        "xpos": arr[:, 9],
        "ypos": arr[:, 10],
        "v_sys": arr[:, 11],
        "vrad": arr[:, 12] if arr.shape[1] > 12 else np.zeros(len(arr)),
    }


def _rings_to_start(rings: dict, free: tuple[str, ...], geometry) -> dict:
    """Map FitMod3D ring arrays → pyuvkin start dict (per-ring or shared)."""
    n = len(rings["radii_arcsec"])
    ny, nx = int(geometry.shape[0]), int(geometry.shape[1])
    ps = float(geometry.pixel_scale)
    # mean centre from FitMod3D pixels → sky
    xpos = float(np.mean(rings["xpos"]))
    ypos = float(np.mean(rings["ypos"]))
    x_g = (xpos - (nx - 1) / 2.0) * ps
    y_g = (ypos - (ny - 1) / 2.0) * ps
    centre_ra, centre_dec = conventions.grid_to_sky(y_g, x_g)

    start: dict[str, float] = {}

    def put(base: str, values):
        values = np.asarray(values, float).ravel()
        if base in free:
            for i in range(n):
                start[ring_param_name(base, i)] = float(values[min(i, values.size - 1)])
        else:
            start[base] = float(np.mean(values))

    put("vrot", rings["vrot"])
    put("velocity_dispersion", rings["velocity_dispersion"])
    put("vrad", rings["vrad"])
    put("inclination", rings["inclination"])
    # PA: BBarolo E of N ↔ pyuvkin phi (keep [0, 360); clip maps into prior)
    put("phi", [float(a) % 360.0 for a in rings["phi"]])
    put("v_sys", rings["v_sys"])
    if "centre_ra" in free:
        for i in range(n):
            xi = float(rings["xpos"][i])
            yi = float(rings["ypos"][i])
            xgi = (xi - (nx - 1) / 2.0) * ps
            ygi = (yi - (ny - 1) / 2.0) * ps
            cra, cdec = conventions.grid_to_sky(ygi, xgi)
            start[ring_param_name("centre_ra", i)] = cra
            start[ring_param_name("centre_dec", i)] = cdec
    else:
        start["centre_ra"] = centre_ra
        start["centre_dec"] = centre_dec
    return start


def seed_start_for_model(
    model, seed: dict, *, clip: bool = True,
) -> dict:
    """Keep only free parameters; optionally clip to prior bounds."""
    start = {k: float(v) for k, v in seed["start"].items()}
    if clip:
        start = _clip_to_model(start, model)
    return start
