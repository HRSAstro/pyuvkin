"""3DBarolo backend (Di Teodoro & Fraternali 2015) through pyBBarolo's GALMOD.

Rings are synthesised from the shared parameters -- exponential surface density
and arctan rotation curve sampled at one ring per render pixel out to an outer
radius set by ``options.rmax`` (arcsec; preferred) or ``n_scale_lengths``
(``"auto"`` by default: freeform maps use the lit-map extent; analytic discs
cover most of the image field) -- and GALMOD builds the unsmoothed cube on a
blank template with a VRAD spectral axis. No beam smoothing is applied (the
PSF lives in the uv operator); the cube is rescaled so ``cube.sum() * dv``
equals ``intensity``. Ported from LensKin's ``bbarolo_utils``.

Freeform surface brightness: GalMod still builds an axi-symmetric ring cube
for the kinematics, then each spatial pixel's spectrum is rescaled so its
integrated flux matches the freeform map (BBarolo's ``NORM=LOCAL``, but
against the SB map rather than the data). Pixels the rings do not cover stay
zero even if the map is lit there, so the rings are extended to the map's
outer lit radius from the disc centre.

Optional: needs pyBBarolo and a built libBBarolo, which are not on PyPI for
every platform. Nothing here is imported unless the backend is asked for.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import sys
from functools import lru_cache

import numpy as np

from .. import conventions
from .base import Renderer
from .parameters import (
    DiscParameters,
    DEFAULT_FREE_RING_PARAMS,
    parse_free_ring_params,
    ring_parameter_names,
    ring_values,
    ring_vrot_kms,
    rotation_curve_kms,
)
from .sb import FreeformSB

try:
    from pyBBarolo import GalMod
except ImportError:  # pragma: no cover - optional dependency
    GalMod = None

_TEMPLATE_DIR = "/tmp/pkbb"
# BBarolo 1.5 strcpy'd FITS paths into char[100] (now patched to FLEN_FILENAME);
# keep templates under a short absolute path regardless.


def bbarolo_phi(phi_deg: float) -> float:
    """pyuvkin ``phi`` -> GALMOD ``phi``. On the blank-template WCS written
    here (CDELT1 < 0, i.e. east left) GALMOD's PA is the astronomical one, so
    the angle passes through; `_to_native` handles the array orientation."""
    return float(conventions.wrap_deg(phi_deg))


def apply_local_surface_brightness(
    cube: np.ndarray, target_jykms: np.ndarray, dv_kms: float,
) -> np.ndarray:
    """Rescale each spatial spectrum so ``cube.sum(0) * dv`` matches ``target``.

    Same idea as BBarolo Galfit's ``NORM=LOCAL``: the ring model supplies the
    line-of-sight velocity distribution per pixel; the map supplies the
    integrated flux. Pixels where the model has no flux stay zero.
    """
    model_mom0 = np.asarray(cube, float).sum(0) * float(dv_kms)
    target = np.asarray(target_jykms, float)
    if model_mom0.shape != target.shape:
        raise ValueError(
            f"freeform map shape {target.shape} does not match the GalMod "
            f"spatial shape {model_mom0.shape}"
        )
    scale = np.zeros_like(model_mom0)
    lit = model_mom0 > 0
    scale[lit] = target[lit] / model_mom0[lit]
    return cube * scale[None]


@contextlib.contextmanager
def _silence_stdio():
    """Mute C/C++ stdout+stderr (BBarolo's HEADER WARNINGs) for a block.

    ``contextlib.redirect_*`` only covers Python-level streams; GalMod writes
    with ``fprintf`` to the real fds. Flush C stdio before restoring so a
    half-written line (e.g. ``... Done``) cannot leak onto the real console.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = (os.dup(1), os.dup(2))
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        try:
            import ctypes
            ctypes.CDLL(None).fflush(None)
        except Exception:
            pass
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        os.close(devnull)


#: BBarolo LINEAR for Hanning-smoothed spectra (σ in channels; FWHM≈2).
HANNING_LINEAR = 0.85


def resolve_linear_channels(raw) -> float | None:
    """Parse ``options.linear`` / ``hanning``.

    Returns σ in channels, or ``None`` to leave GalMod's default alone.
    ``true`` / ``"hanning"`` → 0.85; a number is used as-is; ``false``/unset → None.

    Only enable for native-resolution (non-averaged) spectra — see docs.
    """
    if raw is None or raw is False:
        return None
    if raw is True:
        return HANNING_LINEAR
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("", "false", "none", "off", "0"):
            return None
        if s in ("true", "hanning", "on"):
            return HANNING_LINEAR
        return float(s)
    return float(raw)


def apply_instrumental_broadening(cube: np.ndarray, linear_channels: float) -> np.ndarray:
    """Convolve each spatial spectrum with a Gaussian of σ = ``linear_channels``.

    Matches BBarolo's LINEAR (σ in channels, not FWHM). Flux per spectrum is
    conserved. No-op when ``linear_channels`` ≤ 0.
    """
    from scipy.ndimage import gaussian_filter1d

    sig = float(linear_channels)
    if sig <= 0:
        return cube
    out = np.asarray(cube, float)
    # spectral axis is 0; truncate ≈ 4σ
    return gaussian_filter1d(out, sigma=sig, axis=0, mode="nearest")


@lru_cache(maxsize=32)
def blank_cube_fits_path(n_chan, n_y, n_x, pixel_scale, dv, v_min) -> str:
    """A cached zero cube with a sky + VRAD WCS matching the render grid."""
    from astropy.io import fits

    key = f"v2_n{n_chan}_y{n_y}_x{n_x}_ps{pixel_scale:.8g}_dv{dv:.8g}_v0{v_min:.8g}"
    os.makedirs(_TEMPLATE_DIR, exist_ok=True)
    path = os.path.join(_TEMPLATE_DIR, f"blank_{hashlib.sha1(key.encode()).hexdigest()[:16]}.fits")
    if os.path.exists(path):
        return path
    h = fits.Header()
    cd = pixel_scale / 3600.0
    h["CTYPE1"], h["CTYPE2"], h["CTYPE3"] = "RA---SIN", "DEC--SIN", "VRAD"
    h["CUNIT1"], h["CUNIT2"], h["CUNIT3"] = "deg", "deg", "km/s"
    # non-zero CRVAL1 avoids BBarolo edge cases at RA=0; spectral axis is VRAD
    h["CRVAL1"], h["CRVAL2"], h["CRVAL3"] = 180.0, 0.0, float(v_min)
    h["CRPIX1"], h["CRPIX2"], h["CRPIX3"] = (n_x + 1) / 2.0, (n_y + 1) / 2.0, 1.0
    h["CDELT1"], h["CDELT2"], h["CDELT3"] = -cd, cd, float(dv)
    h["BUNIT"] = "JY/PIXEL"
    # No useful beam: GalMod.smooth() is never called; leave BMAJ/BMIN unset
    # so a mistaken smooth() would fail rather than convolve with a fake beam.
    h["OBJECT"] = "pyuvkin"
    h["TELESCOP"] = "ALMA"
    h["RESTFRQ"] = 1.420405751786e9  # HI; unused for VRAD cubes, silences a warning
    fits.writeto(path, np.zeros((n_chan, n_y, n_x), dtype=np.float32), h, overwrite=True)
    return path


def galmod_ring_radii(
    p: DiscParameters,
    *,
    pixel_scale: float,
    shape: tuple[int, int],
    n_rings=None,
    n_scale_lengths="auto",
    rmax=None,
    sb_map: np.ndarray | None = None,
    yy: np.ndarray | None = None,
    xx: np.ndarray | None = None,
) -> np.ndarray:
    """Ring centres GalMod will be given for these parameters / options.

    Outer radius, in order of precedence:
    1. ``rmax`` (arcsec) if given — used as-is (capped only by the field);
    2. freeform lit-map extent when ``n_scale_lengths`` is ``"auto"``;
    3. ``n_scale_lengths * scale_radius`` (legacy; never shorter than the
       lit map when freeform);
    4. most of the image field when analytic + ``"auto"``.
    """
    ps = float(pixel_scale)
    ny, nx = (int(shape[0]), int(shape[1]))
    field_cap = 0.45 * nx * ps
    hard_cap = 0.5 * np.hypot(ny, nx) * ps
    data_r = None
    if sb_map is not None and yy is not None and xx is not None:
        y0, x0 = conventions.sky_to_grid(p.centre_ra, p.centre_dec)
        lit = sb_map > 0
        if lit.any():
            data_r = float(np.hypot(yy[lit] - y0, xx[lit] - x0).max()) + ps
        else:
            data_r = field_cap
    r_max = _resolve_r_max(
        n_scale_lengths, rmax=rmax,
        scale_radius=float(p.scale_radius), pixel_scale=ps,
        field_cap=field_cap, hard_cap=hard_cap, data_r_max=data_r,
    )
    if n_rings:
        return np.linspace(0.5 * ps, max(r_max, ps), int(n_rings))
    return np.arange(0.5 * ps, r_max + 0.5 * ps, ps)


def _resolve_r_max(
    n_scale_lengths,
    *,
    scale_radius: float,
    pixel_scale: float,
    field_cap: float,
    hard_cap: float,
    data_r_max: float | None,
    rmax=None,
) -> float:
    """Outer ring radius from ``rmax``, ``n_scale_lengths``, and/or the data.

    An explicit ``rmax`` is the outer radius the user asked for (capped by
    the field so GalMod stays on-grid). ``"auto"`` / unset follows the
    freeform lit map or the field edge.
    """
    if rmax is not None and not (
        isinstance(rmax, str) and str(rmax).strip().lower() == "auto"
    ):
        # honour the requested outer radius; do not inflate to the lit map
        return float(min(float(rmax), hard_cap if data_r_max is not None else field_cap))

    auto = n_scale_lengths is None or (
        isinstance(n_scale_lengths, str) and n_scale_lengths.strip().lower() == "auto"
    )
    if auto:
        if data_r_max is not None:
            return float(min(data_r_max, hard_cap))
        return float(field_cap)
    r = float(n_scale_lengths) * max(float(scale_radius), float(pixel_scale))
    if data_r_max is not None:
        # freeform + legacy n_scale_lengths: never truncate inside the lit map
        r = max(r, float(data_r_max))
    return float(min(r, field_cap if data_r_max is None else hard_cap))


class BBaroloRenderer(Renderer):
    name = "bbarolo"
    parameter_names = (
        "centre_ra", "centre_dec", "v_sys", "intensity", "scale_radius",
        "inclination", "phi", "turnover_radius", "maximum_velocity",
        "velocity_dispersion",
    )
    supports_freeform = True

    def __init__(self, geometry, spectral, sb, options=None):
        if GalMod is None:
            raise ImportError(
                "the 'bbarolo' backend needs pyBBarolo with a built libBBarolo: "
                "see docs/install-bbarolo.md "
                "(https://bbarolo.readthedocs.io/en/latest/pybb_install.html)"
            )
        super().__init__(geometry, spectral, sb, options)
        self.rotation_curve = self.options.get("rotation_curve", "arctan")
        self.free_rings = self.rotation_curve == "rings"
        raw_nsl = self.options.get("n_scale_lengths", "auto")
        if raw_nsl is None or (
            isinstance(raw_nsl, str) and raw_nsl.strip().lower() == "auto"
        ):
            self.n_scale_lengths = "auto"
        else:
            self.n_scale_lengths = float(raw_nsl)
        # direct outer radius in arcsec (preferred over n_scale_lengths * scale_radius)
        raw_rmax = self.options.get("rmax", self.options.get("rmax_arcsec"))
        self.rmax = None if raw_rmax is None else (
            "auto" if isinstance(raw_rmax, str) and raw_rmax.strip().lower() == "auto"
            else float(raw_rmax)
        )
        n_rings = self.options.get("n_rings")
        free_opt = self.options.get("free", self.options.get("free_ring_parameters"))
        if self.free_rings:
            if n_rings is None or int(n_rings) < 1:
                raise ValueError(
                    "rotation_curve 'rings' requires model.options.n_rings >= 1"
                )
            self.n_rings = int(n_rings)
            self.free_ring_params = parse_free_ring_params(
                free_opt if free_opt is not None else DEFAULT_FREE_RING_PARAMS
            )
            self.parameter_names = ring_parameter_names(self.n_rings, self.free_ring_params)
        else:
            self.n_rings = int(n_rings) if n_rings is not None else None
            self.free_ring_params = ()
        self.scale_height = float(self.options.get("scale_height_arcsec", 0.0))
        # BBarolo LINEAR: instrumental spectral broadening in channels (σ).
        # Off by default — only needed for native-resolution (non-averaged) data.
        raw_lin = self.options.get(
            "linear", self.options.get("hanning", self.options.get("instrumental_broadening")),
        )
        self.linear_channels = resolve_linear_channels(raw_lin)
        ny, nx = self.render_shape
        self.template = blank_cube_fits_path(
            self.n_chan, ny, nx, self.render_pixel_scale, self.dv, self.spectral.v_min,
        )
        self._sb_map = None
        self._yy = self._xx = None
        if isinstance(sb, FreeformSB):
            self._yy, self._xx = self.render_coordinates()
            self._sb_map = sb.on_grid(self._yy, self._xx, self.render_pixel_scale)

    def _radii(self, p: DiscParameters) -> np.ndarray:
        return galmod_ring_radii(
            p, pixel_scale=self.render_pixel_scale, shape=self.render_shape,
            n_rings=self.n_rings, n_scale_lengths=self.n_scale_lengths,
            rmax=self.rmax, sb_map=self._sb_map, yy=self._yy, xx=self._xx,
        )

    def as_dict(self) -> dict:
        d = super().as_dict()
        d["n_scale_lengths"] = self.n_scale_lengths
        d["rmax_arcsec"] = self.rmax
        d["linear_channels"] = self.linear_channels
        return d

    def _target_map(self, p: DiscParameters) -> np.ndarray:
        """Freeform map scaled so its total equals ``intensity``."""
        total = float(self._sb_map.sum())
        if total <= 0:
            return np.zeros_like(self._sb_map)
        return self._sb_map * (float(p.intensity) / total)

    def render(self, p: DiscParameters) -> np.ndarray:
        ps = self.render_pixel_scale
        ny, nx = self.render_shape
        radii = self._radii(p)
        if radii.size == 0:
            raise RuntimeError("GALMOD ring list is empty")
        if self._sb_map is not None:
            # flat rings: morphology comes from the freeform rescale below
            dens = np.ones_like(radii)
        else:
            dens = np.exp(-radii / max(float(p.scale_radius), 1e-4))
        if self.free_rings:
            vrot = ring_vrot_kms(p, self.n_rings)
            if vrot.size != radii.size:
                raise RuntimeError(
                    f"free-ring VROT length {vrot.size} != GalMod radii {radii.size}"
                )
            inc = ring_values(p, "inclination", self.n_rings)
            phi = np.asarray(
                [bbarolo_phi(a) for a in ring_values(p, "phi", self.n_rings)], dtype=float,
            )
            vdisp = ring_values(p, "velocity_dispersion", self.n_rings)
            vrad = ring_values(p, "vrad", self.n_rings)
            cra = ring_values(p, "centre_ra", self.n_rings)
            cdec = ring_values(p, "centre_dec", self.n_rings)
            vsys = ring_values(p, "v_sys", self.n_rings)
            ypos = np.empty(self.n_rings)
            xpos = np.empty(self.n_rings)
            for i in range(self.n_rings):
                y_g, x_g = conventions.sky_to_grid(cra[i], cdec[i])
                xpos[i] = (nx - 1) / 2.0 + x_g / ps
                ypos[i] = (ny - 1) / 2.0 + y_g / ps
        else:
            vrot = rotation_curve_kms(radii, p, self.rotation_curve)
            y_g, x_g = conventions.sky_to_grid(p.centre_ra, p.centre_dec)
            # FITS pixel frame (0-based): column increases with -RA (west) = +x,
            # row increases with +Dec = +y
            xpos = (nx - 1) / 2.0 + x_g / ps
            ypos = (ny - 1) / 2.0 + y_g / ps
            inc = float(p.inclination)
            phi = bbarolo_phi(p.phi)
            vdisp = float(p.velocity_dispersion)
            vrad = float(getattr(p, "vrad", 0.0) or 0.0)
            vsys = float(p.v_sys)
        # GalMod prints HEADER WARNINGs for missing BMAJ/BMIN/BPA on our blank
        # template; beam is unused (we never smooth). Silence C++ chatter.
        with _silence_stdio():
            gm = GalMod(self.template)
            gm.init(
                radii=radii, xpos=xpos, ypos=ypos, vsys=vsys,
                vrot=vrot, vdisp=vdisp, vrad=vrad,
                z0=self.scale_height, inc=inc, phi=phi, dens=dens,
            )
            gm.set_options(ltype=1)
            model = gm.compute()
        cube = np.asarray(model.data, dtype=float)
        if cube.shape != (self.n_chan, ny, nx):
            raise RuntimeError(f"GALMOD cube shape {cube.shape} != {(self.n_chan, ny, nx)}")
        # FITS order: row 0 is south. Native wants row 0 = north.
        cube = cube[:, ::-1, :]
        if self.linear_channels is not None:
            cube = apply_instrumental_broadening(cube, self.linear_channels)
        if self._sb_map is not None:
            return apply_local_surface_brightness(cube, self._target_map(p), self.dv)
        total = cube.sum() * self.dv
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError("GALMOD returned a cube with no flux")
        return cube * (float(p.intensity) / total)
