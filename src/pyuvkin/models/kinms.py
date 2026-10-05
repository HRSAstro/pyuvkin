"""KinMS backend (Davis et al. 2013): Monte Carlo cloudlets.

Analytic surface brightness: an exponential ``sbProf`` on a log-spaced radial
grid with the shared arctan rotation curve. Freeform: the SB map is sampled as
``inClouds`` at their *sky-plane* positions and given line-of-sight velocities
from the analytic velocity field (`conventions.disc_coordinates`), so KinMS
never re-projects the already-projected morphology -- LensKin's
``KinMSPixelized`` construction.

KinMS's own conventions (cube axes ``(x, y, v)``, ``posAng`` counter-clockwise
from +y with the *receding* side along +y at zero, the sense of its y axis)
are translated here and pinned against the `thindisk` reference by
``tests/test_conventions.py``.
"""

from __future__ import annotations

import numpy as np

from .. import conventions
from .base import Renderer
from .parameters import (
    DEFAULT_DISPERSION_CURVE,
    DiscParameters,
    dispersion_curve_kms,
    normalize_dispersion_curve,
    normalize_rotation_curve,
    rotation_curve_kms,
)
from .sb import FreeformSB

#: first KinMS release that runs unpatched on NumPy 2 / SciPy >= 1.14
#: (``cumulative_trapezoid``, ``np.prod``, and ``np.any(flux_clouds != None)``
#: instead of ``np.any(flux_clouds) != None``, which sent every parametric
#: model down the flux_clouds branch -- the bug LensKin's `kinms_utils` patches)
KINMS_MIN_VERSION = (3, 0, 14)


def _ensure_kinms_compat() -> None:
    """The two removed names older KinMS releases still import. Harmless when
    present; LensKin carries the same shims."""
    import scipy.integrate

    if not hasattr(scipy.integrate, "cumtrapz"):
        from scipy.integrate import cumulative_trapezoid

        def cumtrapz(y, x=None, dx=1.0, axis=-1, initial=None):
            return cumulative_trapezoid(y, x=x, dx=dx, axis=axis, initial=initial)

        scipy.integrate.cumtrapz = cumtrapz
    if not hasattr(np, "product"):
        np.product = np.prod


def _kinms_version() -> tuple[int, ...] | None:
    try:
        from importlib.metadata import version

        return tuple(int(p) for p in version("kinms").split(".")[:3])
    except Exception:  # pragma: no cover
        return None


_ensure_kinms_compat()
try:
    from kinms import KinMS
except ImportError:  # pragma: no cover - optional dependency
    KinMS = None
else:
    _v = _kinms_version()
    if _v is not None and _v < KINMS_MIN_VERSION:
        import logging

        logging.getLogger("pyuvkin").warning(
            "kinms %s is older than %s: its parametric mode mis-normalises the flux "
            "(np.any(flux_clouds) != None); upgrade with `pip install -U kinms`",
            ".".join(map(str, _v)), ".".join(map(str, KINMS_MIN_VERSION)),
        )

DEFAULT_N_SAMPLES = 1_000_000
DEFAULT_CLOUDS_PER_PIXEL = 64


def kinms_pos_ang(phi_deg: float) -> float:
    """pyuvkin ``phi`` (receding side, east of north) -> KinMS ``posAng``.

    KinMS's ``posAng`` runs the opposite way round from east of north in the
    frame `KinMSRenderer._to_native` maps its cube into (measured: a cube
    rendered at ``posAng = phi`` has its receding side at ``-phi``), so the
    angle is negated -- the same conversion LensKin arrived at.
    """
    return float(conventions.wrap_deg(-float(phi_deg)))


class KinMSRenderer(Renderer):
    name = "kinms"
    parameter_names = (
        "centre_ra", "centre_dec", "v_sys", "intensity", "scale_radius",
        "inclination", "phi", "turnover_radius", "maximum_velocity",
        "rotation_beta", "rotation_xi",
        "velocity_dispersion", "dispersion_scale_radius", "vmax_black_hole",
    )
    supports_freeform = True

    def __init__(self, geometry, spectral, sb, options=None):
        if KinMS is None:
            raise ImportError("the 'kinms' backend needs the kinms package: pip install kinms")
        super().__init__(geometry, spectral, sb, options)
        ny, nx = self.render_shape
        if ny != nx:
            raise ValueError("KinMS needs a square grid")
        ps = self.render_pixel_scale
        self.n_samples = int(self.options.get("n_samples", DEFAULT_N_SAMPLES))
        self.disk_thick = float(self.options.get("scale_height_arcsec", 0.0))
        self.rotation_curve = normalize_rotation_curve(
            self.options.get("rotation_curve", "arctan"),
        )
        self.dispersion_curve = normalize_dispersion_curve(
            self.options.get("dispersion_curve", DEFAULT_DISPERSION_CURVE),
        )
        self.kinms = KinMS(
            xs=nx * ps, ys=ny * ps, vs=self.n_chan * self.dv,
            cellSize=ps, dv=self.dv, beamSize=None,
            nSamps=self.n_samples, cleanOut=True, seed=int(self.options.get("seed", 1)),
        )
        # log-spaced radial grid for sbProf / velProf, as LensKin
        self.radii = np.logspace(np.log10(ps / 5.0), np.log10(nx * ps), 2000)
        # velocity of KinMS channel j relative to vOffset: centred axis
        v = self.spectral.model_velocities_kms
        self.v_centre = 0.5 * (v[0] + v[-1])
        self._clouds = None
        if isinstance(sb, FreeformSB):
            xs, ys, zs, w = sb.clouds(
                clouds_per_pixel=int(self.options.get("clouds_per_pixel", DEFAULT_CLOUDS_PER_PIXEL)),
                scale_height_arcsec=self.disk_thick,
                seed=int(self.options.get("cloud_seed", 100)),
            )
            self._clouds = (xs, ys, zs, w)

    # ------------------------------------------------------------ frame maps
    @staticmethod
    def _to_native(cube_xyv: np.ndarray) -> np.ndarray:
        """KinMS ``(x, y, v)`` -> native ``(v, y, x)`` with row 0 = +y.

        KinMS's cube index increases with its y coordinate; native arrays put
        +y in row 0, so y is reversed. Its x index increases with x_KinMS,
        which maps to *east*; native +x is west, so x is reversed too.
        """
        c = np.transpose(cube_xyv, (2, 1, 0))
        return c[:, ::-1, ::-1]

    def _phase_cent(self, p: DiscParameters) -> list[float]:
        """Disc centre in KinMS's frame: x_KinMS = +east = dRA, y = dDec.

        KinMS places ``phaseCent = [0, 0]`` half a pixel from the grid centre
        (measured: a face-on disc's centroid lands at native ``(+ps/2, -ps/2)``
        in ``(y, x)`` for any centre), so the half pixel is taken off here.
        """
        half = 0.5 * self.render_pixel_scale
        return [float(p.centre_ra) - half, float(p.centre_dec) - half]

    def _v_offset(self, p: DiscParameters) -> float:
        """KinMS's velocity axis is centred half a channel from where a
        symmetric ``linspace`` puts it (measured: a zero-dispersion line at a
        channel centre lands half in each of two channels without this)."""
        return float(p.v_sys) - self.v_centre - 0.5 * self.dv

    # -------------------------------------------------------------- render
    def render(self, p: DiscParameters) -> np.ndarray:
        if self._clouds is None:
            cube = self._render_analytic(p)
        else:
            cube = self._render_clouds(p)
        return self._to_native(cube)

    def _render_analytic(self, p: DiscParameters) -> np.ndarray:
        h = max(float(p.scale_radius), 1e-4)
        sbprof = np.exp(-self.radii / h)
        velprof = rotation_curve_kms(self.radii, p, self.rotation_curve)
        sigprof = dispersion_curve_kms(self.radii, p, self.dispersion_curve)
        return self.kinms.model_cube(
            inc=float(p.inclination),
            posAng=kinms_pos_ang(p.phi),
            intFlux=float(p.intensity),
            gasSigma=sigprof,
            diskThick=self.disk_thick,
            sbProf=sbprof, sbRad=self.radii, velProf=velprof, velRad=self.radii,
            inClouds=np.zeros((0, 3)), flux_clouds=None,
            phaseCent=self._phase_cent(p), vOffset=self._v_offset(p),
        )

    def _render_clouds(self, p: DiscParameters) -> np.ndarray:
        xs, ys, zs, w = self._clouds          # native grid x (+west), y (+north)
        R, cos_t, _ = conventions.disc_coordinates(
            ys, xs, p.centre_ra, p.centre_dec, p.phi, p.inclination,
        )
        v_c = rotation_curve_kms(R, p, self.rotation_curve)
        v_los = v_c * np.sin(np.radians(float(p.inclination))) * cos_t
        sigma = dispersion_curve_kms(R, p, self.dispersion_curve)
        if np.any(sigma > 0):
            rng = np.random.RandomState(int(self.options.get("vlos_seed", 7)))
            v_los = v_los + rng.normal(0.0, sigma, size=v_los.shape)
        # KinMS x is +east = -grid x; clouds relative to the disc centre,
        # phaseCent puts the centre back
        in_clouds = np.column_stack([-xs - p.centre_ra, ys - p.centre_dec, zs])
        # `intensity` is fixed to the map's total flux by the configuration
        # unless the user gives it a prior, in which case it rescales the map
        return self.kinms.model_cube(
            inc=0.0, posAng=0.0, gasSigma=0.0,
            intFlux=float(p.intensity),
            inClouds=in_clouds, flux_clouds=np.asarray(w, float), vLOS_clouds=v_los,
            phaseCent=self._phase_cent(p), vOffset=self._v_offset(p),
        )

    def as_dict(self) -> dict:
        d = super().as_dict()
        d["options"] = {
            **d["options"], "n_samples": self.n_samples,
            "scale_height_arcsec": self.disk_thick,
            "rotation_curve": self.rotation_curve,
            "dispersion_curve": self.dispersion_curve,
        }
        return d
