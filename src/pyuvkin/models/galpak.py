"""GalPaK3D backend (Bouché et al. 2015): ``DiskModel._create_cube``.

An exponential (Sérsic n=1) disc with Gaussian thickness and the chosen
rotation curve (``arctan`` by default, to match the shared parameterisation;
GalPaK's ``tanh``, ``exponential``, ``isothermal``, ``NFW`` ... are accepted
through ``options.rotation_curve``). Analytic surface brightness only.

Unit translation: GalPaK works in pixels and channels, its ``radius`` is the
half-light radius (``1.678`` scale lengths) and its ``flux`` is ``cube.sum()``,
so ``flux = intensity / dv``. Its velocity axis has channel ``k`` at
``(k - zo) dv`` relative to systemic, so ``zo = (v_sys - v_min) / dv``.
"""

from __future__ import annotations

import logging
import sys
import types

import numpy as np

from .. import conventions
from .base import Renderer
from .parameters import DiscParameters


def _import_galpak():
    """galpak 1.34 imports ``pkg_resources`` at import time, which recent
    setuptools no longer provides; shim it (as LensKin does)."""
    if "pkg_resources" not in sys.modules:
        try:
            import pkg_resources  # noqa: F401
        except ImportError:
            shim = types.ModuleType("pkg_resources")

            class _Dist:
                version = "1.34.0"

            shim.get_distribution = lambda name: _Dist()
            shim.DistributionNotFound = Exception
            sys.modules["pkg_resources"] = shim
    import logging

    for name in ("GalPaK", "GalPaK: PSF", "GalPaK: Plots"):
        logging.getLogger(name).setLevel(logging.ERROR)
    import galpak

    return galpak


try:
    galpak = _import_galpak()
except ImportError:  # pragma: no cover - optional dependency
    galpak = None

logger = logging.getLogger("pyuvkin")

HALF_LIGHT_PER_SCALE_LENGTH = 1.678
#: GalPaK's own default disc aspect ratio (scale height / half-light radius)
DEFAULT_ASPECT = 0.15


def galpak_pa(phi_deg: float) -> float:
    """pyuvkin ``phi`` -> GalPaK ``pa``. GalPaK rotates by ``pa - 90`` about
    the cube's z axis in its (column = x, row = y) pixel frame, which in the
    native (row 0 = north, column = west) frame runs the opposite way from
    east of north (measured: ``pa = phi`` puts the receding side at ``-phi``)."""
    return float(conventions.wrap_deg(-float(phi_deg)))


class GalPaKRenderer(Renderer):
    name = "galpak"
    parameter_names = (
        "centre_ra", "centre_dec", "v_sys", "intensity", "scale_radius",
        "inclination", "phi", "turnover_radius", "maximum_velocity",
        "velocity_dispersion",
    )
    supports_freeform = False

    def __init__(self, geometry, spectral, sb, options=None):
        if galpak is None:
            raise ImportError(
                "the 'galpak' backend needs galpak: pip install 'galpak==1.34.0'"
            )
        super().__init__(geometry, spectral, sb, options)
        self.rotation_curve = self.options.get("rotation_curve", "arctan")
        # GalPaK's disc has a scale height q x (half-light radius); q is its
        # `aspect` (0.15 by default) and also sets its rotation-mixing
        # dispersion term hz v / r. A small aspect (0.01) makes the disc thin
        # and `velocity_dispersion` the whole dispersion, as in the other
        # backends; see tests/test_conventions.py.
        self.aspect = float(self.options.get("aspect", DEFAULT_ASPECT))
        self.model = galpak.DiskModel(
            flux_profile=self.options.get("flux_profile", "exponential"),
            thickness_profile=self.options.get("thickness_profile", "gaussian"),
            rotation_curve=self.rotation_curve,
            dispersion_profile=self.options.get("dispersion_profile", "thick"),
            aspect=self.aspect,
        )
        # record every effective option, so input_parameters.json says what ran
        self.options.update({
            "rotation_curve": self.rotation_curve, "aspect": self.aspect,
            "flux_profile": self.model.flux_profile,
            "thickness_profile": self.model.thickness_profile,
            "dispersion_profile": self.model.dispersion_profile,
        })
        if self.aspect >= 0.1:
            logger.info(
                "galpak: disc aspect %.2f (scale height / half-light radius); the line width "
                "is velocity_dispersion plus GalPaK's rotation-mixing term. options.aspect = "
                "0.05 gives a thin disc comparable with the other backends.", self.aspect,
            )
        ny, nx = self.render_shape
        self.shape_3d = (self.n_chan, ny, nx)
        self._c = ((ny - 1) / 2.0, (nx - 1) / 2.0)

    def _galaxy(self, p: DiscParameters):
        ps = self.render_pixel_scale
        y_g, x_g = conventions.sky_to_grid(p.centre_ra, p.centre_dec)
        cy, cx = self._c
        # native row 0 = +y, so the row index decreases with y; GalPaK's y is
        # the row index and its x the column index
        x_pix = cx + x_g / ps
        y_pix = cy - y_g / ps
        zo = (float(p.v_sys) - self.spectral.v_min) / self.dv
        g = galpak.GalaxyParameters(
            x=x_pix, y=y_pix, z=zo,
            flux=float(p.intensity) / self.dv,
            radius=HALF_LIGHT_PER_SCALE_LENGTH * max(float(p.scale_radius), 1e-4) / ps,
            inclination=float(p.inclination),
            pa=galpak_pa(p.phi),
            turnover_radius=max(float(p.turnover_radius), 1e-6) / ps,
            maximum_velocity=float(p.maximum_velocity),
            velocity_dispersion=max(float(p.velocity_dispersion), 1e-3),
        )
        return g, zo

    def render(self, p: DiscParameters) -> np.ndarray:
        g, zo = self._galaxy(p)
        cube, _, _, _ = self.model._create_cube(
            galaxy=g, shape=self.shape_3d, z_step_kms=self.dv, zo=zo,
        )
        # GalPaK's cube is (v, row, col); it is fed the native row/column of
        # the centre directly, so the array comes back native already.
        return np.asarray(cube.data, dtype=float)
