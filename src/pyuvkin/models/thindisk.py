"""An analytic, infinitely thin rotating disc -- deterministic and fast.

Each sky pixel is deprojected onto the disc plane (`conventions.disc_coordinates`),
given a line-of-sight velocity

    v_los = v_sys + (v_c(R) cos theta + vrad sin theta) sin i

(``theta`` from the receding major axis; ``vrad`` > 0 is outwards) and a
Gaussian line of width ``σ(R)`` (see ``dispersion_curve``), and the line is
integrated exactly over each channel. No Monte Carlo, so the likelihood is
smooth in every parameter, which optimisers appreciate; and no external
dependency, so it is the reference against which the other backends' angle and
centre conventions are pinned.

Freeform surface brightness is supported directly: the map is resampled onto
the render grid and used as the per-pixel integrated flux.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

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

_SQRT2 = np.sqrt(2.0)


def channel_fractions(v_edges: np.ndarray, v_los: np.ndarray, sigma) -> np.ndarray:
    """Fraction of a Gaussian line centred on ``v_los`` (per pixel) falling in
    each channel: ``(n_chan, ...)``. Exact integral, so flux is conserved
    within the velocity range whatever the dispersion is.

    ``sigma`` may be a scalar or an array broadcastable to ``v_los`` (radial
    dispersion profiles).
    """
    sig = np.asarray(sigma, dtype=float)
    sig = np.maximum(sig, 1e-3)
    z = (v_edges.reshape((-1,) + (1,) * v_los.ndim) - v_los[None]) / (_SQRT2 * sig)
    cdf = 0.5 * (1.0 + erf(z))
    return np.diff(cdf, axis=0)


class ThinDiskRenderer(Renderer):
    name = "thindisk"
    parameter_names = (
        "centre_ra", "centre_dec", "v_sys", "intensity", "scale_radius",
        "inclination", "phi", "turnover_radius", "maximum_velocity",
        "rotation_beta", "rotation_xi",
        "velocity_dispersion", "dispersion_scale_radius",
        "vrad", "vmax_black_hole",
    )
    supports_freeform = True

    def __init__(self, geometry, spectral, sb, options=None):
        super().__init__(geometry, spectral, sb, options)
        self.rotation_curve = normalize_rotation_curve(
            self.options.get("rotation_curve", "arctan"),
        )
        self.dispersion_curve = normalize_dispersion_curve(
            self.options.get("dispersion_curve", DEFAULT_DISPERSION_CURVE),
        )
        self.yy, self.xx = self.render_coordinates()
        v = self.spectral.model_velocities_kms
        self.v_edges = np.concatenate([[v[0] - 0.5 * self.dv], v + 0.5 * self.dv])
        self._sb_map = None
        if isinstance(sb, FreeformSB):
            self._sb_map = sb.on_grid(self.yy, self.xx, self.render_pixel_scale)

    def integrated_map(self, p: DiscParameters, R: np.ndarray) -> np.ndarray:
        """Jy km/s per render pixel."""
        if self._sb_map is not None:
            # `intensity` is fixed to the map total by default; a prior on it
            # rescales the map
            total = self._sb_map.sum()
            return self._sb_map * (float(p.intensity) / total if total > 0 else 0.0)
        h = max(float(p.scale_radius), 1e-4)
        prof = np.exp(-R / h)
        s = prof.sum()
        return prof * (float(p.intensity) / s if s > 0 else 0.0)

    def render(self, p: DiscParameters) -> np.ndarray:
        R, cos_t, sin_t = conventions.disc_coordinates(
            self.yy, self.xx, p.centre_ra, p.centre_dec, p.phi, p.inclination,
        )
        v_c = rotation_curve_kms(R, p, self.rotation_curve)
        v_r = float(getattr(p, "vrad", 0.0) or 0.0)
        sin_i = np.sin(np.radians(float(p.inclination)))
        v_los = float(p.v_sys) + (v_c * cos_t + v_r * sin_t) * sin_i
        sigma = dispersion_curve_kms(R, p, self.dispersion_curve)
        frac = channel_fractions(self.v_edges, v_los, sigma)
        sb = self.integrated_map(p, R)
        return frac * (sb / self.dv)[None]
