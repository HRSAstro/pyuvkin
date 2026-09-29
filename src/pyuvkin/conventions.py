"""Sky <-> grid conventions, in one place.

**Positions.** User-facing positions are sky offsets from the phase centre (or
from the recentred image centre): ``centre_ra`` is dRA·cos(dec) in arcsec,
positive **east**; ``centre_dec`` is dDec, positive north. On the grid
(`grids`), ``y = dDec`` and ``x = -dRA`` -- +x is west, column index
increasing, exactly as pyuvimage's `pointsource.sky_to_grid`.

**Position angle.** ``phi`` is the position angle of the **receding** side
of the kinematic major axis, in degrees east of north on the sky: 0 = the
receding side points north, 90 = east, -90/270 = west. On the grid the unit
vector towards the receding side is therefore ``(dx, dy) = (-sin phi, cos phi)``.

Each backend has its own native angle and centre convention; the conversions
live with the backend and are pinned by tests against the analytic
`thindisk` model, which implements the definitions above directly.
"""

from __future__ import annotations

import numpy as np


def sky_to_grid(d_ra: float, d_dec: float) -> tuple[float, float]:
    """(dRA, dDec) arcsec -> grid (y, x) arcsec."""
    return float(d_dec), float(-d_ra)


def grid_to_sky(y: float, x: float) -> tuple[float, float]:
    return float(-x), float(y)


def major_axis_grid_vector(phi_deg: float) -> tuple[float, float]:
    """Grid ``(dx, dy)`` unit vector towards the receding major-axis side."""
    p = np.radians(float(phi_deg))
    return float(-np.sin(p)), float(np.cos(p))


def wrap_deg(angle):
    """Wrap to ``(-180, 180]``."""
    a = np.asarray(angle, dtype=float)
    out = (a + 180.0) % 360.0 - 180.0
    out = np.where(out == -180.0, 180.0, out)
    return float(out) if out.shape == () else out


def disc_coordinates(
    yy: np.ndarray, xx: np.ndarray, centre_ra: float, centre_dec: float,
    phi_deg: float, inclination_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deprojected disc-plane radius and azimuth of every grid point.

    Returns ``(R, cos_theta, sin_theta)`` where ``theta`` is the azimuth in the
    disc plane measured from the receding major axis, so that the
    line-of-sight component of circular rotation is ``v_c(R) sin i cos theta``.
    """
    y0, x0 = sky_to_grid(centre_ra, centre_dec)
    dx = xx - x0
    dy = yy - y0
    mx, my = major_axis_grid_vector(phi_deg)
    # along the major axis (towards the receding side) and along the minor axis
    u = dx * mx + dy * my
    w = -dx * my + dy * mx
    cos_i = np.cos(np.radians(float(inclination_deg)))
    cos_i = max(abs(cos_i), 1e-6) * (1.0 if cos_i >= 0 else -1.0)
    w_d = w / cos_i
    R = np.hypot(u, w_d)
    safe = np.where(R > 0, R, 1.0)
    return R, np.where(R > 0, u / safe, 0.0), np.where(R > 0, w_d / safe, 0.0)
