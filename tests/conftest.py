import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore", category=UserWarning)

from pyuvkin.grids import CubeGeometry  # noqa: E402
from pyuvkin.spectral import spectral_axis  # noqa: E402
from pyuvkin.models import DiscParameters, backend_available  # noqa: E402


@pytest.fixture
def geometry():
    n, ps = 64, 0.05
    return CubeGeometry(fov_arcsec=n * ps, pixel_scale=ps, shape=(n, n),
                        nyquist_pixel_scale=ps, render_oversample=1)


@pytest.fixture
def spectral():
    f0, nch, dv = 300e9, 24, 20.0
    freqs = f0 * (1 - (np.arange(nch) - nch / 2 + 0.5) * dv / 299792.458)[::-1]
    return spectral_axis(freqs)


@pytest.fixture
def disc():
    return DiscParameters(
        centre_ra=0.2, centre_dec=-0.3, v_sys=35.0, intensity=2.0, scale_radius=0.25,
        inclination=60.0, phi=60.0, turnover_radius=0.1, maximum_velocity=200.0,
        velocity_dispersion=20.0,
    )


def needs(backend):
    return pytest.mark.skipif(not backend_available(backend), reason=f"{backend} not installed")


def cube_moments(cube, spectral, geometry):
    """(flux Jy km/s, centre y, centre x, receding-axis PA, mean velocity)
    measured from a cube; the yardstick every backend is held to."""
    v = spectral.velocities_kms
    yy, xx = geometry.coordinates()
    m0 = cube.sum(0)
    tot = m0.sum()
    m1 = (cube * v[:, None, None]).sum(0) / np.where(m0 > 0, m0, 1)
    cy = (m0 * yy).sum() / tot
    cx = (m0 * xx).sum() / tot
    vmean = (m0 * m1).sum() / tot
    w = m0 * np.clip(m1 - vmean, 0, None)
    dy = (w * (yy - cy)).sum() / w.sum()
    dx = (w * (xx - cx)).sum() / w.sum()
    phi = np.degrees(np.arctan2(-dx, dy))
    return tot * spectral.dv_kms, cy, cx, phi, vmean


def measure_kinematics(cube, spectral, geometry, p):
    """Inclination from the moment-0 axis ratio, v_max sin i and r_t from an
    arctan fit to the major-axis velocity profile, dispersion from the line
    width along the minor axis (away from the centre), all from the cube."""
    from scipy.optimize import curve_fit

    from pyuvkin.results import pv_slice

    v = spectral.velocities_kms
    dv = spectral.dv_kms
    ps = geometry.pixel_scale
    yy, xx = geometry.coordinates()
    m0 = cube.sum(0)
    tot = m0.sum()
    cy = (m0 * yy).sum() / tot
    cx = (m0 * xx).sum() / tot
    iyy = (m0 * (yy - cy) ** 2).sum() / tot
    ixx = (m0 * (xx - cx) ** 2).sum() / tot
    ixy = (m0 * (yy - cy) * (xx - cx)).sum() / tot
    ev = np.linalg.eigvalsh([[ixx, ixy], [ixy, iyy]])
    inclination = np.degrees(np.arccos(np.sqrt(ev[0] / ev[1])))
    s, pv = pv_slice(cube, geometry, (cy, cx), p.phi, slit_pixels=1)
    w = pv.sum(0)
    vbar = (pv * v[:, None]).sum(0) / np.where(w > 1e-3 * w.max(), w, np.nan)
    ok = np.isfinite(vbar) & (np.abs(s) < 3 * p.scale_radius)

    def f(s, vs, vm, rt):
        return vs + np.sign(s) * (2 * vm / np.pi) * np.arctan(np.abs(s) / rt)

    (v_sys, vmax_sini, r_t), _ = curve_fit(f, s[ok], vbar[ok], p0=[0.0, 150.0, 0.1])
    s2, pv2 = pv_slice(cube, geometry, (cy, cx), p.phi + 90.0, slit_pixels=1)
    strip = pv2[:, np.abs(s2) > 2 * ps].sum(1)
    mu = (strip * v).sum() / strip.sum()
    sig = np.sqrt((strip * (v - mu) ** 2).sum() / strip.sum())
    dispersion = np.sqrt(max(sig ** 2 - dv ** 2 / 12.0, 0.0))
    return dict(inclination=inclination, vmax_sini=vmax_sini, turnover_radius=r_t,
                velocity_dispersion=dispersion, v_sys=v_sys)
