"""Every backend must put the same disc at the same place on the same sky."""

import numpy as np
import pytest

from pyuvkin import conventions
from pyuvkin.models import AnalyticSB, FreeformSB, DiscParameters, make_renderer

from conftest import cube_moments, measure_kinematics, needs


def test_sky_to_grid_round_trip():
    y, x = conventions.sky_to_grid(0.3, -0.5)
    assert (y, x) == (-0.5, -0.3)          # +east is -x, +north is +y
    assert conventions.grid_to_sky(y, x) == (0.3, -0.5)


def test_major_axis_vector_north_then_east():
    assert np.allclose(conventions.major_axis_grid_vector(0.0), (0.0, 1.0))    # (x, y) -> north
    assert np.allclose(conventions.major_axis_grid_vector(90.0), (-1.0, 0.0))  # east = -x


@pytest.mark.parametrize("phi", [0.0, 60.0, 120.0, -90.0])
@pytest.mark.parametrize("backend", [
    "thindisk",
    pytest.param("kinms", marks=needs("kinms")),
    pytest.param("galpak", marks=needs("galpak")),
])
def test_backends_agree_on_centre_angle_flux_velocity(backend, phi, geometry, spectral):
    p = DiscParameters(centre_ra=0.3, centre_dec=-0.2, v_sys=35.0, intensity=2.0,
                       scale_radius=0.3, inclination=60.0, phi=phi, turnover_radius=0.1,
                       maximum_velocity=200.0, velocity_dispersion=20.0)
    r = make_renderer(backend, geometry, spectral, AnalyticSB(), {"n_samples": 500_000})
    cube = r.cube(p)
    assert cube.shape == (spectral.n_chan,) + geometry.shape
    flux, cy, cx, phi_meas, vmean = cube_moments(cube, spectral, geometry)
    ygrid, xgrid = conventions.sky_to_grid(p.centre_ra, p.centre_dec)
    assert abs(flux - p.intensity) < 0.05 * p.intensity
    assert abs(cy - ygrid) < 0.02 and abs(cx - xgrid) < 0.02
    assert abs(conventions.wrap_deg(phi_meas - phi)) < 3.0
    assert abs(vmean - p.v_sys) < 3.0


@pytest.mark.parametrize("backend", [
    "thindisk",
    pytest.param("kinms", marks=needs("kinms")),
    pytest.param("bbarolo", marks=needs("bbarolo")),
])
def test_freeform_matches_analytic_morphology(backend, geometry, spectral, disc):
    """A freeform map of the analytic disc's own moment-0 must render to the
    same cube as the analytic disc (to the Monte Carlo noise of KinMS /
    GalMod)."""
    ref = make_renderer("thindisk", geometry, spectral, AnalyticSB(), {}).cube(disc)
    mom0 = ref.sum(0) * spectral.dv_kms
    sb = FreeformSB(map_jykms=mom0, pixel_scale=geometry.pixel_scale, source="test")
    r = make_renderer(backend, geometry, spectral, sb, {"clouds_per_pixel": 64})
    cube = r.cube(disc)
    a = cube_moments(cube, spectral, geometry)
    b = cube_moments(ref, spectral, geometry)
    assert abs(a[0] - b[0]) < 0.02 * b[0]
    assert abs(a[1] - b[1]) < 0.02 and abs(a[2] - b[2]) < 0.02
    assert abs(conventions.wrap_deg(a[3] - b[3])) < 3.0
    assert abs(a[4] - b[4]) < 3.0
    # channel-by-channel spectrum, not just moments
    sa, sb_ = cube.sum((1, 2)), ref.sum((1, 2))
    assert np.max(np.abs(sa - sb_)) < 0.08 * sb_.max()


def test_bbarolo_local_surface_brightness_renorm():
    """BBarolo freeform is NORM=LOCAL against the map: spectra keep their
    shape, integrated flux becomes the target."""
    from pyuvkin.models.bbarolo import apply_local_surface_brightness

    rng = np.random.RandomState(0)
    # fake GalMod cube: arbitrary spectra, non-zero on a patch
    cube = np.zeros((5, 8, 8))
    cube[:, 2:6, 2:6] = rng.random_sample((5, 4, 4)) + 0.1
    dv = 10.0
    target = np.zeros((8, 8))
    target[2:6, 2:6] = np.linspace(0.5, 2.0, 16).reshape(4, 4)
    out = apply_local_surface_brightness(cube, target, dv)
    mom0 = out.sum(0) * dv
    assert np.allclose(mom0[2:6, 2:6], target[2:6, 2:6])
    assert np.allclose(mom0[cube.sum(0) == 0], 0.0)
    # spectrum shape preserved where the model had flux
    y, x = 3, 4
    assert np.allclose(out[:, y, x] / out[:, y, x].sum(), cube[:, y, x] / cube[:, y, x].sum())


KINEMATIC_CASES = [
    DiscParameters(centre_ra=0.1, centre_dec=-0.1, v_sys=20.0, intensity=2.0, scale_radius=0.3,
                   inclination=60.0, phi=40.0, turnover_radius=0.1, maximum_velocity=200.0,
                   velocity_dispersion=30.0),
    DiscParameters(centre_ra=0.0, centre_dec=0.0, v_sys=0.0, intensity=1.0, scale_radius=0.4,
                   inclination=35.0, phi=-70.0, turnover_radius=0.2, maximum_velocity=250.0,
                   velocity_dispersion=50.0),
]


@pytest.mark.parametrize("case", range(len(KINEMATIC_CASES)))
@pytest.mark.parametrize("backend, options", [
    pytest.param("kinms", {"n_samples": 3_000_000}, marks=needs("kinms")),
    # a thin GalPaK disc: aspect 0.15 (its default) adds its own thickness
    # and rotation-mixing dispersion, see test_galpak_default_is_thick
    pytest.param("galpak", {"aspect": 0.05}, marks=needs("galpak")),
])
def test_backends_agree_on_inclination_rotation_curve_and_dispersion(backend, options, case, spectral):
    from pyuvkin.grids import CubeGeometry

    # a field that holds the whole disc: a truncated exponential reads a
    # lower inclination from its moments, and the backends truncate differently
    n, ps = 64, 0.08
    geometry = CubeGeometry(fov_arcsec=n * ps, pixel_scale=ps, shape=(n, n), nyquist_pixel_scale=ps)
    p = KINEMATIC_CASES[case]
    ref = measure_kinematics(make_renderer("thindisk", geometry, spectral, AnalyticSB(), {}).cube(p),
                             spectral, geometry, p)
    got = measure_kinematics(make_renderer(backend, geometry, spectral, AnalyticSB(), options).cube(p),
                             spectral, geometry, p)
    assert abs(got["inclination"] - ref["inclination"]) < 1.0
    assert abs(got["vmax_sini"] - ref["vmax_sini"]) < 0.02 * ref["vmax_sini"]
    assert abs(got["turnover_radius"] - ref["turnover_radius"]) < 0.1 * ref["turnover_radius"]
    assert abs(got["velocity_dispersion"] - ref["velocity_dispersion"]) < 0.05 * ref["velocity_dispersion"]
    assert abs(got["v_sys"] - ref["v_sys"]) < 1.0
    # and the thindisk yardstick itself reads the truth back
    assert abs(ref["vmax_sini"] - p.maximum_velocity * np.sin(np.radians(p.inclination))) < 0.02 * ref["vmax_sini"]
    # the strip estimate carries some in-pixel velocity gradient on this grid
    assert abs(ref["velocity_dispersion"] - p.velocity_dispersion) < 0.1 * p.velocity_dispersion


@needs("galpak")
def test_galpak_default_is_thick(spectral):
    """With GalPaK's own aspect ratio (0.15) the disc is thick and its
    `velocity_dispersion` is only the constant part of the line width, so the
    measured dispersion exceeds the parameter -- documented, and pinned here
    so a change in GalPaK's behaviour is noticed."""
    from pyuvkin.grids import CubeGeometry

    n, ps = 64, 0.08
    geometry = CubeGeometry(fov_arcsec=n * ps, pixel_scale=ps, shape=(n, n), nyquist_pixel_scale=ps)
    p = KINEMATIC_CASES[0]
    got = measure_kinematics(make_renderer("galpak", geometry, spectral, AnalyticSB(), {}).cube(p),
                             spectral, geometry, p)
    ref = measure_kinematics(make_renderer("thindisk", geometry, spectral, AnalyticSB(), {}).cube(p),
                             spectral, geometry, p)
    assert abs(got["inclination"] - ref["inclination"]) < 2.0
    assert abs(got["vmax_sini"] - ref["vmax_sini"]) < 0.03 * ref["vmax_sini"]
    assert got["velocity_dispersion"] > 1.3 * p.velocity_dispersion


@pytest.mark.parametrize("n", [48, 49])
@pytest.mark.parametrize("n_chan", [24, 25])
@pytest.mark.parametrize("backend", [
    pytest.param("kinms", marks=needs("kinms")),
    pytest.param("galpak", marks=needs("galpak")),
])
def test_half_pixel_and_half_channel_offsets_for_odd_and_even_sizes(backend, n, n_chan):
    """KinMS's phaseCent and vOffset corrections were measured on even grids;
    they must hold for odd pixel counts and odd channel counts too."""
    from pyuvkin.grids import CubeGeometry
    from pyuvkin.spectral import spectral_axis

    ps, dv = 0.05, 20.0
    geometry = CubeGeometry(fov_arcsec=n * ps, pixel_scale=ps, shape=(n, n), nyquist_pixel_scale=ps)
    freqs = 300e9 * (1 - (np.arange(n_chan) - (n_chan - 1) / 2) * dv / 299792.458)[::-1]
    spectral = spectral_axis(freqs)
    for ra, dec, v_sys in [(0.3, -0.2, 35.0), (0.0, 0.0, 0.0), (-0.13, 0.27, -47.0)]:
        # kept well inside the field: the backends truncate a disc that runs
        # off the edge differently, which is not what this test is about
        p = DiscParameters(centre_ra=ra, centre_dec=dec, v_sys=v_sys, intensity=2.0, scale_radius=0.2,
                           inclination=60.0, phi=60.0, turnover_radius=0.1, maximum_velocity=200.0,
                           velocity_dispersion=20.0)
        ref = cube_moments(make_renderer("thindisk", geometry, spectral, AnalyticSB(), {}).cube(p),
                           spectral, geometry)
        got = cube_moments(make_renderer(backend, geometry, spectral, AnalyticSB(),
                                         {"n_samples": 2_000_000}).cube(p), spectral, geometry)
        assert abs(got[1] - ref[1]) < 0.002 and abs(got[2] - ref[2]) < 0.002   # centre, arcsec
        assert abs(got[4] - ref[4]) < 0.3                                      # mean velocity
