"""Axisymmetric radial flow in thindisk."""

import numpy as np

from pyuvkin.models import AnalyticSB, DiscParameters, make_renderer


def test_vrad_zero_matches_circular_los(geometry, spectral, disc):
    r = make_renderer("thindisk", geometry, spectral, AnalyticSB(), {})
    circular = r.cube(disc)
    with_zero = r.cube(DiscParameters(**{**disc.as_dict(), "vrad": 0.0}))
    assert np.allclose(circular, with_zero)


def test_vrad_changes_the_cube_and_near_side_sign(geometry, spectral, disc):
    """Inflow (vrad < 0) redshifts the near side (theta = -90 deg)."""
    from pyuvkin import conventions

    r = make_renderer("thindisk", geometry, spectral, AnalyticSB(), {})
    base = r.cube(disc)
    inflow = r.cube(DiscParameters(**{**disc.as_dict(), "vrad": -80.0}))
    assert not np.allclose(base, inflow)

    # first moment shift on the near side should be positive (redshifted)
    v = spectral.velocities_kms
    yy, xx = geometry.coordinates()
    _, _, sin_t = conventions.disc_coordinates(
        yy, xx, disc.centre_ra, disc.centre_dec, disc.phi, disc.inclination,
    )
    near = sin_t < -0.7
    def mom1(cube):
        w = cube.sum(0)
        return (cube * v[:, None, None]).sum(0) / np.where(w > 0, w, 1.0)

    assert float(np.nanmean(mom1(inflow)[near] - mom1(base)[near])) > 0.0
