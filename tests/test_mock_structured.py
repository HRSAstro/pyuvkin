"""The structured mocks are only a fair test if they differ from the fitted
model in exactly one respect at a time."""

import numpy as np
import pytest

from pyuvkin import conventions
from pyuvkin.mock_structured import (
    DEFAULT_TRUTH, MOCKS, SpiralStructure, line_of_sight_velocity, render_structured_cube,
    sersic_b, sersic_profile, spiral_arm_profile, structured_surface_brightness,
)
from pyuvkin.models import AnalyticSB, DiscParameters, make_renderer

from conftest import cube_moments


@pytest.fixture
def disc_params():
    return DiscParameters.from_dict(DEFAULT_TRUTH)


@pytest.mark.parametrize("index, expected", [(0.5, 0.69315), (1.0, 1.67835),
                                             (2.0, 3.67206), (4.0, 7.66925)])
def test_sersic_b_is_exact(index, expected):
    assert sersic_b(index) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("r_eff, index", [(0.18, 2.0), (0.90, 1.0), (0.70, 0.5)])
def test_r_eff_encloses_half_the_light(r_eff, index):
    r = np.linspace(0.0, 40.0 * r_eff, 200_001)
    enclosed = np.cumsum(sersic_profile(r, r_eff, index) * 2 * np.pi * r)
    assert np.interp(0.5, enclosed / enclosed[-1], r) == pytest.approx(r_eff, rel=1e-3)


def test_arms_are_m2_trailing_and_radially_confined():
    s = SpiralStructure()
    theta = np.linspace(-np.pi, np.pi, 2001, endpoint=False)

    # two crests per turn, well separated from the interarm floor
    arm = spiral_arm_profile(np.full_like(theta, s.arm_r_ref), theta, s)
    assert 0.0 <= arm.min() and arm.max() <= 1.0
    assert arm.max() > 10.0 * arm.min()
    crests = np.flatnonzero((arm > np.roll(arm, 1)) & (arm >= np.roll(arm, -1)))
    assert len(crests) == s.n_arms

    # the crest winds towards decreasing theta as the radius grows: trailing.
    # An m-armed pattern repeats every 2*pi/m, so unwrap on that period.
    radii = np.linspace(0.45, 1.0, 12)
    crest_theta = np.unwrap(
        [theta[np.argmax(spiral_arm_profile(np.full_like(theta, r), theta, s))] for r in radii],
        period=2 * np.pi / s.n_arms,
    )
    assert np.all(np.diff(crest_theta) < 0.0)

    # and the window switches the arms off well inside and outside
    for r in (0.05, 2.0):
        assert spiral_arm_profile(np.full_like(theta, r), theta, s).max() < 0.02


def test_surface_brightness_conserves_flux_and_the_bulge_fraction(disc_params, geometry):
    yy, xx = geometry.coordinates(1)
    radius, cos_t, sin_t = conventions.disc_coordinates(
        yy, xx, disc_params.centre_ra, disc_params.centre_dec,
        disc_params.phi, disc_params.inclination,
    )
    s = SpiralStructure()
    arm = spiral_arm_profile(radius, np.arctan2(sin_t, cos_t), s)
    sb = structured_surface_brightness(radius, arm, disc_params.intensity, s)
    assert sb.sum() == pytest.approx(disc_params.intensity, rel=1e-6)

    # turning the arms on moves flux around inside the disc component; it does
    # not change the total or the bulge-to-total ratio
    smooth = structured_surface_brightness(radius, np.zeros_like(arm), disc_params.intensity, s)
    assert smooth.sum() == pytest.approx(disc_params.intensity, rel=1e-6)
    assert np.abs(sb - smooth).sum() > 0.2 * disc_params.intensity

    bulge_only = structured_surface_brightness(
        radius, arm, disc_params.intensity,
        SpiralStructure(**{**vars(s), "bulge_flux_fraction": 1.0}),
    )
    assert bulge_only.sum() == pytest.approx(disc_params.intensity, rel=1e-6)


def test_no_arm_flow_reproduces_the_thindisk_velocity_field(disc_params, geometry):
    """Mock 1's kinematics must be exactly the model `thindisk` fits, so any
    bias it produces comes from the surface brightness alone."""
    yy, xx = geometry.coordinates(1)
    radius, cos_t, sin_t = conventions.disc_coordinates(
        yy, xx, disc_params.centre_ra, disc_params.centre_dec,
        disc_params.phi, disc_params.inclination,
    )
    arm = spiral_arm_profile(radius, np.arctan2(sin_t, cos_t), MOCKS["arms_sb"])
    v = line_of_sight_velocity(radius, cos_t, sin_t, arm, disc_params, MOCKS["arms_sb"])

    from pyuvkin.models.parameters import rotation_curve_kms

    expected = disc_params.v_sys + rotation_curve_kms(radius, disc_params, "arctan") * cos_t * (
        np.sin(np.radians(disc_params.inclination))
    )
    assert np.allclose(v, expected)


def test_axisymmetric_inflow_matches_thindisk_vrad_and_redshifts_near_side(
        disc_params, geometry,
):
    """``arms_inflow`` uses constant axisymmetric ``vrad``, same as thindisk."""
    from pyuvkin.mock_structured import INFLOW_VRAD_KMS

    yy, xx = geometry.coordinates(1)
    radius, cos_t, sin_t = conventions.disc_coordinates(
        yy, xx, disc_params.centre_ra, disc_params.centre_dec,
        disc_params.phi, disc_params.inclination,
    )
    s = MOCKS["arms_inflow"]
    arm = spiral_arm_profile(radius, np.arctan2(sin_t, cos_t), s)
    inflow = DiscParameters(**{**disc_params.as_dict(), "vrad": INFLOW_VRAD_KMS})
    residual = (line_of_sight_velocity(radius, cos_t, sin_t, arm, inflow, s)
                - line_of_sight_velocity(radius, cos_t, sin_t, arm, disc_params, s))

    sin_i = np.sin(np.radians(disc_params.inclination))
    expected = INFLOW_VRAD_KMS * sin_t * sin_i
    assert np.allclose(residual, expected)
    assert np.abs(residual).max() > 40.0

    # near side θ = -90: inflow (vrad < 0) is redshifted
    assert residual[sin_t < -0.5].mean() > 0.0
    assert residual[sin_t > 0.5].mean() < 0.0


def test_inflow_mock_truth_and_fit_free_axisymmetric_vrad():
    from pyuvkin.mock_structured import INFLOW_VRAD_KMS, DEFAULT_TRUTH, fit_settings

    truth = {
        **DEFAULT_TRUTH,
        "vrad": INFLOW_VRAD_KMS,
        "reference_frequency_ghz": 230.0,
        "structure": vars(MOCKS["arms_inflow"]),
    }
    settings = fit_settings(truth, "analytic")
    assert settings["truth"]["vrad"] == INFLOW_VRAD_KMS
    assert "vrad" in settings["priors"]
    assert "vrad_arm_structure" not in (settings["model"].get("options") or {})


def test_the_two_mocks_share_a_morphology_but_not_a_velocity_field(disc_params, geometry,
                                                                   spectral):
    from pyuvkin.mock_structured import MOCK_KINEMATICS

    cubes, maps = {}, {}
    for name, s in MOCKS.items():
        p = DiscParameters(**{**disc_params.as_dict(), **MOCK_KINEMATICS[name]})
        cubes[name], maps[name] = render_structured_cube(geometry, spectral, p, s)
    assert np.allclose(maps["arms_sb"], maps["arms_inflow"])
    assert not np.allclose(cubes["arms_sb"], cubes["arms_inflow"])
    for name, cube in cubes.items():
        assert np.all(np.isfinite(cube))
        assert maps[name].sum() == pytest.approx(disc_params.intensity, rel=1e-6)
        # the cube loses the little flux whose velocity falls outside the band
        assert cube.sum() * spectral.dv_kms == pytest.approx(maps[name].sum(), rel=0.02)


def _mom0_geometry(mom0, geometry):
    """Inclination and position angle as the second moments of a map read
    them -- the geometry a morphology-driven fit is pulled towards."""
    yy, xx = geometry.coordinates()
    total = mom0.sum()
    cy, cx = (mom0 * yy).sum() / total, (mom0 * xx).sum() / total
    iyy = (mom0 * (yy - cy) ** 2).sum() / total
    ixx = (mom0 * (xx - cx) ** 2).sum() / total
    ixy = (mom0 * (yy - cy) * (xx - cx)).sum() / total
    values, vectors = np.linalg.eigh([[ixx, ixy], [ixy, iyy]])
    vx, vy = vectors[:, 1]
    return (np.degrees(np.arccos(np.sqrt(values[0] / values[1]))),
            np.degrees(np.arctan2(-vx, vy)) % 180.0)


def test_the_arms_skew_the_apparent_geometry_but_not_the_kinematics(disc_params, geometry,
                                                                    spectral):
    """The point of mock 1. Strip the arms out and the map reads back the true
    inclination and position angle; put them in and a morphology-driven
    estimate is pulled several degrees away -- while the velocity field still
    has the truth in it exactly. An analytic fit has to reconcile the two; a
    freeform fit is handed the map and does not."""
    smooth, _ = render_structured_cube(geometry, spectral, disc_params,
                                       SpiralStructure(arm_contrast=0.0))
    arms, _ = render_structured_cube(geometry, spectral, disc_params, MOCKS["arms_sb"])

    inc, pa = _mom0_geometry(smooth.sum(0), geometry)
    assert inc == pytest.approx(disc_params.inclination, abs=1.0)
    assert pa == pytest.approx(disc_params.phi, abs=1.0)

    inc_arms, pa_arms = _mom0_geometry(arms.sum(0), geometry)
    assert abs(inc_arms - disc_params.inclination) > 3.0
    assert abs(pa_arms - disc_params.phi) > 5.0

    # the centre and the systemic velocity are unmoved, so an analytic fit
    # still starts somewhere sensible
    _, cy, cx, _, vmean = cube_moments(arms, spectral, geometry)
    _, ref_cy, ref_cx, _, ref_vmean = cube_moments(smooth, spectral, geometry)
    assert abs(cy - ref_cy) < 0.02 and abs(cx - ref_cx) < 0.02
    assert abs(vmean - ref_vmean) < 3.0


def test_an_exponential_disc_cannot_reproduce_the_structured_map(disc_params, geometry,
                                                                 spectral):
    cube, _ = render_structured_cube(geometry, spectral, disc_params, MOCKS["arms_sb"])
    analytic = make_renderer("thindisk", geometry, spectral, AnalyticSB(), {}).cube(disc_params)
    assert np.abs(cube.sum(0) - analytic.sum(0)).sum() > 0.1 * analytic.sum()


def test_from_fits_orientation_flips_spatial_y_not_channels():
    """Regression: np.flipud on a cube flips channels and leaves Dec mirrored."""
    from pyuvkin.mock_structured import _from_fits_orientation

    cube = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    out = _from_fits_orientation(cube)
    assert out.shape == cube.shape
    assert np.array_equal(out[0], cube[0, ::-1, :])
    assert np.array_equal(out[:, 0, :], cube[:, -1, :])
    assert np.array_equal(_from_fits_orientation(cube[0]), cube[0, ::-1, :])


def test_write_structured_mocks_round_trips(tmp_path):
    import json

    from pyuvkin.mock_structured import write_structured_mocks
    from pyuvkin.uvdata import UVData

    written = write_structured_mocks(tmp_path, n_vis=200, n_chan=8, fov=2.0)
    assert written.pop("overview").exists()
    assert set(written) == set(MOCKS)
    for name, d in written.items():
        truth = json.loads((d / "truth.json").read_text())
        assert truth["mock"] == name
        assert truth["structure"] == vars(MOCKS[name])
        uvd = UVData.read(d / "dataset")
        assert uvd.data.shape == (8, 200)
        assert np.all(np.isfinite(uvd.data))
        for sb_type in ("analytic", "freeform", "freeform_matern"):
            settings = json.loads((d / f"settings_{sb_type}.json").read_text())
            kind = "freeform" if sb_type.startswith("freeform") else "analytic"
            assert settings["surface_brightness"]["type"] == kind
            if sb_type == "freeform_matern":
                assert settings["surface_brightness"]["pyuvimage"]["reg"] == "matern"
            # freeform takes the morphology and the flux from the map
            has_sb_priors = {"intensity", "scale_radius"} & set(settings["priors"])
            assert bool(has_sb_priors) == (kind == "analytic")
            # only the inflow mock frees axisymmetric vrad
            assert ("vrad" in settings["priors"]) == (name == "arms_inflow")
            if name == "arms_inflow":
                from pyuvkin.mock_structured import INFLOW_VRAD_KMS
                assert truth["vrad"] == INFLOW_VRAD_KMS
                assert settings["truth"]["vrad"] == INFLOW_VRAD_KMS
            else:
                assert truth.get("vrad", 0.0) == 0.0
