"""Parametric rotation and dispersion curves (Rizzo+2021 §3.2 forms)."""

import numpy as np
import pytest

from pyuvkin.models import AnalyticSB, DiscParameters, make_renderer
from pyuvkin.models.parameters import (
    dispersion_curve_kms,
    normalize_dispersion_curve,
    normalize_rotation_curve,
    rotation_curve_kms,
)


def test_rotation_curve_aliases():
    assert normalize_rotation_curve("multi") == "rix"
    assert normalize_rotation_curve("RIX") == "rix"
    assert normalize_dispersion_curve(None) == "constant"


def test_rix_matches_paper_form_at_sample_radii():
    """V(R) = V_t (1+R_t/R)^β / [1+(R_t/R)^ξ]^(1/ξ)."""
    p = DiscParameters(
        turnover_radius=0.5, maximum_velocity=200.0,
        rotation_beta=0.8, rotation_xi=2.5,
    )
    r = np.array([0.1, 0.5, 1.0, 2.0])
    rt, vt, beta, xi = 0.5, 200.0, 0.8, 2.5
    inv = rt / r
    expect = vt * (1.0 + inv) ** beta / (1.0 + inv ** xi) ** (1.0 / xi)
    got = rotation_curve_kms(r, p, "rix")
    assert np.allclose(got, expect, rtol=1e-6)
    assert rotation_curve_kms(0.0, p, "rix") == 0.0


def test_arctan_unchanged():
    p = DiscParameters(turnover_radius=0.2, maximum_velocity=180.0)
    r = np.linspace(0, 2, 50)
    got = rotation_curve_kms(r, p, "arctan")
    expect = (2.0 * 180.0 / np.pi) * np.arctan(r / 0.2)
    assert np.allclose(got, expect)


def test_exponential_dispersion_profile():
    p = DiscParameters(velocity_dispersion=100.0, dispersion_scale_radius=1.0)
    r = np.array([0.0, 1.0, 2.0])
    got = dispersion_curve_kms(r, p, "exponential")
    expect = 100.0 * np.exp(-r / 1.0)
    assert np.allclose(got, expect)
    assert np.allclose(dispersion_curve_kms(r, p, "constant"), 100.0)


def test_linear_dispersion_profile():
    p = DiscParameters(velocity_dispersion=80.0, dispersion_scale_radius=2.0)
    r = np.array([0.0, 1.0, 2.0, 3.0])
    got = dispersion_curve_kms(r, p, "linear")
    assert got[0] == pytest.approx(80.0)
    assert got[1] == pytest.approx(40.0)
    assert got[2] == pytest.approx(1e-3)  # floored
    assert got[3] == pytest.approx(1e-3)


def test_thindisk_renders_rix_exponential(geometry, spectral, disc):
    p = DiscParameters(**{
        **disc.as_dict(),
        "rotation_beta": 0.7, "rotation_xi": 2.5,
        "dispersion_scale_radius": 0.8,
        "maximum_velocity": 120.0,  # keep |v_los| inside the ±230 km/s window
    })
    const = make_renderer(
        "thindisk", geometry, spectral, AnalyticSB(),
        {"rotation_curve": "arctan", "dispersion_curve": "constant"},
    ).cube(p)
    rix = make_renderer(
        "thindisk", geometry, spectral, AnalyticSB(),
        {"rotation_curve": "rix", "dispersion_curve": "exponential"},
    ).cube(p)
    assert const.shape == rix.shape
    assert not np.allclose(const, rix)
    # flux conserved when the line stays inside the spectral window
    assert abs(rix.sum() * spectral.dv_kms - p.intensity) < 1e-4 * p.intensity
    assert abs(const.sum() * spectral.dv_kms - p.intensity) < 1e-4 * p.intensity


def test_config_accepts_rix_and_exponential_dispersion():
    from pyuvkin.config import DEFAULTS, validate_settings
    import copy

    s = copy.deepcopy(DEFAULTS)
    s["dataset"] = "/tmp/x"
    s["fov"] = 5.0
    s["model"]["rotation_curve"] = "multi"
    s["model"]["dispersion_curve"] = "exponential"
    validate_settings(s)
    assert s["model"]["rotation_curve"] == "rix"
    assert s["model"]["dispersion_curve"] == "exponential"
