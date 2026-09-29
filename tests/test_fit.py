"""Settings, priors and an end-to-end fit to a mock."""

import json

import numpy as np
import pytest

import autofit as af

from pyuvkin import config, mock, priors
from pyuvkin.models import DiscParameters, PARAMETER_NAMES


def test_settings_defaults_merge_and_refuse_unknown_keys(tmp_path):
    s = config.load_settings({"dataset": "d", "fov": 2.0, "model": {"backend": "kinms"}})
    assert s["model"]["backend"] == "kinms" and s["model"]["rotation_curve"] == "arctan"
    assert s["search"]["method"] == "nautilus"
    with pytest.raises(ValueError):
        config.load_settings({"dataset": "d", "fov": 2.0, "fov_arcsec": 2.0})
    with pytest.raises(SystemExit):
        config.load_settings({"fov": 2.0})
    p = tmp_path / "t.json"
    config.write_template(p)
    t = json.loads(p.read_text())
    assert set(t) == set(config.public(config.DEFAULTS))


def test_priors_build_model_fixed_and_free():
    cfg = {
        "phi": {"type": "Uniform", "lower": 0, "upper": 180},
        "inclination": 45.0,
        "maximum_velocity": {"type": "Gaussian", "mean": 200, "sigma": 50, "lower_limit": 0},
        "intensity": {"type": "LogUniform", "lower": 0.1, "upper": 10},
    }
    model = priors.build_model(cfg, PARAMETER_NAMES, fixed={"scale_radius": 0.5})
    assert model.prior_count == 3
    assert isinstance(model.phi, af.UniformPrior)
    assert model.inclination == 45.0 and model.scale_radius == 0.5
    d = priors.describe_model(model)
    assert d["phi"]["type"] == "Uniform" and d["inclination"] == {"type": "fixed", "value": 45.0}
    inst = model.instance_from_unit_vector([0.5] * 3)
    assert isinstance(inst, DiscParameters) and 0 < inst.phi < 180
    with pytest.raises(ValueError):
        priors.build_model({"not_a_parameter": 1.0}, PARAMETER_NAMES)
    with pytest.raises(ValueError):
        priors.build_model({}, PARAMETER_NAMES)


def test_rings_mode_broadcast_priors_and_free_geometry():
    from pyuvkin.models.parameters import (
        make_tilted_ring_class, parse_free_ring_params, ring_parameter_names,
    )
    from pyuvkin.search import tilted_ring_optimiser_defaults

    free = parse_free_ring_params("VROT VDISP INC PA")
    assert free == ("vrot", "velocity_dispersion", "inclination", "phi")
    names = ring_parameter_names(3, free)
    assert "vrot_0" in names and "inclination_2" in names
    assert "inclination" not in names and "centre_ra" in names
    cls = make_tilted_ring_class(3, free)
    cfg = {
        "vrot_i": {"type": "Uniform", "lower": 50, "upper": 400},
        "inclination": {"type": "Uniform", "lower": 20, "upper": 70},
        "phi": {"type": "Uniform", "lower": 0, "upper": 360},
        "velocity_dispersion": {"type": "Uniform", "lower": 5, "upper": 50},
        "centre_ra": 0.0,
        "centre_dec": 0.0,
        "v_sys": 0.0,
        "intensity": 1.0,
        "scale_radius": 0.3,
        "vrot_1": {"type": "Uniform", "lower": 100, "upper": 200},
    }
    model = priors.build_model(cfg, names, parameter_cls=cls)
    assert isinstance(model.vrot_0, af.UniformPrior)
    assert model.vrot_0.lower_limit == 50 and model.vrot_1.lower_limit == 100
    assert isinstance(model.inclination_0, af.UniformPrior)
    assert model.inclination_0.lower_limit == model.inclination_2.lower_limit == 20
    assert model.prior_count == 12  # 3 each of vrot, vdisp, inc, phi
    assert tilted_ring_optimiser_defaults(4) == {"eps": 1.0, "gtol": 1e-2}
    assert abs(tilted_ring_optimiser_defaults(1)["gtol"] - 2.5e-3) < 1e-15
    with pytest.raises(ValueError):
        config.load_settings({
            "dataset": "d", "fov": 2.0,
            "model": {"backend": "thindisk", "rotation_curve": "rings"},
        })


def test_ring_regularize_median_poly_bezier_and_auto():
    from pyuvkin import ring_regularize as rr

    radii = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    noisy = np.array([40.0, 42.0, 80.0, 44.0, 46.0])  # spike at ring 2
    med = rr.regularize_profile(radii, noisy, "median")
    assert np.allclose(med, np.median(noisy))
    const = rr.regularize_profile(radii, noisy, "constant")
    assert np.allclose(const, np.median(noisy))
    line = rr.regularize_profile(radii, np.array([10.0, 20.0, 30.0, 40.0, 50.0]), "poly1")
    assert abs(line[0] - 10) < 1 and abs(line[-1] - 50) < 1
    bez = rr.regularize_profile(radii, noisy, "bezier")
    assert bez.shape == noisy.shape and bez[0] == noisy[0] and bez[-1] == noisy[-1]
    # auto on low-scatter INC → median
    assert rr.resolve_method("auto", radii, np.full(5, 45.0), "inclination") == "median"
    assert rr.needs_twostage(("vrot", "inclination", "phi"))
    assert not rr.needs_twostage(("vrot", "velocity_dispersion"))
    assert rr.stage2_free(("vrot", "inclination", "phi", "velocity_dispersion")) == (
        "vrot", "velocity_dispersion",
    )
    methods = rr.parse_regtype("inc=bezier pa=median")
    assert methods["inclination"] == "bezier" and methods["phi"] == "median"


def test_rmax_sets_outer_galmod_radius():
    from pyuvkin.models.bbarolo import _resolve_r_max, galmod_ring_radii
    from pyuvkin.models.parameters import DiscParameters

    # explicit rmax is the outer radius (not inflated to a larger lit map)
    assert _resolve_r_max(
        5.0, rmax=2.5, scale_radius=0.3, pixel_scale=0.1,
        field_cap=10.0, hard_cap=12.0, data_r_max=8.0,
    ) == 2.5
    # auto still follows the lit map
    assert _resolve_r_max(
        "auto", rmax=None, scale_radius=0.3, pixel_scale=0.1,
        field_cap=10.0, hard_cap=12.0, data_r_max=8.0,
    ) == 8.0
    p = DiscParameters(scale_radius=0.5)
    r = galmod_ring_radii(
        p, pixel_scale=0.2, shape=(64, 64), n_rings=5, rmax=1.0,
    )
    assert r.size == 5 and abs(r[-1] - 1.0) < 1e-9


def test_ring_errors_1sigma_helper():
    from pyuvkin.results import _ring_errors_1sigma

    err = {"vrot_0": [10.0, 12.0], "vrot_1": [5.0, 7.0], "vrot_2": [8.0, 9.0]}
    lo, hi = _ring_errors_1sigma(err, "vrot", 3)
    assert np.allclose(lo, [10, 5, 8]) and np.allclose(hi, [12, 7, 9])
    assert _ring_errors_1sigma(None, "vrot", 3) is None
    assert _ring_errors_1sigma({"inclination_0": [1, 2]}, "vrot", 2) is None


def test_asymmetric_drift_iorio_formula():
    from pyuvkin import asymmetric_drift as ad

    # flat Σ and σ → ADC = 0 → V_circ = V_rot
    r = np.array([1.0, 2.0, 3.0, 4.0])
    dens = np.ones(4)
    sig = np.full(4, 20.0)
    vrot = np.array([100.0, 120.0, 130.0, 135.0])
    out = ad.asymmetric_drift_correction(r, dens, sig, vrot, 45.0, pol1=1, pol2=1)
    assert np.allclose(out["adc2_kms2"], 0.0, atol=1e-6)
    assert np.allclose(out["vcirc_kms"], vrot, atol=1e-6)

    # declining exponential dens + flat σ → positive pressure support (ADC² > 0)
    dens2 = np.exp(-r / 2.0)
    out2 = ad.asymmetric_drift_correction(r, dens2, sig, vrot, 60.0, pol1=-1, pol2=2)
    assert np.all(out2["adc2_kms2"] > 0)
    assert np.all(out2["vcirc_kms"] > vrot)


def test_vrad_freeable_and_linear_option():
    from pyuvkin.models.parameters import parse_free_ring_params, ring_parameter_names
    from pyuvkin.models.bbarolo import resolve_linear_channels, apply_instrumental_broadening

    free = parse_free_ring_params("VROT VDISP VRAD INC PA")
    assert free == ("vrot", "velocity_dispersion", "vrad", "inclination", "phi")
    names = ring_parameter_names(2, free)
    assert "vrad_0" in names and "vrad_1" in names
    # default free set still omits vrad (fixed at 0)
    assert "vrad" not in parse_free_ring_params(None)
    assert resolve_linear_channels(None) is None
    assert resolve_linear_channels(False) is None
    assert resolve_linear_channels(True) == 0.85
    assert resolve_linear_channels("hanning") == 0.85
    assert resolve_linear_channels(0.5) == 0.5
    cube = np.zeros((11, 3, 3))
    cube[5] = 1.0
    out = apply_instrumental_broadening(cube, 0.85)
    assert out.shape == cube.shape and abs(out.sum() - cube.sum()) < 1e-6
    assert out[5, 0, 0] < 1.0 and out[4, 0, 0] > 0


def _tight_settings(ds, truth, out, method):
    s = mock.demo_settings(ds, truth, out, method=method)
    # a small problem: free only the kinematic parameters, the rest at truth
    keep = ("v_sys", "phi", "maximum_velocity")
    s["priors"] = {k: v for k, v in s["priors"].items() if k in keep}
    for k in PARAMETER_NAMES:
        if k not in keep:
            s["priors"][k] = truth[k]
    s["write_plots"] = True
    return s


@pytest.mark.slow
def test_end_to_end_lbfgs_recovers_truth(tmp_path):
    import pyuvkin

    ds, tp = mock.write_mock_dataset(tmp_path / "mock", None, n_vis=250, n_chan=8, sigma_jy=4e-4)
    truth = json.loads(tp.read_text())
    s = _tight_settings(ds, truth, tmp_path / "fit", "lbfgs")
    s["search"]["start"] = "centre"
    r = pyuvkin.run(s)
    rec = r.best_fit_record
    assert rec["kind"] == "optimisation"
    assert rec["fit_quality"]["chi_squared_reduced"] < 1.1
    for name in ("v_sys", "phi", "maximum_velocity"):
        assert abs(rec["max_log_likelihood"][name] - truth[name]) < 0.1 * abs(truth[name]) + 5.0
    out = tmp_path / "fit"
    for f in ("input_parameters.json", "best_fit_parameters.json", "samples.csv",
              "model_cube.fits", "dirty_data.fits", "dirty_model.fits", "dirty_residual.fits",
              "residual_snr.fits", "summary.png"):
        assert (out / f).exists(), f
    inp = json.loads((out / "input_parameters.json").read_text())
    assert inp["free_parameters"] == ["v_sys", "phi", "maximum_velocity"]
    assert inp["priors"]["phi"]["type"] == "Uniform"
    header = (out / "samples.csv").read_text().splitlines()[0].split(",")
    assert header == ["v_sys", "phi", "maximum_velocity", "log_likelihood", "log_prior",
                      "log_posterior", "weight"]
    from astropy.io import fits

    with fits.open(out / "model_cube.fits") as h:
        assert h[0].data.shape[0] == 8 and h[0].header["CTYPE3"].startswith("FREQ")
        # south-up on disk, same flux as the native cube
        assert abs(h[0].data.sum() - r.products.model_cube.sum()) < 1e-6 * r.products.model_cube.sum()
        assert np.allclose(h[0].data[0], r.products.model_cube[0][::-1], rtol=1e-6, atol=1e-12)


@pytest.mark.slow
def test_end_to_end_sampler_writes_posterior(tmp_path):
    import pyuvkin

    ds, tp = mock.write_mock_dataset(tmp_path / "mock", None, n_vis=150, n_chan=6, sigma_jy=6e-4)
    truth = json.loads(tp.read_text())
    s = _tight_settings(ds, truth, tmp_path / "fit", "emcee")
    s["search"] = {"method": "emcee", "nwalkers": 12, "nsteps": 60}
    s["write_cubes"] = False
    r = pyuvkin.run(s)
    rec = r.best_fit_record
    assert rec["kind"] == "bayesian_sampling"
    assert set(rec["median_pdf"]) == {"v_sys", "phi", "maximum_velocity"}
    assert all(lo >= 0 and hi >= 0 for lo, hi in rec["errors_1sigma"].values())
    rows = np.loadtxt(tmp_path / "fit" / "samples.csv", delimiter=",", skiprows=1)
    assert rows.shape[1] == 7 and rows.shape[0] > 100
    assert (tmp_path / "fit" / "cornerplot.png").exists()


def test_relative_paths_resolve_against_the_settings_file(tmp_path, monkeypatch):
    """`pyuvkin fit dir/settings.json` run from elsewhere must find the dataset
    and put `out` next to the settings file, not in the current directory."""
    from pyuvkin import cli

    ds, tp = mock.write_mock_dataset(tmp_path / "mock", None, n_vis=80, n_chan=6, sigma_jy=1e-3)
    truth = json.loads(tp.read_text())
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    s = _tight_settings("../mock/dataset", truth, "fit_out", "lbfgs")
    s["search"].update({"start": "centre", "maxiter": 5})
    s["write_plots"] = False
    (cfg_dir / "settings.json").write_text(json.dumps(s))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert cli.main(["fit", str(cfg_dir / "settings.json")]) == 0
    assert (cfg_dir / "fit_out" / "best_fit_parameters.json").exists()
    assert not (elsewhere / "fit_out").exists()
    inp = json.loads((cfg_dir / "fit_out" / "input_parameters.json").read_text())
    assert inp["data"]["path"] == str((cfg_dir / "../mock/dataset"))


def test_velocity_window_and_v_sys_share_one_zero_point():
    """Without a rest or reference frequency v = 0 is the mean of the *full*
    band, before the window is cut; the window's own mean must not move it."""
    from pyuvkin import api

    uvd, _ = mock.simulate_disc(n_vis=30, n_chan=16, dv_kms=30.0)
    full_mean = float(np.mean(uvd.frequencies))
    s = config.load_settings({"dataset": uvd, "fov": 2.0, "velocity_window_kms": [-40.0, 220.0]})
    sub, spectral = api.prepare_data(s)
    assert sub.n_chan < uvd.n_chan
    assert abs(spectral.reference_frequency_hz - full_mean) < 1.0
    assert spectral.v_min > -60 and spectral.v_max < 240 and abs(np.mean(spectral.velocities_kms)) > 30


def test_fits_map_orientation_and_centre_checks(tmp_path):
    from astropy.io import fits

    from pyuvimage.products import build_header, to_fits_orientation

    from pyuvkin import freeform
    from pyuvkin.spectral import spectral_axis

    n, ps = 20, 0.1
    native = np.zeros((n, n))
    native[3, 15] = 1.0                       # north-east-ish pixel... row 3 = north, col 15 = west
    meta = {"phase_centre_ra_deg": 150.0, "phase_centre_dec_deg": 2.0}
    spectral = spectral_axis(mock.channel_frequencies(230e9, 4, 30.0), reference_frequency_ghz=230.0)
    h = build_header(n_pix=n, pixel_scale_arcsec=ps, meta=meta, bunit="Jy/pixel")
    path = tmp_path / "map.fits"
    fits.writeto(path, to_fits_orientation(native), h, overwrite=True)
    sb, rec = freeform.surface_brightness_from_fits(
        path, spectral, units="jy_kms_per_pixel", expected_centre_deg=(150.0, 2.0))
    assert np.array_equal(sb.map_jykms > 0, native > 0)
    assert rec["snr_threshold"] is None
    # east-to-the-right map: columns must be reversed on loading
    h2 = h.copy()
    h2["CDELT1"] = abs(h["CDELT1"])
    path2 = tmp_path / "map_flipped.fits"
    fits.writeto(path2, to_fits_orientation(native)[:, ::-1], h2, overwrite=True)
    sb2, _ = freeform.surface_brightness_from_fits(
        path2, spectral, units="jy_kms_per_pixel", expected_centre_deg=(150.0, 2.0))
    assert np.array_equal(sb2.map_jykms, sb.map_jykms)
    # a map centred somewhere else is refused
    with pytest.raises(ValueError, match="centred"):
        freeform.surface_brightness_from_fits(
            path, spectral, units="jy_kms_per_pixel", expected_centre_deg=(150.0, 2.0 + 0.3 / 3600))
    # the S/N mask needs the uncertainty map
    unc = np.full((n, n), 0.4)
    upath = tmp_path / "unc.fits"
    fits.writeto(upath, to_fits_orientation(unc), h, overwrite=True)
    sb3, rec3 = freeform.surface_brightness_from_fits(
        path, spectral, units="jy_kms_per_pixel", snr_threshold=3.0, uncertainty_fits=upath,
        expected_centre_deg=(150.0, 2.0))
    assert sb3.total_flux == 0.0 and rec3["masked_flux_fraction"] == 1.0
    assert sb3.uncertainty_jykms is not None


def test_freeform_morphology_scatter_is_reported(tmp_path):
    """The freeform map comes from the same data; morphology_samples re-fits
    perturbed maps and adds their scatter to the record."""
    import pyuvkin

    ds, tp = mock.write_mock_dataset(tmp_path / "mock", None, n_vis=120, n_chan=6, sigma_jy=6e-4)
    truth = json.loads(tp.read_text())
    s = _tight_settings(ds, truth, tmp_path / "fit", "lbfgs")
    s["search"].update({"start": "centre", "maxiter": 20})
    s["surface_brightness"] = {"type": "freeform", "morphology_samples": 2}
    s["priors"].pop("intensity", None)
    s["priors"].pop("scale_radius", None)
    s["write_plots"] = False
    s["write_cubes"] = False
    r = pyuvkin.run(s)
    m = r.best_fit_record["morphology"]
    assert m["n_samples"] == 2 and set(m["scatter_1sigma"]) == {"v_sys", "phi", "maximum_velocity"}
    assert all(len(v) == 2 for v in m["samples"].values())
