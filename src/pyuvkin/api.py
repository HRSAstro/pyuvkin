"""The pipeline: settings -> data -> geometry -> model -> search -> products.

    import pyuvkin
    result = pyuvkin.run("settings.json")
    result = pyuvkin.run({"dataset": "mydata/", "fov": 3.0, "priors": {...}})
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import autofit as af

from . import __version__, config, freeform, memory, priors, results, search
from . import asymmetric_drift, bbarolo_seed, ring_regularize
from .analysis import KinematicAnalysis
from .grids import BASELINE_PERCENTILE, CubeGeometry, resolve_cube_geometry
from .models import AnalyticSB, FreeformSB, make_renderer
from .models.parameters import (
    DiscParameters, DEFAULT_FREE_RING_PARAMS, make_tilted_ring_class,
    parse_free_ring_params, parse_ring_param_name,
)
from .spectral import SpectralAxis, reference_frequency_hz, spectral_axis
from .transform import CubeTransformer, primary_beam_for
from .uvdata import (
    UVData, channel_window_kms, read_dataset, recompute_noise, select_channels,
    shift_image_centre, single_spw,
)

logger = logging.getLogger("pyuvkin")


@dataclass
class RunResult:
    settings: dict
    geometry: CubeGeometry
    spectral: SpectralAxis
    uvdata: UVData
    model: af.Model
    result: object                 # the PyAutoFit result
    best_fit: DiscParameters
    best_fit_record: dict
    input_record: dict
    written: dict
    products: results.CubeProducts | None = None

    @property
    def samples(self):
        return self.result.samples


# ----------------------------------------------------------------- pieces
def prepare_data(settings: dict) -> tuple[UVData, SpectralAxis]:
    dataset = settings["dataset"]
    if isinstance(dataset, (str, Path)):
        dataset = read_dataset(config.resolve_path(settings, dataset))
    uvd = single_spw(dataset, settings.get("spw"))
    if settings["noise"] != "keep":
        uvd = recompute_noise(uvd, settings["noise"])
    sp_cfg = settings["spectral"]
    # one zero point for everything: fixed on the full band *before* any
    # channel selection, so a velocity window and v_sys share it even when no
    # rest or reference frequency is given (then it is the band's mean)
    ref_hz = reference_frequency_hz(
        uvd.frequencies, sp_cfg["rest_frequency_ghz"], sp_cfg["redshift"],
        sp_cfg["reference_frequency_ghz"],
    )
    if all(sp_cfg[k] is None for k in ("rest_frequency_ghz", "redshift", "reference_frequency_ghz")):
        logger.warning(
            "no spectral.rest_frequency_ghz/redshift or reference_frequency_ghz: v = 0 is "
            "the mean of the full band, %.6f GHz", ref_hz / 1e9,
        )
    if settings["velocity_window_kms"] is not None:
        uvd = channel_window_kms(uvd, ref_hz, *settings["velocity_window_kms"])
    elif settings["channels"] is not None:
        uvd = select_channels(uvd, settings["channels"])
    d_ra, d_dec = (float(v) for v in settings["image_centre"])
    if d_ra or d_dec:
        # pyuvimage's shift takes the grid (y, x) pair: y = dDec, x = -dRA
        uvd = shift_image_centre(uvd, (d_dec, -d_ra))
    spectral = spectral_axis(uvd.frequencies, reference_frequency_ghz=ref_hz / 1e9)
    logger.info(
        "dataset: %d visibilities x %d channels, %.6g-%.6g GHz; %.1f km/s channels, "
        "v = %+.0f..%+.0f km/s about %.6f GHz",
        uvd.n_vis, uvd.n_chan, uvd.frequencies.min() / 1e9, uvd.frequencies.max() / 1e9,
        spectral.dv_kms, spectral.v_min, spectral.v_max, spectral.reference_frequency_hz / 1e9,
    )
    return uvd, spectral


def prepare_geometry(settings: dict, uvd: UVData) -> CubeGeometry:
    geometry = resolve_cube_geometry(
        fov_arcsec=float(settings["fov"]),
        max_baseline_wavelengths=uvd.max_baseline_wavelengths,
        effective_baseline_wavelengths=uvd.baseline_percentile_wavelengths(BASELINE_PERCENTILE),
        pixel_scale=settings["pixel_scale"],
        oversample=int(settings["oversample"]),
        render_oversample=int(settings["render_oversample"]),
        n_pixels=settings["n_pixels"],
    )
    logger.info(
        "image grid %dx%d at %.4g\"/pixel (Nyquist of the longest baseline %.4g\"), "
        "model rendered at %.4g\"/pixel",
        *geometry.shape, geometry.pixel_scale, geometry.nyquist_pixel_scale,
        geometry.render_pixel_scale,
    )
    return geometry


def prepare_surface_brightness(
    settings: dict, uvd: UVData, spectral: SpectralAxis, out: Path,
) -> tuple[AnalyticSB | FreeformSB, dict]:
    sb_cfg = settings["surface_brightness"]
    if sb_cfg["type"] == "analytic":
        return AnalyticSB(), {"type": "analytic"}
    pb_cfg = settings["primary_beam"]
    if sb_cfg["map_fits"]:
        sb, rec = freeform.surface_brightness_from_fits(
            config.resolve_path(settings, sb_cfg["map_fits"]), spectral,
            units=sb_cfg["map_units"], snr_threshold=sb_cfg["snr_threshold"],
            uncertainty_fits=(config.resolve_path(settings, sb_cfg["uncertainty_fits"])
                              if sb_cfg["uncertainty_fits"] else None),
            expected_centre_deg=freeform.expected_map_centre_deg(uvd),
        )
    else:
        sb, rec = freeform.reconstruct_surface_brightness(
            uvd, float(settings["fov"]), spectral, out / "surface_brightness",
            snr_threshold=sb_cfg["snr_threshold"],
            pb_correction=bool(pb_cfg["enabled"]),
            dish_diameter_m=pb_cfg["dish_diameter_m"],
            pyuvimage_kwargs=sb_cfg["pyuvimage"],
        )
    if sb.total_flux <= 0:
        raise RuntimeError("the freeform surface-brightness map holds no flux")
    return sb, {"type": "freeform", **rec}


# ------------------------------------------------------------------- run
def run(settings_or_path, **overrides) -> RunResult:
    settings = config.load_settings(settings_or_path, **overrides)
    out = config.resolve_path(settings, settings["out"])
    out.mkdir(parents=True, exist_ok=True)
    _log_to_file(out / "pyuvkin.log")
    logger.info("pyuvkin %s -> %s", __version__, out)
    memory.log_memory("start")

    uvd, spectral = prepare_data(settings)
    geometry = prepare_geometry(settings, uvd)

    pb_cfg = settings["primary_beam"]
    pb = None
    if pb_cfg["enabled"]:
        pb = primary_beam_for(uvd, geometry, pb_cfg["dish_diameter_m"], pb_cfg["pb_factor"])
        if pb is None:
            logger.warning("primary beam requested but no dish diameter is known; not applied")
        else:
            logger.info("primary beam applied: minimum %.4f at the field edge", float(pb.min()))

    sb, sb_record = prepare_surface_brightness(settings, uvd, spectral, out)

    transformer = CubeTransformer(uvd, geometry, settings["transformer"], primary_beam=pb)
    model_cfg = settings["model"]
    options = {
        "rotation_curve": model_cfg["rotation_curve"],
        "dispersion_curve": model_cfg.get("dispersion_curve", "constant"),
        **(model_cfg["options"] or {}),
    }
    renderer = make_renderer(model_cfg["backend"], geometry, spectral, sb, options)
    memory.log_memory("setup")

    fixed = {}
    parameter_names = tuple(renderer.parameter_names)
    priors_cfg = dict(settings["priors"])
    if sb.is_freeform:
        # the map fixes the morphology and the flux; a prior on intensity in
        # the settings still frees it as a rescaling of the map
        fixed = {"intensity": sb.total_flux}
        parameter_names = tuple(n for n in parameter_names if n != "scale_radius")
        if priors_cfg.pop("scale_radius", None) is not None:
            logger.warning("scale_radius is ignored with a freeform surface brightness")
    parameter_cls = DiscParameters
    free_ring = ()
    if model_cfg["rotation_curve"] == "rings":
        free_ring = parse_free_ring_params(
            options.get("free", options.get("free_ring_parameters", DEFAULT_FREE_RING_PARAMS))
        )
        parameter_cls = make_tilted_ring_class(int(options["n_rings"]), free_ring)
    model = priors.build_model(
        priors_cfg, parameter_names, fixed=fixed, parameter_cls=parameter_cls,
    )
    logger.info("free parameters: %s", results.free_names(model))

    analysis = KinematicAnalysis(transformer, renderer)

    truth = settings.get("truth")
    data_path = (str(config.resolve_path(settings, settings["dataset"]))
                 if isinstance(settings["dataset"], (str, Path)) else "in-memory")
    twostage = bool(options.get(
        "twostage",
        ring_regularize.DEFAULT_TWOSTAGE if model_cfg["rotation_curve"] == "rings" else False,
    ))
    regtype = options.get("regtype", ring_regularize.DEFAULT_REGTYPE)
    input_record = {
        "pyuvkin_version": __version__,
        "started": results.timestamp(),
        "settings": {**config.public(settings), "dataset": data_path},
        "data": {
            "path": data_path,
            "n_visibilities": int(uvd.n_vis),
            "n_channels": int(uvd.n_chan),
            "n_samples": int(uvd.n_samples),
            "n_data": int(transformer.n_data),
            "central_frequency_hz": float(uvd.central_frequency),
            "max_baseline_wavelengths": float(uvd.max_baseline_wavelengths),
            "noise_estimate": uvd.meta.get("noise_estimate", "from file"),
            "median_sigma_jy": float(np.median(np.abs(np.asarray(uvd.noise).real))),
            "image_centre_offset_arcsec_yx": uvd.meta.get("image_centre_offset_arcsec"),
            "channel_selection": uvd.meta.get("channel_selection"),
        },
        "spectral": spectral.as_dict(),
        "geometry": geometry.as_dict(),
        "transformer": transformer.as_dict(),
        "primary_beam": {"applied": pb is not None, **{k: v for k, v in pb_cfg.items()}},
        "model": {
            **renderer.as_dict(),
            "twostage": twostage and ring_regularize.needs_twostage(free_ring),
            "regtype": regtype if twostage else None,
            "free_ring_params": list(free_ring) if free_ring else None,
            "adrift": bool(options.get("adrift", asymmetric_drift.DEFAULT_ADRIFT)),
        },
        "surface_brightness": sb_record,
        "priors": priors.describe_model(model),
        "free_parameters": results.free_names(model),
        "search": settings["search"],
        "truth": truth,
    }
    results.write_json(out / "input_parameters.json", input_record)

    # PyAutoFit writes under out/autofit/<name>/<hash>/
    af.conf.instance.output_path = str(out / "autofit")
    # Starting-point FITS/plots can trip BBarolo's GalMod (Abort trap) on some
    # macOS builds when dens profiles are non-flat; skip them — final products
    # are written by pyuvkin anyway.
    try:
        af.conf.instance["output"]["start_point"] = False
    except Exception:
        pass
    method = search.method_of(settings["search"])
    sampler = search.is_sampler(method)
    if not sampler and model_cfg["backend"] in ("kinms", "bbarolo"):
        logger.warning(
            "%s builds its cube from discrete clouds, so the likelihood is piecewise "
            "constant on small scales and finite-difference gradients vanish: %s may "
            "stop early. Prefer a sampler (nautilus, dynesty, emcee) for this backend.",
            model_cfg["backend"], method,
        )

    n_eval_before = analysis.n_evaluations
    tilted_rings = model_cfg["rotation_curve"] == "rings"
    n_rings = int(options["n_rings"]) if tilted_rings else None

    search_cfg = dict(settings["search"])
    seed_info = None
    if tilted_rings and bbarolo_seed.is_bbarolo_seed(search_cfg.get("start")):
        logger.info(
            "search.start=%r: running FitMod3D on the dirty cube for a seed ...",
            search_cfg.get("start"),
        )

        def _prior_centre(name, default):
            prior = getattr(model, name, None)
            if prior is not None and hasattr(prior, "value_for"):
                try:
                    return float(prior.value_for(0.5))
                except Exception:
                    pass
            return float(default)

        radii0 = np.asarray(
            renderer._radii(_seed_dummy_instance(parameter_cls, model)), float,
        )
        dirty = transformer.dirty_cube(transformer.data)
        rest = settings["spectral"].get("rest_frequency_ghz")
        rest_hz = float(rest) * 1e9 if rest else None
        seed_info = bbarolo_seed.rings_from_fitmod3d(
            dirty_native=dirty, uvd=uvd, geometry=geometry, spectral=spectral,
            radii_arcsec=radii0, free=free_ring,
            out_dir=out / "bbarolo_seed",
            rest_frequency_hz=rest_hz,
            scale_height_arcsec=float(options.get("scale_height_arcsec", 0.0)),
            twostage=bool(options.get("twostage", ring_regularize.DEFAULT_TWOSTAGE)),
            init_centre_ra=_prior_centre("centre_ra", 0.0),
            init_centre_dec=_prior_centre("centre_dec", 0.0),
            init_v_sys=_prior_centre("v_sys", 0.0),
            init_inclination=_prior_centre("inclination", 45.0),
            init_phi=_prior_centre("phi", 0.0),
            init_vrot=_prior_centre("vrot_0", _prior_centre("vrot", 200.0)),
            init_vdisp=_prior_centre(
                "velocity_dispersion_0", _prior_centre("velocity_dispersion", 30.0),
            ),
        )
        start = bbarolo_seed.seed_start_for_model(model, seed_info)
        search_cfg["start"] = start
        input_record["bbarolo_seed"] = {
            "radii_arcsec": seed_info["radii_arcsec"],
            "free": seed_info["free"],
            "start": start,
            "rings": {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in seed_info["rings"].items() if k != "path"
            },
            "out_dir": seed_info["out_dir"],
        }
        results.write_json(out / "input_parameters.json", input_record)
        logger.info(
            "FitMod3D seed ready (%d rings); starting uv-plane %s from it",
            len(seed_info["radii_arcsec"]), method,
        )

    fit_result = _run_search(
        search_cfg, model, analysis, name=model_cfg["backend"], tag="stage1",
        tilted_rings=tilted_rings, n_rings=n_rings,
        discrete_clouds=model_cfg["backend"] in ("bbarolo", "kinms"),
    )
    logger.info(
        "stage 1 finished after %d likelihood evaluations",
        analysis.n_evaluations - n_eval_before,
    )
    memory.log_memory("stage1")

    twostage_info = None
    if (
        twostage
        and model_cfg["rotation_curve"] == "rings"
        and ring_regularize.needs_twostage(free_ring)
    ):
        stage1_best = fit_result.samples.max_log_likelihood()
        radii = np.asarray(renderer._radii(stage1_best), float)
        logger.info(
            "TWOSTAGE: regularising geometry (%s) then re-fitting %s ...",
            regtype, list(ring_regularize.stage2_free(free_ring)),
        )
        regularized = ring_regularize.regularize_geometry(
            stage1_best, radii, free_ring, regtype,
        )
        if not regularized:
            logger.warning("TWOSTAGE: nothing to regularise; skipping stage 2")
        else:
            geom_fixed = ring_regularize.geometry_fixed_from_regularized(regularized)
            names2 = tuple(
                n for n in parameter_names
                if (parsed := parse_ring_param_name(n)) is None
                or parsed[0] not in ring_regularize.GEOMETRY_BASES
            )
            priors2 = _priors_for_stage2(priors_cfg, names2)
            fixed2 = {**fixed, **geom_fixed}
            # pin any remaining class attrs (shared centres, etc.) from stage 1
            for name in parameter_cls.parameter_names:
                if name in names2 or name in fixed2:
                    continue
                if hasattr(stage1_best, name):
                    fixed2[name] = float(getattr(stage1_best, name))
            model2 = priors.build_model(
                priors2, names2, fixed=fixed2, parameter_cls=parameter_cls,
            )
            logger.info("stage 2 free parameters: %s", results.free_names(model2))
            # start stage 2 near stage-1 kinematics
            start2 = {
                n: float(getattr(stage1_best, n))
                for n in results.free_names(model2)
                if hasattr(stage1_best, n)
            }
            search2 = {**settings["search"], "start": start2}
            n_eval_s2 = analysis.n_evaluations
            fit_result2 = _run_search(
                search2, model2, analysis, name=f"{model_cfg['backend']}_stage2", tag="stage2",
                tilted_rings=True, n_rings=int(options["n_rings"]),
                discrete_clouds=True,
            )
            logger.info(
                "stage 2 finished after %d likelihood evaluations",
                analysis.n_evaluations - n_eval_s2,
            )
            memory.log_memory("stage2")
            results.write_samples_csv(fit_result, model, out / "samples_stage1.csv")
            stage1_record, _ = results.best_fit_record(
                fit_result, model, analysis, method, sampler, truth,
            )
            twostage_info = {
                "regtype": regtype,
                "radii_arcsec": radii.tolist(),
                "geometry_stage1": ring_regularize.stage1_geometry_record(
                    stage1_best, free_ring, int(options["n_rings"]),
                ),
                "geometry_regularized": {b: v.tolist() for b, v in regularized.items()},
                "stage1": {
                    "free_parameters": stage1_record["free_parameters"],
                    "max_log_likelihood": stage1_record["max_log_likelihood"],
                    "fit_quality": stage1_record["fit_quality"],
                },
            }
            fit_result, model = fit_result2, model2
            input_record["priors_stage2"] = priors.describe_model(model)
            input_record["free_parameters"] = results.free_names(model)
            results.write_json(out / "input_parameters.json", input_record)

    record, best = results.best_fit_record(fit_result, model, analysis, method, sampler, truth)
    if twostage_info is not None:
        record["twostage"] = twostage_info
    if seed_info is not None and "bbarolo_seed" in input_record:
        record["bbarolo_seed"] = input_record["bbarolo_seed"]
    adrift_arrays = None
    if (
        model_cfg["rotation_curve"] == "rings"
        and bool(options.get("adrift", asymmetric_drift.DEFAULT_ADRIFT))
    ):
        pol1 = int(options.get("adrift_pol1", options.get("adriftpol1",
                     asymmetric_drift.DEFAULT_ADRIFT_POL1)))
        pol2 = int(options.get("adrift_pol2", options.get("adriftpol2",
                     asymmetric_drift.DEFAULT_ADRIFT_POL2)))
        try:
            logger.info(
                "ADRIFT: asymmetric drift correction (pol1=%d, pol2=%d) ...", pol1, pol2,
            )
            adrift_arrays = asymmetric_drift.compute_for_rings(
                renderer, best, pol1=pol1, pol2=pol2,
            )
            record["asymmetric_drift"] = asymmetric_drift.result_as_jsonable(adrift_arrays)
        except Exception as e:
            logger.warning("ADRIFT failed: %s", e)
            record["asymmetric_drift"] = {"error": str(e)}
    n_morph = int(settings["surface_brightness"].get("morphology_samples") or 0)
    if sb.is_freeform and n_morph > 0:
        record["morphology"] = morphology_scatter(
            settings, sb, transformer, geometry, spectral, model, best, record, n_morph, model_cfg,
        )
    record["finished"] = results.timestamp()
    record["n_likelihood_evaluations"] = int(analysis.n_evaluations)
    written = {
        "input_parameters": str(results.write_json(out / "input_parameters.json", input_record)),
        "best_fit_parameters": str(results.write_json(out / "best_fit_parameters.json", record)),
        "samples": str(results.write_samples_csv(fit_result, model, out / "samples.csv")),
    }
    if (out / "samples_stage1.csv").exists():
        written["samples_stage1"] = str(out / "samples_stage1.csv")
    if adrift_arrays is not None:
        written["asymdrift"] = str(
            asymmetric_drift.write_asymdrift_txt(adrift_arrays, out / "asymdrift.txt")
        )
    _log_best_fit(record)

    products = None
    if settings["write_cubes"] or settings["write_plots"]:
        products = results.make_cube_products(transformer, analysis, best)
    if settings["write_cubes"]:
        rest = settings["spectral"]["rest_frequency_ghz"]
        written.update(results.write_cubes(
            products, uvd, geometry, spectral, out,
            extra_header={"BACKEND": model_cfg["backend"], "SEARCH": method},
            rest_frequency_hz=float(rest) * 1e9 if rest else None,
            spectral_frame=settings["spectral"]["frame"],
        ))
    if settings["write_plots"]:
        title = (f"{model_cfg['backend']} / {sb.as_dict()['type']} SB / {method}: "
                 f"chi2/N = {record['fit_quality']['chi_squared_reduced']:.4f}")
        written["summary"] = str(results.summary_figure(
            products, geometry, spectral, out, title=title,
            n_channels=int(settings["n_plot_channels"]),
        ))
        written["moment_maps"] = str(results.moment_maps_figure(
            products, geometry, spectral, out, title=title))
        written["channel_maps"] = str(results.channel_maps_figure(
            products, geometry, spectral, out, title=title))
        written["pv_diagram"] = str(results.pv_diagram_figure(
            products, geometry, spectral, best, out, title=title,
            rotation_curve=model_cfg["rotation_curve"],
            ring_radii_arcsec=_ring_radii_for_plot(renderer, best),
            backend=model_cfg["backend"],
            errors_1sigma=record.get("errors_1sigma"),
        ))
        if sampler:
            p = results.corner_figure(fit_result, model, out, truth)
            if p is not None:
                written["cornerplot"] = str(p)
    results.write_json(out / "best_fit_parameters.json", {**record, "written": written})
    logger.info("wrote %s", ", ".join(sorted(Path(v).name for v in written.values())))
    memory.log_memory("done")
    return RunResult(
        settings=settings, geometry=geometry, spectral=spectral, uvdata=uvd, model=model,
        result=fit_result, best_fit=best, best_fit_record=record, input_record=input_record,
        written=written, products=products,
    )


def morphology_scatter(settings, sb, transformer, geometry, spectral, model, best, record, n,
                       model_cfg) -> dict:
    """Re-fit ``n`` perturbed surface-brightness maps and report the scatter
    of the maximum-likelihood kinematics: the part of the error budget the
    main fit cannot see because the map was fixed from the same data."""
    if sb.uncertainty_jykms is None:
        logger.warning("morphology_samples requested but the map has no uncertainty; skipped")
        return {"n_samples": 0, "note": "map has no uncertainty"}
    names = results.free_names(model)
    start = {k: float(getattr(best, k)) for k in names}
    cfg = {"method": "lbfgs", "start": start, "maxiter": int(settings["search"].get("maxiter", 300))}
    options = {
        "rotation_curve": model_cfg["rotation_curve"],
        "dispersion_curve": model_cfg.get("dispersion_curve", "constant"),
        **(model_cfg["options"] or {}),
    }
    rows = []
    logger.info("morphology: re-fitting %d perturbed surface-brightness maps ...", n)
    parameter_cls = DiscParameters
    if model_cfg["rotation_curve"] == "rings":
        free = parse_free_ring_params(
            options.get("free", options.get("free_ring_parameters", DEFAULT_FREE_RING_PARAMS))
        )
        parameter_cls = make_tilted_ring_class(int(options["n_rings"]), free)
    for k in range(n):
        sb_k = sb.perturbed(seed=1000 + k)
        renderer_k = make_renderer(model_cfg["backend"], geometry, spectral, sb_k, options)
        analysis_k = KinematicAnalysis(transformer, renderer_k)
        fixed_k = {"intensity": sb_k.total_flux}
        model_k = priors.build_model(
            _priors_for(settings, sb),
            tuple(n_ for n_ in renderer_k.parameter_names if n_ != "scale_radius"),
            fixed=fixed_k, parameter_cls=parameter_cls,
        )
        search_k, _ = search.build_search(cfg, name=f"{model_cfg['backend']}_morphology", model=model_k,
                                          unique_tag=f"map{k}")
        r = search_k.fit(model=model_k, analysis=analysis_k)
        ml = r.samples.max_log_likelihood()
        rows.append([float(getattr(ml, n_)) for n_ in names])
        logger.info("  map %d/%d: %s", k + 1, n, ", ".join(f"{n_}={v:.4g}" for n_, v in zip(names, rows[-1])))
    arr = np.asarray(rows)
    scatter = dict(zip(names, arr.std(axis=0, ddof=1).tolist() if n > 1 else [0.0] * len(names)))
    out_rec = {
        "n_samples": n,
        "method": "independent Gaussian perturbation of the map per pixel by its 1-sigma "
                  "uncertainty; L-BFGS re-fit from the best fit",
        "scatter_1sigma": scatter,
        "samples": {n_: arr[:, i].tolist() for i, n_ in enumerate(names)},
    }
    if "errors_1sigma" in record:
        out_rec["errors_total_1sigma"] = {
            n_: [float(np.hypot(lo, scatter[n_])), float(np.hypot(hi, scatter[n_]))]
            for n_, (lo, hi) in record["errors_1sigma"].items()
        }
    return out_rec


def _priors_for(settings, sb) -> dict:
    priors_cfg = dict(settings["priors"])
    if sb.is_freeform:
        priors_cfg.pop("scale_radius", None)
    return priors_cfg


def _priors_for_stage2(priors_cfg: dict, parameter_names: tuple[str, ...]) -> dict:
    """Keep priors that apply to stage-2 free parameters (broadcast bases OK)."""
    out = {}
    names = set(parameter_names)
    for key, val in dict(priors_cfg or {}).items():
        if key in names:
            out[key] = val
            continue
        # broadcast base (vrot, velocity_dispersion) or vrot_i alias
        base = key[:-2] if key.endswith("_i") else key
        if any(
            (p := parse_ring_param_name(n)) is not None and p[0] == base
            for n in names
        ):
            out[key] = val
    return out


def _seed_dummy_instance(parameter_cls, model):
    """Instance at prior centres (or defaults) for ring-radius evaluation."""
    from .models.parameters import default_for_param

    vals = {}
    for name in parameter_cls.parameter_names:
        prior = getattr(model, name, None)
        if prior is not None and hasattr(prior, "value_for"):
            try:
                vals[name] = float(prior.value_for(0.5))
                continue
            except Exception:
                pass
        vals[name] = float(default_for_param(name))
    return parameter_cls.from_dict(vals)


def _run_search(search_cfg: dict, model, analysis, *, name: str, tag: str,
                tilted_rings: bool = False, n_rings: int | None = None,
                discrete_clouds: bool = False):
    """Run the configured search (with restarts); return the best PyAutoFit result."""
    method = search.method_of(search_cfg)
    sampler = search.is_sampler(method)
    restarts = search.n_restarts(search_cfg)
    logger.info(
        "running %s %s (%s%s) ...",
        tag, method, "sampling" if sampler else "optimisation",
        f", {restarts} starts" if restarts > 1 else "",
    )
    memory.log_memory(tag)

    probe_starts = None
    if discrete_clouds and not sampler:
        probe_starts = search.pick_probe_starts(model, analysis, search_cfg, restarts)

    fit_result = None
    n_runs = len(probe_starts) if probe_starts is not None else restarts
    for k in range(n_runs):
        if probe_starts is not None:
            cfg = {**search_cfg, "start": probe_starts[k]}
        else:
            cfg = search_cfg if k == 0 else {**search_cfg, "start": "prior"}
        nl_search, _ = search.build_search(
            cfg, name=name, model=model,
            unique_tag=f"{tag}_start{k}" if n_runs > 1 else tag,
            tilted_rings=tilted_rings, n_rings=n_rings,
            discrete_clouds=discrete_clouds,
        )
        r = nl_search.fit(model=model, analysis=analysis)
        ll = float(r.samples.max_log_likelihood_sample.log_likelihood)
        if n_runs > 1:
            logger.info("%s start %d/%d: max log likelihood %.2f", tag, k + 1, n_runs, ll)
        if fit_result is None or ll > float(
            fit_result.samples.max_log_likelihood_sample.log_likelihood
        ):
            fit_result = r
    return fit_result


def _ring_radii_for_plot(renderer, best: DiscParameters):
    """GalMod ring radii for the PV overlay, or None for other backends."""
    if getattr(renderer, "name", None) != "bbarolo":
        return None
    return renderer._radii(best)


def _log_best_fit(record: dict) -> None:
    q = record["fit_quality"]
    logger.info("chi^2/N = %.4f (chi^2 = %.1f, N = %d)", q["chi_squared_reduced"], q["chi_squared"], q["n_data"])
    if record.get("log_evidence") is not None:
        logger.info("log evidence = %.2f", record["log_evidence"])
    for name in record["free_parameters"]:
        line = f"  {name:20s} = {record['max_log_likelihood'][name]:12.5g}"
        if "median_pdf" in record:
            lo, hi = record["errors_1sigma"][name]
            line += f"   (median {record['median_pdf'][name]:.5g} -{lo:.3g} +{hi:.3g})"
        if "truth_comparison" in record and name in record["truth_comparison"]:
            line += f"   truth {record['truth_comparison'][name]['truth']:.5g}"
        logger.info(line)


def _log_to_file(path: Path) -> None:
    root = logging.getLogger("pyuvkin")
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            root.removeHandler(h)
    fh = logging.FileHandler(path, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(fh)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
