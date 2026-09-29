"""The non-linear search: Bayesian sampling or optimisation, from settings.

    "search": {"method": "nautilus", "n_live": 200, "number_of_cores": 1}
    "search": {"method": "dynesty",  "nlive": 100}
    "search": {"method": "emcee",    "nwalkers": 40, "nsteps": 1000}
    "search": {"method": "lbfgs",    "maxiter": 500, "start": "centre", "restarts": 4}
    "search": {"method": "bfgs",     "start": {"phi": 40, "inclination": 50}}

Samplers (nautilus, dynesty, emcee) give posterior samples, evidences (nested
samplers) and errors; optimisers (lbfgs, bfgs) give a maximum-likelihood
point and a trace of the evaluations. Optimisers take two extra keys:
``start`` -- "prior" (a random draw, PyAutoFit's default), "centre" (the
middle of every prior), ``"bbarolo"`` / ``"fitmod3d"`` (rings mode only:
run BBarolo FitMod3D on the dirty cube and start from those rings), or a
dict of starting values -- and ``restarts``, the number of independent
starts to run, keeping the best. L-BFGS-B is given the prior box as bounds
so a finite-difference step can never leave the prior.
Any other keyword in the block is passed to the PyAutoFit search class.
"""

from __future__ import annotations

import logging
import os

import autofit as af

logger = logging.getLogger("pyuvkin")

SAMPLERS = ("nautilus", "dynesty", "dynesty_dynamic", "emcee")
OPTIMISERS = ("lbfgs", "bfgs")
METHODS = SAMPLERS + OPTIMISERS

_CLASSES = {
    "nautilus": af.Nautilus,
    "dynesty": af.DynestyStatic,
    "dynesty_dynamic": af.DynestyDynamic,
    "emcee": af.Emcee,
    "lbfgs": af.LBFGS,
    "bfgs": af.BFGS,
}

#: pyuvkin defaults where PyAutoFit's are sized for a different problem
_DEFAULTS = {
    "nautilus": {"n_live": 200},
    "dynesty": {"nlive": 100},
    "dynesty_dynamic": {"nlive": 100},
    "emcee": {"nwalkers": 40, "nsteps": 1000},
    # finite-difference step in physical units; scipy's 1e-8 is below the
    # level at which a chi^2 of ~1e5 changes reliably
    "lbfgs": {"eps": 1e-6},
    "bfgs": {"eps": 1e-6},
}

#: GalMod / free-ring likelihoods are stepwise; larger FD steps and a
#: gradient tolerance that loosens with the number of rings (``2.5e-3 *
#: n_rings``, so ``1e-2`` at 4 rings).
_TILTED_RING_EPS = 1.0
_TILTED_RING_GTOL_PER_RING = 2.5e-3

_NOT_KWARGS = {"method", "number_of_cores", "name", "path_prefix", "unique_tag", "start", "restarts"}
#: search.auto_correlation (emcee only) is folded into AutoCorrelationsSettings


def tilted_ring_optimiser_defaults(n_rings: int | None = None) -> dict:
    """L-BFGS/BFGS defaults for free tilted-ring fits."""
    n = max(1, int(n_rings) if n_rings is not None else 1)
    return {"eps": _TILTED_RING_EPS, "gtol": _TILTED_RING_GTOL_PER_RING * n}


def resolve_number_of_cores(search_cfg: dict) -> int:
    """``PYUVKIN_CORES`` env, then ``search.number_of_cores``; default 1.

    The forward model holds NUFFT plans (and JAX arrays when the JAX NUFFT is
    used) that do not travel well to worker processes, and a KinMS cube is a
    fraction of a second, so one core is the safe default. Optimisers ignore it.
    """
    env = os.environ.get("PYUVKIN_CORES", "").strip().lower()
    cores = env or str(search_cfg.get("number_of_cores", 1)).strip().lower()
    if cores == "auto":
        return max(1, (os.cpu_count() or 2) - 1)
    return max(1, int(cores))


def method_of(search_cfg: dict) -> str:
    method = str(search_cfg.get("method", "nautilus")).lower()
    if method not in METHODS:
        raise ValueError(f"search.method must be one of {METHODS}, not {method!r}")
    return method


def is_sampler(method: str) -> bool:
    return method in SAMPLERS


def n_restarts(search_cfg: dict) -> int:
    if is_sampler(method_of(search_cfg)):
        return 1
    return max(1, int(search_cfg.get("restarts", 1)))


def initializer_from(start, model: af.Model | None):
    """The autofit initializer for an optimiser's ``start`` setting."""
    if start is None or start == "prior":
        return None
    if start == "centre":
        return af.InitializerBall(lower_limit=0.5, upper_limit=0.5)
    if isinstance(start, dict):
        if model is None:
            raise ValueError("a dict 'start' needs the model")
        points = {}
        for name, value in start.items():
            prior = getattr(model, name, None)
            if not isinstance(prior, af.Prior):
                logger.warning("start value for %r ignored: not a free parameter", name)
                continue
            points[prior] = float(value)
        missing = [p for p in model.priors if p not in points]
        if missing:
            # autofit needs every prior; fill the rest from the prior centres
            for prior in missing:
                points[prior] = float(prior.value_for(0.5))
        return af.InitializerParamStartPoints(points)
    raise ValueError(
        "search.start must be 'prior', 'centre', 'bbarolo'/'fitmod3d' "
        "(rings mode; resolved in api.run), or a dict of values"
    )


def _emcee_auto_correlation(nsteps: int, cfg: dict | None):
    """autofit's emcee reads the autocorrelation over the last ``check_size``
    steps and falls over when the chain is shorter than that; keep the window
    inside the chain. ``search.auto_correlation`` overrides any field."""
    from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelationsSettings

    cfg = dict(cfg or {})
    check_size = int(cfg.pop("check_size", min(100, max(2, nsteps // 2))))
    if check_size >= nsteps:
        check_size = max(2, nsteps // 2)
    return AutoCorrelationsSettings(
        check_for_convergence=bool(cfg.pop("check_for_convergence", True)),
        check_size=check_size,
        required_length=int(cfg.pop("required_length", min(50, check_size))),
        change_threshold=float(cfg.pop("change_threshold", 0.01)),
    )


def build_search(
    search_cfg: dict, name: str = "fit", path_prefix: str | None = None,
    model: af.Model | None = None, unique_tag: str | None = None,
    tilted_rings: bool = False, n_rings: int | None = None,
):
    method = method_of(search_cfg)
    cls = _CLASSES[method]
    kwargs = dict(_DEFAULTS.get(method, {}))
    if tilted_rings and method in OPTIMISERS:
        kwargs.update(tilted_ring_optimiser_defaults(n_rings))
    kwargs.update({k: v for k, v in search_cfg.items() if k not in _NOT_KWARGS})
    if method in SAMPLERS:
        kwargs["number_of_cores"] = resolve_number_of_cores(search_cfg)
        if method == "emcee":
            kwargs["auto_correlation_settings"] = _emcee_auto_correlation(
                int(kwargs["nsteps"]), kwargs.pop("auto_correlation", None),
            )
    else:
        init = initializer_from(search_cfg.get("start"), model)
        if init is not None:
            kwargs["initializer"] = init
        if method == "lbfgs" and "clipper" not in kwargs:
            kwargs["clipper"] = af.ClipperPriorBox()
    logger.info("search: %s %s", method, {k: v for k, v in kwargs.items()
                                          if k not in ("initializer", "clipper")})
    return cls(
        name=search_cfg.get("name", name), path_prefix=path_prefix,
        unique_tag=search_cfg.get("unique_tag", unique_tag), **kwargs,
    ), method
