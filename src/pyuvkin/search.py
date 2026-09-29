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
import numpy as np

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

#: GalMod / KinMS / free-ring likelihoods are stepwise; larger FD steps and a
#: gradient tolerance that loosens with the number of rings (``2.5e-3 *
#: n_rings``, so ``1e-2`` at 4 rings). Parametric cloud backends use the
#: same ``eps`` with a fixed ``gtol``.
_CLOUD_EPS = 1.0
_CLOUD_GTOL = 1e-3
_TILTED_RING_EPS = _CLOUD_EPS
_TILTED_RING_GTOL_PER_RING = 2.5e-3

#: Cheap prior draws to score before L-BFGS on cloud backends (per restart kept).
_CLOUD_PROBES_PER_RESTART = 8
_CLOUD_PROBES_MIN = 24

_NOT_KWARGS = {
    "method", "number_of_cores", "name", "path_prefix", "unique_tag",
    "start", "restarts", "n_probe", "probe_seed",
}
#: search.auto_correlation (emcee only) is folded into AutoCorrelationsSettings


def tilted_ring_optimiser_defaults(n_rings: int | None = None) -> dict:
    """L-BFGS/BFGS defaults for free tilted-ring fits."""
    n = max(1, int(n_rings) if n_rings is not None else 1)
    return {"eps": _TILTED_RING_EPS, "gtol": _TILTED_RING_GTOL_PER_RING * n}


def cloud_optimiser_defaults(n_rings: int | None = None) -> dict:
    """L-BFGS/BFGS defaults for discrete-cloud backends (BBarolo, KinMS)."""
    if n_rings is not None:
        return tilted_ring_optimiser_defaults(n_rings)
    return {"eps": _CLOUD_EPS, "gtol": _CLOUD_GTOL}


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


def n_probe_starts(search_cfg: dict, restarts: int) -> int:
    """How many prior draws to score before refining (0 = skip probing)."""
    if "n_probe" in search_cfg:
        return max(0, int(search_cfg["n_probe"]))
    return max(_CLOUD_PROBES_MIN, int(restarts) * _CLOUD_PROBES_PER_RESTART)


def draw_prior_starts(model: af.Model, n: int, *, seed: int = 0) -> list[dict]:
    """``n`` random draws from the free priors as start dicts."""
    from .results import free_names

    rng = np.random.default_rng(seed)
    names = free_names(model)
    out = []
    for _ in range(max(0, int(n))):
        start = {}
        for name in names:
            prior = getattr(model, name, None)
            if not isinstance(prior, af.Prior):
                continue
            start[name] = float(prior.value_for(float(rng.random())))
        if start:
            out.append(start)
    return out


def centre_start(model: af.Model) -> dict:
    from .results import free_names

    start = {}
    for name in free_names(model):
        prior = getattr(model, name, None)
        if isinstance(prior, af.Prior):
            start[name] = float(prior.value_for(0.5))
    return start


def score_starts(model: af.Model, analysis, starts: list[dict]) -> list[tuple[float, dict]]:
    """Single likelihood evaluation per start; highest LL first."""
    scored = []
    for start in starts:
        try:
            inst = model.instance_from_prior_medians()
            for name, value in start.items():
                if hasattr(inst, name):
                    setattr(inst, name, float(value))
            ll = float(analysis.log_likelihood_function(inst))
        except Exception:
            continue
        if np.isfinite(ll):
            scored.append((ll, start))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def pick_probe_starts(
    model: af.Model, analysis, search_cfg: dict, restarts: int,
) -> list[dict] | None:
    """For cloud-backend optimisers: probe prior draws, return top starts.

    Returns ``None`` to keep the usual start/restart behaviour (user gave an
    explicit dict start, or probing disabled).
    """
    method = method_of(search_cfg)
    if method not in OPTIMISERS:
        return None
    start = search_cfg.get("start", "prior")
    if isinstance(start, dict):
        return None  # honour an explicit seed
    n_probe = n_probe_starts(search_cfg, restarts)
    if n_probe <= 0:
        return None

    candidates: list[dict] = []
    if start == "centre" or start is None or start == "prior":
        c = centre_start(model)
        if c:
            candidates.append(c)
    candidates.extend(draw_prior_starts(model, n_probe, seed=int(search_cfg.get("probe_seed", 0))))

    logger.info(
        "probing %d starts for %s (keeping top %d for refinement) ...",
        len(candidates), method, restarts,
    )
    scored = score_starts(model, analysis, candidates)
    if not scored:
        logger.warning("all probed starts failed; falling back to default starts")
        return None
    logger.info(
        "probe best preliminary log likelihood %.2f (worst kept will be refined next)",
        scored[0][0],
    )
    # unique-ish: drop starts within 1e-3 relative of an already kept vector
    picked: list[dict] = []
    for ll, s in scored:
        if len(picked) >= restarts:
            break
        vec = np.array([s[k] for k in sorted(s)])
        if any(np.allclose(vec, np.array([p[k] for k in sorted(p)]), rtol=0, atol=1e-6) for p in picked):
            continue
        picked.append(s)
    return picked or None


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
    discrete_clouds: bool = False,
):
    method = method_of(search_cfg)
    cls = _CLASSES[method]
    kwargs = dict(_DEFAULTS.get(method, {}))
    if method in OPTIMISERS and (tilted_rings or discrete_clouds):
        kwargs.update(cloud_optimiser_defaults(n_rings if tilted_rings else None))
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
