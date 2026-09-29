"""Priors from the settings JSON -> the PyAutoFit model.

    "priors": {
      "centre_ra":  {"type": "Uniform", "lower": -0.2, "upper": 0.2},
      "intensity":  {"type": "LogUniform", "lower": 0.01, "upper": 10},
      "phi":        {"type": "Gaussian", "mean": 45, "sigma": 20},
      "v_sys":      {"type": "Gaussian", "mean": 0, "sigma": 50, "lower": -300, "upper": 300},
      "inclination": 60.0,                      # a bare number fixes it
      "vrot":       {"type": "Uniform", "lower": 50, "upper": 400},  # all rings
      "vrot_2":     {"type": "Uniform", "lower": 100, "upper": 300}  # override
    }

Every parameter the backend responds to is either given a prior, fixed by a
number (or ``{"type": "fixed", "value": ...}``), or fixed at its default with
a warning -- so a typo in a name is an error, not a silently unfitted
parameter. LensKin's ``UniformPrior`` / ``LogUniformPrior`` / ``lower_limit``
spellings are accepted too.

With ``rotation_curve: "rings"``, a prior on a freeable base name
(``vrot``, ``inclination``, ``phi``, ``velocity_dispersion``, …) broadcasts
to every ``base_i``; per-ring keys override. ``vrot_i`` is accepted as an
alias for the ``vrot`` broadcast.
"""

from __future__ import annotations

import logging

import autofit as af

from .models import DEFAULTS, PARAMETER_NAMES, DiscParameters
from .models.parameters import (
    RING_FREEABLE,
    default_for_param,
    is_ring_param_name,
    parameter_unit,
    parse_ring_param_name,
)

logger = logging.getLogger("pyuvkin")

_TYPES = {
    "uniform": "Uniform", "uniformprior": "Uniform",
    "loguniform": "LogUniform", "loguniformprior": "LogUniform", "log_uniform": "LogUniform",
    "gaussian": "Gaussian", "gaussianprior": "Gaussian", "normal": "Gaussian",
    "loggaussian": "LogGaussian", "loggaussianprior": "LogGaussian",
    "fixed": "fixed", "constant": "fixed",
}


def _bound(cfg: dict, lo_key: str, default):
    for key in (lo_key, f"{lo_key}_limit"):
        if key in cfg:
            return float(cfg[key])
    return default


def prior_from_cfg(name: str, cfg):
    """A number fixes the parameter; a dict describes a prior."""
    if isinstance(cfg, bool):
        raise ValueError(f"prior for {name!r} cannot be a boolean")
    if isinstance(cfg, (int, float)):
        return float(cfg)
    if not isinstance(cfg, dict) or "type" not in cfg:
        raise ValueError(f"prior for {name!r} must be a number or an object with a 'type'")
    kind = _TYPES.get(str(cfg["type"]).replace(" ", "").lower())
    if kind is None:
        raise ValueError(
            f"prior for {name!r}: unknown type {cfg['type']!r}; use Uniform, "
            "LogUniform, Gaussian, LogGaussian or fixed"
        )
    if kind == "fixed":
        return float(cfg["value"])
    lower = _bound(cfg, "lower", None)
    upper = _bound(cfg, "upper", None)
    if kind in ("Uniform", "LogUniform"):
        if lower is None or upper is None:
            raise ValueError(f"prior for {name!r}: {kind} needs 'lower' and 'upper'")
        if upper <= lower:
            raise ValueError(f"prior for {name!r}: upper <= lower")
        if kind == "LogUniform":
            if lower <= 0:
                raise ValueError(f"prior for {name!r}: LogUniform needs lower > 0")
            return af.LogUniformPrior(lower_limit=lower, upper_limit=upper)
        return af.UniformPrior(lower_limit=lower, upper_limit=upper)
    mean = float(cfg["mean"])
    sigma = float(cfg["sigma"])
    kw = {}
    if lower is not None:
        kw["lower_limit"] = lower
    if upper is not None:
        kw["upper_limit"] = upper
    if kind == "LogGaussian":
        if kw:
            raise ValueError(f"prior for {name!r}: LogGaussian takes no limits")
        return af.LogGaussianPrior(mean=mean, sigma=sigma)
    if kw:
        return af.TruncatedGaussianPrior(mean=mean, sigma=sigma, **kw)
    return af.GaussianPrior(mean=mean, sigma=sigma)


def _ring_names_for_base(parameter_names: tuple[str, ...], base: str) -> list[str]:
    return [
        n for n in parameter_names
        if (parsed := parse_ring_param_name(n)) is not None and parsed[0] == base
    ]


def _expand_ring_priors(priors_cfg: dict, parameter_names: tuple[str, ...]) -> dict:
    """Broadcast base-name priors onto every ``base_i`` (per-ring keys win).

    Accepts ``vrot_i`` / ``inclination_i`` / … as aliases for the broadcast.
    """
    cfg = dict(priors_cfg)
    for base in RING_FREEABLE:
        alias = f"{base}_i"
        if alias in cfg:
            if base not in cfg:
                cfg[base] = cfg.pop(alias)
            else:
                cfg.pop(alias)
    for base in RING_FREEABLE:
        if base not in cfg:
            continue
        broadcast = cfg.pop(base)
        ring_names = _ring_names_for_base(parameter_names, base)
        if ring_names:
            for name in ring_names:
                cfg.setdefault(name, broadcast)
        elif base in parameter_names:
            # shared (not free per ring): restore as a normal prior key
            cfg[base] = broadcast
        else:
            raise ValueError(
                f"prior {base!r} is not used by this model "
                f"(not free per ring and not a shared parameter)"
            )
    return cfg


def _known_prior_names(parameter_names: tuple[str, ...]) -> set[str]:
    known = set(PARAMETER_NAMES) | {n for n in parameter_names if is_ring_param_name(n)}
    known |= set(RING_FREEABLE)
    known |= {f"{b}_i" for b in RING_FREEABLE}
    return known


def build_model(
    priors_cfg: dict,
    parameter_names: tuple[str, ...],
    fixed: dict | None = None,
    parameter_cls=DiscParameters,
) -> af.Model:
    """The `af.Model` for one backend.

    ``parameter_names`` are the ones the backend responds to; ``fixed`` are
    values the pipeline has decided (e.g. the freeform surface brightness
    fixing ``intensity``), which a prior in the settings may still override.
    ``parameter_cls`` is ``DiscParameters`` or a tilted-ring class from
    ``make_tilted_ring_class``.
    """
    priors_cfg = _expand_ring_priors(dict(priors_cfg or {}), parameter_names)
    fixed = dict(fixed or {})
    known = _known_prior_names(parameter_names)
    unknown = set(priors_cfg) - known
    if unknown:
        raise ValueError(
            f"unknown parameter(s) in priors: {sorted(unknown)}. "
            f"Known: {sorted(known)}"
        )
    ignored = set(priors_cfg) - set(parameter_names)
    if ignored:
        logger.warning(
            "priors for %s are ignored: this backend does not use them", sorted(ignored),
        )
    cls_names = tuple(getattr(parameter_cls, "parameter_names", PARAMETER_NAMES))
    kwargs = {}
    for name in cls_names:
        if name not in parameter_names:
            kwargs[name] = float(fixed.get(name, default_for_param(name)))
        elif name in priors_cfg:
            kwargs[name] = prior_from_cfg(name, priors_cfg[name])
        elif name in fixed:
            kwargs[name] = float(fixed[name])
        else:
            logger.warning(
                "no prior for %r: fixed at its default %s", name, default_for_param(name),
            )
            kwargs[name] = float(default_for_param(name))
    model = af.Model(parameter_cls, **kwargs)
    if model.prior_count == 0:
        raise ValueError("every parameter is fixed; nothing to fit")
    return model


def free_parameter_names(model: af.Model) -> list[str]:
    return [p[0] for p in model.paths] if model.paths and isinstance(model.paths[0], tuple) else list(model.parameter_names)


def describe_model(model: af.Model) -> dict:
    """The priors as they will be fitted, in JSON form."""
    out = {}
    names = tuple(getattr(model.cls, "parameter_names", PARAMETER_NAMES))
    for name in names:
        value = getattr(model, name)
        if isinstance(value, af.Prior):
            d = {"type": type(value).__name__.replace("Prior", "")}
            for attr in ("lower_limit", "upper_limit", "mean", "sigma"):
                v = getattr(value, attr, None)
                if v is not None and abs(float(v)) != float("inf"):
                    d[attr.replace("_limit", "")] = float(v)
            out[name] = d
        else:
            out[name] = {"type": "fixed", "value": float(value)}
    return out


def units_for(names) -> dict:
    return {n: parameter_unit(n) for n in names}
