"""Settings: one JSON file describes a fit.

``pyuvkin fit settings.json`` (or ``pyuvkin.run("settings.json")``). Every key
has a default except ``dataset`` and ``fov``; unknown keys are refused so a
misspelt option cannot be silently ignored. `docs/fit-config-template.json`
lists every key at its default, and `DEFAULTS` below is that template.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    # ------------------------------------------------------------ the data
    "dataset": None,           # pyuvimage dataset directory or casa_export .npz
    "out": "pyuvkin_out",
    "spw": None,               # which spectral window holds the line (multi-spw data)
    "channels": None,          # [start, stop) channel window, or null for all
    "velocity_window_kms": None,  # [v_lo, v_hi] instead of channels (needs spectral ref)
    "noise": "keep",           # keep | difference | hybrid | scaled (pyuvimage estimators)
    "spectral": {
        "rest_frequency_ghz": None,   # line rest frequency ...
        "redshift": None,             # ... and redshift set the v = 0 reference
        "reference_frequency_ghz": None,  # or give the observed reference directly
        "frame": None,   # spectral frame of the dataset's frequencies for the FITS SPECSYS
                         # (TOPOCENT, LSRK, BARYCENT, ...); pyuvimage exports do not record it
    },
    # ------------------------------------------------------- the geometry
    "fov": None,               # arcsec, full width; must cover all the emission
    "pixel_scale": "auto",     # auto | nyquist | arcsec  (pyuvimage's rules)
    "oversample": 2,           # image grid this much finer than the pixel_scale mesh
    "n_pixels": None,          # or fix the pixel count outright
    "render_oversample": 1,    # render the model this much finer again, then block-sum
    "image_centre": [0.0, 0.0],  # [dRA, dDec] arcsec of the image centre, +east +north
    "transformer": "auto",     # auto | dft | nufft | pynufft
    "primary_beam": {
        "enabled": True,
        "dish_diameter_m": None,   # default: the value stored at import
        "pb_factor": 1.13,
    },
    # ---------------------------------------------------------- the model
    "model": {
        "backend": "thindisk",     # thindisk | kinms | galpak | bbarolo
        "rotation_curve": "arctan",
        # backend options, e.g. kinms: n_samples, scale_height_arcsec, seed,
        # clouds_per_pixel; galpak: thickness_profile, dispersion_profile;
        # bbarolo: n_rings, rmax (arcsec outer radius; preferred),
        # n_scale_lengths ("auto"|float × scale_radius; legacy),
        # scale_height_arcsec, free (may include vrad), twostage, regtype,
        # adrift, adrift_pol1, adrift_pol2,
        # linear / hanning (instrumental spectral σ in channels; off by default)
        "options": {},
    },
    "surface_brightness": {
        "type": "analytic",        # analytic | freeform
        # freeform: where the map comes from
        "map_fits": None,          # a Jy/pixel/channel or Jy km/s map (pyuvimage model_pbcor.fits), or null
        "uncertainty_fits": None,  # its 1-sigma map (pyuvimage uncertainty.fits); needed for the S/N mask
        "map_units": "jy_per_pixel_per_channel",  # or jy_kms_per_pixel
        "snr_threshold": 0.5,      # mask map pixels below this many sigma
        "pyuvimage": {},           # extra keyword arguments to pyuvimage.run for the reconstruction
        # the map is built from the same data the kinematics are fitted to, so
        # the kinematic errors leave out the map's uncertainty; this many
        # perturbed maps are re-fitted (L-BFGS from the best fit) to measure it
        "morphology_samples": 0,
    },
    "priors": {},
    "truth": None,                 # optional true parameters (mocks); recorded and compared
    # --------------------------------------------------------- the search
    "search": {
        "method": "nautilus",
        "number_of_cores": 1,
    },
    # -------------------------------------------------------- the outputs
    "write_cubes": True,
    "write_plots": True,
    "n_plot_channels": 12,
}

_REQUIRED = ("dataset", "fov")


def _merge(defaults: dict, given: dict, path: str = "") -> dict:
    out = copy.deepcopy(defaults)
    for key, value in given.items():
        if key.startswith("_"):
            continue
        if key not in defaults:
            raise ValueError(
                f"unknown setting {path + key!r}. Accepted here: {sorted(defaults)}"
            )
        # free-form blocks are taken as given
        if key in ("priors", "options", "pyuvimage", "truth", "search"):
            out[key] = copy.deepcopy(value)
        elif isinstance(defaults[key], dict) and isinstance(value, dict):
            out[key] = _merge(defaults[key], value, path + key + ".")
        elif key == "dataset":
            out[key] = value          # may be an in-memory UVData; never copied
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_settings(path_or_dict, base_dir: str | Path | None = None, **overrides) -> dict:
    """Read a settings file (or take a dict), apply defaults and overrides.
    Relative paths resolve against the file's directory (or ``base_dir``)."""
    if isinstance(path_or_dict, (str, Path)):
        path = Path(path_or_dict)
        try:
            given = json.loads(path.read_text())
        except FileNotFoundError:
            raise SystemExit(f"no such settings file {path}")
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path} is not valid JSON ({exc})")
        base_dir = path.parent
    else:
        given = dict(path_or_dict)
        # a dict that came from an earlier load_settings keeps its base_dir
        base_dir = Path(base_dir or given.get("_base_dir") or Path.cwd())
    if not isinstance(given, dict):
        raise SystemExit("the settings file must contain a JSON object")
    given.update({k: v for k, v in overrides.items() if v is not None})
    settings = _merge(DEFAULTS, given)
    settings["_base_dir"] = str(base_dir)
    validate_settings(settings)
    return settings


def validate_settings(s: dict) -> None:
    for key in _REQUIRED:
        if s.get(key) is None:
            raise SystemExit(f"setting {key!r} is required")
    if float(s["fov"]) <= 0:
        raise ValueError("fov must be positive")
    if s["surface_brightness"]["type"] not in ("analytic", "freeform"):
        raise ValueError("surface_brightness.type must be 'analytic' or 'freeform'")
    from .models import BACKENDS

    if s["model"]["backend"] not in BACKENDS:
        raise ValueError(f"model.backend must be one of {BACKENDS}")
    from .models.parameters import ROTATION_CURVES

    rc = s["model"]["rotation_curve"]
    if rc not in ROTATION_CURVES:
        raise ValueError(f"model.rotation_curve must be one of {ROTATION_CURVES}")
    if rc == "rings":
        if s["model"]["backend"] != "bbarolo":
            raise ValueError(
                "rotation_curve 'rings' (free tilted-ring VROT) is only "
                "available for the bbarolo backend"
            )
        n_rings = (s["model"].get("options") or {}).get("n_rings")
        if n_rings is None or int(n_rings) < 1:
            raise ValueError(
                "rotation_curve 'rings' requires model.options.n_rings >= 1"
            )
    from .search import method_of

    method_of(s["search"])
    if s["channels"] is not None and s["velocity_window_kms"] is not None:
        raise ValueError("give channels or velocity_window_kms, not both")
    ic = s["image_centre"]
    if not (isinstance(ic, (list, tuple)) and len(ic) == 2):
        raise ValueError("image_centre must be [dRA, dDec] in arcsec")


def resolve_path(settings: dict, value) -> Path:
    """Paths in a settings file are relative to the file."""
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = Path(settings.get("_base_dir", ".")) / p
    return p


def public(settings: dict) -> dict:
    """The settings without private bookkeeping keys."""
    return {k: v for k, v in settings.items() if not k.startswith("_")}


def write_template(path: str | Path) -> None:
    Path(path).write_text(json.dumps(public(DEFAULTS), indent=2) + "\n")
