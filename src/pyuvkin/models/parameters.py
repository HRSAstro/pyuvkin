"""The disc parameters every backend shares, and the class PyAutoFit fits.

One parameter set, one meaning, whatever renders the cube:

| name                  | unit    | meaning                                              |
|-----------------------|---------|------------------------------------------------------|
| centre_ra             | arcsec  | dRA cos(dec) of the disc centre, +east               |
| centre_dec            | arcsec  | dDec of the disc centre, +north                      |
| v_sys                 | km/s    | systemic velocity, relative to the spectral reference|
| intensity             | Jy km/s | velocity-integrated line flux (``cube.sum() * dv``)  |
| scale_radius          | arcsec  | exponential scale length of the surface brightness   |
| inclination           | deg     | 0 face-on, 90 edge-on                                |
| phi                   | deg     | PA of the receding major axis, east of north         |
| turnover_radius       | arcsec  | rotation-curve turnover radius                       |
| maximum_velocity      | km/s    | asymptotic rotation velocity                         |
| rotation_beta         | —       | Rix et al. (1997) inner-slope index β (``rix`` only) |
| rotation_xi           | —       | Rix et al. (1997) sharpness ξ (``rix`` only)         |
| velocity_dispersion   | km/s    | central / constant isotropic gas dispersion σ₀       |
| dispersion_scale_radius | arcsec | scale of radial σ(R) (exponential / linear)        |
| vrad                  | km/s    | in-plane radial velocity, +outwards (thindisk / BBarolo) |
| vmax_black_hole       | km/s    | Keplerian term ``v_bh / sqrt(r)`` (KinMS only)       |

Parametric rotation curves (``model.rotation_curve``): ``arctan``
(``v = (2 v_max / pi) arctan(r / r_t)``), ``tanh``, ``exponential``,
``isothermal``, and ``rix`` (Rix et al. 1997 multi-parameter; Rizzo+2021
eq. 2). Radial dispersion (``model.dispersion_curve``): ``constant``
(default), ``exponential`` (``σ = σ₀ exp(-R/R_σ)``; Rizzo+2021 eq. 4),
or ``linear`` (``σ = σ₀ max(1 - R/R_σ, 0)``).

With ``model.rotation_curve: "rings"`` (BBarolo only) the parametric
``turnover_radius`` / ``maximum_velocity`` are replaced by free per-ring
parameters (classic tilted-ring style). By default that is BBarolo's FREE
set — ``vrot``, ``velocity_dispersion``, ``inclination``, ``phi`` — one
value per ring; centres, ``v_sys`` and ``vrad`` stay shared (``vrad``
defaults to 0). Override with ``model.options.free`` (a list or
BBarolo-style string). Priors broadcast: one ``"vrot"`` /
``"inclination"`` / … entry applies to every ring.
"""

from __future__ import annotations

from functools import lru_cache

PARAMETER_UNITS = {
    "centre_ra": "arcsec",
    "centre_dec": "arcsec",
    "v_sys": "km/s",
    "intensity": "Jy km/s",
    "scale_radius": "arcsec",
    "inclination": "deg",
    "phi": "deg",
    "turnover_radius": "arcsec",
    "maximum_velocity": "km/s",
    "rotation_beta": "",
    "rotation_xi": "",
    "velocity_dispersion": "km/s",
    "dispersion_scale_radius": "arcsec",
    "vrad": "km/s",
    "vmax_black_hole": "km/s",
}

PARAMETER_NAMES = tuple(PARAMETER_UNITS)

#: Parameters that describe the surface brightness alone; not fitted when the
#: surface brightness is freeform (they are then fixed from the SB map).
SURFACE_BRIGHTNESS_PARAMETERS = ("intensity", "scale_radius")

#: Always shared in rings mode (SB normalisation / morphology).
RING_ALWAYS_SHARED = ("intensity", "scale_radius")

#: Can be free per ring (GalMod accepts arrays). Names match our parameters;
#: BBarolo GALFIT FREE uses VROT, VDISP, VRAD, INC, PA, XPOS, YPOS, VSYS, …
RING_FREEABLE = (
    "vrot", "velocity_dispersion", "vrad", "inclination", "phi",
    "centre_ra", "centre_dec", "v_sys",
)

#: BBarolo GALFIT default FREE = "VROT VDISP INC PA"
DEFAULT_FREE_RING_PARAMS = ("vrot", "velocity_dispersion", "inclination", "phi")

_BBAROLO_FREE_ALIASES = {
    "vrot": "vrot", "vdisp": "velocity_dispersion", "inc": "inclination",
    "pa": "phi", "phi": "phi", "xpos": "centre_ra", "ypos": "centre_dec",
    "vsys": "v_sys", "velocity_dispersion": "velocity_dispersion",
    "inclination": "inclination", "centre_ra": "centre_ra",
    "centre_dec": "centre_dec", "v_sys": "v_sys",
    "vrad": "vrad", "v_rad": "vrad",
}

PARAMETRIC_ROTATION_CURVES = ("arctan", "tanh", "exponential", "isothermal", "rix")
ROTATION_CURVES = PARAMETRIC_ROTATION_CURVES + ("rings",)
#: Aliases accepted in settings → canonical name in PARAMETRIC_ROTATION_CURVES.
ROTATION_CURVE_ALIASES = {
    "multi": "rix", "rix97": "rix", "rix1997": "rix",
}

DISPERSION_CURVES = ("constant", "exponential", "linear")
DEFAULT_DISPERSION_CURVE = "constant"

DEFAULT_VROT_KMS = 200.0
DEFAULT_VRAD_KMS = 0.0

DEFAULTS = {
    "centre_ra": 0.0,
    "centre_dec": 0.0,
    "v_sys": 0.0,
    "intensity": 1.0,
    "scale_radius": 0.3,
    "inclination": 45.0,
    "phi": 0.0,
    "turnover_radius": 0.1,
    "maximum_velocity": 200.0,
    "rotation_beta": 0.0,
    "rotation_xi": 3.0,
    "velocity_dispersion": 30.0,
    "dispersion_scale_radius": 1.0,
    "vrad": DEFAULT_VRAD_KMS,
    "vmax_black_hole": 0.0,
}

RING_PARAM_DEFAULTS = {
    "vrot": DEFAULT_VROT_KMS,
    "velocity_dispersion": DEFAULTS["velocity_dispersion"],
    "vrad": DEFAULT_VRAD_KMS,
    "inclination": DEFAULTS["inclination"],
    "phi": DEFAULTS["phi"],
    "centre_ra": DEFAULTS["centre_ra"],
    "centre_dec": DEFAULTS["centre_dec"],
    "v_sys": DEFAULTS["v_sys"],
}

RING_PARAM_UNITS = {
    "vrot": "km/s",
    "velocity_dispersion": "km/s",
    "vrad": "km/s",
    "inclination": "deg",
    "phi": "deg",
    "centre_ra": "arcsec",
    "centre_dec": "arcsec",
    "v_sys": "km/s",
}


def parse_free_ring_params(free) -> tuple[str, ...]:
    """Parse ``options.free``: list, or BBarolo-style ``\"VROT VDISP INC PA\"``."""
    if free is None:
        return DEFAULT_FREE_RING_PARAMS
    if isinstance(free, str):
        tokens = free.replace(",", " ").split()
    else:
        tokens = list(free)
    out = []
    for tok in tokens:
        key = str(tok).strip().lower().replace("-", "_")
        if key not in _BBAROLO_FREE_ALIASES:
            raise ValueError(
                f"unknown free ring parameter {tok!r}; "
                f"choose from {list(RING_FREEABLE)} "
                f"(or BBarolo names VROT VDISP VRAD INC PA XPOS YPOS VSYS)"
            )
        base = _BBAROLO_FREE_ALIASES[key]
        if base not in out:
            out.append(base)
    if "vrot" not in out:
        out.insert(0, "vrot")
    return tuple(out)


def ring_param_name(base: str, i: int) -> str:
    return f"{base}_{int(i)}"


def parse_ring_param_name(name: str) -> tuple[str, int] | None:
    """``('vrot', 3)`` for ``vrot_3``, else None."""
    if "_" not in name:
        return None
    base, _, suffix = name.rpartition("_")
    if base not in RING_FREEABLE or not suffix.isdigit():
        return None
    return base, int(suffix)


def is_ring_param_name(name: str) -> bool:
    return parse_ring_param_name(name) is not None


def is_vrot_name(name: str) -> bool:
    parsed = parse_ring_param_name(name)
    return parsed is not None and parsed[0] == "vrot"


def parameter_unit(name: str) -> str:
    parsed = parse_ring_param_name(name)
    if parsed is not None:
        return RING_PARAM_UNITS[parsed[0]]
    if name in PARAMETER_UNITS:
        return PARAMETER_UNITS[name]
    if name in RING_PARAM_UNITS:
        return RING_PARAM_UNITS[name]
    raise KeyError(name)


def ring_parameter_names(n_rings: int, free=None) -> tuple[str, ...]:
    """Parameter names for rings mode given which bases are free per ring."""
    n = int(n_rings)
    if n < 1:
        raise ValueError("n_rings must be >= 1")
    free = parse_free_ring_params(free)
    shared = list(RING_ALWAYS_SHARED)
    for base in ("centre_ra", "centre_dec", "v_sys", "inclination", "phi",
                 "velocity_dispersion", "vrad"):
        if base not in free:
            shared.append(base)
    names = tuple(shared)
    for base in free:
        names += tuple(ring_param_name(base, i) for i in range(n))
    return names


def ring_values(p, base: str, n_rings: int | None = None):
    """Per-ring array for ``base``, or a length-``n`` fill from the shared attr."""
    import numpy as np

    n = int(n_rings if n_rings is not None else getattr(p, "n_rings"))
    if hasattr(p, ring_param_name(base, 0)):
        return np.asarray(
            [float(getattr(p, ring_param_name(base, i))) for i in range(n)],
            dtype=float,
        )
    return np.full(n, float(getattr(p, base)), dtype=float)


def ring_mean(p, base: str, n_rings: int | None = None) -> float:
    """Mean of a (possibly per-ring) parameter — for plot slits etc."""
    import numpy as np

    return float(np.mean(ring_values(p, base, n_rings)))


def ring_vrot_kms(p, n_rings: int | None = None):
    """``vrot_i`` array on a tilted-ring instance."""
    return ring_values(p, "vrot", n_rings)


def default_for_param(name: str) -> float:
    parsed = parse_ring_param_name(name)
    if parsed is not None:
        return float(RING_PARAM_DEFAULTS[parsed[0]])
    if name in RING_PARAM_DEFAULTS:
        return float(RING_PARAM_DEFAULTS[name])
    return float(DEFAULTS[name])


class DiscParameters:
    """A plain container -- what `af.Model` instantiates for each trial.

    Deliberately holds no reference to any backend, grid or data, so autofit
    can pickle instances and so that the same class serves every backend.
    """

    parameter_names = PARAMETER_NAMES

    def __init__(
        self,
        centre_ra: float = 0.0,
        centre_dec: float = 0.0,
        v_sys: float = 0.0,
        intensity: float = 1.0,
        scale_radius: float = 0.3,
        inclination: float = 45.0,
        phi: float = 0.0,
        turnover_radius: float = 0.1,
        maximum_velocity: float = 200.0,
        rotation_beta: float = 0.0,
        rotation_xi: float = 3.0,
        velocity_dispersion: float = 30.0,
        dispersion_scale_radius: float = 1.0,
        vrad: float = 0.0,
        vmax_black_hole: float = 0.0,
    ):
        self.centre_ra = centre_ra
        self.centre_dec = centre_dec
        self.v_sys = v_sys
        self.intensity = intensity
        self.scale_radius = scale_radius
        self.inclination = inclination
        self.phi = phi
        self.turnover_radius = turnover_radius
        self.maximum_velocity = maximum_velocity
        self.rotation_beta = rotation_beta
        self.rotation_xi = rotation_xi
        self.velocity_dispersion = velocity_dispersion
        self.dispersion_scale_radius = dispersion_scale_radius
        self.vrad = vrad
        self.vmax_black_hole = vmax_black_hole

    @classmethod
    def from_dict(cls, values: dict) -> "DiscParameters":
        unknown = set(values) - set(PARAMETER_NAMES)
        if unknown:
            raise ValueError(f"unknown parameter(s): {sorted(unknown)}")
        return cls(**{**DEFAULTS, **values})

    def as_dict(self) -> dict:
        return {name: float(getattr(self, name)) for name in PARAMETER_NAMES}

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v:.6g}" for k, v in self.as_dict().items())
        return f"DiscParameters({inner})"


def _tilted_ring_from_dict(cls, values: dict):
    kwargs = {}
    broadcast_keys = set(RING_FREEABLE) | {f"{b}_i" for b in RING_FREEABLE}
    for name in cls.parameter_names:
        if name in values:
            kwargs[name] = float(values[name])
            continue
        parsed = parse_ring_param_name(name)
        if parsed is not None:
            base, _ = parsed
            if base in values:
                kwargs[name] = float(values[base])
            elif f"{base}_i" in values:
                kwargs[name] = float(values[f"{base}_i"])
            else:
                kwargs[name] = float(RING_PARAM_DEFAULTS[base])
        else:
            kwargs[name] = float(DEFAULTS[name])
    unknown = set(values) - set(cls.parameter_names) - broadcast_keys
    if unknown:
        raise ValueError(f"unknown parameter(s): {sorted(unknown)}")
    return cls(**kwargs)


def _tilted_ring_as_dict(self) -> dict:
    return {n: float(getattr(self, n)) for n in self.parameter_names}


def _tilted_ring_repr(self) -> str:
    inner = ", ".join(f"{k}={v:.6g}" for k, v in self.as_dict().items())
    return f"TiltedRingParameters({inner})"


@lru_cache(maxsize=64)
def make_tilted_ring_class(n_rings: int, free: tuple[str, ...] = DEFAULT_FREE_RING_PARAMS):
    """DiscParameters-like class with free per-ring parameters.

    ``free`` is a tuple of bases (see ``DEFAULT_FREE_RING_PARAMS``). Built
    dynamically so PyAutoFit can inspect a fixed ``__init__`` signature.
    """
    n_rings = int(n_rings)
    free = parse_free_ring_params(free)
    names = ring_parameter_names(n_rings, free)
    args = []
    for name in names:
        args.append(f"{name}: float = {default_for_param(name)!r}")
    assigns = "\n".join(f"        self.{name} = {name}" for name in names)
    code = (
        f"def __init__(self, {', '.join(args)}):\n"
        f"{assigns}\n"
    )
    ns: dict = {}
    exec(code, ns)  # noqa: S102 — fixed template; n_rings / free vary
    tag = "_".join(free)
    return type(
        f"TiltedRingParameters_{n_rings}_{tag}",
        (),
        {
            "n_rings": n_rings,
            "free_ring_params": free,
            "parameter_names": names,
            "__init__": ns["__init__"],
            "from_dict": classmethod(_tilted_ring_from_dict),
            "as_dict": _tilted_ring_as_dict,
            "__repr__": _tilted_ring_repr,
        },
    )


def normalize_rotation_curve(kind: str) -> str:
    """Canonical ``rotation_curve`` name (resolves aliases; keeps ``rings``)."""
    k = str(kind).strip().lower()
    return ROTATION_CURVE_ALIASES.get(k, k)


def normalize_dispersion_curve(kind: str | None) -> str:
    if kind is None:
        return DEFAULT_DISPERSION_CURVE
    k = str(kind).strip().lower()
    if k not in DISPERSION_CURVES:
        raise ValueError(
            f"unknown dispersion_curve {kind!r}; choose from {DISPERSION_CURVES}"
        )
    return k


def rotation_curve_kms(r_arcsec, p: DiscParameters, kind: str = "arctan"):
    """Circular velocity at radius ``r`` for the shared parameterisation.

    ``rix`` is the Rix et al. (1997) multi-parameter form used in Rizzo+2021
    (their eq. 2)::

        V(R) = V_t (1 + R_t/R)^β / [1 + (R_t/R)^ξ]^(1/ξ)

    with ``maximum_velocity`` → V_t, ``turnover_radius`` → R_t,
    ``rotation_beta`` → β, ``rotation_xi`` → ξ.
    """
    import numpy as np

    kind = normalize_rotation_curve(kind)
    if kind == "rings":
        raise ValueError(
            "rotation_curve 'rings' has no closed form; use ring_vrot_kms "
            "on a tilted-ring instance"
        )
    r = np.asarray(r_arcsec, dtype=float)
    rt = max(float(p.turnover_radius), 1e-6)
    x = r / rt
    vmax = float(p.maximum_velocity)
    if kind == "arctan":
        v = (2.0 * vmax / np.pi) * np.arctan(x)
    elif kind == "tanh":
        v = vmax * np.tanh(x)
    elif kind == "exponential":
        v = vmax * (1.0 - np.exp(-x))
    elif kind == "isothermal":
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(
                x > 0,
                vmax * np.sqrt(np.clip(
                    1.0 - np.arctan(x) / np.where(x > 0, x, 1.0), 0, None,
                )),
                0.0,
            )
    elif kind == "rix":
        v = _rix_rotation_kms(r, rt, vmax, p)
    else:
        raise ValueError(f"unknown rotation curve {kind!r}")
    vbh = float(getattr(p, "vmax_black_hole", 0.0) or 0.0)
    if vbh > 0:
        with np.errstate(divide="ignore"):
            v = np.hypot(v, np.where(r > 0, vbh / np.sqrt(np.where(r > 0, r, 1.0)), 0.0))
    return v


def _rix_rotation_kms(r, rt: float, vmax: float, p: DiscParameters):
    """Rix et al. (1997) / Rizzo+2021 eq. (2)."""
    import numpy as np

    beta = float(getattr(p, "rotation_beta", 0.0) or 0.0)
    xi = max(float(getattr(p, "rotation_xi", 3.0) or 3.0), 1e-6)
    r = np.asarray(r, dtype=float)
    # avoid R=0 singularity; use a tiny floor relative to R_t
    r_safe = np.maximum(r, 1e-8 * rt)
    inv = rt / r_safe
    with np.errstate(over="ignore", invalid="ignore"):
        num = np.power(1.0 + inv, beta)
        den = np.power(1.0 + np.power(inv, xi), 1.0 / xi)
        v = vmax * num / den
    v = np.asarray(v, dtype=float)
    v[~np.isfinite(v)] = 0.0
    # exact centre: solid-body-like limit → 0 for β < 1 (usual)
    return np.where(r > 0, v, 0.0)


def dispersion_curve_kms(r_arcsec, p: DiscParameters, kind: str = "constant"):
    """Isotropic gas dispersion at radius ``r`` (km/s).

    ``constant``
        ``σ = velocity_dispersion`` (σ₀).
    ``exponential``
        ``σ(R) = σ₀ exp(-R / R_σ)`` (Rizzo+2021 eq. 4).
    ``linear``
        ``σ(R) = σ₀ max(1 - R / R_σ, 0)``, floored at 1e-3 km/s.

    ``dispersion_scale_radius`` is R_σ (arcsec).
    """
    import numpy as np

    kind = normalize_dispersion_curve(kind)
    sigma0 = max(float(p.velocity_dispersion), 0.0)
    r = np.asarray(r_arcsec, dtype=float)
    if kind == "constant":
        return np.full(r.shape, max(sigma0, 1e-3), dtype=float) if r.shape else max(sigma0, 1e-3)
    rs = max(float(getattr(p, "dispersion_scale_radius", 1.0) or 1.0), 1e-6)
    if kind == "exponential":
        sig = sigma0 * np.exp(-r / rs)
    elif kind == "linear":
        sig = sigma0 * np.maximum(1.0 - r / rs, 0.0)
    else:
        raise ValueError(f"unknown dispersion_curve {kind!r}")
    return np.maximum(sig, 1e-3)
