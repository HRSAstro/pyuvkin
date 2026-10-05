"""Mocks whose surface brightness is more than an exponential disc.

The morphology is two concentric Sersic components in the plane of the disc --
a compact one and an extended one -- with the extended component modulated by
two trailing logarithmic spiral arms. None of that shape is available to
pyuvkin's analytic surface brightness, so a fit with
``surface_brightness.type: "analytic"`` is deliberately mis-specified while one
with ``"freeform"`` can follow the map. Two mocks are built from the same
morphology, the same uv coverage and the same noise realisation:

    arms_sb       the arms are in the surface brightness only; the velocity
                  field is exactly `thindisk`'s circular field, so any bias in
                  the recovered kinematics comes from the surface brightness
    arms_inflow   same spiral morphology, plus a constant axisymmetric radial
                  inflow (``DiscParameters.vrad``; +outwards). This matches the
                  toy model used on real IFU data (e.g. Price et al. 2021,
                  ApJ 922, 143: constant ``v_r ~ 90`` km/s on top of rotation)
                  and is what `thindisk` can actually fit

**Velocity conventions.** ``theta`` is the disc-plane azimuth measured from the
receding major axis, as `conventions.disc_coordinates` returns it, and rotation
is towards increasing ``theta``. The line-of-sight velocity (positive =
receding) of in-plane motion ``(v_R outwards, v_T along the rotation)`` is

    v_los = v_sys + (v_T cos(theta) + v_R sin(theta)) sin(i)

which reduces to `thindisk`'s ``v_sys + v_c sin(i) cos(theta)`` when the flow
is circular. The near side is at ``theta = -90 deg``, so inflow
(``v_R < 0``) is redshifted there. The arms are wound to trail the rotation,
which is the winding an observer would use to infer the near side.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.special import gammaincinv

from pyuvimage.mock import random_uv_coverage

from . import conventions
from .grids import block_sum, resolve_cube_geometry
from .mock import DEFAULT_META, channel_frequencies
from .models import PARAMETER_NAMES, DiscParameters, FreeformSB
from .models.parameters import rotation_curve_kms
from .models.thindisk import channel_fractions
from .spectral import spectral_axis
from .transform import CubeTransformer
from .uvdata import UVData

#: Disc parameters the structured mocks are built on. Close to
#: `mock.DEFAULT_TRUTH`, so the demo priors cover them, but with a larger disc
#: so that the arms span several beams.
DEFAULT_TRUTH = dict(
    # `scale_radius` is not used by the renderer -- the morphology comes from
    # `SpiralStructure` -- it is only the nominal exponential scale an
    # analytic fit should land near, measured from the rendered map.
    centre_ra=0.05, centre_dec=-0.08, v_sys=15.0, intensity=3.0, scale_radius=0.511,
    inclination=55.0, phi=40.0, turnover_radius=0.25, maximum_velocity=250.0,
    velocity_dispersion=35.0, vrad=0.0, vmax_black_hole=0.0,
)

#: Constant axisymmetric radial velocity (+outwards, km/s) for ``arms_inflow``.
#: Magnitude matches the Price et al. (2021) toy inflow; sign is inflow.
INFLOW_VRAD_KMS = -90.0


@dataclass(frozen=True)
class SpiralStructure:
    """Two Sersic components plus m=2 trailing arms, all in the disc plane.

    Radii are arcsec in the plane of the disc. ``bulge_flux_fraction`` splits
    the total flux between the two components and is held exactly, so turning
    the arms on redistributes flux within the extended component rather than
    changing the bulge-to-total ratio. Kinematics are not stored here -- the
    ``arms_inflow`` mock sets ``DiscParameters.vrad`` instead.
    """

    #: a compact classical bulge and a shallow extended disc. The arms sit at
    #: 0.35-1.15", i.e. 2-6 synthesised beams out, which is where the beam
    #: leaves them with enough arm/interarm contrast to matter.
    bulge_flux_fraction: float = 0.15
    bulge_r_eff: float = 0.24
    bulge_index: float = 2.0
    disc_r_eff: float = 0.70
    disc_index: float = 0.5

    n_arms: int = 2
    arm_contrast: float = 6.0
    pitch_deg: float = 30.0
    arm_sharpness: float = 2.0
    arm_phase_deg: float = 0.0
    arm_r_ref: float = 0.60
    arm_r_in: float = 0.35
    arm_w_in: float = 0.12
    arm_r_out: float = 1.15
    arm_w_out: float = 0.18

    #: the disc is tapered well inside the field, so the mock has no flux at
    #: the edge of the image grid to alias back in
    truncation_radius: float = 1.30
    truncation_width: float = 0.10


def sersic_b(index: float) -> float:
    """``b_n``, the constant that makes ``r_eff`` the half-light radius.

    Solves ``gamma(2n, b) / Gamma(2n) = 1/2`` directly. The usual asymptotic
    expansion would do, but it drifts below n ~ 1 and the disc component here
    is n = 0.5.
    """
    return float(gammaincinv(2.0 * float(index), 0.5))


def sersic_profile(radius, r_eff: float, index: float) -> np.ndarray:
    """Sersic surface brightness, 1 at ``r_eff``; the caller normalises."""
    x = np.maximum(np.asarray(radius, float), 0.0) / max(float(r_eff), 1e-6)
    return np.exp(-sersic_b(index) * (x ** (1.0 / float(index)) - 1.0))


def spiral_arm_profile(radius, theta, s: SpiralStructure) -> np.ndarray:
    """The arm ridge in ``[0, 1]``: 1 on the crest, near 0 between the arms.

    A logarithmic spiral windowed in radius. The phase runs with ``-log(R)``,
    so the crest moves towards decreasing ``theta`` as the radius grows: the
    arms trail the rotation. ``arm_sharpness`` sets how narrow the crest is,
    and with it the interarm floor, ``exp(-2 * arm_sharpness)``.
    """
    r = np.maximum(np.asarray(radius, float), 1e-4)
    phase = -np.log(r / s.arm_r_ref) / np.tan(np.radians(s.pitch_deg))
    crest = np.cos(s.n_arms * (np.asarray(theta, float) - np.radians(s.arm_phase_deg) - phase))
    ridge = np.exp(s.arm_sharpness * (crest - 1.0))
    window = (0.5 * (1.0 + np.tanh((r - s.arm_r_in) / s.arm_w_in))
              * 0.5 * (1.0 - np.tanh((r - s.arm_r_out) / s.arm_w_out)))
    return ridge * window


def structured_surface_brightness(radius, arm, intensity: float,
                                  s: SpiralStructure) -> np.ndarray:
    """Jy km/s per pixel, on whatever grid ``radius`` is given on."""
    taper = 0.5 * (1.0 - np.tanh((np.asarray(radius, float) - s.truncation_radius)
                                 / s.truncation_width))
    bulge = sersic_profile(radius, s.bulge_r_eff, s.bulge_index) * taper
    disc = sersic_profile(radius, s.disc_r_eff, s.disc_index) * taper * (1.0 + s.arm_contrast * arm)

    f = float(np.clip(s.bulge_flux_fraction, 0.0, 1.0))
    out = np.zeros_like(bulge)
    for component, fraction in ((bulge, f), (disc, 1.0 - f)):
        total = component.sum()
        if total > 0 and fraction > 0:
            out = out + component * (fraction * float(intensity) / total)
    return out


def line_of_sight_velocity(radius, cos_theta, sin_theta, arm, p: DiscParameters,
                           s: SpiralStructure, rotation_curve: str = "arctan") -> np.ndarray:
    """``v_sys + (v_c cos(theta) + vrad sin(theta)) sin(i)`` -- exactly
    `thindisk`, including a constant axisymmetric ``vrad`` (+outwards).

    ``arm`` and ``s`` are unused for kinematics (arms are morphology only);
    they remain in the signature so call sites stay uniform.
    """
    del arm, s  # morphology only
    v_c = rotation_curve_kms(radius, p, rotation_curve)
    v_r = float(getattr(p, "vrad", 0.0) or 0.0)
    sin_i = np.sin(np.radians(float(p.inclination)))
    return float(p.v_sys) + (v_c * cos_theta + v_r * sin_theta) * sin_i


def render_structured_cube(geometry, spectral, p: DiscParameters, s: SpiralStructure,
                           rotation_curve: str = "arctan") -> tuple[np.ndarray, np.ndarray]:
    """The cube (Jy/pixel per channel, image grid, data channel order) and the
    velocity-integrated surface brightness on the same grid."""
    yy, xx = geometry.coordinates(geometry.render_oversample)
    radius, cos_t, sin_t = conventions.disc_coordinates(
        yy, xx, p.centre_ra, p.centre_dec, p.phi, p.inclination,
    )
    theta = np.arctan2(sin_t, cos_t)
    arm = spiral_arm_profile(radius, theta, s)
    sb = structured_surface_brightness(radius, arm, p.intensity, s)
    v_los = line_of_sight_velocity(radius, cos_t, sin_t, arm, p, s, rotation_curve)

    v = spectral.model_velocities_kms
    v_edges = np.concatenate([[v[0] - 0.5 * spectral.dv_kms], v + 0.5 * spectral.dv_kms])
    frac = channel_fractions(v_edges, v_los, p.velocity_dispersion)

    cube = frac * (sb / spectral.dv_kms)[None]
    k = geometry.render_oversample
    return spectral.to_data_order(block_sum(cube, k)), block_sum(sb, k)


def _profile_summary(geometry, p: DiscParameters, sb_map: np.ndarray) -> dict:
    """What a one-component fit is really aiming at: the half-light radius of
    the map and the exponential scale length of its azimuthal average."""
    yy, xx = geometry.coordinates(1)
    radius, _, _ = conventions.disc_coordinates(
        yy, xx, p.centre_ra, p.centre_dec, p.phi, p.inclination,
    )
    r, w = radius.ravel(), sb_map.ravel()
    order = np.argsort(r)
    cumulative = np.cumsum(w[order])
    r_half = float(np.interp(0.5, cumulative / cumulative[-1], r[order]))
    # ln I is linear in R for an exponential, so a flux-weighted straight-line
    # fit over the bright part of the disc gives the scale length
    fit = (w > 0) & (radius.ravel() < 1.25)
    slope = np.polyfit(r[fit], np.log(w[fit]), 1, w=np.sqrt(w[fit]))[0]
    return {
        "half_light_radius": r_half,
        "exponential_scale_length": float(-1.0 / slope) if slope < 0 else float("nan"),
        "scale_radius_note": (
            "`scale_radius` is nominal: this morphology is not a single "
            "exponential, so no value is 'true'. Compare an analytic fit "
            "against exponential_scale_length."
        ),
    }


def simulate_structured_disc(
    parameters: DiscParameters | dict | None = None,
    structure: SpiralStructure | None = None,
    *,
    fov: float = 3.0,
    n_vis: int = 2000,
    max_baseline_m: float = 1500.0,
    reference_frequency_ghz: float = 230.0,
    n_chan: int = 16,
    dv_kms: float = 40.0,
    sigma_jy: float = 6e-4,
    pixel_scale: float | str = "auto",
    render_oversample: int = 2,
    seed: int = 0,
    meta: dict | None = None,
) -> tuple[UVData, dict, FreeformSB]:
    """Noisy visibilities of a structured disc, with its truth map alongside.

    ``seed`` fixes the uv coverage and the noise, so two calls differing only
    in ``structure`` give datasets whose difference is purely the model.
    """
    structure = structure or SpiralStructure()
    if parameters is None:
        parameters = DiscParameters.from_dict(DEFAULT_TRUTH)
    elif isinstance(parameters, dict):
        parameters = DiscParameters.from_dict({**DEFAULT_TRUTH, **parameters})

    ref_hz = reference_frequency_ghz * 1e9
    freqs = channel_frequencies(ref_hz, n_chan, dv_kms)
    uvw = random_uv_coverage(n_vis, max_baseline_m, ref_hz, seed=seed)
    meta = {**DEFAULT_META, **(meta or {})}
    placeholder = UVData(
        uvw=uvw, frequencies=freqs,
        data=np.zeros((n_chan, n_vis), complex),
        noise=np.full((n_chan, n_vis), sigma_jy + 1j * sigma_jy),
        meta=meta,
    )
    spectral = spectral_axis(freqs, reference_frequency_ghz=reference_frequency_ghz)
    geometry = resolve_cube_geometry(
        fov, placeholder.max_baseline_wavelengths,
        placeholder.baseline_percentile_wavelengths(95), pixel_scale=pixel_scale,
        oversample=2, render_oversample=render_oversample,
    )
    cube, sb_map = render_structured_cube(geometry, spectral, parameters, structure)
    transformer = CubeTransformer(placeholder, geometry, "dft")
    vis = transformer.model_visibilities(cube)

    rng = np.random.default_rng(seed + 1)
    data = np.zeros((n_chan, n_vis), complex)
    for c, sl in enumerate(transformer.slices):
        data[c] = vis[sl] + rng.normal(0, sigma_jy, n_vis) + 1j * rng.normal(0, sigma_jy, n_vis)
    uvd = UVData(uvw=uvw, frequencies=freqs, data=data, noise=placeholder.noise.copy(), meta=meta)

    truth = {
        **parameters.as_dict(),
        "reference_frequency_ghz": reference_frequency_ghz,
        "sigma_jy": sigma_jy,
        "backend": "structured (2 Sersic + spiral arms)",
        "fov": fov,
        "pixel_scale": geometry.pixel_scale,
        "structure": asdict(structure),
        # This morphology has no single exponential scale length, so the
        # `scale_radius` above is only nominal: the one an analytic fit is
        # aiming at. Record what the map actually is, for comparison.
        **_profile_summary(geometry, parameters, sb_map),
    }
    sb = FreeformSB(map_jykms=sb_map, pixel_scale=geometry.pixel_scale,
                    source="structured mock truth")
    return uvd, truth, sb


def write_truth_sb_fits(path: Path, sb: FreeformSB, meta: dict) -> Path:
    """The truth surface brightness as a FITS map, in pyuvimage's orientation."""
    from astropy.io import fits
    from pyuvimage.products import build_header, to_fits_orientation

    header = build_header(
        n_pix=sb.shape[0], pixel_scale_arcsec=sb.pixel_scale, meta=meta,
        bunit="Jy km/s per pixel",
    )
    fits.writeto(path, to_fits_orientation(sb.map_jykms).astype(np.float32), header,
                 overwrite=True)
    return path


def fit_settings(truth: dict, sb_type: str, method: str = "lbfgs",
                 backend: str = "thindisk") -> dict:
    """Settings for one fit to a structured mock, relative to its directory.

    ``sb_type`` is ``analytic``, ``freeform``, or ``freeform_matern`` (Matérn
    pyuvimage regularisation). Mocks with non-zero truth ``vrad`` free an
    axisymmetric ``vrad`` (+outwards); morphology-only mocks leave it fixed at 0.
    """
    from .mock import demo_settings

    settings = demo_settings("dataset", truth, f"fit_{sb_type}", method=method)
    settings["model"]["backend"] = backend
    if sb_type.startswith("freeform"):
        settings["surface_brightness"] = {"type": "freeform"}
        if sb_type == "freeform_matern":
            settings["surface_brightness"]["pyuvimage"] = {"reg": "matern"}
        # the reconstructed map fixes the morphology and the total flux
        for name in ("intensity", "scale_radius"):
            settings["priors"].pop(name, None)
    else:
        settings["surface_brightness"] = {"type": "analytic"}
    vrad = float(truth.get("vrad") or 0.0)
    if vrad != 0.0:
        settings["priors"]["vrad"] = {"type": "Uniform", "lower": -200.0, "upper": 200.0}
        settings.setdefault("truth", {})["vrad"] = vrad
    return settings


#: Same spiral morphology; ``arms_inflow`` adds constant axisymmetric ``vrad``.
MOCKS = {
    "arms_sb": SpiralStructure(),
    "arms_inflow": SpiralStructure(),
}

#: Extra disc-parameter overrides per mock (on top of `DEFAULT_TRUTH`).
MOCK_KINEMATICS = {
    "arms_sb": {},
    "arms_inflow": {"vrad": INFLOW_VRAD_KMS},
}


def _dirty_moments(uvd: UVData, truth: dict):
    """Dirty cube moments of a mock, plus the grid it was imaged on."""
    spectral = spectral_axis(
        uvd.frequencies, reference_frequency_ghz=truth["reference_frequency_ghz"],
    )
    geometry = resolve_cube_geometry(
        truth["fov"], uvd.max_baseline_wavelengths,
        uvd.baseline_percentile_wavelengths(95), oversample=2,
    )
    transformer = CubeTransformer(uvd, geometry, "dft")
    dirty = transformer.dirty_cube(uvd.data.ravel())
    rms = float(np.median(transformer.rms_per_channel))

    mom0 = dirty.sum(0) * spectral.dv_kms
    mom0_rms = rms * spectral.dv_kms * np.sqrt(spectral.n_chan)
    v = spectral.to_data_order(spectral.model_velocities_kms)
    weight = np.where(dirty > 3.0 * rms, dirty, 0.0)
    total = weight.sum(0)
    mom1 = np.where(total > 0, (weight * v[:, None, None]).sum(0) / np.maximum(total, 1e-30),
                    np.nan)
    mom1 = np.where(mom0 > 10.0 * mom0_rms, mom1, np.nan)
    return geometry, spectral, dirty, rms, mom0, mom1


def overview_figure(path: Path, mocks: dict[str, tuple[UVData, dict, FreeformSB]]) -> Path:
    """One row per mock: what was simulated, what the data look like, and how
    far the velocity field departs from pure rotation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from .results import _show, _sky_extent

    names = list(mocks)
    fig, axes = plt.subplots(len(names), 5, figsize=(20.0, 4.2 * len(names)),
                             squeeze=False, layout="constrained")

    for row, name in enumerate(names):
        uvd, truth, sb = mocks[name]
        structure = SpiralStructure(**truth["structure"])
        p = truth_parameters(truth)
        geometry, spectral, _, rms, mom0, mom1 = _dirty_moments(uvd, truth)

        yy, xx = geometry.coordinates(1)
        radius, cos_t, sin_t = conventions.disc_coordinates(
            yy, xx, p.centre_ra, p.centre_dec, p.phi, p.inclination,
        )
        arm = spiral_arm_profile(radius, np.arctan2(sin_t, cos_t), structure)
        v_los = line_of_sight_velocity(radius, cos_t, sin_t, arm, p, structure)
        circular = DiscParameters(**{**p.as_dict(), "vrad": 0.0})
        v_circ = line_of_sight_velocity(radius, cos_t, sin_t, arm, circular, structure)
        sb_map = sb.map_jykms
        bright = sb_map > 0.01 * sb_map.max()

        extent = _sky_extent(geometry)
        peak, v0 = sb_map.max(), truth["v_sys"]
        faint = matplotlib.colormaps["inferno"].with_extremes(bad="black", under="black")
        panels = [
            (sb_map, "truth surface brightness [Jy km/s/pixel]",
             dict(cmap=faint, norm=LogNorm(peak / 3e2, peak))),
            (mom0, "dirty moment 0 [Jy/beam km/s]", dict(cmap="inferno")),
            (mom1, "dirty moment 1 [km/s]",
             dict(cmap="RdBu_r", vmin=v0 - 200, vmax=v0 + 200)),
            (np.where(bright, v_los, np.nan), "truth $v_{los}$ [km/s]",
             dict(cmap="RdBu_r", vmin=v0 - 200, vmax=v0 + 200)),
            # identically zero for a mock whose arms are morphology only
            (np.where(bright, v_los - v_circ, np.nan),
             "truth $v_{los}$ $-$ circular [km/s]",
             dict(cmap="PuOr_r", vmin=-80, vmax=80)),
        ]
        for col, (image, title, kw) in enumerate(panels):
            ax = axes[row, col]
            im = _show(ax, image, extent, **kw)
            ax.contour(np.flipud(arm), levels=[0.5], colors="c", linewidths=0.6,
                       extent=extent, origin="lower")
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ax.set_xlabel(r'$\Delta$RA ["]', fontsize=8)
            ax.tick_params(labelsize=7)
        axes[row, 0].set_ylabel(f'{name}\n$\\Delta$Dec ["]', fontsize=10)

    fig.suptitle("structured mocks: 2 Sersic components + 2 trailing m=2 arms "
                 "(cyan: arm crest)", fontsize=12)
    fig.savefig(path, dpi=105, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return path


#: primary fits on the comparison axis (analytic vs default freeform)
FIT_CASES = [(mock, sb) for mock in MOCKS for sb in ("analytic", "freeform")]
#: Matérn freeform, overplotted on the freeform slots at alpha=0.5
OVERLAY_CASES = [(mock, "freeform_matern") for mock in MOCKS]
ALL_FIT_CASES = FIT_CASES + OVERLAY_CASES


def _from_fits_orientation(array: np.ndarray) -> np.ndarray:
    """FITS (row 0 = south) -> native (row 0 = north).

    ``np.flipud`` on a (n_chan, ny, nx) cube flips the *channel* axis, which
    leaves the sky mirrored north-south relative to anything computed on the
    native grid (arm crests, disc coordinates). Flip the spatial Y axis.
    """
    array = np.asarray(array, dtype=float)
    if array.ndim == 3:
        return array[:, ::-1, :].copy()
    if array.ndim == 2:
        return np.flipud(array).copy()
    raise ValueError(f"expected a 2-D map or 3-D cube, got shape {array.shape}")


def _read_fit(directory: Path):
    """A finished fit: its parameters and its dirty cubes, back in pyuvkin's
    native orientation (the FITS on disk are flipped for display).

    Sampler fits use the posterior median with the stored 1σ (median−lo,
    hi−median); optimiser fits use the ML point (Fisher 1σ if cached).
    """
    from astropy.io import fits

    from .results import CubeProducts

    record = json.loads((directory / "best_fit_parameters.json").read_text())
    cubes = {name: _from_fits_orientation(fits.getdata(directory / f"{name}.fits"))
             for name in ("dirty_data", "dirty_model", "dirty_residual", "residual_snr")}
    # `rms` is not stored directly, but residual_snr is the residual over it
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.abs(cubes["dirty_residual"] / cubes["residual_snr"])
    products = CubeProducts(
        model_cube=_from_fits_orientation(fits.getdata(directory / "model_cube.fits")),
        dirty_data=cubes["dirty_data"], dirty_model=cubes["dirty_model"],
        rms=np.nanmedian(ratio, axis=(1, 2)),
    )
    if record.get("kind") == "bayesian_sampling" and "median_pdf" in record:
        values = {**record.get("fixed", {}), **record["median_pdf"]}
    else:
        values = {**record.get("fixed", {}), **record["max_log_likelihood"]}
    values.setdefault("vrad", 0.0)
    values["chi_squared_reduced"] = record["fit_quality"]["chi_squared_reduced"]
    values["vmax_sini"] = (values["maximum_velocity"]
                           * np.sin(np.radians(values["inclination"])))
    values["search"] = record.get("search")
    values["kind"] = record.get("kind")
    # sampler fits already carry asymmetric 1σ; L-BFGS gets a diagonal
    # Fisher estimate at the ML point (cached next to the fit)
    if "errors_1sigma" in record:
        values["errors_1sigma"] = {
            n: [float(a), float(b)] for n, (a, b) in record["errors_1sigma"].items()
        }
    else:
        values["errors_1sigma"] = optimiser_errors_1sigma(directory, record)
    _add_derived_errors(values)
    return values, products


#: finite-difference steps for the diagonal Hessian, in each parameter's unit
_HESSIAN_STEPS = {
    "centre_ra": 1e-3, "centre_dec": 1e-3, "v_sys": 0.5,
    "intensity": 0.02, "scale_radius": 5e-3,
    "inclination": 0.25, "phi": 0.25,
    "turnover_radius": 5e-3, "maximum_velocity": 1.0,
    "velocity_dispersion": 0.5, "vrad": 1.0, "vmax_black_hole": 1.0,
}


def _analysis_for_finished_fit(fit_dir: Path):
    """Rebuild the likelihood of a finished fit, reusing a freeform map if any."""
    from . import config
    from .analysis import KinematicAnalysis
    from .api import prepare_data, prepare_geometry, prepare_surface_brightness
    from .models import DiscParameters, make_renderer
    from .transform import CubeTransformer

    tag = fit_dir.name.removeprefix("fit_")
    settings_path = fit_dir.parent / f"settings_{tag}.json"
    given = json.loads(settings_path.read_text())
    # point at this fit's products so freeform does not re-image; paths are
    # relative to the settings file (the mock directory)
    sb = dict(given.get("surface_brightness") or {})
    if sb.get("type") == "freeform":
        rel = fit_dir.name  # e.g. fit_freeform
        sb["map_fits"] = f"{rel}/surface_brightness/model.fits"
        sb["uncertainty_fits"] = f"{rel}/surface_brightness/uncertainty.fits"
        sb["map_units"] = "jy_per_pixel_per_channel"
        given["surface_brightness"] = sb
    given["out"] = fit_dir.name
    settings = config.load_settings(given, base_dir=settings_path.parent)
    uvd, spectral = prepare_data(settings)
    geometry = prepare_geometry(settings, uvd)
    surface, _ = prepare_surface_brightness(settings, uvd, spectral, fit_dir)
    transformer = CubeTransformer(uvd, geometry, settings["transformer"])
    model_cfg = settings["model"]
    options = {
        "rotation_curve": model_cfg["rotation_curve"],
        "dispersion_curve": model_cfg.get("dispersion_curve", "constant"),
        **(model_cfg.get("options") or {}),
    }
    renderer = make_renderer(model_cfg["backend"], geometry, spectral, surface, options)
    return KinematicAnalysis(transformer, renderer), DiscParameters, surface


def optimiser_errors_1sigma(fit_dir: Path, record: dict | None = None) -> dict[str, list[float]]:
    """1σ errors from the diagonal Fisher matrix at the ML point.

    Cached as ``optimiser_errors.json`` next to the fit. Used when the search
    was an optimiser (no posterior). Returns ``{name: [lo, hi]}`` with
    ``lo == hi`` (symmetric).
    """
    fit_dir = Path(fit_dir)
    cache = fit_dir / "optimiser_errors.json"
    if cache.exists():
        return {n: [float(a), float(b)] for n, (a, b) in json.loads(cache.read_text()).items()}

    record = record or json.loads((fit_dir / "best_fit_parameters.json").read_text())
    if record.get("kind") == "bayesian_sampling" and "errors_1sigma" in record:
        return {n: [float(a), float(b)] for n, (a, b) in record["errors_1sigma"].items()}

    names = list(record["free_parameters"])
    vals = {**record.get("fixed", {}), **record["max_log_likelihood"]}
    analysis, parameter_cls, _ = _analysis_for_finished_fit(fit_dir)

    def nll(theta: np.ndarray) -> float:
        trial = dict(vals)
        for n, v in zip(names, theta):
            trial[n] = float(v)
        inst = parameter_cls.from_dict(
            {k: float(trial[k]) for k in parameter_cls.parameter_names if k in trial}
        )
        return -float(analysis.log_likelihood_function(inst))

    theta0 = np.asarray([float(vals[n]) for n in names], dtype=float)
    nll0 = nll(theta0)
    errors: dict[str, list[float]] = {}
    for i, name in enumerate(names):
        eps = float(_HESSIAN_STEPS.get(name, 1e-3))
        # keep the FD step inside a soft bound so a Uniform prior edge does
        # not make one side evaluate nonsense
        e = np.zeros_like(theta0)
        e[i] = eps
        try:
            h = (nll(theta0 + e) - 2.0 * nll0 + nll(theta0 - e)) / (eps * eps)
        except Exception:
            errors[name] = [float("nan"), float("nan")]
            continue
        if np.isfinite(h) and h > 0:
            sig = float(1.0 / np.sqrt(h))
        else:
            sig = float("nan")
        errors[name] = [sig, sig]

    cache.write_text(json.dumps(errors, indent=2) + "\n")
    return errors


def _add_derived_errors(values: dict) -> None:
    """Propagate 1σ onto ``vmax_sini``; leave χ²/N without an error bar."""
    err = dict(values.get("errors_1sigma") or {})
    if "maximum_velocity" in err and "inclination" in err:
        vmax = float(values["maximum_velocity"])
        inc = np.radians(float(values["inclination"]))
        dv_lo, dv_hi = (float(err["maximum_velocity"][0]),
                        float(err["maximum_velocity"][1]))
        di_lo = float(err["inclination"][0]) * np.pi / 180.0
        di_hi = float(err["inclination"][1]) * np.pi / 180.0
        if all(np.isfinite(x) for x in (dv_lo, dv_hi, di_lo, di_hi)):
            err["vmax_sini"] = [
                float(np.hypot(np.sin(inc) * dv_lo, vmax * np.cos(inc) * di_lo)),
                float(np.hypot(np.sin(inc) * dv_hi, vmax * np.cos(inc) * di_hi)),
            ]
    values["errors_1sigma"] = err


def _truth_values(truth: dict) -> dict:
    values = dict(truth)
    values.setdefault("vrad", 0.0)
    values["vmax_sini"] = (truth["maximum_velocity"]
                           * np.sin(np.radians(truth["inclination"])))
    values["chi_squared_reduced"] = 1.0
    return values


#: parameter, axis label, and whether the truth is a target or just a reference
_COMPARED = [
    ("inclination", "inclination [deg]"),
    ("phi", "position angle [deg]"),
    ("maximum_velocity", r"$v_{max}$ [km/s]"),
    ("vmax_sini", r"$v_{max}\,\sin i$ [km/s]"),
    ("vrad", r"$v_{rad}$ [km/s] (+out)"),
    ("turnover_radius", r'$r_{t}$ ["]'),
    ("velocity_dispersion", r"$\sigma_{v}$ [km/s]"),
    ("chi_squared_reduced", r"$\chi^2/N$"),
]


def fit_parameter_figure(path: Path, fits_by_case: dict, truths: dict) -> Path:
    """One panel per parameter: where each of the four fits landed, against
    the truth. Error bars are 1σ (posterior for samplers; diagonal Fisher at
    the ML point for L-BFGS). Matérn freeform is overplotted on the freeform
    x-slots at alpha=0.5 when present."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"analytic": "tab:red", "freeform": "tab:blue",
               "freeform_matern": "tab:blue"}
    markers = {"arms_sb": "o", "arms_inflow": "s"}
    freeform_x = {m: i for i, (m, s) in enumerate(FIT_CASES) if s == "freeform"}
    fig, axes = plt.subplots(2, 4, figsize=(17.0, 7.4), layout="constrained")

    for ax, (name, label) in zip(axes.ravel(), _COMPARED):
        ys, yerr_lo, yerr_hi = [], [], []
        for x, (mock, sb) in enumerate(FIT_CASES):
            values = fits_by_case.get((mock, sb))
            if values is None:
                continue
            y = float(values[name])
            err = (values.get("errors_1sigma") or {}).get(name)
            if err is not None and all(np.isfinite(err)):
                lo, hi = float(err[0]), float(err[1])
            else:
                lo = hi = np.nan
            ys.append(y); yerr_lo.append(lo); yerr_hi.append(hi)
            ax.errorbar(
                [x], [y],
                yerr=None if not np.isfinite(lo) else [[lo], [hi]],
                fmt=markers[mock], color=colours[sb], markersize=11,
                markeredgecolor="k", markeredgewidth=0.5,
                ecolor=colours[sb], elinewidth=1.2, capsize=3.5, zorder=3,
                alpha=1.0,
            )
        for mock, sb in OVERLAY_CASES:
            values = fits_by_case.get((mock, sb))
            if values is None or mock not in freeform_x:
                continue
            x = freeform_x[mock] + 0.22
            y = float(values[name])
            err = (values.get("errors_1sigma") or {}).get(name)
            if err is not None and all(np.isfinite(err)):
                lo, hi = float(err[0]), float(err[1])
            else:
                lo = hi = np.nan
            ys.append(y); yerr_lo.append(lo); yerr_hi.append(hi)
            ax.errorbar(
                [x], [y],
                yerr=None if not np.isfinite(lo) else [[lo], [hi]],
                fmt=markers[mock], color=colours[sb], markersize=11,
                markeredgecolor="k", markeredgewidth=0.5,
                ecolor=colours[sb], elinewidth=1.2, capsize=3.5, zorder=4,
                alpha=0.5,
            )
        truth_levels = sorted({_truth_values(truths[m])[name] for m, _ in FIT_CASES})
        for truth in truth_levels:
            ax.axhline(truth, color="k", linestyle="--", linewidth=1.0, zorder=1)
        lows = [y - (lo if np.isfinite(lo) else 0.0) for y, lo in zip(ys, yerr_lo)]
        highs = [y + (hi if np.isfinite(hi) else 0.0) for y, hi in zip(ys, yerr_hi)]
        lows.extend(truth_levels)
        highs.extend(truth_levels)
        lo, hi = min(lows), max(highs)
        span = hi - lo
        pad = 0.15 * span if span else 0.1
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(range(len(FIT_CASES)))
        ax.set_xticklabels([f"{m}\n{s}" for m, s in FIT_CASES], fontsize=7.5)
        ax.set_xlim(-0.6, len(FIT_CASES) - 0.2)
        ax.axvline(1.5, color="0.85", linewidth=1.0, zorder=0)
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.25)

    handles = [plt.Line2D([], [], marker=markers[m], color=colours[s], linestyle="none",
                          markersize=9, markeredgecolor="k", markeredgewidth=0.5,
                          label=f"{m}, {s}") for m, s in FIT_CASES]
    if any((m, s) in fits_by_case for m, s in OVERLAY_CASES):
        handles.append(plt.Line2D(
            [], [], marker="o", color="tab:blue", linestyle="none", markersize=9,
            markeredgecolor="k", markeredgewidth=0.5, alpha=0.5,
            label="freeform matern (α=0.5)",
        ))
    handles.append(plt.Line2D([], [], color="k", linestyle="--", label="truth"))
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=9)
    posterior = any(v.get("kind") == "bayesian_sampling" for v in fits_by_case.values())
    err_note = ("1σ posterior (median ±)" if posterior
                else "1σ Fisher at the ML point")
    fig.suptitle("kinematics recovered from a 2-Sersic + spiral-arm disc "
                 f"(error bars: {err_note})", fontsize=12)
    fig.savefig(path, dpi=110, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return path


#: kinematic parameters whose bias (fit − truth) is the experiment's answer
_BIAS = [
    ("inclination", "Δ inclination [deg]"),
    ("phi", "Δ position angle [deg]"),
    ("maximum_velocity", r"Δ $v_{max}$ [km/s]"),
    ("vmax_sini", r"Δ $v_{max}\,\sin i$ [km/s]"),
    ("vrad", r"Δ $v_{rad}$ [km/s]"),
    ("turnover_radius", r'Δ $r_{t}$ ["]'),
]


def fit_bias_figure(path: Path, fits_by_case: dict, truths: dict) -> Path:
    """Grouped bars of (fit − truth): freeform should shrink the left-hand
    (morphology-only) biases if the spiral arms were the cause; the right-hand
    (inflow) column tests recovery of a constant axisymmetric ``vrad``.
    Matérn freeform is drawn at alpha=0.5 when present."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"analytic": "tab:red", "freeform": "tab:blue",
               "freeform_matern": "tab:cyan"}
    mocks = list(MOCKS)
    has_matern = any((m, "freeform_matern") in fits_by_case for m in mocks)
    series = ["analytic", "freeform"] + (["freeform_matern"] if has_matern else [])
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), layout="constrained")
    x = np.arange(len(mocks))
    width = 0.22 if has_matern else 0.34
    offsets = {
        2: [-0.5 * width, 0.5 * width],
        3: [-width, 0.0, width],
    }[len(series)]

    for ax, (name, label) in zip(axes.ravel(), _BIAS):
        for i, sb in enumerate(series):
            deltas = []
            for mock in mocks:
                values = fits_by_case.get((mock, sb))
                truth = _truth_values(truths[mock])[name]
                deltas.append(values[name] - truth if values is not None else np.nan)
            ax.bar(x + offsets[i], deltas, width=width, color=colours[sb],
                   edgecolor="k", linewidth=0.5,
                   label="freeform matern" if sb == "freeform_matern" else sb,
                   alpha=0.5 if sb == "freeform_matern" else 1.0, zorder=3)
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([
            "arms in SB only\n(circular kinematics)",
            "arms + axisymmetric\ninflow",
        ], fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.axvline(0.5, color="0.85", linewidth=1.0, zorder=0)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colours[s], edgecolor="k", linewidth=0.5,
                      alpha=0.5 if s == "freeform_matern" else 1.0,
                      label="freeform matern" if s == "freeform_matern" else s)
        for s in series
    ]
    handles.append(plt.Line2D([], [], color="k", linestyle="--", label="truth"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=9)
    fig.suptitle("kinematic bias (fit − truth): does freeform undo the spiral arms?",
                 fontsize=12)
    fig.savefig(path, dpi=110, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return path


def fit_map_figure(path: Path, fits_by_case: dict, products_by_case: dict, truths: dict,
                   geometry, spectral) -> Path:
    """Column per fit: what the model makes of the morphology, what it leaves
    in the moment-0 residual, and what it leaves in the velocity field."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .results import _show, _sky_extent, bright_mask, moments, pv_slice

    extent = _sky_extent(geometry)
    v, dv = spectral.velocities_kms, spectral.dv_kms
    cases = [c for c in ALL_FIT_CASES if c in products_by_case]
    fig, axes = plt.subplots(4, len(cases), figsize=(4.0 * len(cases), 14.2),
                             squeeze=False, layout="constrained")

    # one colour scale per row, set by the data so the columns are comparable
    m0_peak = max(float(np.nanmax(moments(p.dirty_data, v, dv)[0]))
                  for p in products_by_case.values())
    for col, (mock, sb) in enumerate(cases):
        products = products_by_case[(mock, sb)]
        truth = truths[mock]
        p = truth_parameters(truth)
        structure = SpiralStructure(**truth["structure"])
        mask, rms0 = bright_mask(products, spectral)
        d0, d1 = moments(products.dirty_data, v, dv, mask)
        m0, m1 = moments(products.dirty_model, v, dv, mask)

        yy, xx = geometry.coordinates(1)
        radius, cos_t, sin_t = conventions.disc_coordinates(
            yy, xx, p.centre_ra, p.centre_dec, p.phi, p.inclination,
        )
        arm = spiral_arm_profile(radius, np.arctan2(sin_t, cos_t), structure)

        panels = [
            (m0, "model moment 0 [Jy/beam km/s]", dict(cmap="inferno", vmin=0, vmax=m0_peak)),
            ((d0 - m0) / rms0, r"moment-0 residual [$\sigma$]",
             dict(cmap="RdBu_r", vmin=-5, vmax=5)),
            (d1 - m1, "moment 1: data $-$ model [km/s]",
             dict(cmap="PuOr_r", vmin=-40, vmax=40)),
        ]
        for row, (image, title, kw) in enumerate(panels):
            ax = axes[row, col]
            im = _show(ax, image, extent, **kw)
            ax.contour(np.flipud(arm), levels=[0.5], colors="c", linewidths=0.6,
                       extent=extent, origin="lower")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(f'{title}\n$\\Delta$Dec ["]', fontsize=9)

        # a PV cut along the *true* major axis, so every column is the same cut
        centre = conventions.sky_to_grid(truth["centre_ra"], truth["centre_dec"])
        s, pv_d = pv_slice(products.dirty_data, geometry, centre, truth["phi"])
        _, pv_m = pv_slice(products.dirty_model, geometry, centre, truth["phi"])
        ax = axes[3, col]
        im = ax.imshow(pv_d, origin="lower", aspect="auto", cmap="inferno",
                       extent=[s[0], s[-1], v[0] - dv / 2, v[-1] + dv / 2])
        ax.contour(pv_m, levels=np.nanmax(pv_d) * np.array([0.2, 0.4, 0.6, 0.8]),
                   colors="c", linewidths=0.8,
                   extent=[s[0], s[-1], v[0] - dv / 2, v[-1] + dv / 2], origin="lower")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_xlabel('offset along the true major axis ["]', fontsize=8)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel("PV: data, model contours\nvelocity [km/s]", fontsize=9)

        chi2 = fits_by_case[(mock, sb)]["chi_squared_reduced"]
        axes[0, col].set_title(f"{mock}, {sb}\n$\\chi^2/N$ = {chi2:.3f}", fontsize=11)

    fig.suptitle("what each fit leaves behind (cyan: the mock's arm crest)", fontsize=12)
    fig.savefig(path, dpi=105, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return path


def compare_structured_fits(out: str | Path, *, fit_suffix: str = "") -> dict[str, Path]:
    """Draw the comparison figures from finished fits under ``out``.

    Expects each of ``arms_sb`` / ``arms_inflow`` to already have
    ``fit_analytic/`` and ``fit_freeform/`` (or ``fit_*{fit_suffix}/`` when
    ``fit_suffix`` is set, e.g. ``"_nautilus"``). Matérn freeform is optional
    and overplotted when present (skipped for sampler suffixes that have no
    Matérn run).
    """
    from .grids import resolve_cube_geometry
    from .uvdata import UVData

    out = Path(out)
    truths, values, products = {}, {}, {}
    geometry = spectral = None
    # plot keys stay (mock, analytic|freeform[|_matern]); dirs may add a suffix
    cases = list(FIT_CASES)
    if not fit_suffix:
        cases = list(ALL_FIT_CASES)
    for mock, sb_type in cases:
        fit_dir = out / mock / f"fit_{sb_type}{fit_suffix}"
        plot_key = (mock, sb_type)
        if not (fit_dir / "best_fit_parameters.json").exists():
            if (mock, sb_type) in OVERLAY_CASES:
                continue
            raise FileNotFoundError(
                f"missing {fit_dir / 'best_fit_parameters.json'}; "
                f"run `pyuvkin mock-spiral {out} --fit` first"
            )
        truths[mock] = json.loads((out / mock / "truth.json").read_text())
        values[plot_key], products[plot_key] = _read_fit(fit_dir)
        if geometry is None:
            uvd = UVData.read(out / mock / "dataset")
            spectral = spectral_axis(
                uvd.frequencies,
                reference_frequency_ghz=truths[mock]["reference_frequency_ghz"],
            )
            geometry = resolve_cube_geometry(
                truths[mock]["fov"], uvd.max_baseline_wavelengths,
                uvd.baseline_percentile_wavelengths(95), oversample=2,
            )

    written = {
        "parameters": fit_parameter_figure(out / "fit_parameters.png", values, truths),
        "bias": fit_bias_figure(out / "fit_bias.png", values, truths),
    }
    if geometry is not None and not fit_suffix:
        written["maps"] = fit_map_figure(
            out / "fit_maps.png", values, products, truths, geometry, spectral,
        )
    return written


def fit_structured_mocks(out: str | Path, *, refit: bool = False) -> dict[str, Path]:
    """Fit every mock under ``out`` with analytic, freeform, and Matérn freeform
    and draw the comparison. Skips a fit that already has ``best_fit_parameters.json``
    unless ``refit`` is set. Expects `write_structured_mocks` to have run."""
    import shutil

    from .api import run
    from .config import load_settings

    out = Path(out)
    for mock, sb_type in ALL_FIT_CASES:
        fit_dir = out / mock / f"fit_{sb_type}"
        settings_path = out / mock / f"settings_{sb_type}.json"
        # keep settings in sync with the current truth (e.g. vrad prior)
        if (out / mock / "truth.json").exists():
            truth = json.loads((out / mock / "truth.json").read_text())
            settings_path.write_text(
                json.dumps(fit_settings(truth, sb_type), indent=2) + "\n",
            )
        if (fit_dir / "best_fit_parameters.json").exists() and not refit:
            print(f"keeping existing {fit_dir}")
            continue
        if refit and fit_dir.exists():
            # PyAutoFit resumes from autofit/*/.completed; wipe so a changed
            # model (e.g. new vrad prior) actually re-optimises
            shutil.rmtree(fit_dir)
        settings = load_settings(json.loads(settings_path.read_text()),
                                 base_dir=settings_path.parent)
        print(f"fitting {mock} / {sb_type} -> {fit_dir}")
        run(settings)
    return compare_structured_fits(out)


def write_structured_mocks(out: str | Path, *, method: str = "lbfgs",
                           backend: str = "thindisk", **kwargs) -> dict[str, Path]:
    """Write every mock in `MOCKS` under ``out``, each with its dataset, truth,
    truth surface-brightness map and analytic/freeform(/matern) fit settings,
    plus one overview figure comparing them."""
    out = Path(out)
    written, simulated = {}, {}
    sb_types = sorted({sb for _, sb in ALL_FIT_CASES})
    for name, structure in MOCKS.items():
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        parameters = {**DEFAULT_TRUTH, **MOCK_KINEMATICS.get(name, {})}
        uvd, truth, sb = simulate_structured_disc(
            parameters=parameters, structure=structure, **kwargs,
        )
        uvd.write(d / "dataset", overwrite=True)
        truth["mock"] = name
        (d / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
        write_truth_sb_fits(d / "truth_sb.fits", sb, uvd.meta)
        for sb_type in sb_types:
            settings = fit_settings(truth, sb_type, method=method, backend=backend)
            (d / f"settings_{sb_type}.json").write_text(json.dumps(settings, indent=2) + "\n")
        written[name] = d
        simulated[name] = (uvd, truth, sb)
    written["overview"] = overview_figure(out / "mocks_overview.png", simulated)
    return written


def truth_parameters(truth: dict) -> DiscParameters:
    """The `DiscParameters` recorded in a structured mock's ``truth.json``."""
    return DiscParameters.from_dict({k: v for k, v in truth.items() if k in PARAMETER_NAMES})
