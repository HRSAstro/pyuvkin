"""Everything a fit writes.

    input_parameters.json     every setting, resolved: data, geometry, spectral
                              axis, priors as fitted, search, truth if given
    best_fit_parameters.json  maximum-likelihood (and, for samplers, MAP,
                              median and 1/3-sigma intervals) parameters,
                              chi^2, evidence, comparison with the truth
    samples.csv               the chains: one row per sample, columns are the
                              free parameters then log_likelihood, log_prior,
                              log_posterior, weight (PyAutoLens's samples.csv)
    model_cube.fits           best-fit intrinsic sky, Jy/pixel per channel
    dirty_data.fits           naturally weighted dirty cube of the data, Jy/beam
    dirty_model.fits          the same of the model visibilities
    dirty_residual.fits       data - model, Jy/beam
    residual_snr.fits         data - model in units of the per-channel rms
    summary.png               moments, spectrum, channel maps
    cornerplot.png            posterior corner plot (samplers)
    autofit/                  PyAutoFit's own output for the search

FITS cubes carry pyuvimage's WCS (SIN about the phase centre, shifted for a
recentred image) plus a FREQ axis; all are written south-up like every
pyuvimage product.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits

from pyuvimage.products import build_header, to_fits_orientation

from .models.parameters import PARAMETER_NAMES, PARAMETER_UNITS, DiscParameters, parameter_unit

logger = logging.getLogger("pyuvkin")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    return path


# ------------------------------------------------------------- parameters
def free_names(model) -> list[str]:
    """Names of the free parameters in the order autofit lists them."""
    out = []
    for path in model.paths:
        out.append(".".join(path) if isinstance(path, (tuple, list)) else str(path))
    return out


def instance_parameter_names(instance) -> tuple[str, ...]:
    names = getattr(instance, "parameter_names", None)
    if names is not None:
        return tuple(names)
    return PARAMETER_NAMES


def instance_dict(instance) -> dict:
    return {name: float(getattr(instance, name)) for name in instance_parameter_names(instance)}


def best_fit_record(
    result, model, analysis, method: str, is_sampler: bool, truth: dict | None,
) -> tuple[dict, object]:
    """The `best_fit_parameters.json` payload and the instance it is about."""
    samples = result.samples
    names = free_names(model)
    ml = samples.max_log_likelihood()
    ml_values = instance_dict(ml)
    fixed = {n: v for n, v in ml_values.items() if n not in names}
    all_names = instance_parameter_names(ml)

    record = {
        "search": method,
        "kind": "bayesian_sampling" if is_sampler else "optimisation",
        "free_parameters": names,
        "units": {n: parameter_unit(n) for n in all_names},
        "max_log_likelihood": {n: ml_values[n] for n in names},
        "fixed": fixed,
    }
    best = ml
    if is_sampler:
        try:
            mp = samples.max_log_posterior()
            record["max_log_posterior"] = {n: v for n, v in instance_dict(mp).items() if n in names}
        except Exception as e:  # pragma: no cover - depends on sampler
            logger.debug("no MAP: %s", e)
        try:
            median = samples.median_pdf(as_instance=False)
            lo1, hi1 = zip(*samples.values_at_sigma(sigma=1.0, as_instance=False))
            lo3, hi3 = zip(*samples.values_at_sigma(sigma=3.0, as_instance=False))
            record["median_pdf"] = dict(zip(names, map(float, median)))
            record["sigma_1"] = {n: [float(a), float(b)] for n, a, b in zip(names, lo1, hi1)}
            record["sigma_3"] = {n: [float(a), float(b)] for n, a, b in zip(names, lo3, hi3)}
            record["errors_1sigma"] = {
                n: [float(m - a), float(b - m)] for n, m, a, b in zip(names, median, lo1, hi1)
            }
        except Exception as e:  # pragma: no cover
            logger.warning("could not summarise the posterior: %s", e)
        try:
            record["log_evidence"] = float(samples.log_evidence) if samples.log_evidence is not None else None
        except Exception:
            record["log_evidence"] = None
        record["n_samples"] = int(samples.total_samples)
    else:
        record["n_samples"] = int(samples.total_samples)

    quality = analysis.fit_summary(best)
    record["fit_quality"] = quality

    if truth:
        comp = {}
        for n in names:
            if n not in truth:
                continue
            t = float(truth[n])
            row = {"truth": t, "max_log_likelihood": ml_values[n], "offset": ml_values[n] - t}
            if "errors_1sigma" in record:
                lo, hi = record["errors_1sigma"][n]
                med = record["median_pdf"][n]
                err = hi if med < t else lo
                row["median_offset"] = med - t
                row["median_offset_sigma"] = (med - t) / err if err > 0 else None
            comp[n] = row
        record["truth_comparison"] = comp
        try:
            cls = type(best)
            truth_vals = {k: v for k, v in truth.items() if k in all_names or k == "vrot"}
            record["truth_fit_quality"] = analysis.fit_summary(cls.from_dict(truth_vals))
        except Exception as e:  # pragma: no cover
            logger.debug("truth model could not be evaluated: %s", e)
    return record, best


def write_samples_csv(result, model, path: Path) -> Path:
    """One row per sample: free parameters, then log_likelihood, log_prior,
    log_posterior, weight -- PyAutoLens's ``samples.csv`` layout."""
    samples = result.samples
    names = free_names(model)
    params = np.asarray(samples.parameter_lists, dtype=float)
    ll = np.asarray(samples.log_likelihood_list, dtype=float)
    lp = np.asarray(samples.log_prior_list, dtype=float)
    post = np.asarray(samples.log_posterior_list, dtype=float)
    w = np.asarray(samples.weight_list, dtype=float)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(names + ["log_likelihood", "log_prior", "log_posterior", "weight"])
        for i in range(len(ll)):
            writer.writerow(
                [f"{v:.10g}" for v in params[i]]
                + [f"{ll[i]:.10g}", f"{lp[i]:.10g}", f"{post[i]:.10g}", f"{w[i]:.10g}"]
            )
    return path


# ---------------------------------------------------------------- cubes
@dataclass
class CubeProducts:
    model_cube: np.ndarray      # Jy/pixel, intrinsic
    dirty_data: np.ndarray      # Jy/beam
    dirty_model: np.ndarray     # Jy/beam
    rms: np.ndarray             # per channel, Jy/beam
    extra: dict = field(default_factory=dict)

    @property
    def dirty_residual(self) -> np.ndarray:
        return self.dirty_data - self.dirty_model

    @property
    def residual_snr(self) -> np.ndarray:
        return self.dirty_residual / self.rms[:, None, None]


def make_cube_products(transformer, analysis, instance) -> CubeProducts:
    cube = analysis.cube_from(instance)
    model_vis = transformer.model_visibilities(cube)
    return CubeProducts(
        model_cube=cube,
        dirty_data=transformer.dirty_cube(transformer.data),
        dirty_model=transformer.dirty_cube(model_vis),
        rms=transformer.rms_per_channel,
    )


def write_cubes(products: CubeProducts, uvd, geometry, spectral, out: Path, extra_header=None,
                rest_frequency_hz: float | None = None, spectral_frame: str | None = None) -> dict:
    """The frequency axis is written as the dataset carries it. pyuvimage
    exports do not record the MS spectral frame, so ``SPECSYS`` is written
    only when the settings (``spectral.frame``) or the metadata name one.
    ``RESTFRQ`` is the spectral line's rest frequency and is written only
    when one was given; pyuvkin's own zero point goes in ``VREFFRQ``."""
    written = {}
    n = geometry.n_pixels
    freqs = np.asarray(uvd.frequencies, float)
    frame = spectral_frame or uvd.meta.get("spectral_frame") or uvd.meta.get("specsys")

    def header(bunit, more=None):
        h = build_header(
            n_pix=n, pixel_scale_arcsec=geometry.pixel_scale, meta=uvd.meta,
            bunit=bunit, frequencies_hz=freqs,
            extra={**(extra_header or {}), **(more or {})},
        )
        h["ORIGIN"] = "pyuvkin"
        if frame:
            h["SPECSYS"] = str(frame)
        if rest_frequency_hz:
            h["RESTFRQ"] = (float(rest_frequency_hz), "line rest frequency [Hz]")
        h["VREFFRQ"] = (float(spectral.reference_frequency_hz),
                        "pyuvkin v=0 frequency [Hz], radio convention")
        return h

    def w(name, cube, bunit, more=None):
        data = np.stack([to_fits_orientation(c) for c in cube]).astype(np.float32)
        p = out / name
        fits.writeto(p, data, header(bunit, more), overwrite=True)
        written[name] = str(p)

    w("model_cube.fits", products.model_cube, "Jy/pixel")
    w("dirty_data.fits", products.dirty_data, "Jy/beam")
    w("dirty_model.fits", products.dirty_model, "Jy/beam")
    w("dirty_residual.fits", products.dirty_residual, "Jy/beam")
    w("residual_snr.fits", products.residual_snr, "sigma",
      {"RMSMED": (float(np.median(products.rms)), "median per-channel rms [Jy/beam]")})
    return written


# ---------------------------------------------------------------- plots
def moments(cube: np.ndarray, velocities: np.ndarray, dv: float, mask: np.ndarray | None = None):
    m0 = cube.sum(axis=0) * dv
    if mask is None:
        mask = m0 > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = (cube * velocities[:, None, None]).sum(axis=0) / cube.sum(axis=0)
    m1 = np.where(mask, m1, np.nan)
    return m0, m1


def summary_figure(products: CubeProducts, geometry, spectral, out: Path, title: str = "",
                   n_channels: int = 12) -> Path:
    """Compact overview: moments + spectrum on top, channel strips below.

    Channel strips are dirty data / dirty model / residual÷σ (raw residual
    omitted). Constrained layout + tight save keep axis labels inside the PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = spectral.velocities_kms
    dv = spectral.dv_kms
    extent = _sky_extent(geometry)
    d0, d1 = moments(products.dirty_data, v, dv)
    m0, m1 = moments(products.dirty_model, v, dv)
    r0, _ = moments(products.dirty_residual, v, dv)
    mask, rms0 = bright_mask(products, spectral)
    d1 = np.where(mask, d1, np.nan)
    m1 = np.where(mask, m1, np.nan)

    idx = np.unique(np.linspace(0, len(v) - 1, min(n_channels, len(v))).round().astype(int))
    ncol = len(idx)
    vmax0 = float(np.nanmax(np.abs(d0))) or 1.0
    vlim = float(np.nanmax(np.abs(np.concatenate([
        np.nan_to_num(d1).ravel(), np.nan_to_num(m1).ravel(),
    ])))) or 1.0
    vmax_ch = float(np.nanmax(np.abs(products.dirty_data[idx]))) or 1.0

    fig = plt.figure(figsize=(max(11.0, 1.45 * ncol), 9.2), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.04, hspace=0.06, wspace=0.02)
    fig.suptitle(title, fontsize=11)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.0, 1.35], hspace=0.06)

    # ---- moments + spectrum ------------------------------------------------
    top.suptitle("moments and aperture spectrum", fontsize=9)
    axes_t = top.subplots(1, 6, squeeze=False)[0]

    def _panel(ax, img, cmap, lo, hi, label):
        im = _show(ax, img, extent, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(label, fontsize=8, pad=3)
        ax.set_xlabel("dRA [\"]", fontsize=7)
        ax.set_ylabel("dDec [\"]", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_aspect("equal")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=6)
        return im

    _panel(axes_t[0], d0, "inferno", 0, vmax0, "data mom0")
    _panel(axes_t[1], m0, "inferno", 0, vmax0, "model mom0")
    _panel(axes_t[2], r0, "RdBu_r", -vmax0 / 4, vmax0 / 4, "residual mom0")
    _panel(axes_t[3], d1, "RdBu_r", -vlim, vlim, "data mom1")
    _panel(axes_t[4], m1, "RdBu_r", -vlim, vlim, "model mom1")

    ax = axes_t[5]
    ap = mask if mask.any() else np.ones_like(mask, dtype=bool)
    spec_d = (products.dirty_data * ap).sum(axis=(1, 2))
    spec_m = (products.dirty_model * ap).sum(axis=(1, 2))
    ax.step(v, spec_d, where="mid", color="k", lw=1.0, label="data")
    ax.step(v, spec_m, where="mid", color="C3", lw=1.0, label="model")
    ax.step(v, spec_d - spec_m, where="mid", color="0.55", lw=0.9, label="residual")
    ax.axhline(0, color="0.8", lw=0.5)
    ax.set_xlabel("v [km/s]", fontsize=7)
    ax.set_title("spectrum (bright mask)", fontsize=8, pad=3)
    ax.legend(fontsize=6, loc="best", frameon=False)
    ax.tick_params(labelsize=6)
    ax.set_ylabel("Jy/beam × pixels", fontsize=7)

    # ---- channel strips (data / model / resid÷σ) ---------------------------
    bottom.suptitle(
        "channels: dirty data / model / residual÷σ  (shared colour scales per row)",
        fontsize=9,
    )
    axes_b = bottom.subplots(3, ncol, squeeze=False)
    rows = [
        (products.dirty_data, "data", "inferno", 0, vmax_ch),
        (products.dirty_model, "model", "inferno", 0, vmax_ch),
        (products.residual_snr, "resid/σ", "RdBu_r", -5, 5),
    ]
    for r, (cube, label, cmap, lo, hi) in enumerate(rows):
        for c, k in enumerate(idx):
            ax = axes_b[r, c]
            im = _show(ax, cube[k], extent, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{v[k]:+.0f}", fontsize=7, pad=2)
            if c == 0:
                ax.set_ylabel(label, fontsize=8, labelpad=6)
        cb = fig.colorbar(im, ax=axes_b[r, :].tolist(), fraction=0.02, pad=0.01)
        cb.ax.tick_params(labelsize=6)

    path = out / "summary.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return path


def _sky_extent(geometry):
    half = geometry.fov_arcsec / 2.0
    return (half, -half, -half, half)     # dRA decreasing to the right: east left


def _show(ax, img, extent, **kw):
    return ax.imshow(np.flipud(img), origin="lower", extent=extent, **kw)


def bright_mask(products: CubeProducts, spectral) -> tuple[np.ndarray, float]:
    """Where the model moment-0 is bright: above 5 sigma and 10% of its peak."""
    from scipy.ndimage import label

    dv = spectral.dv_kms
    m0 = products.dirty_model.sum(axis=0) * dv
    rms0 = float(np.median(products.rms)) * dv * np.sqrt(len(spectral.velocities_kms))
    mask = m0 > max(5 * rms0, 0.1 * float(np.nanmax(m0)))
    # dirty-beam sidelobes make islands; keep the one holding the peak
    labels, n = label(mask)
    if n > 1:
        peak = np.unravel_index(np.nanargmax(np.where(mask, m0, -np.inf)), m0.shape)
        mask = labels == labels[peak]
    return mask, rms0


def moment_maps_figure(products: CubeProducts, geometry, spectral, out: Path, title: str = "") -> Path:
    """Moments 0, 1, 2 of the dirty data and dirty model, and the residual
    moment 0 with its per-pixel noise. Moments 1 and 2 are read inside the
    bright mask; the residual moment-0 colour scale is in sigma."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = spectral.velocities_kms
    dv = spectral.dv_kms
    extent = _sky_extent(geometry)
    mask, rms0 = bright_mask(products, spectral)

    def moments3(cube):
        m0 = cube.sum(axis=0) * dv
        c = np.clip(cube, 0, None)
        w = c.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            m1 = (c * v[:, None, None]).sum(axis=0) / w
            m2 = np.sqrt(np.clip((c * (v[:, None, None] - m1) ** 2).sum(axis=0) / w, 0, None))
        return m0, np.where(mask, m1, np.nan), np.where(mask, m2, np.nan)

    d = moments3(products.dirty_data)
    m = moments3(products.dirty_model)
    r0 = products.dirty_residual.sum(axis=0) * dv

    vmax0 = float(np.nanmax(np.abs(d[0])))
    vsys = float(np.nanmedian(m[1][mask])) if mask.any() else 0.0
    vlim = float(np.nanmax(np.abs(np.nan_to_num(m[1] - vsys)))) or 1.0
    s2 = float(np.nanmax(np.nan_to_num(m[2]))) or 1.0

    fig, axes = plt.subplots(3, 3, figsize=(12.5, 11.5))
    rows = [("dirty data", d), ("dirty model", m)]
    specs = [
        ("moment 0 [Jy/beam km/s]", "inferno", 0, vmax0),
        (f"moment 1 [km/s]  (v_sys = {vsys:+.0f})", "RdBu_r", vsys - vlim, vsys + vlim),
        ("moment 2 [km/s]", "viridis", 0, s2),
    ]
    for i, (label, mom) in enumerate(rows):
        for j, (name, cmap, lo, hi) in enumerate(specs):
            ax = axes[i, j]
            im = _show(ax, mom[j], extent, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(f"{label}: {name}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            if mask.any():
                ax.contour(np.flipud(mask).astype(float), levels=[0.5], extent=extent,
                           origin="lower", colors="w", linewidths=0.5, alpha=0.6)
    ax = axes[2, 0]
    im = _show(ax, r0 / rms0, extent, cmap="RdBu_r", vmin=-5, vmax=5)
    ax.set_title(f"residual moment 0 [σ]  (σ = {rms0:.3g} Jy/beam km/s)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax = axes[2, 1]
    with np.errstate(invalid="ignore"):
        dm1 = d[1] - m[1]
    im = _show(ax, dm1, extent, cmap="RdBu_r", vmin=-vlim / 4, vmax=vlim / 4)
    ax.set_title("moment 1: data − model [km/s]", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax = axes[2, 2]
    with np.errstate(invalid="ignore"):
        dm2 = d[2] - m[2]
    im = _show(ax, dm2, extent, cmap="RdBu_r", vmin=-s2 / 4, vmax=s2 / 4)
    ax.set_title("moment 2: data − model [km/s]", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for ax in axes.ravel():
        ax.set_xlabel("dRA [\"]", fontsize=8)
        ax.set_ylabel("dDec [\"]", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path = out / "moment_maps.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def channel_maps_figure(products: CubeProducts, geometry, spectral, out: Path, title: str = "",
                        n_channels: int | None = None, n_columns: int = 6) -> Path:
    """Every channel (or ``n_channels`` evenly spaced ones): dirty model, then
    the residual in sigma."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = spectral.velocities_kms
    extent = _sky_extent(geometry)
    n = len(v)
    idx = np.arange(n) if n_channels is None or n_channels >= n else \
        np.unique(np.linspace(0, n - 1, n_channels).round().astype(int))
    ncol = min(n_columns, len(idx))
    nrow = int(np.ceil(len(idx) / ncol))
    vmax = float(np.nanmax(np.abs(products.dirty_model[idx]))) or 1.0

    fig = plt.figure(figsize=(2.2 * ncol + 0.8, 2.2 * nrow * 2 + 1.6), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.04, hspace=0.06, wspace=0.02)
    fig.suptitle(title, fontsize=10)
    top, bottom = fig.subfigures(2, 1, hspace=0.06)
    top.suptitle("dirty model [Jy/beam, 0 to peak]", fontsize=9)
    bottom.suptitle("residual (data − model) in units of the per-channel σ", fontsize=9)
    axes_t = top.subplots(nrow, ncol, squeeze=False)
    axes_b = bottom.subplots(nrow, ncol, squeeze=False)
    for k, c in enumerate(idx):
        r, col = divmod(k, ncol)
        ax = axes_t[r, col]
        im_t = _show(ax, products.dirty_model[c], extent, cmap="inferno", vmin=0, vmax=vmax)
        ax.text(0.04, 0.9, f"{v[c]:+.0f} km/s", transform=ax.transAxes, color="w", fontsize=7)
        ax = axes_b[r, col]
        im_b = _show(ax, products.residual_snr[c], extent, cmap="RdBu_r", vmin=-5, vmax=5)
        ax.text(0.04, 0.9, f"{v[c]:+.0f} km/s", transform=ax.transAxes, color="k", fontsize=7)
    for axes in (axes_t, axes_b):
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes.ravel()[len(idx):]:
            ax.set_visible(False)
    top.colorbar(im_t, ax=axes_t, shrink=0.8, pad=0.02, label="Jy/beam")
    bottom.colorbar(im_b, ax=axes_b, shrink=0.8, pad=0.02, label="σ")
    path = out / "channel_maps.png"
    fig.savefig(path, dpi=110, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return path


def pv_slice(cube: np.ndarray, geometry, centre_yx: tuple[float, float], angle_deg: float,
             slit_pixels: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Position-velocity slice of a native cube through ``centre_yx`` (grid
    arcsec) along the position angle ``angle_deg`` (east of north), averaged
    over a slit ``slit_pixels`` wide. Returns offsets [arcsec] and the
    (n_chan, n_offset) slice. Positive offset is towards the position angle."""
    from scipy.ndimage import map_coordinates

    from .conventions import major_axis_grid_vector

    yy, xx = geometry.coordinates()
    ps = geometry.pixel_scale
    half = geometry.fov_arcsec / 2.0
    n = geometry.shape[0]
    s = np.linspace(-half + ps / 2, half - ps / 2, n)
    ux, uy = major_axis_grid_vector(angle_deg)          # along the slit
    px, py = -uy, ux                                     # across the slit
    yc, xc = centre_yx
    out = np.zeros((cube.shape[0], n))
    offsets = (np.arange(slit_pixels) - (slit_pixels - 1) / 2.0) * ps
    for t in offsets:
        x = xc + s * ux + t * px
        y = yc + s * uy + t * py
        rows = (yy[0, 0] - y) / ps
        cols = (x - xx[0, 0]) / ps
        for c in range(cube.shape[0]):
            out[c] += map_coordinates(cube[c], [rows, cols], order=1, mode="constant", cval=np.nan)
    return s, out / slit_pixels


def _ring_errors_1sigma(errors_1sigma: dict | None, base: str, n_rings: int):
    """Per-ring ``(lo, hi)`` 1σ arrays from ``errors_1sigma``, or ``None``."""
    if not errors_1sigma:
        return None
    from .models.parameters import ring_param_name

    lo = np.full(int(n_rings), np.nan)
    hi = np.full(int(n_rings), np.nan)
    found = False
    shared = errors_1sigma.get(base)
    for i in range(int(n_rings)):
        err = errors_1sigma.get(ring_param_name(base, i), shared)
        if err is None:
            continue
        found = True
        lo[i], hi[i] = float(err[0]), float(err[1])
    return (lo, hi) if found else None


def pv_diagram_figure(products: CubeProducts, geometry, spectral, best, out: Path,
                      title: str = "", rotation_curve: str = "arctan",
                      ring_radii_arcsec=None, backend: str = "",
                      errors_1sigma: dict | None = None) -> Path:
    """PV diagrams of dirty data, dirty model and residual along the fitted
    major axis (top) and minor axis (bottom), with the projected best-fit
    rotation curve ``v_sys ± v_c(R) sin i`` on the major axis.

    ``ring_radii_arcsec`` (BBarolo): GalMod ring centres. With a parametric
    ``rotation_curve`` the overlay samples ``v_c(R)`` at those radii; with
    ``rotation_curve: "rings"`` each marker is the free ``vrot_i``. When
    ``errors_1sigma`` is given (sampler fits), free-ring markers show
    ``±1σ`` on the projected LOS velocity from the ``vrot_i`` uncertainties.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .conventions import sky_to_grid
    from .models.parameters import ring_mean, ring_values, ring_vrot_kms, rotation_curve_kms

    v = spectral.velocities_kms
    order = np.argsort(v)
    free_rings = rotation_curve == "rings"
    n_r = None if ring_radii_arcsec is None else len(np.asarray(ring_radii_arcsec).ravel())
    plot_phi = ring_mean(best, "phi", n_r) if free_rings else float(best.phi)
    plot_inc = ring_mean(best, "inclination", n_r) if free_rings else float(best.inclination)
    plot_vsys = ring_mean(best, "v_sys", n_r) if free_rings else float(best.v_sys)
    plot_cra = ring_mean(best, "centre_ra", n_r) if free_rings else float(best.centre_ra)
    plot_cdec = ring_mean(best, "centre_dec", n_r) if free_rings else float(best.centre_dec)
    centre = sky_to_grid(plot_cra, plot_cdec)
    rms = float(np.median(products.rms))
    cubes = [("dirty data", products.dirty_data), ("dirty model", products.dirty_model),
             ("residual", products.dirty_residual)]
    vmax = float(np.nanmax(np.abs(products.dirty_data))) or 1.0
    rings = None if ring_radii_arcsec is None else np.asarray(ring_radii_arcsec, float).ravel()
    rings = rings if rings is not None and rings.size else None
    ring_vrot = None
    ring_inc = None
    vrot_err = None
    if free_rings and rings is not None:
        ring_vrot = ring_vrot_kms(best, rings.size)
        ring_inc = ring_values(best, "inclination", rings.size)
        vrot_err = _ring_errors_1sigma(errors_1sigma, "vrot", rings.size)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=True, sharey=True,
                             layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04, hspace=0.08, wspace=0.06)
    for i, (axis_name, angle) in enumerate([("major", plot_phi),
                                             ("minor", plot_phi + 90.0)]):
        for j, (label, cube) in enumerate(cubes):
            s, pv = pv_slice(cube, geometry, centre, angle)
            pv = pv[order]
            ax = axes[i, j]
            ds, dvv = s[1] - s[0], spectral.dv_kms
            extent = (s[0] - ds / 2, s[-1] + ds / 2,
                      v[order][0] - dvv / 2, v[order][-1] + dvv / 2)
            if label == "residual":
                im = ax.imshow(pv / rms, origin="lower", extent=extent, aspect="auto",
                               cmap="RdBu_r", vmin=-5, vmax=5)
                ax.set_title(f"{axis_name}: residual [σ]", fontsize=9)
            else:
                im = ax.imshow(pv, origin="lower", extent=extent, aspect="auto",
                               cmap="inferno", vmin=0, vmax=vmax)
                lv = np.array([3, 6, 12, 24, 48, 96]) * rms
                lv = lv[lv < np.nanmax(pv)]
                if lv.size:
                    ax.contour(pv, levels=lv, extent=extent, origin="lower", colors="w",
                               linewidths=0.5, alpha=0.7)
                ax.set_title(f"{axis_name}: {label} [Jy/beam]", fontsize=9)
            if i == 0 and label != "residual":
                r = np.abs(s)
                if free_rings and rings is not None and ring_vrot is not None:
                    sini = np.sin(np.radians(ring_inc))
                    # guide curve: interpolate vrot and sini onto the slit
                    vc = np.interp(r, rings, ring_vrot, left=float(ring_vrot[0]),
                                   right=float(ring_vrot[-1]))
                    si = np.interp(r, rings, sini, left=float(sini[0]),
                                   right=float(sini[-1]))
                    vlos = plot_vsys + np.sign(s) * vc * si
                    ax.plot(s, vlos, color="c", lw=0.7, ls="-", alpha=0.35)
                    mask = rings > 0
                    rr = rings[mask]
                    if rr.size:
                        vcr = ring_vrot[mask]
                        sir = sini[mask]
                        yerr = None
                        if vrot_err is not None:
                            elo, ehi = vrot_err
                            # project δvrot → δ(v_los) at fixed inclination
                            yerr = np.vstack([elo[mask] * sir, ehi[mask] * sir])
                        for sign in (+1.0, -1.0):
                            yy = plot_vsys + sign * vcr * sir
                            lab = "free VROT / ring" if sign > 0 else None
                            if yerr is not None:
                                ax.errorbar(
                                    sign * rr, yy, yerr=yerr, fmt="o", color="c",
                                    ms=4.5, mew=0.6, mec="k", ecolor="c",
                                    elinewidth=0.9, capsize=2.0, capthick=0.7,
                                    label=lab if sign > 0 else None,
                                )
                            else:
                                ax.plot(sign * rr, yy, "o", color="c", ms=4.5,
                                        mew=0.6, mec="k", label=lab)
                elif rings is not None:
                    sini = np.sin(np.radians(plot_inc))
                    vc = rotation_curve_kms(np.maximum(r, 1e-6), best, rotation_curve)
                    vlos = plot_vsys + np.sign(s) * vc * sini
                    ax.plot(s, vlos, color="c", lw=0.7, ls="-", alpha=0.35)
                    rr = rings[rings > 0]
                    if rr.size:
                        vcr = rotation_curve_kms(rr, best, rotation_curve)
                        for sign in (+1.0, -1.0):
                            ax.plot(sign * rr,
                                    plot_vsys + sign * vcr * sini,
                                    "o", color="c", ms=4.5, mew=0.6, mec="k",
                                    label="GalMod rings (parametric v_c)" if sign > 0 else None)
                else:
                    sini = np.sin(np.radians(plot_inc))
                    vc = rotation_curve_kms(np.maximum(r, 1e-6), best, rotation_curve)
                    vlos = plot_vsys + np.sign(s) * vc * sini
                    ax.plot(s, vlos, color="c", lw=1.2, ls="--",
                            label="v_sys ± v_c sin i")
                ax.axhline(plot_vsys, color="c", lw=0.5, alpha=0.5)
            ax.axvline(0.0, color="0.7", lw=0.5)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=6)
            ax.set_ylim(v.min() - spectral.dv_kms / 2, v.max() + spectral.dv_kms / 2)
            ax.tick_params(labelsize=7)
        axes[i, 0].set_ylabel("v [km/s]", fontsize=8, labelpad=4)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.85)
    for ax in axes[1]:
        ax.set_xlabel("offset along slit [\"]  (+ towards PA)", fontsize=8)
    ring_note = ""
    if free_rings and rings is not None:
        ring_note = f"  ·  {rings.size} free tilted rings (VROT+geometry)"
    elif rings is not None:
        ring_note = (f"  ·  {rings.size} GalMod rings, parametric "
                     f"{rotation_curve} v_c")
    elif backend == "bbarolo":
        ring_note = f"  ·  parametric {rotation_curve} v_c (GalMod)"
    fig.suptitle(
        f"{title}\nPA(major) = {plot_phi:.1f}°, centre "
        f"({plot_cra:+.3f}\", {plot_cdec:+.3f}\"), "
        f"slit 3 pix{ring_note}",
        fontsize=9,
    )
    path = out / "pv_diagram.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    return path


def corner_figure(result, model, out: Path, truth: dict | None = None) -> Path | None:
    try:
        import corner
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return None
    samples = result.samples
    names = free_names(model)
    params = np.asarray(samples.parameter_lists, float)
    w = np.asarray(samples.weight_list, float)
    keep = np.isfinite(w) & (w > 0)
    if keep.sum() < 10 or params.shape[1] == 0:
        return None
    labels = [f"{n}\n[{PARAMETER_UNITS[n]}]" for n in names]
    truths = [float(truth[n]) if truth and n in truth else None for n in names] if truth else None
    if truths and all(t is None for t in truths):
        truths = None
    # equal-weight posterior draws: nested-sampling chains carry thousands of
    # negligible-weight points that would otherwise litter the scatter
    pw = w[keep] / w[keep].sum()
    n_draw = int(min(20000, max(2000, 4 * keep.sum())))
    rng = np.random.default_rng(0)
    draws = params[keep][rng.choice(keep.sum(), size=n_draw, p=pw)]
    try:
        fig = corner.corner(
            draws, labels=labels, truths=truths,
            show_titles=True, title_fmt=".3g", quantiles=[0.16, 0.5, 0.84],
            label_kwargs={"fontsize": 8}, title_kwargs={"fontsize": 8},
        )
    except Exception as e:  # pragma: no cover - degenerate posteriors
        logger.warning("corner plot failed: %s", e)
        return None
    path = out / "cornerplot.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
