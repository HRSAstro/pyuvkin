"""Freeform surface brightness: the phase-1 reconstruction.

When ``surface_brightness.type == "freeform"`` the morphology is not a
parametric disc but a regularised maximum-likelihood image of the line's
velocity-integrated emission, obtained by running **pyuvimage** on the same
channels the kinematic fit will use. The channels are first collapsed in the
uv-plane -- every visibility averaged over the line channels, its noise
propagated -- which is the moment-0 in visibility space: one sky, whose
brightness in Jy/pixel is the channel-mean of the line; times the number of
channels and the channel width it is the moment-0 map in Jy km/s per pixel.
(Fitting one sky to all the channels jointly instead would meet a chi^2 that
can never reach 1, because the sky changes from channel to channel, and
pyuvimage would over-regularise against that "noise".) The map --
primary-beam corrected, since the kinematic forward model applies the beam
itself, and masked below an S/N threshold using pyuvimage's uncertainty map
-- is the `FreeformSB` the backends consume. It is the same idea as LensKin's
phase-1 "pixelized" reconstruction, with pyuvimage's Gaussian-process prior
in place of a Delaunay mesh, and without a lens.

A map produced elsewhere (a pyuvimage ``model_pbcor.fits`` from a separate
run, say) can be supplied with ``map_fits`` instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from astropy.io import fits

from .models.sb import FreeformSB
from .spectral import SpectralAxis
from .uvdata import UVData

logger = logging.getLogger("pyuvkin")

#: pyuvimage options a pyuvkin run fixes itself and will not pass through
_RESERVED = {"dataset", "fov", "mode", "out", "image_centre", "write"}


def integrated_from_channel_mean(map_mean_jy: np.ndarray, spectral: SpectralAxis) -> np.ndarray:
    """Jy/pixel per channel (channel mean) -> Jy km/s per pixel."""
    return np.asarray(map_mean_jy, float) * spectral.n_chan * spectral.dv_kms


def collapse_channels(uvd: UVData) -> UVData:
    """Average every visibility over the (unflagged) channels: a single
    channel at the mean frequency whose sky is the channel-mean brightness.
    Noise adds in quadrature; rows flagged in every channel stay flagged.

    Real and imaginary sigmas are pooled in quadrature into one common
    value. Difference estimators (and channel averaging) leave a scatter of
    unequal re/im pairs; pyuvimage's sparse inversion -- and autoarray's
    ``apply_sparse_operator`` -- require ``sigma_re == sigma_im`` on every
    visibility, and pyuvimage only auto-pools when the *median* asymmetry is
    nonzero, which misses a sparse minority of mismatches. Pooling here
    preserves total variance and makes the freeform map build reliably.
    """
    data = np.asarray(uvd.data, complex)
    noise = np.asarray(uvd.noise, complex)
    good = ~np.asarray(uvd.flags, bool) if uvd.flags is not None else np.ones(data.shape, bool)
    n = good.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(good, data, 0).sum(axis=0) / n
        var_re = np.where(good, noise.real**2, 0).sum(axis=0) / n**2
        var_im = np.where(good, noise.imag**2, 0).sum(axis=0) / n**2
        # pool: one sigma whose variance is the mean of the two
        var = 0.5 * (var_re + var_im)
    flagged = n == 0
    mean = np.where(flagged, 0, mean)
    sig = np.where(flagged, 1.0, np.sqrt(var))
    sigma = sig + 1j * sig
    meta = dict(uvd.meta)
    meta["collapsed_channels"] = int(uvd.n_chan)
    meta["noise_pooled_re_im"] = True
    return UVData(
        uvw=uvd.uvw, frequencies=np.array([float(np.mean(uvd.frequencies))]),
        data=mean[None, :], noise=sigma[None, :],
        flags=flagged[None, :] if flagged.any() else None, meta=meta,
        antenna1=uvd.antenna1, antenna2=uvd.antenna2, time=uvd.time,
    )


def reconstruct_surface_brightness(
    uvd: UVData,
    fov: float,
    spectral: SpectralAxis,
    out: Path,
    snr_threshold: float = 0.5,
    pb_correction: bool = True,
    dish_diameter_m: float | None = None,
    pyuvimage_kwargs: dict | None = None,
) -> tuple[FreeformSB, dict]:
    """Run pyuvimage MFS on the selected channels; return the SB map and a
    record of the run. Products are written under ``out`` as pyuvimage writes
    them, so the reconstruction can be inspected on its own."""
    import pyuvimage

    kwargs = {k: v for k, v in (pyuvimage_kwargs or {}).items() if k not in _RESERVED}
    kwargs.setdefault("streaming", False)
    kwargs.setdefault("uncertainty_map", True)
    logger.info(
        "freeform surface brightness: pyuvimage on the %d channels collapsed in the "
        "uv-plane, fov %.3g\" -> %s", uvd.n_chan, fov, out,
    )
    result = pyuvimage.run(
        collapse_channels(uvd), fov=fov, mode="mfs", out=str(out), image_centre="0,0",
        pb_correction=pb_correction, dish_diameter=dish_diameter_m, write=True,
        **kwargs,
    )
    p = result.products[0]
    model = p.model_pbcor if (pb_correction and p.model_pbcor is not None) else p.model_image
    model = np.nan_to_num(np.asarray(model, float), nan=0.0)
    pixel_scale = float(result.geometry.pixel_scale)
    integrated = integrated_from_channel_mean(model, spectral)

    masked_fraction = None
    unc_integrated = None
    if p.uncertainty is not None:
        unc = np.asarray(p.uncertainty, float)
        if pb_correction and p.pb is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                unc = np.where(p.pb > 0.1, unc / np.maximum(p.pb, 0.1), np.inf)
        unc_integrated = np.where(np.isfinite(unc), integrated_from_channel_mean(unc, spectral), 0.0)
    if unc_integrated is not None and snr_threshold is not None and snr_threshold > 0:
        snr = np.where(unc > 0, model / unc, 0.0)
        keep = snr >= float(snr_threshold)
        kept_flux = float(integrated[keep].sum())
        total_flux = float(integrated[integrated > 0].sum())
        masked_fraction = 1.0 - kept_flux / total_flux if total_flux > 0 else None
        integrated = np.where(keep, integrated, 0.0)
        logger.info(
            "  S/N >= %.2f keeps %d of %d pixels and %.1f%% of the positive flux",
            snr_threshold, int(keep.sum()), keep.size,
            100.0 * (1.0 - (masked_fraction or 0.0)),
        )
    integrated = np.clip(integrated, 0.0, None)
    sb = FreeformSB(map_jykms=integrated, pixel_scale=pixel_scale, source="pyuvimage",
                    uncertainty_jykms=unc_integrated)
    record = {
        "source": "pyuvimage",
        "output": str(out),
        "n_channels": int(uvd.n_chan),
        "snr_threshold": snr_threshold,
        "masked_flux_fraction": masked_fraction,
        "total_flux_jy_kms": sb.total_flux,
        "pyuvimage": result.parameters,
    }
    logger.info("  surface-brightness map: %.4g Jy km/s in %d lit pixels at %.4g\"/pixel",
                sb.total_flux, int(np.count_nonzero(integrated)), pixel_scale)
    return sb, record


def expected_map_centre_deg(uvd: UVData) -> tuple[float, float] | None:
    """Sky position of the image centre: the phase centre moved by any
    ``image_centre`` offset (pyuvimage's ``(y, x)`` arcsec convention)."""
    ra = uvd.meta.get("phase_centre_ra_deg")
    dec = uvd.meta.get("phase_centre_dec_deg")
    if ra is None or dec is None:
        return None
    ra, dec = float(ra), float(dec)
    off = uvd.meta.get("image_centre_offset_arcsec")
    if off:
        y0, x0 = float(off[0]), float(off[1])
        dec += y0 / 3600.0
        ra -= (x0 / 3600.0) / max(np.cos(np.radians(dec)), 1e-6)
    return ra, dec


def _read_map(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, float)
        header = hdul[0].header
    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(f"{path} is not a 2D image (shape {data.shape})")
    return data, header


def _checked_native(data: np.ndarray, header: fits.Header, path: Path,
                    expected_centre_deg: tuple[float, float] | None) -> tuple[np.ndarray, float]:
    """FITS -> native orientation, after checking the axes run the right way
    and the map is centred where the kinematic grid is centred."""
    cd1 = header.get("CDELT1", header.get("CD1_1"))
    cd2 = header.get("CDELT2", header.get("CD2_2"))
    if cd1 is None or cd2 is None:
        raise ValueError(f"{path} has no CDELT1/2; cannot tell its pixel scale or orientation")
    cd1, cd2 = float(cd1), float(cd2)
    if abs(abs(cd1) - abs(cd2)) > 1e-6 * abs(cd2):
        raise ValueError(f"{path} has non-square pixels ({cd1 * 3600:.5g}\" x {cd2 * 3600:.5g}\")")
    pixel_scale = abs(cd2) * 3600.0
    native = np.nan_to_num(data, nan=0.0)
    if cd2 > 0:
        native = np.flipud(native)          # FITS row 0 south -> native row 0 north
    else:
        logger.warning("%s has CDELT2 < 0 (north down); rows taken as they are", path)
    if cd1 > 0:
        logger.warning("%s has CDELT1 > 0 (east to the right); columns reversed", path)
        native = native[:, ::-1]
    ny, nx = native.shape
    if expected_centre_deg is not None and "CRVAL1" in header and "CRVAL2" in header:
        # sky position of the map's central pixel, small-angle about CRVAL
        cx, cy = (nx + 1) / 2.0, (ny + 1) / 2.0
        crpix1, crpix2 = float(header.get("CRPIX1", cx)), float(header.get("CRPIX2", cy))
        dec_c = float(header["CRVAL2"]) + (cy - crpix2) * cd2
        ra_c = float(header["CRVAL1"]) + (cx - crpix1) * cd1 / max(np.cos(np.radians(dec_c)), 1e-6)
        ra_e, dec_e = expected_centre_deg
        d_ra = (ra_c - ra_e) * np.cos(np.radians(dec_e)) * 3600.0
        d_dec = (dec_c - dec_e) * 3600.0
        if np.hypot(d_ra, d_dec) > 0.5 * pixel_scale:
            raise ValueError(
                f"{path} is centred {d_ra:+.4f}\", {d_dec:+.4f}\" (RA, Dec) from the image "
                f"centre of the data ({ra_e:.6f}, {dec_e:.6f}); the map must be centred "
                "on the kinematic grid. Reconstruct it with the same image_centre, or set "
                "image_centre to match."
            )
    elif expected_centre_deg is None:
        logger.warning("dataset has no phase centre in its metadata; the map's centre "
                       "cannot be checked against the data")
    else:
        logger.warning("%s has no CRVAL1/2; its centre cannot be checked against the data", path)
    return native, pixel_scale


def surface_brightness_from_fits(
    path: str | Path,
    spectral: SpectralAxis,
    units: str = "jy_per_pixel_per_channel",
    snr_threshold: float | None = None,
    uncertainty_fits: str | Path | None = None,
    expected_centre_deg: tuple[float, float] | None = None,
) -> tuple[FreeformSB, dict]:
    """A user-supplied map (a pyuvimage ``model_pbcor.fits``, say), with the
    matching ``uncertainty.fits`` if the S/N mask is wanted. Orientation and
    centring are checked against the data; see `_checked_native`."""
    path = Path(path)
    data, header = _read_map(path)
    native, pixel_scale = _checked_native(data, header, path, expected_centre_deg)
    if units == "jy_per_pixel_per_channel":
        to_integrated = lambda m: integrated_from_channel_mean(m, spectral)  # noqa: E731
    elif units == "jy_kms_per_pixel":
        to_integrated = lambda m: np.asarray(m, float)  # noqa: E731
    else:
        raise ValueError("map_units must be jy_per_pixel_per_channel or jy_kms_per_pixel")
    integrated = to_integrated(native)
    unc = None
    masked_fraction = None
    if uncertainty_fits:
        udata, uheader = _read_map(Path(uncertainty_fits))
        unative, ups = _checked_native(udata, uheader, Path(uncertainty_fits), expected_centre_deg)
        if unative.shape != native.shape or abs(ups - pixel_scale) > 1e-6 * pixel_scale:
            raise ValueError("the uncertainty map does not match the surface-brightness map's grid")
        unc = to_integrated(unative)
        if snr_threshold is not None and snr_threshold > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                snr = np.where(unc > 0, integrated / unc, 0.0)
            keep = snr >= float(snr_threshold)
            total = float(integrated[integrated > 0].sum())
            kept = float(integrated[keep].sum())
            masked_fraction = 1.0 - kept / total if total > 0 else None
            integrated = np.where(keep, integrated, 0.0)
            logger.info("  S/N >= %.2f keeps %d of %d pixels and %.1f%% of the positive flux",
                        snr_threshold, int(keep.sum()), keep.size,
                        100.0 * (1.0 - (masked_fraction or 0.0)))
    elif snr_threshold is not None and snr_threshold > 0:
        logger.warning(
            "surface_brightness.snr_threshold = %.2f but no uncertainty_fits was given: the "
            "map is used unmasked (negative pixels clipped to zero)", snr_threshold,
        )
    sb = FreeformSB(map_jykms=np.clip(integrated, 0, None), pixel_scale=pixel_scale,
                    source=str(path), uncertainty_jykms=unc)
    return sb, {"source": str(path), "units": units, "uncertainty": str(uncertainty_fits) if uncertainty_fits else None,
                "snr_threshold": snr_threshold, "masked_flux_fraction": masked_fraction,
                "total_flux_jy_kms": sb.total_flux, "pixel_scale_arcsec": pixel_scale}
