"""Cube -> visibilities, channel by channel, and the visibility likelihood.

Every channel keeps its own uv coordinates in wavelengths (no channel-averaging
approximation), so there is one Fourier operator per channel, all on the same
image grid. The operators are pyuvimage's: the direct DFT, autoarray's JAX
NUFFT (nufftax), or the vendored pynufft transformer with its half-pixel
correction -- so a pyuvkin model cube and a pyuvimage image of the same data
sit on the same sky.

The model cube is the *intrinsic* sky in Jy/pixel per channel. If a primary
beam is applied it multiplies each channel image before the transform, so the
visibilities are of the apparent sky, as observed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

import autoarray as aa
import autogalaxy as ag

from pyuvimage import fitting as pv_fitting
from pyuvimage import primary_beam as pv_pb

from .grids import CubeGeometry
from .uvdata import UVData

logger = logging.getLogger("pyuvkin")

TRANSFORMERS = ("auto", "dft", "nufft", "pynufft")

#: The exact DFT is run as a precomputed complex matrix per channel (a matrix
#: product per likelihood call, milliseconds) when all the channels' matrices
#: fit in this many bytes; above it a NUFFT is used when one is available.
#: ``PYUVKIN_DFT_MAX_BYTES`` overrides.
DFT_MATRIX_MAX_BYTES = float(os.environ.get("PYUVKIN_DFT_MAX_BYTES", 1.5e9))


def dft_matrix_bytes(n_samples: int, n_image_pixels: int) -> float:
    return 16.0 * float(n_samples) * float(n_image_pixels)


class MatrixDFT:
    """The direct Fourier transform as a stored matrix, for one channel.

    autoarray's `TransformerDFT` recomputes cos and sin of every
    (pixel, visibility) phase on each call, which at ~1 s per evaluation
    rules it out for a search of thousands of evaluations. Here
    ``E = exp(-2 pi i (x u + y v))`` is built once, so a forward model is a
    matrix-vector product. Same grid, same phase convention, same answer to
    rounding.
    """

    def __init__(self, transformer):
        grid = np.asarray(transformer.grid.array, float)        # (n_pix, 2) radians, (y, x)
        uv = np.asarray(transformer.uv_wavelengths, float)      # (n_vis, 2)
        phase = -2.0 * np.pi * (np.outer(uv[:, 0], grid[:, 1]) + np.outer(uv[:, 1], grid[:, 0]))
        self.matrix = np.exp(1j * phase)                         # (n_vis, n_pix)
        self.inner = transformer

    def forward(self, image_slim: np.ndarray) -> np.ndarray:
        return self.matrix @ np.asarray(image_slim, float)


def resolve_transformer_class(transformer: str, n_samples: int, n_image_pixels: int):
    """Pick the per-channel Fourier operator; say why. ``n_samples`` counts
    every unflagged visibility over all the channels being fitted."""
    if transformer not in TRANSFORMERS:
        raise ValueError(f"transformer must be one of {TRANSFORMERS}, not {transformer!r}")
    jax_ok = pv_fitting.jax_available()
    pynufft_cls = pv_fitting.pynufft_transformer_class()

    if transformer == "dft":
        return ag.TransformerDFT, "dft"
    if transformer == "nufft":
        if not jax_ok:
            raise RuntimeError(
                "transformer 'nufft' needs JAX and nufftax: "
                f"{pv_fitting.jax_path_diagnosis()}"
            )
        return ag.TransformerNUFFT, "nufft"
    if transformer == "pynufft":
        if pynufft_cls is None:
            raise RuntimeError("transformer 'pynufft' needs the pynufft package")
        return pynufft_cls, "pynufft"

    nbytes = dft_matrix_bytes(n_samples, n_image_pixels)
    if nbytes <= DFT_MATRIX_MAX_BYTES:
        logger.info(
            "transformer auto -> dft: the exact DFT matrices take %.2g GB "
            "(limit %.2g GB, PYUVKIN_DFT_MAX_BYTES)", nbytes / 1e9, DFT_MATRIX_MAX_BYTES / 1e9,
        )
        return ag.TransformerDFT, "dft"
    if pynufft_cls is not None:
        logger.info("transformer auto -> pynufft: DFT matrices would take %.2g GB", nbytes / 1e9)
        return pynufft_cls, "pynufft"
    if jax_ok:
        logger.info("transformer auto -> nufft (JAX/nufftax): DFT matrices would take %.2g GB",
                    nbytes / 1e9)
        return ag.TransformerNUFFT, "nufft"
    logger.warning(
        "transformer auto -> dft without stored matrices (%.2g GB would be needed); "
        "each evaluation recomputes the full transform and will be slow. Install "
        "pynufft or JAX+nufftax (%s).", nbytes / 1e9, pv_fitting.jax_path_diagnosis(),
    )
    return ag.TransformerDFT, "dft"


@dataclass
class ChannelData:
    uv: np.ndarray        # (n, 2) wavelengths, unflagged rows only
    data: np.ndarray      # (n,) complex
    noise: np.ndarray     # (n,) complex (sigma_re + i sigma_im)
    transformer: object


class CubeTransformer:
    """Forward model and natural-weighted dirty imaging for one channel set."""

    def __init__(
        self,
        uvd: UVData,
        geometry: CubeGeometry,
        transformer: str = "auto",
        primary_beam: np.ndarray | None = None,
    ):
        self.geometry = geometry
        self.mask = geometry.mask()
        # autoarray's slim order for an all-false mask is row-major over native
        self._slim_index = np.nonzero(~np.asarray(self.mask.array, bool))
        self.primary_beam = None if primary_beam is None else np.asarray(primary_beam, float)
        self.transformer_class, self.transformer_name = resolve_transformer_class(
            transformer, int(uvd.n_samples), geometry.n_image_pixels,
        )
        self.channels: list[ChannelData] = []
        for c in range(uvd.n_chan):
            uv = uvd.uv_wavelengths(c)
            d = uvd.data[c]
            s = uvd.noise[c]
            if uvd.flags is not None:
                keep = ~uvd.flags[c]
                uv, d, s = uv[keep], d[keep], s[keep]
            t = self.transformer_class(
                uv_wavelengths=np.asarray(uv, dtype=float), real_space_mask=self.mask
            )
            if self.transformer_name == "dft":
                t = MatrixDFT(t)
            self.channels.append(ChannelData(
                uv=np.asarray(uv, float), data=np.asarray(d, complex),
                noise=np.asarray(s, complex), transformer=t,
            ))
        self.n_chan = len(self.channels)
        counts = [len(ch.data) for ch in self.channels]
        self.slices = []
        start = 0
        for n in counts:
            self.slices.append(slice(start, start + n))
            start += n
        self.n_samples = start
        self.data = np.concatenate([ch.data for ch in self.channels])
        self.noise = np.concatenate([ch.noise for ch in self.channels])
        sig_re = np.abs(self.noise.real)
        sig_im = np.abs(self.noise.imag)
        self._inv_var_re = 1.0 / sig_re**2
        self._inv_var_im = 1.0 / sig_im**2
        self.noise_normalization = float(
            np.sum(np.log(2.0 * np.pi * sig_re**2)) + np.sum(np.log(2.0 * np.pi * sig_im**2))
        )
        # natural weights and per-channel normalisation for dirty images
        self._weights = [
            1.0 / (0.5 * (np.abs(ch.noise.real) ** 2 + np.abs(ch.noise.imag) ** 2))
            for ch in self.channels
        ]
        self._weight_sums = [float(np.sum(w)) for w in self._weights]

    # ---------------------------------------------------------------- forward
    @property
    def n_data(self) -> int:
        return 2 * self.n_samples

    def _array(self, image: np.ndarray) -> aa.Array2D:
        if self.primary_beam is not None:
            image = image * self.primary_beam
        return aa.Array2D(values=np.asarray(image, dtype=float), mask=self.mask)

    def channel_visibilities(self, channel: int, image: np.ndarray) -> np.ndarray:
        t = self.channels[channel].transformer
        if isinstance(t, MatrixDFT):
            if self.primary_beam is not None:
                image = image * self.primary_beam
            return t.forward(np.asarray(image, float)[self._slim_index])
        vis = t.visibilities_from(image=self._array(image))
        return np.asarray(vis, dtype=complex).reshape(-1)

    def model_visibilities(self, cube: np.ndarray) -> np.ndarray:
        """Flattened (channel-major) model visibilities of a ``(n_chan, ny, nx)``
        cube in the data's channel order."""
        cube = np.asarray(cube)
        if cube.shape != (self.n_chan,) + tuple(self.geometry.shape):
            raise ValueError(
                f"cube shape {cube.shape} != {(self.n_chan,) + tuple(self.geometry.shape)}"
            )
        out = np.empty(self.n_samples, dtype=complex)
        for c, sl in enumerate(self.slices):
            if np.any(cube[c]):
                out[sl] = self.channel_visibilities(c, cube[c])
            else:
                out[sl] = 0.0
        return out

    # ------------------------------------------------------------- likelihood
    def chi_squared(self, model_visibilities: np.ndarray) -> float:
        r = self.data - model_visibilities
        return float(np.sum(r.real**2 * self._inv_var_re) + np.sum(r.imag**2 * self._inv_var_im))

    def log_likelihood(self, model_visibilities: np.ndarray) -> float:
        return -0.5 * (self.chi_squared(model_visibilities) + self.noise_normalization)

    def chi_squared_per_channel(self, model_visibilities: np.ndarray) -> np.ndarray:
        r = self.data - model_visibilities
        chi = r.real**2 * self._inv_var_re + r.imag**2 * self._inv_var_im
        return np.array([float(np.sum(chi[sl])) for sl in self.slices])

    # ---------------------------------------------------------- dirty imaging
    def _adjoint(self, channel: int, values: np.ndarray) -> np.ndarray:
        t = self.channels[channel].transformer
        if isinstance(t, MatrixDFT):
            slim = (t.matrix.conj().T @ np.asarray(values, complex)).real
            out = np.zeros(self.geometry.shape)
            out[self._slim_index] = slim
            return out
        vis = ag.Visibilities(np.asarray(values, dtype=complex))
        img = pv_fitting.adjoint_image(t, vis)
        return np.asarray(img.native, dtype=float)

    def dirty_cube(self, visibilities: np.ndarray) -> np.ndarray:
        """Naturally weighted dirty cube [Jy/beam] of flattened visibilities."""
        out = np.zeros((self.n_chan,) + tuple(self.geometry.shape))
        for c, sl in enumerate(self.slices):
            out[c] = self._adjoint(c, visibilities[sl] * self._weights[c]) / self._weight_sums[c]
        return out

    def dirty_beam(self, channel: int = 0) -> np.ndarray:
        w = self._weights[channel]
        return self._adjoint(channel, w.astype(complex)) / self._weight_sums[channel]

    @property
    def rms_per_channel(self) -> np.ndarray:
        """Analytic dirty-image rms per channel [Jy/beam], natural weighting."""
        return np.array([1.0 / np.sqrt(s) for s in self._weight_sums])

    def as_dict(self) -> dict:
        return {
            "transformer": self.transformer_name,
            "n_channels": self.n_chan,
            "n_samples": int(self.n_samples),
            "n_data": int(self.n_data),
            "primary_beam_applied": self.primary_beam is not None,
        }


def primary_beam_for(
    uvd: UVData, geometry: CubeGeometry, dish_diameter_m: float | None,
    pb_factor: float = pv_pb.DEFAULT_PB_FACTOR,
) -> np.ndarray | None:
    """pyuvimage's Gaussian primary beam on the image grid, at the mean
    frequency of the channels being fitted, centred on the phase centre."""
    dish = dish_diameter_m if dish_diameter_m is not None else uvd.meta.get("dish_diameter_m")
    if not dish:
        return None
    return pv_pb.primary_beam_map(
        shape=geometry.shape,
        pixel_scale=geometry.pixel_scale,
        frequency_hz=float(uvd.central_frequency),
        dish_diameter_m=float(dish),
        pb_factor=pb_factor,
        image_centre_offset_arcsec=uvd.meta.get("image_centre_offset_arcsec"),
    )
