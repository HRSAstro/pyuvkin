"""Image-plane geometry for the forward model.

One square grid, sized from the field of view and the uv coverage exactly as
pyuvimage does it (`pyuvimage.grids.resolve_geometry`): the pixel scale is
``oversample`` times finer than the Nyquist scale of the baseline length that
95% of the samples lie within (``pixel_scale="auto"``), or of the longest
baseline (``"nyquist"``), or an explicit value in arcsec.

Orientation is autoarray-native throughout: ``(y, x)`` arrays, row 0 at +y
(north), column index increasing with +x. **+x is west** (decreasing RA), as
in pyuvimage: ``x = -dRA``, ``y = dDec`` (see `conventions`). Cubes are
``(n_chan, ny, nx)`` in the data's channel order.

The kinematic backends may render on a grid ``render_oversample`` times finer
still and block-sum onto this one, which is flux-conserving and removes the
pixel-centre sampling error of a steep exponential disc without making the
transform any more expensive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyuvimage.grids import ImageGeometry, nyquist_pixel_scale_arcsec, resolve_geometry

#: Which baseline length the automatic pixel scale is sized from -- the same
#: choice as pyuvimage, for the same reason (see `pyuvimage.api.BASELINE_PERCENTILE`).
BASELINE_PERCENTILE = 95.0


@dataclass(frozen=True)
class CubeGeometry:
    fov_arcsec: float
    pixel_scale: float
    shape: tuple[int, int]
    nyquist_pixel_scale: float
    render_oversample: int = 1

    @property
    def n_pixels(self) -> int:
        return int(self.shape[0])

    @property
    def n_image_pixels(self) -> int:
        return int(self.shape[0] * self.shape[1])

    @property
    def render_shape(self) -> tuple[int, int]:
        k = self.render_oversample
        return (self.shape[0] * k, self.shape[1] * k)

    @property
    def render_pixel_scale(self) -> float:
        return self.pixel_scale / self.render_oversample

    def coordinates(self, oversample: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """``(y, x)`` arcsec of every pixel centre, native orientation
        (row 0 = +y), on the grid ``oversample`` times finer than this one."""
        n = self.shape[0] * int(oversample)
        ps = self.fov_arcsec / n
        c = (n - 1) / 2.0
        i = np.arange(n, dtype=float)
        y = (c - i) * ps
        x = (i - c) * ps
        yy, xx = np.meshgrid(y, x, indexing="ij")
        return yy, xx

    def mask(self):
        import autoarray as aa

        return aa.Mask2D.all_false(shape_native=self.shape, pixel_scales=self.pixel_scale)

    def as_dict(self) -> dict:
        return {
            "fov_arcsec": self.fov_arcsec,
            "image_shape": list(self.shape),
            "image_pixel_scale_arcsec": self.pixel_scale,
            "nyquist_pixel_scale_arcsec": self.nyquist_pixel_scale,
            "render_oversample": int(self.render_oversample),
            "render_pixel_scale_arcsec": self.render_pixel_scale,
        }


def block_sum(image: np.ndarray, factor: int) -> np.ndarray:
    """Sum ``factor x factor`` blocks: a fine Jy/pixel image onto the coarse
    grid, conserving flux. Works on 2D images and ``(n_chan, ny, nx)`` cubes."""
    factor = int(factor)
    if factor == 1:
        return image
    a = np.asarray(image)
    ny, nx = a.shape[-2:]
    lead = a.shape[:-2]
    a = a.reshape(lead + (ny // factor, factor, nx // factor, factor))
    return a.sum(axis=(-3, -1))


def resolve_cube_geometry(
    fov_arcsec: float,
    max_baseline_wavelengths: float,
    effective_baseline_wavelengths: float | None = None,
    pixel_scale: float | str = "auto",
    oversample: int = 2,
    render_oversample: int = 1,
    n_pixels: int | None = None,
) -> CubeGeometry:
    """pyuvimage's geometry rules, with the image grid as the product.

    ``pixel_scale`` and ``oversample`` mean what they mean in pyuvimage: the
    "mesh" scale is auto/nyquist/explicit, and the image grid is ``oversample``
    times finer. ``n_pixels`` overrides the pixel count outright (pyuvimage's
    ``mesh_shape``), in which case ``pixel_scale`` is ignored.
    """
    if n_pixels is not None:
        n_pixels = int(n_pixels) + (int(n_pixels) % 2)  # even, like the auto grid
        mesh_shape, oversample = (n_pixels, n_pixels), 1
    else:
        mesh_shape = None
    geom: ImageGeometry = resolve_geometry(
        fov_arcsec=fov_arcsec,
        max_baseline_wavelengths=max_baseline_wavelengths,
        pixel_scale=pixel_scale,
        mesh_shape=mesh_shape,
        oversample=oversample,
        effective_baseline_wavelengths=effective_baseline_wavelengths,
    )
    return CubeGeometry(
        fov_arcsec=float(geom.fov_arcsec),
        pixel_scale=float(geom.pixel_scale),
        shape=tuple(int(s) for s in geom.shape_native),
        nyquist_pixel_scale=float(nyquist_pixel_scale_arcsec(max_baseline_wavelengths)),
        render_oversample=max(1, int(render_oversample)),
    )
