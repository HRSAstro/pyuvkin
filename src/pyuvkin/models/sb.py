"""Surface brightness of the source: analytic (exponential) or freeform.

The **analytic** case is an exponential disc in the disc plane, normalised so
the cube's velocity-integrated flux is ``intensity``; ``scale_radius`` and
``intensity`` are fit parameters.

The **freeform** case takes a velocity-integrated surface-brightness map
(Jy km/s per pixel) on some square grid centred on the image centre -- in
practice pyuvimage's MFS reconstruction of the line channels (see
`freeform.reconstruct_surface_brightness`) -- and treats it as the sky-plane
morphology. Only the kinematics are fitted: the map fixes both the spatial
structure and the total flux, as LensKin's "pixelized" mode does. The map is
a *projected* morphology, so inclination and position angle enter only
through the line-of-sight velocity field, never by re-projecting the map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass(frozen=True)
class AnalyticSB:
    kind: str = "exponential"

    @property
    def is_freeform(self) -> bool:
        return False

    def as_dict(self) -> dict:
        return {"type": "analytic", "profile": self.kind}


@dataclass(frozen=True)
class FreeformSB:
    """A fixed sky-plane map: ``map_jykms[i, j]`` is Jy km/s in the pixel whose
    centre is ``(y[i], x[j])`` arcsec, native orientation (row 0 = +y)."""

    map_jykms: np.ndarray
    pixel_scale: float
    source: str = "user"
    #: 1-sigma per-pixel uncertainty of the map in Jy km/s, when known
    uncertainty_jykms: np.ndarray | None = None

    @property
    def is_freeform(self) -> bool:
        return True

    def perturbed(self, seed: int) -> "FreeformSB":
        """A realisation ``map + N(0, uncertainty)`` on the lit pixels, for
        propagating the map's uncertainty; pixels are drawn independently,
        which overstates small-scale and understates large-scale freedom."""
        if self.uncertainty_jykms is None:
            raise ValueError("this map has no uncertainty to draw from")
        rng = np.random.default_rng(int(seed))
        lit = self.map_jykms > 0
        draw = self.map_jykms + np.where(lit, rng.normal(0.0, 1.0, self.shape) * self.uncertainty_jykms, 0.0)
        return FreeformSB(map_jykms=np.clip(draw, 0.0, None), pixel_scale=self.pixel_scale,
                          source=f"{self.source} (perturbed, seed {seed})",
                          uncertainty_jykms=self.uncertainty_jykms)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(s) for s in self.map_jykms.shape)

    @property
    def total_flux(self) -> float:
        return float(np.nansum(self.map_jykms))

    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        ny, nx = self.shape
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        y = (cy - np.arange(ny)) * self.pixel_scale
        x = (np.arange(nx) - cx) * self.pixel_scale
        return y, x

    def masked(self, threshold_jykms: float) -> "FreeformSB":
        """Zero every pixel below ``threshold_jykms`` (and every negative one)."""
        m = np.where(self.map_jykms >= max(threshold_jykms, 0.0), self.map_jykms, 0.0)
        return FreeformSB(map_jykms=m, pixel_scale=self.pixel_scale, source=self.source,
                          uncertainty_jykms=self.uncertainty_jykms)

    def on_grid(self, yy: np.ndarray, xx: np.ndarray, pixel_scale: float) -> np.ndarray:
        """The map resampled onto another grid, in Jy km/s per *new* pixel.

        Bilinear in surface brightness density, then renormalised so the total
        flux is exactly conserved. Points outside the map are zero.
        """
        y, x = self.coordinates()
        density = self.map_jykms / self.pixel_scale**2
        # RegularGridInterpolator wants ascending axes; y descends with row
        interp = RegularGridInterpolator(
            (y[::-1], x), density[::-1, :], method="linear", bounds_error=False, fill_value=0.0,
        )
        pts = np.column_stack([yy.ravel(), xx.ravel()])
        out = interp(pts).reshape(yy.shape) * pixel_scale**2
        out = np.clip(out, 0.0, None)
        s = out.sum()
        if s > 0 and self.total_flux > 0:
            out *= self.total_flux / s
        return out

    def clouds(
        self, clouds_per_pixel: int = 64, scale_height_arcsec: float = 0.0, seed: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample the map as point clouds for KinMS: grid ``x``, ``y`` (arcsec),
        ``z`` (arcsec, exponential scale height) and relative flux weights
        summing to one. Uniform jitter within each pixel, so the cloud
        distribution has the map's resolution and nothing finer."""
        y, x = self.coordinates()
        yy, xx = np.meshgrid(y, x, indexing="ij")
        keep = self.map_jykms > 0
        fx, fy, fw = xx[keep], yy[keep], self.map_jykms[keep]
        k = max(1, int(clouds_per_pixel))
        rng = np.random.RandomState(int(seed))
        n = fx.size * k
        xs = np.repeat(fx, k) + (rng.random_sample(n) - 0.5) * self.pixel_scale
        ys = np.repeat(fy, k) + (rng.random_sample(n) - 0.5) * self.pixel_scale
        if scale_height_arcsec > 0:
            zs = scale_height_arcsec * rng.exponential(1.0, n) * rng.choice([-1.0, 1.0], size=n)
        else:
            zs = np.zeros(n)
        w = np.repeat(fw / k, k)
        return xs, ys, zs, w / w.sum()

    def as_dict(self) -> dict:
        return {
            "type": "freeform",
            "source": self.source,
            "shape": list(self.shape),
            "pixel_scale_arcsec": float(self.pixel_scale),
            "total_flux_jy_kms": self.total_flux,
            "n_lit_pixels": int(np.count_nonzero(self.map_jykms > 0)),
        }
