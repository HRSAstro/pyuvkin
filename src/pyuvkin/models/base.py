"""What a kinematic backend has to provide.

A `Renderer` is built once per fit from the geometry, the spectral axis, the
surface-brightness choice and backend options, and then asked for a cube per
trial. The cube contract is fixed for every backend:

* shape ``(n_chan, ny, nx)`` on the *image* grid (`CubeGeometry.shape`), in
  the **data's channel order** (`SpectralAxis.to_data_order` does this);
* units Jy per pixel per channel, so ``cube.sum() * dv == intensity``
  (up to what falls outside the field or the velocity range);
* native orientation, row 0 = +y (north), +x = west;
* the *intrinsic* sky: no primary beam, no PSF, no lensing.
"""

from __future__ import annotations

import numpy as np

from ..grids import CubeGeometry, block_sum
from ..spectral import SpectralAxis
from .parameters import DiscParameters
from .sb import AnalyticSB, FreeformSB


class Renderer:
    name = "abstract"
    #: parameters this backend actually responds to (the rest are fixed)
    parameter_names: tuple[str, ...] = ()
    supports_freeform = False

    def __init__(
        self,
        geometry: CubeGeometry,
        spectral: SpectralAxis,
        sb: AnalyticSB | FreeformSB,
        options: dict | None = None,
    ):
        self.geometry = geometry
        self.spectral = spectral
        self.sb = sb
        self.options = dict(options or {})
        if sb.is_freeform and not self.supports_freeform:
            raise ValueError(
                f"the {self.name!r} backend cannot use a freeform surface "
                "brightness; use 'thindisk', 'kinms' or 'bbarolo'."
            )

    # ------------------------------------------------------------ interface
    def render(self, p: DiscParameters) -> np.ndarray:
        """Cube on the *render* grid, ascending model velocity order."""
        raise NotImplementedError

    def cube(self, p: DiscParameters) -> np.ndarray:
        """The contract above: image grid, data channel order, Jy/pixel."""
        c = self.render(p)
        c = block_sum(c, self.geometry.render_oversample)
        c = self.spectral.to_data_order(c)
        if not np.all(np.isfinite(c)):
            raise FloatingPointError(f"{self.name} produced a non-finite cube")
        return np.ascontiguousarray(c, dtype=float)

    def as_dict(self) -> dict:
        return {"backend": self.name, "options": self.options, "surface_brightness": self.sb.as_dict()}

    # -------------------------------------------------------------- helpers
    @property
    def dv(self) -> float:
        return self.spectral.dv_kms

    @property
    def n_chan(self) -> int:
        return self.spectral.n_chan

    @property
    def render_shape(self) -> tuple[int, int]:
        return self.geometry.render_shape

    @property
    def render_pixel_scale(self) -> float:
        return self.geometry.render_pixel_scale

    def render_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        return self.geometry.coordinates(self.geometry.render_oversample)
