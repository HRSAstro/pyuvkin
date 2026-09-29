"""Kinematic backends, one interface.

    renderer = models.make_renderer("kinms", geometry, spectral, sb, options)
    cube = renderer.cube(parameters)          # (n_chan, ny, nx) Jy/pixel
"""

from __future__ import annotations

from .parameters import DEFAULTS, PARAMETER_NAMES, PARAMETER_UNITS, DiscParameters, make_tilted_ring_class
from .sb import AnalyticSB, FreeformSB

BACKENDS = ("thindisk", "kinms", "galpak", "bbarolo")


def renderer_class(name: str):
    name = str(name).lower()
    if name == "thindisk":
        from .thindisk import ThinDiskRenderer

        return ThinDiskRenderer
    if name == "kinms":
        from .kinms import KinMSRenderer

        return KinMSRenderer
    if name == "galpak":
        from .galpak import GalPaKRenderer

        return GalPaKRenderer
    if name == "bbarolo":
        from .bbarolo import BBaroloRenderer

        return BBaroloRenderer
    raise ValueError(f"unknown backend {name!r}; choose from {BACKENDS}")


def make_renderer(name, geometry, spectral, sb, options=None):
    return renderer_class(name)(geometry, spectral, sb, options)


def backend_available(name: str) -> bool:
    try:
        cls = renderer_class(name)
    except ValueError:
        return False
    if name == "kinms":
        from . import kinms

        return kinms.KinMS is not None
    if name == "galpak":
        from . import galpak

        return galpak.galpak is not None
    if name == "bbarolo":
        from . import bbarolo

        return bbarolo.GalMod is not None
    return cls is not None


__all__ = [
    "BACKENDS", "DEFAULTS", "PARAMETER_NAMES", "PARAMETER_UNITS", "DiscParameters",
    "make_tilted_ring_class", "AnalyticSB", "FreeformSB", "make_renderer",
    "renderer_class", "backend_available",
]
