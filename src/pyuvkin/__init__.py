"""pyuvkin: kinematic modelling of interferometric spectral-line data by
forward modelling in the uv-plane.

Data handling is pyuvimage's (same dataset directories, same noise model,
same uv conventions). The sky is a 3D kinematic model (KinMS, GalPaK, BBarolo
or the built-in thin disc) whose cube is Fourier transformed channel by
channel and compared to the visibilities; nothing is ever CLEANed or
re-convolved.
"""

import logging
import os

logging.getLogger("pyuvkin").addHandler(logging.NullHandler())

# autofit compares the workspace version on import and prints a banner; there
# is no workspace here.
os.environ.setdefault("PYAUTO_SKIP_WORKSPACE_VERSION_CHECK", "1")

# pyuvimage's guards: enable float64 in JAX before anything imports it, and
# hide a broken JAX install from the PyAuto packages.
import pyuvimage  # noqa: E402,F401

__version__ = "0.1.0"


def __getattr__(name):
    if name in ("run", "RunResult"):
        from . import api

        return getattr(api, name)
    raise AttributeError(f"module 'pyuvkin' has no attribute {name!r}")


__all__ = ["run", "RunResult", "__version__"]
