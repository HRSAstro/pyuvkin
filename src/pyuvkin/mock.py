"""Mock line cubes in pyuvimage's dataset format, for tests and demos.

    uvd, truth = mock.simulate_disc(parameters, ...)
    mock.write_mock_dataset("mock/", parameters, ...)   # dataset + truth.json

The model is rendered by one of pyuvkin's own backends and forward-modelled
with `CubeTransformer`, so a fit to the mock exercises exactly the code a fit
to real data does. `pyuvimage.mock.random_uv_coverage` supplies the uv
sampling.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pyuvimage.mock import random_uv_coverage

from .grids import resolve_cube_geometry
from .models import PARAMETER_NAMES, AnalyticSB, DiscParameters, make_renderer
from .spectral import C_KM_S, spectral_axis
from .transform import CubeTransformer
from .uvdata import UVData

DEFAULT_TRUTH = dict(
    centre_ra=0.05, centre_dec=-0.08, v_sys=15.0, intensity=2.0, scale_radius=0.25,
    inclination=55.0, phi=40.0, turnover_radius=0.15, maximum_velocity=250.0,
    velocity_dispersion=40.0, vmax_black_hole=0.0,
)

DEFAULT_META = {
    "phase_centre_ra_deg": 150.0,
    "phase_centre_dec_deg": 2.0,
    "dish_diameter_m": 12.0,
    "telescope": "MOCK",
    "noise_estimate": "mock",
}


def channel_frequencies(
    reference_frequency_hz: float, n_chan: int, dv_kms: float, v_centre_kms: float = 0.0,
) -> np.ndarray:
    """Uniform channels in frequency, descending like a real ALMA export, so
    the radio velocities are ascending and centred on ``v_centre_kms``."""
    k = np.arange(n_chan) - (n_chan - 1) / 2.0
    v = v_centre_kms + k * dv_kms
    return reference_frequency_hz * (1.0 - v / C_KM_S)


def simulate_disc(
    parameters: DiscParameters | dict | None = None,
    *,
    fov: float = 3.0,
    n_vis: int = 2000,
    max_baseline_m: float = 1500.0,
    reference_frequency_ghz: float = 230.0,
    n_chan: int = 24,
    dv_kms: float = 30.0,
    sigma_jy: float = 5e-4,
    backend: str = "thindisk",
    options: dict | None = None,
    pixel_scale: float | str = "auto",
    seed: int = 0,
    meta: dict | None = None,
) -> tuple[UVData, dict]:
    """A noisy visibility cube of an analytic disc. Returns the dataset and
    the truth (parameters plus the spectral reference) as a dict."""
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
        placeholder.baseline_percentile_wavelengths(95), pixel_scale=pixel_scale, oversample=2,
    )
    renderer = make_renderer(backend, geometry, spectral, AnalyticSB(), options)
    cube = renderer.cube(parameters)
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
        "backend": backend,
        "fov": fov,
        "pixel_scale": geometry.pixel_scale,
    }
    return uvd, truth


def write_mock_dataset(out: str | Path, parameters=None, **kwargs) -> tuple[Path, Path]:
    """Write ``out/dataset/`` (pyuvimage format) and ``out/truth.json``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    uvd, truth = simulate_disc(parameters, **kwargs)
    ds = uvd.write(out / "dataset", overwrite=True)
    truth_path = out / "truth.json"
    truth_path.write_text(json.dumps(truth, indent=2) + "\n")
    return Path(ds), truth_path


def demo_settings(dataset: str | Path, truth: dict, out: str | Path, method: str = "lbfgs") -> dict:
    """Settings for a fit to a mock: broad priors around the truth."""
    t = truth
    search = {"method": method}
    if str(method).lower() in ("lbfgs", "bfgs"):
        # prior centres + a few restarts; cloud backends need this because
        # finite-difference gradients vanish with the default tiny eps
        search.update({"start": "centre", "restarts": 4})
    return {
        "dataset": str(dataset),
        "out": str(out),
        "fov": t.get("fov", 3.0),
        "spectral": {"reference_frequency_ghz": t["reference_frequency_ghz"]},
        "primary_beam": {"enabled": False},
        "model": {"backend": "thindisk", "rotation_curve": "arctan"},
        "priors": {
            "centre_ra": {"type": "Uniform", "lower": -0.3, "upper": 0.3},
            "centre_dec": {"type": "Uniform", "lower": -0.3, "upper": 0.3},
            "v_sys": {"type": "Uniform", "lower": -60, "upper": 60},
            "intensity": {"type": "LogUniform", "lower": 0.2, "upper": 20},
            "scale_radius": {"type": "Uniform", "lower": 0.05, "upper": 0.8},
            "inclination": {"type": "Uniform", "lower": 20, "upper": 80},
            "phi": {"type": "Uniform", "lower": -90, "upper": 170},
            "turnover_radius": {"type": "Uniform", "lower": 0.02, "upper": 0.6},
            "maximum_velocity": {"type": "Uniform", "lower": 50, "upper": 500},
            "velocity_dispersion": {"type": "Uniform", "lower": 5, "upper": 120},
        },
        "truth": {k: v for k, v in t.items() if k in PARAMETER_NAMES},
        "search": search,
    }
