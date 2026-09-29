"""The spectral axis: channel frequencies -> velocities the model works in.

Velocities are **radio convention**, ``v = c (1 - nu / nu_ref)``, relative to
a reference frequency. That reference is the line's expected observed
frequency (``rest_frequency / (1 + z)``) when a rest frequency and redshift
are given, so that ``v_sys = 0`` means "at the catalogue redshift"; otherwise
it is the mean channel frequency, and ``v_sys`` is relative to the window
centre. Either way the model's systemic velocity is a free parameter, so the
choice changes only what its number means.

The kinematic backends build cubes on an *ascending* velocity axis with a
uniform step; the data's channels ascend in frequency, i.e. descend in
velocity. `SpectralAxis.to_data_order` maps between the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_KM_S = 299792.458


def radio_velocity_kms(frequencies_hz, reference_frequency_hz: float) -> np.ndarray:
    f = np.asarray(frequencies_hz, dtype=float)
    return C_KM_S * (1.0 - f / float(reference_frequency_hz))


def reference_frequency_hz(
    frequencies_hz,
    rest_frequency_ghz: float | None = None,
    redshift: float | None = None,
    reference_frequency_ghz: float | None = None,
) -> float:
    if reference_frequency_ghz is not None:
        return float(reference_frequency_ghz) * 1e9
    if rest_frequency_ghz is not None:
        if redshift is None:
            raise ValueError("rest_frequency_ghz needs a redshift")
        return float(rest_frequency_ghz) * 1e9 / (1.0 + float(redshift))
    return float(np.mean(np.asarray(frequencies_hz, dtype=float)))


@dataclass(frozen=True)
class SpectralAxis:
    """Velocities of the data channels and the model's uniform velocity grid."""

    frequencies_hz: np.ndarray       # data order
    reference_frequency_hz: float
    velocities_kms: np.ndarray       # data order, radio convention
    dv_kms: float                    # channel width (positive)
    model_velocities_kms: np.ndarray  # ascending, same channels

    @property
    def n_chan(self) -> int:
        return int(len(self.frequencies_hz))

    @property
    def model_to_data(self) -> np.ndarray:
        """Index array: ``cube_data = cube_model[model_to_data]``.

        Model channel ``j`` has velocity ``model_velocities_kms[j]`` (ascending);
        data channel ``k`` has ``velocities_kms[k]``. Same set of values, so
        the map is a permutation.
        """
        j = np.searchsorted(self.model_velocities_kms, self.velocities_kms)
        return np.clip(j, 0, self.n_chan - 1)

    def to_data_order(self, cube_model: np.ndarray) -> np.ndarray:
        """Reorder a ``(n_chan, ny, nx)`` model cube from ascending velocity
        to the data's channel order."""
        return np.ascontiguousarray(cube_model[self.model_to_data])

    @property
    def v_min(self) -> float:
        return float(self.model_velocities_kms[0])

    @property
    def v_max(self) -> float:
        return float(self.model_velocities_kms[-1])

    def as_dict(self) -> dict:
        return {
            "reference_frequency_hz": self.reference_frequency_hz,
            "n_channels": self.n_chan,
            "channel_width_kms": self.dv_kms,
            "velocity_range_kms": [self.v_min, self.v_max],
            "frequency_range_hz": [
                float(np.min(self.frequencies_hz)), float(np.max(self.frequencies_hz))
            ],
        }


#: Channels are assumed evenly spaced in frequency; a spread larger than this
#: (relative to the step) is refused, since the backends need a uniform dv.
UNIFORM_STEP_TOLERANCE = 1e-3


def spectral_axis(
    frequencies_hz,
    rest_frequency_ghz: float | None = None,
    redshift: float | None = None,
    reference_frequency_ghz: float | None = None,
) -> SpectralAxis:
    f = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
    if f.size < 2:
        raise ValueError("a cube needs at least two channels")
    steps = np.diff(f)
    if np.any(steps == 0) or np.any(np.sign(steps) != np.sign(steps[0])):
        raise ValueError("channel frequencies must be strictly monotonic")
    if np.max(np.abs(steps - steps[0])) > UNIFORM_STEP_TOLERANCE * abs(steps[0]):
        raise ValueError(
            "channel frequencies are not evenly spaced; the kinematic models "
            "need a uniform velocity step. Regrid or bin the data first."
        )
    ref = reference_frequency_hz(f, rest_frequency_ghz, redshift, reference_frequency_ghz)
    v = radio_velocity_kms(f, ref)
    # constant frequency step -> constant velocity step (radio convention)
    dv = float(abs(v[1] - v[0]))
    model_v = np.sort(v)
    return SpectralAxis(
        frequencies_hz=f,
        reference_frequency_hz=ref,
        velocities_kms=v,
        dv_kms=dv,
        model_velocities_kms=model_v,
    )
