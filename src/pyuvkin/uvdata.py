"""Visibility data: pyuvimage's `UVData`, plus channel selection for cubes.

A pyuvkin dataset *is* a pyuvimage dataset (``pyuvimage import obs.ms out/``
or a ``casa_export.py`` .npz): uvw in metres, per-channel frequencies,
Stokes-I visibilities and a per-visibility noise map estimated from the data.
Everything about reading, noise re-estimation, recentring and the sign of
``v`` is pyuvimage's and is imported rather than copied.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pyuvimage.uvdata import (  # noqa: F401  (re-exported)
    C_M_S,
    V_SIGN,
    MultiSpwUVData,
    UVData,
    read_dataset,
    recompute_noise,
    shift_image_centre,
)


def single_spw(dataset, spw: int | None = None) -> UVData:
    """A cube fit needs one spectral window: the line's.

    A multi-spw dataset is a list of ragged windows with no common channel
    axis, so the window holding the line has to be named (``spw``) unless
    there is only one.
    """
    if isinstance(dataset, MultiSpwUVData):
        if dataset.n_spw == 1:
            return dataset.spws[0]
        if spw is None:
            raise ValueError(
                f"the dataset holds {dataset.n_spw} spectral windows; a cube "
                "fit uses one. Set \"spw\" to the index of the window with "
                "the line."
            )
        return dataset.spws[int(spw)]
    return dataset


def select_channels(uvd: UVData, channels) -> UVData:
    """Restrict to a channel range ``[start, stop)`` or an explicit list.

    `UVData.select` takes one channel; a cube fit wants a contiguous window
    around the line (and nothing else -- continuum-only channels far from the
    line add data the model predicts as zero, which is fine, but cost time).
    """
    if channels is None:
        return uvd
    if isinstance(channels, dict):
        start = channels.get("start", 0)
        stop = channels.get("stop", uvd.n_chan)
        idx = np.arange(int(start), int(stop))
    elif isinstance(channels, (list, tuple)) and len(channels) == 2 and all(
        isinstance(c, (int, np.integer)) for c in channels
    ):
        idx = np.arange(int(channels[0]), int(channels[1]))
    else:
        idx = np.asarray(channels, dtype=int)
    if idx.size == 0:
        raise ValueError("channel selection is empty")
    if idx.min() < 0 or idx.max() >= uvd.n_chan:
        raise ValueError(
            f"channel selection {idx.min()}..{idx.max()} outside 0..{uvd.n_chan - 1}"
        )
    meta = dict(uvd.meta)
    meta["channel_selection"] = [int(idx[0]), int(idx[-1]) + 1]
    return replace(
        uvd,
        frequencies=uvd.frequencies[idx],
        data=uvd.data[idx],
        noise=uvd.noise[idx],
        flags=None if uvd.flags is None else uvd.flags[idx],
        weight_sigma=None if uvd.weight_sigma is None else uvd.weight_sigma[idx],
        meta=meta,
    )


def channel_window_kms(
    uvd: UVData, reference_frequency_hz: float, v_lo: float, v_hi: float
) -> UVData:
    """Keep the channels whose (radio) velocity lies in ``[v_lo, v_hi]`` km/s."""
    from .spectral import radio_velocity_kms

    v = radio_velocity_kms(uvd.frequencies, reference_frequency_hz)
    keep = np.flatnonzero((v >= min(v_lo, v_hi)) & (v <= max(v_lo, v_hi)))
    if keep.size == 0:
        raise ValueError(
            f"no channels between {v_lo} and {v_hi} km/s (data span "
            f"{v.min():.1f}..{v.max():.1f} km/s)"
        )
    return select_channels(uvd, keep)


def n_data(uvd: UVData) -> int:
    """Real and imaginary parts of every unflagged sample."""
    return 2 * uvd.n_samples
