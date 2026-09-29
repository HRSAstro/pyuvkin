"""Spectral axis, channel selection, transformer and mock consistency."""

import numpy as np
import pytest

import autogalaxy as ag

from pyuvkin import mock, uvdata
from pyuvkin.grids import resolve_cube_geometry
from pyuvkin.models import AnalyticSB, DiscParameters, PARAMETER_NAMES, make_renderer
from pyuvkin.spectral import C_KM_S, spectral_axis
from pyuvkin.transform import CubeTransformer


def test_spectral_axis_radio_velocities_and_order():
    f0 = 230e9
    freqs = mock.channel_frequencies(f0, 8, 25.0)
    sp = spectral_axis(freqs, reference_frequency_ghz=230.0)
    assert sp.n_chan == 8 and abs(sp.dv_kms - 25.0) < 1e-9
    assert np.allclose(sp.velocities_kms, C_KM_S * (1 - freqs / f0))
    # descending frequency -> ascending velocity, so model and data orders match
    assert np.all(np.diff(sp.model_velocities_kms) > 0)
    cube = np.arange(8)[:, None, None] * np.ones((8, 2, 2))
    assert np.allclose(sp.to_data_order(cube)[:, 0, 0], np.argsort(np.argsort(sp.velocities_kms)))


def test_spectral_reference_from_rest_frequency_and_redshift():
    freqs = mock.channel_frequencies(345.796e9 / 3.0, 6, 40.0)
    sp = spectral_axis(freqs, rest_frequency_ghz=345.796, redshift=2.0)
    assert abs(sp.reference_frequency_hz - 345.796e9 / 3.0) < 1.0
    assert abs(np.mean(sp.velocities_kms)) < 1e-6


def test_spectral_axis_refuses_nonuniform_channels():
    freqs = np.array([230.0e9, 230.02e9, 230.05e9])
    with pytest.raises(ValueError):
        spectral_axis(freqs)


def test_select_channels_and_velocity_window():
    uvd, _ = mock.simulate_disc(n_vis=50, n_chan=12, dv_kms=30.0)
    sub = uvdata.select_channels(uvd, [2, 7])
    assert sub.n_chan == 5 and np.allclose(sub.frequencies, uvd.frequencies[2:7])
    assert sub.meta["channel_selection"] == [2, 7]
    win = uvdata.channel_window_kms(uvd, 230e9, -50.0, 50.0)
    sp = spectral_axis(win.frequencies, reference_frequency_ghz=230.0)
    assert sp.v_min >= -50.0 - 15.0 and sp.v_max <= 50.0 + 15.0
    assert 3 <= win.n_chan <= 4


def test_matrix_dft_matches_autoarray_and_truth_chi2():
    uvd, truth = mock.simulate_disc(n_vis=200, n_chan=6, seed=3)
    sp = spectral_axis(uvd.frequencies, reference_frequency_ghz=230.0)
    geo = resolve_cube_geometry(3.0, uvd.max_baseline_wavelengths,
                                uvd.baseline_percentile_wavelengths(95))
    p = DiscParameters.from_dict({k: v for k, v in truth.items() if k in PARAMETER_NAMES})
    cube = make_renderer("thindisk", geo, sp, AnalyticSB(), {}).cube(p)
    tr = CubeTransformer(uvd, geo, "dft")
    vis = tr.model_visibilities(cube)
    ref = []
    for c in range(uvd.n_chan):
        t = ag.TransformerDFT(uv_wavelengths=uvd.uv_wavelengths(c), real_space_mask=geo.mask())
        ref.append(np.asarray(t.visibilities_from(
            image=ag.Array2D(values=cube[c], mask=geo.mask()))).reshape(-1))
    ref = np.concatenate(ref)
    assert np.max(np.abs(vis - ref)) < 1e-10 * np.max(np.abs(ref))
    # the mock was made with the same operator and Gaussian noise
    assert abs(tr.chi_squared(vis) / tr.n_data - 1.0) < 0.1
    # zero-spacing flux of a channel equals the channel's image sum
    assert abs(vis[tr.slices[0]].real.max() - cube[0].sum()) < 0.05 * cube[0].sum() + 1e-6


def test_dirty_cube_has_analytic_rms():
    uvd, _ = mock.simulate_disc(n_vis=300, n_chan=4, sigma_jy=1e-3, seed=5)
    geo = resolve_cube_geometry(3.0, uvd.max_baseline_wavelengths,
                                uvd.baseline_percentile_wavelengths(95))
    tr = CubeTransformer(uvd, geo, "dft")
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1e-3, tr.n_samples) + 1j * rng.normal(0, 1e-3, tr.n_samples)
    dirty = tr.dirty_cube(noise)
    assert abs(dirty.std() / tr.rms_per_channel.mean() - 1.0) < 0.2


def test_flagged_rows_are_dropped():
    uvd, _ = mock.simulate_disc(n_vis=100, n_chan=3)
    flags = np.zeros(uvd.data.shape, bool)
    flags[1, :40] = True
    uvd.flags = flags
    geo = resolve_cube_geometry(3.0, uvd.max_baseline_wavelengths,
                                uvd.baseline_percentile_wavelengths(95))
    tr = CubeTransformer(uvd, geo, "dft")
    assert tr.n_samples == 300 - 40 and tr.n_data == 2 * (300 - 40)


def test_collapse_channels_pools_re_im_noise():
    """Freeform's MFS step needs sigma_re == sigma_im on every visibility
    (autoarray's sparse operator refuses any mismatch)."""
    from pyuvkin.freeform import collapse_channels

    uvd, _ = mock.simulate_disc(n_vis=50, n_chan=4, sigma_jy=1e-3, seed=2)
    # deliberately unequal re/im, including a minority of large mismatches
    noise = np.asarray(uvd.noise, complex).copy()
    noise.real[:] = 1e-3
    noise.imag[:] = 1.05e-3
    noise.real[:, :5] = 1e-3
    noise.imag[:, :5] = 1.2e-3
    uvd = type(uvd)(
        uvw=uvd.uvw, frequencies=uvd.frequencies, data=uvd.data, noise=noise,
        flags=uvd.flags, meta=dict(uvd.meta),
        antenna1=uvd.antenna1, antenna2=uvd.antenna2, time=uvd.time,
    )
    collapsed = collapse_channels(uvd)
    assert collapsed.n_chan == 1
    assert np.allclose(collapsed.noise.real, collapsed.noise.imag)
    # total variance conserved: 0.5*(sig_re^2 + sig_im^2) after channel mean
    assert collapsed.meta.get("noise_pooled_re_im") is True
