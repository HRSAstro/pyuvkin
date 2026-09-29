# pyuvkin

Kinematic modelling of interferometric spectral-line data by forward
modelling in the uv-plane.

A 3D kinematic model of the source (a rotating disc) is rendered as a cube,
Fourier transformed channel by channel to the observed uv points, and
compared with the calibrated visibilities. Nothing is ever CLEANed or
re-convolved; dirty images appear only as diagnostics. Data handling is
[pyuvimage](../pyuvimage)'s — the same dataset directories, noise model and
uv conventions — so a pyuvkin cube and a pyuvimage image of the same data sit
on the same sky. 

## Install

```bash
conda activate native_env          # arm64, JAX-enabled
pip install -e ~/Work/pyuvimage
pip install -e . --no-deps
pip install kinms galpak==1.34.0   # optional backends
```

The `bbarolo` backend needs a from-source [pyBBarolo](https://bbarolo.readthedocs.io/)
build (not a simple pip install on Apple Silicon). See
[`docs/install-bbarolo.md`](docs/install-bbarolo.md).



## Use

```bash
pyuvkin template settings.json     # every setting at its default
pyuvkin import obs.ms data/ --spw 3          # pyuvimage's importer
pyuvkin fit settings.json [--method lbfgs] [--backend kinms]
pyuvkin demo out/ --method nautilus          # mock + fit, end to end
```

The CLI is the `pyuvkin` console script (`pip install -e .`); there is no
`python -m pyuvkin` entry point.

or in Python

```python
import pyuvkin
result = pyuvkin.run("settings.json")
result.best_fit_record["median_pdf"]
```

A minimal settings file:

```json
{
  "dataset": "data/",
  "out": "fits/co32",
  "fov": 3.0,
  "spectral": {"rest_frequency_ghz": 345.796, "redshift": 2.31},
  "velocity_window_kms": [-500, 500],
  "model": {"backend": "kinms", "options": {"n_samples": 500000}},
  "surface_brightness": {"type": "analytic"},
  "priors": {
    "centre_ra":  {"type": "Uniform", "lower": -0.5, "upper": 0.5},
    "centre_dec": {"type": "Uniform", "lower": -0.5, "upper": 0.5},
    "v_sys":      {"type": "Uniform", "lower": -200, "upper": 200},
    "intensity":  {"type": "LogUniform", "lower": 0.1, "upper": 50},
    "scale_radius": {"type": "Uniform", "lower": 0.02, "upper": 1.0},
    "inclination":  {"type": "Uniform", "lower": 10, "upper": 85},
    "phi":          {"type": "Uniform", "lower": -180, "upper": 180},
    "turnover_radius":  {"type": "Uniform", "lower": 0.01, "upper": 1.0},
    "maximum_velocity": {"type": "Uniform", "lower": 20, "upper": 800},
    "velocity_dispersion": {"type": "Uniform", "lower": 1, "upper": 300}
  },
  "search": {"method": "nautilus", "n_live": 300}
}
```

[`docs/settings.md`](docs/settings.md) explains every key and its options;
`docs/fit-config-template.json` lists them at their defaults. Copy-ready
backend starters: [`examples/template_bbarolo.json`](examples/template_bbarolo.json),
[`examples/template_galpak.json`](examples/template_galpak.json),
[`examples/template_kinms.json`](examples/template_kinms.json). Worked examples:
[`examples/ALMAQuest.json`](examples/ALMAQuest.json),
[`examples/ruby_co76.json`](examples/ruby_co76.json),
[`examples/REBELS_15.json`](examples/REBELS_15.json). Unknown keys are
refused. A parameter given a number instead of a prior is fixed. Paths are
relative to the settings file.

## The model

One parameter set, whatever renders it:


| name                                  | unit         | meaning                                                                                                        |
| ------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------- |
| `centre_ra`, `centre_dec`             | arcsec       | disc centre, dRA cos δ (+east) and dDec (+north) from the image centre                                         |
| `v_sys`                               | km/s         | systemic velocity relative to the spectral reference (radio convention)                                        |
| `intensity`                           | Jy km/s      | velocity-integrated line flux                                                                                  |
| `scale_radius`                        | arcsec       | exponential scale length `h` of analytic SB (`I ∝ exp(−R/h)`); not the outer radius (`options.rmax`)           |
| `inclination`                         | deg          | 0 face-on, 90 edge-on                                                                                          |
| `phi`                                 | deg          | position angle of the receding major axis, east of north                                                       |
| `turnover_radius`, `maximum_velocity` | arcsec, km/s | parametric rotation curves (`model.rotation_curve`: arctan, tanh, exponential, isothermal)                     |
| `vrot_i`, per-ring `inclination_i`, `phi_i`, … | …   | free tilted rings when `backend: bbarolo` and `rotation_curve: rings` (see settings doc)                       |
| `velocity_dispersion`                 | km/s         | intrinsic isotropic dispersion                                                                                 |
| `vmax_black_hole`                     | km/s         | Keplerian `v_bh/√r` term (KinMS/thindisk)                                                                      |


The spectral reference (v = 0) is `rest_frequency_ghz / (1 + redshift)`, or
`reference_frequency_ghz`, or — with neither — the mean frequency of the full
band before any channel selection, so `velocity_window_kms` and `v_sys`
share one zero point. `spectral.frame` (e.g. `LSRK`, `TOPOCENT`) names the
frame of the data's frequencies for the FITS `SPECSYS`; pyuvimage exports do
not record it, so it is omitted unless given. `RESTFRQ` is written only when
a rest frequency was given; pyuvkin's zero point goes in `VREFFRQ`.

### Backends (`model.backend`)


| backend    | needs            | freeform SB | notes                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------- | ---------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `thindisk` | —                | yes         | built-in analytic thin disc, exact channel integration; fast and smooth, the reference the others are calibrated against                                                                                                                                                                                                                                                                                                                      |
| `kinms`    | `kinms`          | yes         | Monte Carlo clouds; `options`: `n_samples`, `scale_height_arcsec`, `seed`, `clouds_per_pixel`, `cloud_seed`, `vlos_seed`                                                                                                                                                                                                                                                                                                                      |
| `galpak`   | `galpak==1.34.0` | no          | `options`: `aspect` (scale height / half-light radius, GalPaK's 0.15 by default), `flux_profile`, `thickness_profile`, `dispersion_profile`. At the default aspect the disc is thick and `velocity_dispersion` is only the constant part of the line width (GalPaK adds a rotation-mixing term `h_z v / r`); `aspect: 0.05` gives a thin disc whose inclination, rotation curve and dispersion match the other backends (pinned by the tests) |
| `bbarolo`  | `pyBBarolo`      | yes         | GalMod; parametric curves or `rotation_curve: rings` (free VROT per ring). Freeform via `NORM=LOCAL`. `options`: `n_rings`, `rmax`, `free`, `twostage`, `regtype`, `scale_height_arcsec`, `adrift`, optional `linear`/`hanning`. `search.start: bbarolo` runs FitMod3D on the dirty cube to seed ring fits. Install: [`docs/install-bbarolo.md`](docs/install-bbarolo.md) |


`tests/test_conventions.py` pins every backend to the thin disc: centre,
position angle, flux and systemic velocity (on odd and even grids and channel
counts), and inclination, `v_max sin i`, turnover radius and dispersion read
back from the cubes. KinMS and BBarolo build cubes from discrete clouds, so
their likelihoods are not smooth: use a sampler with them, not L-BFGS.
KinMS ≥ 3.0.14 is required (older releases need NumPy 2 / SciPy
patches; the two removed names are shimmed here, the flux-normalisation bug
is not).

### Surface brightness (`surface_brightness.type`)

- `analytic` — exponential disc (`intensity`, `scale_radius`).
- `freeform` — the line channels are collapsed in the
uv-plane (moment-0 in visibility space) and imaged with pyuvimage; the
primary-beam-corrected, S/N-masked map (Jy km/s per pixel) fixes the
morphology and `intensity`, and only the kinematics are fitted (give
`intensity` a prior to let the map rescale). `thindisk` reads the map
per pixel; `kinms` samples it as clouds; `bbarolo` builds ring kinematics
then rescales each sky pixel to the map. The reconstruction is written to
`out/surface_brightness/`; a ready map can be given with `map_fits` (plus
`uncertainty_fits` for the S/N mask) — its orientation and centre are
checked against the data. Because the map and the kinematics come from the
same visibilities, the kinematic errors leave out the map's uncertainty;
`morphology_samples: N` re-fits N maps perturbed by their per-pixel 1σ and
reports the scatter (`morphology` in `best_fit_parameters.json`, with
`errors_total_1sigma`). The perturbation is per pixel and independent, so
it is an estimate, not a marginalisation.



### Search (`search.method`)

Bayesian sampling: `nautilus` (default), `dynesty`, `dynesty_dynamic`,
`emcee`. Optimisation: `lbfgs` (L-BFGS-B, bounded by the prior box), `bfgs`.
Optimisers take `start` (`"prior"`, `"centre"`, `"bbarolo"` / `"fitmod3d"` for
BBarolo free rings, or a dict of values) and `restarts`. Free-ring L-BFGS
defaults to `eps: 1` and `gtol: 2.5e-3 × n_rings` unless overridden. Other
keys pass through to the PyAutoFit search (`n_live`, `nlive`, `nwalkers`/`nsteps`,
`maxiter`, `eps`, ...); `number_of_cores` or `PYUVKIN_CORES` (integer or
`auto`) parallelise a sampler only.

## Outputs


| file                                                                              | contents                                                                                                                         |
| --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `input_parameters.json`                                                           | every setting resolved, data summary, grid, spectral axis, priors as fitted, truth if given                                      |
| `best_fit_parameters.json`                                                        | maximum-likelihood parameters; for samplers also MAP, median and 1σ/3σ intervals, log evidence; χ², N, comparison with the truth |
| `samples.csv`                                                                     | the chain: free parameters, `log_likelihood`, `log_prior`, `log_posterior`, `weight` (PyAutoLens layout)                         |
| `model_cube.fits`                                                                 | best-fit intrinsic sky, Jy/pixel per channel, with FREQ axis                                                                     |
| `dirty_data.fits`, `dirty_model.fits`, `dirty_residual.fits`, `residual_snr.fits` | naturally weighted dirty cubes (Jy/beam) and the residual in σ                                                                   |
| `moment_maps.png`                                                                 | moments 0, 1, 2 of the dirty data and dirty model, residual moment 0 in σ, moment 1/2 differences                                |
| `channel_maps.png`                                                                | every channel: dirty data with dirty-model contours (3, 6, 12, ... σ), residual in σ                                             |
| `pv_diagram.png`                                                                  | position-velocity slices along the fitted major and minor axes with the projected rotation curve                                 |
| `summary.png`, `cornerplot.png`                                                   | one-page overview; posterior corner (samplers)                                                                                   |
| `samples_stage1.csv`                                                              | TWOSTAGE ring fits: stage-1 optimiser trace                                                                                      |
| `surface_brightness/`                                                             | freeform map reconstruction (pyuvimage products)                                                                                 |
| `bbarolo_seed/`                                                                   | FitMod3D seed run (`search.start: bbarolo`)                                                                                      |
| `pyuvkin.log`                                                                     | appended run log (includes memory: process RSS and host RAM available)                                                           |
| `autofit/`                                                                        | PyAutoFit's own output tree for the search                                                                                       |




## Geometry and speed

The image grid follows pyuvimage's rules (`pixel_scale` auto, `oversample`,
`n_pixels`); `render_oversample` renders the model finer and block-sums.
The exact DFT is stored as one matrix per channel when all of them fit in
`PYUVKIN_DFT_MAX_BYTES` (1.5 GB default) — a few ms per evaluation; larger
problems fall back to pynufft or the JAX NUFFT. A ten-parameter Nautilus fit
to a small mock takes ~3 minutes; L-BFGS about a minute.

## Tests

```bash
pytest            # conventions, data, an L-BFGS and an emcee fit to mocks
```

