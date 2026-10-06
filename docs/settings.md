# Settings file reference

One JSON file describes a fit: `pyuvkin fit settings.json` or
`pyuvkin.run("settings.json")`. `pyuvkin template settings.json` writes every
key at its default (`docs/fit-config-template.json`). Backend starters live under
`examples/template_{bbarolo,galpak,kinms}.json`; `examples/ALMAQuest.json`
is a filled-in starting point (swap `dataset` / `spectral` / `image_centre`
for your source).

Ground rules:

- Only `dataset` and `fov` are required; everything else has a default.
- Unknown keys are refused, so a misspelt option is an error rather than
silently ignored. Keys beginning with `_` (e.g. `"_notes"`) are ignored
and can hold comments. Inside `search`, `priors`, `model.options` and
`surface_brightness.pyuvimage` every key is passed through as given.
- Paths (`dataset`, `out`, `map_fits`, `uncertainty_fits`) are relative to
the settings file, not to the working directory.
- CLI flags `--out`, `--method`, `--backend`, `--cores` override the file.

Sky conventions used throughout: offsets are `[dRA, dDec]` in arcsec with
+east, +north (dRA already includes cos dec); position angles are east of
north; velocities are radio velocities `v = c (1 - nu / nu_ref)` relative
to the spectral reference.

## Data


| key                   | default         | meaning                                                                                                                                                                                                                                                                       |
| --------------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset`             | required        | pyuvimage dataset: a directory with `data.fits` etc., or a `casa_export.py` `.npz`. Make one from a measurement set with `pyuvkin import obs.ms out --spw N` (needs python-casacore) or with `pyuvimage/casa_export.py` inside CASA. The data should be continuum-subtracted. |
| `out`                 | `"pyuvkin_out"` | output directory (created; PyAutoFit's own files go under `out/autofit/`).                                                                                                                                                                                                    |
| `spw`                 | `null`          | which spectral window holds the line when the dataset has several; single-window datasets need nothing.                                                                                                                                                                       |
| `channels`            | `null`          | channel window as `[start, stop)` (stop exclusive), `{"start": a, "stop": b}`, or an explicit list of indices. `null` keeps all channels.                                                                                                                                     |
| `velocity_window_kms` | `null`          | `[v_lo, v_hi]` km/s about the spectral reference; keeps the channels inside it. Give this **or** `channels`, not both. Channels the model predicts as zero are harmless but cost time, so window to the line plus some line-free margin.                                      |
| `noise`               | `"keep"`        | per-visibility noise map: `keep` the one stored at import, or re-estimate with pyuvimage's `difference` (successive-time differences), `scaled` (MS weights, scaled) or `hybrid`. The last two need the export to carry weights.                                              |




### `spectral` — the velocity zero point


| key                       | default | meaning                                                                                                                                                                                                   |
| ------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rest_frequency_ghz`      | `null`  | line rest frequency, e.g. 115.271 (CO 1-0), 230.538 (2-1), 345.796 (3-2), 461.041 (4-3), 806.652 (7-6), 1900.537 ([CII])                                                                                  |
| `redshift`                | `null`  | source redshift; with the rest frequency it sets `nu_ref = nu_rest / (1 + z)`                                                                                                                             |
| `reference_frequency_ghz` | `null`  | alternatively the observed reference frequency directly                                                                                                                                                   |
| `frame`                   | `null`  | spectral frame of the dataset's frequencies (`TOPOCENT`, `LSRK`, `BARYCENT`), written to the FITS `SPECSYS` header. Exports do not record it; an unregridded MS is usually `TOPOCENT`. Header label only. |


`v = 0` is at `nu_ref`, and `v_sys` is measured from it, so a good redshift
lets `v_sys` have a narrow prior. If nothing is given, `v = 0` is the mean
of the full band (with a warning). The reference is fixed on the full band
*before* any channel selection, so `velocity_window_kms` and `v_sys` share it.

## Geometry


| key                 | default  | meaning                                                                                                                                                                                                                                                                                                                                                    |
| ------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fov`               | required | field of view, arcsec, full width. Must cover all the line emission (and the primary-beam-corrected sidelobes you care about); cost grows with the square of it. 3-6" is typical for a high-z galaxy.                                                                                                                                                      |
| `pixel_scale`       | `"auto"` | `"auto"`: `oversample` times finer than the Nyquist scale of the baseline length 95% of the samples lie within; `"nyquist"`: same for the longest baseline; or a number in arcsec. A value coarser than Nyquist warns: the longest baselines are then not fully used.                                                                                      |
| `oversample`        | `2`      | the image grid is this much finer than the pixel-scale mesh (pyuvimage's meaning)                                                                                                                                                                                                                                                                          |
| `n_pixels`          | `null`   | fix the pixel count outright instead                                                                                                                                                                                                                                                                                                                       |
| `render_oversample` | `1`      | render the model this many times finer again and block-sum onto the image grid; flux-conserving, and removes pixel-centre sampling error for discs with `scale_radius` of a pixel or two                                                                                                                                                                   |
| `image_centre`      | `[0, 0]` | `[dRA, dDec]` arcsec of the grid centre from the phase centre. Set it to the source position: the fit needs the source inside `fov`, and a small centred field is far cheaper than a large one from the phase centre. The disc's `centre_ra`/`centre_dec` are then measured from this point. Check on the dirty image (`dirty_data.fits`) or a CASA image. |
| `transformer`       | `"auto"` | how model cubes become visibilities: `dft` (exact, precomputed per-channel matrices; used by `auto` when they fit in `PYUVKIN_DFT_MAX_BYTES`, 1.5 GB by default), `pynufft`, or `nufft` (JAX). `auto` falls back to `pynufft` when the matrices are too large.                                                                                             |




### `primary_beam`


| key               | default | meaning                                                                                  |
| ----------------- | ------- | ---------------------------------------------------------------------------------------- |
| `enabled`         | `true`  | multiply the model by a Gaussian primary beam about the phase centre before transforming |
| `dish_diameter_m` | `null`  | dish diameter; default is the value stored at import (12 m for ALMA)                     |
| `pb_factor`       | `1.13`  | FWHM = `pb_factor * lambda / D`                                                          |




## Model



### `model`


| key              | default      | meaning                                                                                                                                                                                                                                            |
| ---------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend`             | `"thindisk"` | what renders the cube; see below                                                                                                                                                                                                                                                                                                                                                          |
| `rotation_curve`      | `"arctan"`   | Parametric: `arctan` (`v = (2 v_max / pi) arctan(r / r_t)`), `tanh`, `exponential`, `isothermal`, `rix` (Rix et al. 1997 multi-parameter: `V = V_t (1+R_t/R)^β / [1+(R_t/R)^ξ]^(1/ξ)`; also accepts alias `multi`) — uses `maximum_velocity` / `turnover_radius`, and for `rix` also `rotation_beta` / `rotation_xi`. `rings` (BBarolo only): free tilted-ring VROT — one `vrot_i` per ring; requires `options.n_rings`. |
| `dispersion_curve`    | `"constant"` | Radial gas dispersion for parametric models: `constant` (`σ = velocity_dispersion`), `exponential` (`σ = σ₀ exp(−R/R_σ)`; Rizzo+2021), `linear` (`σ = σ₀ max(1−R/R_σ, 0)`). `R_σ` is `dispersion_scale_radius`. Not used with `rings` (use per-ring `velocity_dispersion`) or galpak (use `options.dispersion_profile`).                                                                 |
| `options`             | `{}`         | backend-specific, below                                                                                                                                                                                                                                                                                                                                                                   |



| backend    | needs                                       | freeform SB | `options`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------- | ------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `thindisk` | —                                           | yes         | none beyond shared `rotation_curve` / `dispersion_curve`. Built-in analytic infinitely thin disc with exact channel integration; fast, smooth likelihood, works with the optimisers. The reference the others are pinned to.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `kinms`    | `pip install kinms` (>= 3.0.14)             | yes         | `n_samples` (clouds per model, default 5e5; more is smoother and slower), `scale_height_arcsec` (0 = thin), `seed`, and for freeform maps `clouds_per_pixel`, `cloud_seed`, `vlos_seed`. Monte Carlo, so the likelihood is noisy: use a sampler, not `lbfgs`/`bfgs`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `galpak`   | `galpak==1.34.0`                            | no          | `aspect` (scale height / half-light radius, GalPaK's 0.15 by default; `0.05` matches the thin backends), `flux_profile` (`exponential`, `gaussian`, `de_vaucouleurs`, `sersicN2`), `thickness_profile` (`gaussian`, `exponential`, `sech2`, `none`), `dispersion_profile` (`thick`, `thin`, `infinitely_thin`). At the default aspect `velocity_dispersion` is only the constant part of the line width; GalPaK adds a rotation-mixing term `h_z v / r`.                                                                                                                                                                                                                                                                                                              |
| `bbarolo`  | `pyBBarolo` ([install](install-bbarolo.md)) | yes         | `n_rings` (required for `rotation_curve: "rings"`), `free` (which quantities vary per ring; default `VROT VDISP INC PA`; may include `VRAD`), `twostage` (default true: regularise geometry then re-fit VROT/VDISP/VRAD), `regtype` (`auto` / `median` / `bezier` / poly degree), `rmax` (outer ring radius in arcsec; preferred), `n_scale_lengths` (legacy), `scale_height_arcsec`, `adrift` (+ `adrift_pol1`/`adrift_pol2`), `linear` / `hanning` (optional instrumental spectral broadening in channels; **off by default**, only for native-resolution non-averaged spectra). Freeform: GalMod supplies axi-symmetric kinematics, then each sky pixel is rescaled to the map (`NORM=LOCAL`). Build from source — see `[install-bbarolo.md](install-bbarolo.md)`. |




### The fitted parameters

Every backend shares one parameter set for parametric rotation curves. Each
parameter either gets a prior (fitted), a number (fixed), or is fixed at its
default with a warning.


| name                                              | unit           | default   | meaning                                                                                                                                                                                            |
| ------------------------------------------------- | -------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `centre_ra`                                       | arcsec         | 0         | disc centre, dRA from the image centre, +east                                                                                                                                                      |
| `centre_dec`                                      | arcsec         | 0         | disc centre, dDec from the image centre, +north                                                                                                                                                    |
| `v_sys`                                           | km/s           | 0         | systemic velocity relative to the spectral reference                                                                                                                                               |
| `intensity`                                       | Jy km/s        | 1         | velocity-integrated line flux of the model (before primary-beam attenuation). With a freeform map it rescales the map.                                                                             |
| `scale_radius`                                    | arcsec         | 0.3       | exponential *scale length* `h` of the analytic surface brightness (`I ∝ exp(−R/h)`). Not the outer radius — that is `options.rmax`. Ignored (with a warning) for freeform SB.                      |
| `inclination`                                     | deg            | 45        | 0 face-on, 90 edge-on                                                                                                                                                                              |
| `phi`                                             | deg            | 0         | position angle of the *receding* major axis, east of north. Spans 360 deg: a prior of `[-90, 170]` or `[0, 360]` covers every orientation once; `[0, 180]` leaves the sense of rotation ambiguous. |
| `turnover_radius`                                 | arcsec         | 0.1       | rotation-curve turnover radius `r_t` (parametric curves only; dropped for `rings`)                                                                                                                                                 |
| `maximum_velocity`                                | km/s           | 200       | asymptotic rotation velocity (not `v sin i`; parametric only; dropped for `rings`)                                                                                                                                                 |
| `rotation_beta`                                   | —              | 0         | Rix et al. (1997) β (inner slope); only used when `rotation_curve` is `rix`                                                                                                                                                        |
| `rotation_xi`                                     | —              | 3         | Rix et al. (1997) ξ (sharpness of the turnover); only used when `rotation_curve` is `rix`                                                                                                                                          |
| `velocity_dispersion`                             | km/s           | 30        | intrinsic isotropic gas dispersion σ₀ (constant σ, or the central value for `exponential` / `linear`); the channel width is integrated over separately                                                                            |
| `dispersion_scale_radius`                         | arcsec         | 1         | `R_σ` for `dispersion_curve` `exponential` or `linear`; ignored for `constant`                                                                                                                                                     |
| `vmax_black_hole`                                 | km/s           | 0         | Keplerian term `v_bh / sqrt(r)` added in quadrature (`thindisk`, `kinms`)                                                                                                                                                          |
| `vrot_i`                                          | km/s           | 200       | free circular velocity of ring `i` when `rotation_curve` is `rings`                                                                                                                                |
| `vrad_i`                                          | km/s           | 0         | free radial (in-plane) flow of ring `i` when `vrad` is listed in `options.free`; otherwise fixed at 0                                                                                              |
| `inclination_i`, `phi_i`, `velocity_dispersion_i` | deg, deg, km/s | 45, 0, 30 | per-ring geometry / dispersion when listed in `options.free` (default: all three free with `vrot`)                                                                                                 |
| `centre_ra_i`, `centre_dec_i`, `v_sys_i`          | …              | …         | only if added to `options.free` (shared by default)                                                                                                                                                |




### Free tilted rings (`rotation_curve: "rings"`)

BBarolo only. Matches GALFIT's per-ring model: by default each ring has its
own `vrot`, `velocity_dispersion`, `inclination` and `phi` (BBarolo's
`FREE = VROT VDISP INC PA`); centres and `v_sys` stay shared. GalMod is
given arrays for the free quantities. Change the free set with
`options.free` (list or BBarolo-style string, e.g. `"VROT INC PA"`).

**TWOSTAGE** (default `true`): after stage 1, geometric parameters that were
free per ring are radially regularised (`options.regtype`, default
`auto` — median / constant / linear / bezier, as in BBarolo) and stage 2
re-fits only `vrot` and `velocity_dispersion` with that geometry fixed.
Set `"twostage": false` to keep fully independent rings. Stage-1 samples
are written to `samples_stage1.csv`; the regularised geometry is recorded
under `twostage` in `best_fit_parameters.json`.

```json
"model": {
  "backend": "bbarolo",
  "rotation_curve": "rings",
  "options": {
    "n_rings": 8,
    "rmax": 3.0,
    "free": ["vrot", "inclination", "phi", "velocity_dispersion"],
    "twostage": true,
    "regtype": "auto",
    "adrift": true
  }
},
"priors": {
  "vrot": {"type": "Uniform", "lower": 50, "upper": 400},
  "inclination": {"type": "Uniform", "lower": 20, "upper": 80},
  "phi": {"type": "Uniform", "lower": 0, "upper": 360},
  "velocity_dispersion": {"type": "Uniform", "lower": 5, "upper": 80},
  "vrot_3": {"type": "Uniform", "lower": 100, "upper": 250}
}
```

`options.rmax` (arcsec) sets the outermost GalMod ring directly and is
honoured as given (capped only so rings stay on the image). Prefer it over
`n_scale_lengths` (which multiplies the fitted `scale_radius`, and with
freeform never truncates inside the lit map). With `rmax` unset / `"auto"`,
freeform rings follow the lit map and analytic rings cover most of the field.
`scale_radius` remains the exponential SB scale length for analytic discs —
it is not the outer radius and cannot be dropped in that case.

**ADRIFT** (`"adrift": true`): after the fit, apply the Iorio et al. (2017)
asymmetric-drift correction (same formula as BBarolo). Regularise σ² and the
face-on surface-density profile with polynomials (`adrift_pol1`,
`adrift_pol2`, default 3; set `adrift_pol1: -1` to keep raw σ²), then

```
ADC² = −R σ² (d ln σ² / dR + d ln Σ / dR)
V_circ = √(V_rot² + ADC²)
```

**LINEAR / Hanning** (`"linear": true` or a float, also `hanning`):
optional instrumental spectral broadening after GalMod, matching BBarolo's
`LINEAR` (Gaussian σ in **channels**; `true` → 0.85 ≈ Hanning with
FWHM ≈ 2 channels). **Off by default.** Only enable this for
**native-resolution** spectra that have not been channel-averaged; averaged
data already include enough spectral smoothing that an extra LINEAR term
biases `velocity_dispersion` low.

One prior per base name broadcasts to every ring; `vrot_i` is accepted as
an alias for the `vrot` broadcast. Per-ring keys (`vrot_3`, …) override.
Do not set priors on `turnover_radius` / `maximum_velocity` in this mode.
Prefer a sampler rather than L-BFGS when many rings are free. Sampler fits
plot `±1σ` VROT error bars on the major-axis PV markers.

`regtype` values: `auto` (BBarolo default), `median`, `bezier`,
`constant`, an integer polynomial degree (`0` = constant, `1` = linear),
or keyed `"inc=bezier pa=median"`.

**FitMod3D seed** (optional, not a separate likelihood): with
`rotation_curve: "rings"`, set `search.start` to `"bbarolo"` or
`"fitmod3d"`. pyuvkin runs BBarolo's FitMod3D on the naturally weighted
dirty cube (image-plane residual), writes `out/bbarolo_seed/` (ring tables,
seed cube), clips the result into your priors, and uses it as the first
L-BFGS/BFGS start. Restarts still draw from `"prior"`. The seed and ring
values are copied into `input_parameters.json` and
`best_fit_parameters.json` under `bbarolo_seed`. This is for a fair
comparison with BBarolo's ring finder while the uv-plane fit still uses
pyuvkin's likelihood; it does not replace the sampler/optimiser.

### `priors`

```json
"priors": {
  "centre_ra":   {"type": "Uniform",    "lower": -0.3, "upper": 0.3},
  "intensity":   {"type": "LogUniform", "lower": 0.1,  "upper": 50},
  "phi":         {"type": "Gaussian",   "mean": 45, "sigma": 20},
  "v_sys":       {"type": "Gaussian",   "mean": 0, "sigma": 50, "lower": -300, "upper": 300},
  "inclination": 60.0,
  "vmax_black_hole": {"type": "fixed", "value": 0}
}
```


| type          | needs                                      | notes                                                                             |
| ------------- | ------------------------------------------ | --------------------------------------------------------------------------------- |
| `Uniform`     | `lower`, `upper`                           |                                                                                   |
| `LogUniform`  | `lower > 0`, `upper`                       | for scale-free positives: `intensity`, radii                                      |
| `Gaussian`    | `mean`, `sigma`; optional `lower`, `upper` | truncated when limits are given (a truncation is also what bounds the optimisers) |
| `LogGaussian` | `mean`, `sigma`                            | no limits                                                                         |
| `fixed`       | `value`                                    | or just write the number                                                          |


Type names are case-insensitive and LensKin's spellings (`UniformPrior`,
`lower_limit`, ...) are accepted. Name typos are errors.

### `surface_brightness`


| key                  | default                      | meaning                                                                                                                                                                                                                                                                                                                                                |
| -------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `type`               | `"analytic"`                 | `analytic`: exponential disc (`intensity`, `scale_radius`). `freeform`: the line channels are collapsed in the uv-plane and imaged with pyuvimage; the primary-beam-corrected, S/N-masked Jy km/s map fixes the morphology, and only the kinematics (plus `intensity` as a rescaling, if given a prior) are fitted. `thindisk`, `kinms` and `bbarolo`. |
| `map_fits`           | `null`                       | use a ready map instead of reconstructing one: a pyuvimage `model_pbcor.fits` (Jy/pixel/channel) or a Jy km/s map. Its orientation, pixel scale and centre are checked against the data; it must be made with the same `image_centre`.                                                                                                                 |
| `uncertainty_fits`   | `null`                       | the map's 1-sigma map (pyuvimage `uncertainty.fits`); needed for the S/N mask and `morphology_samples`                                                                                                                                                                                                                                                 |
| `map_units`          | `"jy_per_pixel_per_channel"` | or `"jy_kms_per_pixel"`                                                                                                                                                                                                                                                                                                                                |
| `snr_threshold`      | `0.5`                        | mask map pixels below this many sigma                                                                                                                                                                                                                                                                                                                  |
| `pyuvimage`          | `{}`                         | extra keyword arguments to `pyuvimage.run` for the reconstruction (regularisation etc.); `dataset`, `fov`, `mode`, `out`, `image_centre` are set for you                                                                                                                                                                                               |
| `morphology_samples` | `0`                          | the map comes from the same visibilities as the kinematics, so the kinematic errors omit its uncertainty; `N > 0` re-fits N maps perturbed by their per-pixel 1-sigma (L-BFGS from the best fit) and reports the scatter as `morphology` / `errors_total_1sigma` in `best_fit_parameters.json`. An estimate, not a marginalisation.                    |


`truth` (default `null`): optional dict of true parameter values for mocks;
recorded in the outputs and compared with the fit.

## Search

```json
"search": {"method": "nautilus", "n_live": 200, "number_of_cores": 4}
"search": {"method": "dynesty",  "nlive": 100}
"search": {"method": "emcee",    "nwalkers": 40, "nsteps": 1000}
"search": {"method": "lbfgs",    "maxiter": 500, "start": "centre", "restarts": 4}
"search": {"method": "lbfgs",    "start": "bbarolo", "restarts": 4}
"search": {"method": "bfgs",     "start": {"phi": 40, "inclination": 50}}
```


| key                | default      | meaning                                                                                                                                                                                                                                                                                                                  |
| ------------------ | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `method`           | `"nautilus"` | samplers `nautilus`, `dynesty`, `dynesty_dynamic`, `emcee` give posterior samples, errors and (nested samplers) the evidence; optimisers `lbfgs`, `bfgs` give the maximum-likelihood point only, with a trace of evaluations in `samples.csv`. Optimisers are 10-100x cheaper; use them to check the setup, then sample. |
| `number_of_cores`  | `1`          | parallel likelihood evaluations for samplers only (optimisers ignore it). Env `PYUVKIN_CORES` overrides; may be an integer or `auto` (all CPUs minus one). Default 1 because NUFFT/JAX state does not fork cleanly.                                                                                                      |
| `start`            | `"prior"`    | optimisers only: `prior` (random draw), `centre` (middle of every prior), `bbarolo` / `fitmod3d` (rings + BBarolo backend only: FitMod3D seed on the dirty cube), or a dict of starting values for some/all parameters                                                                                                 |
| `restarts`         | `1`          | optimisers only: independent starts, keeping the best. For `bbarolo`/`kinms`, pyuvkin first scores `n_probe` prior draws (default `max(24, 8×restarts)`) and refines only the top `restarts`.                                                                                                                                 |
| `n_probe`          | auto         | cloud backends + optimisers: number of prior draws to score before L-BFGS; `0` disables probing. Ignored when `start` is an explicit dict.                                                                                                                                                                                  |
| `auto_correlation` | —            | emcee only: `{"check_size", "required_length", "change_threshold", "check_for_convergence"}`                                                                                                                                                                                                                             |
| anything else      |              | passed to the PyAutoFit search class: `n_live` (nautilus, default 200), `nlive` (dynesty, 100), `nwalkers`/`nsteps` (emcee, 40/1000), `maxiter`, `eps` (finite-difference step for lbfgs/bfgs; default `1e-6`, or `1` with `gtol: 1e-3` for `bbarolo`/`kinms`, and `gtol: 2.5e-3 × n_rings` when `rotation_curve` is `rings`), `gtol`, ...                       |


L-BFGS-B is bounded by the prior box, so a finite-difference step cannot
leave the prior; a parameter sitting exactly on a bound in the result means
the bound is doing the work.

## Outputs


| key               | default | meaning                                                                                                                                                                                  |
| ----------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `write_cubes`     | `true`  | FITS cubes: `dirty_data`, `dirty_model`, `dirty_residual`, `residual_snr` (natural weighting, Jy/beam) and `model_cube` (Jy/pixel), with the data WCS including the `image_centre` shift |
| `write_plots`                  | `true`  | `summary.png`, `moment_maps.png`, `channel_maps.png`, `pv_diagram.png`, and `cornerplot.png` for samplers |
| `n_plot_channels`              | `12`    | channels shown in `channel_maps.png`                                                                       |
| `moment_mask_snr`              | `5`     | bright-mask cut on dirty-model moment-0: keep pixels above this many σ₀ (and see peak fraction below)      |
| `moment_mask_peak_fraction`    | `0.1`   | also require `m0 > fraction × peak(m0)`; the mask threshold is `max(snr×σ₀, fraction×peak)`. Moment 1/2 and the aperture spectrum use this mask. |


Always written: `input_parameters.json` (the resolved settings, data
summary, geometry, priors), `best_fit_parameters.json` (maximum likelihood,
and for samplers median/1-sigma/3-sigma, evidence; chi^2 per channel;
comparison with `truth`; optional `twostage`, `bbarolo_seed`, `morphology`),
`samples.csv` (one row per sample or evaluation: parameters, log likelihood,
log prior, weight), and `pyuvkin.log` (appended each run: INFO lines including
process RSS and host available memory at major milestones). TWOSTAGE ring fits
also write `samples_stage1.csv`. Freeform SB reconstruction goes under
`surface_brightness/`; a FitMod3D seed run adds `bbarolo_seed/`.

### Logging

The CLI prints INFO to stderr; each fit also appends to `out/pyuvkin.log`
(with timestamps). At start, after model setup, each search stage, and at
the end, pyuvkin logs process RSS and host available/total memory (no extra
dependencies). Use `-v` / `--verbose` for DEBUG on stderr.

Environment variables that affect a run without editing the JSON:

| variable | default | effect |
| --- | --- | --- |
| `PYUVKIN_DFT_MAX_BYTES` | `1.5e9` | `transformer: auto` uses exact DFT matrices when all channels fit below this size |
| `PYUVKIN_CORES` | — | overrides `search.number_of_cores` for samplers (`auto` = CPUs − 1) |

## Checklist for a new dataset

1. Continuum-subtract and export a window of a few hundred km/s beyond the
  line on each side; set `spectral.rest_frequency_ghz` + `redshift`.
2. Find the source offset on a dirty image; set `image_centre` and a `fov`
  about three times the source extent.
3. Start with `thindisk` + `lbfgs` (`restarts: 4`) and broad priors; look at
  `moment_maps.png` and the chi^2/N. Tighten priors, then switch to
  `nautilus` for posteriors, or `kinms` / `bbarolo` for a thick disc,
  freeform SB, or free tilted rings (`examples/template_*.json`).

