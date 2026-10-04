# Spiral-arm structured mocks

Internal notes for the `pyuvkin mock-spiral` experiment: morphology vs
kinematics, freeform surface brightness, and constant axisymmetric `vrad`.
Numbers below are for the specific mocks under `out/spiral_mocks/` and will
change if you retune noise, FOV, or priors.

## What the mocks are

Two mocks share morphology, uv coverage and noise; they differ only in the
velocity field:

| mock | surface brightness | kinematics |
| --- | --- | --- |
| `arms_sb` | 2 Sérsic components + trailing m=2 arms | circular (`thindisk` with `vrad = 0`) |
| `arms_inflow` | same | circular + constant axisymmetric inflow `vrad = −90` km/s (+outwards) |

The inflow matches the toy model used on real IFU data in Price et al. (2021),
ApJ 922, 143 (constant \(v_r \sim 90\) km/s on top of rotation). Arm-confined
radial flow was tried earlier and is **not** what `thindisk` can fit; the
mocks use disc-wide constant `vrad` so recovery is a fair test.

Truth parameters (both mocks unless noted): \(i = 55^\circ\), \(\phi = 40^\circ\),
\(v_\mathrm{max} = 250\) km/s, \(r_t = 0.25''\), \(\sigma_v = 35\) km/s;
`arms_inflow` has `vrad = −90`.

## How to run

```bash
# write datasets + settings (+ overview figure)
pyuvkin mock-spiral out/spiral_mocks

# L-BFGS analytic + freeform (+ Matérn freeform); then comparison plots
pyuvkin mock-spiral out/spiral_mocks --fit
pyuvkin mock-spiral out/spiral_mocks --plot   # rebuild plots only

# Nautilus (separate settings; does not overwrite L-BFGS outs)
pyuvkin fit out/spiral_mocks/arms_inflow/settings_freeform_nautilus.json
# … likewise settings_analytic_nautilus.json, arms_sb/…
```

Each mock directory holds `dataset/`, `truth.json`, `truth_sb.fits`, and
`settings_*.json`. Comparison figures land in `out/spiral_mocks/`:
`mocks_overview.png`, `fit_parameters.png`, `fit_bias.png`, `fit_maps.png`.

Freeform variants:

- `settings_freeform.json` — pyuvimage default (adaptive) regularisation
- `settings_freeform_matern.json` — Matérn (`pyuvimage.reg: matern`)
- `settings_*_nautilus.json` — same models with `search.method: nautilus`

On a laptop, prefer `"number_of_cores": 1` for Nautilus: each worker rebuilds
the ~1.4 GB DFT, so 4 cores can exhaust ~17 GB RAM even when a single process
only needs ~2 GB.

## Geometry bias from the arms alone

Moment-0 second moments of the map (no kinematics) are pulled by the arms by
roughly ~5° in inclination and ~9° in position angle relative to truth. The
`arms_sb` velocity field is still exactly circular at the true \(i\), \(\phi\).
An analytic SB fit has to reconcile skewed light with correct rotation;
freeform takes the map and does not.

## L-BFGS results (`thindisk`)

| | χ²/N | inclination | phi | \(v_\mathrm{max}\) | \(v_\mathrm{max}\sin i\) | vrad |
| --- | --- | --- | --- | --- | --- | --- |
| truth (`arms_sb`) | | 55.0 | 40.0 | 250 | 204.8 | 0 |
| truth (`arms_inflow`) | | 55.0 | 40.0 | 250 | 204.8 | −90 |
| `arms_sb`, analytic | 1.875 | 64.4 | 37.5 | 231 | 208.7 | (fixed 0) |
| `arms_sb`, freeform | 1.018 | 48.5 | 41.3 | 279 | 208.8 | (fixed 0) |
| `arms_inflow`, analytic | 1.830 | 64.4 | 34.9 | 242 | 218.7 | −64.0 |
| `arms_inflow`, freeform | 1.017 | 55.4 | 39.1 | 247 | 203.1 | −87.2 |

Matérn freeform tracks default freeform closely (e.g. `arms_inflow` `vrad`
≈ −87.5 vs −87.2).

## Nautilus results (posterior median ± 1σ)

| | med \(i\) | med \(\phi\) | med \(v_\mathrm{max}\) | med vrad |
| --- | --- | --- | --- | --- |
| `arms_sb`, analytic | 64.53 ± 0.11 | 37.45 ± 0.07 | 231.2 ± 0.6 | (fixed 0) |
| `arms_sb`, freeform | 54.34 ± 0.30 | 40.04 ± 0.12 | 251.1 ± 1.2 | (fixed 0) |
| `arms_inflow`, analytic | 64.45 ± 0.11 | 34.89 ± 0.11 | 242.5 ± 0.6 | −63.7 ± 0.4 |
| `arms_inflow`, freeform | 55.35 ± 0.25 | 39.18 ± 0.22 | 246.5 ± 1.1 | **−87.6 ± 0.7** |

Formal errors are tiny next to surface-brightness-driven biases. Freeform
recovers the constant inflow; analytic undershoots `vrad` while locking onto
the usual high-inclination / wrong-φ compromise.

Rebuild the comparison plots from Nautilus products (median ± 1σ):

```python
from pyuvkin.mock_structured import compare_structured_fits
compare_structured_fits("out/spiral_mocks", fit_suffix="_nautilus")
```

## Takeaways

1. Spiral arms in the light alone bias analytic kinematic fits; freeform SB
   largely removes that for circular discs.
2. Constant axisymmetric `vrad` is recoverable with freeform once morphology
   is fixed; analytic still trades `vrad` against \(i\) / \(\phi\).
3. Arm-tied non-circular flow is a stress test of model misspecification, not
   a recovery test for the current `thindisk` `vrad` parameter.
