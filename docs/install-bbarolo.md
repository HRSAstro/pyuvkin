# Installing pyBBarolo for the `bbarolo` backend

pyuvkin's `bbarolo` backend uses [pyBBarolo](https://bbarolo.readthedocs.io/)
`GalMod` to render cubes in the uv-plane likelihood. Optionally,
`search.start: "bbarolo"` / `"fitmod3d"` runs FitMod3D on the dirty cube to
seed a free-ring fit (see [`settings.md`](settings.md)); that is not the main
fitter. It is not on PyPI as a pre-built wheel for every platform: you build
BBarolo from source and install the Python wrapper into the same environment
as pyuvkin.

Upstream docs: https://bbarolo.readthedocs.io/en/latest/pybb_install.html

## Dependencies

Needed libraries (Homebrew names on macOS):

- **CFITSIO** — `cfitsio`
- **FFTW3** — `fftw`
- **WCSLIB** — `wcslib`

On Apple Silicon use **arm64 Homebrew** (`/opt/homebrew`), not the Intel
prefix (`/usr/local`). Mixing an arm64 Python (`native_env`) with an x86_64
`libBBarolo.dylib` fails at import.

```bash
brew install cfitsio fftw wcslib
```

## Build

```bash
git clone -b master --single-branch https://github.com/editeodoro/Bbarolo
cd Bbarolo

./configure \
  --with-cfitsio="$(brew --prefix cfitsio)" \
  --with-fftw3="$(brew --prefix fftw)" \
  --with-wcslib="$(brew --prefix wcslib)"
```

Check the generated `Makefile` before compiling. On Homebrew Apple Silicon,
`configure` sometimes drops the CFITSIO `-L` path and points WCS includes at
the wrong directory. The lines should look like:

```make
CFITSIOINC = -I/opt/homebrew/opt/cfitsio/include
CFITSIOLIB = -L/opt/homebrew/opt/cfitsio/lib -lcfitsio
FFTW3INC   = -I/opt/homebrew/opt/fftw/include
FFTW3LIB   = -L/opt/homebrew/opt/fftw/lib -lfftw3
WCSINC     = -I/opt/homebrew/opt/wcslib/include
WCSLIB     = -L/opt/homebrew/opt/wcslib/lib -lwcs
```

If `CFITSIOLIB` is only `-lcfitsio`, add the `-L.../lib`. If `WCSINC` ends in
`include/wcslib`, change it to `include` (the sources `#include <wcslib/wcs.h>`).

Do **not** run `make clean` after configure: that target also cleans the Qt
GUI and fails when qmake is missing or points at another machine. To wipe an
old build (e.g. leftover x86_64 objects) without touching the GUI:

```bash
rm -rf src/Build
make
make lib
make pybb
```

`make` builds the `BBarolo` binary; `make lib` / `make pybb` put
`libBBarolo.dylib` into `pyBBarolo/`. Confirm architecture:

```bash
file BBarolo pyBBarolo/libBBarolo.dylib
# both should say arm64 on Apple Silicon (or x86_64 on Intel)
```

## Install into the conda env

Activate the same env you use for pyuvkin (`native_env`), then:

```bash
conda activate native_env
cd /path/to/Bbarolo
python setup.py install
```

`setup.py install` is deprecated but is what upstream ships. If the installed
copy in `site-packages/pyBBarolo` is stale (old dylib or unpatched Python),
copy from the build tree:

```bash
SP="$CONDA_PREFIX/lib/python*/site-packages/pyBBarolo"
cp -f pyBBarolo/*.py pyBBarolo/libBBarolo.dylib* pyBBarolo/libBBarolo*.a $SP/
```

### NumPy 2

Recent NumPy removed `np.int` and `np.bool`. Upstream pyBBarolo still uses
them. Add this near the top of `pyBBarolo/BB_interface.py` and
`pyBBarolo/pyBBarolo.py` (after `import numpy as np`), and use `np.intc` for
the ctypes int pointer dtype in `BB_interface.py`:

```python
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
```

```python
# in BB_interface.py type definitions:
array_1d_int = ndpointer(dtype=np.intc, ndim=1, flags="CONTIGUOUS")
```

Re-copy into `site-packages` (or reinstall) after patching.

### FITS header buffer overflow (Abort trap: 6)

BBarolo 1.5's `src/Arrays/header.cpp` uses undersized stack buffers for
CFITSIO string reads (`char comment[72]` vs `FLEN_COMMENT` 73,
`name[20]` / `Bunit[20]` vs `FLEN_VALUE` 71, `filename[100]` vs
`FLEN_FILENAME` 1025). On modern macOS this aborts with
`Abort trap: 6` / `__stack_chk_fail` as soon as GalMod opens a FITS file
(often right after the `FREQ0-RESTFREQ` / `OBJECT` header warnings).

Before building, widen those buffers to the CFITSIO `FLEN_*` constants (a
patched tree already has this if you rebuilt with the pyuvkin install notes).
In `header_read` and the other `comment[72]` sites:

```cpp
char comment[FLEN_COMMENT];
char filename[FLEN_FILENAME];
char Bunit[FLEN_VALUE], Btype[FLEN_VALUE], name[FLEN_VALUE],
     Tel[FLEN_VALUE], Dunit3[FLEN_VALUE], Keys[FLEN_CARD];
// and for CTYPE/CUNIT arrays:
Ctype[i] = new char[FLEN_VALUE];
Cunit[i] = new char[FLEN_VALUE];
```

Then `rm -rf src/Build && make && make lib && make pybb` and copy the new
`libBBarolo.dylib*` into the env's `site-packages/pyBBarolo/`.

pyuvkin also keeps GalMod templates under `/tmp/pkbb/` (short paths) and only
passes options current pyBBarolo accepts (`ltype`; not the older
`densflux` / `empty` / `outfolder` keywords).

## Check

```bash
python -c "from pyBBarolo import GalMod; print(GalMod)"
```

Then in settings (parametric disc):

```json
"model": { "backend": "bbarolo", "rotation_curve": "arctan" }
```

Free tilted rings (one `vrot_i` per ring, BBarolo-style `FREE` set) use
`rotation_curve: "rings"` and `options.n_rings` — see
[`settings.md`](settings.md) and `examples/template_bbarolo.json`.

Freeform surface brightness is supported (`NORM=LOCAL`-style rescale of each
sky pixel to the map). GalMod builds cubes from discrete clouds, so the
likelihood is piecewise and finite-difference gradients can vanish: prefer a
sampler (`nautilus`, …) for final uncertainties, or L-BFGS with the tilted-ring
defaults (`eps: 1`, `gtol: 2.5e-3 × n_rings`) and optional FitMod3D seeding.

## Quick failures

| Symptom | Likely cause |
| --- | --- |
| `Could not find the CFITSIO library` | Point `--with-cfitsio` at `$(brew --prefix cfitsio)` (`/opt/homebrew/...` on arm64) |
| `ld: library 'cfitsio' not found` | Add `-L.../cfitsio/lib` to `CFITSIOLIB` in the Makefile |
| `wcslib/wcs.h: file not found` | Set `WCSINC = -I$(brew --prefix wcslib)/include` |
| `make clean` / qmake errors under `src/GUI` | Skip GUI clean; `rm -rf src/Build` instead |
| Link ignores `.o` files / wrong arch | Old objects from another architecture; wipe `src/Build` and rebuild |
| `ImportError` / `AttributeError` on `libBBarolo.dylib` | dylib arch ≠ Python arch, or dylib not next to the installed package |
| `np.int` / `np.bool` AttributeError | Apply the NumPy 2 shims above |
| `Abort trap: 6` on `GalMod(...)` after header warnings | Undersized buffers in `header.cpp` — apply the FLEN_* patch above and rebuild |
| `Option densflux/empty/outfolder unknown` | Current pyBBarolo dropped those GalMod options; pyuvkin only sets `ltype` |