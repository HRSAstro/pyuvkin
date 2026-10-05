"""Command line.

    pyuvkin fit settings.json [--out DIR] [--method lbfgs]
    pyuvkin template settings.json          # every setting at its default
    pyuvkin import MS OUT [...]             # pyuvimage's importer, unchanged
    pyuvkin mock OUT [--backend kinms ...]  # a mock dataset + truth.json
    pyuvkin mock-spiral [OUT]               # mocks with 2 Sersic + 2 spiral arms
    pyuvkin demo [OUT] [--method lbfgs]     # mock + analytic and freeform fits
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

from . import __version__


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("pyuvkin").setLevel(logging.DEBUG if verbose else logging.INFO)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pyuvkin",
        description="Kinematic modelling of interferometric line data in the uv-plane.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"pyuvkin {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fit = sub.add_parser("fit", help="fit a kinematic model to a dataset")
    p_fit.add_argument("settings", help="JSON settings file (see `pyuvkin template`)")
    p_fit.add_argument("--out", help="override the output directory")
    p_fit.add_argument("--method", help="override search.method (nautilus, dynesty, emcee, lbfgs, ...)")
    p_fit.add_argument("--backend", help="override model.backend")
    p_fit.add_argument("--cores", type=int, help="override search.number_of_cores")

    p_tpl = sub.add_parser("template", help="write a settings file with every default")
    p_tpl.add_argument("path", nargs="?", default="pyuvkin-settings.json")

    p_imp = sub.add_parser(
        "import", help="convert a CASA MS to a dataset (delegates to pyuvimage import)",
        add_help=False,
    )
    p_imp.add_argument("rest", nargs=argparse.REMAINDER)

    p_mock = sub.add_parser("mock", help="write a mock line dataset and its truth")
    p_mock.add_argument("out")
    p_mock.add_argument("--backend", default="thindisk")
    p_mock.add_argument("--n-vis", type=int, default=2000)
    p_mock.add_argument("--n-chan", type=int, default=24)
    p_mock.add_argument("--dv", type=float, default=30.0, help="channel width, km/s")
    p_mock.add_argument("--sigma", type=float, default=2e-3, help="per-visibility noise, Jy")
    p_mock.add_argument("--fov", type=float, default=3.0)
    p_mock.add_argument("--seed", type=int, default=0)
    p_mock.add_argument("--truth", help="JSON file or string of disc parameters to use")

    p_spiral = sub.add_parser(
        "mock-spiral",
        help="write the structured mocks (2 Sersic + 2 spiral arms) used to compare "
             "analytic and freeform surface brightness",
    )
    p_spiral.add_argument("out", nargs="?", default="pyuvkin_spiral_mocks")
    p_spiral.add_argument("--method", default="lbfgs", help="search.method in the fit settings")
    p_spiral.add_argument("--backend", default="thindisk", help="model.backend in the fit settings")
    p_spiral.add_argument("--n-vis", type=int, default=2000)
    p_spiral.add_argument("--n-chan", type=int, default=16)
    p_spiral.add_argument("--dv", type=float, default=40.0, help="channel width, km/s")
    p_spiral.add_argument("--sigma", type=float, default=6e-4, help="per-visibility noise, Jy")
    p_spiral.add_argument("--fov", type=float, default=3.0)
    p_spiral.add_argument("--seed", type=int, default=0)
    p_spiral.add_argument(
        "--fit", action="store_true",
        help="also fit every mock with both surface-brightness models and plot the comparison",
    )
    p_spiral.add_argument(
        "--plot", action="store_true",
        help="rebuild the comparison plots from finished fits (no new fitting)",
    )
    p_spiral.add_argument(
        "--refit", action="store_true",
        help="with --fit, redo fits even if best_fit_parameters.json already exists",
    )

    p_demo = sub.add_parser(
        "demo",
        help="mock a disc, then fit it with analytic and freeform surface brightness",
    )
    p_demo.add_argument("out", nargs="?", default="pyuvkin_demo")
    p_demo.add_argument("--method", default="lbfgs")
    p_demo.add_argument("--backend", default="thindisk", help="backend used for the fit")
    p_demo.add_argument(
        "--freeform", action="store_true",
        help="(deprecated) freeform-only; demo always runs both analytic and freeform",
    )
    p_demo.add_argument("--n-vis", type=int, default=1500)
    p_demo.add_argument("--n-chan", type=int, default=20)
    p_demo.add_argument("--sigma", type=float, default=2e-3, help="per-visibility noise, Jy")

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if args.command == "template":
        from .config import write_template

        write_template(args.path)
        print(f"wrote {args.path}")
        return 0

    if args.command == "import":
        from pyuvimage.cli import main as pv_main

        return int(pv_main(["import", *args.rest]) or 0)

    if args.command == "fit":
        from .api import run

        overrides = {}
        if args.out:
            overrides["out"] = args.out
        settings_path = Path(args.settings)
        given = json.loads(settings_path.read_text())
        if args.method:
            given.setdefault("search", {})["method"] = args.method
        if args.cores is not None:
            given.setdefault("search", {})["number_of_cores"] = args.cores
        if args.backend:
            given.setdefault("model", {})["backend"] = args.backend
        from . import config

        settings = config.load_settings(given, base_dir=settings_path.parent, **overrides)
        result = run(settings)
        _print_best(result)
        return 0

    if args.command == "mock":
        from .mock import write_mock_dataset

        truth = None
        if args.truth:
            t = Path(args.truth)
            truth = json.loads(t.read_text() if t.exists() else args.truth)
        ds, tp = write_mock_dataset(
            args.out, truth, backend=args.backend, n_vis=args.n_vis, n_chan=args.n_chan,
            dv_kms=args.dv, sigma_jy=args.sigma, fov=args.fov, seed=args.seed,
        )
        print(f"wrote {ds} and {tp}")
        return 0

    if args.command == "mock-spiral":
        from .mock_structured import (
            MOCKS, compare_structured_fits, fit_structured_mocks, write_structured_mocks,
        )

        out = Path(args.out)
        mocks_ready = all((out / name / "dataset").exists() for name in MOCKS)

        # --plot alone: rebuild figures from finished fits, leave the mocks alone
        if args.plot and not args.fit:
            for name, path in compare_structured_fits(out).items():
                print(f"wrote {name} -> {path}")
            return 0

        # don't wipe finished fits by rewriting the mocks underneath them
        if mocks_ready and args.fit and not args.refit:
            print(f"keeping existing mocks under {out}")
        else:
            written = write_structured_mocks(
                out, method=args.method, backend=args.backend, n_vis=args.n_vis,
                n_chan=args.n_chan, dv_kms=args.dv, sigma_jy=args.sigma, fov=args.fov,
                seed=args.seed,
            )
            overview = written.pop("overview", None)
            for name, d in written.items():
                print(f"wrote {name} -> {d}")
            if overview:
                print(f"wrote overview -> {overview}")

        if args.fit:
            for name, path in fit_structured_mocks(out, refit=args.refit).items():
                print(f"wrote {name} -> {path}")
            return 0

        print("\nfit each with, for example:")
        for name in MOCKS:
            for sb in ("analytic", "freeform"):
                print(f"  pyuvkin fit {out / name / f'settings_{sb}.json'}")
        print(f"\nor: pyuvkin mock-spiral {out} --fit")
        print(f"then: pyuvkin mock-spiral {out} --plot")
        return 0

    if args.command == "demo":
        from .api import run
        from .mock import demo_settings, write_mock_dataset

        if args.freeform:
            logging.getLogger("pyuvkin").warning(
                "--freeform is deprecated: demo always fits analytic and freeform SB",
            )

        out = Path(args.out)
        ds, tp = write_mock_dataset(
            out / "mock", None, n_vis=args.n_vis, n_chan=args.n_chan, sigma_jy=args.sigma,
        )
        truth = json.loads(tp.read_text())
        base = demo_settings(ds, truth, out / "fit_analytic", method=args.method)
        base["model"]["backend"] = args.backend

        # Cloud backends have a stepwise likelihood: seed L-BFGS from a cheap
        # smooth thindisk fit so both SB modes land in the right basin.
        seed_start = None
        if (
            args.backend in ("bbarolo", "kinms")
            and str(args.method).lower() in ("lbfgs", "bfgs")
        ):
            seed_out = out / "fit_thindisk_seed"
            seed_settings = {
                **base,
                "out": str(seed_out),
                "model": {
                    "backend": "thindisk",
                    "rotation_curve": base["model"].get("rotation_curve", "arctan"),
                    "dispersion_curve": base["model"].get("dispersion_curve", "constant"),
                },
                "search": {"method": "lbfgs", "start": "centre", "restarts": 2, "maxiter": 200},
                "write_cubes": False,
                "write_plots": False,
            }
            print(f"seeding {args.backend} from a quick thindisk L-BFGS -> {seed_out}")
            seed = run(seed_settings)
            seed_start = {
                n: float(seed.best_fit_record["max_log_likelihood"][n])
                for n in seed.best_fit_record["free_parameters"]
            }

        modes = (
            ("analytic", out / "fit_analytic", {"type": "analytic"}),
            ("freeform", out / "fit_freeform", {"type": "freeform"}),
        )
        for label, fit_out, sb in modes:
            settings = copy.deepcopy(base)
            settings["out"] = str(fit_out)
            settings["surface_brightness"] = dict(sb)
            if label == "freeform":
                for name in ("intensity", "scale_radius"):
                    settings["priors"].pop(name, None)
            if seed_start is not None:
                # freeform drops intensity/scale_radius from the seed
                settings["search"]["start"] = {
                    k: v for k, v in seed_start.items() if k in settings["priors"]
                }
                settings["search"]["restarts"] = 1
            (out / f"settings_{label}.json").write_text(json.dumps(settings, indent=2) + "\n")
            print(f"\n=== {label} surface brightness -> {fit_out} ===")
            result = run(settings)
            _print_best(result)

        # convenience pointer used by older scripts / docs
        (out / "settings.json").write_text(
            (out / "settings_analytic.json").read_text()
        )
        return 0

    parser.error(f"unknown command {args.command}")
    return 2


def _print_best(result) -> None:
    rec = result.best_fit_record
    print(f"\nchi^2/N = {rec['fit_quality']['chi_squared_reduced']:.4f}")
    for name in rec["free_parameters"]:
        line = f"  {name:20s} {rec['max_log_likelihood'][name]:12.5g}"
        if "median_pdf" in rec:
            lo, hi = rec["errors_1sigma"][name]
            line += f"   median {rec['median_pdf'][name]:.5g} -{lo:.3g} +{hi:.3g}"
        tc = rec.get("truth_comparison", {}).get(name)
        if tc:
            line += f"   truth {tc['truth']:.5g}"
        print(line)
    print("outputs:", ", ".join(sorted(Path(v).name for v in result.written.values())))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
