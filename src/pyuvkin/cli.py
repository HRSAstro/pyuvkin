"""Command line.

    pyuvkin fit settings.json [--out DIR] [--method lbfgs]
    pyuvkin template settings.json          # every setting at its default
    pyuvkin import MS OUT [...]             # pyuvimage's importer, unchanged
    pyuvkin mock OUT [--backend kinms ...]  # a mock dataset + truth.json
    pyuvkin demo [OUT] [--method nautilus]  # mock, fit, products
"""

from __future__ import annotations

import argparse
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
    p_mock.add_argument("--sigma", type=float, default=5e-4, help="per-visibility noise, Jy")
    p_mock.add_argument("--fov", type=float, default=3.0)
    p_mock.add_argument("--seed", type=int, default=0)
    p_mock.add_argument("--truth", help="JSON file or string of disc parameters to use")

    p_demo = sub.add_parser("demo", help="mock a disc, fit it, write every product")
    p_demo.add_argument("out", nargs="?", default="pyuvkin_demo")
    p_demo.add_argument("--method", default="lbfgs")
    p_demo.add_argument("--backend", default="thindisk", help="backend used for the fit")
    p_demo.add_argument("--freeform", action="store_true", help="fit with a freeform surface brightness")
    p_demo.add_argument("--n-vis", type=int, default=1500)
    p_demo.add_argument("--n-chan", type=int, default=20)
    p_demo.add_argument("--sigma", type=float, default=5e-4)

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

    if args.command == "demo":
        from .api import run
        from .mock import demo_settings, write_mock_dataset

        out = Path(args.out)
        ds, tp = write_mock_dataset(
            out / "mock", None, n_vis=args.n_vis, n_chan=args.n_chan, sigma_jy=args.sigma,
        )
        truth = json.loads(tp.read_text())
        settings = demo_settings(ds, truth, out / "fit", method=args.method)
        settings["model"]["backend"] = args.backend
        if args.freeform:
            settings["surface_brightness"] = {"type": "freeform"}
            for name in ("intensity", "scale_radius"):
                settings["priors"].pop(name, None)
        # Cloud backends have a stepwise likelihood: seed L-BFGS from a cheap
        # smooth thindisk fit so we land in the right basin.
        if (
            args.backend in ("bbarolo", "kinms")
            and str(args.method).lower() in ("lbfgs", "bfgs")
        ):
            seed_out = out / "fit_thindisk_seed"
            seed_settings = {
                **settings,
                "out": str(seed_out),
                "model": {"backend": "thindisk", "rotation_curve": settings["model"].get("rotation_curve", "arctan")},
                "search": {"method": "lbfgs", "start": "centre", "restarts": 2, "maxiter": 200},
                "write_cubes": False,
                "write_plots": False,
            }
            print(f"seeding {args.backend} from a quick thindisk L-BFGS -> {seed_out}")
            seed = run(seed_settings)
            settings["search"]["start"] = {
                n: float(seed.best_fit_record["max_log_likelihood"][n])
                for n in seed.best_fit_record["free_parameters"]
            }
            settings["search"]["restarts"] = 1
        (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
        result = run(settings)
        _print_best(result)
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
