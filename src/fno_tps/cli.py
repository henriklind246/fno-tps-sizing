from __future__ import annotations

import argparse
from pathlib import Path
import json

from fno_tps.config import load_study_config
from fno_tps.data import generate_dataset
from fno_tps.evaluation import evaluate_checkpoint
from fno_tps.figure_data import build_figure_data
from fno_tps.pilot import run_pilot_matrix
from fno_tps.problem import TPSProblem
from fno_tps.screening import (
    DEFAULT_HORIZONS,
    DEFAULT_INCIDENT_AMPLITUDES,
    DEFAULT_INCIDENT_HORIZONS,
    DEFAULT_INCIDENT_PULSE_WIDTHS,
    DEFAULT_PULSE_WIDTHS,
    DEFAULT_REFERENCE_AMPLITUDE,
    DEFAULT_SCREENING_THICKNESSES,
    run_incident_radiation_screen,
    run_pulse_width_screen,
)
from fno_tps.sizing import run_full_sizing
from fno_tps.training import train_model
from fno_tps.verification import run_verification, select_horizon, thin_bond_study


def _figure_numbers(value: str) -> list[int]:
    try:
        numbers = sorted({int(part) for part in value.split(",") if part.strip()})
    except ValueError:
        raise SystemExit(
            f"--figures expects a comma-separated list such as 1,2,3,4; got {value!r}."
        ) from None
    if not numbers:
        raise SystemExit("--figures requires at least one figure number.")
    return numbers


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("conf/demo.yaml"))
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Repeatable dotted configuration override.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fno-tps",
        description="FV-verified TPS sizing with time-conditioned FNOs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="Run FV verification.")
    _common(verify)
    verify.add_argument("--output", type=Path, default=Path("reports/verification.json"))
    verify.add_argument("--quick", action="store_true", help="Run the energy/interface gate only.")

    horizon = subparsers.add_parser("horizon", help="Run pilot horizon selection.")
    _common(horizon)
    horizon.add_argument("--output", type=Path, default=Path("reports/horizon.json"))
    horizon.add_argument("--pilot-cases", type=int, default=3)

    pilot = subparsers.add_parser(
        "pilot",
        help="Run the matched FV sizing-regime pilot matrix.",
    )
    _common(pilot)
    pilot.add_argument("--output", type=Path, default=Path("reports/pilot"))
    pilot.add_argument("--scenario-count", type=int, default=5)
    pilot.add_argument(
        "--backing-thicknesses",
        type=float,
        nargs="+",
        default=None,
        metavar="METERS",
        help=(
            "Optional backing-thickness sweep. The full scenario-by-TPS grid "
            "is run independently for every value."
        ),
    )

    pulse_screen = subparsers.add_parser(
        "pulse-screen",
        help=(
            "Run the legacy constant-property linear-regression screen. "
            "Nonlinear production studies must use incident-screen."
        ),
    )
    _common(pulse_screen)
    pulse_screen.add_argument(
        "--output",
        type=Path,
        default=Path("reports/pulse-screen"),
    )
    pulse_screen.add_argument(
        "--pulse-widths",
        type=float,
        nargs="+",
        default=list(DEFAULT_PULSE_WIDTHS),
        metavar="SECONDS",
    )
    pulse_screen.add_argument(
        "--horizons",
        type=float,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        metavar="SECONDS",
    )
    pulse_screen.add_argument(
        "--thicknesses",
        type=float,
        nargs="+",
        default=list(DEFAULT_SCREENING_THICKNESSES),
        metavar="METERS",
    )
    pulse_screen.add_argument(
        "--reference-amplitude",
        type=float,
        default=DEFAULT_REFERENCE_AMPLITUDE,
        metavar="W_PER_M2",
    )

    incident_screen = subparsers.add_parser(
        "incident-screen",
        help=(
            "Directly screen incident amplitudes with nonlinear hot-face "
            "reradiation."
        ),
    )
    _common(incident_screen)
    incident_screen.add_argument(
        "--output",
        type=Path,
        default=Path("reports/incident-screen"),
    )
    incident_screen.add_argument(
        "--pulse-widths",
        type=float,
        nargs="+",
        default=list(DEFAULT_INCIDENT_PULSE_WIDTHS),
        metavar="SECONDS",
    )
    incident_screen.add_argument(
        "--amplitudes",
        type=float,
        nargs="+",
        default=list(DEFAULT_INCIDENT_AMPLITUDES),
        metavar="W_PER_M2",
    )
    incident_screen.add_argument(
        "--horizons",
        type=float,
        nargs="+",
        default=list(DEFAULT_INCIDENT_HORIZONS),
        metavar="SECONDS",
    )
    incident_screen.add_argument(
        "--thicknesses",
        type=float,
        nargs="+",
        default=list(DEFAULT_SCREENING_THICKNESSES),
        metavar="METERS",
    )

    thin = subparsers.add_parser("thin-bond", help="Run the thin-bond limiting study.")
    _common(thin)
    thin.add_argument("--output", type=Path, default=Path("reports/thin_bond.json"))

    generate = subparsers.add_parser("generate", help="Generate trajectory data.")
    _common(generate)
    generate.add_argument("--output", type=Path, default=Path("data/demo"))
    generate.add_argument("--cases-per-stratum", type=int, default=None)
    generate.add_argument("--max-cases", type=int, default=None)
    generate.add_argument("--allow-demo", action="store_true")

    train = subparsers.add_parser("train", help="Train one representation.")
    _common(train)
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument(
        "--representation",
        choices=("summary", "temporal_global", "temporal_local"),
        required=True,
    )
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--device", default="auto")
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes. 0 builds inputs on the main process.",
    )

    evaluate = subparsers.add_parser("evaluate", help="Evaluate all saved targets.")
    _common(evaluate)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="test",
    )
    evaluate.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    evaluate.add_argument("--device", default="auto")

    size = subparsers.add_parser("size", help="Run the full surrogate/FV sizing matrix.")
    _common(size)
    size.add_argument("--scenarios", type=Path, required=True)
    size.add_argument("--checkpoint", type=Path, required=True)
    size.add_argument("--output", type=Path, required=True)
    size.add_argument("--allow-demo", action="store_true")
    size.add_argument("--device", default="auto")
    size.add_argument("--batch-size", type=int, default=64)

    figures = subparsers.add_parser(
        "figures",
        help="Build and render the four-figure result set.",
    )
    _common(figures)
    figures.add_argument("--output", type=Path, default=Path("reports/figures"))
    figures.add_argument(
        "--sizing",
        type=Path,
        action="append",
        default=[],
        metavar="DIR",
        help="Repeatable `fno-tps size` output directory, one per representation.",
    )
    figures.add_argument("--data", type=Path, default=None, help="Figure 1 dataset.")
    figures.add_argument("--checkpoint", type=Path, default=None)
    figures.add_argument("--scenarios", type=Path, default=None)
    figures.add_argument(
        "--figures",
        default="1,2,3,4",
        metavar="LIST",
        help="Comma-separated subset of figures to build and render.",
    )
    figures.add_argument("--case-id", default=None, help="Force the figure 1 case.")
    figures.add_argument(
        "--case-selector",
        choices=("critical", "worst-error"),
        default="critical",
    )
    figures.add_argument("--scenario-id", default=None, help="Force the figure 2 scenario.")
    figures.add_argument(
        "--primary-representation",
        choices=("summary", "temporal_global", "temporal_local"),
        default=None,
        help="Sizing run behind figures 2 and 3; defaults to the checkpoint's.",
    )
    figures.add_argument("--allow-demo", action="store_true")
    figures.add_argument(
        "--no-render",
        action="store_true",
        help="Write the artifacts without importing matplotlib.",
    )
    figures.add_argument("--device", default="auto")
    figures.add_argument("--batch-size", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_study_config(args.config, args.set)
    if args.command == "verify":
        report = run_verification(
            config,
            args.output,
            include_slow=not args.quick,
        )
    elif args.command == "horizon":
        cases = TPSProblem(config).sample_cases(cases_per_stratum=1)
        cases.sort(key=lambda case: (case.event_count, case.defect_count), reverse=True)
        report = select_horizon(config, cases[: args.pilot_cases])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    elif args.command == "pilot":
        report = run_pilot_matrix(
            config,
            args.output,
            scenario_count=args.scenario_count,
            backing_thicknesses=(
                None
                if args.backing_thicknesses is None
                else tuple(args.backing_thicknesses)
            ),
        )
    elif args.command == "pulse-screen":
        report = run_pulse_width_screen(
            config,
            args.output,
            pulse_widths=args.pulse_widths,
            horizons=args.horizons,
            thicknesses=args.thicknesses,
            reference_amplitude=args.reference_amplitude,
        )
    elif args.command == "incident-screen":
        report = run_incident_radiation_screen(
            config,
            args.output,
            pulse_widths=args.pulse_widths,
            amplitudes=args.amplitudes,
            horizons=args.horizons,
            thicknesses=args.thicknesses,
        )
    elif args.command == "thin-bond":
        report = thin_bond_study(config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    elif args.command == "generate":
        config.require_authoritative(allow_demo=args.allow_demo)
        bundle = generate_dataset(
            config,
            args.output,
            cases_per_stratum=args.cases_per_stratum,
            max_cases=args.max_cases,
        )
        report = {
            "output": str(bundle.root.resolve()),
            "case_count": len(bundle.cases),
            "shape": list(bundle.trajectories.shape),
            "validity": bundle.manifest["validity"],
        }
    elif args.command == "train":
        report = train_model(
            config,
            args.data,
            args.run_dir,
            args.representation,
            epochs=args.epochs,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    elif args.command == "evaluate":
        report = evaluate_checkpoint(
            config,
            args.data,
            args.checkpoint,
            args.output,
            split=args.split,
            device=args.device,
        )
    elif args.command == "size":
        report = run_full_sizing(
            config,
            args.scenarios,
            args.checkpoint,
            args.output,
            allow_demo=args.allow_demo,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        config.require_authoritative(allow_demo=args.allow_demo)
        numbers = _figure_numbers(args.figures)
        report = build_figure_data(
            config,
            args.output,
            sizing_dirs=args.sizing,
            data_dir=args.data,
            checkpoint_path=args.checkpoint,
            scenario_path=args.scenarios,
            case_id=args.case_id,
            case_selector=args.case_selector,
            scenario_id=args.scenario_id,
            primary_representation=args.primary_representation,
            figures=numbers,
            device=args.device,
            batch_size=args.batch_size,
        )
        if not args.no_render:
            # Imported lazily: matplotlib lives in the optional `plot` extra.
            from fno_tps.figures import render_from_directory

            report = {
                **report,
                "rendered": render_from_directory(args.output, figures=numbers),
            }
        for note in report["notes"]:
            print(f"note: {note}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
