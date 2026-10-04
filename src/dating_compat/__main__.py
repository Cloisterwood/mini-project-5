"""Command-line entry point for diagnostics and dataset auditing."""

import argparse
import json
from pathlib import Path

from dating_compat.doctor import diagnose


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dating-compat")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check dependencies and a real tensor operation")
    doctor.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    doctor.add_argument("--output", type=Path, help="Save the diagnostic report as JSON")
    for name in ("acquire", "explore"):
        command = commands.add_parser(name, help="Acquire pinned data" if name == "acquire" else "Audit and split data offline")
        command.add_argument("--data-dir", type=Path, default=Path("data"))
        command.add_argument("--reports-dir", type=Path, default=Path("reports/quarantine-v1") if name == "explore" else Path("reports"))
        if name == "explore":
            command.add_argument("--quarantine-policy", type=Path, default=Path("configs/quarantine-v1.json"))
    quality = commands.add_parser("quality", help="Prepare human review and compute the frozen quality gate")
    quality.add_argument("--data-dir", type=Path, default=Path("data"))
    quality.add_argument("--output", type=Path, default=Path("reports/quality_report.json"))
    quality.add_argument("--responses", type=Path, default=Path("docs/human_review_responses.csv"))
    review = commands.add_parser("review", help="Guided terminal entry of human ratings; saves and resumes per pair")
    review.add_argument("--data-dir", type=Path, default=Path("data"))
    review.add_argument("--responses", type=Path, default=Path("docs/human_review_responses.csv"))
    review.add_argument("--pair", type=int, help="Revisit a specific sample number (1–20)")
    baseline_parser = commands.add_parser("baseline", help="Run or verify the frozen, gated base-model baseline")
    baseline_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    commands.add_parser("prepare-training", help="Separate governance preflight exports fit/validation inputs")
    smoke_parser = commands.add_parser("train-smoke", help="Non-scored local checkpoint/resume and speed check")
    smoke_parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    train_parser = commands.add_parser("train", help="Run one authorized fit/validation-only experiment")
    train_parser.add_argument("--run-id", required=True)
    train_parser.add_argument("--config", type=Path, default=Path("configs/train-default.yaml"))
    train_parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    train_parser.add_argument("--resume", type=Path)
    validation_parser = commands.add_parser("validate-run", help="Compare completed selected checkpoint on validation only")
    validation_parser.add_argument("--run-id", required=True)
    commands.add_parser("select-run", help="Freeze the final validation-only choice after all three experiments")
    args = parser.parse_args(argv)
    if args.command == "select-run":
        from dating_compat.data import DataError
        from dating_compat.selection import freeze_selection
        try:
            selected = freeze_selection()
        except (DataError, OSError) as exc:
            print(f"SELECTION ERROR: {exc}")
            return 1
        print(f"SELECTION FROZEN: {selected['selected_run']}")
        return 0
    if args.command == "validate-run":
        from dating_compat.data import DataError
        from dating_compat.validation import validate_run
        try:
            validate_run(args.run_id)
        except (DataError, OSError) as exc:
            print(f"VALIDATION ERROR: {exc}")
            return 1
        print("VALIDATION RUN PASS")
        return 0
    if args.command in {"prepare-training", "train-smoke", "train"}:
        from dating_compat.data import DataError
        try:
            if args.command == "prepare-training":
                from dating_compat.training_data import prepare_training
                prepare_training()
                print("TRAINING PREFLIGHT PASS")
            else:
                from dating_compat.train import train_run, train_smoke
                if args.command == "train-smoke":
                    train_smoke(args.device)
                    print("TRAINING SMOKE PASS")
                else:
                    train_run(args.run_id, args.config, args.device, args.resume)
                    print("TRAINING RUN COMPLETE")
        except (DataError, OSError) as exc:
            print(f"TRAINING ERROR: {exc}")
            return 1
        return 0
    if args.command == "baseline":
        from dating_compat.data import DataError
        from dating_compat.evaluate import baseline
        try:
            baseline(args.data_dir)
        except (DataError, OSError) as exc:
            print(f"BASELINE ERROR: {exc}")
            return 1
        print("BASELINE PASS")
        return 0
    if args.command == "review":
        from dating_compat.data import DataError
        from dating_compat.review_guide import guided_review
        try:
            return guided_review(args.data_dir, args.responses, args.pair)
        except (DataError, OSError) as exc:
            print(f"REVIEW ERROR: {exc}")
            return 1
    if args.command == "quality":
        from dating_compat.data import DataError
        from dating_compat.quality import quality_report
        try:
            report = quality_report(args.data_dir, args.output, args.responses)
        except (DataError, OSError) as exc:
            print(f"QUALITY ERROR: {exc}")
            return 1
        status = report["gate"]["status"]
        print("QUALITY GATE PASS" if status == "passed" else f"QUALITY GATE {status.upper()}")
        if report["gate"]["failed_dimensions"]:
            print("Failed dimensions: " + ", ".join(report["gate"]["failed_dimensions"]))
        return 0 if status == "passed" else 1
    if args.command != "doctor":
        from dating_compat.data import DataError, acquire
        try:
            if args.command == "acquire":
                acquire(args.data_dir, args.reports_dir)
                print("ACQUISITION PASS")
                return 0
            from dating_compat.explore import explore
            report = explore(args.data_dir, args.reports_dir, args.quarantine_policy)
            if report["training_blockers"]:
                print("DATA GATE BLOCKED: " + "; ".join(report["training_blockers"]))
                return 1
            print("DATA EXPLORATION PASS")
            return 0
        except (DataError, OSError) as exc:
            print(f"DATA ERROR: {exc}")
            return 1
    report = diagnose(args.device)
    rendered = json.dumps(report, indent=2, allow_nan=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print("ENVIRONMENT PASS" if report["passed"] else "ENVIRONMENT FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
