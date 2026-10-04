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
    args = parser.parse_args(argv)
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
