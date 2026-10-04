"""Command-line entry point. Only environment diagnostics are implemented in S1."""

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
    args = parser.parse_args(argv)
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
