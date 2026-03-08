from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

from .connectors import build_duration_request_from_rows, build_payload_from_rows, read_table
from .engine import analyze, parse_payload
from .planning import bayesian_duration_conversion, frequentist_duration_conversion
from .reporting import build_markdown_report
from .text_parser import parse_duration_prompt, parse_variant_lines


def build_parser() -> argparse.ArgumentParser:
    epilog = (
        "Examples:\n"
        "  bayestest analyze --input input.json --output output.json --report report.md\n"
        "  bayestest analyze-file --input data.csv --mapping mapping.json\n"
        "  bayestest analyze-file --input data.xlsx --mapping mapping.json --sheet Sheet1\n"
        "  bayestest analyze-text --text \"Variant A: 100 conversions out of 2000 visitors\\n"
        "Variant B: 125 conversions out of 2000 visitors\"\n"
        "  bayestest duration --prompt-text \"Traffic: 50000 visitors/day\\nBaseline: 4%\\nMDE: 5%\\nLooks: 10\"\n"
        "  bayestest duration --input duration_inputs.xlsx --mapping duration_mapping.json --sheet Sheet1\n"
    )
    parser = argparse.ArgumentParser(
        prog="bayestest",
        description=(
            "Agent-friendly CLI for Bayesian and frequentist (sequential or fixed-horizon) A/B/n decisions."
        ),
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    analyze_cmd = sub.add_parser(
        "analyze",
        help="Analyze one JSON experiment payload.",
        description="Run Bayesian or frequentist analysis (sequential or fixed-horizon) from a structured JSON payload.",
    )
    analyze_cmd.add_argument("--input", required=True, help="Path to input JSON payload.")
    analyze_cmd.add_argument(
        "--output",
        required=False,
        help="Path to output JSON. If omitted, writes to stdout.",
    )
    analyze_cmd.add_argument(
        "--report",
        required=False,
        help="Optional path to markdown report output.",
    )

    sub.add_parser("example-input", help="Print an example input JSON to stdout.")
    sub.add_parser("example-mapping", help="Print an example mapping JSON for CSV/XLSX.")
    sub.add_parser("example-duration-prompt", help="Print a duration prompt text example.")
    sub.add_parser("example-duration-mapping", help="Print a duration mapping JSON for CSV/XLSX.")

    analyze_file_cmd = sub.add_parser(
        "analyze-file",
        help="Analyze CSV/XLSX using a mapping file.",
        description="Use mapping JSON to normalize arbitrary CSV/XLSX columns into the bayestest contract.",
    )
    analyze_file_cmd.add_argument("--input", required=True, help="Path to CSV/XLSX.")
    analyze_file_cmd.add_argument("--mapping", required=True, help="Path to mapping JSON.")
    analyze_file_cmd.add_argument("--sheet", required=False, help="Excel sheet name.")
    analyze_file_cmd.add_argument(
        "--output",
        required=False,
        help="Path to output JSON. If omitted, writes to stdout.",
    )
    analyze_file_cmd.add_argument(
        "--report",
        required=False,
        help="Optional path to markdown report output.",
    )

    analyze_text_cmd = sub.add_parser(
        "analyze-text",
        help="Parse pasted variant lines and analyze.",
        description="Conversation mode: parse plain-text variant stats and run analysis.",
    )
    analyze_text_cmd.add_argument("--text", required=False, help="Raw pasted text.")
    analyze_text_cmd.add_argument("--text-file", required=False, help="Path to pasted text file.")
    analyze_text_cmd.add_argument("--experiment-name", default="text_input_experiment")
    analyze_text_cmd.add_argument("--method", default="bayesian")
    analyze_text_cmd.add_argument("--primary-metric", default="conversion_rate")
    analyze_text_cmd.add_argument("--output", required=False)
    analyze_text_cmd.add_argument("--report", required=False)

    duration_cmd = sub.add_parser(
        "duration",
        help="Estimate test duration from assumptions.",
        description=(
            "Estimate experiment runtime from assumptions.\n"
            "Supports direct args, --prompt-text, and CSV/XLSX + mapping."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    duration_cmd.add_argument("--method", choices=["frequentist", "bayesian"], required=False)
    duration_cmd.add_argument("--baseline-rate", type=float, required=False, help="Baseline conversion rate, decimal (e.g. 0.04)")
    duration_cmd.add_argument("--relative-mde", type=float, required=False, help="Relative MDE, decimal (e.g. 0.05 for +5%%)")
    duration_cmd.add_argument("--daily-traffic", type=int, required=False, help="Total daily traffic across variants")
    duration_cmd.add_argument("--n-variants", type=int, default=2)
    duration_cmd.add_argument("--alpha", type=float, default=0.05)
    duration_cmd.add_argument("--power", type=float, default=0.8)
    duration_cmd.add_argument("--max-looks", type=int, default=10)
    duration_cmd.add_argument("--prob-threshold", type=float, default=0.95)
    duration_cmd.add_argument("--max-expected-loss", type=float, default=0.001)
    duration_cmd.add_argument("--assurance-target", type=float, default=0.8)
    duration_cmd.add_argument("--max-days", type=int, default=60)
    duration_cmd.add_argument("--output", required=False)
    duration_cmd.add_argument("--prompt-text", required=False, help="Natural-language assumptions text")
    duration_cmd.add_argument("--input", required=False, help="CSV/XLSX input for duration assumptions")
    duration_cmd.add_argument("--mapping", required=False, help="JSON mapping for duration table columns")
    duration_cmd.add_argument("--sheet", required=False, help="Excel sheet name for duration input")

    doctor_cmd = sub.add_parser(
        "doctor",
        help="Run environment and dependency checks for agents/CI.",
        description="Check Python version, required dependencies, and CLI readiness.",
    )
    doctor_cmd.add_argument("--strict", action="store_true", help="Exit non-zero if any check fails.")
    doctor_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")

    return parser


def example_payload() -> dict:
    return {
        "experiment_name": "checkout_button_color",
        "method": "bayesian",
        "primary_metric": "conversion_rate",
        "alpha": 0.05,
        "look_index": 3,
        "max_looks": 10,
        "variants": [
            {
                "name": "control",
                "visitors": 100000,
                "conversions": 4000,
                "is_control": True,
            },
            {
                "name": "treatment_a",
                "visitors": 100000,
                "conversions": 4200,
                "is_control": False,
            },
        ],
        "guardrails": [
            {
                "name": "bounce_rate",
                "control": 0.36,
                "treatment": 0.365,
                "direction": "decrease",
                "max_relative_change": 0.03,
            },
            {
                "name": "p95_latency_ms",
                "control": 420.0,
                "treatment": 430.0,
                "direction": "decrease",
                "max_relative_change": 0.05,
            },
        ],
        "decision_thresholds": {
            "bayes_prob_beats_control": 0.95,
            "max_expected_loss": 0.001,
        },
        "samples": 50000,
        "random_seed": 7,
    }


def example_mapping() -> dict:
    return {
        "experiment_name": "checkout_button_color",
        "method": "bayesian",
        "primary_metric": "conversion_rate",
        "alpha": 0.05,
        "look_index": 3,
        "max_looks": 10,
        "columns": {
            "variant": "variant_name",
            "visitors": "users",
            "conversions": "orders",
            "is_control": "is_control",
            "revenue_sum": "revenue_sum",
            "revenue_sum_squares": "revenue_sum_squares"
        },
        "control": {
            "column": "variant_name",
            "value": "control"
        },
        "guardrails": [
            {
                "name": "p95_latency_ms",
                "control": 420.0,
                "treatment": 430.0,
                "direction": "decrease",
                "max_relative_change": 0.05
            }
        ],
        "decision_thresholds": {
            "bayes_prob_beats_control": 0.95,
            "max_expected_loss": 0.001
        },
        "samples": 50000,
        "random_seed": 7
    }


def example_duration_prompt() -> str:
    return (
        "Traffic: 50000 visitors/day\n"
        "Baseline: 4%\n"
        "MDE: 5%\n"
        "Alpha: 0.05\n"
        "Power: 0.8\n"
        "Variants: 3\n"
        "Looks: 10\n"
    )


def example_duration_mapping() -> dict:
    return {
        "method": "frequentist",
        "columns": {
            "method": "method",
            "baseline_rate": "baseline_rate",
            "relative_mde": "relative_mde",
            "daily_traffic": "daily_traffic",
            "n_variants": "n_variants",
            "alpha": "alpha",
            "power": "power",
            "max_looks": "max_looks",
            "prob_threshold": "prob_threshold",
            "max_expected_loss": "max_expected_loss",
            "assurance_target": "assurance_target",
            "max_days": "max_days",
        },
    }

def _write_text(path: str, content: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")


def run_doctor() -> dict:
    checks = []

    py_ok = sys.version_info >= (3, 9)
    checks.append(
        {
            "name": "python_version",
            "passed": py_ok,
            "detail": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "required": ">=3.9",
        }
    )

    for dep, required in [("numpy", ">=1.22"), ("openpyxl", ">=3.1.0")]:
        try:
            version = importlib.metadata.version(dep)
            checks.append(
                {
                    "name": f"dependency_{dep}",
                    "passed": True,
                    "detail": version,
                    "required": required,
                }
            )
        except importlib.metadata.PackageNotFoundError:
            checks.append(
                {
                    "name": f"dependency_{dep}",
                    "passed": False,
                    "detail": "not installed",
                    "required": required,
                }
            )

    all_passed = all(c["passed"] for c in checks)
    return {"ok": all_passed, "checks": checks}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "example-input":
        print(json.dumps(example_payload(), indent=2))
        return 0
    if args.command == "example-mapping":
        print(json.dumps(example_mapping(), indent=2))
        return 0
    if args.command == "example-duration-prompt":
        print(example_duration_prompt())
        return 0
    if args.command == "example-duration-mapping":
        print(json.dumps(example_duration_mapping(), indent=2))
        return 0

    if args.command == "analyze-text":
        text = args.text
        if not text and args.text_file:
            text = Path(args.text_file).read_text(encoding="utf-8")
        if not text:
            raise ValueError("Provide --text or --text-file.")
        variants = parse_variant_lines(text)
        payload = {
            "experiment_name": args.experiment_name,
            "method": args.method,
            "primary_metric": args.primary_metric,
            "variants": variants,
        }
        analysis_input = parse_payload(payload)
        result = analyze(analysis_input)
        result_json = json.dumps(result.to_dict(), indent=2)
        if args.output:
            _write_text(args.output, result_json + "\n")
        else:
            print(result_json)
        if args.report:
            _write_text(args.report, build_markdown_report(result))
        return 0

    if args.command == "duration":
        method = args.method
        baseline_rate = args.baseline_rate
        relative_mde = args.relative_mde
        daily_traffic = args.daily_traffic

        if args.input and args.mapping:
            rows = read_table(args.input, sheet=args.sheet)
            duration_mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
            file_req = build_duration_request_from_rows(rows, duration_mapping)
            method = method or file_req["method"]
            baseline_rate = baseline_rate if baseline_rate is not None else file_req["baseline_rate"]
            relative_mde = relative_mde if relative_mde is not None else file_req["relative_mde"]
            daily_traffic = daily_traffic if daily_traffic is not None else file_req["daily_traffic"]
            args.n_variants = file_req["n_variants"]
            args.alpha = file_req["alpha"]
            args.power = file_req["power"]
            args.max_looks = file_req["max_looks"]
            args.prob_threshold = file_req["prob_threshold"]
            args.max_expected_loss = file_req["max_expected_loss"]
            args.assurance_target = file_req["assurance_target"]
            args.max_days = file_req["max_days"]

        if args.prompt_text:
            parsed = parse_duration_prompt(args.prompt_text)
            method = method or "frequentist"
            baseline_rate = baseline_rate if baseline_rate is not None else parsed["baseline_rate"]
            relative_mde = relative_mde if relative_mde is not None else parsed["relative_mde"]
            daily_traffic = daily_traffic if daily_traffic is not None else parsed["daily_total_traffic"]
            args.alpha = args.alpha if args.alpha is not None else parsed["alpha"]
            args.power = args.power if args.power is not None else parsed["power"]
            args.n_variants = args.n_variants if args.n_variants is not None else parsed["n_variants"]
            args.max_looks = args.max_looks if args.max_looks is not None else parsed["max_looks"]

        if not method:
            method = input("Method (frequentist/bayesian): ").strip().lower()
        if baseline_rate is None:
            baseline_rate = float(input("Baseline conversion rate (decimal, e.g. 0.04): ").strip())
        if relative_mde is None:
            relative_mde = float(input("Relative MDE (decimal, e.g. 0.05): ").strip())
        if daily_traffic is None:
            daily_traffic = int(input("Total daily traffic: ").strip())

        if method == "frequentist":
            plan = frequentist_duration_conversion(
                baseline_rate=baseline_rate,
                relative_mde=relative_mde,
                daily_total_traffic=daily_traffic,
                n_variants=args.n_variants,
                alpha=args.alpha,
                power=args.power,
                max_looks=args.max_looks,
            )
        else:
            plan = bayesian_duration_conversion(
                baseline_rate=baseline_rate,
                relative_mde=relative_mde,
                daily_total_traffic=daily_traffic,
                n_variants=args.n_variants,
                prob_threshold=args.prob_threshold,
                max_expected_loss=args.max_expected_loss,
                assurance_target=args.assurance_target,
                max_days=args.max_days,
            )

        plan_json = json.dumps(
            {
                "method": plan.method,
                "estimated_days": plan.estimated_days,
                "n_per_variant": plan.n_per_variant,
                "assumptions": plan.assumptions,
                "diagnostics": plan.diagnostics,
            },
            indent=2,
        )
        if args.output:
            _write_text(args.output, plan_json + "\n")
        else:
            print(plan_json)
        return 0

    if args.command == "doctor":
        result = run_doctor()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            status = "OK" if result["ok"] else "FAIL"
            print(f"Doctor status: {status}")
            for check in result["checks"]:
                icon = "PASS" if check["passed"] else "FAIL"
                print(
                    f"- {icon} {check['name']}: {check['detail']} "
                    f"(required {check['required']})"
                )
        if args.strict and not result["ok"]:
            return 1
        return 0

    if args.command == "analyze-file":
        mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
        rows = read_table(args.input, sheet=args.sheet)
        input_payload = build_payload_from_rows(rows, mapping)
    else:
        input_payload = json.loads(Path(args.input).read_text(encoding="utf-8"))

    analysis_input = parse_payload(input_payload)
    result = analyze(analysis_input)
    result_json = json.dumps(result.to_dict(), indent=2)

    if args.output:
        _write_text(args.output, result_json + "\n")
    else:
        print(result_json)

    if args.report:
        report = build_markdown_report(result)
        _write_text(args.report, report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
