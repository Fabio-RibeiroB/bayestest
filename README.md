# bayestest

`bayestest` is an agent-friendly Python package and CLI for A/B/n decisions.

> Early release (alpha): this project is work in progress and not production-hardened yet.

It supports:
- Bayesian conversion-rate decisions (Beta-Binomial)
- Bayesian ARPU probability-to-win from aggregate revenue stats
- Frequentist sequential decisions (O'Brien-Fleming alpha spending)
- Multi-variant (`A/B/n`) comparisons against one control
- Guardrail checks (latency, bounce rate, error rate, etc.)
- SRM detection (sample ratio mismatch) for data quality
- Structured JSON output for agents and automations
- Markdown report generation for human review
- CSV/XLSX ingestion with mapping files

## Install (uv)

```bash
uv sync --group test
```

Run commands through `uv` so the managed environment is always used.

## Requirements Checklist

- Python `>=3.9`
- Dependencies: `numpy>=1.22`, `openpyxl>=3.1.0`
- Test dependencies available via `uv sync --group test`
- CLI available via `uv run bayestest --help`
- Exactly one control variant in analysis inputs
- Required variant fields: `name`, `visitors`, `conversions`
- ARPU mode also requires: `revenue_sum`, `revenue_sum_squares`
- CSV/XLSX mode requires mapping JSON (`example-mapping` or `example-duration-mapping`)

## CLI quickstart

Generate an input template:

```bash
uv run bayestest example-input > input.json
```

Run analysis and write both JSON + report:

```bash
uv run bayestest analyze \
  --input input.json \
  --output output.json \
  --report report.md
```

Analyze directly from CSV/XLSX:

```bash
uv run bayestest analyze-file \
  --input experiment.xlsx \
  --mapping mapping.json \
  --sheet Sheet1 \
  --output output.json \
  --report report.md
```

Run all bundled demos:

```bash
make demo
```

Check environment readiness:

```bash
uv run bayestest doctor
uv run bayestest doctor --json
uv run bayestest doctor --strict
```

Estimate duration from assumptions:

```bash
uv run bayestest duration \
  --method frequentist \
  --baseline-rate 0.04 \
  --relative-mde 0.05 \
  --daily-traffic 50000 \
  --n-variants 3 \
  --max-looks 10
```

Estimate Bayesian duration (assurance simulation):

```bash
uv run bayestest duration \
  --method bayesian \
  --baseline-rate 0.04 \
  --relative-mde 0.05 \
  --daily-traffic 50000 \
  --n-variants 3 \
  --max-days 60
```

Analyze pasted variant text:

```bash
uv run bayestest analyze-text \
  --text "Variant A: 100 conversions out of 2000 visitors\nVariant B: 125 conversions out of 2000 visitors" \
  --experiment-name pasted_example
```

Estimate duration from CSV/XLSX:

```bash
uv run bayestest example-duration-mapping > duration_mapping.json
uv run bayestest duration \
  --input duration_inputs.xlsx \
  --mapping duration_mapping.json \
  --sheet Sheet1
```

## Development

Run the test suite:

```bash
uv run pytest
```

## Agent Playbook

1. Detect available columns in source data (CSV/XLSX).
2. Build `mapping.json` to map client fields into `bayestest` fields.
3. Run `bayestest analyze-file ...`.
4. Read `recommendation.action`, `decision_confidence`, and `risk_flags`.
5. If `action=continue_collecting_data`, schedule next look.
6. If `action=investigate_data_quality`, resolve SRM/tracking before any ship decision.
7. For planning questions ("how long should we run?"), run `bayestest duration`.
8. For pasted stats messages, run `bayestest analyze-text` to convert free text into analysis.
9. For spreadsheet planning assumptions, run `bayestest duration --input ... --mapping ...`.

## Input schema

Top-level fields:
- `experiment_name` (str)
- `method` (`"bayesian"` or `"frequentist_sequential"`)
- `primary_metric` (`"conversion_rate"` or `"arpu"`)
- `alpha` (float, default `0.05`)
- `look_index` and `max_looks` (ints, sequential mode)
- `information_fraction` (optional float in `(0, 1]`, overrides look/max_looks)
- `variants` (list): exactly one row must include `"is_control": true`
- `guardrails` (optional list)
- `decision_thresholds` (optional):
  - `bayes_prob_beats_control` (default `0.95`)
  - `max_expected_loss` (default `0.001`)
- `samples` (default `50000`, Bayesian mode)
- `random_seed` (default `7`)

Variant row:

```json
{
  "name": "control",
  "visitors": 100000,
  "conversions": 4000,
  "is_control": true
}
```

For `primary_metric: "arpu"`, each variant also needs:
- `revenue_sum`
- `revenue_sum_squares`

## Mapping-based ingestion (CSV/XLSX)

Generate mapping template:

```bash
uv run bayestest example-mapping > mapping.json
```

Mapping keys:
- `columns.variant`, `columns.visitors`, `columns.conversions`
- optional `columns.is_control`
- optional `columns.revenue_sum`, `columns.revenue_sum_squares`
- control detection fallback: `control.column` + `control.value`

This lets agents reshape arbitrary business exports into a consistent contract.

## Examples

1. Bayesian conversion-rate A/B/n:

```json
{
  "experiment_name": "homepage_cta",
  "method": "bayesian",
  "primary_metric": "conversion_rate",
  "variants": [
    {"name": "control", "visitors": 50000, "conversions": 2000, "is_control": true},
    {"name": "v1", "visitors": 50000, "conversions": 2080, "is_control": false},
    {"name": "v2", "visitors": 50000, "conversions": 2140, "is_control": false}
  ]
}
```

2. Bayesian ARPU probability-to-win:

```json
{
  "experiment_name": "pricing_page",
  "method": "bayesian",
  "primary_metric": "arpu",
  "variants": [
    {"name": "control", "visitors": 10000, "conversions": 550, "revenue_sum": 22000, "revenue_sum_squares": 150000, "is_control": true},
    {"name": "v1", "visitors": 10000, "conversions": 570, "revenue_sum": 23500, "revenue_sum_squares": 170000, "is_control": false}
  ]
}
```

3. Sequential ARPU (early look):

```json
{
  "experiment_name": "checkout_flow",
  "method": "frequentist_sequential",
  "primary_metric": "arpu",
  "alpha": 0.05,
  "look_index": 3,
  "max_looks": 10,
  "variants": [
    {"name": "control", "visitors": 12000, "conversions": 610, "revenue_sum": 21000, "revenue_sum_squares": 150000, "is_control": true},
    {"name": "v1", "visitors": 12000, "conversions": 640, "revenue_sum": 22400, "revenue_sum_squares": 167000, "is_control": false}
  ]
}
```

## Input validation errors

Common errors and fixes:
- `Exactly one variant must have is_control=true`:
  mark one and only one control row.
- `conversions cannot exceed visitors`:
  fix aggregation query or mapped columns.
- `ARPU requires revenue_sum and revenue_sum_squares`:
  include both revenue aggregate columns.
- `primary_metric must be 'conversion_rate' or 'arpu'`:
  fix mapping or input payload values.

## Agent output contract

`recommendation` contains:
- `action` (`ship_*`, `continue_collecting_data`, `do_not_ship`, `investigate_data_quality`, `stop_and_rollback`)
- `rationale`
- `decision_confidence` (0 to 1)
- `next_best_action`
- `risk_flags` (e.g. `srm_detected`, `guardrail_failure`)

## Notes

- This tool currently uses aggregate statistics for ARPU.
- For production, add metric-specific robust models, stronger QA checks, and regression tests.
