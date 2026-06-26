#!/usr/bin/env python3
"""Run predictive planner success-improvement campaign across checkpoints and planner configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.benchmark.predictive_planner_config import build_predictive_planner_algo_config
from scripts.validation.predictive_eval_common import load_seed_manifest, make_subset_scenarios

_CONTRACT_VERSION = "benchmark-reset-v2"
_TRAINING_FAMILY = "prediction_planner"


@dataclass
class EvalResult:
    """Aggregate result for one checkpoint/config pair."""

    checkpoint: str
    variant: str
    suite: str
    episodes: int
    success_rate: float
    success_ci_low: float
    success_ci_high: float
    mean_min_distance: float
    mean_avg_speed: float
    jsonl_path: str


class MissingCampaignArtifactError(RuntimeError):
    """Raised when a campaign stage does not produce its required JSONL artifact."""

    def __init__(
        self,
        *,
        jsonl_path: Path,
        suite_name: str,
        checkpoint: str,
        variant_name: str,
        reason: str,
    ) -> None:
        """Initialize the missing-artifact error with structured context."""
        self.jsonl_path = jsonl_path
        self.suite_name = suite_name
        self.checkpoint = checkpoint
        self.variant_name = variant_name
        self.reason = reason
        super().__init__(
            "Predictive success campaign expected JSONL artifact was not usable: "
            f"path={jsonl_path} suite={suite_name} variant={variant_name} "
            f"checkpoint={checkpoint} reason={reason}"
        )

    def to_failure_payload(self) -> dict[str, str]:
        """Return structured failure details for campaign summaries."""
        return {
            "type": type(self).__name__,
            "message": str(self),
            "jsonl_path": str(self.jsonl_path),
            "suite": str(self.suite_name),
            "checkpoint": str(self.checkpoint),
            "variant": str(self.variant_name),
            "reason": str(self.reason),
        }


def parse_args() -> argparse.Namespace:
    """Parse campaign CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Validate the scenario matrix, hard-seed manifest, and planner grid, then exit "
            "without requiring checkpoints or running evaluation."
        ),
    )
    parser.add_argument(
        "--scenario-matrix",
        type=Path,
        default=Path("configs/scenarios/classic_interactions.yaml"),
    )
    parser.add_argument(
        "--hard-seed-manifest",
        type=Path,
        default=Path("configs/benchmarks/predictive_hard_seeds_v1.yaml"),
    )
    parser.add_argument(
        "--planner-grid",
        type=Path,
        default=Path("configs/benchmarks/predictive_sweep_planner_grid_v1.yaml"),
    )
    parser.add_argument("--horizon", type=int, default=140)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--closed-loop-gate-baseline-variant",
        type=str,
        default=None,
        help=(
            "Optional baseline planner-grid variant for a fail-closed local gate. "
            "When set, the best ranked variant must improve closed-loop success over this row."
        ),
    )
    parser.add_argument(
        "--closed-loop-gate-min-global-success-delta",
        type=float,
        default=0.0,
        help="Minimum best-minus-baseline global success-rate delta for the closed-loop gate.",
    )
    parser.add_argument(
        "--closed-loop-gate-min-hard-success-delta",
        type=float,
        default=0.0,
        help="Minimum best-minus-baseline hard-suite success-rate delta for the closed-loop gate.",
    )
    parser.add_argument(
        "--closed-loop-gate-max-min-distance-regression",
        type=float,
        default=float("inf"),
        help=(
            "Maximum allowed global mean-min-distance regression in meters. "
            "Use a finite value to reject success gains that give back too much clearance."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/tmp/predictive_planner/campaigns/latest_success_campaign"),
    )
    return parser.parse_args()


def _load_planner_variants(path: Path) -> list[dict]:
    """Load planner sweep variants from YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    variants = payload.get("variants", [])
    if not isinstance(variants, list):
        raise TypeError(f"planner grid variants must be a list: {path}")
    out = []
    names: set[str] = set()
    for index, item in enumerate(variants, start=1):
        if not isinstance(item, dict):
            raise TypeError(f"planner grid variant #{index} must be a mapping: {path}")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"planner grid variant #{index} has no name: {path}")
        if name in names:
            raise ValueError(f"planner grid contains duplicate variant name: {name}")
        names.add(name)
        params = item.get("params", {})
        if not isinstance(params, dict):
            raise TypeError(f"planner grid variant params must be a mapping: {name}")
        out.append({"name": name, "params": params})
    if not out:
        raise RuntimeError(f"No planner variants found in {path}")
    return out


def _episode_success(row: dict) -> bool:
    """Resolve episode success with collision-aware fallback semantics."""
    metrics = row.get("metrics", {})
    if "success_rate" in metrics:
        value = metrics.get("success_rate")
        if value is None or value == "":
            return False
        return float(value) >= 0.5
    value = metrics.get("success", False)
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return False
    return float(value) >= 0.5


def _bootstrap_ci(values: np.ndarray, n_samples: int, seed: int) -> tuple[float, float]:
    """Bootstrap 95% CI for the mean."""
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty((n_samples,), dtype=float)
    n = values.size
    for i in range(n_samples):
        sample = values[rng.integers(0, n, size=n)]
        means[i] = float(np.mean(sample))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _base_algo_cfg(checkpoint: str) -> dict:
    """Base predictive planner algorithm config."""
    return build_predictive_planner_algo_config(checkpoint_path=checkpoint, device="cpu")


def _run_eval(
    *,
    scenarios_or_path: Path | list[dict],
    suite_name: str,
    checkpoint: str,
    variant_name: str,
    algo_cfg: dict,
    args: argparse.Namespace,
    output_dir: Path,
) -> EvalResult:
    """Run one evaluation and return aggregated metrics."""
    safe_ckpt = _checkpoint_token(checkpoint)
    tag = f"{suite_name}__{variant_name}__{safe_ckpt}"
    algo_cfg_path = output_dir / f"{tag}_algo.yaml"
    jsonl_path = output_dir / f"{tag}.jsonl"
    algo_cfg_path.write_text(yaml.safe_dump(algo_cfg, sort_keys=False), encoding="utf-8")
    if jsonl_path.exists():
        jsonl_path.unlink()

    run_map_batch(
        scenarios_or_path,
        jsonl_path,
        schema_path=Path("robot_sf/benchmark/schemas/episode.schema.v1.json"),
        algo="prediction_planner",
        algo_config_path=str(algo_cfg_path),
        horizon=int(args.horizon),
        dt=float(args.dt),
        workers=int(args.workers),
        resume=False,
        benchmark_profile="experimental",
    )
    if not jsonl_path.exists():
        raise MissingCampaignArtifactError(
            jsonl_path=jsonl_path,
            suite_name=suite_name,
            checkpoint=checkpoint,
            variant_name=variant_name,
            reason="missing_jsonl",
        )

    rows: list[dict] = []
    for line_number, line in enumerate(
        jsonl_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"Malformed JSON in {jsonl_path} at line {line_number}: {exc}"
            ) from exc
        if isinstance(payload, dict):
            rows.append(payload)
    if not rows:
        raise MissingCampaignArtifactError(
            jsonl_path=jsonl_path,
            suite_name=suite_name,
            checkpoint=checkpoint,
            variant_name=variant_name,
            reason="empty_jsonl",
        )
    success_vals = np.asarray([1.0 if _episode_success(row) else 0.0 for row in rows], dtype=float)
    min_dist_vals = np.asarray(
        [
            float(row.get("metrics", {}).get("min_distance"))
            for row in rows
            if row.get("metrics", {}).get("min_distance") is not None
        ],
        dtype=float,
    )
    speed_vals = np.asarray(
        [float(row.get("metrics", {}).get("avg_speed", 0.0)) for row in rows],
        dtype=float,
    )
    ci_low, ci_high = _bootstrap_ci(
        success_vals,
        n_samples=int(args.bootstrap_samples),
        seed=int(args.bootstrap_seed),
    )
    return EvalResult(
        checkpoint=checkpoint,
        variant=variant_name,
        suite=suite_name,
        episodes=len(rows),
        success_rate=float(np.mean(success_vals)) if success_vals.size > 0 else 0.0,
        success_ci_low=ci_low,
        success_ci_high=ci_high,
        mean_min_distance=float(np.mean(min_dist_vals)) if min_dist_vals.size > 0 else float("nan"),
        mean_avg_speed=float(np.mean(speed_vals)) if speed_vals.size > 0 else 0.0,
        jsonl_path=str(jsonl_path),
    )


def _rank_key(hard: EvalResult, global_res: EvalResult) -> tuple[float, float, float, float, float]:
    """Ranking key preferring actual success before clearance in hard-suite ties."""
    hard_clearance = (
        hard.mean_min_distance if np.isfinite(hard.mean_min_distance) else float("-inf")
    )
    global_clearance = (
        global_res.mean_min_distance if np.isfinite(global_res.mean_min_distance) else float("-inf")
    )
    return (
        hard.success_rate,
        global_res.success_rate,
        hard_clearance,
        global_clearance,
        global_res.mean_avg_speed,
    )


def _closed_loop_gate_result(
    ranked: list[dict],
    *,
    baseline_variant: str | None,
    min_global_success_delta: float,
    min_hard_success_delta: float,
    max_min_distance_regression: float,
) -> dict[str, object] | None:
    """Evaluate an optional closed-loop gate over ranked campaign results."""
    if not baseline_variant:
        return None
    if not ranked:
        return {
            "passed": False,
            "reason": "no_ranked_results",
            "baseline_variant": baseline_variant,
        }

    baseline = next(
        (row for row in ranked if str(row.get("variant")) == str(baseline_variant)),
        None,
    )
    if baseline is None:
        return {
            "passed": False,
            "reason": "baseline_variant_missing",
            "baseline_variant": baseline_variant,
            "candidate_variant": str(ranked[0].get("variant", "not_available")),
        }

    best = ranked[0]
    hard_delta = float(best["hard"]["success_rate"]) - float(baseline["hard"]["success_rate"])
    global_delta = float(best["global"]["success_rate"]) - float(baseline["global"]["success_rate"])
    best_global_min_distance = float(best["global"]["mean_min_distance"])
    baseline_global_min_distance = float(baseline["global"]["mean_min_distance"])
    min_distance_delta = best_global_min_distance - baseline_global_min_distance

    reasons: list[str] = []
    if global_delta < float(min_global_success_delta):
        reasons.append("global_success_delta_below_gate")
    if hard_delta < float(min_hard_success_delta):
        reasons.append("hard_success_delta_below_gate")
    if math.isfinite(float(max_min_distance_regression)):
        if not (
            math.isfinite(best_global_min_distance) and math.isfinite(baseline_global_min_distance)
        ):
            reasons.append("global_min_distance_not_finite")
        elif min_distance_delta < -float(max_min_distance_regression):
            reasons.append("global_min_distance_regression_above_gate")

    return {
        "passed": not reasons,
        "reason": "passed" if not reasons else ",".join(reasons),
        "baseline_variant": baseline_variant,
        "candidate_variant": best["variant"],
        "thresholds": {
            "min_global_success_delta": float(min_global_success_delta),
            "min_hard_success_delta": float(min_hard_success_delta),
            "max_min_distance_regression": float(max_min_distance_regression),
        },
        "deltas": {
            "hard_success": hard_delta,
            "global_success": global_delta,
            "global_mean_min_distance": min_distance_delta,
        },
        "baseline": baseline,
        "candidate": best,
    }


def _checkpoint_token(checkpoint: str) -> str:
    """Return collision-resistant token for checkpoint artifact naming."""
    ckpt_hash = hashlib.sha1(str(Path(checkpoint).resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{Path(checkpoint).stem.replace('.', '_')}_{ckpt_hash}"


def _checkpoint_label(path_str: str) -> str:
    """Return compact checkpoint label for reports."""
    p = Path(path_str)
    if len(p.parts) >= 2:
        return "/".join(p.parts[-2:])
    return p.name


def _nan_to_none(value: object) -> object:
    """Recursively convert NaN floats to ``None`` for JSON compatibility."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _nan_to_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(v) for v in value]
    return value


def _format_optional_float(value: object) -> str:
    """Format optional numeric report fields without masking missing gate data."""
    if value is None:
        return "not_available"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "not_available"


def _write_success_reports(summary: dict, output_dir: Path) -> tuple[Path, Path]:
    """Write successful or gate-failed campaign reports."""
    ranked = summary["ranked"]
    top = summary["best"]
    gate = summary.get("closed_loop_gate")

    json_path = output_dir / "campaign_summary.json"
    json_path.write_text(json.dumps(_nan_to_none(summary), indent=2), encoding="utf-8")

    md_lines = [
        "# Predictive Success Campaign",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Scenario matrix: `{summary['scenario_matrix']}`",
        f"- Hard manifest: `{summary['hard_seed_manifest']}`",
        f"- Planner grid: `{summary['planner_grid']}`",
        f"- Candidates: `{summary['num_candidates']}`",
        "",
        "## Best Candidate",
        "",
        f"- Checkpoint: `{top['checkpoint']}`",
        f"- Variant: `{top['variant']}`",
        f"- Hard success: `{top['hard']['success_rate']:.4f}` "
        f"(95% CI `{top['hard']['success_ci_low']:.4f}`..`{top['hard']['success_ci_high']:.4f}`)",
        f"- Hard mean min-distance: `{top['hard']['mean_min_distance']:.4f}`",
        f"- Global success: `{top['global']['success_rate']:.4f}` "
        f"(95% CI `{top['global']['success_ci_low']:.4f}`.."
        f"`{top['global']['success_ci_high']:.4f}`)",
        f"- Global mean min-distance: `{top['global']['mean_min_distance']:.4f}`",
        "",
        "## Ranking (top 10)",
        "",
        "| Rank | Variant | Checkpoint | Hard SR | Hard MinDist | Global SR | Global MinDist |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(ranked[:10], start=1):
        md_lines.append(
            "| "
            f"{i} | {row['variant']} | {_checkpoint_label(row['checkpoint'])} | "
            f"{row['hard']['success_rate']:.4f} | {row['hard']['mean_min_distance']:.4f} | "
            f"{row['global']['success_rate']:.4f} | {row['global']['mean_min_distance']:.4f} |"
        )

    if gate is not None:
        deltas = gate.get("deltas", {})
        if not isinstance(deltas, dict):
            deltas = {}
        md_lines.extend(
            [
                "",
                "## Closed-Loop Gate",
                "",
                f"- Status: `{'passed' if gate['passed'] else 'failed'}`",
                f"- Reason: `{gate['reason']}`",
                f"- Baseline variant: `{gate['baseline_variant']}`",
                f"- Candidate variant: `{gate.get('candidate_variant', 'not_available')}`",
                f"- Global success delta: `{_format_optional_float(deltas.get('global_success'))}`",
                f"- Hard success delta: `{_format_optional_float(deltas.get('hard_success'))}`",
                f"- Global mean-min-distance delta: "
                f"`{_format_optional_float(deltas.get('global_mean_min_distance'))}`",
            ]
        )

    md_path = output_dir / "campaign_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return json_path, md_path


def _run_campaign_results(
    *,
    args: argparse.Namespace,
    variants: list[dict],
    hard_scenarios: list[dict],
) -> list[dict]:
    """Run each checkpoint and planner-grid variant pair."""
    results: list[dict] = []
    for checkpoint in args.checkpoints:
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        for variant in variants:
            name = str(variant["name"])
            cfg = _base_algo_cfg(checkpoint)
            cfg.update(dict(variant.get("params", {})))
            hard_res = _run_eval(
                scenarios_or_path=hard_scenarios,
                suite_name="hard",
                checkpoint=checkpoint,
                variant_name=name,
                algo_cfg=cfg,
                args=args,
                output_dir=args.output_dir,
            )
            global_res = _run_eval(
                scenarios_or_path=args.scenario_matrix,
                suite_name="global",
                checkpoint=checkpoint,
                variant_name=name,
                algo_cfg=cfg,
                args=args,
                output_dir=args.output_dir,
            )
            results.append(
                {
                    "checkpoint": checkpoint,
                    "variant": name,
                    "config": cfg,
                    "hard": hard_res.__dict__,
                    "global": global_res.__dict__,
                    "ranking_key": list(_rank_key(hard_res, global_res)),
                }
            )
    return results


def main() -> int:
    """Execute full campaign and write machine + human-readable reports."""
    args = parse_args()
    variants = _load_planner_variants(args.planner_grid)
    hard_manifest = load_seed_manifest(args.hard_seed_manifest)
    hard_scenarios = make_subset_scenarios(args.scenario_matrix, hard_manifest)
    if not hard_scenarios:
        raise RuntimeError("Hard-case manifest did not match any scenarios.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "check_only", False):
        summary = {
            "contract_version": _CONTRACT_VERSION,
            "training_family": _TRAINING_FAMILY,
            "artifact_role": "predictive_success_campaign_manifest_check",
            "status": "checked_no_evaluation_run",
            "generated_at": datetime.now(UTC).isoformat(),
            "scenario_matrix": str(args.scenario_matrix),
            "hard_seed_manifest": str(args.hard_seed_manifest),
            "planner_grid": str(args.planner_grid),
            "hard_manifest_entries": len(hard_manifest),
            "matched_hard_scenarios": len(hard_scenarios),
            "planner_variants": [str(variant["name"]) for variant in variants],
            "num_planner_variants": len(variants),
            "scope_note": (
                "Structural manifest/grid check only; no checkpoint evaluation, benchmark run, "
                "or maneuver-authority success interpretation was performed."
            ),
        }
        json_path = args.output_dir / "manifest_check_summary.json"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote manifest check summary: {json_path}")
        return 0

    if not getattr(args, "checkpoints", None):
        raise SystemExit("--checkpoints is required unless --check-only is set")

    results: list[dict] = []

    try:
        results = _run_campaign_results(
            args=args,
            variants=variants,
            hard_scenarios=hard_scenarios,
        )
    except (Exception, SystemExit) as exc:
        if isinstance(exc, MissingCampaignArtifactError):
            failure = exc.to_failure_payload()
        else:
            message = str(exc)
            if isinstance(exc, SystemExit) and not message:
                message = str(getattr(exc, "code", ""))
            failure = {"type": type(exc).__name__, "message": message}
        failure_summary = {
            "contract_version": _CONTRACT_VERSION,
            "training_family": _TRAINING_FAMILY,
            "artifact_role": "predictive_success_campaign",
            "status": "failed",
            "generated_at": datetime.now(UTC).isoformat(),
            "scenario_matrix": str(args.scenario_matrix),
            "hard_seed_manifest": str(args.hard_seed_manifest),
            "planner_grid": str(args.planner_grid),
            "horizon": int(args.horizon),
            "dt": float(args.dt),
            "workers": int(args.workers),
            "partial_results": results,
            "failure": failure,
        }
        json_path = args.output_dir / "campaign_summary.json"
        json_path.write_text(
            json.dumps(_nan_to_none(failure_summary), indent=2),
            encoding="utf-8",
        )
        md_path = args.output_dir / "campaign_report.md"
        md_path.write_text(
            "\n".join(
                [
                    "# Predictive Success Campaign",
                    "",
                    "- Status: `failed`",
                    f"- Failure type: `{failure['type']}`",
                    f"- Failure: `{failure['message']}`",
                    f"- JSON summary: `{json_path}`",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"summary": str(json_path), "report": str(md_path)}, indent=2))
        return 2

    ranked = sorted(
        results,
        key=lambda r: tuple(r["ranking_key"]),
        reverse=True,
    )

    top = ranked[0]
    summary = {
        "contract_version": _CONTRACT_VERSION,
        "training_family": _TRAINING_FAMILY,
        "artifact_role": "predictive_success_campaign",
        "generated_at": datetime.now(UTC).isoformat(),
        "scenario_matrix": str(args.scenario_matrix),
        "hard_seed_manifest": str(args.hard_seed_manifest),
        "planner_grid": str(args.planner_grid),
        "horizon": int(args.horizon),
        "dt": float(args.dt),
        "workers": int(args.workers),
        "bootstrap_samples": int(args.bootstrap_samples),
        "num_candidates": len(ranked),
        "best": top,
        "ranked": ranked,
    }
    gate = _closed_loop_gate_result(
        ranked,
        baseline_variant=args.closed_loop_gate_baseline_variant,
        min_global_success_delta=float(args.closed_loop_gate_min_global_success_delta),
        min_hard_success_delta=float(args.closed_loop_gate_min_hard_success_delta),
        max_min_distance_regression=float(args.closed_loop_gate_max_min_distance_regression),
    )
    if gate is not None:
        summary["closed_loop_gate"] = gate
    if gate is not None and not bool(gate["passed"]):
        summary["status"] = "failed_closed_loop_gate"

    json_path, md_path = _write_success_reports(summary, args.output_dir)
    print(json.dumps({"summary": str(json_path), "report": str(md_path)}, indent=2))
    if gate is not None and not bool(gate["passed"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
