"""Deterministic SNQI-v2 family, seed bootstrap, diagnostics and offline records.

Every scored report is emitted together with the preregistered V2-F family.
Undefined source metrics and incomplete planner/seed coverage fail closed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import spearmanr

from robot_sf.benchmark.fallback_policy import (
    runtime_fallback_or_degraded_marker,
    summarize_benchmark_availability,
)
from robot_sf.benchmark.snqi.bootstrap import bootstrap_stability
from robot_sf.benchmark.snqi.compute import compute_snqi_v2, normalize_snqi_v2_terms
from robot_sf.benchmark.snqi.v2_spec import FAMILY, QUALITY_TERMS, TERMS, WEIGHTS, SnqiV2Spec

if TYPE_CHECKING:
    from pathlib import Path

CLAIM_BOUNDARY = (
    "SNQI-v2 is a declared benchmark aggregate over simulator quantities. It is not a validated "
    "measure of human comfort or safety and admits no deployment ranking on its own."
)


def family_vectors() -> list[dict[str, Any]]:
    """Return 2,000 rejection-sampled Dirichlet draws, 11 grid members, 2 relaxed strata.

    Rejection conditions the uniform simplex draw on every quality weight >=.02;
    adding a floor after drawing would be a different distribution.
    """
    rng = np.random.default_rng(FAMILY["seed"])
    vectors: list[dict[str, Any]] = []

    def append(
        name: str,
        quality: Sequence[float],
        *,
        success: float = 1,
        collision: float = 2,
        kind: str = "stratified",
    ) -> None:
        vectors.append(
            {
                "name": name,
                "kind": kind,
                "weights": {
                    "S": success,
                    "C": collision,
                    **dict(zip(QUALITY_TERMS, map(float, quality), strict=True)),
                },
            }
        )

    while len(vectors) < FAMILY["draws"]:
        quality = rng.dirichlet(FAMILY["dirichlet_alpha"]) * FAMILY["quality_mass"]
        if np.min(quality) >= FAMILY["minimum_weight"]:
            append(f"dirichlet_{len(vectors):04d}", quality)
    append("equal", [0.19] * 5)
    for term in QUALITY_TERMS:
        append(f"heavy_{term}", [0.55 if key == term else 0.10 for key in QUALITY_TERMS])
        remaining = 0.95 - WEIGHTS[term]
        append(
            f"leave_one_out_{term}",
            [0.02 if key == term else WEIGHTS[key] * 0.93 / remaining for key in QUALITY_TERMS],
        )
    for collision in (0.5, 1.0):
        append(
            f"relaxed_S0.5_C{collision}",
            [WEIGHTS[key] for key in QUALITY_TERMS],
            success=0.5,
            collision=collision,
            kind="relaxed_strata",
        )
    return vectors


def _rho(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Report undefined constant-rank correlations as null, never fabricated agreement.

    Returns:
        Validated result described above.
    """
    if len(a) < 2 or len(set(a)) == 1 or len(set(b)) == 1:
        return None
    return float(spearmanr(a, b).statistic)


def score_episode(episode: Mapping[str, Any], spec: SnqiV2Spec) -> dict[str, Any]:
    """Copy an episode and add v2 fields while preserving every legacy metric.

    Returns:
        Validated result described above.
    """
    validate_episode_execution(episode)
    metrics = episode.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("SNQI-v2 episode requires metrics")
    inputs = {**metrics, "executed_steps": episode.get("steps")}
    normalized = normalize_snqi_v2_terms(inputs, spec)
    return {
        **episode,
        "metrics": {
            **metrics,
            "snqi_v2": compute_snqi_v2(inputs, spec),
            "snqi_v2_terms": normalized,
        },
    }


def _planner(episode: Mapping[str, Any]) -> str:
    """Return explicit arm identity including kinematics where recorded."""
    name = episode.get("planner_key", episode.get("algo"))
    if not isinstance(name, str) or not name:
        raise ValueError("SNQI-v2 report requires explicit planner identity")
    kinematics = episode.get("kinematics")
    return f"{name}::{kinematics}" if kinematics else name


def _summary(values: Sequence[float | None]) -> dict[str, Any]:
    finite = [v for v in values if v is not None]
    return {
        "median": float(np.median(finite)) if finite else None,
        "minimum": min(finite) if finite else None,
        "defined": len(finite),
        "undefined": len(values) - len(finite),
    }


def build_family_report(
    episodes: Sequence[Mapping[str, Any]], spec: SnqiV2Spec, *, bootstrap_samples: int = 2000
) -> dict[str, Any]:
    """Compute family sensitivity and paired seed-bootstrap intervals from raw records.

    Returns:
        Validated result described above.
    """
    if not episodes or bootstrap_samples < 1:
        raise ValueError("SNQI-v2 family needs episodes and positive bootstrap samples")
    scored = [score_episode(episode, spec) for episode in episodes]
    groups = sorted({_planner(episode) for episode in scored})
    grouped = {key: [episode for episode in scored if _planner(episode) == key] for key in groups}
    means = np.array(
        [
            [
                np.mean([ep["metrics"]["snqi_v2_terms"][term] for ep in grouped[key]])
                for term in TERMS
            ]
            for key in groups
        ]
    )
    signs = np.array([1, -1, -1, -1, -1, -1, -1])
    declared = means @ (np.array([spec.weights[t] for t in TERMS]) * signs)

    def ordering(scores: np.ndarray) -> list[int]:
        return sorted(range(len(groups)), key=lambda i: (-scores[i], groups[i]))

    declared_order = ordering(declared)
    declared_ranks = {i: rank + 1 for rank, i in enumerate(declared_order)}
    vectors = family_vectors()
    stratified = [v for v in vectors if v["kind"] == "stratified"]
    results = []
    flips = np.zeros((len(groups), len(groups)))
    top1 = dict.fromkeys(groups, 0.0)
    top3 = []
    for vector in vectors:
        values = means @ (np.array([vector["weights"][t] for t in TERMS]) * signs)
        order = ordering(values)
        ranks = {i: rank + 1 for rank, i in enumerate(order)}
        row = {
            **vector,
            "ordering": [groups[i] for i in order],
            "spearman_declared": _rho(declared, values),
            "spearman_success": _rho(means[:, 0], values),
        }
        if vector["name"].startswith("leave_one_out_"):
            row["rank_change"] = {groups[i]: ranks[i] - declared_ranks[i] for i in ranks}
        results.append(row)
        if vector["kind"] != "stratified":
            continue
        winners = np.flatnonzero(np.isclose(values, values.max(), rtol=0, atol=1e-12))
        for i in winners:
            top1[groups[i]] += 1 / len(winners) / len(stratified)
        n_top = min(3, len(groups))
        top3.append(len(set(order[:n_top]) & set(declared_order[:n_top])) / n_top)
        flips += (declared[:, None] - declared[None, :]) * (
            values[:, None] - values[None, :]
        ) < -1e-12
    paired_cells = [
        {(ep.get("scenario_id"), ep.get("seed")) for ep in grouped[key]} for key in groups
    ]
    for key, cells in zip(groups, paired_cells, strict=True):
        if (
            any(scenario is None or seed is None for scenario, seed in cells)
            or len(cells) != len(grouped[key])
            or cells != paired_cells[0]
        ):
            raise ValueError(
                "SNQI-v2 family requires complete unique paired scenario/seed coverage"
            )
    seed_sets = [{ep.get("seed") for ep in grouped[key]} for key in groups]
    if any(None in seeds or seeds != seed_sets[0] for seeds in seed_sets):
        raise ValueError("SNQI-v2 paired seed bootstrap requires identical explicit seed coverage")
    seeds = sorted(seed_sets[0])
    by_seed = np.array(
        [
            [
                np.mean([ep["metrics"]["snqi_v2"] for ep in grouped[key] if ep["seed"] == seed])
                for seed in seeds
            ]
            for key in groups
        ]
    )
    rng = np.random.default_rng(FAMILY["seed"])
    draws = rng.integers(0, len(seeds), size=(bootstrap_samples, len(seeds)))
    intervals = np.quantile(by_seed[:, draws].mean(axis=2), [0.025, 0.975], axis=1)
    seed_records = [
        {"algo": key, "metrics": {"snqi": float(by_seed[i, j])}}
        for i, key in enumerate(groups)
        for j in range(len(seeds))
    ]
    stability = (
        bootstrap_stability(
            seed_records,
            spec.weights,
            rng=np.random.default_rng(FAMILY["seed"]),
            samples=bootstrap_samples,
        )
        if len(groups) > 1
        else {"status": "not_available", "reason": "one planner"}
    )
    return {
        "schema_version": "snqi-v2-family.v1",
        "family": "V2-F",
        "seed": FAMILY["seed"],
        "claim_boundary": CLAIM_BOUNDARY,
        "provenance": spec.provenance(),
        "episode_count": len(scored),
        "stratified_count": len(stratified),
        "tie_policy": "average ranks for rho; split top-1 credit; lexical top-3 boundary",
        "bootstrap": {
            "unit": "seed",
            "confidence_intervals_paired": True,
            "samples": bootstrap_samples,
            "confidence": 0.95,
            "seed_count": len(seeds),
            "stability": {
                **stability,
                "paired": False,
                "resampling": "independent_per_planner_seed_resampling",
            },
        },
        "declared_ranking": [
            {
                "planner": groups[i],
                "mean_snqi_v2": float(declared[i]),
                "ci95": [float(intervals[0, i]), float(intervals[1, i])],
                "rank": declared_ranks[i],
            }
            for i in declared_order
        ],
        "spearman_against_declared": _summary(
            [r["spearman_declared"] for r in results if r["kind"] == "stratified"]
        ),
        "spearman_against_success": _summary(
            [r["spearman_success"] for r in results if r["kind"] == "stratified"]
        ),
        "top1_frequency": top1,
        "top3_set_stability": _summary(top3),
        "pairwise_order_flip_frequency": {
            left: {right: float(flips[i, j] / len(stratified)) for j, right in enumerate(groups)}
            for i, left in enumerate(groups)
        },
        "vectors": results,
    }


def write_v2_reports(
    episodes: Sequence[Mapping[str, Any]],
    spec: SnqiV2Spec,
    reports_dir: Path,
    *,
    bootstrap_samples: int = 2000,
) -> dict[str, str]:
    """Emit the inseparable diagnostics/family pair, returning their artifact paths.

    Returns:
        Validated result described above.
    """
    family = build_family_report(episodes, spec, bootstrap_samples=bootstrap_samples)
    scored = [score_episode(episode, spec) for episode in episodes]
    terms = {term: [ep["metrics"]["snqi_v2_terms"][term] for ep in scored] for term in TERMS}
    diagnostics = {
        "schema_version": "snqi-v2-diagnostics.v1",
        "claim_boundary": CLAIM_BOUNDARY,
        "provenance": spec.provenance(),
        "episode_count": len(scored),
        "family_report": "snqi_v2_family.json",
        "sources": spec.sources,
        "normalized_term_means": {key: float(np.mean(values)) for key, values in terms.items()},
        "clipped_at_one_fraction": {
            key: float(np.mean(np.array(values) >= 1)) for key, values in terms.items()
        },
        "term_spearman": {a: {b: _rho(terms[a], terms[b]) for b in TERMS} for a in TERMS},
        "strata_counts": {
            "collision": sum(ep["metrics"]["snqi_v2_terms"]["C"] > 0 for ep in scored),
            "collision_free_success": sum(
                ep["metrics"]["snqi_v2_terms"]["C"] == 0
                and ep["metrics"]["snqi_v2_terms"]["S"] == 1
                for ep in scored
            ),
            "collision_free_failure": sum(
                ep["metrics"]["snqi_v2_terms"]["C"] == 0
                and ep["metrics"]["snqi_v2_terms"]["S"] == 0
                for ep in scored
            ),
        },
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    for name, payload in (("family", family), ("diagnostics", diagnostics)):
        json_path = reports_dir / f"snqi_v2_{name}.json"
        md_path = reports_dir / f"snqi_v2_{name}.md"
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        json_path.write_text(text, encoding="utf-8")
        _write_markdown_report(md_path, name, payload)
        artifacts[f"snqi_v2_{name}_json"] = str(json_path)
        artifacts[f"snqi_v2_{name}_md"] = str(md_path)
    return artifacts


def read_episode_files(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Read JSONL strictly, preserving undefined numeric values for the score validator.

    Returns:
        Validated result described above.
    """
    records = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected episode object")
            records.append(record)
    if not records:
        raise ValueError("SNQI-v2 requires at least one episode")
    return records


def enrich_campaign_v2(
    run_entries: Sequence[Mapping[str, Any]],
    spec: SnqiV2Spec,
    reports_dir: Path,
    *,
    repo_root: Path,
    bootstrap_samples: int = 2000,
) -> dict[str, str]:
    """Validate all native run records before enriching JSONL and writing both reports.

    Writes use sibling temporary files and atomic replacement. Repeating enrichment
    is idempotent; legacy metrics and episode identity remain intact.

    Returns:
        Validated result described above.
    """
    files: dict[Path, list[dict[str, Any]]] = {}
    all_records = []
    for entry in run_entries:
        if entry.get("status") != "ok":
            raise ValueError("SNQI-v2 refuses failed, unavailable or degraded campaign arms")
        path_value = entry.get("episodes_path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError("SNQI-v2 run requires episodes_path")
        path = (repo_root / path_value).resolve()
        if path in files:
            raise ValueError("SNQI-v2 duplicate campaign episode path")
        if (
            "summary" in entry
            and not summarize_benchmark_availability(entry["summary"]).benchmark_success
        ):
            raise ValueError("SNQI-v2 refuses fallback/degraded run summaries")
        records = read_episode_files([path])
        planner = entry.get("planner", {})
        if not planner.get("key"):
            raise ValueError("SNQI-v2 run requires explicit planner.key")
        enriched = [
            score_episode(
                {**ep, "planner_key": planner["key"], "kinematics": planner.get("kinematics")}, spec
            )
            for ep in records
        ]
        spec.validate_evaluation_seeds([ep["seed"] for ep in records])
        files[path] = enriched
        all_records.extend(
            {**ep, "planner_key": planner["key"], "kinematics": planner.get("kinematics")}
            for ep in enriched
        )
    artifacts = write_v2_reports(
        all_records, spec, reports_dir, bootstrap_samples=bootstrap_samples
    )
    for path, records in files.items():
        temporary = path.with_suffix(path.suffix + ".snqi-v2.tmp")
        temporary.write_text(
            "".join(json.dumps(ep, separators=(",", ":")) + "\n" for ep in records),
            encoding="utf-8",
        )
        temporary.replace(path)
    return artifacts


def _write_markdown_report(path: Path, name: str, payload: Mapping[str, Any]) -> None:
    """Write readable summaries; keep all 2,013 detailed vectors in the companion JSON."""
    lines = [
        f"# SNQI-v2 {name}",
        "",
        CLAIM_BOUNDARY,
        "",
        "Read with [the full family report](snqi_v2_family.json).",
        "",
    ]
    if name == "family":
        lines.extend(
            ["| Rank | Planner | Mean | Seed-bootstrap 95% interval |", "| --- | --- | --- | --- |"]
        )
        for row in payload["declared_ranking"]:
            lines.append(
                f"| {row['rank']} | {row['planner']} | {row['mean_snqi_v2']:.6f} | "
                f"[{row['ci95'][0]:.6f}, {row['ci95'][1]:.6f}] |"
            )
        compact = {
            key: value
            for key, value in payload.items()
            if key not in {"vectors", "bootstrap", "declared_ranking"}
        }
        compact["bootstrap"] = {
            key: value for key, value in payload["bootstrap"].items() if key != "stability"
        }
        compact["leave_one_out_rank_change"] = {
            row["name"]: row["rank_change"] for row in payload["vectors"] if "rank_change" in row
        }
        compact["relaxed_strata"] = [
            row for row in payload["vectors"] if row["kind"] == "relaxed_strata"
        ]
    else:
        compact = dict(payload)
    lines.extend(
        ["", "```json", json.dumps(compact, indent=2, sort_keys=True, allow_nan=False), "```", ""]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_episode_execution(episode: Mapping[str, Any]) -> None:
    """Reject planner fallback markers without interpreting unrelated metric availability.

    Paired-effect metrics can legitimately be unavailable on an otherwise native
    episode. Only the planner status and its runtime subtree determine execution.
    """
    metadata = episode.get("algorithm_metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("SNQI-v2 malformed algorithm metadata")
    direct = {
        key: metadata[key]
        for key in ("status", "execution_mode", "fallback_used", "fallback_triggered", "degraded")
        if key in metadata
    }
    direct["planner_runtime"] = metadata.get("planner_runtime")
    if runtime_fallback_or_degraded_marker(direct) is not None:
        raise ValueError("SNQI-v2 refuses fallback/degraded episode metadata")
