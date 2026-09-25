"""Deterministic SNQI-v2 family, seed bootstrap, diagnostics and offline records.

Every scored report is emitted together with the preregistered V2-F family.
Undefined source metrics and incomplete planner/seed coverage fail closed.
"""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from robot_sf.benchmark.fallback_policy import (
    summarize_benchmark_availability,
)
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.result_provenance import (
    manifest_path_for_result_jsonl,
    validate_result_provenance_manifest,
    write_result_provenance_manifest,
)
from robot_sf.benchmark.snqi.bootstrap import bootstrap_stability
from robot_sf.benchmark.snqi.compute import compute_snqi_v2, normalize_snqi_v2_terms
from robot_sf.benchmark.snqi.v2_spec import (
    FAMILY,
    QUALITY_TERMS,
    TERMS,
    WEIGHTS,
    SnqiV2Spec,
    parse_v2_json,
)
from robot_sf.benchmark.spawn_validity import (
    RESPAWN_COLLISION_WINDOW_S,
    SPAWN_VALIDITY_SCHEMA_VERSION,
)

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


def score_episode(
    episode: Mapping[str, Any], spec: SnqiV2Spec, *, expected_algorithm: str | None = None
) -> dict[str, Any]:
    """Copy an episode and add v2 fields while preserving every legacy metric.

    Returns:
        Validated result described above.
    """
    validate_episode_execution(episode, expected_algorithm=expected_algorithm)
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
    episodes: Sequence[Mapping[str, Any]],
    spec: SnqiV2Spec,
    *,
    bootstrap_samples: int = 2000,
    expected_algorithms: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compute family sensitivity and paired seed-bootstrap intervals from raw records.

    Returns:
        Validated result described above.
    """
    if not episodes or bootstrap_samples < 1:
        raise ValueError("SNQI-v2 family needs episodes and positive bootstrap samples")
    scored = [
        score_episode(
            episode, spec, expected_algorithm=(expected_algorithms or {}).get(_planner(episode))
        )
        for episode in episodes
    ]
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
    expected_algorithms: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Emit the inseparable diagnostics/family pair, returning their artifact paths.

    Returns:
        Validated result described above.
    """
    family = build_family_report(
        episodes, spec, bootstrap_samples=bootstrap_samples, expected_algorithms=expected_algorithms
    )
    scored = [
        score_episode(
            episode, spec, expected_algorithm=(expected_algorithms or {}).get(_planner(episode))
        )
        for episode in episodes
    ]
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


def read_episode_files(paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    """Yield JSONL objects one at a time without retaining raw lines or prior episodes."""
    seen = False
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = parse_v2_json(line)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: expected episode object")
                seen = True
                yield record
    if not seen:
        raise ValueError("SNQI-v2 requires at least one episode")


def compact_report_episode(
    episode: Mapping[str, Any], spec: SnqiV2Spec, *, expected_algorithm: str | None = None
) -> dict[str, Any]:
    """Validate a raw episode and retain only report identities and scalar score inputs.

    Force samples, simulation/planner traces and unrelated metrics never enter the
    report working set. Runtime validation precedes projection so dropping metadata
    cannot hide a fallback marker.

    Returns:
        An independent compact record accepted by the unchanged report calculations.
    """
    scored = score_episode(episode, spec, expected_algorithm=expected_algorithm)
    return {
        **{
            key: scored[key]
            for key in (
                "planner_key",
                "algo",
                "kinematics",
                "scenario_id",
                "seed",
                "steps",
                "episode_id",
                "config_hash",
                "git_hash",
            )
            if key in scored
        },
        "metrics": {source: scored["metrics"].get(source) for source in spec.sources.values()},
    }


def _stage_v2_file(
    path: Path, temporary: Path, spec: SnqiV2Spec, planner: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Stream one enriched JSONL file.

    Returns:
        Compact scalar report records with producer row identities.
    """
    records = []
    with temporary.open("w", encoding="utf-8") as output:
        for episode in read_episode_files([path]):
            spec.validate_evaluation_seeds([episode["seed"]])
            declared_kinematics = planner.get("kinematics")
            row_kinematics = episode.get("kinematics")
            if (
                declared_kinematics is not None
                and row_kinematics is not None
                and row_kinematics != declared_kinematics
            ):
                raise ValueError("SNQI-v2 planner/episode kinematics identity mismatch")
            kinematics = declared_kinematics if declared_kinematics is not None else row_kinematics
            scored_episode = {**episode, "planner_key": planner["key"]}
            if kinematics is not None:
                scored_episode["kinematics"] = kinematics
            enriched = score_episode(
                scored_episode,
                spec,
                expected_algorithm=planner.get("algo"),
            )
            output.write(json.dumps(enriched, separators=(",", ":")) + "\n")
            records.append(
                compact_report_episode(enriched, spec, expected_algorithm=planner.get("algo"))
            )
    return records


def _temporary_sibling(path: Path) -> Path:
    """Reserve a task-owned sibling for atomic publication.

    Returns:
        Path of the newly created empty temporary file.
    """
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".snqi-v2.tmp", delete=False
    ) as temporary:
        return Path(temporary.name)


def _stage_v2_provenance(
    path: Path,
    temporary: Path,
    sidecar: Path,
    staged_sidecar: Path,
    spec: SnqiV2Spec,
    records: Sequence[Mapping[str, Any]],
    original_hash: str,
) -> None:
    """Rebind a verified producer sidecar with explicit enrichment lineage."""
    payload = parse_v2_json(sidecar.read_text(encoding="utf-8"))
    validate_result_provenance_manifest(payload)
    artifacts = [
        entry for entry in payload["raw_artifacts"] if entry.get("kind") == "episodes_jsonl"
    ]
    if (
        len(artifacts) != 1
        or artifacts[0].get("sha256") != original_hash
        or Path(artifacts[0].get("path", "")).resolve() != path
    ):
        raise ValueError("SNQI-v2 producer sidecar does not bind the original episode file")
    if len(payload["rows"]) != len(records):
        raise ValueError("SNQI-v2 producer sidecar row count mismatch")
    for index, (bound, row) in enumerate(zip(payload["rows"], records, strict=True)):
        if (
            bound.get("jsonl_line") != index
            or any(
                bound.get(key) != row.get(key)
                for key in ("episode_id", "scenario_id", "seed", "config_hash")
            )
            or bound.get("repo_commit") != row.get("git_hash")
        ):
            raise ValueError("SNQI-v2 producer sidecar row identity mismatch")
    enriched_hash = sha256_file(temporary)
    previous = payload.get("snqi_v2_enrichment", {})
    if previous and previous.get("output_sha256") != original_hash:
        raise ValueError("SNQI-v2 existing enrichment lineage is stale")
    payload["snqi_v2_enrichment"] = {
        "operation": "add_snqi_v2_fields",
        "input_sha256": previous.get("input_sha256", original_hash),
        "output_sha256": enriched_hash,
        "episode_count": len(records),
        "specification": spec.provenance(),
    }
    artifacts[0]["sha256"] = enriched_hash
    validate_result_provenance_manifest(payload)
    write_result_provenance_manifest(staged_sidecar, payload)


def _validated_run_input(
    entry: Mapping[str, Any], repo_root: Path
) -> tuple[Path, Mapping[str, Any]]:
    """Check arm availability and its explicit file and planner identity.

    Returns:
        Resolved episode file and planner descriptor.
    """
    if entry.get("status") != "ok":
        raise ValueError("SNQI-v2 refuses failed, unavailable or degraded campaign arms")
    path_value = entry.get("episodes_path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("SNQI-v2 run requires episodes_path")
    if (
        "summary" in entry
        and not summarize_benchmark_availability(entry["summary"]).benchmark_success
    ):
        raise ValueError("SNQI-v2 refuses fallback/degraded run summaries")
    planner = entry.get("planner", {})
    if (
        not isinstance(planner, Mapping)
        or not isinstance(planner.get("key"), str)
        or not planner["key"]
        or (planner.get("algo") is not None and not isinstance(planner["algo"], str))
    ):
        raise ValueError("SNQI-v2 run requires explicit planner.key and a string planner.algo")
    return (repo_root / path_value).resolve(), planner


def _validated_v2_record_algorithms(
    records: Sequence[Mapping[str, Any]],
    planner: Mapping[str, Any],
    planner_identities: set[str],
) -> dict[str, str]:
    """Validate one staged run's identities and return its algorithm bindings.

    Returns:
        Mapping from each validated planner/kinematics identity to its algorithm.
    """
    identities = {_planner(record) for record in records}
    algorithms = [record.get("algo") for record in records]
    if (
        len(identities) != 1
        or any(not isinstance(algo, str) or not algo for algo in algorithms)
        or len(set(algorithms)) != 1
    ):
        raise ValueError("SNQI-v2 run must have one planner/kinematics identity and one algorithm")
    identity = next(iter(identities))
    if identity in planner_identities:
        raise ValueError("SNQI-v2 duplicate planner/kinematics identity")
    planner_identities.add(identity)
    algorithm = algorithms[0]
    declared_algorithm = planner.get("algo")
    bound_algorithm = declared_algorithm if declared_algorithm is not None else algorithm
    return {_planner(record): bound_algorithm for record in records}


def enrich_campaign_v2(
    run_entries: Sequence[Mapping[str, Any]],
    spec: SnqiV2Spec,
    reports_dir: Path,
    *,
    repo_root: Path,
    bootstrap_samples: int = 2000,
) -> dict[str, str]:
    """Stage enriched files one episode at a time, then validate the complete report pair.

    Memory scales with scalar score inputs and identities plus the largest episode,
    not the complete decoded force/trace payload. All source files remain untouched
    until scoring and paired-coverage checks pass. Sibling files are then atomically
    replaced individually; this is not a multi-file filesystem transaction.
    Repeating enrichment is idempotent and preserves legacy field values.

    Returns:
        Paths to the mandatory diagnostics and family JSON/Markdown artifacts.
    """
    staged: dict[Path, Path] = {}
    all_records = []
    original_hashes: dict[Path, str] = {}
    expected_algorithms = {}
    planner_identities: set[str] = set()
    try:
        for entry in run_entries:
            path, planner = _validated_run_input(entry, repo_root)
            if path in staged:
                raise ValueError("SNQI-v2 duplicate campaign episode path")
            sidecar = manifest_path_for_result_jsonl(path)
            if not sidecar.is_file():
                raise ValueError("SNQI-v2 requires the producer provenance sidecar")
            original_hashes[path] = sha256_file(path)
            original_hashes[sidecar] = sha256_file(sidecar)
            staged[path] = _temporary_sibling(path)
            records = _stage_v2_file(path, staged[path], spec, planner)
            expected_algorithms.update(
                _validated_v2_record_algorithms(records, planner, planner_identities)
            )
            staged[sidecar] = _temporary_sibling(sidecar)
            _stage_v2_provenance(
                path, staged[path], sidecar, staged[sidecar], spec, records, original_hashes[path]
            )
            all_records.extend(records)
        artifacts = write_v2_reports(
            all_records,
            spec,
            reports_dir,
            bootstrap_samples=bootstrap_samples,
            expected_algorithms=expected_algorithms,
        )
        if any(sha256_file(path) != digest for path, digest in original_hashes.items()):
            raise ValueError("SNQI-v2 source or sidecar changed during enrichment")
        for path, temporary in staged.items():
            temporary.replace(path)
        return artifacts
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


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


def _spawn_route_completed(episode: Mapping[str, Any]) -> bool:
    """Bind the producer exception to explicit outcome and termination status.

    Returns:
        Whether the row consistently declares a completed route.
    """
    return (
        episode.get("status") == "success"
        and episode["outcome"]["route_complete"] is True
        and episode["outcome"]["timeout_event"] is False
    )


def _validate_spawn_outcome(episode: Mapping[str, Any]) -> None:
    """Present producer spawn blocks require the accompanying canonical outcome."""
    outcome = episode.get("outcome")
    if not isinstance(outcome, Mapping) or any(
        not isinstance(outcome.get(key), bool)
        for key in ("route_complete", "collision_event", "timeout_event")
    ):
        raise ValueError("SNQI-v2 malformed spawn_validity: canonical outcome required")


def _spawn_clearance_negative(value: Any, *, obstacle: bool = False) -> bool:
    """Validate a producer clearance scalar and return whether it denotes contact.

    Returns:
        Whether the measured clearance is negative; absent measurements are false.
    """
    if value is None:
        return False
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("SNQI-v2 malformed spawn_validity: invalid clearance scalar")
    if not math.isfinite(value) and not (obstacle and value == math.inf):
        raise ValueError("SNQI-v2 malformed spawn_validity: nonfinite clearance")
    return value < 0.0


def _validate_reset_clearance(clearance: Mapping[str, Any]) -> None:
    """Check the verdict and geometry fields emitted by reset_spawn_clearance."""
    required = {
        "robot_pedestrian_min_surface_clearance_m",
        "robot_obstacle_min_surface_clearance_m",
        "overlapping_pedestrian_rows",
        "pedestrian_overlap",
        "obstacle_overlap",
        "overlap",
    }
    if not required.issubset(clearance):
        raise ValueError("SNQI-v2 malformed spawn_validity: incomplete reset clearance")
    rows = clearance["overlapping_pedestrian_rows"]
    if (
        not isinstance(rows, list)
        or any(isinstance(row, bool) or not isinstance(row, int) or row < 0 for row in rows)
        or rows != sorted(set(rows))
    ):
        raise ValueError("SNQI-v2 malformed spawn_validity: invalid overlapping pedestrian rows")
    pedestrian = _spawn_clearance_negative(clearance["robot_pedestrian_min_surface_clearance_m"])
    obstacle = _spawn_clearance_negative(
        clearance["robot_obstacle_min_surface_clearance_m"], obstacle=True
    )
    if not (
        pedestrian == bool(rows)
        and clearance["pedestrian_overlap"] is pedestrian
        and clearance["obstacle_overlap"] is obstacle
        and clearance["overlap"] is (pedestrian or obstacle)
    ):
        raise ValueError("SNQI-v2 inconsistent spawn_validity: reset clearance verdict disagrees")


def _validate_spawn_validity_shape(block: Mapping[str, Any]) -> None:
    """Require the complete schema emitted by build_spawn_validity for present blocks."""
    required = {
        "schema_version",
        "reset_clearance",
        "reset_clearance_status",
        "reset_clearance_error",
        "reset_overlap",
        "respawn_overlap_events",
        "respawn_overlap_collisions",
        "invalid_run",
        "invalid_reason",
    }
    if not required.issubset(block):
        raise ValueError("SNQI-v2 malformed spawn_validity: missing producer fields")
    clearance = block["reset_clearance"]
    available = isinstance(clearance, Mapping) and bool(clearance)
    checks = (
        block["schema_version"] == SPAWN_VALIDITY_SCHEMA_VERSION,
        isinstance(block["invalid_run"], bool),
        block["invalid_reason"] is None or isinstance(block["invalid_reason"], str),
        isinstance(block["reset_overlap"], bool),
        clearance is None or (available and isinstance(clearance.get("overlap"), bool)),
        clearance is not None or block["reset_overlap"] is False,
        block["reset_clearance_status"] == ("available" if available else "unavailable"),
        block["reset_clearance_error"] is None
        or (not available and isinstance(block["reset_clearance_error"], str)),
    )
    if not all(checks):
        raise ValueError("SNQI-v2 malformed spawn_validity: invalid producer field types")
    if clearance is not None:
        _validate_reset_clearance(clearance)
    for key in ("respawn_overlap_events", "respawn_overlap_collisions"):
        events = block[key]
        if not isinstance(events, list) or any(not isinstance(event, Mapping) for event in events):
            raise ValueError("SNQI-v2 malformed spawn_validity: invalid respawn telemetry")


def _validate_respawn_event(event: Mapping[str, Any]) -> None:
    """Validate every event emitted by pedestrian respawn, even without collisions."""
    if not {"group_id", "ped_rows", "step", "positions"}.issubset(event):
        raise ValueError("SNQI-v2 malformed spawn_validity: incomplete respawn event")
    if any(
        isinstance(event[key], bool) or not isinstance(event[key], int) or event[key] < 0
        for key in ("group_id", "step")
    ):
        raise ValueError("SNQI-v2 malformed spawn_validity: invalid respawn identity/step")
    rows = event["ped_rows"]
    if (
        not isinstance(rows, list)
        or any(isinstance(row, bool) or not isinstance(row, int) or row < 0 for row in rows)
        or rows != sorted(set(rows))
    ):
        raise ValueError("SNQI-v2 malformed spawn_validity: invalid respawn pedestrian rows")
    positions = event["positions"]
    if (
        not isinstance(positions, list)
        or len(positions) != len(rows)
        or any(
            not isinstance(point, list)
            or len(point) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in point
            )
            for point in positions
        )
    ):
        raise ValueError("SNQI-v2 malformed spawn_validity: invalid respawn positions")


def _validate_respawn_attribution(episode: Mapping[str, Any], block: Mapping[str, Any]) -> None:
    """Bind attributed collisions to declared respawn group, pedestrian and timing."""
    for event in block["respawn_overlap_events"]:
        _validate_respawn_event(event)
    collisions = block["respawn_overlap_collisions"]
    if not collisions:
        return
    params = episode.get("scenario_params", {})
    dt = params.get("run_dt") if isinstance(params, Mapping) else None
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("SNQI-v2 malformed spawn_validity: respawn attribution requires run_dt")
    for collision in collisions:
        _validate_respawn_collision(collision, block["respawn_overlap_events"], float(dt))


def _validate_respawn_collision(
    collision: Mapping[str, Any], events: list[Mapping[str, Any]], dt: float
) -> None:
    """Refuse attribution without matching source event and the producer timing window."""
    group, row = collision.get("group_id"), collision.get("ped_row")
    times = [collision.get("respawn_time_s"), collision.get("collision_time_s")]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (group, row)
    ) or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in times
    ):
        raise ValueError("SNQI-v2 malformed spawn_validity: invalid attributed collision")
    respawn_time, collision_time = times
    matched = any(
        event["group_id"] == group
        and row in event["ped_rows"]
        and event["step"] * dt == respawn_time
        for event in events
    )
    elapsed = collision_time - respawn_time
    if not matched or not -dt - 1e-9 <= elapsed <= RESPAWN_COLLISION_WINDOW_S + 1e-9:
        raise ValueError("SNQI-v2 inconsistent spawn_validity: unmatched respawn collision")


_MISSING_SPAWN_VALIDITY = object()


def _spawn_validity_block(episode: Mapping[str, Any], *, required: bool) -> Any:
    """Return producer metadata, preserving the legacy-absence sentinel."""
    if "spawn_validity" not in episode:
        if required:
            raise ValueError("SNQI-v2 calibration requires spawn_validity producer metadata")
        return _MISSING_SPAWN_VALIDITY
    return episode["spawn_validity"]


def _validate_spawn_validity(episode: Mapping[str, Any], *, required: bool = False) -> None:
    """Refuse invalid or ambiguous spawn admission while preserving legacy absence."""
    block = _spawn_validity_block(episode, required=required)
    if block is _MISSING_SPAWN_VALIDITY:
        return
    if not isinstance(block, Mapping) or not isinstance(block.get("invalid_run"), bool):
        raise ValueError("SNQI-v2 malformed spawn_validity: explicit boolean invalid_run required")
    _validate_spawn_validity_shape(block)
    _validate_spawn_outcome(episode)
    _validate_respawn_attribution(episode, block)
    if block["invalid_run"]:
        raise ValueError("SNQI-v2 refuses spawn_validity.invalid_run episode")
    if block.get("invalid_reason") is not None:
        raise ValueError("SNQI-v2 malformed spawn_validity: valid row has an invalid_reason")
    if "schema_version" in block and block["schema_version"] != SPAWN_VALIDITY_SCHEMA_VERSION:
        raise ValueError("SNQI-v2 malformed spawn_validity: unsupported schema_version")
    reset_overlap = block.get("reset_overlap", False)
    respawn_collisions = block.get("respawn_overlap_collisions", [])
    if not isinstance(reset_overlap, bool) or not isinstance(respawn_collisions, list):
        raise ValueError("SNQI-v2 malformed spawn_validity: invalid overlap telemetry")
    clearance = block.get("reset_clearance")
    if clearance is not None and (
        not isinstance(clearance, Mapping)
        or clearance.get("overlap", reset_overlap) is not reset_overlap
    ):
        raise ValueError("SNQI-v2 inconsistent spawn_validity: reset overlap disagrees")
    if reset_overlap or respawn_collisions:
        # The producer exempts completed routes, including earlier collisions;
        # a non-success termination cannot claim that exception.
        if not _spawn_route_completed(episode):
            raise ValueError("SNQI-v2 inconsistent spawn_validity: overlap row marked valid")


def validate_episode_execution(
    episode: Mapping[str, Any],
    *,
    expected_algorithm: str | None = None,
    require_spawn_validity: bool = False,
) -> None:
    """Apply canonical execution classification with independently declared arm identity.

    Direct callers get no guarded exception without context. Import lazily because
    release acceptance imports campaign entrypoints, which in turn consume reports.
    Metric availability is separate from planner execution and is not scanned.
    """
    from robot_sf.benchmark.release_acceptance import _status_markers  # noqa: PLC0415

    _validate_spawn_validity(episode, required=require_spawn_validity)
    if not isinstance(episode.get("algorithm_metadata", {}), Mapping):
        raise ValueError("SNQI-v2 malformed algorithm metadata")
    # This companion describes optional posthoc metrics, not planner execution.
    metadata = episode.get("algorithm_metadata", {})
    execution = {
        **episode,
        "algorithm_metadata": {
            key: value for key, value in metadata.items() if key != "paired_effect_metric_producer"
        },
    }
    if _status_markers(execution, "episode", expected_algorithm=expected_algorithm):
        raise ValueError("SNQI-v2 refuses fallback/degraded episode metadata")
