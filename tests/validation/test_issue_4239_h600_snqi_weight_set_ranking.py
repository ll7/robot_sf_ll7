"""Tests for the issue #4239 h600 SNQI weight-set ranking packet builder."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/validation/build_issue_4239_h600_snqi_weight_set_ranking.py"


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "planner_key",
        "seed",
        "success",
        "time_to_goal_norm",
        "collisions",
        "near_misses",
        "comfort_exposure",
        "force_exceed_events",
        "jerk_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _base_rows(planners: list[str]) -> list[dict[str, object]]:
    values = {
        "alpha": (1.0, 0.20, 0.0, 1.0, 0.01, 1.0, 0.10),
        "bravo": (0.8, 0.50, 1.0, 4.0, 0.03, 4.0, 0.40),
        "charlie": (0.9, 0.30, 0.0, 2.0, 0.02, 2.0, 0.20),
        "delta": (0.9, 0.30, 0.0, 2.0, 0.02, 2.0, 0.20),
    }
    rows: list[dict[str, object]] = []
    for planner in planners:
        success, time_norm, collisions, near_misses, comfort, force, jerk = values[planner]
        for seed in (111, 112):
            rows.append(
                {
                    "planner_key": planner,
                    "seed": seed,
                    "success": success,
                    "time_to_goal_norm": time_norm,
                    "collisions": collisions,
                    "near_misses": near_misses,
                    "comfort_exposure": comfort,
                    "force_exceed_events": force,
                    "jerk_mean": jerk,
                }
            )
    return rows


def _fixture(tmp_path: Path, *, tie_rows: bool = False) -> tuple[Path, Path]:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "README.md").write_text("# Fixture evidence\n", encoding="utf-8")
    confirm = tmp_path / "output/confirm/13268/reports/seed_episode_rows.csv"
    extended = tmp_path / "output/extended/13273/reports/seed_episode_rows.csv"
    if tie_rows:
        _write_rows(confirm, _base_rows(["charlie", "delta"]))
        _write_rows(extended, _base_rows(["charlie", "delta"]))
    else:
        _write_rows(confirm, _base_rows(["alpha", "bravo"]))
        _write_rows(extended, _base_rows(["alpha", "bravo", "charlie"]))
    manifest = {
        "schema_version": "issue_4195_h600_aggregation.v1.source_manifest",
        "generated_outputs": ["README.md", "source_manifest.json"],
        "runs": [
            {
                "job_id": "13268",
                "run_label": "confirm",
                "seed_episode_rows": str(confirm),
                "campaign_summary": str(
                    tmp_path / "output/confirm/13268/reports/campaign_summary.json"
                ),
                "campaign": {
                    "scenario_matrix_hash": "scenariohash",
                    "comparability_mapping_hash": "mappinghash",
                },
            },
            {
                "job_id": "13273",
                "run_label": "extended_roster",
                "seed_episode_rows": str(extended),
                "campaign_summary": str(
                    tmp_path / "output/extended/13273/reports/campaign_summary.json"
                ),
                "campaign": {
                    "scenario_matrix_hash": "scenariohash",
                    "comparability_mapping_hash": "mappinghash",
                },
            },
        ],
    }
    source_manifest = evidence / "source_manifest.json"
    _write_json(source_manifest, manifest)
    baseline = tmp_path / "snqi_baseline.json"
    _write_json(
        baseline,
        {
            "collisions": {"med": 0.0, "p95": 2.0},
            "near_misses": {"med": 0.0, "p95": 5.0},
            "force_exceed_events": {"med": 0.0, "p95": 5.0},
            "jerk_mean": {"med": 0.0, "p95": 1.0},
        },
    )
    weights_v2 = tmp_path / "weights_v2.json"
    weights_v3 = tmp_path / "weights_v3.json"
    canonical = tmp_path / "canonical.json"
    _write_json(
        weights_v2,
        {
            "w_success": 0.2,
            "w_time": 0.3,
            "w_collisions": 0.05,
            "w_near": 0.18,
            "w_comfort": 0.15,
            "w_force_exceed": 0.08,
            "w_jerk": 0.04,
        },
    )
    _write_json(
        weights_v3,
        {
            "w_success": 0.2,
            "w_time": 0.1,
            "w_collisions": 0.1,
            "w_near": 0.31,
            "w_comfort": 0.18,
            "w_force_exceed": 0.07,
            "w_jerk": 0.05,
        },
    )
    _write_json(
        canonical,
        {
            "w_success": 1.0,
            "w_time": 0.8,
            "w_collisions": 2.0,
            "w_near": 1.0,
            "w_comfort": 0.5,
            "w_force_exceed": 1.5,
            "w_jerk": 3.0,
        },
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": "h600-snqi-weight-set-ranking.v1",
                "issue": 4239,
                "source_manifest": str(source_manifest),
                "claim_boundary": "Diagnostic h600 fixture only; no canonical SNQI decision.",
                "weight_sets": [
                    {"id": "default_uniform_1p0", "kind": "synthetic_uniform", "weights": None},
                    {"id": "camera_ready_v2", "kind": "shipped_json", "path": str(weights_v2)},
                    {"id": "camera_ready_v3", "kind": "shipped_json", "path": str(weights_v3)},
                    {
                        "id": "model_canonical_v1",
                        "kind": "shipped_json",
                        "path": str(canonical),
                        "caveat": "known jerk-artifact and canonical-label conflict source",
                    },
                ],
                "baseline": {"path": str(baseline)},
                "run_inputs": [
                    {"run_label": "confirm", "job_id": 13268},
                    {"run_label": "extended_roster", "job_id": 13273},
                ],
                "deduplication": {"tolerance": 1e-9},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config, evidence


def _run_builder(config: Path, evidence: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config),
            "--output-dir",
            str(evidence),
            "--generated-at",
            "2026-07-03T00:00:00Z",
        ],
        check=False,
        text=True,
        capture_output=True,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_happy_path_writes_rank_agreement_audit_snippet_and_checksums(tmp_path: Path) -> None:
    """Fixture raw rows produce the full compact issue #4239 packet."""
    config, evidence = _fixture(tmp_path)

    result = _run_builder(config, evidence)

    assert result.returncode == 0, result.stderr
    expected = {
        "snqi_weight_set_h600_preflight.json",
        "snqi_weight_set_h600_deduplication_audit.csv",
        "snqi_weight_set_h600_deduplication_audit.md",
        "snqi_weight_set_h600_rank_table.csv",
        "snqi_weight_set_h600_rank_table.md",
        "snqi_weight_set_h600_pairwise_agreement.csv",
        "snqi_weight_set_h600_pairwise_agreement.md",
        "snqi_weight_set_h600_report.json",
        "snqi_weight_set_h600_diss331_comment.md",
        "SHA256SUMS",
    }
    assert expected <= {path.name for path in evidence.iterdir()}
    preflight = json.loads((evidence / "snqi_weight_set_h600_preflight.json").read_text())
    assert preflight["status"] == "ready"
    rank_rows = _read_csv(evidence / "snqi_weight_set_h600_rank_table.csv")
    assert {row["weight_set_id"] for row in rank_rows} == {
        "default_uniform_1p0",
        "camera_ready_v2",
        "camera_ready_v3",
        "model_canonical_v1",
    }
    assert {row["planner_key"] for row in rank_rows} == {"alpha", "bravo", "charlie"}
    assert "extended_13273" in {row["source_run"] for row in rank_rows}
    agreement_rows = _read_csv(evidence / "snqi_weight_set_h600_pairwise_agreement.csv")
    assert len(agreement_rows) == 6
    snippet = (evidence / "snqi_weight_set_h600_diss331_comment.md").read_text(encoding="utf-8")
    assert "does not choose canonical weights" in snippet
    sums = (evidence / "SHA256SUMS").read_text(encoding="utf-8")
    assert "snqi_weight_set_h600_rank_table.csv" in sums
    assert "seed_episode_rows.csv" not in {path.name for path in evidence.iterdir()}


def test_shared_planner_mismatch_fails_closed_with_audit(tmp_path: Path) -> None:
    """Shared planners are not double-counted when confirm and extended metrics disagree."""
    config, evidence = _fixture(tmp_path)
    extended = tmp_path / "output/extended/13273/reports/seed_episode_rows.csv"
    rows = _base_rows(["alpha", "bravo", "charlie"])
    rows[0]["near_misses"] = 3.0
    _write_rows(extended, rows)

    result = _run_builder(config, evidence)

    assert result.returncode == 2
    preflight = json.loads((evidence / "snqi_weight_set_h600_preflight.json").read_text())
    assert preflight["status"] == "blocked_shared_planner_mismatch"
    audit = _read_csv(evidence / "snqi_weight_set_h600_deduplication_audit.csv")
    assert any(row["status"] == "mismatch" for row in audit)
    assert not (evidence / "snqi_weight_set_h600_rank_table.csv").exists()


def test_missing_time_to_goal_norm_blocks_without_zero_fill(tmp_path: Path) -> None:
    """The all-ones set needs time_to_goal_norm and must not silently fill it as zero."""
    config, evidence = _fixture(tmp_path)
    confirm = tmp_path / "output/confirm/13268/reports/seed_episode_rows.csv"
    rows = _base_rows(["alpha", "bravo"])
    for row in rows:
        row["time_to_goal_norm"] = ""
    _write_rows(confirm, rows)

    result = _run_builder(config, evidence)

    assert result.returncode == 2
    preflight = json.loads((evidence / "snqi_weight_set_h600_preflight.json").read_text())
    assert preflight["status"] == "blocked_missing_snqi_terms"
    assert any(
        issue["weight_set_id"] == "default_uniform_1p0"
        and "time_to_goal_norm" in issue["missing_terms"]
        for issue in preflight["issues"]
    )


def test_missing_jerk_blocks_model_canonical_v1(tmp_path: Path) -> None:
    """The model canonical set has an active jerk weight, so missing jerk_mean blocks it."""
    config, evidence = _fixture(tmp_path)
    confirm = tmp_path / "output/confirm/13268/reports/seed_episode_rows.csv"
    rows = _base_rows(["alpha", "bravo"])
    for row in rows:
        row["jerk_mean"] = ""
    _write_rows(confirm, rows)

    result = _run_builder(config, evidence)

    assert result.returncode == 2
    preflight = json.loads((evidence / "snqi_weight_set_h600_preflight.json").read_text())
    assert any(
        issue["weight_set_id"] == "model_canonical_v1" and "jerk_mean" in issue["missing_terms"]
        for issue in preflight["issues"]
    )


def test_non_mapping_config_fails_closed(tmp_path: Path) -> None:
    """A config that parses as a non-mapping (e.g. a YAML list) is rejected, not crashed on."""
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(["not", "a", "mapping"]), encoding="utf-8")
    evidence = tmp_path / "evidence"

    result = _run_builder(config, evidence)

    assert result.returncode != 0
    assert "must be a mapping" in result.stderr


def test_uniform_all_ones_distinct_from_model_canonical_and_ties_deterministic(
    tmp_path: Path,
) -> None:
    """Uniform all-ones is explicit and deterministic ties use averaged ranks."""
    config, evidence = _fixture(tmp_path, tie_rows=True)

    result = _run_builder(config, evidence)

    assert result.returncode == 0, result.stderr
    report = json.loads((evidence / "snqi_weight_set_h600_report.json").read_text())
    weights = {item["id"]: item["weights"] for item in report["weight_sets"]}
    assert set(weights["default_uniform_1p0"].values()) == {1.0}
    assert weights["default_uniform_1p0"] != weights["model_canonical_v1"]
    rank_rows = _read_csv(evidence / "snqi_weight_set_h600_rank_table.csv")
    uniform_rows = [row for row in rank_rows if row["weight_set_id"] == "default_uniform_1p0"]
    assert [row["planner_key"] for row in uniform_rows] == ["charlie", "delta"]
    assert {row["rank"] for row in uniform_rows} == {"1.5"}
