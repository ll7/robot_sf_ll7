"""Tests issue #4206 mechanism-level policy-structure cross-cut builder."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml

from robot_sf.benchmark.failure_mechanism_taxonomy import (
    GEOMETRY_ONLY_FIELDS,
    MECHANISM_SCHEMA_VERSION,
    REQUIRED_MECHANISM_FIELDS,
    TRACE_VERIFIED_EVIDENCE_MODES,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/validation/build_issue_4206_policy_structure_mechanism_crosscut.py"
CONFIG = REPO_ROOT / "configs/analysis/issue_4206_policy_structure_mechanism_crosscut.yaml"

_SPEC = importlib.util.spec_from_file_location("_issue_4206_builder", SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_builder_uses_canonical_failure_mechanism_taxonomy_contract() -> None:
    """Builder and trace-capable pre-registration validator share one schema owner."""
    assert _MODULE.MECHANISM_SCHEMA_VERSION == MECHANISM_SCHEMA_VERSION
    assert _MODULE.REQUIRED_MECHANISM_FIELDS == REQUIRED_MECHANISM_FIELDS
    assert _MODULE.TRACE_VERIFIED_EVIDENCE_MODES == TRACE_VERIFIED_EVIDENCE_MODES
    assert set(_MODULE.GEOMETRY_ONLY_FIELDS) == GEOMETRY_ONLY_FIELDS


def _write_rows(root: Path, rows: list[dict[str, object]]) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True)
    fields = sorted({field for row in rows for field in row})
    with (reports / "seed_episode_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (reports / "campaign_summary.json").write_text('{"status": "fixture"}\n', encoding="utf-8")


def _base_row(planner_key: str, success: float, collision: float = 0.0) -> dict[str, object]:
    return {
        "episode_id": f"ep-{planner_key}",
        "scenario_id": "classic_static_deadlock",
        "planner_key": planner_key,
        "seed": 1,
        "repeat_index": 0,
        "success": success,
        "collision": collision,
        "near_miss": 0,
        "timeout": 1.0 - success,
        "progress": success,
        "snqi": success - collision,
        "geometry_bucket": "cross_trap",
        "mechanism_schema_version": "failure_mechanism_taxonomy.v1",
        "mechanism_label": "static_deadlock_or_local_minimum",
        "mechanism_confidence": "observed_mechanism",
        "mechanism_evidence_mode": "paired_trace",
        "mechanism_evidence_uri": "docs/context/issue_2220_failure_mechanism_taxonomy.md",
        "mechanism_case_id": f"fixture-{planner_key}",
        "mechanism_caveat": "",
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_happy_path_writes_expected_outputs_and_checksums(tmp_path: Path) -> None:
    """Trace-labeled rows produce rank/probe/agreement tables and checksums."""
    confirm = tmp_path / "confirm"
    extended = tmp_path / "extended"
    rows = [
        _base_row("prediction_planner", 1.0),
        _base_row("hybrid_rule_v3_fast_progress_static_escape", 0.4),
        _base_row("ppo", 0.7),
        _base_row("orca", 0.2, collision=1.0),
    ]
    _write_rows(confirm, rows)
    _write_rows(extended, rows)
    packet = tmp_path / "packet.json"
    packet.write_text('{"source_job_id": 13175}\n', encoding="utf-8")
    output = tmp_path / "evidence"

    summary = _MODULE.build_packet(
        config_path=CONFIG,
        confirm_root=confirm,
        extended_root=extended,
        job13175_packet=packet,
        output_dir=output,
        generated_at="2026-07-03T00:00:00Z",
    )

    assert summary["status"] == "analysis_ready_trace_verified"
    expected = {
        "README.md",
        "metadata.json",
        "mechanism_crosscut_report.json",
        "mechanism_crosscut_report.md",
        "mechanism_by_structural_class.csv",
        "f_c4ii_probe_predictive_dominance.csv",
        "f_c4ii_probe_local_minimum_failures.csv",
        "geometry_vs_mechanism_agreement.csv",
        "missing_instrumentation.json",
        "claim_boundary.md",
        "SHA256SUMS",
    }
    assert expected <= {path.name for path in output.iterdir()}
    probe = _read_csv(output / "f_c4ii_probe_predictive_dominance.csv")
    assert probe[0]["predictive_beats_both"] == "True"
    local_probe = _read_csv(output / "f_c4ii_probe_local_minimum_failures.csv")
    assert {row["structural_class"] for row in local_probe} >= {
        "predictive",
        "constraint_first_hybrid",
        "learned_policy",
    }
    sums = (output / "SHA256SUMS").read_text(encoding="utf-8")
    digest = hashlib.sha256((output / "metadata.json").read_bytes()).hexdigest()
    assert f"{digest}  metadata.json" in sums


def test_missing_mechanism_label_blocks_f_c4ii_tables(tmp_path: Path) -> None:
    """Rows without required mechanism fields produce a blocked instrumentation packet."""
    confirm = tmp_path / "confirm"
    extended = tmp_path / "extended"
    row = _base_row("prediction_planner", 1.0)
    for field in _MODULE.REQUIRED_MECHANISM_FIELDS:
        row.pop(field, None)
    _write_rows(confirm, [row])
    _write_rows(extended, [row])
    output = tmp_path / "evidence"
    summary = _MODULE.build_packet(
        config_path=CONFIG,
        confirm_root=confirm,
        extended_root=extended,
        job13175_packet=tmp_path / "packet.json",
        output_dir=output,
        generated_at="2026-07-03T00:00:00Z",
    )

    assert summary["status"] == "blocked_missing_trace_verified_mechanism_labels"
    missing = json.loads((output / "missing_instrumentation.json").read_text(encoding="utf-8"))
    assert missing["missing_row_count"] == 2
    assert missing["geometry_bucket_substitution_rejected"] is True
    assert missing["input_provenance"]["config"]["exists"] is True
    assert missing["input_provenance"]["confirm_h600_13268"]["seed_episode_rows_exists"] is True
    assert missing["input_provenance"]["extended_h600_13273"]["seed_episode_rows_exists"] is True
    assert missing["input_provenance"]["continuity_h500_job13175"]["packet_exists"] is False
    assert _read_csv(output / "f_c4ii_probe_predictive_dominance.csv") == []


def test_geometry_bucket_labels_are_rejected_as_substitutes(tmp_path: Path) -> None:
    """Geometry-only labels cannot satisfy the mechanism taxonomy contract."""
    confirm = tmp_path / "confirm"
    extended = tmp_path / "extended"
    row = _base_row("prediction_planner", 1.0)
    for field in _MODULE.REQUIRED_MECHANISM_FIELDS:
        row.pop(field, None)
    row["geometry_label"] = "static_deadlock_or_local_minimum"
    _write_rows(confirm, [row])
    _write_rows(extended, [row])
    output = tmp_path / "evidence"

    _MODULE.build_packet(
        config_path=CONFIG,
        confirm_root=confirm,
        extended_root=extended,
        job13175_packet=tmp_path / "packet.json",
        output_dir=output,
        generated_at="2026-07-03T00:00:00Z",
    )

    missing = json.loads((output / "missing_instrumentation.json").read_text(encoding="utf-8"))
    assert missing["geometry_bucket_substitution_rejected"] is True
    assert "mechanism_label" in missing["required_fields"]


def test_non_trace_verified_mechanism_label_blocks_f_c4ii_tables(tmp_path: Path) -> None:
    """Present labels still fail closed unless trace status is verified."""

    confirm = tmp_path / "confirm"
    extended = tmp_path / "extended"
    row = _base_row("prediction_planner", 1.0)
    row["mechanism_evidence_mode"] = "geometry_proxy"
    _write_rows(confirm, [row])
    _write_rows(extended, [row])
    output = tmp_path / "evidence"

    summary = _MODULE.build_packet(
        config_path=CONFIG,
        confirm_root=confirm,
        extended_root=extended,
        job13175_packet=tmp_path / "packet.json",
        output_dir=output,
        generated_at="2026-07-03T00:00:00Z",
    )

    assert summary["status"] == "blocked_missing_trace_verified_mechanism_labels"
    missing = json.loads((output / "missing_instrumentation.json").read_text(encoding="utf-8"))
    assert (
        "mechanism_evidence_mode=trace_verified_source"
        in missing["missing_rows_sample"][0]["missing_fields"]
    )


def test_unknown_planner_key_is_unclassified_and_excluded(tmp_path: Path) -> None:
    """Unknown planners remain visible in metadata but are excluded from F-C4(ii) ranks."""
    confirm = tmp_path / "confirm"
    extended = tmp_path / "extended"
    rows = [_base_row("prediction_planner", 1.0), _base_row("new_planner", 0.0)]
    _write_rows(confirm, rows)
    _write_rows(extended, rows)
    output = tmp_path / "evidence"
    summary = _MODULE.build_packet(
        config_path=CONFIG,
        confirm_root=confirm,
        extended_root=extended,
        job13175_packet=tmp_path / "packet.json",
        output_dir=output,
        generated_at="2026-07-03T00:00:00Z",
    )

    assert summary["unclassified_planner_rows_excluded_from_f_c4ii"] == 2
    rank_rows = _read_csv(output / "mechanism_by_structural_class.csv")
    assert {row["structural_class"] for row in rank_rows} == {"predictive"}


def test_fallback_rows_remain_visible_not_success_evidence(tmp_path: Path) -> None:
    """Fallback/degraded rows are counted and make cells ineligible for F-C4(ii)."""
    confirm = tmp_path / "confirm"
    extended = tmp_path / "extended"
    row = _base_row("prediction_planner", 1.0)
    row["status"] = "fallback"
    _write_rows(confirm, [row])
    _write_rows(extended, [row])
    output = tmp_path / "evidence"

    _MODULE.build_packet(
        config_path=CONFIG,
        confirm_root=confirm,
        extended_root=extended,
        job13175_packet=tmp_path / "packet.json",
        output_dir=output,
        generated_at="2026-07-03T00:00:00Z",
    )

    rank_rows = _read_csv(output / "mechanism_by_structural_class.csv")
    assert rank_rows[0]["fallback_or_degraded_count"] == "2"
    assert rank_rows[0]["eligible_f_c4ii"] == "False"


def test_sidecar_mechanism_labels_are_accepted(tmp_path: Path) -> None:
    """A conventional reports/mechanism_labels.csv sidecar can supply trace labels."""
    confirm = tmp_path / "confirm"
    extended = tmp_path / "extended"
    row = _base_row("prediction_planner", 1.0)
    sidecar = {field: row.pop(field) for field in _MODULE.REQUIRED_MECHANISM_FIELDS}
    _write_rows(confirm, [row])
    _write_rows(extended, [row])
    for root in (confirm, extended):
        with (root / "reports" / "mechanism_labels.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "scenario_id",
                    "planner_key",
                    "seed",
                    "repeat_index",
                    *_MODULE.REQUIRED_MECHANISM_FIELDS,
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "scenario_id": row["scenario_id"],
                    "planner_key": row["planner_key"],
                    "seed": row["seed"],
                    "repeat_index": row["repeat_index"],
                    **sidecar,
                }
            )
    output = tmp_path / "evidence"

    summary = _MODULE.build_packet(
        config_path=CONFIG,
        confirm_root=confirm,
        extended_root=extended,
        job13175_packet=tmp_path / "packet.json",
        output_dir=output,
        generated_at="2026-07-03T00:00:00Z",
    )

    assert summary["status"] == "analysis_ready_trace_verified"


def test_row_key_preserves_zero_seed_and_repeat_values() -> None:
    """Sidecar identity keys must distinguish zero from missing values."""
    row = {
        "episode_id": 0,
        "scenario_id": "scenario",
        "planner_key": "planner",
        "seed": 0,
        "repeat_index": 0,
    }

    assert _MODULE._row_key(row) == ("0", "scenario", "planner", "0", "0")


def test_progress_mean_preserves_zero_progress_over_ratio() -> None:
    """Zero progress is a real value, not a reason to fall back to progress_ratio."""
    row = _base_row("prediction_planner", 1.0)
    row["structural_class"] = "predictive"
    row["progress"] = 0.0
    row["progress_ratio"] = 1.0

    [summary] = _MODULE._summarize_groups([row])

    assert summary["progress_mean"] == 0.0


def test_sidecar_episode_id_only_does_not_attach_to_other_keys(tmp_path: Path) -> None:
    """Mechanism sidecars must not attach by episode id alone."""
    confirm = tmp_path / "confirm"
    extended = tmp_path / "extended"
    row = _base_row("prediction_planner", 1.0)
    row["episode_id"] = "shared-episode"
    for field in _MODULE.REQUIRED_MECHANISM_FIELDS:
        row.pop(field, None)
    _write_rows(confirm, [row])
    _write_rows(extended, [row])
    sidecar = _base_row("other_planner", 0.0)
    sidecar["episode_id"] = "shared-episode"
    for root in (confirm, extended):
        with (root / "reports" / "mechanism_labels.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["episode_id", *_MODULE.REQUIRED_MECHANISM_FIELDS]
            )
            writer.writeheader()
            writer.writerow(
                {"episode_id": sidecar["episode_id"]}
                | {field: sidecar[field] for field in _MODULE.REQUIRED_MECHANISM_FIELDS}
            )

    summary = _MODULE.build_packet(
        config_path=CONFIG,
        confirm_root=confirm,
        extended_root=extended,
        job13175_packet=tmp_path / "packet.json",
        output_dir=tmp_path / "evidence",
        generated_at="2026-07-03T00:00:00Z",
    )

    assert summary["status"] == "blocked_missing_trace_verified_mechanism_labels"


def test_score_tolerates_missing_snqi_mean() -> None:
    """Rank scoring keeps missing SNQI fail-closed instead of crashing."""
    score = _MODULE._score(
        {
            "success_rate": 1.0,
            "collision_event_rate": 0.0,
            "near_miss_event_rate": 0.0,
            "timeout_rate": 0.0,
            "snqi_mean": None,
        }
    )

    assert score[-1] == 9999.0


def test_geometry_agreement_table_does_not_claim_unverified_agreement() -> None:
    """Geometry comparison table must label uncomputed comparisons explicitly."""
    table = _MODULE._agreement_table(
        [_base_row("prediction_planner", 1.0)],
        [
            {
                "mechanism_label": "static_deadlock_or_local_minimum",
                "structural_class": "predictive",
                "mechanism_rank": 1,
            }
        ],
    )

    assert table[0]["agreement_status"] == "geometry_present_not_rank_compared"
    assert table[0]["conclusion_survives"] == ""


def test_cli_json_summary(tmp_path: Path) -> None:
    """The CLI returns a machine-readable blocked summary when inputs are absent."""
    output = tmp_path / "evidence"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(CONFIG),
            "--confirm-root",
            str(tmp_path / "missing-confirm"),
            "--extended-root",
            str(tmp_path / "missing-extended"),
            "--job13175-packet",
            str(tmp_path / "packet.json"),
            "--output-dir",
            str(output),
            "--generated-at",
            "2026-07-03T00:00:00Z",
            "--json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    # Absent run directories are an artifact-retrieval block, not a missing-label block.
    assert payload["status"] == "blocked_missing_input_artifacts"


def test_missing_input_artifacts_reports_retrieval_block_not_label_block(tmp_path: Path) -> None:
    """Absent run outputs fail closed as a retrieval gap with an artifact-retrieval next action."""
    output = tmp_path / "evidence"

    summary = _MODULE.build_packet(
        config_path=CONFIG,
        confirm_root=tmp_path / "missing-confirm",
        extended_root=tmp_path / "missing-extended",
        job13175_packet=tmp_path / "packet.json",
        output_dir=output,
        generated_at="2026-07-03T00:00:00Z",
    )

    assert summary["status"] == "blocked_missing_input_artifacts"
    missing = json.loads((output / "missing_instrumentation.json").read_text(encoding="utf-8"))
    assert missing["status"] == "blocked_missing_input_artifacts"
    assert missing["blocker_kinds"] == ["missing_input_artifact"]
    # The follow-up must point at hydrating the run outputs, not at exporter instrumentation.
    assert "retriev" in missing["next_action"].lower()
    assert "13268" in missing["followup_issue_skeleton"]["title"]
    # Provenance still records that the declared roots are absent on this host.
    assert missing["input_provenance"]["confirm_h600_13268"]["root_exists"] is False
    assert _read_csv(output / "f_c4ii_probe_predictive_dominance.csv") == []


def test_present_rows_missing_labels_stay_a_label_block(tmp_path: Path) -> None:
    """Rows present but unlabeled stay a mechanism-label block, distinct from the retrieval block."""
    confirm = tmp_path / "confirm"
    extended = tmp_path / "extended"
    row = _base_row("prediction_planner", 1.0)
    for field in _MODULE.REQUIRED_MECHANISM_FIELDS:
        row.pop(field, None)
    _write_rows(confirm, [row])
    _write_rows(extended, [row])
    output = tmp_path / "evidence"

    summary = _MODULE.build_packet(
        config_path=CONFIG,
        confirm_root=confirm,
        extended_root=extended,
        job13175_packet=tmp_path / "packet.json",
        output_dir=output,
        generated_at="2026-07-03T00:00:00Z",
    )

    assert summary["status"] == "blocked_missing_trace_verified_mechanism_labels"
    missing = json.loads((output / "missing_instrumentation.json").read_text(encoding="utf-8"))
    assert missing["blocker_kinds"] == ["missing_mechanism_labels"]
    assert "label" in missing["next_action"].lower()


def _native_writer_episode(
    planner_key: str,
    *,
    seed: int,
    success: float,
    trace_verified: bool,
) -> dict[str, object]:
    """Build one benchmark episode record for the native seed-row writer."""

    taxonomy = {
        "label": "static_deadlock_or_local_minimum",
        "source": "trace_review_issue_2220",
        "trace_status": "trace_verified" if trace_verified else "geometry_only",
        "confidence": "observed_mechanism",
        "case_id": f"fixture-{planner_key}-{seed}",
        "caveat": "",
    }
    return {
        "episode_id": f"ep-{planner_key}-{seed}",
        "scenario_id": "classic_static_deadlock",
        "seed": seed,
        "algo": planner_key,
        "planner_key": planner_key,
        "kinematics": "differential_drive",
        "failure_mechanism_taxonomy": taxonomy,
        "metrics": {
            "success": success,
            "collisions": 0.0,
            "near_misses": 1.0,
            "time_to_goal_norm": 1.0,
            "snqi": success,
        },
    }


def _write_native_seed_rows(root: Path, rows: list[dict[str, object]]) -> None:
    """Persist native writer rows to reports/seed_episode_rows.csv."""

    reports = root / "reports"
    reports.mkdir(parents=True)
    fields = sorted({field for row in rows for field in row})
    with (reports / "seed_episode_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: ("" if row.get(field) is None else row[field]) for field in fields}
            )
    (reports / "campaign_summary.json").write_text('{"status": "fixture"}\n', encoding="utf-8")


def test_native_writer_rows_are_consumed_by_builder(tmp_path: Path) -> None:
    """End-to-end: the real seed-row writer output is consumed as native fields.

    This closes the writer->builder contract for issue #4242: rows produced by
    ``robot_sf.benchmark.seed_variance.build_seed_episode_rows`` (the #4301/#4255
    native instrumentation) carry the canonical ``mechanism_*`` columns the #4206
    builder requires, without any hand-authored taxonomy fields in the test.
    """

    from robot_sf.benchmark.seed_variance import build_seed_episode_rows

    trace_records = [
        _native_writer_episode("prediction_planner", seed=1, success=1.0, trace_verified=True),
        _native_writer_episode(
            "hybrid_rule_v3_fast_progress_static_escape",
            seed=1,
            success=0.4,
            trace_verified=True,
        ),
        _native_writer_episode("ppo", seed=1, success=0.7, trace_verified=True),
    ]
    native_rows = build_seed_episode_rows(trace_records)
    # The writer emitted the canonical consumption columns without a sidecar.
    assert all(
        row["mechanism_schema_version"] == "failure_mechanism_taxonomy.v1" for row in native_rows
    )
    assert all(row["mechanism_evidence_mode"] == "paired_trace" for row in native_rows)

    confirm = tmp_path / "confirm"
    extended = tmp_path / "extended"
    _write_native_seed_rows(confirm, native_rows)
    _write_native_seed_rows(extended, native_rows)
    output = tmp_path / "evidence"

    summary = _MODULE.build_packet(
        config_path=CONFIG,
        confirm_root=confirm,
        extended_root=extended,
        job13175_packet=tmp_path / "packet.json",
        output_dir=output,
        generated_at="2026-07-03T00:00:00Z",
    )

    assert summary["status"] == "analysis_ready_trace_verified"
    assert summary["accepted_rows"] >= 3


def test_native_writer_geometry_only_rows_are_rejected(tmp_path: Path) -> None:
    """Non-trace-verified native rows fail closed to unknown, blocking F-C4(ii)."""

    from robot_sf.benchmark.seed_variance import build_seed_episode_rows

    native_rows = build_seed_episode_rows(
        [_native_writer_episode("prediction_planner", seed=1, success=1.0, trace_verified=False)]
    )
    # The writer refuses geometry-only mechanism labels: label collapses to unknown.
    assert native_rows[0]["mechanism_label"] == "unknown"
    assert native_rows[0]["mechanism_evidence_mode"] == "unknown"

    confirm = tmp_path / "confirm"
    extended = tmp_path / "extended"
    _write_native_seed_rows(confirm, native_rows)
    _write_native_seed_rows(extended, native_rows)
    output = tmp_path / "evidence"

    summary = _MODULE.build_packet(
        config_path=CONFIG,
        confirm_root=confirm,
        extended_root=extended,
        job13175_packet=tmp_path / "packet.json",
        output_dir=output,
        generated_at="2026-07-03T00:00:00Z",
    )

    assert summary["status"] == "blocked_missing_trace_verified_mechanism_labels"


def test_config_declares_forbidden_geometry_substitution() -> None:
    """The checked-in config owns classes and forbids geometry fallback."""
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert payload["issue"] == 4206
    assert payload["taxonomy_source"] == "docs/context/issue_2220_failure_mechanism_taxonomy.md"
    assert (
        payload["forbidden_fallback"]["geometry_buckets_may_substitute_mechanism_labels"] is False
    )


def test_config_declares_retained_mechanism_sidecar() -> None:
    """The checked-in config wires the retained #4242 sidecar as a declared input (#4305 path)."""
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    declared = payload["declared_mechanism_sidecars"]
    assert declared == [
        "docs/context/evidence/issue_3810_h600_interpretation_2026-07/"
        "h600_mechanism_labels_sidecar.csv"
    ]
    # The retained sidecar it points at must actually exist in the repo.
    assert (REPO_ROOT / declared[0]).exists()


def _unlabeled_seed_rows(root: Path, planners: list[str]) -> list[dict[str, object]]:
    """Write seed_episode_rows.csv rows that carry no mechanism fields (predate trace capture)."""

    rows = []
    for planner in planners:
        row = _base_row(planner, 1.0)
        for field in _MODULE.REQUIRED_MECHANISM_FIELDS:
            row.pop(field, None)
        rows.append(row)
    _write_rows(root, rows)
    return rows


def _write_sidecar_csv(
    path: Path, source_rows: list[dict[str, object]], mechanism: dict[str, object]
) -> None:
    """Write an external mechanism-label sidecar keyed to ``source_rows``."""

    key_fields = ("episode_id", "scenario_id", "planner_key", "seed", "repeat_index")
    rows = [{**{k: row[k] for k in key_fields}, **mechanism} for row in source_rows]
    fields = sorted({field for row in rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _config_with_declared_sidecar(tmp_path: Path, sidecar: Path) -> Path:
    """Return a config path identical to the checked-in one but pointing at ``sidecar``."""

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["declared_mechanism_sidecars"] = [str(sidecar)]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_is_not_derivable_row_matches_only_missing_trace() -> None:
    """The classifier flags predates-trace unknowns, not geometry-only or real labels."""

    missing_trace = {
        "mechanism_schema_version": "failure_mechanism_taxonomy.v1",
        "mechanism_label": "unknown",
        "mechanism_confidence": "unknown",
        "mechanism_backfill_status": "not_derivable_missing_trace",
    }
    geometry_only = {**missing_trace, "mechanism_backfill_status": ""}
    geometry_only["mechanism_caveat"] = "not_derivable_non_trace_verified_mechanism"
    real = _base_row("prediction_planner", 1.0)

    assert _MODULE._is_not_derivable_row(missing_trace) is True
    assert _MODULE._is_not_derivable_row(geometry_only) is False
    assert _MODULE._is_not_derivable_row(real) is False


def test_declared_sidecar_all_not_derivable_reports_predates_trace_capture_block(
    tmp_path: Path,
) -> None:
    """Declared sidecar consumed, all labels not_derivable -> precise predates-trace block.

    This is the retained h600 reality: the confirm/extended rows predate trace capture (#4301), so
    every declared-sidecar label is `not_derivable_missing_trace`. The builder must consume the
    sidecar (proving the #4305 path works) yet still fail closed with the precise blocker and a
    trace-instrumented-rerun follow-up, not the generic "add a sidecar" action.
    """

    planners = ["prediction_planner", "ppo", "orca"]
    confirm = tmp_path / "confirm"
    extended = tmp_path / "extended"
    confirm_rows = _unlabeled_seed_rows(confirm, planners)
    _unlabeled_seed_rows(extended, planners)
    # Mirror the retained #4242 backfill sidecar: v1 schema, all-unknown label, missing-trace marker.
    unknown_mechanism = {
        "mechanism_schema_version": "failure_mechanism_taxonomy.v1",
        "mechanism_label": "unknown",
        "mechanism_confidence": "unknown",
        "mechanism_evidence_mode": "unknown",
        "mechanism_evidence_uri": "",
        "mechanism_case_id": "",
        "mechanism_backfill_status": "not_derivable_missing_trace",
        "mechanism_caveat": "not_derivable_missing_trace",
    }
    sidecar = tmp_path / "sidecar.csv"
    _write_sidecar_csv(sidecar, confirm_rows, unknown_mechanism)
    config_path = _config_with_declared_sidecar(tmp_path, sidecar)
    output = tmp_path / "evidence"

    summary = _MODULE.build_packet(
        config_path=config_path,
        confirm_root=confirm,
        extended_root=extended,
        job13175_packet=tmp_path / "packet.json",
        output_dir=output,
        generated_at="2026-07-03T00:00:00Z",
    )

    assert summary["status"] == "blocked_trace_labels_not_derivable_predates_trace_capture"
    missing = json.loads((output / "missing_instrumentation.json").read_text(encoding="utf-8"))
    assert missing["blocker_kinds"] == ["mechanism_labels_not_derivable_predates_trace_capture"]
    assert "trace" in missing["next_action"].lower()
    assert "sidecar" not in missing["followup_issue_skeleton"]["title"].lower()
    # Provenance proves the declared sidecar was actually consumed.
    declared = missing["input_provenance"]["declared_mechanism_sidecars"]
    assert declared and declared[0]["exists"] is True
    # No F-C4(ii) tables are emitted while blocked.
    assert _read_csv(output / "f_c4ii_probe_predictive_dominance.csv") == []


def test_declared_sidecar_supplies_trace_labels_unblocks_tables(tmp_path: Path) -> None:
    """A declared sidecar with real trace labels unblocks the F-C4(ii) tables (#4305 path)."""

    planners = [
        "prediction_planner",
        "hybrid_rule_v3_fast_progress_static_escape",
        "ppo",
        "orca",
    ]
    confirm = tmp_path / "confirm"
    extended = tmp_path / "extended"
    confirm_rows = _unlabeled_seed_rows(confirm, planners)
    _unlabeled_seed_rows(extended, planners)
    trace_mechanism = {
        "mechanism_schema_version": "failure_mechanism_taxonomy.v1",
        "mechanism_label": "static_deadlock_or_local_minimum",
        "mechanism_confidence": "observed_mechanism",
        "mechanism_evidence_mode": "paired_trace",
        "mechanism_evidence_uri": "docs/context/issue_2220_failure_mechanism_taxonomy.md",
        "mechanism_case_id": "fixture-trace-case",
        "mechanism_caveat": "",
    }
    sidecar = tmp_path / "sidecar.csv"
    _write_sidecar_csv(sidecar, confirm_rows, trace_mechanism)
    config_path = _config_with_declared_sidecar(tmp_path, sidecar)
    output = tmp_path / "evidence"

    summary = _MODULE.build_packet(
        config_path=config_path,
        confirm_root=confirm,
        extended_root=extended,
        job13175_packet=tmp_path / "packet.json",
        output_dir=output,
        generated_at="2026-07-03T00:00:00Z",
    )

    assert summary["status"] == "analysis_ready_trace_verified"
    assert summary["accepted_rows"] >= len(planners)
    probe = _read_csv(output / "f_c4ii_probe_predictive_dominance.csv")
    assert probe != []
