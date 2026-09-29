"""Small release-row fixtures for the 0.0.7/0.0.8 slot comparator."""

from __future__ import annotations

import copy
import csv
import gzip
import json
import sys
import tarfile
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pytest

import scripts.analysis.compare_release_0_0_7_to_0_0_8 as comparator
from scripts.analysis.compare_release_0_0_7_to_0_0_8 import (
    BASELINE_CAMPAIGN,
    compare,
    write_report,
)

OLD_SOURCE = "07f7e8d43084de748915e1b1eb8b2a1603357c6e"
NEW_SOURCE = "0" * 40
SCENARIO = {
    "manifest": "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml",
    "sha256": "03fc83302f707dd1b27c0fa81c4e45e36e8354a4413171d09365926f62bb5c2c",
}


def _row(scenario: str, seed: int, *, planner: str = "goal", track: str | None = None) -> dict:
    row = {
        "version": "v1",
        "episode_id": f"{scenario}-{seed}",
        "scenario_id": scenario,
        "seed": seed,
        "algo": planner,
        "git_hash": OLD_SOURCE,
        "outcome": {"route_complete": True, "collision_event": False, "timeout_event": False},
        "metrics": {"collisions": 0, "path_length": 3.0, "nested": {"near_miss": 0.0}},
    }
    if track is not None:
        row["benchmark_track"] = track
    return row


def _archive(
    tmp_path: Path, rows: list[dict], *, run: str = "goal__differential_drive"
) -> tuple[Path, str]:
    bundle = tmp_path / "baseline.tar.gz"
    manifest = {
        "source_sha": OLD_SOURCE,
        "scenario": {"matrix_path": SCENARIO["manifest"], "matrix_sha256": SCENARIO["sha256"]},
    }
    members = {
        "publication/payload/release/release_manifest.resolved.json": json.dumps(manifest).encode(),
        f"publication/payload/runs/{run}/episodes.jsonl": b"".join(
            (json.dumps(row) + "\n").encode() for row in rows
        ),
    }
    with tarfile.open(bundle, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
    return bundle, sha256(bundle.read_bytes()).hexdigest()


def _root(tmp_path: Path, rows: list[dict], *, run: str = "goal__differential_drive") -> Path:
    root = tmp_path / "successor"
    path = root / "runs" / run / "episodes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    updated = copy.deepcopy(rows)
    for row in updated:
        row["git_hash"] = NEW_SOURCE
    path.write_text("".join(json.dumps(row) + "\n" for row in updated), encoding="utf-8")
    (root / "campaign_manifest.json").write_text(
        json.dumps({"campaign_id": "synthetic-0.0.8", "git": {"commit": NEW_SOURCE}}),
        encoding="utf-8",
    )
    return root


def _preserved_root(tmp_path: Path, rows: list[dict]) -> tuple[Path, str]:
    root = tmp_path / "preserved-0.0.7"
    release = {
        "source_sha": OLD_SOURCE,
        "scenario": {"matrix_path": SCENARIO["manifest"], "matrix_sha256": SCENARIO["sha256"]},
    }
    files = {
        "campaign_manifest.json": json.dumps(
            {"campaign_id": BASELINE_CAMPAIGN, "git": {"commit": OLD_SOURCE}}
        ).encode(),
        "release/release_manifest.resolved.json": json.dumps(release).encode(),
        "runs/goal__differential_drive/episodes.jsonl": b"".join(
            (json.dumps(row) + "\n").encode() for row in rows
        ),
    }
    entries = []
    for name, payload in files.items():
        stored = gzip.compress(payload)
        path = root / f"{name}.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(stored)
        entries.append(
            {
                "path": name,
                "stored_path": f"{name}.gz",
                "sha256": sha256(payload).hexdigest(),
                "stored_sha256": sha256(stored).hexdigest(),
            }
        )
    manifest = {
        "schema": "campaign-preservation-manifest.v1",
        "campaign_id": BASELINE_CAMPAIGN,
        "files": entries,
    }
    digest = sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["manifest_digest"] = f"sha256:{digest}"
    (root / "campaign_preservation_manifest.json").write_text(json.dumps(manifest))
    return root, digest


def _compare(bundle: Path, root: Path, digest: str, rules: Path | None = None) -> dict:
    return compare(
        bundle,
        root,
        classification_file=rules,
        baseline_sha256=digest,
        baseline_source=OLD_SOURCE,
        expected_scenario_identity=SCENARIO,
    )


def _rules(path: Path, entries: list[dict]) -> Path:
    path.write_text(
        json.dumps({"schema_version": "slot-paired-classifications.v1", "rules": entries})
    )
    return path


def _rule(field: str, *, scenario: str = "s1", classification: str = "route changes") -> dict:
    return {
        "planner": "goal",
        "scenario_id": scenario,
        "field": field,
        "classification": classification,
        "issue": "#9870",
        "explanation": "Versioned route input changed this slot.",
        "evidence": "docs/analysis/route-comparison.json#s1",
    }


def test_synthetic_successor_built_from_baseline_rows_reports_paired_and_release_only(
    tmp_path: Path,
) -> None:
    baseline_rows = [_row("s1", 111), _row("s2", 111)]
    bundle, digest = _archive(tmp_path, baseline_rows)
    new_row = copy.deepcopy(baseline_rows[0])
    new_row["git_hash"] = NEW_SOURCE
    new_row["metrics"]["path_length"] = 4.5
    new_row["outcome"]["route_complete"] = False
    added = _row("s3", 111)
    added["git_hash"] = NEW_SOURCE
    root = _root(tmp_path, [new_row, added])
    rules = _rules(
        tmp_path / "rules.json",
        [
            _rule("metrics.path_length"),
            _rule("outcome.route_complete"),
            _rule("__row__", scenario="s2"),
            _rule("__row__", scenario="s3"),
        ],
    )

    report = _compare(bundle, root, digest, rules)
    assert (report["paired_rows"], report["only_0_0_7"], report["only_0_0_8"]) == (1, 1, 1)
    assert report["status"] == "classified"
    assert report["unexplained_count"] == 0
    assert {(row["scenario_id"], row["field"], row["presence"]) for row in report["findings"]} == {
        ("s1", "metrics.path_length", "paired"),
        ("s1", "outcome.route_complete", "paired"),
        ("s2", "__row__", "only_0_0_7"),
        ("s3", "__row__", "only_0_0_8"),
    }
    assert (
        next(row for row in report["findings"] if row["field"] == "metrics.path_length")[
            "delta_0_0_8_minus_0_0_7"
        ]
        == 1.5
    )
    assert all(
        row["classification"] and row["explanation"] and row["evidence"]
        for row in report["findings"]
    )
    output = tmp_path / "report"
    write_report(report, output)
    with (output / "findings.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 4
    summary = next(
        row
        for row in report["planner_scenario_metrics"]
        if row["scenario_id"] == "s1" and row["field"] == "metrics.path_length"
    )
    assert summary["mean_paired_delta"] == 1.5


def test_unexplained_difference_and_metric_presence_are_visible(tmp_path: Path) -> None:
    row = _row("s1", 111)
    bundle, digest = _archive(tmp_path, [row])
    changed = copy.deepcopy(row)
    changed["metrics"]["new_metric"] = 0.25
    root = _root(tmp_path, [changed])

    report = _compare(bundle, root, digest)

    assert report["status"] == "unexplained"
    assert report["unexplained_count"] == 1
    assert report["findings"][0]["field"] == "metrics.new_metric"
    assert report["findings"][0]["old_value"] == "<missing>"


def test_numeric_tolerance_and_track_are_part_of_slot(tmp_path: Path) -> None:
    row = _row("s1", 111, track="oracle_full_state")
    bundle, digest = _archive(tmp_path, [row])
    changed = copy.deepcopy(row)
    changed["metrics"]["path_length"] += 5e-13
    root = _root(tmp_path, [changed])
    assert _compare(bundle, root, digest)["findings"] == []

    changed["benchmark_track"] = "lidar_2d"
    root = _root(tmp_path, [changed])
    report = _compare(bundle, root, digest)
    assert report["paired_rows"] == 0
    assert (report["only_0_0_7"], report["only_0_0_8"]) == (1, 1)


def test_versioned_v4_arm_is_replacement_not_paired_correction(tmp_path: Path) -> None:
    old = _row("s1", 111, planner="hybrid_rule_v3_fast_progress_static_escape")
    bundle, digest = _archive(
        tmp_path, [old], run="hybrid_rule_v3_fast_progress_static_escape__differential_drive"
    )
    new = _row("s1", 111, planner="hybrid_rule_v4_fast_progress_static_escape")
    root = _root(
        tmp_path, [new], run="hybrid_rule_v4_fast_progress_static_escape__differential_drive"
    )

    report = _compare(bundle, root, digest)

    assert report["paired_rows"] == 0
    assert {finding["presence"] for finding in report["findings"]} == {"only_0_0_7", "only_0_0_8"}
    assert {finding["replacement_planner"] for finding in report["findings"]} == {
        "hybrid_rule_v3_fast_progress_static_escape",
        "hybrid_rule_v4_fast_progress_static_escape",
    }


def test_duplicate_slot_and_ambiguous_classification_fail_closed(tmp_path: Path) -> None:
    row = _row("s1", 111)
    bundle, digest = _archive(tmp_path, [row, row])
    root = _root(tmp_path, [row])
    with pytest.raises(ValueError, match="duplicate slot"):
        _compare(bundle, root, digest)

    bundle, digest = _archive(tmp_path, [row])
    changed = copy.deepcopy(row)
    changed["metrics"]["path_length"] = 5.0
    root = _root(tmp_path, [changed])
    rules = _rules(
        tmp_path / "rules.json", [_rule("metrics.path_length"), _rule("metrics.path_length")]
    )
    with pytest.raises(ValueError, match="ambiguous classification"):
        _compare(bundle, root, digest, rules)


def test_baseline_checksum_and_source_are_pinned(tmp_path: Path) -> None:
    row = _row("s1", 111)
    bundle, digest = _archive(tmp_path, [row])
    root = _root(tmp_path, [row])
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _compare(bundle, root, "f" * 64)
    with pytest.raises(ValueError, match="source mismatch"):
        compare(
            bundle,
            root,
            baseline_sha256=digest,
            baseline_source="f" * 40,
            expected_scenario_identity=SCENARIO,
        )


def test_cli_writes_findings_and_exits_nonzero_for_unexplained(tmp_path: Path, monkeypatch) -> None:
    row = _row("s1", 111)
    bundle, digest = _archive(tmp_path, [row])
    changed = copy.deepcopy(row)
    changed["metrics"]["path_length"] = 4.0
    root = _root(tmp_path, [changed])
    real_compare = comparator.compare
    monkeypatch.setattr(
        comparator,
        "compare",
        lambda bundle_arg, root_arg, classification_file=None, baseline_root=None: real_compare(
            bundle_arg,
            root_arg,
            baseline_root=baseline_root,
            classification_file=classification_file,
            baseline_sha256=digest,
            baseline_source=OLD_SOURCE,
            expected_scenario_identity=SCENARIO,
        ),
    )
    output = tmp_path / "cli-report"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare",
            "--baseline-bundle",
            str(bundle),
            "--successor-root",
            str(root),
            "--output-dir",
            str(output),
        ],
    )

    assert comparator.main() == 1
    assert json.loads((output / "report.json").read_text())["unexplained_count"] == 1


def test_cold_restored_published_artifact_shape_verifies_checksums(tmp_path: Path) -> None:
    row = _row("s1", 111)
    baseline_root, digest = _preserved_root(tmp_path, [row])
    successor = _root(tmp_path, [row])
    report = compare(
        None,
        successor,
        baseline_root=baseline_root,
        baseline_manifest_digest=digest,
        expected_scenario_identity=SCENARIO,
    )
    assert report["paired_rows"] == 1
    assert report["baseline"]["preservation_manifest_digest"] == digest

    path = baseline_root / "runs/goal__differential_drive/episodes.jsonl.gz"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="stored SHA-256 mismatch"):
        compare(
            None,
            successor,
            baseline_root=baseline_root,
            baseline_manifest_digest=digest,
            expected_scenario_identity=SCENARIO,
        )


def test_published_0_0_7_row_sample_with_nonfinite_metric_sentinels(tmp_path: Path) -> None:
    sample = Path(__file__).parent / "fixtures/issue_9668_0_0_7_goal_sample.jsonl"
    baseline_rows = [json.loads(line) for line in sample.read_text().splitlines()]
    bundle, digest = _archive(tmp_path, baseline_rows)
    successor_rows = copy.deepcopy(baseline_rows)
    successor_rows[0]["metrics"]["socnavbench_path_length"] += 1.0
    root = _root(tmp_path, successor_rows)
    rule = _rule("metrics.socnavbench_path_length", scenario="classic_bottleneck_low")
    rules = _rules(tmp_path / "rules.json", [rule])

    report = _compare(bundle, root, digest, rules)

    assert report["paired_rows"] == 2
    assert report["unexplained_count"] == 0
    assert len(report["findings"]) == 1
    assert report["findings"][0]["delta_0_0_8_minus_0_0_7"] == 1.0
    nonfinite = next(
        item
        for item in report["planner_scenario_metrics"]
        if item["field"] == "metrics.metric_values.min_predicted_separation_m"
    )
    assert nonfinite["paired_count"] == 2
    assert nonfinite["mean_paired_delta"] is None
    write_report(report, tmp_path / "published-sample-report")
    json.loads((tmp_path / "published-sample-report/report.json").read_text())
