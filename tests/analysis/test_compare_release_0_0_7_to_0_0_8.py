"""Small release-row fixtures for the 0.0.7/0.0.8 slot comparator."""

from __future__ import annotations

import copy
import csv
import gzip
import json
import subprocess
import sys
import tarfile
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pytest

import scripts.analysis.compare_release_0_0_7_to_0_0_8 as comparator
from scripts.analysis.compare_release_0_0_7_to_0_0_8 import (
    BASELINE_CAMPAIGN,
    V4_SLOT_REPLACEMENTS,
    compare,
    write_report,
)

OLD_SOURCE = "07f7e8d43084de748915e1b1eb8b2a1603357c6e"
SCENARIO = {
    "manifest": "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml",
    "sha256": "03fc83302f707dd1b27c0fa81c4e45e36e8354a4413171d09365926f62bb5c2c",
}
CONFIG_RUNTIME_HASH = "a" * 16
SCENARIO_RUNTIME_HASH = "b" * 16


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
    _, _, _, source_commit = _successor_contract(tmp_path)
    root = tmp_path / "successor"
    path = root / "runs" / run / "episodes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    updated = copy.deepcopy(rows)
    for row in updated:
        row["git_hash"] = source_commit
    path.write_text("".join(json.dumps(row) + "\n" for row in updated), encoding="utf-8")
    (root / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "campaign_id": "synthetic-0.0.8",
                "git": {"commit": source_commit},
                "config_hash": CONFIG_RUNTIME_HASH,
                "scenario_matrix": "configs/scenarios/synthetic.yaml",
                "scenario_matrix_hash": SCENARIO_RUNTIME_HASH,
            }
        ),
        encoding="utf-8",
    )
    return root


def _successor_contract(tmp_path: Path) -> tuple[Path, str, Path, str]:
    source = tmp_path / "successor-source"
    manifest_path = tmp_path / "successor-manifest.json"
    if manifest_path.exists():
        payload = manifest_path.read_bytes()
        return (
            manifest_path,
            sha256(payload).hexdigest(),
            source,
            json.loads(payload)["source_commit"],
        )
    scenario = source / "configs/scenarios/synthetic.yaml"
    config = source / "configs/benchmarks/synthetic.yaml"
    scenario.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    scenario.write_text("scenarios: []\n")
    planner_bindings = {}
    planner_lines = ["  - key: goal"]
    for key in V4_SLOT_REPLACEMENTS.values():
        name = f"configs/algos/{key}.yaml"
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"name: {key}\n")
        planner_lines.extend([f"  - key: {key}", f"    algo_config: {name}"])
        planner_bindings[key] = {"path": name, "sha256": sha256(path.read_bytes()).hexdigest()}
    config.write_text(
        "scenario_matrix: configs/scenarios/synthetic.yaml\nplanners:\n"
        + "\n".join(planner_lines)
        + "\n"
    )
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "configs"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "synthetic successor identity",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "schema_version": "slot-paired-successor.v1",
        "release": "0.0.8",
        "campaign_id": "synthetic-0.0.8",
        "source_commit": commit,
        "campaign_config": {
            "path": "configs/benchmarks/synthetic.yaml",
            "sha256": sha256(config.read_bytes()).hexdigest(),
            "runtime_hash": CONFIG_RUNTIME_HASH,
        },
        "scenario_matrix": {
            "path": "configs/scenarios/synthetic.yaml",
            "sha256": sha256(scenario.read_bytes()).hexdigest(),
            "runtime_hash": SCENARIO_RUNTIME_HASH,
        },
        "versioned_planner_bindings": planner_bindings,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, sha256(manifest_path.read_bytes()).hexdigest(), source, commit


def _successor_kwargs(tmp_path: Path) -> dict:
    path, digest, source, _ = _successor_contract(tmp_path)
    return {
        "successor_manifest": path,
        "successor_manifest_sha256": digest,
        "successor_source_root": source,
    }


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
        **_successor_kwargs(root.parent),
    )


def _rules(path: Path, entries: list[dict]) -> Path:
    path.write_text(
        json.dumps({"schema_version": "slot-paired-classifications.v2", "rules": entries})
    )
    return path


def _rule(
    field: str,
    *,
    scenario: str = "s1",
    classification: str = "route changes",
    predicate: dict | None = None,
    seed: int = 111,
) -> dict:
    return {
        "rule_id": f"{scenario}-{field}",
        "slots": [
            {
                "planner": "goal",
                "kinematics": "differential_drive",
                "scenario_id": scenario,
                "benchmark_track": "",
                "seeds": [seed],
            }
        ],
        "fields": [field],
        "predicate": predicate or {"sign": "positive", "max_abs_delta": 2.0},
        "max_findings": 1,
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
    new_row["metrics"]["path_length"] = 4.5
    new_row["outcome"]["route_complete"] = False
    added = _row("s3", 111)
    root = _root(tmp_path, [new_row, added])
    preliminary = _compare(bundle, root, digest)
    finding_ids = {
        (item["scenario_id"], item["field"]): item["finding_id"] for item in preliminary["findings"]
    }
    rules = _rules(
        tmp_path / "rules.json",
        [
            _rule("metrics.path_length"),
            _rule(
                "outcome.route_complete",
                predicate={"finding_ids": [finding_ids["s1", "outcome.route_complete"]]},
            ),
            _rule(
                "__row__", scenario="s2", predicate={"finding_ids": [finding_ids["s2", "__row__"]]}
            ),
            _rule(
                "__row__", scenario="s3", predicate={"finding_ids": [finding_ids["s3", "__row__"]]}
            ),
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
    first, second = _rule("metrics.path_length"), _rule("metrics.path_length")
    second["rule_id"] = "second"
    rules = _rules(tmp_path / "rules.json", [first, second])
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
            **_successor_kwargs(tmp_path),
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
        lambda bundle_arg, root_arg, classification_file=None, baseline_root=None, **kwargs: (
            real_compare(
                bundle_arg,
                root_arg,
                baseline_root=baseline_root,
                classification_file=classification_file,
                baseline_sha256=digest,
                baseline_source=OLD_SOURCE,
                expected_scenario_identity=SCENARIO,
                **kwargs,
            )
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
            "--successor-manifest",
            str(_successor_kwargs(tmp_path)["successor_manifest"]),
            "--successor-manifest-sha256",
            _successor_kwargs(tmp_path)["successor_manifest_sha256"],
            "--successor-source-root",
            str(_successor_kwargs(tmp_path)["successor_source_root"]),
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
        **_successor_kwargs(tmp_path),
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
            **_successor_kwargs(tmp_path),
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


def test_rule_explains_one_bounded_change_but_not_second_injected_change(
    tmp_path: Path, monkeypatch
) -> None:
    rows = [_row("s1", 111), _row("s1", 112)]
    bundle, digest = _archive(tmp_path, rows)
    changed = copy.deepcopy(rows)
    changed[0]["metrics"]["path_length"] += 1.0
    changed[1]["metrics"]["path_length"] += 100.0
    root = _root(tmp_path, changed)
    rule = _rule("metrics.path_length")
    rule["slots"][0]["seeds"] = "all"
    rules = _rules(tmp_path / "rules.json", [rule])
    report = _compare(bundle, root, digest, rules)
    assert report["unexplained_count"] == 1
    assert report["rules"][0]["covered_findings"] == [report["findings"][0]["finding_id"]]
    assert report["findings"][1]["classification"] == "unexplained"

    real_compare = comparator.compare
    monkeypatch.setattr(
        comparator,
        "compare",
        lambda bundle_arg, root_arg, **kwargs: real_compare(
            bundle_arg,
            root_arg,
            baseline_sha256=digest,
            baseline_source=OLD_SOURCE,
            expected_scenario_identity=SCENARIO,
            **kwargs,
        ),
    )
    successor = _successor_kwargs(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare",
            "--baseline-bundle",
            str(bundle),
            "--successor-root",
            str(root),
            "--successor-manifest",
            str(successor["successor_manifest"]),
            "--successor-manifest-sha256",
            successor["successor_manifest_sha256"],
            "--successor-source-root",
            str(successor["successor_source_root"]),
            "--classification-file",
            str(rules),
            "--output-dir",
            str(tmp_path / "bounded-report"),
        ],
    )
    assert comparator.main() == 1
    assert (
        json.loads((tmp_path / "bounded-report/report.json").read_text())["unexplained_count"] == 1
    )


def test_bogus_successor_config_hash_exits_two(tmp_path: Path, monkeypatch) -> None:
    row = _row("s1", 111)
    bundle, digest = _archive(tmp_path, [row])
    root = _root(tmp_path, [row])
    campaign_path = root / "campaign_manifest.json"
    campaign = json.loads(campaign_path.read_text())
    campaign["config_hash"] = "bogus"
    campaign_path.write_text(json.dumps(campaign))
    successor = _successor_kwargs(tmp_path)
    with pytest.raises(ValueError, match="config_hash differs"):
        _compare(bundle, root, digest)
    real_compare = comparator.compare
    monkeypatch.setattr(
        comparator,
        "compare",
        lambda bundle_arg, root_arg, **kwargs: real_compare(
            bundle_arg,
            root_arg,
            baseline_sha256=digest,
            baseline_source=OLD_SOURCE,
            expected_scenario_identity=SCENARIO,
            **kwargs,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare",
            "--baseline-bundle",
            str(bundle),
            "--successor-root",
            str(root),
            "--successor-manifest",
            str(successor["successor_manifest"]),
            "--successor-manifest-sha256",
            successor["successor_manifest_sha256"],
            "--successor-source-root",
            str(successor["successor_source_root"]),
            "--output-dir",
            str(tmp_path / "invalid-report"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        comparator.main()
    assert exc.value.code == 2
    assert not (tmp_path / "invalid-report/report.json").exists()


def test_rule_over_max_findings_explains_none(tmp_path: Path) -> None:
    rows = [_row("s1", 111), _row("s1", 112)]
    bundle, digest = _archive(tmp_path, rows)
    changed = copy.deepcopy(rows)
    for row in changed:
        row["metrics"]["path_length"] += 1.0
    root = _root(tmp_path, changed)
    rule = _rule("metrics.path_length")
    rule["slots"][0]["seeds"] = "all"
    rules = _rules(tmp_path / "rules.json", [rule])

    report = _compare(bundle, root, digest, rules)

    assert report["unexplained_count"] == 2
    assert report["rules"][0]["over_limit"] is True
    assert len(report["rules"][0]["matching_findings"]) == 2
    assert report["rules"][0]["covered_findings"] == []


def test_successor_planner_binding_hash_must_match_source(tmp_path: Path) -> None:
    row = _row("s1", 111)
    bundle, digest = _archive(tmp_path, [row])
    root = _root(tmp_path, [row])
    manifest_path = tmp_path / "successor-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    key = next(iter(manifest["versioned_planner_bindings"]))
    manifest["versioned_planner_bindings"][key]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="planner binding SHA-256 mismatch"):
        _compare(bundle, root, digest)
