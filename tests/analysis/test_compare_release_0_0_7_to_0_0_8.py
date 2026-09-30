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
from robot_sf.benchmark.utils import _config_hash
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


def _row(scenario: str, seed: int, *, planner: str = "goal", track: str | None = None) -> dict:
    algo = "hybrid_rule_local_planner" if planner in V4_SLOT_REPLACEMENTS.values() else planner
    row = {
        "version": "v1",
        "episode_id": f"{scenario}-{seed}",
        "scenario_id": scenario,
        "seed": seed,
        "algo": algo,
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


def _root(
    tmp_path: Path,
    rows: list[dict],
    *,
    run: str = "goal__differential_drive",
    enable_all_v4: bool = False,
) -> Path:
    manifest_path, _, source, source_commit = _successor_contract(
        tmp_path, rows=rows, run=run, enable_all_v4=enable_all_v4
    )
    manifest = json.loads(manifest_path.read_text())
    _, _, _, scoped_hashes, _ = comparator._runtime_successor_identity(
        source, source_commit, "configs/benchmarks/synthetic.yaml", {}
    )
    root = tmp_path / "successor"
    path = root / "runs" / run / "episodes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    updated = copy.deepcopy(rows)
    run_planner = run.rsplit("__", 1)[0]
    for row in updated:
        row["git_hash"] = source_commit
        algo = row["algo"]
        binding = manifest["versioned_planner_bindings"].get(run_planner)
        config = (
            {"speed": 1, "planner_variant": "hybrid_rule_v4_clearance_braking", "name": run_planner}
            if binding
            else {}
        )
        scenario = {
            "name": row["scenario_id"],
            "id": row["scenario_id"],
            "map_file": "maps/svg_maps/classic_crossing.svg",
            "algo": algo,
            "seed": row["seed"],
        }
        if row.get("benchmark_track"):
            scenario["benchmark_track"] = row["benchmark_track"]
        row["scenario_params"] = scenario
        row["config_hash"] = _config_hash(scenario)
        row["algorithm_metadata"] = {
            "algorithm": algo,
            "config": config,
            "config_hash": _config_hash(config),
        }
        row["provenance"] = {
            "commit_hash": source_commit,
            "config_identity": {
                "algo": algo,
                "algo_config_path": binding["path"] if binding else None,
                "scenario_matrix_hash": scoped_hashes[(run_planner, run.rsplit("__", 1)[1])],
                "campaign_config_hash": manifest["campaign_config"]["runtime_hash"],
            },
        }
    _, _, runtime_rows, _, _ = comparator._runtime_successor_identity(
        source,
        source_commit,
        "configs/benchmarks/synthetic.yaml",
        {
            (
                run_planner,
                run.rsplit("__", 1)[1],
                row["scenario_id"],
                row["seed"],
                row.get("benchmark_track") or "",
            ): {"_provenance": {"scenario_params": row["scenario_params"]}}
            for row in updated
        },
    )
    for row in updated:
        slot = (
            run_planner,
            run.rsplit("__", 1)[1],
            row["scenario_id"],
            row["seed"],
            row.get("benchmark_track") or "",
        )
        if slot in runtime_rows:
            row["scenario_params"].update(runtime_rows[slot]["controls"])
            row["scenario_params"]["robot_config"] = {
                "type": slot[1],
                **({"command_mode": "vx_vy"} if slot[1] == "holonomic" else {}),
            }
        row["config_hash"] = _config_hash(row["scenario_params"])
    path.write_text("".join(json.dumps(row) + "\n" for row in updated), encoding="utf-8")
    (root / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "campaign_id": "synthetic-0.0.8",
                "git": {"commit": source_commit},
                "config_hash": manifest["campaign_config"]["runtime_hash"],
                "scenario_matrix": "configs/scenarios/synthetic.yaml",
                "scenario_matrix_hash": manifest["scenario_matrix"]["runtime_hash"],
            }
        ),
        encoding="utf-8",
    )
    return root


def _successor_contract(
    tmp_path: Path,
    *,
    rows: list[dict] | None = None,
    run: str = "goal__differential_drive",
    enable_all_v4: bool = False,
) -> tuple[Path, str, Path, str]:
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
    # The pinned source must contain actual production runtime modules. A
    # sparse local clone keeps each synthetic campaign fixture cheap.
    subprocess.run(
        [
            "git",
            "clone",
            "--shared",
            "--quiet",
            "--sparse",
            str(Path(__file__).parents[2]),
            str(source),
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "sparse-checkout",
            "add",
            "robot_sf",
            "fast-pysf",
            "configs/benchmarks",
            "configs/scenarios",
            "configs/algos",
            "configs/policy_search",
            "maps/svg_maps",
        ],
        check=True,
    )
    scenario = source / "configs/scenarios/synthetic.yaml"
    config = source / "configs/benchmarks/synthetic.yaml"
    scenario.parent.mkdir(parents=True, exist_ok=True)
    config.parent.mkdir(parents=True, exist_ok=True)
    map_path = source / "maps/svg_maps/classic_crossing.svg"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>\n")
    selected = rows or [_row("s1", 111)]
    by_scenario: dict[str, dict] = {}
    for row in selected:
        item = by_scenario.setdefault(row["scenario_id"], {"seeds": set(), "track": None})
        item["seeds"].add(row["seed"])
        item["track"] = row.get("benchmark_track") or item["track"]
    scenario.write_text(
        "".join(
            f"- name: {name}\n  map_file: ../../maps/svg_maps/classic_crossing.svg\n"
            f"  seeds: {sorted(item['seeds'])}\n"
            + (f"  benchmark_track: {item['track']}\n" if item["track"] else "")
            for name, item in by_scenario.items()
        )
    )
    selected_planner = run.rsplit("__", 1)[0]
    planner_bindings = {}
    planner_lines = [
        "  - key: goal",
        "    algo: goal",
        f"    enabled: {str(selected_planner == 'goal').lower()}",
    ]
    base_planner = source / "configs/algos/hybrid_rule_v4_clearance_braking.yaml"
    base_planner.parent.mkdir(parents=True, exist_ok=True)
    base_planner.write_text("speed: 1\nplanner_variant: hybrid_rule_v4_clearance_braking\n")
    template = (
        source
        / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"
    )
    template_text = template.read_text(encoding="utf-8")
    for key in V4_SLOT_REPLACEMENTS.values():
        name = f"configs/algos/{key}.yaml"
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "release_parameter_freeze:\n"
            "  status: frozen\n"
            "  implementation_family: hybrid_rule_v4_clearance_braking\n"
            f"  replaces_0_0_7_slot: {comparator.V4_PREDECESSORS[key]}\n"
            "base_config_path: configs/algos/hybrid_rule_v4_clearance_braking.yaml\n"
            f"params:\n  name: {key}\n"
        )
        frozen = f"configs/policy_search/candidates/{key}_s30_h600_release_0_0_8_frozen.yaml"
        assert frozen in template_text
        template_text = template_text.replace(frozen, name)
        planner_lines.extend(
            [
                f"  - key: {key}",
                "    algo: hybrid_rule_local_planner",
                f"    algo_config: {name}",
                f"    enabled: {str(enable_all_v4 or selected_planner == key).lower()}",
            ]
        )
        planner_bindings[key] = {"path": name, "sha256": sha256(path.read_bytes()).hexdigest()}
    template.write_text(template_text, encoding="utf-8")
    config.write_text(
        "scenario_matrix: configs/scenarios/synthetic.yaml\nhorizon: 600\ndt: 0.1\n"
        + ("kinematics_matrix: [holonomic]\n" if run.endswith("__holonomic") else "")
        + "planners:\n"
        + "\n".join(planner_lines)
        + "\n"
    )
    subprocess.run(["git", "-C", str(source), "add", "--sparse", "configs", "maps"], check=True)
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
    config_hash, scenario_hash, _, _, _ = comparator._runtime_successor_identity(
        source, commit, "configs/benchmarks/synthetic.yaml", {}
    )
    manifest = {
        "schema_version": "slot-paired-successor.v1",
        "release": "0.0.8",
        "campaign_id": "synthetic-0.0.8",
        "source_commit": commit,
        "campaign_config": {
            "path": "configs/benchmarks/synthetic.yaml",
            "sha256": sha256(config.read_bytes()).hexdigest(),
            "runtime_hash": config_hash,
        },
        "scenario_matrix": {
            "path": "configs/scenarios/synthetic.yaml",
            "sha256": sha256(scenario.read_bytes()).hexdigest(),
            "runtime_hash": scenario_hash,
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


def _assert_cli_exit_two(
    tmp_path: Path, bundle: Path, root: Path, digest: str, monkeypatch
) -> None:
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
    output = tmp_path / "invalid-control-report"
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
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        comparator.main()
    assert exc.value.code == 2
    assert not (output / "report.json").exists()


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


def test_configured_v4_arms_absent_from_goal_only_campaign_exit_two(tmp_path: Path) -> None:
    """A goal-only result cannot claim a complete four-arm v4 successor campaign."""
    row = _row("s1", 111)
    bundle, digest = _archive(tmp_path, [row])
    root = _root(tmp_path, [row], enable_all_v4=True)

    with pytest.raises(ValueError, match="planner arm .* has no rows"):
        _compare(bundle, root, digest)


def test_missing_extra_and_duplicate_successor_slots_are_unclassifiable_findings(
    tmp_path: Path, monkeypatch
) -> None:
    """Completeness errors stay visible and cannot be hidden with analyst rules."""
    rows = [_row("s1", 111), _row("s1", 112)]
    bundle, digest = _archive(tmp_path, rows)
    root = _root(tmp_path, rows)
    path = root / "runs/goal__differential_drive/episodes.jsonl"
    recorded = [json.loads(line) for line in path.read_text().splitlines()]
    extra = copy.deepcopy(recorded[0])
    extra["seed"] = 113
    path.write_text("".join(json.dumps(row) + "\n" for row in [recorded[0], recorded[0], extra]))

    report = _compare(bundle, root, digest)
    structural = [item for item in report["findings"] if item["field"] == "__slot__"]
    assert {(item["seed"], item["presence"]) for item in structural} == {
        (111, "duplicate_0_0_8"),
        (112, "missing_0_0_8"),
        (113, "extra_0_0_8"),
    }
    assert report["unexplained_count"] >= 3
    rules = _rules(
        tmp_path / "rules.json",
        [
            _rule(
                "__slot__",
                predicate={"finding_ids": [item["finding_id"] for item in structural]},
            )
        ],
    )
    assert all(
        item["classification"] == "unexplained"
        for item in _compare(bundle, root, digest, rules)["findings"]
        if item["field"] == "__slot__"
    )

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
    output = tmp_path / "incomplete-report"
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
            str(output),
        ],
    )
    assert comparator.main() == 1
    assert (output / "findings.csv").exists()

    extra["provenance"]["config_identity"]["scenario_matrix_hash"] = "f" * 16
    path.write_text("".join(json.dumps(row) + "\n" for row in [recorded[0], extra]))
    with pytest.raises(ValueError, match="scenario_matrix_hash differs from pinned scoped runner"):
        _compare(bundle, root, digest)

    duplicate = copy.deepcopy(recorded[0])
    duplicate["provenance"]["config_identity"]["scenario_matrix_hash"] = "f" * 16
    path.write_text("".join(json.dumps(row) + "\n" for row in [recorded[0], duplicate]))
    with pytest.raises(ValueError, match="scenario_matrix_hash differs from pinned scoped runner"):
        _compare(bundle, root, digest)


def test_mixed_scoped_matrix_hashes_exit_two_before_report(tmp_path: Path, monkeypatch) -> None:
    """Rows from differently scoped runner inputs cannot share one campaign."""
    rows = [_row("s1", 111), _row("s2", 111)]
    bundle, digest = _archive(tmp_path, rows)
    root = _root(tmp_path, rows)
    path = root / "runs/goal__differential_drive/episodes.jsonl"
    recorded = [json.loads(line) for line in path.read_text().splitlines()]
    recorded[1]["provenance"]["config_identity"]["scenario_matrix_hash"] = "f" * 16
    path.write_text("".join(json.dumps(row) + "\n" for row in recorded))

    with pytest.raises(ValueError, match="scenario_matrix_hash differs from pinned scoped runner"):
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
    successor = _successor_kwargs(tmp_path)
    output = tmp_path / "mixed-report"
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
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        comparator.main()
    assert exc.value.code == 2
    assert not (output / "report.json").exists()

    recorded[1]["provenance"]["config_identity"]["scenario_matrix_hash"] = recorded[0][
        "provenance"
    ]["config_identity"]["scenario_matrix_hash"]
    recorded[1]["provenance"]["config_identity"]["campaign_config_hash"] = "f" * 16
    path.write_text("".join(json.dumps(row) + "\n" for row in recorded))
    with pytest.raises(ValueError, match="campaign_config_hash differs from pinned campaign"):
        _compare(bundle, root, digest)


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


def test_forged_runtime_hashes_matching_result_root_exit_two(tmp_path: Path, monkeypatch) -> None:
    row = _row("s1", 111)
    bundle, digest = _archive(tmp_path, [row])
    root = _root(tmp_path, [row])
    manifest_path = tmp_path / "successor-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    campaign_path = root / "campaign_manifest.json"
    campaign = json.loads(campaign_path.read_text())
    for binding, field in (
        ("campaign_config", "config_hash"),
        ("scenario_matrix", "scenario_matrix_hash"),
    ):
        manifest[binding]["runtime_hash"] = "f" * len(manifest[binding]["runtime_hash"])
        campaign[field] = manifest[binding]["runtime_hash"]
    manifest_path.write_text(json.dumps(manifest))
    campaign_path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="runtime_hash differs from pinned source"):
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
    successor = _successor_kwargs(tmp_path)
    output = tmp_path / "forged-report"
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
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        comparator.main()
    assert exc.value.code == 2
    assert not (output / "report.json").exists()


def test_runtime_hash_and_planner_resolution_use_pinned_code(tmp_path: Path) -> None:
    manifest_path, _, source, commit = _successor_contract(tmp_path)
    original = json.loads(manifest_path.read_text())["campaign_config"]["runtime_hash"]
    utils = source / "robot_sf/benchmark/utils.py"
    code = utils.read_text()
    target = "return hashlib.sha256(data).hexdigest()[:16]"
    assert target in code
    utils.write_text(code.replace(target, 'return "a" * 16', 1))
    resolver = source / "robot_sf/benchmark/map_runner_policies/map_runner_policy_resolution.py"
    code = resolver.read_text()
    anchor = "    manifest = (\n        dict(algo_config)"
    assert anchor in code
    resolver.write_text(
        code.replace(
            anchor,
            '    if default_algo == "goal":\n        return default_algo, {"pinned_runtime_marker": True}\n'
            + anchor,
            1,
        )
    )
    subprocess.run(["git", "-C", str(source), "add", "robot_sf/benchmark"], check=True)
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
            "different runtime hash",
        ],
        check=True,
    )
    changed_commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert changed_commit != commit
    slot = ("goal", "differential_drive", "s1", 111, "")
    rows = {
        slot: {
            "_provenance": {
                "scenario_params": {
                    "name": "s1",
                    "id": "s1",
                    "map_file": "maps/svg_maps/classic_crossing.svg",
                    "algo": "goal",
                    "seed": 111,
                }
            }
        }
    }
    observed, _, runtime_rows, _, _ = comparator._runtime_successor_identity(
        source, changed_commit, "configs/benchmarks/synthetic.yaml", rows
    )
    assert observed == "a" * 16
    assert observed != original
    assert runtime_rows[slot]["config"] == {"pinned_runtime_marker": True}


def test_forged_row_map_with_recomputed_self_hash_exits_two(tmp_path: Path, monkeypatch) -> None:
    row = _row("s1", 111)
    bundle, digest = _archive(tmp_path, [row])
    root = _root(tmp_path, [row])
    path = root / "runs/goal__differential_drive/episodes.jsonl"
    forged = json.loads(path.read_text())
    forged["scenario_params"]["map_file"] = "maps/svg_maps/forged.svg"
    forged["config_hash"] = _config_hash(forged["scenario_params"])
    path.write_text(json.dumps(forged) + "\n")
    with pytest.raises(ValueError, match="map_file differs from pinned scenario"):
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
    successor = _successor_kwargs(tmp_path)
    output = tmp_path / "forged-map-report"
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
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        comparator.main()
    assert exc.value.code == 2
    assert not (output / "report.json").exists()


def test_row_identity_alias_must_match_pinned_scenario(tmp_path: Path) -> None:
    row = _row("s1", 111)
    bundle, digest = _archive(tmp_path, [row])
    root = _root(tmp_path, [row])
    path = root / "runs/goal__differential_drive/episodes.jsonl"
    forged = json.loads(path.read_text())
    forged["scenario_params"]["id"] = "s2"
    forged["config_hash"] = _config_hash(forged["scenario_params"])
    path.write_text(json.dumps(forged) + "\n")
    with pytest.raises(ValueError, match="scenario provenance differs from run slot"):
        _compare(bundle, root, digest)


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("run_horizon", 300),
        ("run_dt", 0.2),
        ("record_forces", False),
        ("observation_mode", "wrong_mode"),
        ("record_simulation_step_trace", True),
        ("seed", 112),
    ],
)
def test_recorded_episode_controls_must_match_pinned_h600_campaign(
    tmp_path: Path, field: str, wrong: object, monkeypatch
) -> None:
    row = _row("s1", 111)
    bundle, digest = _archive(tmp_path, [row])
    root = _root(tmp_path, [row])
    path = root / "runs/goal__differential_drive/episodes.jsonl"
    changed = json.loads(path.read_text())
    changed["scenario_params"][field] = wrong
    changed["config_hash"] = _config_hash(changed["scenario_params"])
    path.write_text(json.dumps(changed) + "\n")
    with pytest.raises(ValueError, match="pinned (episode controls|slot)"):
        _compare(bundle, root, digest)
    if field == "run_horizon":
        _assert_cli_exit_two(tmp_path, bundle, root, digest, monkeypatch)


def test_differential_drive_row_declaring_holonomic_exits_two(tmp_path: Path, monkeypatch) -> None:
    row = _row("s1", 111)
    bundle, digest = _archive(tmp_path, [row])
    root = _root(tmp_path, [row])
    path = root / "runs/goal__differential_drive/episodes.jsonl"
    changed = json.loads(path.read_text())
    changed["scenario_params"]["robot_config"]["type"] = "holonomic"
    changed["config_hash"] = _config_hash(changed["scenario_params"])
    path.write_text(json.dumps(changed) + "\n")
    with pytest.raises(ValueError, match="robot_config.type differs from pinned scoped runner"):
        _compare(bundle, root, digest)
    _assert_cli_exit_two(tmp_path, bundle, root, digest, monkeypatch)


def test_holonomic_row_command_mode_must_match_scoped_runner(tmp_path: Path, monkeypatch) -> None:
    row = _row("s1", 111)
    run = "goal__holonomic"
    bundle, digest = _archive(tmp_path, [row], run=run)
    root = _root(tmp_path, [row], run=run)
    path = root / "runs" / run / "episodes.jsonl"
    changed = json.loads(path.read_text())
    changed["scenario_params"]["robot_config"]["command_mode"] = "unicycle_vw"
    changed["config_hash"] = _config_hash(changed["scenario_params"])
    path.write_text(json.dumps(changed) + "\n")
    with pytest.raises(
        ValueError, match="holonomic command_mode differs from pinned scoped runner"
    ):
        _compare(bundle, root, digest)
    _assert_cli_exit_two(tmp_path, bundle, root, digest, monkeypatch)


def test_v4_scenario_algorithm_override_to_goal_exits_two(tmp_path: Path, monkeypatch) -> None:
    v4 = "hybrid_rule_v4_fast_progress_static_escape"
    row = _row("s1", 111, planner=v4)
    bundle, digest = _archive(tmp_path, [_row("s1", 111)])
    root = _root(tmp_path, [row], run=f"{v4}__differential_drive")
    source = tmp_path / "successor-source"
    config = source / f"configs/algos/{v4}.yaml"
    config.write_text(config.read_text() + "scenario_algo_overrides:\n  s1:\n    algo: goal\n")
    subprocess.run(["git", "-C", str(source), "add", "--sparse", str(config)], check=True)
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
            "override v4 with goal",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest_path = tmp_path / "successor-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_commit"] = commit
    manifest["versioned_planner_bindings"][v4]["sha256"] = sha256(config.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    path = root / "runs" / f"{v4}__differential_drive" / "episodes.jsonl"
    changed = json.loads(path.read_text())
    changed["git_hash"] = commit
    changed["algo"] = "goal"
    changed["scenario_params"]["algo"] = "goal"
    changed["scenario_params"]["observation_mode"] = "goal_state"
    changed["scenario_params"]["observation_level"] = "oracle_full_state"
    changed["algorithm_metadata"]["algorithm"] = "goal"
    changed["algorithm_metadata"]["config"] = {}
    changed["algorithm_metadata"]["config_hash"] = _config_hash({})
    changed["provenance"]["commit_hash"] = commit
    changed["provenance"]["config_identity"]["algo"] = "goal"
    changed["config_hash"] = _config_hash(changed["scenario_params"])
    path.write_text(json.dumps(changed) + "\n")
    campaign_path = root / "campaign_manifest.json"
    campaign = json.loads(campaign_path.read_text())
    campaign["git"]["commit"] = commit
    campaign_path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="v4 slot resolves wrong algorithm"):
        _compare(bundle, root, digest)
    _assert_cli_exit_two(tmp_path, bundle, root, digest, monkeypatch)


def test_v4_key_bound_to_v3_config_fails_even_with_re_pinned_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    v4 = "hybrid_rule_v4_fast_progress_static_escape"
    row = _row("s1", 111, planner=v4)
    bundle, digest = _archive(tmp_path, [row], run=f"{v4}__differential_drive")
    root = _root(tmp_path, [row], run=f"{v4}__differential_drive")
    source = tmp_path / "successor-source"
    config = source / f"configs/algos/{v4}.yaml"
    config.write_text(
        "base_config_path: configs/algos/hybrid_rule_v3_teb_like_rollout.yaml\n"
        f"params:\n  name: {v4}\n"
    )
    (source / "configs/algos/hybrid_rule_v3_teb_like_rollout.yaml").write_text(
        "speed: 1\nplanner_variant: hybrid_rule_v3_teb_like_rollout\n"
    )
    subprocess.run(["git", "-C", str(source), "add", "--sparse", "configs/algos"], check=True)
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
            "swap reviewed v4 binding to v3",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest_path = tmp_path / "successor-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_commit"] = commit
    manifest["versioned_planner_bindings"][v4]["sha256"] = sha256(config.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    row_path = root / "runs" / f"{v4}__differential_drive" / "episodes.jsonl"
    recorded = json.loads(row_path.read_text())
    recorded["git_hash"] = commit
    recorded["provenance"]["commit_hash"] = commit
    row_path.write_text(json.dumps(recorded) + "\n")
    campaign_path = root / "campaign_manifest.json"
    campaign = json.loads(campaign_path.read_text())
    campaign["git"]["commit"] = commit
    campaign_path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="reviewed v4 lineage"):
        _compare(bundle, root, digest)
    _assert_cli_exit_two(tmp_path, bundle, root, digest, monkeypatch)


def test_goal_row_under_v4_run_directory_exits_two(tmp_path: Path, monkeypatch) -> None:
    row = _row("s1", 111)
    bundle, digest = _archive(tmp_path, [row])
    v4 = "hybrid_rule_v4_fast_progress_static_escape"
    root = _root(tmp_path, [row], run=f"{v4}__differential_drive")
    with pytest.raises(ValueError, match="row algorithm differs"):
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
    successor = _successor_kwargs(tmp_path)
    output = tmp_path / "wrong-planner-report"
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
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        comparator.main()
    assert exc.value.code == 2
    assert not (output / "report.json").exists()


def test_effective_planner_config_and_referenced_path_must_match(tmp_path: Path) -> None:
    v4 = "hybrid_rule_v4_fast_progress_static_escape"
    row = _row("s1", 111, planner=v4)
    bundle, digest = _archive(tmp_path, [row], run=f"{v4}__differential_drive")
    root = _root(tmp_path, [row], run=f"{v4}__differential_drive")
    path = root / "runs" / f"{v4}__differential_drive" / "episodes.jsonl"
    original = json.loads(path.read_text())
    original["algorithm_metadata"]["config"] = {"name": "wrong"}
    path.write_text(json.dumps(original) + "\n")
    with pytest.raises(ValueError, match="effective planner config differs"):
        _compare(bundle, root, digest)
    original["algorithm_metadata"]["config"] = {
        "speed": 1,
        "planner_variant": "hybrid_rule_v4_clearance_braking",
        "name": v4,
    }
    original["provenance"]["config_identity"]["algo_config_path"] = "configs/algos/other.yaml"
    path.write_text(json.dumps(original) + "\n")
    with pytest.raises(ValueError, match="run provenance differs"):
        _compare(bundle, root, digest)
    original["provenance"]["config_identity"]["algo_config_path"] = json.loads(
        (tmp_path / "successor-manifest.json").read_text()
    )["versioned_planner_bindings"][v4]["path"]
    original["provenance"]["commit_hash"] = "f" * 40
    path.write_text(json.dumps(original) + "\n")
    with pytest.raises(ValueError, match="run provenance source differs"):
        _compare(bundle, root, digest)


def test_broad_rules_are_reported_first_without_rejection(tmp_path: Path) -> None:
    row = _row("s1", 111)
    bundle, digest = _archive(tmp_path, [row])
    changed = copy.deepcopy(row)
    changed["metrics"]["path_length"] += 1.0
    root = _root(tmp_path, [changed])
    all_seeds = _rule("metrics.path_length")
    all_seeds["slots"][0]["seeds"] = "all"
    large_bound = _rule("metrics.collisions", scenario="s2")
    large_bound["predicate"]["max_abs_delta"] = 5.0
    rules = _rules(tmp_path / "rules.json", [all_seeds, large_bound])

    report = _compare(bundle, root, digest, rules)
    assert report["status"] == "classified"
    assert report["broad_rules"] == [
        {"rule_id": all_seeds["rule_id"], "reasons": ["seeds: all", "max_abs_delta: 2.0"]},
        {"rule_id": large_bound["rule_id"], "reasons": ["max_abs_delta: 5.0"]},
    ]
    report = compare(
        bundle,
        root,
        classification_file=rules,
        broad_rule_bound_threshold=6.0,
        baseline_sha256=digest,
        baseline_source=OLD_SOURCE,
        expected_scenario_identity=SCENARIO,
        **_successor_kwargs(tmp_path),
    )
    assert [rule["rule_id"] for rule in report["broad_rules"]] == [all_seeds["rule_id"]]
    output = tmp_path / "broad-report"
    write_report(report, output)
    summary = (output / "summary.md").read_text()
    assert summary.index("## Broad rules") < summary.index("## Rule coverage")


@pytest.mark.parametrize(
    "declared", ["goal", "hybrid_rule_v3_fast_progress_static_escape_continuous"]
)
def test_pinned_v4_lineage_rejects_contradictory_config_declaration(tmp_path: Path, declared: str):
    """Optional lineage declarations cannot contradict the source-pinned arm map."""
    _, _, source, _ = _successor_contract(tmp_path, enable_all_v4=True)
    key = "hybrid_rule_v4_fast_progress_static_escape"
    path = source / f"configs/algos/{key}.yaml"
    text = path.read_text()
    assert "replaces_0_0_7_slot: hybrid_rule_v3_fast_progress_static_escape\n" in text
    path.write_text(
        text.replace(
            "replaces_0_0_7_slot: hybrid_rule_v3_fast_progress_static_escape\n",
            f"replaces_0_0_7_slot: {declared}\n",
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(Path(comparator.__file__).with_name("_pinned_successor_runtime.py")),
        ],
        cwd=source,
        input=json.dumps(
            {
                "config_path": "configs/benchmarks/synthetic.yaml",
                "versioned_keys": sorted(V4_SLOT_REPLACEMENTS.values()),
                "rows": [],
            }
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert f"successor planner binding lacks reviewed v4 lineage: {key}" in result.stderr

@pytest.mark.parametrize("mixed_successor_definitions", [False, True])
def test_changed_definitions_suppress_paired_metric_delta(tmp_path, mixed_successor_definitions):
    """Read actual JSONL bytes and fence metrics even when numerical values agree."""
    old = _row("s", 1001)
    other = _row("s", 1002)
    bundle, digest = _archive(tmp_path, [old, other] if mixed_successor_definitions else [old])
    new = copy.deepcopy(old)
    new["metric_schema_version"] = "robot-sf-metrics.v2"
    new["metrics"]["metric_schema_version"] = "robot-sf-metrics.v2"
    root = _root(tmp_path, [new, other] if mixed_successor_definitions else [new])
    report = _compare(bundle, root, digest)
    summary = next(
        x for x in report["planner_scenario_metrics"] if x["field"] == "metrics.path_length"
    )
    assert summary["mean_paired_delta"] is None
    assert summary["metric_comparability"] == "incompatible_definitions"
    finding = next(x for x in report["findings"] if x["field"] == "metrics.path_length")
    assert finding["classification"] == "metric_definition_change"
    assert finding["delta_0_0_8_minus_0_0_7"] is None
