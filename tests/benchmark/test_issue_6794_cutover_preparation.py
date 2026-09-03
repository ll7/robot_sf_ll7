"""Tests for the Issue #6794 checkpoint-cutover preparation contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark import issue_6794_cutover_preparation as preparation_module
from robot_sf.benchmark.issue_6794_cutover_preparation import (
    compare_parity_rows,
    main,
    validate_preparation_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs/benchmarks/issue_6794_phase_c_parity_preparation_v1.yaml"


def test_preparation_contract_freezes_current_bytes_and_load_paths() -> None:
    """The checked-in packet validates without staging or running a campaign."""
    report = validate_preparation_contract(REPO_ROOT, CONFIG)

    assert report["status"] == "prepared_not_executed"
    assert report["claim_boundary"] == "provenance_and_protocol_only"
    assert set(report["checkpoints"]) == {"default_ppo", "ga3c_cadrl"}
    assert len(report["load_paths"]) == 8
    protocol = report["parity_protocol"]
    assert protocol["seeds"] == [111, 112, 113]
    assert len(protocol["scenario_ids"]) == 48
    assert protocol["after_mode"] == "registry_release_identity_preparation"
    assert protocol["after_resolution_mode"] == "registry_release_hydrated_checkpoint"
    assert protocol["planner_arms"][1]["execution_mode"] == "adapter"
    assert protocol["planner_arms"][1]["adapter_name"] == "SACADRLPlannerAdapter"
    assert all(row["runtime_probe"] == "deferred_until_hydration" for row in report["load_paths"])
    assert report["checkpoints"]["ga3c_cadrl"]["release_bundle_files"]
    assert report["checkpoints"]["ga3c_cadrl"]["load_contract"] == {
        "source_shape": "tensorflow_checkpoint_prefix_verified",
        "registry_local_path": "in_tree_prefix_present",
        "runtime_loader_probe": "deferred_until_hydration",
    }


def test_preparation_contract_accepts_relative_config_path() -> None:
    """The public validator supports its documented repository-relative default."""
    report = validate_preparation_contract(REPO_ROOT)

    assert report["status"] == "prepared_not_executed"


def test_checkpoint_kind_and_registry_local_path_are_pinned() -> None:
    """Ignored checkpoint-shape or resolver-path edits fail the preparation contract."""
    config = preparation_module.yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    registry = preparation_module.load_registry(REPO_ROOT / "model/registry.yaml")
    checkpoint = dict(config["checkpoint_snapshots"]["ga3c_cadrl"])
    checkpoint["registry_local_path"] = "model/ga3c_cadrl/IROS18/network_01900000.index"
    with pytest.raises(ValueError, match="local path disagrees"):
        preparation_module._checkpoint_report(REPO_ROOT, registry, "ga3c_cadrl", checkpoint)

    checkpoint = dict(config["checkpoint_snapshots"]["ga3c_cadrl"])
    checkpoint["registry_release_asset_name"] = "different.tar.gz"
    with pytest.raises(ValueError, match="release asset disagrees"):
        preparation_module._checkpoint_report(REPO_ROOT, registry, "ga3c_cadrl", checkpoint)

    checkpoint = dict(config["checkpoint_snapshots"]["ga3c_cadrl"])
    checkpoint["kind"] = "single_file"
    with pytest.raises(ValueError, match="single_file kind needs one source path"):
        preparation_module._checkpoint_report(REPO_ROOT, registry, "ga3c_cadrl", checkpoint)


def test_preparation_contract_rejects_escape_paths_and_malformed_arms(tmp_path: Path) -> None:
    """Declared inputs and protocol arms fail closed before any file is consumed."""
    outside = tmp_path.parent / "outside-checkpoint.txt"
    outside.write_text("not a checkpoint", encoding="utf-8")
    link = tmp_path / "checkpoint.txt"
    link.symlink_to(outside)

    with pytest.raises(ValueError, match="resolve within the repository"):
        preparation_module._repo_declared_path(tmp_path, "checkpoint.txt", name="checkpoint")
    with pytest.raises(ValueError, match="repository-relative"):
        preparation_module._repo_declared_path(
            tmp_path, "../outside-checkpoint.txt", name="checkpoint"
        )
    with pytest.raises(ValueError, match="two mapping arms"):
        preparation_module._validate_protocol_arms({"planner_arms": [{"key": "ppo"}, "malformed"]})


def test_load_path_inventory_rejects_escape_and_unknown_checkpoint() -> None:
    """Load-path selectors stay confined and reference a declared checkpoint."""
    with pytest.raises(ValueError, match="repository-relative"):
        preparation_module._validate_load_paths(
            REPO_ROOT,
            {
                "checkpoint_snapshots": {"default": {}},
                "load_path_inventory": [
                    {
                        "id": "escape",
                        "checkpoint": "default",
                        "path": "../outside.py",
                        "selector": "anything",
                    }
                ],
            },
        )
    with pytest.raises(ValueError, match="unknown checkpoint"):
        preparation_module._validate_load_paths(
            REPO_ROOT,
            {
                "checkpoint_snapshots": {"default": {}},
                "load_path_inventory": [
                    {
                        "id": "unknown",
                        "checkpoint": "missing",
                        "path": "README.md",
                        "selector": "anything",
                    }
                ],
            },
        )


def test_preparation_contract_rejects_protocol_type_coercion() -> None:
    """Boolean lookalikes cannot satisfy frozen numeric or arm contracts."""
    config = preparation_module.yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["parity_protocol"]["workers"] = True
    with pytest.raises(ValueError, match="workers must be an integer"):
        preparation_module._validate_protocol(REPO_ROOT, config["parity_protocol"])

    config = preparation_module.yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["parity_protocol"]["comparison"]["float_rel_tolerance"] = False
    with pytest.raises(ValueError, match="finite number"):
        preparation_module._validate_protocol(REPO_ROOT, config["parity_protocol"])

    config = preparation_module.yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["parity_protocol"]["planner_arms"][0]["execution_mode"] = "adapter"
    with pytest.raises(ValueError, match="unexpected execution_mode"):
        preparation_module._validate_protocol(REPO_ROOT, config["parity_protocol"])


def test_preparation_config_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    """Duplicate YAML members cannot silently replace a preparation gate."""
    config = tmp_path / "duplicate.yaml"
    config.write_text(
        "schema_version: legacy-checkpoint-cutover-preparation.v1\n"
        "schema_version: legacy-checkpoint-cutover-preparation.v1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key"):
        preparation_module._load_preparation_config(config)


def test_release_component_set_must_match_declared_sources() -> None:
    """A bundle cannot silently omit or add registry components."""
    digest = "a" * 64
    with pytest.raises(ValueError, match="member set|component set"):
        preparation_module._verify_release_components(
            "bundle",
            {
                "registry_per_file_sha256": {
                    "one.data": digest,
                    "one.meta": digest,
                    "extra.meta": digest,
                }
            },
            {
                "per_file_sha256": {
                    "one.data": digest,
                    "one.meta": digest,
                    "extra.meta": digest,
                },
                "bundle_files": ["model/one.data", "model/one.meta", "model/extra.meta"],
            },
            {"model/one.data": digest, "model/one.meta": digest},
        )


def test_source_digest_keys_and_paths_must_match_exactly(tmp_path: Path) -> None:
    """A checkpoint cannot hide duplicate paths or unverified digest entries."""
    source = tmp_path / "checkpoint.bin"
    source.write_bytes(b"checkpoint")
    digest = preparation_module._sha256(source)

    with pytest.raises(ValueError, match="duplicate source path"):
        preparation_module._verify_source_files(
            tmp_path,
            "checkpoint",
            ["checkpoint.bin", "checkpoint.bin"],
            {"checkpoint.bin": digest},
        )
    with pytest.raises(ValueError, match="source_sha256 keys"):
        preparation_module._verify_source_files(
            tmp_path,
            "checkpoint",
            ["checkpoint.bin"],
            {"checkpoint.bin": digest, "unlisted.bin": digest},
        )


def test_bundle_source_basenames_must_be_unique() -> None:
    """Basename-keyed registry bundles cannot represent ambiguous source paths."""
    digest = "a" * 64
    with pytest.raises(ValueError, match="basenames must be unique"):
        preparation_module._verify_release_components(
            "bundle",
            {"registry_per_file_sha256": {"checkpoint.data": digest}},
            {
                "per_file_sha256": {"checkpoint.data": digest},
                "bundle_files": ["left/checkpoint.data", "right/checkpoint.data"],
            },
            {"left/checkpoint.data": digest, "right/checkpoint.data": digest},
        )


def _strict_comparison_args() -> dict:
    """Return the frozen comparator bindings for a small fixture matrix."""
    protocol = validate_preparation_contract(REPO_ROOT, CONFIG)["parity_protocol"]
    return {
        "expected_keys": [("ppo", "fixture.scenario", seed) for seed in (111, 112)],
        "expected_execution_modes": {"ppo": "native"},
        "expected_algorithms": {"ppo": "ppo"},
        "expected_adapter_names": {},
        "required_provenance_fields": protocol["required_provenance_fields"],
        "expected_provenance": {
            "before": {"ppo": dict(protocol["expected_provenance"]["before"]["ppo"])},
            "after": {"ppo": dict(protocol["expected_provenance"]["after"]["ppo"])},
        },
        "episode_schema_path": REPO_ROOT / protocol["episode_schema"],
    }


def _row(  # noqa: PLR0913
    seed: int,
    *,
    delta: float = 0.0,
    status: str = "native",
    planner_key: str = "ppo",
    scenario_id: str = "fixture.scenario",
    execution_mode: str = "native",
    algorithm: str = "ppo",
    readiness_status: str = "native",
    availability_status: str = "available",
    benchmark_success: object = True,
    provenance: dict | None = None,
) -> dict:
    """Return one complete synthetic canonical parity row."""
    supports_native = execution_mode == "native"
    supports_adapter = execution_mode in {"adapter", "mixed"}
    algorithm_metadata = {
        "algorithm": algorithm,
        "canonical_algorithm": algorithm,
        "status": "ok",
        "planner_kinematics": {
            "execution_mode": execution_mode,
            "supports_native_commands": supports_native,
            "supports_adapter_commands": supports_adapter,
            "adapter_name": (
                "SACADRLPlannerAdapter" if algorithm == "sacadrl" else "ppo_action_to_unicycle"
            ),
            "adapter_active": supports_adapter,
        },
    }
    row = {
        "version": "v1",
        "episode_id": f"{scenario_id}--{seed}",
        "planner_key": planner_key,
        "scenario_id": scenario_id,
        "seed": seed,
        "row_status": status,
        "execution_mode": execution_mode,
        "readiness_status": readiness_status,
        "availability_status": availability_status,
        "benchmark_success": benchmark_success,
        "benchmark_success_basis": "all",
        "termination_reason": "success",
        "metrics": {
            "success": 1.0,
            "collisions": 0.0,
            "near_misses": 1.0 + delta,
            "time_to_goal_norm": 0.4,
            "snqi": 0.2,
        },
        "algorithm_metadata": algorithm_metadata,
        "config_hash": "a" * 16,
        "git_hash": "b" * 40,
        "outcome": {
            "route_complete": True,
            "collision_event": False,
            "timeout_event": False,
        },
        "integrity": {"contradictions": []},
        "status": "success",
    }
    if provenance is not None:
        parity_provenance: dict[str, object] = {}
        for field, value in provenance.items():
            if field in {"config_hash", "git_hash"}:
                row[field] = value
                continue
            if not field.startswith("parity_provenance."):
                continue
            cursor = parity_provenance
            parts = field.split(".")[1:]
            for part in parts[:-1]:
                nested = cursor.setdefault(part, {})
                assert isinstance(nested, dict)
                cursor = nested
            cursor[parts[-1]] = value
        parity_provenance.setdefault("resolution_receipt_sha256", "c" * 64)
        receipt = parity_provenance.setdefault("resolution_receipt", {})
        assert isinstance(receipt, dict)
        receipt.setdefault("cache_path", "output/fixture-cache")
        receipt.setdefault(
            "resolved_path",
            (
                next(iter(parity_provenance["source_sha256"]), "model/fixture-checkpoint")
                if parity_provenance.get("resolution_mode") == "in_tree_checkpoint"
                else "output/fixture-cache/resolved-checkpoint"
            ),
        )
        row["parity_provenance"] = parity_provenance
    return row


def _write_rows(path: Path, rows: list[dict]) -> None:
    """Write test JSONL rows."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_compare_parity_rows_accepts_unchanged_native_fixture(tmp_path: Path) -> None:
    """The future parity harness accepts identical rows without executing a campaign."""
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    args = _strict_comparison_args()
    before_rows = [
        _row(seed, provenance=args["expected_provenance"]["before"]["ppo"]) for seed in (111, 112)
    ]
    after_rows = [
        _row(seed, provenance=args["expected_provenance"]["after"]["ppo"]) for seed in (111, 112)
    ]
    _write_rows(before, before_rows)
    _write_rows(after, after_rows)

    report = compare_parity_rows(before, after, **args)

    assert report["status"] == "passed"
    assert report["compared_rows"] == 2
    assert all(delta["delta"] == 0.0 for delta in report["metric_deltas"])


def test_compare_parity_rows_rejects_status_and_metric_drift(tmp_path: Path) -> None:
    """Any status drift or metric delta beyond the frozen tolerance fails closed."""
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    args = _strict_comparison_args()
    _write_rows(
        before,
        [_row(111, provenance=args["expected_provenance"]["before"]["ppo"])],
    )
    _write_rows(
        after,
        [
            _row(
                111,
                delta=1e-6,
                status="fallback",
                provenance=args["expected_provenance"]["after"]["ppo"],
            )
        ],
    )

    report = compare_parity_rows(before, after, **args)

    assert report["status"] == "failed"
    assert any("status drift" in blocker for blocker in report["blockers"])
    assert any("metric drift" in blocker for blocker in report["blockers"])


def test_compare_parity_rows_uses_execution_mode_for_native_admission(tmp_path: Path) -> None:
    """Native execution is a separate field from the evidence row status."""
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    args = _strict_comparison_args()
    _write_rows(
        before,
        [_row(111, provenance=args["expected_provenance"]["before"]["ppo"])],
    )
    _write_rows(
        after,
        [
            _row(
                111,
                execution_mode="adapter",
                provenance=args["expected_provenance"]["after"]["ppo"],
            )
        ],
    )

    report = compare_parity_rows(before, after, **args)

    assert report["status"] == "failed"
    assert any("execution mode drift" in blocker for blocker in report["blockers"])


def test_compare_parity_rows_rejects_non_success_episode_status(tmp_path: Path) -> None:
    """A successful benchmark flag cannot mask a failed canonical episode status."""
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    args = _strict_comparison_args()
    before_row = _row(111, provenance=args["expected_provenance"]["before"]["ppo"])
    after_row = _row(111, provenance=args["expected_provenance"]["after"]["ppo"])
    after_row["status"] = "failure"
    _write_rows(before, [before_row])
    _write_rows(after, [after_row])

    report = compare_parity_rows(before, after, **args)

    assert report["status"] == "failed"
    assert any("non-success episode status" in blocker for blocker in report["blockers"])


def test_compare_parity_rows_rejects_unhydrated_after_resolution(tmp_path: Path) -> None:
    """The future release arm cannot report the in-tree source as its resolved path."""
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    args = _strict_comparison_args()
    before_row = _row(111, provenance=args["expected_provenance"]["before"]["ppo"])
    after_row = _row(111, provenance=args["expected_provenance"]["after"]["ppo"])
    source_path = next(
        iter(args["expected_provenance"]["after"]["ppo"]["parity_provenance.source_sha256"])
    )
    after_row["parity_provenance"]["resolution_receipt"]["resolved_path"] = source_path
    _write_rows(before, [before_row])
    _write_rows(after, [after_row])

    report = compare_parity_rows(before, after, **args)

    assert report["status"] == "failed"
    assert any("isolated hydrated path" in blocker for blocker in report["blockers"])


def test_compare_parity_rows_rejects_nested_preflight_fallback(
    tmp_path: Path,
) -> None:
    """Nested canonical preflight fallback cannot pass behind a native row label."""
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    args = _strict_comparison_args()
    before_row = _row(111, provenance=args["expected_provenance"]["before"]["ppo"])
    after_row = _row(111, provenance=args["expected_provenance"]["after"]["ppo"])
    after_row["preflight"] = {"status": "fallback"}
    _write_rows(before, [before_row])
    _write_rows(after, [after_row])

    report = compare_parity_rows(before, after, **args)

    assert report["status"] == "failed"
    assert any("fallback/degraded marker" in blocker for blocker in report["blockers"])
    assert any("not canonically benchmark-available" in blocker for blocker in report["blockers"])


def test_compare_parity_rows_accepts_sacadrl_only_as_an_adapter(
    tmp_path: Path,
) -> None:
    """The SACADRL arm is admissible only with its canonical adapter metadata."""
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    protocol = validate_preparation_contract(REPO_ROOT, CONFIG)["parity_protocol"]
    args = {
        "expected_keys": [("sacadrl", "fixture.scenario", 111)],
        "expected_execution_modes": {"sacadrl": "adapter"},
        "expected_algorithms": {"sacadrl": "sacadrl"},
        "expected_adapter_names": {"sacadrl": "SACADRLPlannerAdapter"},
        "required_provenance_fields": protocol["required_provenance_fields"],
        "expected_provenance": {
            "before": {"sacadrl": dict(protocol["expected_provenance"]["before"]["sacadrl"])},
            "after": {"sacadrl": dict(protocol["expected_provenance"]["after"]["sacadrl"])},
        },
        "episode_schema_path": REPO_ROOT / protocol["episode_schema"],
    }
    _write_rows(
        before,
        [
            _row(
                111,
                planner_key="sacadrl",
                execution_mode="adapter",
                algorithm="sacadrl",
                readiness_status="adapter",
                provenance=args["expected_provenance"]["before"]["sacadrl"],
            )
        ],
    )
    _write_rows(
        after,
        [
            _row(
                111,
                planner_key="sacadrl",
                execution_mode="adapter",
                algorithm="sacadrl",
                readiness_status="adapter",
                provenance=args["expected_provenance"]["after"]["sacadrl"],
            )
        ],
    )

    report = compare_parity_rows(before, after, **args)

    assert report["status"] == "passed"


def test_compare_parity_rows_rejects_coerced_identity_and_status_types(tmp_path: Path) -> None:
    """Self-consistent boolean/string lookalikes cannot pass parity comparison."""
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    args = _strict_comparison_args()
    malformed_before = _row(111, provenance=args["expected_provenance"]["before"]["ppo"])
    malformed_after = _row(111, provenance=args["expected_provenance"]["after"]["ppo"])
    malformed_before["seed"] = True
    malformed_after["seed"] = True
    _write_rows(before, [malformed_before])
    _write_rows(after, [malformed_after])
    with pytest.raises(ValueError, match="seed"):
        compare_parity_rows(before, after, **args)

    malformed_before = _row(111, provenance=args["expected_provenance"]["before"]["ppo"])
    malformed_after = _row(111, provenance=args["expected_provenance"]["after"]["ppo"])
    malformed_before["benchmark_success"] = "true"
    malformed_after["benchmark_success"] = "true"
    _write_rows(before, [malformed_before])
    _write_rows(after, [malformed_after])
    report = compare_parity_rows(before, after, **args)
    assert report["status"] == "failed"
    assert any("invalid status field type" in blocker for blocker in report["blockers"])


def test_compare_parity_rows_rejects_ambiguous_or_nonfinite_jsonl(tmp_path: Path) -> None:
    """JSONL duplicate keys and non-finite extensions fail before comparison."""
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    args = _strict_comparison_args()
    serialized = json.dumps(
        _row(111, provenance=args["expected_provenance"]["before"]["ppo"])
    ).replace('"seed": 111', '"seed": 111, "seed": 111', 1)
    before.write_text(serialized + "\n", encoding="utf-8")
    after.write_text(serialized + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        compare_parity_rows(before, after, **args)

    malformed = _row(111, provenance=args["expected_provenance"]["before"]["ppo"])
    malformed["diagnostics"] = float("nan")
    _write_rows(before, [malformed])
    _write_rows(after, [malformed])
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        compare_parity_rows(before, after, **args)


def _full_protocol_rows(protocol: dict, side: str) -> list[dict]:
    """Build a complete synthetic 48-scenario, two-arm protocol output."""
    rows: list[dict] = []
    for arm in protocol["planner_arms"]:
        arm_key = str(arm["key"])
        for scenario_id in protocol["scenario_ids"]:
            for seed in protocol["seeds"]:
                rows.append(
                    _row(
                        seed,
                        planner_key=arm_key,
                        scenario_id=scenario_id,
                        execution_mode=str(arm["execution_mode"]),
                        algorithm=str(arm["algo"]),
                        readiness_status=(
                            "adapter" if arm["execution_mode"] == "adapter" else "native"
                        ),
                        provenance=protocol["expected_provenance"][side][arm_key],
                    )
                )
    return rows


def test_cli_uses_protocol_metric_paths_for_parity_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI comparisons resolve frozen metric names under the row metrics mapping."""
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    protocol = validate_preparation_contract(REPO_ROOT, CONFIG)["parity_protocol"]
    _write_rows(before, _full_protocol_rows(protocol, "before"))
    _write_rows(after, _full_protocol_rows(protocol, "after"))

    exit_code = main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--config",
            str(CONFIG),
            "--before-episodes",
            str(before),
            "--after-episodes",
            str(after),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["comparison"]["status"] == "passed"
    assert payload["comparison"]["expected_rows"] == 288
    assert payload["comparison"]["compared_rows"] == 288


def test_cli_rejects_one_sided_comparison_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI requires both before and after outputs for a comparison."""
    exit_code = main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--config",
            str(CONFIG),
            "--before-episodes",
            str(tmp_path / "before.jsonl"),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "failed"
    assert "supplied together" in payload["blockers"][0]
