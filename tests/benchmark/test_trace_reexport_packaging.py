"""Synthetic contract tests for deterministic issue #5756 trace packaging."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tarfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import pytest

from robot_sf.analysis_workbench.simulation_trace_export import load_simulation_trace_export
from robot_sf.benchmark.trace_reexport_packaging import (
    EXECUTION_COMMIT,
    PPO_CHECKPOINT_SHA256,
    FrozenTraceReexportContract,
    TraceReexportPackagingError,
    campaign_expectations,
    canonical_sha256,
    expected_outcomes_payload_for_rows,
    package_trace_reexport,
)
from robot_sf.benchmark.utils import _config_hash
from scripts.tools.build_simulation_trace_export import build_simulation_trace_export

REPO_ROOT = Path(__file__).resolve().parents[2]


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_json_bytes(row) for row in rows))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _outcome(planner: str, scenario: str, seed: int) -> dict[str, bool]:
    if scenario == "classic_doorway_medium":
        success = seed % 2 == 1
    elif planner == "goal":
        success = seed % 2 == 0
    else:
        success = seed % 2 == 1
    return {
        "success": success,
        "route_complete": success,
        "collision_event": not success,
        "timeout_event": False,
    }


def _release_row(planner: str, scenario: str, seed: int) -> dict[str, Any]:
    outcome = _outcome(planner, scenario, seed)
    return {
        "episode_id": f"release-{planner}-{scenario}-{seed}",
        "scenario_id": scenario,
        "seed": seed,
        "metrics": {"success": outcome["success"]},
        "outcome": {key: outcome[key] for key in outcome if key != "success"},
    }


def _trace_frame() -> dict[str, Any]:
    return {
        "step": 0,
        "time_s": 0.1,
        "robot": {"position": [0.0, 0.0], "heading": 0.0, "velocity": [0.0, 0.0]},
        "pedestrians": [],
        "planner": {
            "event": "step",
            "selected_action": {"linear_velocity": 0.0, "angular_velocity": 0.0},
        },
    }


def _rerun_row(planner: str, scenario: str, seed: int, *, prefix: str = "rerun") -> dict[str, Any]:
    outcome = _outcome(planner, scenario, seed)
    params = {
        "id": scenario,
        "algo": planner,
        "record_forces": True,
        "record_planner_decision_trace": True,
        "record_simulation_step_trace": True,
        "run_horizon": 600,
        "run_dt": 0.1,
    }
    metadata: dict[str, Any] = {
        "planner_kinematics": {"robot_kinematics": "differential_drive"},
        "planner_decision_trace": {
            "schema_version": "planner-decision-trace.v1",
            "steps": [],
        },
        "simulation_step_trace": {
            "schema_version": "simulation-step-trace.v1",
            "steps": [_trace_frame()],
        },
    }
    if planner == "ppo":
        metadata["planner_runtime"] = {
            "checkpoint_provenance": {
                "checkpoint_sha256": PPO_CHECKPOINT_SHA256,
                "load_succeeded": True,
                "fallback_triggered": False,
            }
        }
    return {
        "episode_id": f"{prefix}-{planner}-{scenario}-{seed}",
        "scenario_id": scenario,
        "seed": seed,
        "git_hash": EXECUTION_COMMIT,
        "scenario_params": params,
        "config_hash": _config_hash(params),
        "metrics": {"success": outcome["success"]},
        "outcome": {key: outcome[key] for key in outcome if key != "success"},
        "algorithm_metadata": metadata,
    }


def _add_tar_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(data)
    member.mtime = 0
    archive.addfile(member, io.BytesIO(data))


@dataclass
class SyntheticInputs:
    """Mutable synthetic release, request, and rerun inputs for one test."""

    root: Path
    release_bundle: Path
    request_manifest: Path
    outputs: dict[str, Path]
    release_rows: dict[str, list[dict[str, Any]]]
    contract: FrozenTraceReexportContract

    def kwargs(self, output_dir: Path) -> dict[str, Any]:
        return {
            "release_bundle": self.release_bundle,
            "request_manifest": self.request_manifest,
            "canary_output": self.outputs["canary"],
            "ppo_output": self.outputs["ppo"],
            "goal_output": self.outputs["goal"],
            "output_dir": output_dir,
            "repo_root": REPO_ROOT,
            "contract": self.contract,
        }

    def output_rows(self, label: str) -> list[dict[str, Any]]:
        planner = "goal" if label == "goal" else "ppo"
        return _read_jsonl(
            self.outputs[label] / "runs" / f"{planner}__differential_drive" / "episodes.jsonl"
        )

    def write_output_rows(self, label: str, rows: list[dict[str, Any]]) -> None:
        planner = "goal" if label == "goal" else "ppo"
        _write_jsonl(
            self.outputs[label] / "runs" / f"{planner}__differential_drive" / "episodes.jsonl",
            rows,
        )

    def manifest(self, label: str) -> dict[str, Any]:
        return json.loads((self.outputs[label] / "campaign_manifest.json").read_text())

    def write_manifest(self, label: str, payload: dict[str, Any]) -> None:
        _write_json(self.outputs[label] / "campaign_manifest.json", payload)

    def rebuild_release(
        self,
        *,
        repin_embedded: bool = True,
        member_prefix: str = "bundle/",
    ) -> None:
        streams = {
            planner: b"".join(_json_bytes(row) for row in rows)
            for planner, rows in self.release_rows.items()
        }
        files = [
            {
                "path": f"runs/{planner}__differential_drive/episodes.jsonl",
                "sha256": _sha256_bytes(data),
            }
            for planner, data in sorted(streams.items())
        ]
        with tarfile.open(self.release_bundle, "w") as archive:
            for planner, data in streams.items():
                _add_tar_bytes(
                    archive,
                    f"{member_prefix}payload/runs/{planner}__differential_drive/episodes.jsonl",
                    data,
                )
            _add_tar_bytes(
                archive,
                f"{member_prefix}publication_manifest.json",
                _json_bytes({"files": files}),
            )
        updates: dict[str, str] = {"release_bundle_sha256": _sha256_file(self.release_bundle)}
        if repin_embedded:
            updates.update(
                release_goal_jsonl_sha256=_sha256_bytes(streams["goal"]),
                release_ppo_jsonl_sha256=_sha256_bytes(streams["ppo"]),
            )
        self.contract = replace(self.contract, **updates)
        self.refresh_expected_digest()

    def repin_request(self) -> None:
        self.contract = replace(
            self.contract,
            request_manifest_sha256=_sha256_file(self.request_manifest),
        )
        self.refresh_expected_digest()

    def refresh_expected_digest(self) -> None:
        indexed = {
            (planner, str(row["scenario_id"]), int(row["seed"])): row
            for planner, rows in self.release_rows.items()
            for row in rows
        }
        payload = expected_outcomes_payload_for_rows(indexed, contract=self.contract)
        self.contract = replace(self.contract, expected_outcomes_sha256=canonical_sha256(payload))


@pytest.fixture
def synthetic_inputs(tmp_path: Path) -> SyntheticInputs:
    expectations = campaign_expectations(REPO_ROOT)
    release_rows = {"goal": [], "ppo": []}
    for expectation in expectations.values():
        if expectation.label == "canary":
            continue
        release_rows[expectation.planner].extend(
            _release_row(expectation.planner, scenario, seed)
            for scenario in expectation.scenarios
            for seed in expectation.seeds
        )

    outputs: dict[str, Path] = {}
    for label, expectation in expectations.items():
        root = tmp_path / label
        outputs[label] = root
        rows = [
            _rerun_row(expectation.planner, scenario, seed, prefix=f"{label}-rerun")
            for scenario in expectation.scenarios
            for seed in expectation.seeds
        ]
        _write_jsonl(
            root / "runs" / f"{expectation.planner}__differential_drive" / "episodes.jsonl", rows
        )
        planner_entry: dict[str, Any] = {"key": expectation.planner}
        if expectation.planner == "ppo":
            planner_entry["checkpoint_provenance"] = {
                "checkpoint_sha256": PPO_CHECKPOINT_SHA256,
                "load_succeeded": True,
                "fallback_triggered": False,
            }
        _write_json(
            root / "campaign_manifest.json",
            {
                "name": expectation.name,
                "scenario_matrix": "configs/scenarios/classic_interactions_francis2023.yaml",
                "scenario_matrix_hash": expectation.scenario_matrix_hash,
                "config_hash": expectation.config_hash,
                "scenario_candidates": list(expectation.scenario_candidates),
                "git": {"commit": EXECUTION_COMMIT},
                "seed_policy": {"resolved_seeds": list(expectation.seeds)},
                "planners": [planner_entry],
            },
        )

    request_rows = [
        {
            "planner": planner,
            "scenario_id": row["scenario_id"],
            "seed": str(row["seed"]),
            "episode_id": row["episode_id"],
            "episode_id_status": "found",
        }
        for planner, rows in release_rows.items()
        for row in rows
    ]
    request_manifest = tmp_path / "requests.json"
    _write_json(
        request_manifest,
        {
            "schema_version": "issue_5446_trace_reexport_list.v1",
            "n_tuples": 90,
            "tuples": request_rows,
        },
    )
    inputs = SyntheticInputs(
        root=tmp_path,
        release_bundle=tmp_path / "release.tar",
        request_manifest=request_manifest,
        outputs=outputs,
        release_rows=release_rows,
        contract=FrozenTraceReexportContract(
            release_bundle_sha256="",
            request_manifest_sha256=_sha256_file(request_manifest),
            release_goal_jsonl_sha256="",
            release_ppo_jsonl_sha256="",
            expected_outcomes_sha256="",
        ),
    )
    inputs.rebuild_release()
    return inputs


def _tree_digests(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def _input_digests(synthetic_inputs: SyntheticInputs) -> dict[str, Any]:
    return {
        "release_bundle": _sha256_file(synthetic_inputs.release_bundle),
        "request_manifest": _sha256_file(synthetic_inputs.request_manifest),
        **{label: _tree_digests(path) for label, path in synthetic_inputs.outputs.items()},
    }


def test_package_success_is_isolated_deterministic_and_idempotent(
    synthetic_inputs: SyntheticInputs,
) -> None:
    output = synthetic_inputs.root / "package"
    isolated_line_counts: list[int] = []

    def tracing_builder(source: Path, **kwargs: Any) -> dict[str, Any]:
        isolated_line_counts.append(len(source.read_text().splitlines()))
        return build_simulation_trace_export(source, **kwargs)

    package_trace_reexport(**synthetic_inputs.kwargs(output), trace_builder=tracing_builder)
    before = _tree_digests(output)
    marker_mtime = (output / "package_complete.json").stat().st_mtime_ns
    receipt = json.loads((output / "mapping_receipt.json").read_text())
    release_by_key = {
        (planner, row["scenario_id"], row["seed"]): row
        for planner, rows in synthetic_inputs.release_rows.items()
        for row in rows
    }
    rerun_by_key = {
        ("goal" if label == "goal" else "ppo", row["scenario_id"], row["seed"]): row
        for label in ("ppo", "goal")
        for row in synthetic_inputs.output_rows(label)
    }

    assert isolated_line_counts == [1] * 90
    assert len(receipt["rows"]) == 90
    assert all(row["release_episode_id"] != row["rerun_episode_id"] for row in receipt["rows"])
    assert len(list((output / "traces").rglob("*.json"))) == 90
    for row in receipt["rows"]:
        key = (row["planner"], row["scenario_id"], row["seed"])
        trace = load_simulation_trace_export(output / row["trace_uri"])
        assert trace.source.episode_id == row["rerun_episode_id"]
        assert row["release_row_sha256"] == _sha256_bytes(_json_bytes(release_by_key[key]))
        assert row["rerun_row_sha256"] == _sha256_bytes(_json_bytes(rerun_by_key[key]))
        assert _sha256_file(output / row["trace_uri"]) == row["trace_sha256"]

    package_trace_reexport(**synthetic_inputs.kwargs(output))
    assert _tree_digests(output) == before
    assert (output / "package_complete.json").stat().st_mtime_ns == marker_mtime


def test_package_accepts_release_members_rooted_at_payload(
    synthetic_inputs: SyntheticInputs,
) -> None:
    """A valid bundle need not wrap its payload directory in another directory."""
    synthetic_inputs.rebuild_release(member_prefix="")

    output = synthetic_inputs.root / "root-payload-package"
    package_trace_reexport(**synthetic_inputs.kwargs(output))

    receipt = json.loads((output / "mapping_receipt.json").read_text())
    assert len(receipt["rows"]) == 90


@pytest.mark.parametrize("case", ["missing", "duplicate", "extra"])
def test_request_manifest_requires_exact_tuple_set(
    synthetic_inputs: SyntheticInputs, case: str
) -> None:
    payload = json.loads(synthetic_inputs.request_manifest.read_text())
    if case == "missing":
        payload["tuples"].pop()
        payload["n_tuples"] = 89
    elif case == "duplicate":
        payload["tuples"][0] = copy.deepcopy(payload["tuples"][1])
    else:
        payload["tuples"][0]["seed"] = "999"
    _write_json(synthetic_inputs.request_manifest, payload)
    synthetic_inputs.repin_request()

    with pytest.raises(TraceReexportPackagingError):
        package_trace_reexport(**synthetic_inputs.kwargs(synthetic_inputs.root / "package"))


@pytest.mark.parametrize("case", ["missing", "duplicate", "extra"])
def test_rerun_output_requires_exact_tuple_set(
    synthetic_inputs: SyntheticInputs, case: str
) -> None:
    rows = synthetic_inputs.output_rows("ppo")
    if case == "missing":
        rows.pop()
    elif case == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    else:
        extra = copy.deepcopy(rows[0])
        extra["seed"] = 999
        rows.append(extra)
    synthetic_inputs.write_output_rows("ppo", rows)

    with pytest.raises(TraceReexportPackagingError):
        package_trace_reexport(**synthetic_inputs.kwargs(synthetic_inputs.root / "package"))


def test_release_duplicate_tuple_is_ambiguous(synthetic_inputs: SyntheticInputs) -> None:
    synthetic_inputs.release_rows["goal"].append(
        copy.deepcopy(synthetic_inputs.release_rows["goal"][0])
    )
    synthetic_inputs.rebuild_release()

    with pytest.raises(TraceReexportPackagingError, match="duplicate/ambiguous"):
        package_trace_reexport(**synthetic_inputs.kwargs(synthetic_inputs.root / "package"))


@pytest.mark.parametrize("case", ["release", "request", "embedded"])
def test_input_digest_mismatch_fails_closed(synthetic_inputs: SyntheticInputs, case: str) -> None:
    if case == "release":
        synthetic_inputs.release_bundle.write_bytes(
            synthetic_inputs.release_bundle.read_bytes() + b"x"
        )
    elif case == "request":
        synthetic_inputs.request_manifest.write_bytes(
            synthetic_inputs.request_manifest.read_bytes() + b"\n"
        )
    else:
        synthetic_inputs.release_rows["ppo"][0]["metrics"]["success"] ^= True
        synthetic_inputs.rebuild_release(repin_embedded=False)

    with pytest.raises(TraceReexportPackagingError, match="SHA-256 mismatch"):
        package_trace_reexport(**synthetic_inputs.kwargs(synthetic_inputs.root / "package"))


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("commit", "execution commit"),
        ("campaign_config", "config_hash"),
        ("scenario_matrix", "scenario_matrix_hash"),
        ("seed_set", "resolved seed"),
        ("manifest_checkpoint", "checkpoint"),
        ("row_config", "scenario/config hash"),
        ("trace_flag", "record_simulation_step_trace"),
        ("decision_trace", "planner decision trace"),
        ("kinematics", "differential-drive"),
        ("row_checkpoint", "checkpoint"),
    ],
)
def test_rerun_provenance_mismatch_fails_closed(
    synthetic_inputs: SyntheticInputs, case: str, match: str
) -> None:
    if case in {"campaign_config", "scenario_matrix", "seed_set", "manifest_checkpoint"}:
        manifest = synthetic_inputs.manifest("canary")
        if case == "campaign_config":
            manifest["config_hash"] = "wrong"
        elif case == "scenario_matrix":
            manifest["scenario_matrix_hash"] = "wrong"
        elif case == "seed_set":
            manifest["seed_policy"]["resolved_seeds"] = [999]
        else:
            manifest["planners"][0]["checkpoint_provenance"]["checkpoint_sha256"] = "wrong"
        synthetic_inputs.write_manifest("canary", manifest)
    else:
        rows = synthetic_inputs.output_rows("canary")
        row = rows[0]
        if case == "commit":
            row["git_hash"] = "wrong"
        elif case == "row_config":
            row["config_hash"] = "wrong"
        elif case == "trace_flag":
            row["scenario_params"]["record_simulation_step_trace"] = False
            row["config_hash"] = _config_hash(row["scenario_params"])
        elif case == "decision_trace":
            row["algorithm_metadata"].pop("planner_decision_trace")
        elif case == "kinematics":
            row["algorithm_metadata"]["planner_kinematics"]["robot_kinematics"] = "holonomic"
        else:
            provenance = row["algorithm_metadata"]["planner_runtime"]["checkpoint_provenance"]
            provenance["checkpoint_sha256"] = "wrong"
        synthetic_inputs.write_output_rows("canary", rows)

    with pytest.raises(TraceReexportPackagingError, match=match):
        package_trace_reexport(**synthetic_inputs.kwargs(synthetic_inputs.root / "package"))


@pytest.mark.parametrize(
    "field",
    ["success", "route_complete", "collision_event", "timeout_event"],
)
def test_all_four_outcome_booleans_must_match(
    synthetic_inputs: SyntheticInputs, field: str
) -> None:
    rows = synthetic_inputs.output_rows("goal")
    container = rows[0]["metrics"] if field == "success" else rows[0]["outcome"]
    container[field] = not container[field]
    synthetic_inputs.write_output_rows("goal", rows)

    with pytest.raises(TraceReexportPackagingError, match="outcome mismatch"):
        package_trace_reexport(**synthetic_inputs.kwargs(synthetic_inputs.root / "package"))


def test_invalid_trace_schema_leaves_no_partial_complete_output(
    synthetic_inputs: SyntheticInputs,
) -> None:
    rows = synthetic_inputs.output_rows("goal")
    del rows[0]["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["planner"][
        "selected_action"
    ]
    synthetic_inputs.write_output_rows("goal", rows)
    output = synthetic_inputs.root / "package"

    with pytest.raises(ValueError, match="selected_action"):
        package_trace_reexport(**synthetic_inputs.kwargs(output))

    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.staging-*"))


def test_failed_repackage_does_not_replace_previous_complete_output(
    synthetic_inputs: SyntheticInputs,
) -> None:
    output = synthetic_inputs.root / "package"
    package_trace_reexport(**synthetic_inputs.kwargs(output))
    before = _tree_digests(output)
    rows = synthetic_inputs.output_rows("goal")
    rows[0]["outcome"]["timeout_event"] = not rows[0]["outcome"]["timeout_event"]
    synthetic_inputs.write_output_rows("goal", rows)

    with pytest.raises(TraceReexportPackagingError, match="outcome mismatch"):
        package_trace_reexport(**synthetic_inputs.kwargs(output))

    assert _tree_digests(output) == before
    assert json.loads((output / "package_complete.json").read_text())["status"] == "complete"
    assert not list(output.parent.glob(f".{output.name}.staging-*"))


def test_markerless_output_is_preserved(
    synthetic_inputs: SyntheticInputs,
) -> None:
    output = synthetic_inputs.root / "markerless"
    output.mkdir()
    (output / "unrelated.bin").write_bytes(b"unrelated output content")
    before_output = _tree_digests(output)
    before_inputs = _input_digests(synthetic_inputs)

    with pytest.raises(TraceReexportPackagingError, match="not a complete trace package"):
        package_trace_reexport(**synthetic_inputs.kwargs(output))

    assert _tree_digests(output) == before_output
    assert _input_digests(synthetic_inputs) == before_inputs
    assert not list(output.parent.glob(f".{output.name}.staging-*"))


@pytest.mark.parametrize(
    ("output_factory", "match"),
    [
        (lambda inputs: inputs.outputs["canary"], "overlaps raw canary input"),
        (lambda inputs: inputs.outputs["ppo"] / "nested" / "package", "overlaps raw PPO input"),
        (lambda inputs: inputs.root, "overlaps raw"),
    ],
    ids=["exact-input-alias", "input-ancestor-overlap", "input-descendant-overlap"],
)
def test_output_input_overlaps_are_rejected_without_mutation(
    synthetic_inputs: SyntheticInputs,
    output_factory: Callable[[SyntheticInputs], Path],
    match: str,
) -> None:
    output = output_factory(synthetic_inputs)
    before_output = _tree_digests(output)
    before_inputs = _input_digests(synthetic_inputs)

    with pytest.raises(TraceReexportPackagingError, match=match):
        package_trace_reexport(**synthetic_inputs.kwargs(output))

    assert _tree_digests(output) == before_output
    assert _input_digests(synthetic_inputs) == before_inputs
    assert not (synthetic_inputs.outputs["ppo"] / "nested").exists()


def test_resolver_mapping_receipt_adapts_a_complete_package(
    synthetic_inputs: SyntheticInputs,
) -> None:
    """The packager's complete package becomes a resolver-valid 90/90 mapping."""
    from robot_sf.benchmark.candidate_trace_resolution import (
        load_episode_mapping,
        load_episode_requests,
        resolve_episode_requests,
        validate_candidate_trace_resolution,
    )
    from robot_sf.benchmark.trace_reexport_packaging import (
        build_resolver_mapping_receipt,
        default_resolver_mapping_path,
    )

    package = synthetic_inputs.root / "package"
    package_trace_reexport(**synthetic_inputs.kwargs(package))
    package_before = _tree_digests(package)

    receipt_path = default_resolver_mapping_path(package)
    payload = build_resolver_mapping_receipt(package, output_path=receipt_path)
    assert payload["schema_version"] == "issue_5756_trace_mapping_receipt.v1"
    assert payload["n_rows"] == 90
    assert receipt_path.is_file()

    request_manifest = load_episode_requests(
        synthetic_inputs.request_manifest, expected_sha256=None
    )
    mapping = load_episode_mapping(receipt_path, expected_provenance=payload["provenance"])
    assert mapping.provenance["release_tag"] == "0.0.3"
    resolution = resolve_episode_requests(request_manifest, mapping)
    assert resolution["summary"]["n_resolved"] == 90
    assert validate_candidate_trace_resolution(resolution)["ok"]

    # A derived resolver receipt is not a package member. Repeating the conversion
    # leaves the complete marker/digests valid and produces identical canonical data.
    assert _tree_digests(package) == package_before
    assert build_resolver_mapping_receipt(package, output_path=receipt_path) == payload
    assert _tree_digests(package) == package_before
    package_trace_reexport(**synthetic_inputs.kwargs(package))
    assert _tree_digests(package) == package_before


def test_resolver_mapping_receipt_rejects_output_inside_complete_package(
    synthetic_inputs: SyntheticInputs,
) -> None:
    """An explicit interior destination cannot invalidate a complete package."""
    from robot_sf.benchmark.trace_reexport_packaging import build_resolver_mapping_receipt

    package = synthetic_inputs.root / "package"
    package_trace_reexport(**synthetic_inputs.kwargs(package))
    package_before = _tree_digests(package)

    with pytest.raises(TraceReexportPackagingError, match="outside the immutable complete package"):
        build_resolver_mapping_receipt(
            package,
            output_path=package / "resolver_mapping_receipt.json",
        )

    assert _tree_digests(package) == package_before


def test_resolver_cli_default_preserves_complete_package(
    synthetic_inputs: SyntheticInputs,
) -> None:
    """The operator-facing default writes the derived receipt beside the package."""
    from robot_sf.benchmark.trace_reexport_packaging import default_resolver_mapping_path
    from scripts.tools.package_issue_5756_trace_reexport import main

    package = synthetic_inputs.root / "package"
    package_trace_reexport(**synthetic_inputs.kwargs(package))
    package_before = _tree_digests(package)

    assert main(["to-resolver-mapping", "--package-dir", str(package)]) == 0
    assert default_resolver_mapping_path(package).is_file()
    assert _tree_digests(package) == package_before

    assert main(["to-resolver-mapping", "--package-dir", str(package)]) == 0
    assert _tree_digests(package) == package_before


def test_resolver_mapping_receipt_fails_closed_on_incomplete_package(
    synthetic_inputs: SyntheticInputs,
) -> None:
    """An empty directory cannot be adapted into a resolver mapping."""
    from robot_sf.benchmark.trace_reexport_packaging import build_resolver_mapping_receipt

    incomplete = synthetic_inputs.root / "incomplete"
    incomplete.mkdir()
    with pytest.raises(TraceReexportPackagingError, match="complete trace package"):
        build_resolver_mapping_receipt(incomplete)
