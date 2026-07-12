"""Tests for the issue #5419 authorization-gated dense DPCBF episode executor.

This is the slice that turns ``execute_run_plan`` from an always-raise gate into a bounded,
explicitly authorized *local* executor. The tests pin the fail-closed contract the issue asks
for:

- **No / wrong authorization ID** -> raises before any output file is created.
- **Correct authorization ID** -> all three arms are dispatched through the canonical benchmark
  runner in packet order, each with the shared scenario manifest and its distinct algorithm
  config, and a machine-readable manifest is written.
- **A failing arm** stays a visible caveat in the manifest, blocks an overall ``complete``
  status, and does not cause later arms to be misreported as successful.
- **A repeated invocation** safely resumes matching provenance or fails closed on mismatch.
- One tiny **smoke** proves the real ``run_batch`` integration with a reduced horizon/repeats
  through a test-only fixture (not the tracked result packet), making no performance claim.

Claim boundary: diagnostic/contract tests only. Even the smoke runs a reduced, test-only
fixture and asserts *that episodes were produced*, never a safety-performance or
collision-reduction result. Failed/degraded rows are caveats, never success evidence.
"""

from __future__ import annotations

import copy
import json
import pathlib
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest
import yaml

from robot_sf.benchmark.issue_4142_dpcbf_dense_readiness import (
    PACKET_PATH,
    REQUIRED_ARMS,
)
from robot_sf.benchmark.issue_4142_dpcbf_dense_runner import (
    EPISODE_SCHEMA_PATH,
    EXECUTION_MANIFEST_FILENAME,
    EXECUTION_MANIFEST_SCHEMA_VERSION,
    REQUIRED_AUTHORIZATION_ID,
    DenseComparisonExecutionGatedError,
    DenseComparisonProvenanceMismatchError,
    _resolve_execution_inputs,
    build_run_plan,
    execute_run_plan,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# A well-formed synthetic packet whose arm configs pass readiness. ``algorithm`` is
# ``simple_policy`` so the smoke can run real episodes on CPU without MPC dependencies.
_RUNNABLE_PACKET = {
    "schema_version": "robot_sf.issue_4142_dpcbf_dense_comparison.v1",
    "canonical_command": "uv run python scripts/tools/run_issue_4142_dpcbf_dense_comparison.py",
    "scenario_manifest": "configs/scenarios/sets/dense.yaml",
    "algorithm": "simple_policy",
    "algorithm_configs": {
        "cbf_off": "configs/algos/off.yaml",
        "cbf_collision_cone_on": "configs/algos/cone.yaml",
        "cbf_dynamic_parabolic_v1_on": "configs/algos/dpcbf.yaml",
    },
    "execution": {
        "base_seed": 7,
        "repeats": 1,
        "horizon": 3,
        "dt": 0.1,
        "workers": 1,
        "video_enabled": False,
        "resume": True,
    },
    "runtime_cbf_arms": [
        {"enabled": False, "arm_key": "cbf_off"},
        {"enabled": True, "arm_key": "cbf_collision_cone_on"},
        {
            "enabled": True,
            "arm_key": "cbf_dynamic_parabolic_v1_on",
            "variant": "dynamic_parabolic_cbf_v1",
        },
    ],
    "summary_contract": {
        "evidence_tier": "bounded_runtime_comparison",
        "fallback_rows_are_success_evidence": False,
        "excluded_row_statuses": ["fallback", "degraded", "failed", "ineligible"],
        "required_arms": [
            "cbf_off",
            "cbf_collision_cone_on",
            "cbf_dynamic_parabolic_v1_on",
        ],
    },
}


def _write_runnable_tree(root: pathlib.Path, packet: dict) -> pathlib.Path:
    """Materialize a minimal repo tree with a *runnable* scenario manifest under ``root``."""
    (root / "configs/algos").mkdir(parents=True, exist_ok=True)
    (root / "configs/scenarios/sets").mkdir(parents=True, exist_ok=True)
    (root / "configs/research").mkdir(parents=True, exist_ok=True)

    # Abstract (non-map) scenarios the canonical runner can execute quickly on CPU. Written as
    # a multi-document YAML stream so ``load_scenario_matrix`` returns them directly instead of
    # routing through the map-oriented manifest loader (which expects a name + map reference).
    scenarios = [
        {
            "id": "dpcbf-smoke-a",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": 1,
        },
        {
            "id": "dpcbf-smoke-b",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": 1,
        },
    ]
    (root / "configs/scenarios/sets/dense.yaml").write_text(
        yaml.safe_dump_all(scenarios, sort_keys=False), encoding="utf-8"
    )
    (root / "configs/algos/off.yaml").write_text("algorithm: simple_policy\n", encoding="utf-8")
    (root / "configs/algos/cone.yaml").write_text(
        "cbf_safety_filter:\n  enabled: true\n", encoding="utf-8"
    )
    (root / "configs/algos/dpcbf.yaml").write_text(
        "cbf_safety_filter:\n  enabled: true\n  variant: dynamic_parabolic_cbf_v1\n",
        encoding="utf-8",
    )
    packet_path = root / "configs/research/packet.yaml"
    packet_path.write_text(yaml.safe_dump(packet, sort_keys=False), encoding="utf-8")
    return packet_path


def _success_summary(total: int = 1) -> dict[str, Any]:
    """A run_batch summary that reports every scheduled job written with no failures."""
    return {"total_jobs": total, "written": total, "failures": []}


class _RecordingRunBatch:
    """A fake ``run_batch`` that records each call and returns a caller-chosen summary."""

    def __init__(self, summary_for=None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._summary_for = summary_for or (lambda idx, kwargs: _success_summary())

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        idx = len(self.calls)
        self.calls.append(kwargs)
        return self._summary_for(idx, kwargs)


def _config_stem(path: str) -> str:
    """Return the arm config filename for order/identity assertions."""
    return pathlib.Path(path).name


# --- Authorization gate (fail closed before any write) ---


def test_no_authorization_raises_before_creating_output(tmp_path: pathlib.Path) -> None:
    """Without an authorization ID the executor raises and creates no output directory."""
    out_dir = tmp_path / "out"
    plan = build_run_plan(repo_root=REPO_ROOT, packet_path=PACKET_PATH, output_dir=out_dir)
    fake = _RecordingRunBatch()

    with pytest.raises(DenseComparisonExecutionGatedError):
        execute_run_plan(plan, repo_root=REPO_ROOT, run_batch_fn=fake)

    assert not out_dir.exists()
    assert fake.calls == []  # no arm dispatched


def test_wrong_authorization_raises_before_creating_output(tmp_path: pathlib.Path) -> None:
    """A wrong authorization ID (incl. a truthy non-matching string) fails closed, no writes."""
    out_dir = tmp_path / "out"
    plan = build_run_plan(repo_root=REPO_ROOT, packet_path=PACKET_PATH, output_dir=out_dir)
    fake = _RecordingRunBatch()

    with pytest.raises(DenseComparisonExecutionGatedError):
        execute_run_plan(plan, authorization="not-the-id", repo_root=REPO_ROOT, run_batch_fn=fake)
    with pytest.raises(DenseComparisonExecutionGatedError):
        # A bare boolean-ish truthy value is also insufficient.
        execute_run_plan(plan, authorization="true", repo_root=REPO_ROOT, run_batch_fn=fake)

    assert not out_dir.exists()
    assert fake.calls == []


def test_unresolved_plan_cannot_execute_even_with_authorization(tmp_path: pathlib.Path) -> None:
    """A blocked plan raises before authorization even matters -- no output created."""
    packet = copy.deepcopy(_RUNNABLE_PACKET)
    packet["runtime_cbf_arms"] = packet["runtime_cbf_arms"][:2]  # drop the DPCBF arm
    packet["summary_contract"]["required_arms"] = ["cbf_off", "cbf_collision_cone_on"]
    packet_path = _write_runnable_tree(tmp_path, packet)
    out_dir = tmp_path / "out"
    plan = build_run_plan(repo_root=tmp_path, packet_path=packet_path, output_dir=out_dir)
    assert plan.is_executable_in_principle is False

    with pytest.raises(DenseComparisonExecutionGatedError):
        execute_run_plan(
            plan,
            authorization=REQUIRED_AUTHORIZATION_ID,
            repo_root=tmp_path,
            run_batch_fn=_RecordingRunBatch(),
        )
    assert not out_dir.exists()


# --- Authorized dispatch + manifest ---


def test_authorized_run_dispatches_all_arms_in_packet_order(tmp_path: pathlib.Path) -> None:
    """Correct ID dispatches the three arms in order, shared manifest, distinct configs."""
    out_dir = tmp_path / "out"
    plan = build_run_plan(repo_root=REPO_ROOT, packet_path=PACKET_PATH, output_dir=out_dir)
    fake = _RecordingRunBatch()

    manifest = execute_run_plan(
        plan,
        authorization=REQUIRED_AUTHORIZATION_ID,
        repo_root=REPO_ROOT,
        run_batch_fn=fake,
    )

    # One dispatch per required arm, in packet order (identified by algorithm config).
    assert len(fake.calls) == len(REQUIRED_ARMS)
    dispatched_configs = [_config_stem(c["algo_config_path"]) for c in fake.calls]
    assert dispatched_configs == [_config_stem(job.algorithm_config) for job in plan.arms]
    # All arms share the one scenario manifest; each arm has a distinct algo config + output.
    assert len({c["scenarios_or_path"] for c in fake.calls}) == 1
    assert len({c["algo_config_path"] for c in fake.calls}) == len(REQUIRED_ARMS)
    assert len({c["out_path"] for c in fake.calls}) == len(REQUIRED_ARMS)
    # Bounded execution inputs are threaded through to the runner.
    for call in fake.calls:
        assert call["horizon"] == plan.execution_inputs.horizon
        assert call["repeats_override"] == plan.execution_inputs.repeats
        assert call["video_enabled"] is False

    # Manifest reflects a complete run and is persisted under the output directory.
    assert manifest.status == "complete"
    assert tuple(a.arm_key for a in manifest.arms) == REQUIRED_ARMS
    assert all(a.status == "executed" for a in manifest.arms)
    manifest_file = out_dir / EXECUTION_MANIFEST_FILENAME
    assert manifest_file.is_file()


def test_manifest_contains_required_provenance_fields(tmp_path: pathlib.Path) -> None:
    """The persisted manifest carries the full provenance the issue requires."""
    out_dir = tmp_path / "out"
    plan = build_run_plan(repo_root=REPO_ROOT, packet_path=PACKET_PATH, output_dir=out_dir)
    deterministic_now = datetime(2026, 7, 12, 12, 0, 0, tzinfo=UTC)
    execute_run_plan(
        plan,
        authorization=REQUIRED_AUTHORIZATION_ID,
        repo_root=REPO_ROOT,
        run_batch_fn=_RecordingRunBatch(),
        now_fn=lambda: deterministic_now,
    )
    payload = json.loads((out_dir / EXECUTION_MANIFEST_FILENAME).read_text(encoding="utf-8"))

    assert payload["schema_version"] == EXECUTION_MANIFEST_SCHEMA_VERSION
    assert payload["packet_schema_version"] == plan.packet_schema_version
    assert payload["plan_schema_version"] == plan.schema_version
    assert payload["authorization_id"] == REQUIRED_AUTHORIZATION_ID
    assert "git_sha" in payload and isinstance(payload["git_dirty"], bool)
    assert payload["effective_arguments"]["horizon"] == plan.execution_inputs.horizon
    assert payload["started_at"] == "2026-07-12T12:00:00+00:00"
    assert payload["ended_at"] == "2026-07-12T12:00:00+00:00"
    assert payload["status"] == "complete"
    # Fail-closed exclusion is carried into the manifest.
    for status in ("fallback", "degraded", "failed", "ineligible"):
        assert status in payload["excluded_row_statuses"]
    # Per-arm output paths and statuses are all present.
    assert {a["arm_key"] for a in payload["arms"]} == set(REQUIRED_ARMS)
    assert all(a["output_jsonl"] for a in payload["arms"])


def test_failing_arm_stays_visible_and_blocks_complete(tmp_path: pathlib.Path) -> None:
    """A failing middle arm is a visible caveat; later arms keep their true status."""
    out_dir = tmp_path / "out"
    plan = build_run_plan(repo_root=REPO_ROOT, packet_path=PACKET_PATH, output_dir=out_dir)

    def summary_for(idx: int, _kwargs: dict[str, Any]) -> dict[str, Any]:
        if idx == 1:  # the second (collision-cone) arm fails hard
            raise RuntimeError("simulated arm failure")
        return _success_summary()

    fake = _RecordingRunBatch(summary_for=summary_for)
    manifest = execute_run_plan(
        plan,
        authorization=REQUIRED_AUTHORIZATION_ID,
        repo_root=REPO_ROOT,
        run_batch_fn=fake,
    )

    # All arms were attempted in order; the failure did not abort the run.
    assert len(fake.calls) == len(REQUIRED_ARMS)
    by_arm = {a.arm_key: a for a in manifest.arms}
    assert by_arm["cbf_collision_cone_on"].status == "failed"
    assert by_arm["cbf_collision_cone_on"].error is not None
    # The later arm ran and reports its real (executed) status -- not misreported off a failure.
    assert by_arm["cbf_dynamic_parabolic_v1_on"].status == "executed"
    # A failing arm blocks an overall complete status.
    assert manifest.status == "results_incomplete"


def test_arm_writing_fewer_than_scheduled_is_not_success(tmp_path: pathlib.Path) -> None:
    """An arm that writes fewer episodes than scheduled is a caveat, never success."""
    out_dir = tmp_path / "out"
    plan = build_run_plan(repo_root=REPO_ROOT, packet_path=PACKET_PATH, output_dir=out_dir)

    def summary_for(idx: int, _kwargs: dict[str, Any]) -> dict[str, Any]:
        if idx == 0:
            return {"total_jobs": 4, "written": 1, "failures": []}  # partial write
        return _success_summary()

    manifest = execute_run_plan(
        plan,
        authorization=REQUIRED_AUTHORIZATION_ID,
        repo_root=REPO_ROOT,
        run_batch_fn=_RecordingRunBatch(summary_for=summary_for),
    )
    by_arm = {a.arm_key: a for a in manifest.arms}
    assert by_arm["cbf_off"].status == "failed"
    assert manifest.status == "results_incomplete"


# --- Provenance-safe resume ---


def test_repeat_run_matching_provenance_resumes(tmp_path: pathlib.Path) -> None:
    """A second run with identical provenance does not fail closed."""
    out_dir = tmp_path / "out"
    plan = build_run_plan(repo_root=REPO_ROOT, packet_path=PACKET_PATH, output_dir=out_dir)
    execute_run_plan(
        plan,
        authorization=REQUIRED_AUTHORIZATION_ID,
        repo_root=REPO_ROOT,
        run_batch_fn=_RecordingRunBatch(),
    )
    # Re-run: same plan, same git provenance -> allowed (no mismatch raised).
    manifest = execute_run_plan(
        plan,
        authorization=REQUIRED_AUTHORIZATION_ID,
        repo_root=REPO_ROOT,
        run_batch_fn=_RecordingRunBatch(),
    )
    assert manifest.status == "complete"


def test_repeat_run_mismatched_provenance_fails_closed(tmp_path: pathlib.Path) -> None:
    """A pre-existing manifest with a different provenance key blocks a silent resume."""
    out_dir = tmp_path / "out"
    plan = build_run_plan(repo_root=REPO_ROOT, packet_path=PACKET_PATH, output_dir=out_dir)
    execute_run_plan(
        plan,
        authorization=REQUIRED_AUTHORIZATION_ID,
        repo_root=REPO_ROOT,
        run_batch_fn=_RecordingRunBatch(),
    )
    # Tamper the recorded provenance so the next run would mix incompatible artifacts.
    manifest_file = out_dir / EXECUTION_MANIFEST_FILENAME
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    payload["provenance_key"] = "incompatible-provenance"
    manifest_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DenseComparisonProvenanceMismatchError):
        execute_run_plan(
            plan,
            authorization=REQUIRED_AUTHORIZATION_ID,
            repo_root=REPO_ROOT,
            run_batch_fn=_RecordingRunBatch(),
        )


def test_non_mapping_manifest_fails_closed(tmp_path: pathlib.Path) -> None:
    """A syntactically valid non-object manifest cannot be used for resume."""
    out_dir = tmp_path / "out"
    plan = build_run_plan(repo_root=REPO_ROOT, packet_path=PACKET_PATH, output_dir=out_dir)
    execute_run_plan(
        plan,
        authorization=REQUIRED_AUTHORIZATION_ID,
        repo_root=REPO_ROOT,
        run_batch_fn=_RecordingRunBatch(),
    )
    (out_dir / EXECUTION_MANIFEST_FILENAME).write_text("[]", encoding="utf-8")

    with pytest.raises(DenseComparisonProvenanceMismatchError, match="not a dictionary"):
        execute_run_plan(
            plan,
            authorization=REQUIRED_AUTHORIZATION_ID,
            repo_root=REPO_ROOT,
            run_batch_fn=_RecordingRunBatch(),
        )


def test_changed_execution_inputs_fail_closed_on_resume(tmp_path: pathlib.Path) -> None:
    """Changed worker/schema provenance cannot reuse an existing execution manifest."""
    out_dir = tmp_path / "out"
    plan = build_run_plan(repo_root=REPO_ROOT, packet_path=PACKET_PATH, output_dir=out_dir)
    execute_run_plan(
        plan,
        authorization=REQUIRED_AUTHORIZATION_ID,
        repo_root=REPO_ROOT,
        run_batch_fn=_RecordingRunBatch(),
    )
    changed_inputs = replace(plan.execution_inputs, workers=2)
    changed_plan = replace(plan, execution_inputs=changed_inputs)

    with pytest.raises(DenseComparisonProvenanceMismatchError):
        execute_run_plan(
            changed_plan,
            authorization=REQUIRED_AUTHORIZATION_ID,
            repo_root=REPO_ROOT,
            schema_path=tmp_path / "alternate-episode-schema.json",
            run_batch_fn=_RecordingRunBatch(),
        )


def test_non_boolean_resume_is_a_plan_blocker() -> None:
    """String values cannot silently change resume/append semantics."""
    inputs, blockers = _resolve_execution_inputs({"execution": {"resume": "false"}})
    assert inputs.resume is True
    assert any("resume must be a boolean" in blocker for blocker in blockers)


# --- Smoke: real run_batch integration (reduced horizon/repeats, no performance claim) ---


def test_smoke_real_run_batch_writes_episodes(tmp_path: pathlib.Path) -> None:
    """Authorized run through the *real* ``run_batch`` writes per-arm JSONL for every arm.

    This proves the executor is wired to the canonical benchmark runner. It uses a reduced,
    test-only fixture (horizon 3, 1 repeat, ``simple_policy``) and asserts only that episodes
    were produced -- it makes no safety-performance or collision-reduction claim.
    """
    packet_path = _write_runnable_tree(tmp_path, copy.deepcopy(_RUNNABLE_PACKET))
    out_dir = tmp_path / "out"
    plan = build_run_plan(repo_root=tmp_path, packet_path=packet_path, output_dir=out_dir)
    assert plan.is_executable_in_principle is True
    assert plan.execution_inputs.horizon == 3  # bounded fixture inputs honored

    manifest = execute_run_plan(
        plan,
        authorization=REQUIRED_AUTHORIZATION_ID,
        repo_root=tmp_path,
        # The episode schema is a real repo asset, not part of the synthetic fixture tree.
        schema_path=REPO_ROOT / EPISODE_SCHEMA_PATH,
    )

    assert manifest.status == "complete"
    assert tuple(a.arm_key for a in manifest.arms) == REQUIRED_ARMS
    for arm in manifest.arms:
        assert arm.status == "executed"
        assert arm.written > 0
        artifact = tmp_path / arm.output_jsonl
        assert artifact.is_file()
        assert artifact.read_text(encoding="utf-8").strip()  # at least one episode row
