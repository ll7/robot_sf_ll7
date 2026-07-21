"""Tests for the issue #5303 durable transfer-run archival + provenance stage.

This slice (successor to PR #5845) adds the provenance-pinned archival of the
K x N transfer matrix: every run directory records an ``execution_context.txt``
(hostname/CPU/threads/commit) and a ``receipt_manifest.json`` with SHA-256
digests, exactly per the evidence-grade promotion plan. It does NOT run the
planner replay campaign (ops queue) and makes no benchmark claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.adversarial.provenance import (
    ExecutionContext,
    ReceiptItem,
    gather_execution_context,
    sha256_of_file,
    write_execution_context,
    write_receipt_manifest,
)
from robot_sf.adversarial.transfer_matrix import (
    DEFAULT_TRANSFER_ROSTER,
    PlannerEval,
    archive_transfer_run,
    build_transfer_matrix,
    select_certified_configs,
)

_TARGET_PLANNER = DEFAULT_TRANSFER_ROSTER[0]
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _certified_candidate(start_x: float, *, seed: int, objective: float) -> dict:
    return {
        "candidate": {
            "start": {"x": start_x, "y": 2.0, "theta": 0.0},
            "goal": {"x": 5.0, "y": 2.0, "theta": 0.0},
            "scenario_seed": seed,
        },
        "objective_value": objective,
        "certification_status": {
            "schema_version": "scenario_cert.v1",
            "status": "passed",
            "details": {
                "certificates": [
                    {"benchmark_eligibility": "eligible", "classification": "hard_but_solvable"}
                ]
            },
        },
    }


def _manifest(tmp_path: Path, *, candidates: list[dict], policy: str = _TARGET_PLANNER) -> Path:
    payload = {
        "schema_version": "adversarial-search-manifest.v1",
        "config": {
            "policy": policy,
            "scenario_template": "configs/scenarios/templates/crossing_ttc.yaml",
        },
        "candidates": candidates,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _evals_for_configs(configs, *, robustness: float, failed: bool):
    evals = []
    for cfg in configs:
        for planner in DEFAULT_TRANSFER_ROSTER:
            evals.append(
                PlannerEval(
                    config_id=cfg.config_id,
                    planner=planner,
                    robustness=robustness,
                    failed=failed,
                    seed=cfg.scenario_seed,
                )
            )
    return evals


def _built_matrix(tmp_path: Path, *, robustness: float, failed: bool):
    m = _manifest(
        tmp_path,
        candidates=[
            _certified_candidate(0.1 + i * 0.01, seed=700 + i, objective=float(i)) for i in range(6)
        ],
    )
    configs = select_certified_configs([m], target_planner=_TARGET_PLANNER, K=6)
    evals = _evals_for_configs(configs, robustness=robustness, failed=failed)
    return build_transfer_matrix(configs, evals)


def test_gather_execution_context_records_thread_env_and_schema():
    ctx = gather_execution_context(repo_root=_REPO_ROOT)
    assert isinstance(ctx, ExecutionContext)
    assert ctx.schema_version == "adversarial_execution_context.v1"
    assert ctx.hostname
    assert ctx.commit_sha
    assert "OMP_NUM_THREADS" in ctx.thread_env
    # When unset, the contract records "unset" rather than dropping the key.
    assert ctx.thread_env["OMP_NUM_THREADS"] in (None, "unset") or isinstance(
        ctx.thread_env["OMP_NUM_THREADS"], str
    )


def test_write_execution_context_persists_json(tmp_path):
    path = write_execution_context(tmp_path, repo_root=tmp_path)
    assert path.name == "execution_context.txt"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "adversarial_execution_context.v1"
    assert payload["hostname"]


def test_sha256_of_file_is_stable_and_distinct(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello", encoding="utf-8")
    b.write_text("world", encoding="utf-8")
    assert sha256_of_file(a) == sha256_of_file(a)
    assert sha256_of_file(a) != sha256_of_file(b)
    assert len(sha256_of_file(a)) == 64


def test_write_receipt_manifest_records_digests(tmp_path):
    item_a = tmp_path / "matrix.json"
    item_a.write_text("{}", encoding="utf-8")
    manifest_path = write_receipt_manifest(
        tmp_path,
        run_id="run-x",
        items=[
            ReceiptItem(
                artifact="transfer_matrix_json", path=item_a.name, digest=sha256_of_file(item_a)
            )
        ],
        execution_context_path="execution_context.txt",
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "adversarial_receipt_manifest.v1"
    assert payload["run_id"] == "run-x"
    assert payload["items"][0]["digest"] == sha256_of_file(item_a)


def test_archive_transfer_run_writes_pinned_artifacts(tmp_path):
    matrix = _built_matrix(tmp_path, robustness=-1.0, failed=True)
    run_dir = archive_transfer_run(
        matrix,
        archive_root=tmp_path / "archive",
        run_id="test-run",
        repo_root=_REPO_ROOT,
    )

    # Durable artifacts all present, under the adversarial archive subpath.
    assert run_dir.relative_to(tmp_path / "archive").parts[0] == "transfer_matrix"
    assert (run_dir / "transfer_matrix.json").exists()
    assert (run_dir / "transfer_report.md").exists()
    context_path = run_dir / "execution_context.txt"
    receipt_path = run_dir / "receipt_manifest.json"
    assert context_path.exists() and receipt_path.exists()

    # Receipt digests match the actual files and reference the context.
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    by_artifact = {item["artifact"]: item for item in receipt["items"]}
    assert by_artifact["transfer_matrix_json"]["digest"] == sha256_of_file(
        run_dir / "transfer_matrix.json"
    ), "transfer_matrix_json digest mismatch"
    assert by_artifact["transfer_report_md"]["digest"] == sha256_of_file(
        run_dir / "transfer_report.md"
    ), "transfer_report_md digest mismatch"
    assert by_artifact["execution_context"]["digest"] == sha256_of_file(context_path), (
        "execution_context digest mismatch"
    )
    assert receipt["execution_context_path"] == context_path.name, "execution_context_path mismatch"
    context = json.loads(context_path.read_text(encoding="utf-8"))
    assert context["commit_sha"], "archive execution context must pin a git commit"


def test_archive_transfer_run_rejects_under_sized_matrix(tmp_path):
    matrix = _built_matrix(tmp_path, robustness=1.0, failed=False)
    # Force fewer than 3 planners to exercise fail-closed archival.
    from robot_sf.adversarial.transfer_matrix import TransferMatrix

    small = TransferMatrix(
        target_planner=matrix.target_planner,
        configs=matrix.configs[:1],
        config_ids=matrix.config_ids[:1],
        planners=DEFAULT_TRANSFER_ROSTER[:2],
        cells=matrix.cells[:2],
        ranking=matrix.ranking[:2],
    )
    try:
        archive_transfer_run(small, archive_root=tmp_path / "archive", repo_root=tmp_path)
        raise AssertionError("expected ValueError for under-sized matrix")
    except ValueError:
        pass


def test_archive_transfer_run_rejects_unsafe_and_duplicate_run_ids(tmp_path):
    """Durable run IDs stay contained and never overwrite an existing archive."""
    matrix = _built_matrix(tmp_path, robustness=-1.0, failed=True)
    archive_root = tmp_path / "archive"

    with pytest.raises(ValueError, match="single 1-128 character path component"):
        archive_transfer_run(
            matrix,
            archive_root=archive_root,
            run_id="../outside",
            repo_root=_REPO_ROOT,
        )

    archive_transfer_run(
        matrix,
        archive_root=archive_root,
        run_id="immutable-run",
        repo_root=_REPO_ROOT,
    )
    with pytest.raises(FileExistsError):
        archive_transfer_run(
            matrix,
            archive_root=archive_root,
            run_id="immutable-run",
            repo_root=_REPO_ROOT,
        )


def test_archive_transfer_run_requires_resolved_git_commit(tmp_path):
    """Provenance-pinned archival fails before writing when commit lookup fails."""
    matrix = _built_matrix(tmp_path, robustness=-1.0, failed=True)
    archive_root = tmp_path / "archive"

    with pytest.raises(RuntimeError, match="without a resolved git commit"):
        archive_transfer_run(
            matrix,
            archive_root=archive_root,
            run_id="missing-commit",
            repo_root=tmp_path,
        )

    assert not (archive_root / "transfer_matrix" / "missing-commit").exists()
