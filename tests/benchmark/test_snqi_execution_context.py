"""Context admission uses recorded data, setup fences and a dev1004 fixture environment."""
# evidence-writer-exempt: only exact-byte producer copies and intentional digest corruption
# use raw writes; each is marked with the shared write_review_sidecar. Other fixtures use write_json.

import json
from copy import deepcopy
from pathlib import Path

import pytest

from robot_sf.benchmark.map_runner import map_runner_episode as episode
from robot_sf.benchmark.result_provenance import build_execution_context_provenance
from robot_sf.evidence.writers import write_json, write_review_sidecar

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/context/evidence/2026-10-04_freeze008_calibration"
ENV = "ROBOT_SF_SNQI_V2_CALIBRATION_CONTEXT"


def test_worker_refuses_context_difference_before_episode_setup(monkeypatch):
    """Base reaches the setup fence; the fixed worker must refuse without setup/reset."""
    context = build_execution_context_provenance()
    context["cpu_model"] = "deliberately different CPU"
    monkeypatch.setenv(ENV, json.dumps(context))
    reached = []

    def setup(**kwargs):
        reached.append("setup")
        raise AssertionError("episode setup reached before execution-context admission")

    monkeypatch.setattr(episode, "_resolve_episode_run_context", setup)
    with pytest.raises(ValueError, match="execution context mismatch: cpu_model"):
        episode.run_map_episode(
            {},
            1004,
            horizon=1,
            dt=0.1,
            record_forces=True,
            snqi_weights=None,
            snqi_baseline=None,
            algo="ppo",
            scenario_path=Path("dev.yaml"),
            policy_builder=lambda *a: pytest.fail("planner construction forbidden"),
        )
    assert reached == []


def reference():
    return json.loads((EVIDENCE / "determinism-receipt.json").read_bytes())["execution_contexts"][
        "original"
    ]


@pytest.mark.parametrize(
    "field",
    [
        "cpu_model",
        "platform",
        "python_version",
        "numpy_version",
        "numba_version",
        "kernel",
        "glibc",
        "thread_env",
        "torch_version",
        "stable_baselines3_version",
    ],
)
def test_same_node_cannot_bypass_a_numerical_context_difference(field):
    from robot_sf.benchmark.snqi.execution_context import assert_context_equal

    expected = reference()
    expected.update(
        kernel="6.8.0-136-generic",
        glibc="2.39",
        torch_version="recorded torch",
        stable_baselines3_version="recorded SB3",
    )
    observed = deepcopy(expected)
    observed[field] = {"OMP_NUM_THREADS": "2"} if field == "thread_env" else "changed"
    with pytest.raises(ValueError, match=field):
        assert_context_equal(observed, expected)


def test_equal_numerical_context_on_another_node_is_admissible():
    from robot_sf.benchmark.snqi.execution_context import assert_context_equal

    expected = reference()
    observed = deepcopy(expected)
    observed["node_identity_sha256"] = "a" * 64
    assert_context_equal(observed, expected)


@pytest.mark.parametrize(
    "field",
    ["cpu_model", "platform", "python_version", "numpy_version", "numba_version", "thread_env"],
)
def test_missing_reference_context_refuses(field):
    from robot_sf.benchmark.snqi.execution_context import assert_context_equal

    expected = reference()
    expected.pop(field)
    with pytest.raises(ValueError, match="missing " + field):
        assert_context_equal(reference(), expected)


@pytest.mark.parametrize("mutation", ["none", "anchors", "receipt", "missing"])
def test_actual_delivered_proof_binds_context_bytes(tmp_path, mutation):
    from robot_sf.benchmark.snqi.execution_context import load_calibration_context

    for name in [
        "anchors.v2.0.acquired.json",
        "acquisition-proof.json",
        "determinism-receipt.json",
    ]:
        (tmp_path / name).write_bytes((EVIDENCE / name).read_bytes())
        write_review_sidecar(tmp_path / name)
    anchors = tmp_path / "anchors.v2.0.acquired.json"
    if mutation == "anchors":
        anchors.write_bytes(anchors.read_bytes() + b" ")
        write_review_sidecar(anchors)
    if mutation == "receipt":
        p = tmp_path / "determinism-receipt.json"
        p.write_bytes(p.read_bytes() + b" ")
        write_review_sidecar(p)
    if mutation == "missing":
        (tmp_path / "acquisition-proof.json").unlink()
    if mutation == "none":
        assert load_calibration_context(anchors) == reference()
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            load_calibration_context(anchors)


def test_guard_restores_environment_and_records_worker_context(monkeypatch):
    from robot_sf.benchmark.snqi.execution_context import (
        admit_episode_context,
        episode_context_guard,
    )

    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(variable, "1")
    live = build_execution_context_provenance()
    monkeypatch.setenv(ENV, "prior")
    with episode_context_guard(live):
        record = admit_episode_context("ppo")
        assert record["cpu_model"] == live["cpu_model"] and "hostname" not in record
        assert admit_episode_context("goal") is None
    assert __import__("os").environ[ENV] == "prior"
    with episode_context_guard(None):
        assert admit_episode_context("ppo") is None


@pytest.mark.parametrize("mutation", ["none", "missing", "different", "empty"])
def test_every_learned_row_is_checked_not_only_batch_context(tmp_path, mutation):
    from robot_sf.benchmark.snqi.execution_context import verify_episode_contexts

    ctx = reference()
    rows = [
        {"algo": "ppo", "seed": 1004, "algorithm_metadata": {"execution_context": deepcopy(ctx)}},
        {
            "algo": "guarded_ppo",
            "seed": 1005,
            "algorithm_metadata": {"execution_context": deepcopy(ctx)},
        },
    ]
    if mutation == "missing":
        rows[1]["algorithm_metadata"].pop("execution_context")
    if mutation == "different":
        rows[1]["algorithm_metadata"]["execution_context"]["numpy_version"] = "changed"
    if mutation == "empty":
        rows = []
    for row in rows:
        path = tmp_path / "runs" / row["algo"] / "episodes.jsonl"
        path.parent.mkdir(parents=True)
        write_json(path, row, indent=None)
    if mutation == "none":
        assert verify_episode_contexts(tmp_path, ctx) == 2
    else:
        with pytest.raises(ValueError, match="execution context"):
            verify_episode_contexts(tmp_path, ctx)


@pytest.mark.parametrize("mode", ["preflight", "run"])
@pytest.mark.parametrize("custody", ["valid", "missing"])
def test_release_cli_refuses_before_source_admission_or_campaign(
    monkeypatch, capsys, mode, custody
):
    """Exercise the actual production CLI, with unchanged paired evidence and no episodes."""
    from types import SimpleNamespace

    from robot_sf.benchmark.snqi import v2_binding
    from scripts.tools import run_benchmark_release as runner

    manifest = SimpleNamespace(canonical_campaign_config_path=Path("dev.yaml"), source_sha="a" * 40)
    cfg = SimpleNamespace(
        snqi_v2_binding={"source_bound": True}, snqi_v2_spec=SimpleNamespace(diagnostic=False)
    )
    monkeypatch.setattr(runner, "load_release_manifest", lambda _path: manifest)
    monkeypatch.setattr(runner, "load_campaign_config", lambda _path: cfg)
    monkeypatch.setattr(v2_binding, "bind_acquired_anchors", lambda cfg, **_kw: cfg)
    observed = reference()
    observed["cpu_model"] = "different CPU despite the same recorded node"
    monkeypatch.setattr(runner, "build_execution_context_provenance", lambda: observed)

    def forbidden(*_args, **_kwargs):
        pytest.fail("context refusal must precede source admission and campaign execution")

    monkeypatch.setattr(runner, "_current_source_commit", forbidden)
    monkeypatch.setattr(runner, "run_campaign", forbidden)
    args = ["--manifest", "dev.yaml", "--mode", mode]
    if custody == "valid":
        args.extend(["--snqi-v2-anchors", str(EVIDENCE / "anchors.v2.0.acquired.json")])
    assert runner.main(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "snqi_v2_execution_context_refused"
    assert payload["campaign_execution_status"] == "not_started"
    assert payload["benchmark_success"] is False
    reason = "mismatch: cpu_model" if custody == "valid" else "requires acquired anchor custody"
    assert reason in payload["status_reason"]


def test_malformed_worker_reference_refuses(monkeypatch):
    from robot_sf.benchmark.snqi.execution_context import admit_episode_context

    monkeypatch.setenv(ENV, "[]")
    with pytest.raises(ValueError, match="must be a mapping"):
        admit_episode_context("ppo")


@pytest.mark.parametrize("context_rows", ["missing", "different", "equal"])
def test_recorded_context_gate_precedes_full_release_acceptance(
    monkeypatch,
    capsys,
    tmp_path: Path,
    context_rows: str | None,
) -> None:
    """Missing/different learned-row context cannot reach release acceptance/publication."""
    from types import SimpleNamespace

    from scripts.tools import run_benchmark_release
    from tests.tools.test_run_benchmark_release import (
        _admit_checkpoint_receipt,
        _default_spawn_matrix_preflight_passes,
        _make_campaign_tree,
        _manifest_fixture,
    )

    _default_spawn_matrix_preflight_passes.__wrapped__(monkeypatch)
    campaign_root = _make_campaign_tree(tmp_path)
    manifest = SimpleNamespace(
        **_manifest_fixture().__dict__,
        schema_version="benchmark-release-manifest.v0.2",
    )
    cfg = SimpleNamespace(export_publication_bundle=True)
    context_args = []
    if context_rows is not None:
        manifest.source_sha = None
        from robot_sf.benchmark.snqi import v2_binding

        evidence = (
            Path(__file__).resolve().parents[2]
            / "docs/context/evidence/2026-10-04_freeze008_calibration"
        )
        context = json.loads((evidence / "determinism-receipt.json").read_bytes())[
            "execution_contexts"
        ]["original"]
        monkeypatch.setattr(
            run_benchmark_release, "_snqi_v2_evaluation_seed_receipt", lambda *_a, **_kw: {}
        )
        cfg.snqi_v2_binding = {"source_bound": True}
        cfg.snqi_v2_spec = SimpleNamespace(diagnostic=False)
        monkeypatch.setattr(v2_binding, "bind_acquired_anchors", lambda cfg, **_kw: cfg)
        monkeypatch.setattr(
            run_benchmark_release, "build_execution_context_provenance", lambda: context
        )
        context_args = ["--snqi-v2-anchors", str(evidence / "anchors.v2.0.acquired.json")]
        if context_rows != "missing":
            recorded = dict(context)
            if context_rows == "different":
                recorded["numpy_version"] = "different"
            path = campaign_root / "runs/ppo/episodes.jsonl"
            path.parent.mkdir(parents=True)
            write_json(
                path,
                {
                    "algo": "ppo",
                    "seed": 1004,
                    "algorithm_metadata": {"execution_context": recorded},
                },
                indent=None,
            )
    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: cfg)
    monkeypatch.setattr(run_benchmark_release, "check_orca_rvo2_preflight", lambda cfg: None)
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_release_manifest",
        lambda *args, **kwargs: {"status": "valid", "problem_count": 0, "problems": []},
    )
    monkeypatch.setattr(
        run_benchmark_release, "build_resolved_release_manifest", lambda *a, **k: {}
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "run_campaign",
        lambda *args, **kwargs: {
            "campaign_root": str(campaign_root),
            "benchmark_success": True,
            "status": "benchmark_success",
            "status_reason": "core rows passed",
            "exit_code": 0,
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "build_release_provenance",
        lambda *args, **kwargs: {
            "benchmark_protocol_version": "0.1.0",
            "release_id": "full",
            "release_tag": manifest.release_tag,
            "manifest_path": "manifest.yaml",
            "manifest_sha256": "a" * 64,
            "canonical_campaign_config": "campaign.yaml",
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_full_benchmark_release_acceptance",
        lambda *args, **kwargs: {
            "status": "invalid",
            "benchmark_success": False,
            "blockers": ["independent scientific sources are not admitted"],
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "_build_publication_payload",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("publication must be blocked by full acceptance")
        ),
    )
    receipt = _admit_checkpoint_receipt(monkeypatch, tmp_path)
    smoke_receipt = tmp_path / "runtime_smoke_result.json"
    write_json(smoke_receipt, {})
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_runtime_smoke_result",
        lambda *args, **kwargs: {
            "schema_version": "benchmark-runtime-smoke-admission.v1",
            "status": "admitted",
        },
    )
    monkeypatch.setattr(run_benchmark_release, "_current_source_commit", lambda: "a" * 40)

    exit_code = run_benchmark_release.main(
        [
            "--manifest",
            "manifest.yaml",
            "--checkpoint-receipt",
            str(receipt),
            "--runtime-smoke-receipt",
            str(smoke_receipt),
            *context_args,
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    if context_rows in {"missing", "different"}:
        assert exit_code == 2
        assert payload["status"] == "snqi_v2_episode_context_refused"
        assert payload["benchmark_success"] is False
        expected_reason = "census empty" if context_rows == "missing" else "mismatch: numpy_version"
        assert expected_reason in payload["status_reason"]
        assert not (campaign_root / "release" / "release_result.json").exists()
        return
    persisted = json.loads(
        (campaign_root / "release" / "release_result.json").read_text(encoding="utf-8")
    )
    assert exit_code == 2
    assert payload["campaign_benchmark_success"] is True
    assert payload["benchmark_success"] is False
    assert payload["release_benchmark_success"] is False
    assert payload["release_status"] == "full_release_acceptance_failed"
    assert payload["release_exit_code"] == 2
    assert payload["publication_bundle"] is None
    assert persisted["release_acceptance"]["blockers"] == [
        "independent scientific sources are not admitted"
    ]


def test_rehashed_caller_receipt_cannot_redefine_calibration_context(tmp_path):
    """Coupled custody hashes are insufficient; only the committed calibration record is used."""
    import hashlib

    from robot_sf.benchmark.snqi.execution_context import load_calibration_context

    for name in [
        "anchors.v2.0.acquired.json",
        "acquisition-proof.json",
        "determinism-receipt.json",
    ]:
        (tmp_path / name).write_bytes((EVIDENCE / name).read_bytes())
        write_review_sidecar(tmp_path / name)
    receipt = json.loads((tmp_path / "determinism-receipt.json").read_bytes())
    receipt["execution_contexts"]["original"]["cpu_model"] = "caller-chosen CPU"
    receipt["execution_contexts"]["repeat"]["cpu_model"] = "caller-chosen CPU"
    write_json(tmp_path / "determinism-receipt.json", receipt)
    proof = json.loads((tmp_path / "acquisition-proof.json").read_bytes())
    proof["determinism_receipt"]["sha256"] = hashlib.sha256(
        (tmp_path / "determinism-receipt.json").read_bytes()
    ).hexdigest()
    write_json(tmp_path / "acquisition-proof.json", proof)
    with pytest.raises(ValueError, match="differs from committed calibration reference"):
        load_calibration_context(tmp_path / "anchors.v2.0.acquired.json")
