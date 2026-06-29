"""Tests for the #3204 proxy-based checkpoint-selection readiness preflight.

These exercise the fail-closed boundary: missing mapping/proxy metadata and degenerate
artifact states must report ``blocked`` (exit 2), and a fully-resolved fixture must report
``ready`` (exit 0). All inputs are synthetic fixtures under ``tmp_path``; no real checkpoints,
training, or network access are involved.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _REPO_ROOT / "scripts" / "research" / "check_predictive_checkpoint_proxy_readiness.py"
_spec = importlib.util.spec_from_file_location("check_predictive_checkpoint_proxy_readiness", _TOOL)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

_HARD_SEED_FIXTURE = "configs/benchmarks/predictive_hard_seeds_v1.yaml"


def _write_config(
    root: Path,
    *,
    min_resolvable: int = 2,
    min_epochs: int = 4,
    known_blockers: list[dict[str, str]] | None = None,
) -> Path:
    """Write a minimal but valid readiness contract pointing at the real hard-seed fixture."""
    config = {
        "schema_version": "predictive-checkpoint-proxy-readiness.v1",
        "hard_seed_fixture": _HARD_SEED_FIXTURE,
        "checkpoint_selector": {
            "registry_tag": "predictive",
            "min_resolvable_checkpoints": min_resolvable,
        },
        "proxy_summary_contract": {
            "require_enabled": True,
            "min_proxy_epochs": min_epochs,
            "min_success_spread": 1.0e-9,
        },
    }
    if known_blockers is not None:
        config["known_blockers"] = known_blockers
    path = root / "proxy_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _write_registry(root: Path, present_count: int, absent_count: int) -> Path:
    """Write a registry whose predictive checkpoints have present/absent local_paths."""
    models = []
    for i in range(present_count):
        ckpt = root / f"present_{i}.pt"
        ckpt.write_text("x", encoding="utf-8")
        models.append(
            {"model_id": f"predictive_present_{i}", "local_path": str(ckpt), "tags": ["predictive"]}
        )
    for i in range(absent_count):
        models.append(
            {
                "model_id": f"predictive_absent_{i}",
                "local_path": str(root / "missing" / f"absent_{i}.pt"),
                "tags": ["predictive"],
            }
        )
    # A non-predictive entry that must be ignored by the selector.
    models.append({"model_id": "ppo_other", "local_path": str(root), "tags": ["ppo"]})
    path = root / "registry.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "models": models}), encoding="utf-8")
    return path


def _hist(pairs):
    return [
        {"epoch": float(i + 1), "val_ade": a, "success_rate": s} for i, (a, s) in enumerate(pairs)
    ]


def _write_summary(root: Path, *, enabled: bool, pairs) -> Path:
    """Write a training summary JSON in the analyzer's proxy.history schema."""
    summary = {"model_id": "m", "proxy": {"enabled": enabled, "history": _hist(pairs)}}
    path = root / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def test_blocked_when_config_missing(tmp_path):
    """A missing readiness config fails closed."""
    registry = _write_registry(tmp_path, present_count=6, absent_count=0)
    report = mod.check_readiness(
        config_path=tmp_path / "does_not_exist.yaml",
        registry_path=registry,
        repo_root=_REPO_ROOT,
    )
    assert report["status"] == "blocked"
    assert report["prerequisites"]["readiness_config"]["status"] == "blocked"


def test_failed_when_config_not_a_mapping(tmp_path):
    """A config YAML whose top-level document is not a mapping reports a clean failed status."""
    registry = _write_registry(tmp_path, present_count=6, absent_count=0)
    bad_config = tmp_path / "proxy_config.yaml"
    bad_config.write_text(yaml.safe_dump([1, 2, 3]), encoding="utf-8")
    report = mod.check_readiness(
        config_path=bad_config, registry_path=registry, repo_root=_REPO_ROOT
    )
    assert report["status"] == "blocked"
    cfg_check = report["prerequisites"]["readiness_config"]
    assert cfg_check["status"] == "failed"
    assert any("must be a mapping" in m for m in cfg_check["messages"])


def test_blocked_when_checkpoint_local_path_is_directory(tmp_path):
    """A predictive local_path pointing at a directory must not count as a resolvable checkpoint."""
    config = _write_config(tmp_path, min_resolvable=1)
    ckpt_dir = tmp_path / "predictive_dir"
    ckpt_dir.mkdir()
    models = [{"model_id": "predictive_dir_0", "local_path": str(ckpt_dir), "tags": ["predictive"]}]
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump({"version": 1, "models": models}), encoding="utf-8")
    report = mod.check_readiness(config_path=config, registry_path=registry, repo_root=_REPO_ROOT)
    ckpt = report["prerequisites"]["checkpoint_artifacts"]
    assert ckpt["status"] == "blocked"
    assert ckpt["mapping"]["resolvable_count"] == 0
    assert ckpt["mapping"]["blocked_by_status"] == {"not_checkpoint_file": 1}
    assert ckpt["mapping"]["candidates"][0]["status"] == "not_checkpoint_file"
    assert ckpt["mapping"]["candidates"][0]["reason"] == "local_path resolves to a directory"
    assert report["status"] == "blocked"


def test_blocked_mapping_classifies_invalid_local_path(tmp_path):
    """A predictive registry entry without local_path is an explicit blocked mapping row."""
    config = _write_config(tmp_path, min_resolvable=1)
    models = [{"model_id": "predictive_missing_path", "tags": ["predictive"]}]
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump({"version": 1, "models": models}), encoding="utf-8")

    report = mod.check_readiness(config_path=config, registry_path=registry, repo_root=_REPO_ROOT)

    ckpt = report["prerequisites"]["checkpoint_artifacts"]
    assert report["status"] == "blocked"
    assert ckpt["mapping"]["blocked_by_status"] == {"invalid_local_path": 1}
    assert ckpt["mapping"]["candidates"][0]["status"] == "invalid_local_path"
    assert ckpt["mapping"]["candidates"][0]["artifact_scope"] == "invalid"


def test_blocked_mapping_classifies_worktree_local_output_artifact(tmp_path):
    """A missing output/ checkpoint is reported as worktree-local, not durable evidence."""
    config = _write_config(tmp_path, min_resolvable=1)
    models = [
        {
            "model_id": "predictive_output_tmp",
            "local_path": "output/tmp/predictive/run/checkpoint.pt",
            "tags": ["predictive"],
        }
    ]
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump({"version": 1, "models": models}), encoding="utf-8")

    report = mod.check_readiness(config_path=config, registry_path=registry, repo_root=_REPO_ROOT)

    ckpt = report["prerequisites"]["checkpoint_artifacts"]
    candidate = ckpt["mapping"]["candidates"][0]
    assert report["status"] == "blocked"
    assert ckpt["mapping"]["blocked_by_status"] == {"missing_local_path": 1}
    assert candidate["status"] == "missing_local_path"
    assert candidate["artifact_scope"] == "worktree_local_output"
    assert candidate["resolved_path"].endswith("output/tmp/predictive/run/checkpoint.pt")


def test_blocked_when_too_few_checkpoints_resolve(tmp_path):
    """Fewer locally-resolvable checkpoints than the minimum fails closed (mapping metadata)."""
    config = _write_config(tmp_path, min_resolvable=6)
    registry = _write_registry(tmp_path, present_count=2, absent_count=4)
    report = mod.check_readiness(config_path=config, registry_path=registry, repo_root=_REPO_ROOT)
    assert report["status"] == "blocked"
    ckpt = report["prerequisites"]["checkpoint_artifacts"]
    assert ckpt["status"] == "blocked"
    mapping = ckpt["mapping"]
    assert mapping["candidate_count"] == 6  # the ppo entry is excluded
    assert mapping["resolvable_count"] == 2
    assert mapping["blocked_by_status"] == {"missing_local_path": 4}
    # Absent checkpoints are surfaced by model_id for actionable triage.
    assert any("predictive_absent_0" in m for m in ckpt["messages"])


def test_blocked_when_proxy_summary_degenerate(tmp_path):
    """A summary with no hard-success spread (the all-zero probe) fails closed."""
    config = _write_config(tmp_path, min_resolvable=2, min_epochs=4)
    registry = _write_registry(tmp_path, present_count=2, absent_count=0)
    summary = _write_summary(
        tmp_path, enabled=True, pairs=[(1.0, 0.0), (0.9, 0.0), (0.8, 0.0), (0.7, 0.0)]
    )
    report = mod.check_readiness(
        config_path=config,
        registry_path=registry,
        repo_root=_REPO_ROOT,
        training_summary=summary,
    )
    assert report["status"] == "blocked"
    summary_check = report["prerequisites"]["proxy_training_summary"]
    assert summary_check["status"] == "blocked"
    assert summary_check["summary"]["verdict"] == "inconclusive"


def test_blocked_when_proxy_disabled(tmp_path):
    """A summary with proxy.enabled=false fails closed even with spread."""
    config = _write_config(tmp_path, min_resolvable=2, min_epochs=2)
    registry = _write_registry(tmp_path, present_count=2, absent_count=0)
    summary = _write_summary(tmp_path, enabled=False, pairs=[(1.0, 0.1), (0.8, 0.4)])
    report = mod.check_readiness(
        config_path=config,
        registry_path=registry,
        repo_root=_REPO_ROOT,
        training_summary=summary,
    )
    assert report["status"] == "blocked"
    assert any(
        "proxy.enabled" in m for m in report["prerequisites"]["proxy_training_summary"]["messages"]
    )


def test_blocked_when_summary_not_a_mapping(tmp_path):
    """A training summary whose JSON is not an object fails closed without crashing."""
    config = _write_config(tmp_path, min_resolvable=2, min_epochs=2)
    registry = _write_registry(tmp_path, present_count=2, absent_count=0)
    bad_summary = tmp_path / "summary.json"
    bad_summary.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    report = mod.check_readiness(
        config_path=config,
        registry_path=registry,
        repo_root=_REPO_ROOT,
        training_summary=bad_summary,
    )
    assert report["status"] == "blocked"
    summary_check = report["prerequisites"]["proxy_training_summary"]
    assert summary_check["status"] == "failed"
    assert any("JSON object" in m for m in summary_check["messages"])


def test_blocked_when_known_blocker_configured(tmp_path):
    """Configured known blockers are surfaced and keep preflight fail-closed."""
    config = _write_config(
        tmp_path,
        min_resolvable=2,
        min_epochs=2,
        known_blockers=[
            {
                "id": "degenerate_hardcase_proxy_probe_v1",
                "status": "blocked",
                "revival_condition": "provide a non-degenerate proxy summary",
            }
        ],
    )
    registry = _write_registry(tmp_path, present_count=2, absent_count=0)
    summary = _write_summary(tmp_path, enabled=True, pairs=[(1.0, 0.1), (0.8, 0.4)])
    report = mod.check_readiness(
        config_path=config,
        registry_path=registry,
        repo_root=_REPO_ROOT,
        training_summary=summary,
    )
    blockers = report["prerequisites"]["known_blockers"]
    assert report["status"] == "blocked"
    assert blockers["status"] == "blocked"
    assert blockers["blockers"][0]["id"] == "degenerate_hardcase_proxy_probe_v1"


def test_ready_when_known_blocker_resolved(tmp_path):
    """Resolved known blockers stay in the map without blocking readiness."""
    config = _write_config(
        tmp_path,
        min_resolvable=2,
        min_epochs=2,
        known_blockers=[
            {
                "id": "degenerate_hardcase_proxy_probe_v1",
                "status": "resolved",
                "revival_condition": "non-degenerate proxy summary supplied",
            }
        ],
    )
    registry = _write_registry(tmp_path, present_count=2, absent_count=0)
    summary = _write_summary(tmp_path, enabled=True, pairs=[(1.0, 0.1), (0.8, 0.4)])
    report = mod.check_readiness(
        config_path=config,
        registry_path=registry,
        repo_root=_REPO_ROOT,
        training_summary=summary,
    )
    blockers = report["prerequisites"]["known_blockers"]
    assert report["status"] == "ready"
    assert blockers["status"] == "passed"
    assert blockers["blockers"][0]["status"] == "resolved"


def test_ready_when_inputs_present_and_summary_has_spread(tmp_path):
    """All inputs resolve and the summary has non-degenerate spread -> ready (exit 0)."""
    config = _write_config(tmp_path, min_resolvable=6, min_epochs=4)
    registry = _write_registry(tmp_path, present_count=6, absent_count=0)
    summary = _write_summary(
        tmp_path, enabled=True, pairs=[(1.0, 0.1), (0.9, 0.3), (0.8, 0.1), (0.7, 0.2)]
    )
    report = mod.check_readiness(
        config_path=config,
        registry_path=registry,
        repo_root=_REPO_ROOT,
        training_summary=summary,
    )
    assert report["status"] == "ready", report["errors"]
    assert all(p["status"] == "passed" for p in report["prerequisites"].values())


def test_main_exit_codes(tmp_path, capsys):
    """The CLI returns exit 2 when blocked and 0 when ready, in JSON mode."""
    config = _write_config(tmp_path, min_resolvable=6)
    blocked_registry = _write_registry(tmp_path, present_count=1, absent_count=5)
    code = mod.main(
        [
            "--config",
            str(config),
            "--registry",
            str(blocked_registry),
            "--repo-root",
            str(_REPO_ROOT),
            "--json",
        ]
    )
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"

    ready_registry = _write_registry(tmp_path, present_count=6, absent_count=0)
    code = mod.main(
        [
            "--config",
            str(config),
            "--registry",
            str(ready_registry),
            "--repo-root",
            str(_REPO_ROOT),
            "--json",
        ]
    )
    assert code == 0


def test_real_registry_predictive_checkpoints_blocked():
    """Against the live registry, predictive checkpoints do not resolve -> blocked (the #3204 state).

    This pins the diagnostic recorded on the issue: every predictive local_path points at a
    non-durable output/tmp path that is absent in a clean checkout. If checkpoints are later
    hydrated this test will flip, which is the intended revival signal.
    """
    config_path = _REPO_ROOT / "configs" / "research" / "predictive_checkpoint_proxy_v1.yaml"
    registry_path = _REPO_ROOT / "model" / "registry.yaml"
    report = mod.check_readiness(
        config_path=config_path, registry_path=registry_path, repo_root=_REPO_ROOT
    )
    ckpt = report["prerequisites"]["checkpoint_artifacts"]
    assert ckpt["mapping"]["candidate_count"] >= 6
    # Documented current state: insufficient predictive checkpoints resolve locally.
    assert ckpt["status"] == "blocked"
    assert report["status"] == "blocked"
