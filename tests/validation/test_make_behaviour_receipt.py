"""Producer coverage for behaviour-receipt payloads and headers."""

import json
import subprocess
from pathlib import Path

import pytest

from scripts.ci import behaviour_receipt as gate
from scripts.ci import make_behaviour_receipt as producer


def _git_repo(root: Path):
    root.mkdir(exist_ok=True)

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    git("init", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    return git


def _sweep_rows(root: Path, label: str, *, failing: bool) -> Path:
    out = root / label
    out.mkdir()
    rows = []
    for seed in producer.DEFAULT_SEEDS:
        rows.append(
            {
                "arm": "goal",
                "scenario": "open",
                "seed": seed,
                "success": not (failing and seed == producer.DEV_SEED_MIN),
                "collisions": 0,
                "execution_status": "written",
            }
        )
    (out / "episodes_main.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    (out / "README.md").write_text(f"{label}\n", encoding="utf-8")
    return out


def _scope() -> dict:
    return {
        "arms": ["goal"],
        "maps": ["open"],
        "vehicle_id": "t60",
        "exceptions": [],
        "arm_algorithms": {"goal": "goal"},
    }


def test_latest_release_selects_published_0_0_8_from_release_name(monkeypatch):
    """The 0.0.8 publication is not tagged as plain 0.0.8, so inspect names too."""
    releases = [
        [
            {
                "draft": False,
                "prerelease": False,
                "tag_name": "paper-matrix-v2-h600-s30-2026-10-373dbfde4f39667cf9e8732dabe7df5118bdeab1",
                "name": "Robot SF benchmark data 0.0.8 - main only",
                "published_at": "2026-10-09T17:39:53Z",
            },
            {
                "draft": False,
                "prerelease": False,
                "tag_name": "0.0.7",
                "name": "Robot SF 0.0.7",
                "published_at": "2026-09-17T16:20:07Z",
            },
        ]
    ]
    monkeypatch.setattr(
        gate.subprocess,
        "check_output",
        lambda *args, **kwargs: json.dumps(releases),
    )
    assert (
        gate.latest_release("ll7/robot_sf_ll7")
        == "paper-matrix-v2-h600-s30-2026-10-373dbfde4f39667cf9e8732dabe7df5118bdeab1"
    )


def test_synthetic_receipt_produced_by_tool_validates_with_real_gate(monkeypatch, tmp_path):
    """Produce committed rows, print a compact header, then admit it through check_receipt."""
    git = _git_repo(tmp_path)
    git("commit", "--allow-empty", "-m", "execution source")
    run_source = git("rev-parse", "HEAD")
    scope_path = tmp_path / "scope.json"
    scope = _scope()
    scope_path.write_text(json.dumps(scope), encoding="utf-8")
    head_sweep = _sweep_rows(tmp_path, "head-sweep", failing=True)
    baseline_sweep = _sweep_rows(tmp_path, "baseline-sweep", failing=False)
    scheduler = producer._existing_sweep(
        run_source,
        head_sweep,
        artifact_uri="https://example.org/head-sweep",
        artifact_sha256=None,
        job_id="123",
    )
    baseline = producer._existing_sweep(
        "c" * 40,
        baseline_sweep,
        artifact_uri="https://example.org/baseline-sweep",
        artifact_sha256=None,
        job_id="456",
    )
    rows, classifications, totals = producer.build_rows_and_classifications(
        producer._load_sweep_rows(head_sweep),
        producer._load_sweep_rows(baseline_sweep),
        scope,
        classification_class="known_limitation",
        evidence_base_uri="https://example.org/classifications",
    )
    audit_path = tmp_path / "audit.json"
    audit_sha = producer.write_real_row_audit(
        audit_path,
        source_sha=run_source,
        rows=rows,
        classifications=classifications,
    )
    producer.write_receipt_and_header(
        repo_root=tmp_path,
        receipt_id="synthetic",
        head_sha=run_source,
        scheduler=scheduler,
        baseline=baseline,
        baseline_release="0.0.8",
        baseline_body_id="t60",
        baseline_config_sha256="d" * 64,
        baseline_differences=["synthetic behaviour-change fixture"],
        scope=scope,
        rows=rows,
        classifications=classifications,
        totals=totals,
        audit_uri="https://example.org/audit.json",
        audit_sha256=audit_sha,
        audit_source_sha=run_source,
        refute_review_uri="https://example.org/refute-review",
        output_dir=tmp_path / "receipts/behaviour",
    )
    git("add", "receipts/behaviour/synthetic.json")
    git("commit", "-m", "receipt payload")
    final_head = git("rev-parse", "HEAD")
    _, header = producer.write_receipt_and_header(
        repo_root=tmp_path,
        receipt_id="synthetic",
        head_sha=final_head,
        scheduler=scheduler,
        baseline=baseline,
        baseline_release="0.0.8",
        baseline_body_id="t60",
        baseline_config_sha256="d" * 64,
        baseline_differences=["synthetic behaviour-change fixture"],
        scope=scope,
        rows=rows,
        classifications=classifications,
        totals=totals,
        audit_uri="https://example.org/audit.json",
        audit_sha256=audit_sha,
        audit_source_sha=run_source,
        refute_review_uri="https://example.org/refute-review",
        output_dir=tmp_path / "receipts/behaviour",
    )
    assert git("status", "--short", "--", "receipts/behaviour/synthetic.json") == ""
    body = "<!-- behaviour-change-receipt:v2\n" + json.dumps(header) + "\n-->"
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "SCOPE_PATH", scope_path)
    monkeypatch.setattr(gate, "latest_release", lambda repo: "0.0.8")
    monkeypatch.setattr(gate, "release_source", lambda release: "c" * 40)
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", final_head)
    assert gate.check_receipt(body, ["robot_sf/planner/guarded_ppo.py"], "ll7/robot_sf_ll7") == []


def test_seed_guard_names_wider_dev_range_but_keeps_current_gate_contract():
    assert producer._check_seed_range([1001, 1030]) == [1001, 1030]
    with pytest.raises(ValueError, match="current behaviour gate admits only seeds"):
        producer._check_seed_range([1031])
    with pytest.raises(ValueError, match="outside development range"):
        producer._check_seed_range([1201])
