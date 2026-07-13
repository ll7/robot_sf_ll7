"""Tests for the issue #5355 factorial campaign-readiness CLI gate.

The library gate (``assess_campaign_readiness``) is covered in
``tests/test_prediction_mpc_factorial.py``; these tests cover the executable
CLI contract: exit codes, report artifact emission, and JSON output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.validation.check_issue_5355_factorial_campaign_readiness import main

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/research/prediction_mpc_factorial_v1.yaml"


def _resolved_config_and_registry(tmp_path: Path) -> tuple[Path, Path]:
    """Copy the real config with dependencies resolved and pin it in a registry."""
    source = CONFIG_PATH.read_text(encoding="utf-8")
    resolved_text = source.replace("status: open", "status: resolved")
    assert "status: open" not in resolved_text
    cfg = tmp_path / "prediction_mpc_factorial_resolved.yaml"
    cfg.write_text(resolved_text, encoding="utf-8")

    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "campaign_id": "issue_5355_prediction_mpc_factorial_v1",
                "config_path": str(cfg),
                "config_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return cfg, registry


class TestReadinessCLI:
    """Exit-code and output contract for the fail-closed gate."""

    def test_default_config_is_resolved_from_script_location(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        code = main([])
        assert code == 1
        out = capsys.readouterr().out
        assert "NOT READY" in out
        assert "#5351" in out

    def test_real_config_exits_nonzero_on_open_dependencies(self, capsys):
        """The landed packet is blocked only on the open #5351 dependency -> exit 1."""
        code = main(["--config", str(CONFIG_PATH)])
        assert code == 1
        out = capsys.readouterr().out
        assert "NOT READY" in out
        assert "#5351" in out

    def test_json_flag_emits_machine_readable_report(self, capsys):
        code = main(["--config", str(CONFIG_PATH), "--json"])
        assert code == 1
        report = json.loads(capsys.readouterr().out)
        assert report["issue"] == 5355
        assert report["ready"] is False
        assert report["criteria"]["preregistration_config_valid"]["ready"] is True
        assert report["criteria"]["dependencies_resolved"]["ready"] is False
        assert any("#5351" in blocker for blocker in report["blockers"])

    def test_out_writes_report_artifact(self, tmp_path, capsys):
        out_path = tmp_path / "nested" / "readiness.json"
        code = main(["--config", str(CONFIG_PATH), "--out", str(out_path)])
        assert code == 1
        assert out_path.is_file()
        written = json.loads(out_path.read_text(encoding="utf-8"))
        assert written["config_path"] == str(CONFIG_PATH)
        assert written["ready"] is False

    def test_ready_config_exits_zero(self, tmp_path, capsys):
        cfg, registry = _resolved_config_and_registry(tmp_path)
        code = main(["--config", str(cfg), "--registry", str(registry)])
        assert code == 0
        out = capsys.readouterr().out
        assert "READY" in out
        assert "NOT READY" not in out

    def test_invalid_config_fails_closed(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("issue: 5355\nschema_version: wrong\n", encoding="utf-8")
        code = main(["--config", str(bad)])
        assert code == 1


class TestResolved5353Reconciliation:
    """Regression coverage for issue #5483: the #5353 capability-matrix
    dependency is resolved and the gate reports #5351 as the sole remaining
    blocker, still failing closed until analysis and the campaign complete.
    """

    def test_5353_no_longer_blocks_the_gate(self, capsys):
        """After reconciliation, the readiness report must NOT cite #5353 as a blocker."""
        code = main(["--config", str(CONFIG_PATH), "--json"])
        assert code == 1
        report = json.loads(capsys.readouterr().out)
        blockers = report["blockers"]
        assert not any("#5353" in blocker for blocker in blockers)
        assert report["criteria"]["dependencies_resolved"]["ready"] is False

    def test_5351_remains_the_only_dependency_blocker(self, capsys):
        """The sole remaining dependency blocker must be #5351 (hierarchical analysis)."""
        code = main(["--config", str(CONFIG_PATH), "--json"])
        assert code == 1
        report = json.loads(capsys.readouterr().out)
        dep_blockers = [b for b in report["blockers"] if b.startswith("dependencies_resolved:")]
        assert len(dep_blockers) == 1
        assert "#5351" in dep_blockers[0]
        assert "#5353" not in dep_blockers[0]

    def test_gate_still_fails_closed_with_5351_open(self, capsys):
        """Resolving #5353 alone is insufficient: the gate stays NOT READY."""
        code = main(["--config", str(CONFIG_PATH)])
        assert code == 1
        out = capsys.readouterr().out
        assert "NOT READY" in out
        assert "#5351" in out

    def test_5353_resolved_status_in_config(self):
        """The tracked factorial config must declare #5353 resolved and #5351 open."""
        import yaml

        payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        statuses = {int(d["issue"]): str(d.get("status")) for d in payload["dependencies"]}
        assert statuses[5353] == "resolved"
        assert statuses[5351] == "open"
