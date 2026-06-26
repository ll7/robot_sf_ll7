"""Issue #3654 vulnerable-user proxy scenario-pack scaffold tests."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from robot_sf.benchmark.cli import cli_main
from robot_sf.training.scenario_loader import load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_PATH = (
    REPO_ROOT / "configs/scenarios/sets/vulnerable_user_proxy_pack_v0_deferred_issue3654.yaml"
)


def test_vulnerable_user_proxy_pack_loads_as_deferred_manifest(capsys) -> None:
    """The pack is structurally valid but explicitly deferred from benchmark use."""
    rc = cli_main(["validate-config", "--matrix", str(PACK_PATH)])
    report = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert report["num_scenarios"] == 3
    assert report["source"]["schema_version"] == "robot_sf.scenario_matrix.v1"

    scenarios = load_scenarios(PACK_PATH)
    assert [scenario["name"] for scenario in scenarios] == [
        "vulnerable_user_proxy_slow_crossing_low",
        "vulnerable_user_proxy_waiting_dwell_medium",
        "vulnerable_user_proxy_occluded_emergence_low",
    ]

    for scenario in scenarios:
        metadata = scenario["metadata"]
        assert metadata["pack_id"] == "issue_3654_vulnerable_user_proxy_v0"
        assert metadata["status"] == "deferred"
        assert metadata["enabled_by_default"] is False
        assert metadata["benchmark_evidence"] is False
        assert "not a real-world user-group claim" in metadata["claim_boundary"]
        assert "opt-in benchmark config" in metadata["requires_before_benchmark_use"]


def test_vulnerable_user_proxy_pack_is_not_referenced_by_default_benchmarks() -> None:
    """Default benchmark configs must not pick up the deferred pack implicitly."""
    pack_name = PACK_PATH.name
    benchmark_configs = sorted((REPO_ROOT / "configs/benchmarks").rglob("*.yaml"))

    referencing_configs = []
    for path in benchmark_configs:
        text = path.read_text(encoding="utf-8")
        if pack_name in text or "issue_3654_vulnerable_user_proxy_v0" in text:
            referencing_configs.append(path)

    assert referencing_configs == []


def test_vulnerable_user_proxy_pack_avoids_accessibility_claim_text() -> None:
    """The deferred scaffold must not make accessibility or paper-facing claims."""
    payload = yaml.safe_load(PACK_PATH.read_text(encoding="utf-8"))
    serialized = yaml.safe_dump(payload, sort_keys=True).lower()

    forbidden_terms = [
        "accessibility",
        "disabled user",
        "paper claim",
        "dissertation claim",
        "benchmark evidence true",
    ]
    assert not any(term in serialized for term in forbidden_terms)
