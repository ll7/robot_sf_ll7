"""Focused tests for the retained failed-campaign diagnosis example (issue #8902).

The tests protect the example's smoke/diagnostic contract: canonical validators are
called in order, partial rows stay diagnostic-only, and ambiguous evidence never turns
into retry or scientific-success advice.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOSSIER = ROOT / "examples" / "fixtures" / "failed_campaign_dossier.json"
DRIVER = ROOT / "examples" / "advanced" / "39_failed_campaign_diagnosis.py"


def _driver():
    spec = importlib.util.spec_from_file_location("failed_campaign_diagnosis", DRIVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = _driver()
CASES = json.loads(DOSSIER.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", sorted(CASES))
def test_named_dossier_cases_are_fail_closed(case: str) -> None:
    """Each retained-file fixture maps to its declared local diagnostic lane."""

    report = driver.diagnose_dossier(DOSSIER, case=case)

    assert report["schema"] == "failed_campaign_diagnosis.v1"
    assert report["diagnostic_outcome"] == CASES[case]["expected_outcome"]
    assert report["rerun"]["authorized"] is False
    assert all(item["promotable"] is False for item in report["retained_useful_artifacts"])


def test_base_report_preserves_causal_order_and_partial_rows() -> None:
    """The default dossier exposes execution first while retaining one useful row."""

    report = driver.diagnose_dossier(DOSSIER)
    gates = report["observed_facts"]

    assert [gate["name"] for gate in gates] == [
        "identity",
        "startup_environment",
        "execution",
        "row_accounting",
        "producer_finalizer",
        "artifact_integrity",
        "preservation",
    ]
    assert report["first_decisive_failure"]["name"] == "execution"
    assert report["row_accounting"] == {
        "expected_rows": 2,
        "observed_rows": 1,
        "retained_diagnostic_rows": 1,
        "row_statuses": ["native"],
    }
    assert report["retained_useful_artifacts"][0]["status"] == "diagnostic_only"
    assert report["observed_facts"][5]["status"] == "pass"
    assert report["observed_facts"][6]["findings"] == ["non_durable_custody"]


def test_repeated_diagnosis_is_byte_stable() -> None:
    """Temporary materialization and validator reports do not add timestamps or paths."""

    first = driver.diagnose_dossier(DOSSIER)
    second = driver.diagnose_dossier(DOSSIER)
    assert first == second
    assert str(ROOT) not in json.dumps(first)


def test_cli_emits_json_without_submitting_or_writing_to_fixture(capsys) -> None:
    """The example command returns diagnostic success and leaves the dossier unchanged."""

    before = DOSSIER.read_bytes()
    assert driver.main(["--dossier", str(DOSSIER), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["diagnostic_outcome"] == "retry_requires_differential"
    assert DOSSIER.read_bytes() == before
