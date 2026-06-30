"""Tests issue #3808 TTC near-miss diagnostic decision packet renderer."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/validation/render_ttc_near_miss_decision_packet.py"

_SPEC = importlib.util.spec_from_file_location("_issue_3808_ttc_packet", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_all_fixtures_cover_issue_3808_decision_cases() -> None:
    """All required fixtures render deterministic packet statuses."""
    packets = _MODULE.build_packets()

    assert set(packets) == {
        "closing",
        "opening",
        "missing-timing",
        "unsupported-trajectory",
    }
    assert packets["closing"].diagnostic_status == "ok"
    assert packets["opening"].diagnostic_status == "no-approaching-pairs"
    assert packets["missing-timing"].diagnostic_status == "unsupported-inputs"
    assert packets["unsupported-trajectory"].diagnostic_status == "unsupported-inputs"


def test_json_render_is_strict_and_keeps_claim_boundary() -> None:
    """JSON output is reviewable and cannot contain NaN metric sentinels."""
    packets = _MODULE.build_packets()
    payload = json.loads(_MODULE.render_packets_json(packets))

    assert payload["issue"] == 3808
    assert "no canonical metric replacement" in payload["claim_boundary"]
    assert payload["fixtures"]["opening"]["diagnostic"]["near_miss_ttc__min_ttc_s"] is None
    assert payload["fixtures"]["missing-timing"]["diagnostic"] == {}


def test_markdown_render_names_fixtures_and_non_claims() -> None:
    """Markdown dry-run packet includes fixture labels and claim caveats."""
    text = _MODULE.render_packets_markdown(_MODULE.build_packets())

    assert "Fixture: `closing`" in text
    assert "Fixture: `unsupported-trajectory`" in text
    assert "Cannot Claim Before Canonical Metric Change" in text
    assert "benchmark ranking" in text


def test_cli_json_single_fixture() -> None:
    """CLI dry-run renders a selected fixture as strict JSON."""
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fixture",
            "opening",
            "--format",
            "json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert set(payload["fixtures"]) == {"opening"}
    assert payload["fixtures"]["opening"]["diagnostic_status"] == "no-approaching-pairs"


def test_cli_json_output_path_writes_packet(tmp_path: Path) -> None:
    """CLI can materialize a deterministic packet for review handoff."""
    output_path = tmp_path / "nested" / "ttc_packet.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fixture",
            "missing-timing",
            "--format",
            "json",
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(payload["fixtures"]) == {"missing-timing"}
    assert payload["fixtures"]["missing-timing"]["diagnostic_status"] == "unsupported-inputs"
