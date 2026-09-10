"""Focused tests for the optional-capability-handling tutorial (issue #8740)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = REPO_ROOT / "examples/advanced/36_optional_capability_handling.py"


def _load_tutorial_module():
    """Load the tutorial example by file path (numeric filenames are not importable)."""
    spec = importlib.util.spec_from_file_location("tutorial_capability", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["tutorial_capability"] = module
    spec.loader.exec_module(module)
    return module


_tutorial = _load_tutorial_module()

EXPECTED_CODES = {
    "core_available",
    "extra_missing",
    "model_unavailable",
    "model_unknown",
    "dataset_unavailable",
    "runtime_unsupported",
}


def test_all_probes_report_stable_reason_codes() -> None:
    """Every probe reports a known code; unavailable states never read as success."""
    statuses = _tutorial.collect_statuses()
    assert {status.reason_code for status in statuses} == EXPECTED_CODES
    unavailable = [status for status in statuses if not status.available]
    assert len(unavailable) == 5
    assert all(status.remedy for status in unavailable)


def test_text_and_json_formats_agree_on_reason_codes() -> None:
    """Friendly and JSON output preserve the same reason codes."""
    statuses = _tutorial.collect_statuses()
    text = _tutorial.format_text(statuses)
    payload = json.loads(_tutorial.format_json(statuses))
    assert [entry["reason_code"] for entry in payload] == [
        status.reason_code for status in statuses
    ]
    for status in statuses:
        assert status.reason_code in text


def test_unknown_reason_codes_fail_closed() -> None:
    """An unrecognized reason code raises instead of silently passing."""
    statuses = _tutorial.collect_statuses()
    tampered = list(statuses) + [
        type(statuses[0])(
            capability="mystery",
            available=False,
            reason_code="something_new",
            detail="x",
            remedy="y",
        )
    ]
    with pytest.raises(ValueError, match="Unknown capability reason codes"):
        _tutorial.fail_on_unknown_status(tampered)


def test_example_runs_headless_with_stable_output() -> None:
    """The tutorial executes as a subprocess with deterministic reason codes."""
    first = subprocess.run(
        [sys.executable, str(_MODULE_PATH), "--format", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr[-2000:]
    second = subprocess.run(
        [sys.executable, str(_MODULE_PATH), "--format", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert second.returncode == 0
    assert second.stdout == first.stdout
    assert {entry["reason_code"] for entry in json.loads(first.stdout)} == EXPECTED_CODES


def test_manifest_entry_is_registered() -> None:
    """The tutorial example must be registered and CI-enabled in the manifest."""
    import yaml

    manifest = yaml.safe_load(Path("examples/examples_manifest.yaml").read_text())
    entries = [
        entry
        for entry in manifest["examples"]
        if entry["path"] == "advanced/36_optional_capability_handling.py"
    ]
    assert len(entries) == 1
    assert entries[0]["ci_enabled"] is True
