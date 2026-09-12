"""Focused tests for the optional-capability-handling tutorial (issue #8740)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
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


def _write_cwd_collision_fixtures(root: Path) -> None:
    """Create legacy relative fixtures that must not satisfy isolated probes."""
    fixture_dir = root / "output" / "tutorial_fixtures"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "absent_model.zip").write_bytes(b"not a model")
    (fixture_dir / "absent_annotations.txt").write_text(
        "1 0 0 0 1 1 0 0 0 Pedestrian\n1 1 0 0 1 1 0 0 0 Pedestrian\n",
        encoding="utf-8",
    )


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


def test_formatters_fail_closed_on_unknown_reason_codes() -> None:
    """Programmatic renderers cannot serialize an unrecognized status as valid output."""
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
        _tutorial.format_text(tampered)
    with pytest.raises(ValueError, match="Unknown capability reason codes"):
        _tutorial.format_json(tampered)


def test_synthetic_unavailable_probes_ignore_ambient_collisions(
    tmp_path: Path, monkeypatch
) -> None:
    """Ambient module, registry, and CWD fixtures cannot create synthetic success."""
    import yaml

    from robot_sf.models import registry as model_registry
    from robot_sf.sim import registry as simulator_registry

    ambient_module = tmp_path / f"{_tutorial._MISSING_EXTRA}.py"
    ambient_module.write_text("ambient = True\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, _tutorial._MISSING_EXTRA, types.ModuleType("ambient_fixture"))

    _write_cwd_collision_fixtures(tmp_path)
    ambient_model = tmp_path / "ambient_model.zip"
    ambient_model.write_bytes(b"ambient model")
    ambient_registry = tmp_path / "ambient_registry.yaml"
    ambient_registry.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "models": [
                    {
                        "model_id": _tutorial._MISSING_MODEL_ID,
                        "local_path": str(ambient_model),
                        "local_only": True,
                    },
                    {
                        "model_id": _tutorial._UNKNOWN_MODEL_ID,
                        "local_path": str(ambient_model),
                        "local_only": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_registry, "DEFAULT_REGISTRY_PATH", ambient_registry)
    monkeypatch.setitem(simulator_registry._REGISTRY, _tutorial._MISSING_BACKEND, lambda: None)
    monkeypatch.chdir(tmp_path)

    statuses = {
        status.capability: status
        for status in (
            _tutorial.check_optional_extra(),
            _tutorial.check_model_artifact(),
            _tutorial.check_unknown_model_id(),
            _tutorial.check_dataset_artifact(),
            _tutorial.check_external_runtime(),
        )
    }

    assert {status.reason_code for status in statuses.values()} == EXPECTED_CODES - {
        "core_available"
    }
    assert all(not status.available for status in statuses.values())


def test_synthetic_extra_remedy_is_actionable_without_a_fake_extra() -> None:
    """The synthetic extra message explains the fixture boundary and real next step."""
    remedy = _tutorial.check_optional_extra().remedy
    assert remedy.startswith("Fixture-only diagnostic:")
    assert "no documented extra exists" in remedy
    assert "follow its package documentation" in remedy
    assert remedy.endswith("then retry.")


def test_absent_model_fixture_is_cwd_independent(tmp_path: Path, monkeypatch) -> None:
    """A colliding CWD fixture cannot turn the guaranteed-missing model probe into success."""
    fixture = tmp_path / "output" / "tutorial_fixtures" / "absent_model.zip"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(b"not a model")
    monkeypatch.chdir(tmp_path)
    status = _tutorial.check_model_artifact()
    assert status.reason_code == "model_unavailable"
    assert status.available is False


def test_example_runs_headless_with_stable_output(tmp_path: Path) -> None:
    """The tutorial stays deterministic when launched from a colliding non-repository CWD."""
    _write_cwd_collision_fixtures(tmp_path)
    (tmp_path / f"{_tutorial._MISSING_EXTRA}.py").write_text("ambient = True\n", encoding="utf-8")
    environment = os.environ.copy()
    python_path = [str(tmp_path), str(REPO_ROOT)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)

    first = subprocess.run(
        [sys.executable, str(_MODULE_PATH), "--format", "json"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr[-2000:]
    second = subprocess.run(
        [sys.executable, str(_MODULE_PATH), "--format", "json"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert second.returncode == 0
    assert second.stdout == first.stdout
    assert {entry["reason_code"] for entry in json.loads(first.stdout)} == EXPECTED_CODES


def test_manifest_entry_is_registered(tmp_path: Path, monkeypatch) -> None:
    """The tutorial example must be registered and CI-enabled in the manifest."""
    import yaml

    monkeypatch.chdir(tmp_path)
    manifest_path = REPO_ROOT / "examples" / "examples_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    entries = [
        entry
        for entry in manifest["examples"]
        if entry["path"] == "advanced/36_optional_capability_handling.py"
    ]
    assert len(entries) == 1
    assert entries[0]["ci_enabled"] is True


def test_tutorial_reference_points_to_existing_troubleshooting_section() -> None:
    """The tutorial's documentation reference must resolve to the user guide."""
    source = _MODULE_PATH.read_text(encoding="utf-8")
    assert "docs/user-guide.md#8-troubleshoot" in source
    assert (REPO_ROOT / "docs/user-guide.md").is_file()
