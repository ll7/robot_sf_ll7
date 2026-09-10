"""Tests for check-only example prerequisite validation (issue #8735).

The tests cover the stable statuses required by the issue - ``ready``,
``missing_model``, ``missing_map``, ``missing_extra``, ``invalid_checksum``,
``unsupported_legacy_reference`` - plus the acceptance criteria that the
examples CLI and direct script invocation report the same status, that the
check path performs no network access and imports no training framework, and
that the JSON payload is deterministic.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from robot_sf.examples import prerequisites as prerequisites_module
from robot_sf.examples.manifest_loader import load_manifest
from robot_sf.examples.prerequisites import (
    CHECK_SCHEMA,
    STATUS_INVALID_CHECKSUM,
    STATUS_MISSING_EXTRA,
    STATUS_MISSING_MAP,
    STATUS_MISSING_MODEL,
    STATUS_OPERATOR_INPUT_REQUIRED,
    STATUS_READY,
    STATUS_UNSUPPORTED_LEGACY_REFERENCE,
    check_example_prerequisites,
    check_script_prerequisites,
    report_to_dict,
)
from robot_sf.examples_cli import examples_cli_main

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL_MANIFEST = load_manifest(validate_paths=True)

_SCRIPTS = (
    "advanced/06_pedestrian_env_factory.py",
    "advanced/09_defensive_policy.py",
    "advanced/10_offensive_policy.py",
    "advanced/11_ego_pedestrian_policy.py",
    "advanced/32_demo_adversarial_pedestrian.py",
)


def _sha256_bytes(data: bytes) -> str:
    """Return the SHA256 hex digest of a byte payload."""

    return hashlib.sha256(data).hexdigest()


def _write_repo(
    tmp_path: Path,
    examples: list[dict[str, object]],
    *,
    files: dict[str, bytes] | None = None,
    registry: list[dict[str, object]] | None = None,
) -> Path:
    """Create a fixture repository and return its manifest path."""

    examples_dir = tmp_path / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "fixture",
        "categories": [
            {
                "slug": "advanced",
                "title": "Advanced",
                "description": "Fixture category.",
                "order": 0,
            }
        ],
        "examples": examples,
    }
    for example in examples:
        script = examples_dir / str(example["path"])
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text('"""Fixture example."""\n', encoding="utf-8")
    manifest_path = examples_dir / "examples_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    for relative, content in (files or {}).items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    if registry is not None:
        registry_path = tmp_path / "model" / "registry.yaml"
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(yaml.safe_dump({"models": registry}, sort_keys=False))
    return manifest_path


def _example(path: str, prerequisites: list[str]) -> dict[str, object]:
    """Build one fixture example entry."""

    return {
        "path": path,
        "name": path,
        "summary": "Fixture example.",
        "category_slug": "advanced",
        "prerequisites": prerequisites,
        "ci_enabled": False,
        "ci_reason": "fixture",
    }


def _missing_ok_checks(report, prerequisite: str):
    """Return the single check for a prerequisite."""

    matches = [check for check in report.checks if check.prerequisite == prerequisite]
    assert len(matches) == 1, matches
    return matches[0]


def test_complete_prerequisites_report_ready_with_registry_provenance(tmp_path: Path) -> None:
    """A present map plus a checksum-verified registered model is ready."""

    model_bytes = b"fixture-model-bytes"
    manifest_path = _write_repo(
        tmp_path,
        [_example("advanced/ready.py", ["maps/ok.svg", "model/registered.zip"])],
        files={"maps/ok.svg": b"<svg/>", "model/registered.zip": model_bytes},
        registry=[
            {
                "model_id": "fixture_registered",
                "local_path": "model/registered.zip",
                "github_release": {
                    "url": "https://example.invalid/registered.zip",
                    "sha256": _sha256_bytes(model_bytes),
                },
            }
        ],
    )
    manifest = load_manifest(manifest_path, validate_paths=True)

    report = check_example_prerequisites(manifest, "advanced/ready")

    assert report.status == STATUS_READY
    assert report.ready is True
    model_check = _missing_ok_checks(report, "model/registered.zip")
    assert model_check.status == STATUS_READY
    assert model_check.model_registry is not None
    assert model_check.model_registry["registered"] is True
    assert model_check.observed_sha256 == _sha256_bytes(model_bytes)


def test_missing_model_and_map_report_distinct_statuses(tmp_path: Path) -> None:
    """Missing models and maps keep their explicit non-ready statuses."""

    manifest_path = _write_repo(
        tmp_path,
        [_example("advanced/missing.py", ["model/absent.zip", "maps/absent.svg"])],
    )
    manifest = load_manifest(manifest_path, validate_paths=True)

    report = check_example_prerequisites(manifest, "advanced/missing")

    assert report.status == STATUS_MISSING_MODEL
    assert _missing_ok_checks(report, "model/absent.zip").status == STATUS_MISSING_MODEL
    assert _missing_ok_checks(report, "maps/absent.svg").status == STATUS_MISSING_MAP


def test_checksum_mismatch_is_reported(tmp_path: Path) -> None:
    """A present model with the wrong bytes never reports ready."""

    manifest_path = _write_repo(
        tmp_path,
        [_example("advanced/checksum.py", ["model/registered.zip"])],
        files={"model/registered.zip": b"tampered"},
        registry=[
            {
                "model_id": "fixture_registered",
                "local_path": "model/registered.zip",
                "github_release": {
                    "url": "https://example.invalid/registered.zip",
                    "sha256": _sha256_bytes(b"expected"),
                },
            }
        ],
    )
    manifest = load_manifest(manifest_path, validate_paths=True)

    report = check_example_prerequisites(manifest, "advanced/checksum")

    assert report.status == STATUS_INVALID_CHECKSUM
    check = _missing_ok_checks(report, "model/registered.zip")
    assert check.expected_sha256 == _sha256_bytes(b"expected")
    assert check.observed_sha256 == _sha256_bytes(b"tampered")


def test_legacy_reference_reports_registry_replacement(tmp_path: Path) -> None:
    """A pre-registry in-tree path reports the registry-backed replacement."""

    manifest_path = _write_repo(
        tmp_path,
        [_example("advanced/legacy.py", ["model/run_043"])],
        files={"model/run_043/README.md": b"stub"},
        registry=[
            {
                "model_id": "legacy_ppo_run_043",
                "local_path": "output/model_cache/legacy_ppo_run_043/legacy_ppo_run_043.zip",
                "github_release": {
                    "url": "https://example.invalid/legacy_ppo_run_043.zip",
                    "sha256": _sha256_bytes(b"legacy"),
                },
            }
        ],
    )
    manifest = load_manifest(manifest_path, validate_paths=True)

    report = check_example_prerequisites(manifest, "advanced/legacy")

    assert report.status == STATUS_UNSUPPORTED_LEGACY_REFERENCE
    check = _missing_ok_checks(report, "model/run_043")
    assert check.model_registry is not None
    assert check.model_registry["model_id"] == "legacy_ppo_run_043"
    assert "resolve_model_path" in (check.acquisition or "")
    assert "robot-sf models download legacy_ppo_run_043" in (check.acquisition or "")


def test_unregistered_model_present_is_ready_and_missing_is_model_status(tmp_path: Path) -> None:
    """Unregistered paths fall back to filesystem checks without registry detail."""

    manifest_path = _write_repo(
        tmp_path,
        [
            _example(
                "advanced/unregistered.py",
                ["model/unregistered_present.zip", "model/unregistered_absent.zip"],
            )
        ],
        files={"model/unregistered_present.zip": b"bytes"},
    )
    manifest = load_manifest(manifest_path, validate_paths=True)

    report = check_example_prerequisites(manifest, "advanced/unregistered")

    assert report.status == STATUS_MISSING_MODEL
    present = _missing_ok_checks(report, "model/unregistered_present.zip")
    assert present.status == STATUS_READY
    assert present.model_registry is None
    absent = _missing_ok_checks(report, "model/unregistered_absent.zip")
    assert absent.status == STATUS_MISSING_MODEL


def test_missing_extra_reports_unimportable_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dependency-sync prerequisite fails closed when a probe module is absent."""

    monkeypatch.setitem(
        prerequisites_module._EXTRA_IMPORT_PROBES,
        "uv sync --extra fixture",
        ("fixture_module_that_does_not_exist",),
    )
    manifest_path = _write_repo(
        tmp_path,
        [_example("advanced/extra.py", ["uv sync --extra fixture"])],
    )
    manifest = load_manifest(manifest_path, validate_paths=True)

    report = check_example_prerequisites(manifest, "advanced/extra")

    assert report.status == STATUS_MISSING_EXTRA
    assert _missing_ok_checks(report, "uv sync --extra fixture").status == STATUS_MISSING_EXTRA


def test_operator_placeholder_is_explicit(tmp_path: Path) -> None:
    """Placeholder paths are reported instead of being treated as ready."""

    manifest_path = _write_repo(
        tmp_path,
        [_example("advanced/operator.py", ["examples/recordings/<file>.pkl"])],
    )
    manifest = load_manifest(manifest_path, validate_paths=True)

    report = check_example_prerequisites(manifest, "advanced/operator")

    assert report.status == STATUS_OPERATOR_INPUT_REQUIRED


def test_report_json_is_deterministic_and_matches_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI and the library return byte-identical JSON payloads."""

    manifest_path = _write_repo(
        tmp_path,
        [_example("advanced/cli.py", ["maps/ok.svg"])],
        files={"maps/ok.svg": b"<svg/>"},
    )
    manifest = load_manifest(manifest_path, validate_paths=True)

    expected = report_to_dict(check_example_prerequisites(manifest, "advanced/cli"))
    exit_code = examples_cli_main(
        ["check", "advanced/cli", "--manifest", str(manifest_path), "--format", "json"]
    )
    captured = capsys.readouterr()

    assert json.loads(captured.out) == expected
    assert exit_code == 0
    assert expected["schema"] == CHECK_SCHEMA
    assert set(expected) == {"schema", "example_id", "example_path", "status", "ready", "checks"}


@pytest.mark.parametrize("script", _SCRIPTS)
def test_direct_script_check_matches_manifest_and_imports_no_training_framework(
    script: str,
) -> None:
    """Each named example resolves its own manifest entry without heavy imports."""

    runner = (
        "import contextlib, io, json, runpy, sys\n"
        "script = sys.argv[1]\n"
        "sys.argv = [script, '--check', '--format', 'json']\n"
        "buf = io.StringIO()\n"
        "status = None\n"
        "with contextlib.redirect_stdout(buf):\n"
        "    try:\n"
        "        runpy.run_path(script, run_name='__main__')\n"
        "    except SystemExit as exc:\n"
        "        status = exc.code\n"
        "payload = json.loads(buf.getvalue())\n"
        "print(json.dumps({\n"
        "    'exit': status,\n"
        "    'payload': payload,\n"
        "    'training_imports': [m for m in ('torch', 'stable_baselines3', 'gymnasium')\n"
        "                         if m in sys.modules],\n"
        "}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", runner, str(_REPO_ROOT / "examples" / script)],
        capture_output=True,
        text=True,
        check=True,
        cwd=_REPO_ROOT,
    )
    result = json.loads(completed.stdout)

    query = script.removesuffix(".py")
    expected = check_example_prerequisites(_REAL_MANIFEST, query)
    assert result["payload"]["status"] == expected.status
    assert result["payload"]["schema"] == CHECK_SCHEMA
    assert result["exit"] == (0 if expected.ready else 1)
    assert result["training_imports"] == []


def test_script_helpers_are_self_consistent() -> None:
    """``check_script_prerequisites`` resolves the same report as the manifest query."""

    script_path = _REPO_ROOT / "examples" / "advanced" / "10_offensive_policy.py"
    report = check_script_prerequisites(script_path)
    expected = check_example_prerequisites(_REAL_MANIFEST, "advanced/10_offensive_policy")
    assert report == expected
