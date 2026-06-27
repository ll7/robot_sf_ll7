"""Tests for the re-export readiness preflight (issue #3203).

Cover the three operator-facing readiness states (``fresh``, ``stale``,
``blocked``) using small on-disk fixtures so the classification stays
verifiable without running any benchmark campaign.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from scripts.tools import reexport_readiness_preflight as preflight
from scripts.tools.reexport_readiness_preflight import ReexportReadiness

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, content: bytes) -> str:
    """Write ``content`` to ``path`` and return its sha256 digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _bundle_payload(*, checksum: str, command: str, source_commit: str | None) -> dict:
    """Build a dissertation-bundle manifest payload for the fixtures."""
    artifact = {
        "artifact_id": "tab_scenario_horizon",
        "output_path": "table.md",
        "sha256": checksum,
        "generation_command": command,
    }
    if source_commit is not None:
        artifact["source_commit"] = source_commit
    payload = {
        "schema_version": "dissertation_artifact_bundle.v1",
        "generation_command": command,
        "artifacts": [artifact],
    }
    if source_commit is not None:
        payload["source_commit"] = source_commit
    return payload


def _make_repo_root(tmp_path: Path, *, with_config: bool, with_script: bool) -> Path:
    """Create a fake repo root with optional config and generation script."""
    repo_root = tmp_path / "repo"
    if with_config:
        cfg = repo_root / "configs" / "campaign.yaml"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text("name: scenario_horizon\n", encoding="utf-8")
    if with_script:
        script = repo_root / "scripts" / "tools" / "run_campaign.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("# entry point\n", encoding="utf-8")
    repo_root.mkdir(parents=True, exist_ok=True)
    return repo_root


_COMMAND = "uv run python scripts/tools/run_campaign.py --config configs/campaign.yaml --mode run"


def test_fresh_bundle_with_matching_payload(tmp_path: Path) -> None:
    """Payload present + matching checksum => fresh, no re-export needed."""
    bundle = tmp_path / "bundle"
    digest = _write(bundle / "payload" / "table.md", b"horizon table\n")
    repo_root = _make_repo_root(tmp_path, with_config=True, with_script=True)
    manifest = bundle / "artifact_manifest.json"
    manifest.write_text(
        json.dumps(_bundle_payload(checksum=digest, command=_COMMAND, source_commit="abc123")),
        encoding="utf-8",
    )

    report = preflight.assess_bundle(manifest, repo_root=repo_root)

    assert report.state is ReexportReadiness.FRESH
    assert report.reexport_needed is False
    assert report.exit_code == 0
    assert report.missing_inputs == []


def test_stale_bundle_with_inputs_present_is_reexport_ready(tmp_path: Path) -> None:
    """Missing payload but all inputs present => stale (re-export unblocked)."""
    bundle = tmp_path / "bundle"
    # Manifest references a payload file that is never written -> checksum fails.
    repo_root = _make_repo_root(tmp_path, with_config=True, with_script=True)
    manifest = bundle / "artifact_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            _bundle_payload(checksum="deadbeef" * 8, command=_COMMAND, source_commit="abc123")
        ),
        encoding="utf-8",
    )

    report = preflight.assess_bundle(manifest, repo_root=repo_root)

    assert report.state is ReexportReadiness.STALE
    assert report.reexport_needed is True
    # stale is actionable-but-unblocked, so it is not a hard failure.
    assert report.exit_code == 0
    assert report.missing_inputs == []
    assert {item.name for item in report.required_inputs} == {
        "campaign_config",
        "generation_script",
        "source_commit",
    }


def test_blocked_when_config_input_missing(tmp_path: Path) -> None:
    """Re-export needed but the campaign config is absent => blocked."""
    bundle = tmp_path / "bundle"
    repo_root = _make_repo_root(tmp_path, with_config=False, with_script=True)
    manifest = bundle / "artifact_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            _bundle_payload(checksum="deadbeef" * 8, command=_COMMAND, source_commit="abc123")
        ),
        encoding="utf-8",
    )

    report = preflight.assess_bundle(manifest, repo_root=repo_root)

    assert report.state is ReexportReadiness.BLOCKED
    assert report.reexport_needed is True
    assert report.exit_code == 1
    assert [item.name for item in report.missing_inputs] == ["campaign_config"]


def test_blocked_when_provenance_commit_missing(tmp_path: Path) -> None:
    """Re-export needed and no source_commit recorded => blocked on provenance."""
    bundle = tmp_path / "bundle"
    repo_root = _make_repo_root(tmp_path, with_config=True, with_script=True)
    manifest = bundle / "artifact_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(_bundle_payload(checksum="deadbeef" * 8, command=_COMMAND, source_commit=None)),
        encoding="utf-8",
    )

    report = preflight.assess_bundle(manifest, repo_root=repo_root)

    assert report.state is ReexportReadiness.BLOCKED
    assert "source_commit" in {item.name for item in report.missing_inputs}


def test_placeholder_config_is_treated_as_missing(tmp_path: Path) -> None:
    """A placeholder config path (e.g. ``<config>``) cannot be reproduced."""
    bundle = tmp_path / "bundle"
    repo_root = _make_repo_root(tmp_path, with_config=True, with_script=True)
    manifest = bundle / "artifact_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    command = "uv run python scripts/tools/run_campaign.py --config <config> --mode run"
    manifest.write_text(
        json.dumps(
            _bundle_payload(checksum="deadbeef" * 8, command=command, source_commit="abc123")
        ),
        encoding="utf-8",
    )

    report = preflight.assess_bundle(manifest, repo_root=repo_root)

    assert report.state is ReexportReadiness.BLOCKED
    config_input = next(i for i in report.required_inputs if i.name == "campaign_config")
    assert config_input.present is False
    assert "placeholder" in config_input.detail


def test_malformed_manifest_is_blocked(tmp_path: Path) -> None:
    """An unloadable manifest is blocked, not silently fresh."""
    manifest = tmp_path / "broken.json"
    manifest.write_text("{not valid json", encoding="utf-8")

    report = preflight.assess_bundle(manifest, repo_root=tmp_path)

    assert report.state is ReexportReadiness.BLOCKED
    assert report.exit_code == 1


def test_cli_writes_json_report(tmp_path: Path) -> None:
    """The CLI emits a JSON report and a non-zero exit code when blocked."""
    bundle = tmp_path / "bundle"
    repo_root = _make_repo_root(tmp_path, with_config=False, with_script=True)
    manifest = bundle / "artifact_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            _bundle_payload(checksum="deadbeef" * 8, command=_COMMAND, source_commit="abc123")
        ),
        encoding="utf-8",
    )
    out = tmp_path / "report.json"

    exit_code = preflight.main(
        [str(manifest), "--repo-root", str(repo_root), "--json-out", str(out)]
    )

    assert exit_code == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["state"] == "blocked"
    assert report["schema"] == preflight.SCHEMA_VERSION
    assert "campaign_config" in report["missing_inputs"]
