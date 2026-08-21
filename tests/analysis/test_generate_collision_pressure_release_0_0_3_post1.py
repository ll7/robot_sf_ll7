"""Tests for the deterministic issue-7724 release packet generator."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from robot_sf.benchmark.event_ledger import build_event_ledger
from scripts.analysis import generate_collision_pressure_release_0_0_3_post1 as generator

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "robot_sf" / "benchmark" / "schemas" / "collision_pressure_report.v1.json"


def _event(partner_type: str = "pedestrian") -> dict[str, object]:
    """Build one valid typed collision event for the fixture ledger."""
    return {
        "collision_partner_type": partner_type,
        "collision_partner_id": "pedestrian-1",
        "collision_time": 1.0,
        "relative_speed_at_contact": 0.2,
        "clearance_series_source": "trace.clearance",
        "exact_event_source": "trace.contact",
    }


def _row(episode_id: str, *, family: str, collision: bool) -> dict[str, Any]:
    """Build one release-like episode row with a canonical event ledger."""
    record: dict[str, Any] = {
        "episode_id": episode_id,
        "scenario_id": f"scenario-{episode_id}",
        "scenario_family": family,
        "scenario_params": {"metadata": {"archetype": family}},
        "seed": 7,
        "algo": "goal",
        "metrics": {"collisions": 1.0 if collision else 0.0},
        "outcome": {"collision_event": collision, "route_complete": not collision},
        "termination_reason": "collision" if collision else "success",
        "git_hash": generator.EXPECTED_ROW_PRODUCTION_COMMIT,
    }
    record["event_ledger"] = build_event_ledger(
        record,
        collision_events=[_event()] if collision else [],
    )
    return record


def _json_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize one fixture manifest deterministically."""
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _archive_member_bytes() -> dict[str, bytes]:
    """Return the minimal release payload needed by the fail-closed verifier."""
    root = generator.EXPECTED_BUNDLE_ROOT
    selected = _row("episode-1", family="doorway", collision=True)
    excluded = _row("episode-2", family="other", collision=False)
    episodes = (
        json.dumps(selected, sort_keys=True) + "\n" + json.dumps(excluded, sort_keys=True) + "\n"
    ).encode()
    publication = {
        "bundle_name": root,
        "publication_channels": {"release_tag": generator.EXPECTED_RELEASE_TAG},
        "provenance": {
            "repository": {"commit": generator.EXPECTED_PUBLICATION_COMMIT},
            "commit_reconciliation": {
                "publication_commit": generator.EXPECTED_PUBLICATION_COMMIT,
                "execution_commit": generator.EXPECTED_ROW_PRODUCTION_COMMIT,
            },
        },
    }
    payload_manifest = {
        "git_hash": generator.EXPECTED_PUBLICATION_COMMIT,
        "benchmark_release": {
            "release_tag": generator.EXPECTED_RELEASE_TAG,
            "release_id": generator.EXPECTED_RELEASE_ID,
        },
    }
    release_manifest = {
        "release_tag": generator.EXPECTED_RELEASE_TAG,
        "release_id": generator.EXPECTED_RELEASE_ID,
        "planners": {"keys": ["goal"]},
        "kinematics": {"matrix": ["differential_drive"]},
    }
    breakdown = (
        b"scenario_family,episodes,collisions_mean,total_collision_count_mean,"
        b"ped_collision_count_mean,obstacle_collision_count_mean\n"
        b"doorway,1,1,1,1,0\n"
    )
    return {
        "publication_manifest.json": _json_bytes(publication),
        "payload/manifest.json": _json_bytes(payload_manifest),
        "payload/release/release_manifest.resolved.json": _json_bytes(release_manifest),
        "payload/runs/goal__differential_drive/episodes.jsonl": episodes,
        "payload/reports/scenario_family_breakdown.csv": breakdown,
    }


def _write_fixture_archive(path: Path) -> Path:
    """Write a small checksum-complete release archive for contract tests."""
    root = generator.EXPECTED_BUNDLE_ROOT
    members = _archive_member_bytes()
    checksum_lines = [
        f"{hashlib.sha256(payload).hexdigest()}  {name}"
        for name, payload in sorted(members.items())
    ]
    checksums = ("\n".join(checksum_lines) + "\n").encode()
    with tarfile.open(path, "w:gz") as archive:
        root_info = tarfile.TarInfo(f"{root}/")
        root_info.type = tarfile.DIRTYPE
        root_info.mode = 0o755
        archive.addfile(root_info)
        for name, payload in sorted(members.items()):
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(payload)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
        checksum_info = tarfile.TarInfo(f"{root}/checksums.sha256")
        checksum_info.size = len(checksums)
        checksum_info.mode = 0o644
        archive.addfile(checksum_info, io.BytesIO(checksums))
    return path


def _patch_fixture_contract(monkeypatch: pytest.MonkeyPatch, archive: Path) -> None:
    """Patch only immutable count expectations for the minimal fixture."""
    selected_digest = hashlib.sha256(
        json.dumps(["goal::episode-1"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    values = {
        "EXPECTED_BUNDLE_SHA256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "EXPECTED_RUN_COUNT": 1,
        "EXPECTED_ROWS_PER_RUN": 2,
        "EXPECTED_TOTAL_ROWS": 2,
        "ELIGIBLE_FAMILIES": ("doorway",),
        "EXPECTED_ELIGIBLE_ROWS": 1,
        "EXPECTED_DENOMINATOR_DIGEST": selected_digest,
        "EXPECTED_CONTACT_EPISODES": 1,
        "EXPECTED_COLLISION_EVENTS": 1,
        "EXPECTED_PARTNER_EPISODE_COUNTS": {
            "pedestrian": 1,
            "static_geometry": 0,
            "boundary": 0,
            "goal_artifact": 0,
        },
        "EXPECTED_OBSTACLE_EPISODES": 0,
        "EXPECTED_OVERLAP_COUNTS": {
            "pedestrian_only": 1,
            "obstacle_only": 0,
            "pedestrian_and_obstacle": 0,
        },
        "EXPECTED_FAMILY_COUNTS": {
            "doorway": {"eligible_episode_count": 1, "contact_episode_count": 1}
        },
        "EXPECTED_MISSING_OPTIONAL_FIELDS": {
            "collision_partner_id": 0,
            "relative_speed_at_contact": 0,
        },
        "EXPECTED_MISSING_PARTNER_TYPES": {},
        "EXPECTED_PAYLOAD_CHECKSUM_ENTRIES": 5,
    }
    for name, value in values.items():
        monkeypatch.setattr(generator, name, value)


@pytest.fixture
def fixture_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a release archive whose identity is patched to its computed digest."""
    archive = _write_fixture_archive(tmp_path / "fixture.tar.gz")
    _patch_fixture_contract(monkeypatch, archive)
    return archive


def test_release_packet_reconciles_schema_and_event_ledger(
    fixture_archive: Path, tmp_path: Path
) -> None:
    """A checksum-complete release slice yields a schema-valid, reconciled packet."""
    output_dir = tmp_path / "packet"
    result = generator.build_packet(fixture_archive, output_dir)

    report = json.loads((output_dir / "collision_pressure_report.json").read_text())
    jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text())).validate(report)
    assert report["selection"]["excluded_row_count"] == 0
    assert report["denominator"]["eligible_episode_key_sha256"] == (
        "4916860570c6880720f68a883e55183ef8f0d6cef0e0791d72f686d24b0335c7"
    )
    assert report["counts"]["partner_type_episode_counts"]["pedestrian"] == 1
    assert result["row_summary"]["total_release_rows"] == 2
    manifest = json.loads((output_dir / "source_manifest.json").read_text())
    assert manifest["adapter"]["scenario_family"]["source_locator"] == (
        "scenario_params.metadata.archetype"
    )
    assert manifest["missing_optional_fields"] == {
        "collision_partner_id": 0,
        "relative_speed_at_contact": 0,
    }


def test_release_packet_regeneration_is_byte_deterministic(
    fixture_archive: Path, tmp_path: Path
) -> None:
    """The same verified archive produces byte-identical packet files."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    generator.build_packet(fixture_archive, first)
    generator.build_packet(fixture_archive, second)

    first_files = sorted(path.name for path in first.iterdir() if path.is_file())
    second_files = sorted(path.name for path in second.iterdir() if path.is_file())
    assert first_files == second_files
    for name in first_files:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_release_archive_checksum_mismatch_fails_closed(
    fixture_archive: Path, tmp_path: Path
) -> None:
    """A changed archive is rejected before extraction or report generation."""
    tampered = tmp_path / "tampered.tar.gz"
    tampered.write_bytes(fixture_archive.read_bytes() + b"tampered")

    with pytest.raises(generator.CollisionPressureReleaseError, match="SHA-256 mismatch"):
        generator.verify_release_archive(tampered)


def test_module_cli_subprocess_executes_main(tmp_path: Path) -> None:
    """The report module's subprocess boundary must invoke its real CLI."""
    rows = _row("episode-1", family="family_a", collision=True)
    rows_path = tmp_path / "episodes.jsonl"
    rows_path.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    command = [
        sys.executable,
        "-m",
        "robot_sf.benchmark.collision.collision_pressure_report",
        "--rows",
        str(rows_path),
        "--eligible-family",
        "family_a",
        "--source-commit",
        generator.EXPECTED_ROW_PRODUCTION_COMMIT,
        "--release-id",
        "release-test",
        "--bundle-id",
        "bundle-test",
        "--input-checksum",
        f"episodes.jsonl={'a' * 64}",
        "--json-out",
        str(json_path),
        "--csv-out",
        str(csv_path),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr
    assert json_path.is_file()
    assert csv_path.is_file()
    assert "collision-pressure report written" in completed.stdout
