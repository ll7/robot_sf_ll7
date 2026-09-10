"""Focused contract tests for the credential-safe locator snapshot tool."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.tools import locator_snapshot as tool

FIXTURES = Path(__file__).parent / "fixtures" / "locator_snapshot"
GOOD_REGISTRY = FIXTURES / "registry.good.yaml"
GOOD_OVERLAY = FIXTURES / "overlay.good.yaml"
BAD_DUPLICATE = FIXTURES / "registry.bad_duplicate.yaml"
BAD_EXPIRED = FIXTURES / "registry.bad_expired.yaml"
SECRET = "PRIVATE-OVERLAY-SECRET-8857"
PRIVATE_HOST = "robot-sf-private-invalid"
GOOD_ARGS = ["--registry", str(GOOD_REGISTRY), "--overlay", str(GOOD_OVERLAY)]
ALIAS = {
    "artifact_class": "model",
    "locator_class": "model_alias",
    "locator": "robot-sf/risk:latest",
}
LOCAL = {"locator_class": "local_path", "locator": "payload.bin"}
EXPECTED_CLASSES = {"local_path", "artifact_uri", "model_alias", "dataset_root", "durable_mirror"}
EXPECTED_DESTINATIONS = {
    "durable_mirror",
    "release_artifact",
    "registry_entry",
    "tracked_path",
    "none",
}


def _entry(logical_id: str, digest: str, **overrides: Any) -> dict[str, Any]:
    """Return one structurally valid registry entry."""
    entry: dict[str, Any] = {
        "logical_id": logical_id,
        "owner": "issue-8857",
        "artifact_class": "artifact",
        "source_issue": "8857",
        "content_digest": digest,
        "schema_version": "fixture.v1",
        "locator_class": "artifact_uri",
        "locator": "https://artifacts.example.invalid/robot-sf/fixture.bin",
        "availability": "available",
        "durable_destination_class": "durable_mirror",
        "consumer_references": ["scripts/tools/locator_snapshot.py"],
    }
    entry.update(overrides)
    return entry


def _registry(tmp_path: Path, entries: list[dict[str, Any]], name: str = "registry.yaml") -> Path:
    """Write a synthetic registry fixture."""
    path = tmp_path / name
    payload = yaml.safe_dump({"schema": tool.REGISTRY_SCHEMA, "entries": entries}, sort_keys=False)
    path.write_text(payload, encoding="utf-8")
    return path


def _overlay(tmp_path: Path, entries: dict[str, Any], name: str = "overlay.yaml") -> Path:
    """Write a synthetic private overlay fixture."""
    path = tmp_path / name
    payload = yaml.safe_dump({"schema": tool.OVERLAY_SCHEMA, "entries": entries}, sort_keys=False)
    path.write_text(payload, encoding="utf-8")
    return path


def _codes(result: tool.SnapshotResult) -> set[str]:
    """Return the set of fail-closed issue codes."""
    return {issue.code for issue in result.issues}


def test_good_fixture_is_byte_stable_and_covers_locator_classes() -> None:
    """The curated fixture passes and repeated builds are byte-identical."""
    kwargs = {"overlay_path": GOOD_OVERLAY, "as_of": "2026-09-10T00:00:00Z"}
    first = tool.build_locator_snapshot([GOOD_REGISTRY], **kwargs)
    rendered = first.render_snapshot_json()
    assert first.ok, [issue.message for issue in first.issues]
    assert rendered == tool.build_locator_snapshot([GOOD_REGISTRY], **kwargs).render_snapshot_json()
    snapshot = json.loads(rendered)
    classes = {e["locator_class"] for e in snapshot["entries"]}
    destinations = {e["durable_destination_class"] for e in snapshot["entries"]}
    assert EXPECTED_CLASSES <= classes and EXPECTED_DESTINATIONS <= destinations
    assert len(snapshot["entries"]) == 6 and snapshot["as_of"] == "2026-09-10T00:00:00Z"
    assert "observed_at" not in rendered


def test_private_overlay_values_are_redacted_everywhere(tmp_path: Path, capsys: Any) -> None:
    """Overlay locators and secrets never reach snapshots, indexes, or diagnostics."""
    overlay = _overlay(
        tmp_path, {"campaign.secret": {"locator": f"s3://{PRIVATE_HOST}/x/{SECRET}"}}
    )
    secret = _entry(
        "campaign.secret", "a" * 64, artifact_class="campaign", locator_class="private_overlay"
    )
    registry = _registry(tmp_path, [secret])
    result = tool.build_locator_snapshot([registry], overlay_path=overlay)
    rendered = result.render_snapshot_json() + result.render_recovery_index_markdown()
    assert result.ok and "s3://" not in rendered
    assert SECRET not in rendered and PRIVATE_HOST not in rendered
    args = ["--check", "--registry", str(registry), "--overlay", str(overlay)]
    assert tool.main([*args, "--format", "json"]) == 0
    captured = capsys.readouterr()
    assert (
        SECRET not in captured.out + captured.err
        and PRIVATE_HOST not in captured.out + captured.err
    )


def test_check_exit_codes_no_partial_outputs_and_normal_mode(tmp_path: Path) -> None:
    """Good fixtures exit 0, bad fixtures exit 2, and failures write no output files."""
    assert tool.main(["--check", *GOOD_ARGS, "--format", "json"]) == 0
    assert tool.main(["--check", "--registry", str(BAD_DUPLICATE), "--format", "json"]) == 2
    assert tool.main(["--check", "--registry", str(BAD_EXPIRED), "--format", "json"]) == 2
    snapshot_out, index_out = tmp_path / "s.json", tmp_path / "r.md"
    write_args = ["--snapshot-out", str(snapshot_out), "--recovery-index-out", str(index_out)]
    assert tool.main([*GOOD_ARGS, *write_args]) == 0
    expected = tool.build_locator_snapshot([GOOD_REGISTRY], overlay_path=GOOD_OVERLAY)
    assert snapshot_out.read_text(encoding="utf-8") == expected.render_snapshot_json()
    assert "Locator Recovery Index" in index_out.read_text(encoding="utf-8")
    snapshot_out.unlink()
    index_out.unlink()
    assert tool.main(["--registry", str(BAD_DUPLICATE), *write_args]) == 2
    assert not snapshot_out.exists() and not index_out.exists()


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"owner": ""}, "missing_owner"),
        ({"availability": "expired"}, "expired_locator"),
        ({"availability": "missing"}, "missing_target"),
        ({"availability": "transfer_incomplete"}, "incomplete_transfer"),
        ({"locator_class": "signed_url"}, "signed_locator"),
        (
            {"locator": "https://artifacts.example.invalid/f.bin?X-Amz-Signature=abc"},
            "signed_locator",
        ),
        ({"locator": None}, "missing_locator"),
        ({"locator_class": "local_path", "locator": "does_not_exist.bin"}, "missing_target"),
        (ALIAS | {"durable_destination_class": "none"}, "mutable_alias_not_durable"),
    ],
)
def test_fail_closed_conditions(tmp_path: Path, overrides: dict[str, Any], expected: str) -> None:
    """Each fail-closed condition produces its explicit diagnostic code."""
    entry = _entry("artifact.fail_closed", "a" * 64, **overrides)
    result = tool.build_locator_snapshot([_registry(tmp_path, [entry])])
    assert expected in _codes(result) and not result.ok


def test_conflicts_overlay_requirements_orphans_and_redaction(tmp_path: Path) -> None:
    """Duplicate/conflicting IDs fail; private-class entries need an overlay; values stay hidden."""
    same = _registry(tmp_path, [_entry("dup", "a" * 64), _entry("dup", "a" * 64)])
    codes = _codes(tool.build_locator_snapshot([same]))
    assert "duplicate_logical_id" in codes and "conflicting_bytes" not in codes
    conflict = _registry(tmp_path, [_entry("dup", "a" * 64), _entry("dup", "b" * 64)], "c.yaml")
    assert {"duplicate_logical_id", "conflicting_bytes"} <= _codes(
        tool.build_locator_snapshot([conflict])
    )
    digest_overlay = _overlay(
        tmp_path,
        {"dup": {"locator": f"s3://{PRIVATE_HOST}/x", "content_digest": "c" * 64}},
        "dg.yaml",
    )
    only = _registry(tmp_path, [_entry("dup", "a" * 64)], "o.yaml")
    assert "conflicting_bytes" in _codes(
        tool.build_locator_snapshot([only], overlay_path=digest_overlay)
    )
    needs = _entry(
        "campaign.needs", "a" * 64, artifact_class="campaign", locator_class="private_overlay"
    )
    assert "missing_private_overlay" in _codes(
        tool.build_locator_snapshot([_registry(tmp_path, [needs], "n.yaml")])
    )
    orphan = _overlay(
        tmp_path, {f"orphan.{SECRET}": {"locator": f"s3://{PRIVATE_HOST}/x"}}, "orph.yaml"
    )
    registry = _registry(tmp_path, [_entry("owned", "b" * 64, locator=None)], "owned.yaml")
    result = tool.build_locator_snapshot([registry], overlay_path=orphan)
    assert "orphan_overlay_entry" in _codes(result)
    assert SECRET not in json.dumps(result.to_check_report_dict())
    kept = _overlay(tmp_path, {"owned": {"locator": f"s3://{PRIVATE_HOST}/y"}}, "kept.yaml")
    result2 = tool.build_locator_snapshot([registry], overlay_path=kept)
    assert result2.ok and result2.entries[0].locator_source == "private_overlay"
    assert result2.entries[0].verification == "unresolved"
    bad_class = _overlay(tmp_path, {"owned": {"locator": "s3://x.invalid", "locator_class": "bad"}})
    invalid = tool.build_locator_snapshot([registry], overlay_path=bad_class)
    assert not invalid.ok and "invalid_overlay_entry" in _codes(invalid)
    assert invalid.entries[0].locator_class == "artifact_uri"


def test_readable_locator_and_destination_verify_digest(tmp_path: Path) -> None:
    """Readable local bytes are verified; mismatched bytes fail closed."""
    content = b"locator-snapshot-unit-payload\n"
    (tmp_path / "payload.bin").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    entry = _entry(
        "artifact.local",
        digest,
        **LOCAL
        | {"durable_destination_class": "tracked_path", "durable_destination": "payload.bin"},
    )
    result = tool.build_locator_snapshot([_registry(tmp_path, [entry])])
    assert result.ok and result.entries[0].verification == "verified"
    assert result.entries[0].destination_verification == "verified"
    wrong = _entry("artifact.wrong", "a" * 64, **LOCAL)
    assert "digest_mismatch" in _codes(
        tool.build_locator_snapshot([_registry(tmp_path, [wrong], "w.yaml")])
    )


def test_pinned_alias_warns_and_unavailable_is_recorded(tmp_path: Path) -> None:
    """A pinned alias warns; an unavailable locator is recorded, not failed."""
    alias = _entry(
        "model.latest",
        "a" * 64,
        **ALIAS
        | {
            "durable_destination_class": "registry_entry",
            "durable_destination": "model-registry://risk/v1",
        },
    )
    result = tool.build_locator_snapshot([_registry(tmp_path, [alias])])
    assert result.ok and any(w.code == "mutable_alias_current_locator" for w in result.warnings)
    unavailable = _entry(
        "artifact.unavailable",
        "b" * 64,
        **LOCAL | {"locator": "not_there.bin", "availability": "unavailable"},
    )
    result2 = tool.build_locator_snapshot([_registry(tmp_path, [unavailable], "u.yaml")])
    assert result2.ok and result2.entries[0].verification == "missing"
