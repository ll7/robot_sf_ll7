"""Real-byte disclosure witnesses, independent of sealed episode execution."""

import hashlib
import json
import shutil
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.evidence.writers import write_json
from robot_sf.evidence.writers import write_text as marked_write_text

write_text = partial(marked_write_text, issue_ref="robot_sf#10110")

ROOT = Path(__file__).resolve().parents[2]
NOTES = "docs/release/0.0.8/release_notes.md"
# Independent oracles: adopted content and section, not production requirements.
DISCLOSURES = [
    ("D-039", "Scripted pedestrian speed", "The declared scripted pedestrian speed"),
    ("D-040", "Group forces", "Group gaze and group repulsion"),
    ("D-044", "Elevator geometry", "The elevator scenario interior walls"),
    ("D-053", "Learned reference", "The plain ppo arm replaces"),
    ("D-055", "Narrow passages", "Flagged rows:"),
    ("D-056", "Upstream queue", "The three-width doorway slice stays"),
    ("D-057", "Crowd speed and body size", "Crowd desired speed and hard cap"),
    ("D-058", "Pedestrian robot response", "Simulated pedestrians slow down"),
    ("D-059", "Robot motion", "The 0.0.8 release robot uses"),
    ("D-060", "Occupancy grid", "The occupancy-grid rasteriser"),
    ("D-061", "Collision geometry and TTC", "Time to collision is centre-based"),
    ("D-062", "Cross-release interpretation", "0.0.7 and 0.0.8 are different benchmarks"),
    ("D-063", "Small-crowd groups", "The groups setting remains"),
    ("D-065", "Non-release sampler test", "The 400-step safety ruling applies"),
    ("D-075", "Doorway slice admission", "The benchmark-doorway-width-slice.v1 dataset"),
    ("D-076", "Versioned release notes", "The versioned 0.0.8 release notes live"),
]


def _remove(text, prefix):
    blocks = text.split("\n\n")
    matching = [block for block in blocks if block.startswith(prefix)]
    assert len(matching) == 1
    return "\n\n".join(block for block in blocks if block != matching[0]), matching[0]


@pytest.fixture
def notes_repo(tmp_path):
    for relative in (
        NOTES,
        "docs/release/0.0.8/decisions.md",
        "robot_sf/benchmark/release_notes.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return tmp_path


def test_actual_notes_bind_every_decision_and_source_digest(notes_repo):
    from robot_sf.benchmark.release_notes import notes_gate

    receipt = notes_gate(notes_repo, source_commit="a" * 40)
    assert receipt["decisions"] == [row[0] for row in DISCLOSURES]
    assert receipt["notes_sha256"] == hashlib.sha256((ROOT / NOTES).read_bytes()).hexdigest()
    assert (
        receipt["checker_sha256"]
        == hashlib.sha256((ROOT / "robot_sf/benchmark/release_notes.py").read_bytes()).hexdigest()
    )
    assert (
        receipt["decisions_sha256"]
        == hashlib.sha256((ROOT / "docs/release/0.0.8/decisions.md").read_bytes()).hexdigest()
    )
    assert (
        notes_gate(notes_repo, source_commit="a" * 40, phase="publication", receipt=receipt)
        == receipt
    )


@pytest.mark.parametrize("decision,section,prefix", DISCLOSURES, ids=[r[0] for r in DISCLOSURES])
def test_each_deleted_disclosure_refused(notes_repo, decision, section, prefix):
    from robot_sf.benchmark.release_notes import notes_gate

    path = notes_repo / NOTES
    changed, _ = _remove(path.read_text(), prefix)
    write_text(path, changed)
    with pytest.raises(ValueError, match=f"{decision}: missing required statement in {section}"):
        notes_gate(notes_repo, source_commit="a" * 40)


@pytest.mark.parametrize("decision,section,prefix", DISCLOSURES, ids=[r[0] for r in DISCLOSURES])
def test_keywords_and_statements_in_unrelated_prose_refused(notes_repo, decision, section, prefix):
    from robot_sf.benchmark.release_notes import notes_gate

    path = notes_repo / NOTES
    changed, paragraph = _remove(path.read_text(), prefix)
    write_text(path, changed + "\n\n## Unrelated prose\n\n" + paragraph + "\n")
    with pytest.raises(ValueError, match=f"{decision}: missing required statement in {section}"):
        notes_gate(notes_repo, source_commit="a" * 40)


@pytest.mark.parametrize(
    "placeholder",
    ["PLACEHOLDER_SEALED_SUCCESS_COUNT", "TODO", "TBD", "{{sealed_count}}", "<sealed-count>"],
)
def test_placeholder_mint_allowed_publication_refused(notes_repo, placeholder):
    from robot_sf.benchmark.release_notes import notes_gate

    path = notes_repo / NOTES
    write_text(path, path.read_text() + "\n\n## Campaign outcomes\n\n" + placeholder + "\n")
    receipt = notes_gate(notes_repo, source_commit="a" * 40)
    with pytest.raises(ValueError, match="publication refuses placeholders"):
        notes_gate(notes_repo, source_commit="a" * 40, phase="publication", receipt=receipt)


def test_review_marker_is_not_a_publication_placeholder(notes_repo):
    from robot_sf.benchmark.release_notes import notes_gate

    path = notes_repo / NOTES
    write_text(path, path.read_text())
    receipt = notes_gate(notes_repo, source_commit="a" * 40)
    assert (
        notes_gate(notes_repo, source_commit="a" * 40, phase="publication", receipt=receipt)
        == receipt
    )


def test_stale_notes_rejected_even_when_disclosures_pass(notes_repo):
    from robot_sf.benchmark.release_notes import notes_gate

    receipt = notes_gate(notes_repo, source_commit="a" * 40)
    path = notes_repo / NOTES
    write_text(path, path.read_text() + "\nUnrelated update.\n")
    with pytest.raises(ValueError, match="digest mismatch against mint receipt"):
        notes_gate(notes_repo, source_commit="a" * 40, phase="publication", receipt=receipt)


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "schema",
        "status",
        "source",
        "path",
        "decisions",
        "checker",
        "register",
        "invalid-phase",
        "duplicate",
        "negation",
        "malformed",
    ],
)
def test_fail_closed_receipt_and_section_controls(notes_repo, case):
    from robot_sf.benchmark.release_notes import notes_gate

    receipt = notes_gate(notes_repo, source_commit="a" * 40)
    fields = {
        "schema": "schema_version",
        "status": "status",
        "source": "source_commit",
        "path": "notes_path",
        "decisions": "decisions",
        "checker": "checker_sha256",
        "register": "decisions_sha256",
    }
    if case in fields:
        receipt[fields[case]] = "wrong"
    elif case == "missing":
        receipt = None
    elif case == "malformed":
        receipt = []
    elif case in {"duplicate", "negation"}:
        path = notes_repo / NOTES
        text = path.read_text()
        if case == "duplicate":
            text += "\n\n## Robot motion\n"
        else:
            text = text.replace("The declared scripted", "It is false that the declared scripted")
        write_text(path, text)
        with pytest.raises(
            ValueError, match="duplicate section" if case == "duplicate" else "D-039"
        ):
            notes_gate(notes_repo, source_commit="a" * 40)
        return
    with pytest.raises(ValueError):
        notes_gate(
            notes_repo,
            source_commit="a" * 40,
            phase="wrong" if case == "invalid-phase" else "publication",
            receipt=receipt,
        )


def test_preflight_and_direct_publication_recheck_mint_receipt(notes_repo, monkeypatch):
    from robot_sf import release_cli
    from robot_sf.benchmark.release_notes import RECEIPT_NAME, gate_manifest, notes_gate

    identity = notes_repo / "output/identity.json"
    identity.parent.mkdir()
    receipt = notes_gate(notes_repo, source_commit="a" * 40)
    write_json(identity.parent / RECEIPT_NAME, receipt)
    manifest = SimpleNamespace(
        canonical_campaign_config_path=Path("campaign_0_0_8.yaml"),
        resolved_identity_path=identity,
        source_sha="a" * 40,
    )
    assert gate_manifest(manifest, notes_repo) == receipt
    monkeypatch.setattr(release_cli, "get_repository_root", lambda: notes_repo)
    release_cli._publication_notes_gate(manifest)
    path = notes_repo / NOTES
    write_text(path, path.read_text() + "\nStale.\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        gate_manifest(manifest, notes_repo)
    with pytest.raises(release_cli.zenodo_publisher.ZenodoPublisherError, match="digest mismatch"):
        release_cli._publication_notes_gate(manifest)
    manifest.resolved_identity_path = None
    with pytest.raises(ValueError, match="requires production mint receipt"):
        gate_manifest(manifest, notes_repo)
    manifest.release_kind = "development_rehearsal"
    assert gate_manifest(manifest, notes_repo) is None
    assert gate_manifest(SimpleNamespace(), notes_repo) is None


@pytest.mark.parametrize("blank_lines", [1, 2, 3, 4])
def test_duplicate_heading_refused_across_blank_line_counts(notes_repo, blank_lines):
    from robot_sf.benchmark.release_notes import notes_gate

    path = notes_repo / NOTES
    write_text(path, path.read_text() + "\n" * blank_lines + "## Robot motion\n")
    with pytest.raises(ValueError, match="duplicate section: Robot motion"):
        notes_gate(notes_repo, source_commit="a" * 40)


def test_exported_notes_checked_against_retained_digest(notes_repo, monkeypatch):
    from robot_sf.benchmark import artifact_publication as publication
    from robot_sf.benchmark.release_notes import RECEIPT_NAME, notes_gate

    receipt = notes_gate(notes_repo, source_commit="a" * 40)
    resolved = {
        "canonical_campaign_config": "campaign_0_0_8.yaml",
        "source_sha": "a" * 40,
        "release_notes_gate": receipt,
    }
    monkeypatch.setattr(publication, "get_repository_root", lambda: notes_repo)
    run = notes_repo / "run"
    (run / "release").mkdir(parents=True)
    roles = publication._release_notes_files(run, notes_repo, resolved)
    payload = notes_repo / "payload"
    for source, destination in roles.values():
        target = payload / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (payload / "release").mkdir()
    write_json(payload / "release/release_manifest.resolved.json", resolved)
    violations = []
    publication._preflight_release_notes(payload, violations)
    assert violations == []
    path = payload / "release_metadata/release_notes.md"
    write_text(path, path.read_text() + "\nStale exported prose.\n")
    publication._preflight_release_notes(payload, violations)
    assert len(violations) == 1 and "digest mismatch" in violations[0]
    write_json(payload / "release_metadata" / RECEIPT_NAME, {**receipt, "source_commit": "b" * 40})
    violations = []
    publication._preflight_release_notes(payload, violations)
    assert "bundle receipt differs from mint receipt" in violations[0]


def _resign_notes(bundle, notes):
    """Keep ordinary checksums and manifest entries consistent after a notes edit."""
    relative = "payload/release_metadata/release_notes.md"
    digest = hashlib.sha256(notes.read_bytes()).hexdigest()
    checksums = bundle / "checksums.sha256"
    lines = [line for line in checksums.read_text().splitlines() if not line.endswith(relative)]
    marked_write_text(
        checksums,
        "# AI-GENERATED NEEDS-REVIEW\n" + "\n".join(lines) + "\n" + digest + "  " + relative + "\n",
    )
    path = bundle / "publication_manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["files"] = [entry for entry in manifest["files"] if entry.get("path") != relative]
    manifest["files"].append({"path": relative, "sha256": digest})
    write_json(path, manifest)


def test_publication_preflight_route_refuses_resigned_stale_notes(notes_repo, monkeypatch):
    from robot_sf.benchmark import artifact_publication as publication
    from robot_sf.benchmark.release_notes import RECEIPT_NAME, notes_gate
    from tests.validation.test_publication_preflight import _build_bundle

    source = "a" * 40
    bundle = _build_bundle(notes_repo / "bundle-test", publication_commit=source)
    payload = bundle / "payload"
    receipt = notes_gate(notes_repo, source_commit=source)
    resolved = {
        "canonical_campaign_config": "campaign_0_0_8.yaml",
        "source_sha": source,
        "release_notes_gate": receipt,
    }
    write_json(payload / "release/release_manifest.resolved.json", resolved)
    target = payload / "release_metadata"
    target.mkdir()
    shutil.copy2(notes_repo / NOTES, target / "release_notes.md")
    write_json(target / RECEIPT_NAME, receipt)
    # The marked JSON receipt must be byte-identical to the one in resolved metadata.
    resolved["release_notes_gate"] = json.loads((target / RECEIPT_NAME).read_bytes())
    write_json(payload / "release/release_manifest.resolved.json", resolved)
    monkeypatch.setattr(publication, "get_repository_root", lambda: notes_repo)
    _resign_notes(bundle, target / "release_notes.md")
    assert publication.verify_publication_bundle_preflight(bundle)["status"] == "pass"
    notes = target / "release_notes.md"
    write_text(notes, notes.read_text() + "\nUnrelated editorial change.\n")
    # Re-signing a changed file cannot substitute for the production mint digest.
    _resign_notes(bundle, notes)
    with pytest.raises(
        publication.PublicationPreflightError, match="digest mismatch against mint receipt"
    ):
        publication.verify_publication_bundle_preflight(bundle)


@pytest.mark.parametrize("consistent", [False, True])
def test_diagnostic_notes_exemption_requires_consistent_run_markers(notes_repo, consistent):
    from robot_sf.benchmark import artifact_publication as publication

    resolved = {
        "canonical_campaign_config": "campaign_0_0_8.yaml",
        "release_kind": "development_rehearsal",
    }
    payload = notes_repo / "payload"
    (payload / "release").mkdir(parents=True)
    write_json(payload / "release/release_manifest.resolved.json", resolved)
    write_json(
        payload / "release/release_result.json",
        {
            "benchmark_release": {
                "release_kind": "development_rehearsal" if consistent else "benchmark-data"
            },
            "release_eligible": not consistent,
            "release_benchmark_success": not consistent,
        },
    )
    violations = []
    publication._preflight_release_notes(payload, violations)
    if consistent:
        assert violations == []
        assert publication._release_notes_files(payload, notes_repo, resolved) == {}
    else:
        assert len(violations) == 1 and "diagnostic exemption disagrees" in violations[0]
        with pytest.raises(ValueError, match="diagnostic exemption disagrees"):
            publication._release_notes_files(payload, notes_repo, resolved)


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({"release_kind": "benchmark-doorway-width-slice.v1"}, True),
        ({"release_kind": "benchmark-width-slice"}, True),
        ({"release_tag": "release-0.0.8-final"}, True),
        ({"scenario_matrix": "scenarios_0_0_8.yaml"}, True),
        ({"scenario_matrix_path": Path("scenarios_0_0_8.yaml")}, True),
        ({"canonical_campaign_config": "campaign_0_0_8.yaml"}, True),
        ({"canonical_campaign_config_path": Path("campaign_0_0_8.yaml")}, True),
        ({"canonical_campaign_config": "paper_experiment_matrix_v2_h600_s30.yaml"}, False),
        ({}, False),
    ],
)
def test_shared_release_detector_handles_native_and_archived_fields(fields, expected):
    from robot_sf.benchmark.release_notes import is_release_0_0_8

    assert is_release_0_0_8(fields) is expected
    assert is_release_0_0_8(SimpleNamespace(**fields)) is expected
