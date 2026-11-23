"""Cold publication score validation, independent of legacy SNQI exclusion."""

from __future__ import annotations

import json

import pytest

from robot_sf.benchmark.artifact_publication import _publication_snqi_evidence


def _payload(tmp_path, metrics):
    """Stage one scored row and the explicit v2 diagnostics declaration."""
    (tmp_path / "reports").mkdir(parents=True)
    (tmp_path / "reports/snqi_v2_diagnostics.json").write_text(
        json.dumps(
            {
                "schema_version": "snqi-v2-diagnostics.v1",
                "episode_count": 1,
            }
        )
    )
    (tmp_path / "campaign_manifest.json").write_text(json.dumps({"legacy_snqi": "excluded"}))
    (tmp_path / "runs/goal").mkdir(parents=True)
    (tmp_path / "runs/goal/episodes.jsonl").write_text(json.dumps({"metrics": metrics}) + "\n")
    return tmp_path


def _metrics(success=True, collisions=0, score=1.0):
    """Hand-calculated nominal score: all quality penalties are zero."""
    return {
        "success": success,
        "total_collision_count": collisions,
        "snqi_v2": score,
        "snqi_v2_terms": {
            "S": int(success),
            "C": int(collisions > 0),
            "T": 0.0,
            "N": 0.0,
            "F": 0.0,
            "J": 0.0,
            "K": 0.0,
        },
    }


@pytest.mark.parametrize(
    "success,collisions,score",
    [
        (True, 0, -999),
        (True, 0, 1.01),
        (False, 0, 0.01),
        (False, 0, -1.01),
        (True, 1, 0.0),
        (False, 1, -999),
        (True, 0, float("nan")),
        (True, 0, True),
    ],
)
def test_v2_rejects_impossible_scores_despite_legacy_exclusion(
    tmp_path, success, collisions, score
):
    """Excluded legacy scoring must not bypass the v2 numerical gate."""
    result = _publication_snqi_evidence(_payload(tmp_path, _metrics(success, collisions, score)))
    assert result["checked"] is True
    assert any("SNQI-v2" in reason for reason in result["violations"])


@pytest.mark.parametrize(
    "success,collisions,score", [(True, 0, 1), (False, 0, 0), (True, 1, -1), (False, 1, -2)]
)
def test_v2_accepts_hand_calculated_scores(tmp_path, success, collisions, score):
    """Each stratum permits the nominal score with zero quality penalties."""
    result = _publication_snqi_evidence(_payload(tmp_path, _metrics(success, collisions, score)))
    assert result["checked"] is True
    assert result["violations"] == []


@pytest.mark.parametrize(
    "field,value",
    [("snqi_v2", None), ("snqi_v2_terms", {}), ("success", 2), ("total_collision_count", -1)],
)
def test_v2_refuses_incomplete_or_malformed_rows(tmp_path, field, value):
    """A declared v2 publication cannot silently omit required numerical inputs."""
    metrics = _metrics()
    metrics[field] = value
    result = _publication_snqi_evidence(_payload(tmp_path, metrics))
    assert result["checked"] is True
    assert result["violations"]


def test_v2_refuses_score_without_diagnostics(tmp_path):
    """A scored row itself activates the gate even if diagnostics were deleted."""
    payload = _payload(tmp_path, _metrics())
    (payload / "reports/snqi_v2_diagnostics.json").unlink()
    result = _publication_snqi_evidence(payload)
    assert result["checked"] is True
    assert any("diagnostics" in reason for reason in result["violations"])


@pytest.mark.parametrize("score", [1.0, -999.0])
def test_publication_preflight_enforces_v2_numerical_result(tmp_path, score):
    """The public admission gate accepts a valid score and refuses the audited bypass."""
    import hashlib

    from robot_sf.benchmark.artifact_publication import (
        PublicationPreflightError,
        verify_publication_bundle_preflight,
    )

    payload = _payload(tmp_path / "payload", _metrics(score=score))
    row_path = payload / "runs/goal/episodes.jsonl"
    row = json.loads(row_path.read_text())
    row["event_ledger"] = {"software_commit": "abc123"}
    row_path.write_text(json.dumps(row) + "\n")
    entries = [
        {
            "path": p.relative_to(payload).as_posix(),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in payload.rglob("*")
        if p.is_file()
    ]
    (tmp_path / "publication_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "benchmark-publication-bundle.v1",
                "files": entries,
                "provenance": {"repository": {"commit": "abc123"}},
                "publication_channels": {
                    "doi": "10.5281/zenodo.1234567",
                    "release_tag": "test-v2",
                    "release_url": "https://github.com/ll7/robot_sf_ll7/releases/tag/test-v2",
                },
            }
        )
    )
    (tmp_path / "checksums.sha256").write_text(
        "".join(f"{e['sha256']}  payload/{e['path']}\n" for e in entries)
    )
    if score == 1:
        report = verify_publication_bundle_preflight(tmp_path, require_release_reconciliation=False)
        assert report["evidence"]["snqi_field_consistency"]["checked"] is True
    else:
        with pytest.raises(PublicationPreflightError, match="SNQI-v2 score violates"):
            verify_publication_bundle_preflight(tmp_path, require_release_reconciliation=False)


def test_v2_campaign_declaration_cannot_be_stripped_of_all_scores(tmp_path):
    """The campaign's v2 contract activates admission even if score rows and reports are stripped."""
    payload = _payload(tmp_path, _metrics())
    (payload / "reports/snqi_v2_diagnostics.json").unlink()
    (payload / "runs/goal/episodes.jsonl").write_text(
        json.dumps({"metrics": {"success": True}}) + "\n"
    )
    (payload / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "legacy_snqi": "excluded",
                "metrics": {"snqi_v2_version": "SNQI-v2"},
            }
        )
    )
    result = _publication_snqi_evidence(payload)
    assert result["checked"] is True
    assert result["invalid_rows"] == 1
    assert result["violations"]
