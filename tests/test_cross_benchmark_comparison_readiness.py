"""Fixture tests for the cross-benchmark comparison readiness helper (#3287).

These tests exercise the presence-only classifier against synthetic repository roots so the
ready / blocked / waived logic is covered without depending on the real checkout layout, on
external benchmark assets, or on any campaign execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts.tools.cross_benchmark_comparison_readiness import (
    CAMPAIGN_MANIFEST_PATH,
    LIMITATIONS_TEMPLATE_PATH,
    PREREQUISITE_FAMILIES,
    REQUIRED_LIMITATION_SECTIONS,
    RUN_GATES,
    SOCIAL_NAV_EXTERNAL_ASSET_IDS,
    CampaignManifestError,
    WaiverError,
    _parse_waiver_args,
    evaluate_readiness,
    main,
    render_text,
    validate_campaign_manifest,
    validate_waivers,
)

if TYPE_CHECKING:
    from pathlib import Path


def _touch(repo_root: Path, rel: Path) -> None:
    """Create an empty file (and parents) at ``rel`` under ``repo_root``."""
    target = repo_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")


def _stage_local_prerequisites(repo_root: Path) -> None:
    """Materialize every locally-checkable prerequisite path.

    External-blocker families have no required paths, so they are intentionally left blocked.
    """
    for family in PREREQUISITE_FAMILIES:
        for rel in family.required_paths:
            if rel.suffix:
                _touch(repo_root, rel)
            else:
                (repo_root / rel).mkdir(parents=True, exist_ok=True)


def _family(report: dict, family_id: str) -> dict:
    """Return the report entry for ``family_id``."""
    return next(fam for fam in report["families"] if fam["id"] == family_id)


def test_local_families_ready_external_blocked(tmp_path: Path) -> None:
    """All file-backed families classify ready; the external-asset family stays blocked."""
    _stage_local_prerequisites(tmp_path)
    report = evaluate_readiness(tmp_path)

    assert _family(report, "converter")["status"] == "ready"
    assert _family(report, "metric_wrapper")["status"] == "ready"
    assert _family(report, "policy_metadata")["status"] == "ready"

    external = _family(report, "external_assets")
    assert external["status"] == "blocked"
    assert external["external_blocker"] is True
    # The external blocker keeps the whole campaign blocked even with every local file present.
    assert report["prerequisites_status"] == "blocked"


def test_missing_converter_path_reports_blocked_with_blockers(tmp_path: Path) -> None:
    """A missing converter artifact yields a blocked family that names the missing path."""
    _stage_local_prerequisites(tmp_path)
    converter = next(f for f in PREREQUISITE_FAMILIES if f.family_id == "converter")
    missing = converter.required_paths[0]
    (tmp_path / missing).unlink()

    report = evaluate_readiness(tmp_path)
    entry = _family(report, "converter")

    assert entry["status"] == "blocked"
    assert missing.as_posix() in entry["missing_paths"]
    assert report["prerequisites_status"] == "blocked"


def test_waiver_clears_external_blocker(tmp_path: Path) -> None:
    """An explicit waiver with a reason turns the external blocker into ``waived``."""
    _stage_local_prerequisites(tmp_path)
    report = evaluate_readiness(
        tmp_path, {"external_assets": "assets staged out-of-band on the cluster"}
    )

    external = _family(report, "external_assets")
    assert external["status"] == "waived"
    assert external["waiver_reason"] == "assets staged out-of-band on the cluster"
    # With every local family ready and the only external blocker waived, prerequisites clear.
    assert report["prerequisites_status"] == "ready"


def test_waiver_does_not_satisfy_a_genuinely_missing_local_path(tmp_path: Path) -> None:
    """Waiving one family does not paper over a different blocked family."""
    _stage_local_prerequisites(tmp_path)
    converter = next(f for f in PREREQUISITE_FAMILIES if f.family_id == "converter")
    (tmp_path / converter.required_paths[0]).unlink()

    report = evaluate_readiness(tmp_path, {"external_assets": "staged on the cluster"})

    assert _family(report, "converter")["status"] == "blocked"
    assert report["prerequisites_status"] == "blocked"


def test_waiver_without_reason_is_rejected() -> None:
    """A waiver must carry an explicit, non-empty reason."""
    with pytest.raises(WaiverError):
        validate_waivers({"external_assets": "   "})


def test_waiver_for_unknown_family_is_rejected() -> None:
    """Waiving a family that does not exist is a hard error, not a silent no-op."""
    with pytest.raises(WaiverError):
        validate_waivers({"not_a_family": "because"})


def test_parse_waiver_args_splits_on_first_colon() -> None:
    """Reasons may contain colons; only the first colon separates id from reason."""
    parsed = _parse_waiver_args(["external_assets:staged at 12:00 on cluster"])
    assert parsed == {"external_assets": "staged at 12:00 on cluster"}


def test_parse_waiver_args_requires_reason() -> None:
    """A bare family id with no colon is rejected so waivers stay explicit."""
    with pytest.raises(WaiverError):
        _parse_waiver_args(["external_assets"])


def test_campaign_never_authorized_even_when_prerequisites_clear(tmp_path: Path) -> None:
    """The helper is presence-only: cleared prerequisites must not imply campaign authorization."""
    _stage_local_prerequisites(tmp_path)
    report = evaluate_readiness(tmp_path, {"external_assets": "staged on the cluster"})

    assert report["prerequisites_status"] == "ready"
    assert report["campaign_authorized"] is False
    assert report["run_gates"] == list(RUN_GATES)
    assert report["run_gates"], "standing run gates must be reported"


def test_main_exit_code_tracks_prerequisites(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """main() exits 1 while blocked and 0 once prerequisites clear (presence-only signal)."""
    _stage_local_prerequisites(tmp_path)
    assert main(["--repo-root", str(tmp_path)]) == 1

    capsys.readouterr()
    exit_code = main(["--repo-root", str(tmp_path), "--waive", "external_assets:staged on cluster"])
    assert exit_code == 0


def test_main_rejects_malformed_waiver(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A malformed CLI waiver returns the dedicated error exit code (2)."""
    assert main(["--repo-root", str(tmp_path), "--waive", "external_assets"]) == 2
    # Diagnostics are written to stderr so stdout stays reserved for the report.
    assert "error:" in capsys.readouterr().err


def test_render_text_marks_status_and_gates(tmp_path: Path) -> None:
    """The text rendering surfaces per-family status and the standing run gates."""
    _stage_local_prerequisites(tmp_path)
    report = evaluate_readiness(tmp_path)
    text = render_text(report)

    assert "Cross-benchmark policy comparison readiness (#3287)" in text
    assert "Run gates" in text
    assert "BLOCKED" in text


def test_external_asset_ids_exist_in_canonical_registry() -> None:
    """The referenced external asset ids must stay valid against the canonical registry.

    This guards against drift between this helper's documented external blockers and the
    canonical external-data owner in ``scripts/tools/manage_external_data.py``.
    """
    from scripts.tools.manage_external_data import list_assets

    registry_ids = {asset.asset_id for asset in list_assets()}
    for asset_id in SOCIAL_NAV_EXTERNAL_ASSET_IDS:
        assert asset_id in registry_ids, f"{asset_id} missing from canonical external-data registry"


def test_real_checkout_reports_converter_and_metric_ready_external_blocked() -> None:
    """Sanity check against the live checkout: converter + metric wrapper present, assets not.

    This is a presence-only assertion about the current repository layout; it does not run any
    campaign. The converter prerequisite (#3285) landed on main, so it now classifies ``ready``;
    the external-asset family stays ``blocked`` because those assets are never staged in-repo.
    """
    report = evaluate_readiness()  # real REPO_ROOT
    assert _family(report, "converter")["status"] == "ready"
    assert _family(report, "metric_wrapper")["status"] == "ready"
    assert _family(report, "policy_metadata")["status"] == "ready"
    assert _family(report, "external_assets")["status"] == "blocked"


def test_issue_3287_campaign_manifest_scaffold_validates() -> None:
    """Real scaffold pins blocked status, limitations, provenance, and no equivalence claim."""
    manifest = validate_campaign_manifest()

    assert manifest["status"] == "blocked_prerequisite"
    assert manifest["campaign_authorized"] is False
    assert manifest["direct_equivalence_claim_allowed"] is False
    assert manifest["limitations_template"] == LIMITATIONS_TEMPLATE_PATH.as_posix()
    assert set(REQUIRED_LIMITATION_SECTIONS).issubset(manifest["limitations_sections"])
    assert manifest["external_asset_provenance"]


def test_issue_3287_manifest_paths_are_readiness_inputs() -> None:
    """The policy-metadata family tracks both scaffold files named by maintainer plan."""
    policy_metadata = next(
        family for family in PREREQUISITE_FAMILIES if family.family_id == "policy_metadata"
    )

    assert CAMPAIGN_MANIFEST_PATH in policy_metadata.required_paths
    assert LIMITATIONS_TEMPLATE_PATH in policy_metadata.required_paths


def test_campaign_manifest_validation_rejects_equivalence_claim(tmp_path: Path) -> None:
    """The scaffold contract fails closed if a manifest permits direct equivalence."""
    source = CAMPAIGN_MANIFEST_PATH
    target = tmp_path / source.name
    target.write_text(
        source.read_text(encoding="utf-8").replace(
            "direct_equivalence_claim_allowed: false",
            "direct_equivalence_claim_allowed: true",
        ),
        encoding="utf-8",
    )

    with pytest.raises(CampaignManifestError, match="direct_equivalence_claim_allowed"):
        validate_campaign_manifest(target)


def test_main_validate_manifest_keeps_campaign_blocked() -> None:
    """CLI manifest validation succeeds but still reports blocked campaign prerequisites."""
    assert main(["--validate-manifest"]) == 1
