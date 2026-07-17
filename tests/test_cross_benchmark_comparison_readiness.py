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
    CANARY_METRIC_ID_ROBOT_SF,
    CANARY_METRIC_ID_SOCNAVBENCH,
    CANARY_PER_SUITE_METRIC_IDS,
    CANARY_SLICE_FORBIDDEN_TOKENS,
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
    validate_canary_slice,
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


# ---------------------------------------------------------------------------
# Issue #5783 canary-slice validation tests
# ---------------------------------------------------------------------------


def test_real_canary_slice_is_concrete_and_authorized() -> None:
    """The checked-in #5783 canary slice must be authorized with no placeholder fields."""
    report = validate_canary_slice()
    assert report.ok is True
    assert report.canary_authorized is True
    assert report.status == "authorized_canary"
    assert report.policy_id
    assert report.policy_version
    assert report.algo
    assert report.algo_config
    assert report.robot_sf_scenario_id
    assert report.socnavbench_scenario_id
    assert isinstance(report.seed, int)
    assert report.external_asset_id
    assert report.limitation_flags
    assert report.per_suite_metric_ids == {
        "robot_sf": CANARY_METRIC_ID_ROBOT_SF,
        "socnavbench": CANARY_METRIC_ID_SOCNAVBENCH,
    }
    assert report.errors == []


def test_real_canary_slice_external_asset_in_registry() -> None:
    """The canary slice asset id must be a real external-data registry entry (no drift)."""
    from scripts.tools.manage_external_data import list_assets

    report = validate_canary_slice()
    registry_ids = {asset.asset_id for asset in list_assets()}
    assert report.external_asset_id in registry_ids


def test_canary_slice_rejects_unauthorized_status(tmp_path: Path) -> None:
    """A slice that is not authorized_canary must fail closed."""
    source = CAMPAIGN_MANIFEST_PATH
    target = tmp_path / source.name
    text = source.read_text(encoding="utf-8").replace(
        "status: authorized_canary", "status: blocked_prerequisite"
    )
    target.write_text(text, encoding="utf-8")
    report = validate_canary_slice(target)
    assert report.ok is False
    assert any("status" in err for err in report.errors)


def test_canary_slice_rejects_forbidden_placeholder_tokens(tmp_path: Path) -> None:
    """Any forbidden placeholder/blocked token in the slice must fail closed."""
    source = CAMPAIGN_MANIFEST_PATH
    target = tmp_path / source.name
    text = source.read_text(encoding="utf-8").replace(
        "version: tau_low_v1", "version: to_be_selected"
    )
    target.write_text(text, encoding="utf-8")
    report = validate_canary_slice(target)
    assert report.ok is False
    assert any("forbidden token" in err for err in report.errors)


def test_canary_slice_rejects_missing_block(tmp_path: Path) -> None:
    """A manifest with no canary_slice block must fail closed as missing."""
    target = tmp_path / "no_slice.yaml"
    target.write_text(
        "schema_version: cross_benchmark_policy_comparison.issue_3287.v1\n", encoding="utf-8"
    )
    report = validate_canary_slice(target)
    assert report.ok is False
    assert report.status == "missing"


def test_canary_slice_rejects_non_mapping_root(tmp_path: Path) -> None:
    """A malformed YAML root must return a fail-closed report, not raise AttributeError."""
    target = tmp_path / "non_mapping.yaml"
    target.write_text("- canary_slice\n", encoding="utf-8")
    report = validate_canary_slice(target)
    assert report.ok is False
    assert any("manifest root" in err for err in report.errors)


def test_canary_slice_rejects_non_mapping_nested_blocks(tmp_path: Path) -> None:
    """Malformed policy and scenario blocks must be reported independently."""
    target = tmp_path / "non_mapping_blocks.yaml"
    target.write_text(
        "canary_slice:\n"
        "  status: authorized_canary\n"
        "  canary_authorized: true\n"
        "  policy: []\n"
        "  scenario_mapping: scalar\n"
        "  metric_id: canary.metric\n"
        "  limitation_flags: [geometry] \n",
        encoding="utf-8",
    )
    report = validate_canary_slice(target)
    assert report.ok is False
    assert any("policy must be a mapping" in err for err in report.errors)
    assert any("scenario_mapping must be a mapping" in err for err in report.errors)


def test_main_validate_canary_slice_exit_codes(capsys: pytest.CaptureFixture) -> None:
    """CLI canary-slice validation exits 0 on the real slice and emits the OK line.

    The OK line must render the per-suite metric IDs (not the now-retired singular
    ``metric_id``), so a regression that drops back to ``metric=None`` is caught here.
    """
    assert main(["--validate-canary-slice"]) == 0
    out = capsys.readouterr().out
    assert "canary slice OK" in out
    assert "metrics=" in out
    assert CANARY_METRIC_ID_ROBOT_SF in out
    assert CANARY_METRIC_ID_SOCNAVBENCH in out
    assert "metric=None" not in out


def test_canary_slice_forbidden_tokens_cover_known_placeholders() -> None:
    """Guard against drift: the forbidden-token set covers the standing placeholder markers."""
    for token in ("tbd", "blocked_prerequisite", "to_be_selected"):
        assert token in CANARY_SLICE_FORBIDDEN_TOKENS


def test_per_suite_metric_ids_mirror_code_constants() -> None:
    """The readiness helper mirrors the code metric IDs exactly (no import coupling).

    The canary code in ``robot_sf/benchmark/socnavbench_canary.py`` is the source of truth for the
    two per-suite IDs; this readiness helper mirrors them locally so it stays import-free. If the
    code constants change, this test fails closed so the two cannot silently drift apart.
    """
    from robot_sf.benchmark.socnavbench_canary import (
        ROBOT_SF_METRIC_ID,
        SOCNAVBENCH_METRIC_ID,
    )

    assert CANARY_METRIC_ID_ROBOT_SF == ROBOT_SF_METRIC_ID
    assert CANARY_METRIC_ID_SOCNAVBENCH == SOCNAVBENCH_METRIC_ID
    assert CANARY_PER_SUITE_METRIC_IDS == {
        "robot_sf": ROBOT_SF_METRIC_ID,
        "socnavbench": SOCNAVBENCH_METRIC_ID,
    }


def test_canary_slice_rejects_legacy_singular_metric_id(tmp_path: Path) -> None:
    """The old colliding singular metric_id must fail closed, not capture one half silently."""
    source = CAMPAIGN_MANIFEST_PATH
    target = tmp_path / source.name
    text = source.read_text(encoding="utf-8")
    text = text.replace(
        "  per_suite_metric_ids:\n"
        "    robot_sf: robot_sf.path_length_ratio.distance_over_displacement\n"
        "    socnavbench: socnavbench.path_length_ratio.displacement_over_distance\n",
        "  metric_id: socnavbench.path_length_ratio\n",
    )
    target.write_text(text, encoding="utf-8")
    report = validate_canary_slice(target)
    assert report.ok is False
    assert any("per_suite_metric_ids" in err for err in report.errors)
    assert any("legacy/singular" in err for err in report.errors)


def test_canary_slice_rejects_colliding_per_suite_id(tmp_path: Path) -> None:
    """A per-suite id that does not match the code constant must fail closed."""
    source = CAMPAIGN_MANIFEST_PATH
    target = tmp_path / source.name
    text = source.read_text(encoding="utf-8").replace(
        "socnavbench.path_length_ratio.displacement_over_distance",
        "socnavbench.path_length_ratio",  # the old colliding singular id
    )
    target.write_text(text, encoding="utf-8")
    report = validate_canary_slice(target)
    assert report.ok is False
    assert any("displacement_over_distance" in err for err in report.errors)


def test_canary_slice_rejects_missing_per_suite_suite_key(tmp_path: Path) -> None:
    """Dropping one suite key (e.g. robot_sf) must fail closed."""
    source = CAMPAIGN_MANIFEST_PATH
    target = tmp_path / source.name
    text = source.read_text(encoding="utf-8")
    text = text.replace("    robot_sf: robot_sf.path_length_ratio.distance_over_displacement\n", "")
    target.write_text(text, encoding="utf-8")
    report = validate_canary_slice(target)
    assert report.ok is False
    assert any("missing suite key" in err for err in report.errors)


def test_canary_slice_rejects_duplicate_per_suite_ids(tmp_path: Path) -> None:
    """Both suites must use distinct metric ids; equating reciprocal definitions is a bug."""
    source = CAMPAIGN_MANIFEST_PATH
    target = tmp_path / source.name
    text = source.read_text(encoding="utf-8").replace(
        "    socnavbench: socnavbench.path_length_ratio.displacement_over_distance",
        "    socnavbench: robot_sf.path_length_ratio.distance_over_displacement",
    )
    target.write_text(text, encoding="utf-8")
    report = validate_canary_slice(target)
    assert report.ok is False
    assert any("distinct metric ids" in err for err in report.errors)
