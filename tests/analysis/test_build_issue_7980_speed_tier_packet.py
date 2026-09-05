"""Contract tests for the issue #7980 source-bound diagnostic speed-tier packet."""

# evidence-writer-exempt: authenticated-receipt tests write only throwaway pytest tmp_path
# fixtures, including exact gzip bytes and deliberately malformed inputs; no repository evidence
# is emitted by this test module.

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.result_interpretation_packet import (
    compute_packet_digest,
    load_result_interpretation_packet,
    write_deterministic_json,
)
from scripts.analysis import build_issue_7980_speed_tier_packet as builder
from scripts.analysis.build_issue_7980_speed_tier_packet import (
    _canonical_manifest_digest,
    _review_sidecar_path,
    _review_sidecar_payload,
    _validate_source_receipt,
    _validate_synthesis,
    build_packet,
    decode_source_binding,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "docs/context/evidence/issue_6102_robot_speed_tier_recovery"
PACKET_PATH = EVIDENCE_DIR / "result_interpretation_packet.issue_7980.v1.json"
RECOVERY_MANIFEST_PATH = EVIDENCE_DIR / "recovery_manifest.json"
PREREGISTRATION_PATH = (
    REPO_ROOT / "configs/benchmarks/issue_5578_robot_speed_tier_preregistration.yaml"
)
EXPECTED_PACKET_DIGEST = "b08b5e58cef49d3c847623542d4fb13701a9fcb6649e024708d70f8df7b7d1fc"
EXPECTED_ROW_DIGESTS = (
    "c204a1741a2d4bf77a1e757eb9614d77b6f721794bd51182bc34d922c2c48858",
    "35a90600f347363b74cf8c98fe8022a2b5b439cf3aea5e69f6a4c8e11707f85f",
    "4caed594079a9d111ce475c642a9f5c759fbacf5f48268a1f5bf1ba01fe8e76d",
    "7fcc6319de3394b8e1ba8bc46833a72d7ff1b2cda31c2b4b840b7fae26581755",
    "ac772454829b9f6928bd131c76f546a3730bda2188081f8e8f58eb805d4ed425",
    "35a15306eadb718fc6c55e1b7e15624d01e8bd1f2e63cfcde93ec75fec735e0f",
    "9599efa384b734e4411b14a9be91c4ccc4eca14a9c006348607e9a7faa77900b",
    "dbba6cde004d69943b4bb2ea4a743e1cb37086ffcc9a5d85e9d9606faeaacf91",
    "0b565aba99e530f7f47b971811bd40e78863d894ca8e950579dfa0767e081e64",
    "d96841359c6fc858d10ccdf4fd7c844fe715c8f4b994db138c26bc8fc0af2e7e",
    "f6cb2401cc4dadc5897f67eb759373d5196061fc8f820a01a3b3fa838ff3b657",
    "0ddfa9be2a83bdcaeba2fce548d4fb6c692f87915cbd8bd70a04ad72c1c07c48",
    "f4a42c36899b803b1c5b421f6ae83b8bfc480fdace6bee31bc27933a39aa8662",
    "36779665df005ac2e67f70e03337eb6b746d358457c18cfc2f0165b71a238fb4",
    "378c8425d14c9d4d0521052205bb39ade3828200699b14f0710ae7bc2efc23b3",
    "10ffa01b2e14debbc19806ec09318bd9a96bb455da238a51a5cba8d38c136e26",
    "869836455d0912e8132b54cf80f5255f52cdd6e73c87539a4ca4cfc2666e9d58",
    "da6fc95ea79f1c275d754d120ad76e00637d90e353e9f55f942dae1136490fcf",
    "bd20370c154d6328a13b9b596cc5871322f3c5ff5ffa7a69dc93af9acc28a79c",
    "4329f4aa7ecfce521dda97029f920a95926a6a5070a185f0fbca5c221a182522",
    "451a17f611fad1fc7699cf8ebeaf5117b5b82571a7ff44b0ca0b3fba08704051",
    "88777184ab19f655e4566fe65e205722063216eeced4e48c63241cbab76a0c1a",
    "c16f79550eb1f908a68e430ed1aa16abed26e25734ff8c1e9e575192d53aa80a",
    "d9cd5779fe605a91b3a043a7c620f59fbd0647ad46012c9372d80be77006bdac",
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _packet_bindings() -> tuple[dict, list[dict]]:
    packet = _load_json(PACKET_PATH)
    bindings = [decode_source_binding(metric["sensitivity"][0]) for metric in packet["metrics"]]
    return packet, bindings


def _synthetic_synthesis(rows: list[dict]) -> dict:
    return {
        "schema_version": "robot_sf.issue_5578_speed_tier_synthesis_adapter.v1",
        "per_cell_count": 2160,
        "native_cell_count": 2160,
        "excluded_cell_count": 0,
        "all_native": True,
        "grid_complete": True,
        "evidence_status": "native_grid_synthesis_complete_provenance_unverified",
        "decision_table": rows,
    }


def _validation_inputs() -> tuple[dict, str, dict, dict]:
    _, bindings = _packet_bindings()
    rows = [copy.deepcopy(binding["canonical_decision_row"]) for binding in bindings]
    recovery = _load_json(RECOVERY_MANIFEST_PATH)
    preregistration = yaml.safe_load(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    synthesis_sha = recovery["local_artifact_sha256"]["synthesis.json"]
    return _synthetic_synthesis(rows), synthesis_sha, recovery, preregistration


def _authenticated_receipt_inputs(
    tmp_path: Path,
) -> tuple[dict, Path, str, list[dict], dict]:
    """Build one byte-backed authenticated receipt without external credentials."""

    rows = [
        {"test_id": "planner__cap_3_0__success_rate", "effect": 0.01},
        {"test_id": "planner__cap_4_0__success_rate", "effect": -0.02},
    ]
    synthesis_path = tmp_path / "synthesis.json"
    source_bytes = json.dumps(
        {"decision_table": rows}, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    synthesis_path.write_bytes(source_bytes)
    synthesis_sha256 = hashlib.sha256(source_bytes).hexdigest()

    member_path = tmp_path / "synthesis.json.gz"
    member_path.write_bytes(gzip.compress(source_bytes, compresslevel=6, mtime=0))
    member_sha256 = hashlib.sha256(member_path.read_bytes()).hexdigest()
    manifest = {
        "schema": "campaign-preservation-manifest.v1",
        "campaign_id": "fixture-authenticated-receipt",
        "generated_at": "2026-08-30T00:00:00Z",
        "source_root": str(tmp_path),
        "source_host": "fixture",
        "compression": "gzip-6",
        "stage_root": str(tmp_path),
        "files": [
            {
                "path": "synthesis.json",
                "bytes": len(source_bytes),
                "sha256": synthesis_sha256,
                "stored_path": "synthesis.json.gz",
                "stored_bytes": member_path.stat().st_size,
                "stored_sha256": member_sha256,
                "stored_md5_b64": "fixture-not-used-by-this-contract",
            }
        ],
        "totals": {
            "files": 1,
            "source_bytes": len(source_bytes),
            "stored_bytes": member_path.stat().st_size,
            "compression_ratio": round(len(source_bytes) / member_path.stat().st_size, 4),
        },
    }
    manifest["manifest_digest"] = _canonical_manifest_digest(manifest)
    manifest_path = tmp_path / "campaign_preservation_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    crosswalk = {
        "row_crosswalk": [
            {
                "test_id": row["test_id"],
                "canonical_row_sha256": hashlib.sha256(
                    json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
            for row in rows
        ]
    }
    crosswalk_path = tmp_path / "source_row_crosswalk.json"
    crosswalk_path.write_text(json.dumps(crosswalk), encoding="utf-8")

    qualified_name = "ll7/robot_sf/test-speed-tier-artifact:v0"
    identity = {
        "qualified_name": qualified_name,
        "version": "v0",
        "member": "synthesis.json.gz",
        "member_sha256": member_sha256,
        "source_member": "synthesis.json",
        "source_sha256": synthesis_sha256,
        "manifest_digest": manifest["manifest_digest"],
    }
    receipt = {
        "schema_version": "issue_7980_source_ingestion_receipt.v1",
        "source_ingestion_status": "authenticated_immutable_source_hydrated",
        "independent_of_packet": True,
        "synthesis_sha256": synthesis_sha256,
        "source_path": str(crosswalk_path),
        "source_sha256": hashlib.sha256(crosswalk_path.read_bytes()).hexdigest(),
        "source_artifact": identity,
        "immutable_hydration_receipt": {
            **identity,
            "status": "verified",
            "hydrated_member_path": str(member_path),
            "manifest_member": "campaign_preservation_manifest.json",
            "preservation_manifest_path": str(manifest_path),
        },
    }
    recovery = {
        "durable_artifact": {
            "artifact_name": qualified_name,
            "version": "v0",
            "manifest_sha256": manifest["manifest_digest"].removeprefix("sha256:"),
        }
    }
    return receipt, synthesis_path, synthesis_sha256, rows, recovery


def test_tracked_packet_loads_under_generic_v1_contract() -> None:
    """Protect the durable packet's public schema and semantic validation contract."""

    packet = load_result_interpretation_packet(PACKET_PATH)

    assert packet.packet_id == "issue_7980_robot_speed_tier_contrast_binding_diagnostic"
    assert packet.evidence.tier == "smoke_diagnostic"
    assert packet.evidence.admission_state == "diagnostic_only"
    assert len(packet.metrics) == len(packet.decisions) == 24
    assert compute_packet_digest(packet) == EXPECTED_PACKET_DIGEST
    assert "source-complete" not in PACKET_PATH.read_text(encoding="utf-8")


def test_tracked_packet_producer_commit_contains_the_exact_builder() -> None:
    """Keep producer identity resolvable without self-referencing the output commit."""

    packet = _load_json(PACKET_PATH)
    producer = packet["producer"]
    commit = producer["commit"]
    committed_blob = subprocess.run(
        ["git", "rev-parse", f"{commit}:scripts/analysis/build_issue_7980_speed_tier_packet.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    current_blob = subprocess.run(
        ["git", "hash-object", "scripts/analysis/build_issue_7980_speed_tier_packet.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert len(commit) == 40
    assert committed_blob == current_blob
    assert f"git worktree add --detach <fresh-worktree> {commit}" in producer["command"]
    assert "scripts/analysis/build_issue_7980_speed_tier_packet.py" in producer["command"]
    assert f"--producer-commit {commit}" in producer["command"]


def test_all_24_packet_rows_match_the_immutable_source_binding() -> None:
    """Prove every contrast is represented once with exact custody and statistics."""

    packet, bindings = _packet_bindings()
    recovery = _load_json(RECOVERY_MANIFEST_PATH)
    expected_source_sha = recovery["local_artifact_sha256"]["synthesis.json"]
    metrics = {metric["metric_id"]: metric for metric in packet["metrics"]}
    decisions = {decision["metric_id"]: decision for decision in packet["decisions"]}
    rows = [binding["canonical_decision_row"] for binding in bindings]

    assert len(rows) == len({row["test_id"] for row in rows}) == 24
    assert Counter(row["classification"] for row in rows) == {
        "no_material_shift": 10,
        "inconclusive": 8,
        "intervention_not_activated": 6,
    }
    assert (
        tuple(
            hashlib.sha256(
                json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for row in sorted(rows, key=lambda item: item["test_id"])
        )
        == EXPECTED_ROW_DIGESTS
    )
    for binding in bindings:
        row = binding["canonical_decision_row"]
        test_id = row["test_id"]
        metric = metrics[test_id]
        decision = decisions[test_id]
        assert binding["source_artifact"] == {
            "reason": "independent source-ingestion receipt is unavailable",
            "sha256": expected_source_sha,
            "status": "pending",
        }
        assert binding["preregistration"] == {
            "path": "configs/benchmarks/issue_5578_robot_speed_tier_preregistration.yaml",
            "schema_version": "robot_sf.issue_5578_robot_speed_tier_preregistration.v1",
        }
        assert binding["paired_denominator"] == metric["denominator"] == 180
        assert metric["support"] == metric["support_threshold"] == 180
        assert metric["effect"] == row["pooled_delta_mean"] == decision["effect"]
        assert decision["contrast_result"]["effect"] == row["pooled_delta_mean"]
        assert decision["comparator"] == {
            "reference": "cap_2_0_nominal",
            "comparison": row["speed_tier_id"],
            "direction": "comparison_minus_reference",
        }
        for field in (
            "pooled_delta_se",
            "harm_bound",
            "noninferiority_bound",
            "p_value_harm_raw",
            "p_value_harm_holm",
            "p_value_noninferiority_raw",
            "p_value_noninferiority_holm",
            "directional_family_alpha",
            "familywise_alpha",
            "intervention_activated",
        ):
            assert field in row


def test_directional_bounds_and_thresholds_remain_exact() -> None:
    """Prevent the two directional tests from being collapsed into one ambiguous interval."""

    packet, bindings = _packet_bindings()
    metrics = {metric["metric_id"]: metric for metric in packet["metrics"]}
    expected_thresholds = {
        "success_rate": -0.05,
        "collision_rate": 0.02,
        "near_miss_rate": 0.05,
    }

    for binding in bindings:
        row = binding["canonical_decision_row"]
        metric = metrics[row["test_id"]]
        bounds = {
            row["harm_bound_type"]: row["harm_bound"],
            row["noninferiority_bound_type"]: row["noninferiority_bound"],
        }
        assert set(bounds) == {"lower", "upper"}
        assert metric["uncertainty"]["ci_low"] == bounds["lower"]
        assert metric["uncertainty"]["ci_high"] == bounds["upper"]
        assert binding["harm_threshold"] == expected_thresholds[row["metric"]]
        assert metric["null_value"] == binding["harm_threshold"]
        assert metric["multiplicity"] == {
            "declared": True,
            "method": "holm_bonferroni_per_planner_directional_family",
            "n_comparisons": 6,
        }


def test_nonactivated_prediction_rows_remain_invalid() -> None:
    """Protect the key fail-closed exclusion against accidental null-effect promotion."""

    packet, bindings = _packet_bindings()
    decisions = {decision["metric_id"]: decision for decision in packet["decisions"]}
    inactive = [
        binding["canonical_decision_row"]
        for binding in bindings
        if not binding["canonical_decision_row"]["intervention_activated"]
    ]

    assert len(inactive) == 6
    assert {row["planner_id"] for row in inactive} == {"prediction_planner"}
    assert {row["classification"] for row in inactive} == {"intervention_not_activated"}
    assert {row["speed_tier_id"] for row in inactive} == {"cap_3_0", "cap_4_0"}
    assert all(decisions[row["test_id"]]["outcome"] == "invalid" for row in inactive)
    assert all(
        decisions[row["test_id"]]["outcome"] == "inconclusive"
        for row in (binding["canonical_decision_row"] for binding in bindings)
        if row["intervention_activated"]
    )


def test_source_references_match_tracked_bytes_and_nested_synthesis_digest() -> None:
    """Bind the packet to durable tracked metadata rather than ignored local hydration."""

    packet = _load_json(PACKET_PATH)
    recovery = _load_json(RECOVERY_MANIFEST_PATH)
    sources = {source["source_id"]: source for source in packet["sources"]}

    for source in sources.values():
        source_path = REPO_ROOT / source["path"]
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source["sha256"]
    assert recovery["local_artifact_sha256"]["synthesis.json"] == (
        "e6bb7a3553c623e07ef48260325cfe1e161dba71cc6c068dcb412df7062808c0"
    )


def test_generated_artifacts_have_exact_review_sidecars() -> None:
    """Bind every marker-sensitive artifact to the shared review-sidecar contract."""

    artifacts = (
        PACKET_PATH,
        EVIDENCE_DIR / "result_interpretation_caption.issue_7980.txt",
        EVIDENCE_DIR / "SHA256SUMS.issue_7980",
        EVIDENCE_DIR / "packet_digest_review.issue_7980.json",
        EVIDENCE_DIR / "source_row_crosswalk.issue_7980.fixture.json",
        EVIDENCE_DIR / "source_ingestion_receipt.issue_7980.fixture.json",
    )
    for artifact in artifacts:
        assert _load_json(_review_sidecar_path(artifact)) == _review_sidecar_payload(artifact)

    assert (
        (EVIDENCE_DIR / "result_interpretation_caption.issue_7980.txt")
        .read_text(encoding="utf-8")
        .startswith("<!-- AI-GENERATED (robot_sf#7980) - NEEDS-REVIEW -->\n")
    )
    assert (
        (EVIDENCE_DIR / "SHA256SUMS.issue_7980")
        .read_text(encoding="utf-8")
        .startswith("# AI-GENERATED NEEDS-REVIEW\n")
    )


def test_packet_digest_review_is_exact_and_non_approving() -> None:
    """Keep the independently reviewable digest artifact bounded to identity only."""

    review = _load_json(EVIDENCE_DIR / "packet_digest_review.issue_7980.json")

    assert review["packet_digest"] == EXPECTED_PACKET_DIGEST
    assert review["domain_approval"] is False
    assert review["review_status"] == ("diagnostic_only_pending_source_proof_and_domain_approval")


def test_validation_rejects_immutable_synthesis_digest_drift() -> None:
    """Ensure a different artifact member cannot inherit the reviewed custody receipt."""

    synthesis, _, recovery, preregistration = _validation_inputs()

    with pytest.raises(ValueError, match="digest does not match"):
        _validate_synthesis(
            synthesis,
            synthesis_sha256="0" * 64,
            recovery_manifest=recovery,
            preregistration=preregistration,
        )


def test_validation_rejects_duplicate_or_missing_contrasts() -> None:
    """Ensure row accounting cannot hide a missing registered contrast behind a duplicate."""

    synthesis, synthesis_sha, recovery, preregistration = _validation_inputs()
    synthesis["decision_table"][1] = copy.deepcopy(synthesis["decision_table"][0])

    with pytest.raises(ValueError, match="duplicate test IDs"):
        _validate_synthesis(
            synthesis,
            synthesis_sha256=synthesis_sha,
            recovery_manifest=recovery,
            preregistration=preregistration,
        )


def test_validation_rejects_activation_classification_disagreement() -> None:
    """Ensure an inactive speed manipulation cannot be relabeled as a null effect."""

    synthesis, synthesis_sha, recovery, preregistration = _validation_inputs()
    row = next(
        item
        for item in synthesis["decision_table"]
        if item["classification"] == "no_material_shift"
    )
    row["intervention_activated"] = False
    row["activation_diagnostics_summary"]["intervention_activated"] = False

    with pytest.raises(ValueError, match="activation state and classification disagree"):
        _validate_synthesis(
            synthesis,
            synthesis_sha256=synthesis_sha,
            recovery_manifest=recovery,
            preregistration=preregistration,
        )


def test_validation_rejects_non_integral_scenario_count() -> None:
    """Do not truncate a non-integral scenario count into the frozen six-scenario contract."""

    synthesis, synthesis_sha, recovery, preregistration = _validation_inputs()
    synthesis["decision_table"][0]["n_scenarios"] = 6.5

    with pytest.raises(ValueError, match="n_scenarios must equal"):
        _validate_synthesis(
            synthesis,
            synthesis_sha256=synthesis_sha,
            recovery_manifest=recovery,
            preregistration=preregistration,
        )


def test_validation_rejects_row_identity_mismatch() -> None:
    """Keep source row IDs bound to their planner, tier, and metric fields."""

    synthesis, synthesis_sha, recovery, preregistration = _validation_inputs()
    synthesis["decision_table"][0]["metric"] = "success_rate"

    with pytest.raises(ValueError, match="test_id must match"):
        _validate_synthesis(
            synthesis,
            synthesis_sha256=synthesis_sha,
            recovery_manifest=recovery,
            preregistration=preregistration,
        )


def test_validation_rejects_non_prediction_inactive_rows() -> None:
    """Keep the six invalid rows bound to the non-activated prediction-planner contrasts."""

    synthesis, synthesis_sha, recovery, preregistration = _validation_inputs()
    active = next(row for row in synthesis["decision_table"] if row["planner_id"] == "orca")
    inactive = next(
        row
        for row in synthesis["decision_table"]
        if row["classification"] == "intervention_not_activated"
    )
    active["classification"], inactive["classification"] = (
        inactive["classification"],
        active["classification"],
    )
    active["intervention_activated"], inactive["intervention_activated"] = (
        inactive["intervention_activated"],
        active["intervention_activated"],
    )
    active["activation_diagnostics_summary"]["intervention_activated"] = active[
        "intervention_activated"
    ]
    inactive["activation_diagnostics_summary"]["intervention_activated"] = inactive[
        "intervention_activated"
    ]

    with pytest.raises(ValueError, match="non-activated source rows must match"):
        _validate_synthesis(
            synthesis,
            synthesis_sha256=synthesis_sha,
            recovery_manifest=recovery,
            preregistration=preregistration,
        )


def test_independent_source_crosswalk_validates_rows_without_packet_reuse() -> None:
    """Require a separately supplied fixture crosswalk without upgrading evidence status."""

    synthesis, synthesis_sha, _, _ = _validation_inputs()
    receipt = _load_json(EVIDENCE_DIR / "source_ingestion_receipt.issue_7980.fixture.json")
    rows = sorted(synthesis["decision_table"], key=lambda item: item["test_id"])

    validated = _validate_source_receipt(
        receipt,
        synthesis_sha256=synthesis_sha,
        synthesis_path=PACKET_PATH,
        rows=rows,
        recovery_manifest=_load_json(RECOVERY_MANIFEST_PATH),
    )

    assert validated["source_ingestion_status"] == "fixture_verified"
    assert validated["independent_of_packet"] is True


def test_fixture_receipt_build_declares_durable_source_and_validates_packet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Build a receipt-backed packet through the generic schema validator end to end."""

    synthesis, synthesis_sha256, _, _ = _validation_inputs()
    synthesis_path = tmp_path / "synthesis.json"
    synthesis_path.write_text(json.dumps(synthesis), encoding="utf-8")
    receipt_path = EVIDENCE_DIR / "source_ingestion_receipt.issue_7980.fixture.json"
    real_sha256 = builder._sha256

    def synthetic_synthesis_sha256(path: Path) -> str:
        """Use the reviewed synthesis digest for the compact row-only test fixture."""

        if path.resolve() == synthesis_path.resolve():
            return synthesis_sha256
        return real_sha256(path)

    monkeypatch.setattr(builder, "_sha256", synthetic_synthesis_sha256)
    producer_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    packet = build_packet(
        synthesis_path=synthesis_path,
        recovery_manifest_path=RECOVERY_MANIFEST_PATH,
        previous_packet_path=EVIDENCE_DIR / "result_interpretation_packet.v1.json",
        preregistration_path=PREREGISTRATION_PATH,
        producer_commit=producer_commit,
        source_receipt_path=receipt_path,
    )
    output = tmp_path / "receipt-backed-packet.json"
    write_deterministic_json(packet, output)
    loaded = load_result_interpretation_packet(output)

    sources = {source["source_id"]: source for source in packet["sources"]}
    assert set(sources) == {"recovery_manifest", "issue_7980_source_receipt"}
    receipt_source = sources["issue_7980_source_receipt"]
    assert receipt_source["path"] == (
        "docs/context/evidence/issue_6102_robot_speed_tier_recovery/"
        "source_row_crosswalk.issue_7980.fixture.json"
    )
    assert receipt_source["sha256"] == _load_json(receipt_path)["source_sha256"]
    assert receipt_source["commit"] == producer_commit
    assert receipt_source["tracked_commit"]
    assert all("issue_7980_source_receipt" in metric["source_ids"] for metric in packet["metrics"])
    assert loaded.evidence.admission_state == "diagnostic_only"
    assert {decision.outcome for decision in loaded.decisions} == {"inconclusive", "invalid"}


def test_source_receipt_fails_closed_when_authenticated_hydration_is_unavailable() -> None:
    """Do not let a manifest digest or packet-derived rows imply immutable source custody."""

    synthesis, synthesis_sha, _, _ = _validation_inputs()
    receipt = _load_json(EVIDENCE_DIR / "source_ingestion_receipt.issue_7980.fixture.json")
    receipt["source_ingestion_status"] = "recorded_but_not_hydrated"

    with pytest.raises(ValueError, match="authenticated immutable source hydration is unavailable"):
        _validate_source_receipt(
            receipt,
            synthesis_sha256=synthesis_sha,
            synthesis_path=PACKET_PATH,
            rows=sorted(synthesis["decision_table"], key=lambda item: item["test_id"]),
            recovery_manifest=_load_json(RECOVERY_MANIFEST_PATH),
        )


def test_source_crosswalk_tampering_is_rejected() -> None:
    """A changed source row cannot inherit the independent receipt's digest."""

    synthesis, synthesis_sha, _, _ = _validation_inputs()
    receipt = _load_json(EVIDENCE_DIR / "source_ingestion_receipt.issue_7980.fixture.json")
    synthesis["decision_table"][0]["pooled_delta_mean"] += 0.001

    with pytest.raises(ValueError, match="row crosswalk does not match"):
        _validate_source_receipt(
            receipt,
            synthesis_sha256=synthesis_sha,
            synthesis_path=PACKET_PATH,
            rows=sorted(synthesis["decision_table"], key=lambda item: item["test_id"]),
            recovery_manifest=_load_json(RECOVERY_MANIFEST_PATH),
        )


def test_packet_reuse_as_source_path_is_rejected() -> None:
    """A receipt cannot turn the successor packet into independent source evidence."""

    synthesis, synthesis_sha, _, _ = _validation_inputs()
    receipt = _load_json(EVIDENCE_DIR / "source_ingestion_receipt.issue_7980.fixture.json")
    receipt["source_path"] = str(PACKET_PATH.relative_to(REPO_ROOT))
    receipt["source_sha256"] = hashlib.sha256(PACKET_PATH.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="must not reuse the supplied synthesis path"):
        _validate_source_receipt(
            receipt,
            synthesis_sha256=synthesis_sha,
            synthesis_path=PACKET_PATH,
            rows=sorted(synthesis["decision_table"], key=lambda item: item["test_id"]),
            recovery_manifest=_load_json(RECOVERY_MANIFEST_PATH),
        )


def test_supplied_source_path_digest_is_checked() -> None:
    """A receipt cannot claim custody for source bytes whose digest changed."""

    synthesis, synthesis_sha, _, _ = _validation_inputs()
    receipt = _load_json(EVIDENCE_DIR / "source_ingestion_receipt.issue_7980.fixture.json")
    receipt["source_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="source path digest does not match"):
        _validate_source_receipt(
            receipt,
            synthesis_sha256=synthesis_sha,
            synthesis_path=PACKET_PATH,
            rows=sorted(synthesis["decision_table"], key=lambda item: item["test_id"]),
            recovery_manifest=_load_json(RECOVERY_MANIFEST_PATH),
        )


def test_authenticated_receipt_cross_checks_artifact_member_manifest_and_rows(
    tmp_path: Path,
) -> None:
    """Accept one receipt only when every immutable identity and byte boundary agrees."""

    receipt, synthesis_path, synthesis_sha256, rows, recovery = _authenticated_receipt_inputs(
        tmp_path
    )

    validated = _validate_source_receipt(
        receipt,
        synthesis_sha256=synthesis_sha256,
        synthesis_path=synthesis_path,
        rows=rows,
        recovery_manifest=recovery,
    )

    assert validated["source_ingestion_status"] == "authenticated_immutable_source_hydrated"
    assert validated["source_artifact"]["member"] == "synthesis.json.gz"


@pytest.mark.parametrize("field", ["source_artifact", "immutable_hydration_receipt"])
def test_authenticated_receipt_rejects_empty_identity_mappings(tmp_path: Path, field: str) -> None:
    """Empty self-declarations cannot stand in for authenticated immutable custody."""

    receipt, synthesis_path, synthesis_sha256, rows, recovery = _authenticated_receipt_inputs(
        tmp_path
    )
    receipt[field] = {}

    with pytest.raises(ValueError, match="must be a non-empty mapping"):
        _validate_source_receipt(
            receipt,
            synthesis_sha256=synthesis_sha256,
            synthesis_path=synthesis_path,
            rows=rows,
            recovery_manifest=recovery,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("qualified_name", "ll7/robot_sf/wrong:v0", "qualified_name"),
        ("member", "synthesis.json", "member"),
        ("source_sha256", "0" * 64, "source_sha256"),
        ("manifest_digest", "sha256:" + "0" * 64, "manifest_digest"),
    ],
)
def test_authenticated_receipt_rejects_artifact_identity_or_digest_drift(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    """Wrong artifact, member, source, or manifest identities fail before admission wording."""

    receipt, synthesis_path, synthesis_sha256, rows, recovery = _authenticated_receipt_inputs(
        tmp_path
    )
    receipt["source_artifact"][field] = value

    with pytest.raises(ValueError, match=message):
        _validate_source_receipt(
            receipt,
            synthesis_sha256=synthesis_sha256,
            synthesis_path=synthesis_path,
            rows=rows,
            recovery_manifest=recovery,
        )


def test_authenticated_receipt_rejects_compressed_member_digest_drift(
    tmp_path: Path,
) -> None:
    """A changed compressed member cannot inherit a valid artifact/source receipt."""

    receipt, synthesis_path, synthesis_sha256, rows, recovery = _authenticated_receipt_inputs(
        tmp_path
    )
    member_path = Path(receipt["immutable_hydration_receipt"]["hydrated_member_path"])
    member_path.write_bytes(member_path.read_bytes() + b"drift")

    with pytest.raises(ValueError, match="member digest drifted"):
        _validate_source_receipt(
            receipt,
            synthesis_sha256=synthesis_sha256,
            synthesis_path=synthesis_path,
            rows=rows,
            recovery_manifest=recovery,
        )
