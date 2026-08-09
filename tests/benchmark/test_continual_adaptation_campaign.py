"""Tests for continual-adaptation campaign integration (issue #6657).

Covers evidence-bundle construction, promotion-result wiring, fail-closed
behaviour on fallback/degraded execution, and the promotion-gate round-trip
through ``check_continual_adaptation_run``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from robot_sf.benchmark.continual_adaptation_campaign import (
    CONTINUAL_ADAPTATION_EVIDENCE_SCHEMA_VERSION,
    ContinualAdaptationCampaignError,
    ContinualAdaptationEvidenceBundle,
    build_evidence_bundle_ref,
    build_promotion_results,
    build_result_reference,
    prepare_promotion_manifest,
    validate_promotion_readiness,
    verify_local_result_references,
    write_evidence_bundle,
    write_promotion_manifest,
)
from robot_sf.benchmark.continual_adaptation_campaign import (
    build_continual_adaptation_evidence as _build_continual_adaptation_evidence,
)
from robot_sf.research.continual_adaptation_protocol import (
    CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION,
    PROTOCOL_STATUS_VALID,
    check_continual_adaptation_run,
    derive_adapted_policy_identifier,
)
from scripts.benchmark import run_continual_adaptation_campaign as campaign_cli

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMOTION_FIXTURE_PATH = (
    REPO_ROOT / "configs" / "benchmark" / "continual_adaptation_promotion_fixture.yaml"
)

_BASELINE_DIGEST = "a" * 64
_WRAPPER_DIGEST = "b" * 64


def _checksum(digest: str = _BASELINE_DIGEST) -> dict:
    return {"algorithm": "sha256", "digest": digest}


def _threshold(bound: float, direction: str = "at_most") -> dict:
    return {"metric": "success_rate_delta", "bound": bound, "direction": direction}


def _build_evidence(
    manifest: dict,
    **kwargs: object,
) -> ContinualAdaptationEvidenceBundle:
    """Build evidence with exact deterministic fixture bytes."""
    options = dict(kwargs)
    options.setdefault("nominal_content", b'{"result_type":"nominal"}\n')
    options.setdefault("shift_content", b'{"result_type":"shift"}\n')
    options.setdefault("forgetting_content", b'{"result_type":"forgetting"}\n')
    return _build_continual_adaptation_evidence(manifest, **options)


def _manifest(**overrides: object) -> dict:
    """Return a minimal valid manifest with promotion_decision 'experimental'."""
    manifest = {
        "schema_version": CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION,
        "run_id": "test_campaign_run",
        "issue": 6657,
        "claim_boundary": "metadata-only campaign integration test",
        "baseline_policy": {
            "identifier": "ppo_baseline_v1",
            "checksum": _checksum(),
        },
        "safety_wrapper": {
            "identifier": "robot_sf.gym_env.safety_wrapper",
            "checksum": _checksum(_WRAPPER_DIGEST),
            "mutation_permitted": False,
        },
        "adaptation": {
            "allowed_parameters": ["policy_net.head."],
            "experience_budget": {
                "bounded": True,
                "steps": 100000,
                "units": "gradient_steps",
            },
        },
        "scenarios": {
            "adaptation": ["train_a", "train_b"],
            "evaluation": ["eval_a", "eval_b"],
        },
        "shifts": [
            {
                "id": "friction_low",
                "kind": "friction",
                "description": "lowered floor friction",
                "parameters": {"friction_coefficient": 0.4},
            }
        ],
        "thresholds": {
            "nominal": _threshold(-0.02, "at_most"),
            "shift": _threshold(0.05, "at_least"),
            "forgetting": _threshold(-0.02, "at_most"),
        },
        "promotion_decision": {
            "decision": "experimental",
            "rationale": "test fixture",
        },
    }
    manifest.update(overrides)
    return manifest


class TestBuildResultReference:
    """Tests for :func:`build_result_reference`."""

    def test_builds_reference_with_content(self) -> None:
        ref = build_result_reference("runs/nominal.json", content="test-content")
        assert ref["uri"] == "runs/nominal.json"
        assert ref["checksum"]["algorithm"] == "sha256"
        assert len(ref["checksum"]["digest"]) == 64

    def test_builds_reference_with_explicit_checksum(self) -> None:
        checksum = {"algorithm": "sha256", "digest": "c" * 64}
        ref = build_result_reference("runs/shift.json", checksum=checksum)
        assert ref["checksum"] == checksum

    def test_empty_uri_fails_closed(self) -> None:
        with pytest.raises(ContinualAdaptationCampaignError, match="non-empty"):
            build_result_reference("", content="data")

    def test_neither_content_nor_checksum_fails_closed(self) -> None:
        with pytest.raises(ContinualAdaptationCampaignError, match="exactly one"):
            build_result_reference("runs/nominal.json")

    def test_both_content_and_checksum_fails_closed(self) -> None:
        with pytest.raises(ContinualAdaptationCampaignError, match="exactly one"):
            build_result_reference(
                "runs/nominal.json",
                content="data",
                checksum={"algorithm": "sha256", "digest": "c" * 64},
            )

    @pytest.mark.parametrize(
        "checksum",
        [
            {"algorithm": "md5", "digest": "c" * 32},
            {"algorithm": "sha256", "digest": "C" * 64},
            {"algorithm": "sha256", "digest": "short"},
        ],
    )
    def test_invalid_explicit_checksum_fails_closed(self, checksum: dict[str, str]) -> None:
        """Unsupported or malformed digests cannot become evidence references."""
        with pytest.raises(ContinualAdaptationCampaignError, match="checksum|sha256"):
            build_result_reference("runs/nominal.json", checksum=checksum)


class TestBuildEvidenceBundleRef:
    """Tests for :func:`build_evidence_bundle_ref`."""

    def test_builds_complete_reference(self) -> None:
        ref = build_evidence_bundle_ref(
            identifier="evidence_v1",
            uri="evidence/bundle.yaml",
            policy_identifier="ppo_baseline_v1#continual-adaptation@sha256:abc123",
            baseline_identifier="ppo_baseline_v1",
            content="evidence-content",
        )
        assert ref["identifier"] == "evidence_v1"
        assert ref["policy_identifier"] == "ppo_baseline_v1#continual-adaptation@sha256:abc123"
        assert ref["uri"] == "evidence/bundle.yaml"
        assert ref["checksum"]["algorithm"] == "sha256"

    def test_baseline_identifier_collision_fails_closed(self) -> None:
        with pytest.raises(ContinualAdaptationCampaignError, match="must not equal"):
            build_evidence_bundle_ref(
                identifier="ppo_baseline_v1",
                uri="evidence/bundle.yaml",
                policy_identifier="adapted_id",
                baseline_identifier="ppo_baseline_v1",
                content="data",
            )

    def test_empty_identifier_fails_closed(self) -> None:
        with pytest.raises(ContinualAdaptationCampaignError, match="non-empty"):
            build_evidence_bundle_ref(
                identifier="",
                uri="evidence/bundle.yaml",
                policy_identifier="adapted_id",
                baseline_identifier="ppo_baseline_v1",
                content="data",
            )

    def test_empty_policy_identifier_fails_closed(self) -> None:
        with pytest.raises(ContinualAdaptationCampaignError, match="non-empty"):
            build_evidence_bundle_ref(
                identifier="evidence_v1",
                uri="evidence/bundle.yaml",
                policy_identifier="",
                baseline_identifier="ppo_baseline_v1",
                content="data",
            )


class TestBuildContinualAdaptationEvidence:
    """Tests for :func:`build_continual_adaptation_evidence`."""

    def test_builds_evidence_with_derived_identifier(self) -> None:
        manifest = _manifest()
        evidence = _build_evidence(
            manifest,
            nominal_uri="runs/nominal.json",
            shift_uri="runs/shift.json",
            forgetting_uri="runs/forgetting.json",
            evidence_bundle_uri="evidence/bundle.yaml",
            evidence_bundle_identifier="evidence_v1",
        )
        assert evidence.schema_version == CONTINUAL_ADAPTATION_EVIDENCE_SCHEMA_VERSION
        assert evidence.baseline_policy_identifier == "ppo_baseline_v1"
        assert evidence.derived_adapted_policy_identifier != "ppo_baseline_v1"
        assert evidence.derived_adapted_policy_identifier.startswith("ppo_baseline_v1#")
        assert evidence.execution_mode == "native"
        assert evidence.is_promotion_ready
        assert evidence.blockers == []

    def test_evidence_bundle_ref_names_derived_identifier(self) -> None:
        manifest = _manifest()
        derived = derive_adapted_policy_identifier(manifest)
        evidence = _build_evidence(
            manifest,
            nominal_uri="runs/nominal.json",
            shift_uri="runs/shift.json",
            forgetting_uri="runs/forgetting.json",
            evidence_bundle_uri="evidence/bundle.yaml",
            evidence_bundle_identifier="evidence_v1",
        )
        assert evidence.evidence_bundle_ref["policy_identifier"] == derived

    @pytest.mark.parametrize(
        "mode",
        [
            "fallback",
            "degraded",
            "failed",
            "missing",
            "duplicate",
            "provenance-invalid",
            "unknown",
            "simulated",
            "",
            "FALLBACK",
        ],
    )
    def test_every_non_native_status_fails_closed(self, mode: str) -> None:
        """Only an explicitly native execution record may feed promotion metadata."""
        manifest = _manifest()
        with pytest.raises(ContinualAdaptationCampaignError, match="allowed native record"):
            _build_evidence(
                manifest,
                nominal_uri="runs/nominal.json",
                shift_uri="runs/shift.json",
                forgetting_uri="runs/forgetting.json",
                evidence_bundle_uri="evidence/bundle.yaml",
                evidence_bundle_identifier="evidence_v1",
                execution_mode=mode,
            )

    def test_missing_exact_result_content_fails_closed(self) -> None:
        """URI strings alone cannot stand in for the bytes referenced as evidence."""
        with pytest.raises(ContinualAdaptationCampaignError, match="exactly one"):
            _build_continual_adaptation_evidence(
                _manifest(),
                nominal_uri="runs/nominal.json",
                shift_uri="runs/shift.json",
                forgetting_uri="runs/forgetting.json",
                evidence_bundle_uri="evidence/bundle.yaml",
                evidence_bundle_identifier="evidence_v1",
            )

    def test_duplicate_result_uri_fails_closed(self) -> None:
        """Distinct result roles must not silently alias one artifact."""
        with pytest.raises(ContinualAdaptationCampaignError, match="URIs must be distinct"):
            _build_evidence(
                _manifest(),
                nominal_uri="runs/shared.json",
                shift_uri="runs/shared.json",
                forgetting_uri="runs/forgetting.json",
                evidence_bundle_uri="evidence/bundle.yaml",
                evidence_bundle_identifier="evidence_v1",
            )

    def test_non_metadata_claim_boundary_fails_closed(self) -> None:
        """The approved implementation-only claim boundary remains explicit."""
        manifest = _manifest(claim_boundary="empirical adaptation run")
        with pytest.raises(ContinualAdaptationCampaignError, match="metadata-only"):
            _build_evidence(
                manifest,
                nominal_uri="runs/nominal.json",
                shift_uri="runs/shift.json",
                forgetting_uri="runs/forgetting.json",
                evidence_bundle_uri="evidence/bundle.yaml",
                evidence_bundle_identifier="evidence_v1",
            )

    def test_bundle_payload_and_reference_are_deterministic(self) -> None:
        """Identical inputs produce identical payload bytes and bundle references."""
        kwargs = {
            "nominal_uri": "runs/nominal.json",
            "shift_uri": "runs/shift.json",
            "forgetting_uri": "runs/forgetting.json",
            "evidence_bundle_uri": "evidence/bundle.yaml",
            "evidence_bundle_identifier": "evidence_v1",
        }
        first = _build_evidence(_manifest(), **kwargs)
        second = _build_evidence(_manifest(), **kwargs)

        assert first.to_payload_dict() == second.to_payload_dict()
        assert first.evidence_bundle_ref == second.evidence_bundle_ref
        assert "created_utc" not in first.to_dict()

    def test_evidence_boundary_stamped(self) -> None:
        manifest = _manifest()
        evidence = _build_evidence(
            manifest,
            nominal_uri="runs/nominal.json",
            shift_uri="runs/shift.json",
            forgetting_uri="runs/forgetting.json",
            evidence_bundle_uri="evidence/bundle.yaml",
            evidence_bundle_identifier="evidence_v1",
        )
        assert "no_benchmark_or_paper_evidence" in evidence.evidence_boundary


class TestBuildPromotionResults:
    """Tests for :func:`build_promotion_results`."""

    def test_builds_all_four_refs(self) -> None:
        manifest = _manifest()
        evidence = _build_evidence(
            manifest,
            nominal_uri="runs/nominal.json",
            shift_uri="runs/shift.json",
            forgetting_uri="runs/forgetting.json",
            evidence_bundle_uri="evidence/bundle.yaml",
            evidence_bundle_identifier="evidence_v1",
        )
        results = build_promotion_results(evidence)
        assert set(results.keys()) == {
            "nominal_result",
            "shift_result",
            "forgetting_result",
            "evidence_bundle",
        }
        for key in results:
            assert "uri" in results[key]
            assert "checksum" in results[key]


class TestPreparePromotionManifest:
    """Tests for :func:`prepare_promotion_manifest`."""

    def test_sets_promote_decision_and_results(self) -> None:
        manifest = _manifest()
        evidence = _build_evidence(
            manifest,
            nominal_uri="runs/nominal.json",
            shift_uri="runs/shift.json",
            forgetting_uri="runs/forgetting.json",
            evidence_bundle_uri="evidence/bundle.yaml",
            evidence_bundle_identifier="evidence_v1",
        )
        promoted = prepare_promotion_manifest(manifest, evidence)
        assert promoted["promotion_decision"]["decision"] == "promote"
        assert "results" in promoted
        assert "nominal_result" in promoted["results"]

    def test_original_manifest_not_mutated(self) -> None:
        manifest = _manifest()
        evidence = _build_evidence(
            manifest,
            nominal_uri="runs/nominal.json",
            shift_uri="runs/shift.json",
            forgetting_uri="runs/forgetting.json",
            evidence_bundle_uri="evidence/bundle.yaml",
            evidence_bundle_identifier="evidence_v1",
        )
        prepare_promotion_manifest(manifest, evidence)
        assert manifest["promotion_decision"]["decision"] == "experimental"
        assert "results" not in manifest


class TestPromotionGateRoundTrip:
    """Round-trip: build evidence, prepare manifest, validate promotion gate."""

    def test_promotion_gate_passes(self) -> None:
        manifest = _manifest()
        evidence = _build_evidence(
            manifest,
            nominal_uri="runs/nominal.json",
            shift_uri="runs/shift.json",
            forgetting_uri="runs/forgetting.json",
            evidence_bundle_uri="evidence/bundle.yaml",
            evidence_bundle_identifier="evidence_v1",
        )
        promoted = prepare_promotion_manifest(manifest, evidence)
        report = check_continual_adaptation_run(promoted)
        assert report.protocol_status == PROTOCOL_STATUS_VALID
        assert report.promotion_ready is True
        assert report.blockers == []

    def test_validate_promotion_readiness_confirms(self) -> None:
        manifest = _manifest()
        evidence = _build_evidence(
            manifest,
            nominal_uri="runs/nominal.json",
            shift_uri="runs/shift.json",
            forgetting_uri="runs/forgetting.json",
            evidence_bundle_uri="evidence/bundle.yaml",
            evidence_bundle_identifier="evidence_v1",
        )
        promoted = prepare_promotion_manifest(manifest, evidence)
        validation = validate_promotion_readiness(promoted)
        assert validation.is_promotion_ready
        assert validation.blockers == []


class TestPromotionFixture:
    """Tests against the committed promotion fixture YAML."""

    def test_fixture_passes_promotion_gate(self) -> None:
        from robot_sf.research.continual_adaptation_protocol import load_continual_adaptation_run

        manifest = load_continual_adaptation_run(PROMOTION_FIXTURE_PATH)
        report = check_continual_adaptation_run(manifest)
        assert report.protocol_status == PROTOCOL_STATUS_VALID
        assert report.promotion_decision == "promote"
        assert report.promotion_ready is True
        assert report.blockers == []

    def test_fixture_evidence_bundle_names_derived_identifier(self) -> None:
        from robot_sf.research.continual_adaptation_protocol import load_continual_adaptation_run

        manifest = load_continual_adaptation_run(PROMOTION_FIXTURE_PATH)
        derived = derive_adapted_policy_identifier(manifest)
        evidence_ref = manifest["results"]["evidence_bundle"]
        assert evidence_ref["policy_identifier"] == derived
        assert evidence_ref["identifier"] != manifest["baseline_policy"]["identifier"]

    def test_fixture_checksums_bind_exact_committed_files(self) -> None:
        """Every committed fixture digest binds the bytes at its declared path."""
        from robot_sf.research.continual_adaptation_protocol import load_continual_adaptation_run

        manifest = load_continual_adaptation_run(PROMOTION_FIXTURE_PATH)
        verified = verify_local_result_references(manifest, REPO_ROOT)

        assert set(verified) == {
            "nominal_result",
            "shift_result",
            "forgetting_result",
            "evidence_bundle",
        }
        for name, path in verified.items():
            expected = manifest["results"][name]["checksum"]["digest"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected

    def test_fixture_checksum_mismatch_fails_closed(self) -> None:
        """A syntactically valid but incorrect checksum is rejected explicitly."""
        from robot_sf.research.continual_adaptation_protocol import load_continual_adaptation_run

        manifest = load_continual_adaptation_run(PROMOTION_FIXTURE_PATH)
        manifest["results"]["nominal_result"]["checksum"]["digest"] = "0" * 64

        with pytest.raises(ContinualAdaptationCampaignError, match="checksum mismatch"):
            verify_local_result_references(manifest, REPO_ROOT)

    def test_fixture_path_escape_fails_closed(self) -> None:
        """Local verification never dereferences an artifact outside its root."""
        from robot_sf.research.continual_adaptation_protocol import load_continual_adaptation_run

        manifest = load_continual_adaptation_run(PROMOTION_FIXTURE_PATH)
        manifest["results"]["nominal_result"]["uri"] = "../outside.json"

        with pytest.raises(ContinualAdaptationCampaignError, match="safe repository-relative"):
            verify_local_result_references(manifest, REPO_ROOT)


class TestWriteEvidenceBundle:
    """Tests for :func:`write_evidence_bundle` and :func:`write_promotion_manifest`."""

    def test_write_evidence_bundle(self, tmp_path: Path) -> None:
        """Written bundle bytes match the non-self-referential checksum."""
        manifest = _manifest()
        evidence = _build_evidence(
            manifest,
            nominal_uri="runs/nominal.json",
            shift_uri="runs/shift.json",
            forgetting_uri="runs/forgetting.json",
            evidence_bundle_uri="evidence/bundle.yaml",
            evidence_bundle_identifier="evidence_v1",
        )
        out_path = tmp_path / "bundle.yaml"
        path = write_evidence_bundle(evidence, out_path)
        assert path.exists()
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert loaded["schema_version"] == CONTINUAL_ADAPTATION_EVIDENCE_SCHEMA_VERSION
        assert loaded["derived_adapted_policy_identifier"] != "ppo_baseline_v1"
        assert "evidence_bundle_ref" not in loaded
        assert "created_utc" not in loaded
        expected_digest = evidence.evidence_bundle_ref["checksum"]["digest"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_digest

    def test_write_evidence_bundle_no_overwrite(self, tmp_path: Path) -> None:
        """The default writer refuses to replace an existing bundle."""
        manifest = _manifest()
        evidence = _build_evidence(
            manifest,
            nominal_uri="runs/nominal.json",
            shift_uri="runs/shift.json",
            forgetting_uri="runs/forgetting.json",
            evidence_bundle_uri="evidence/bundle.yaml",
            evidence_bundle_identifier="evidence_v1",
        )
        out_path = tmp_path / "bundle.yaml"
        write_evidence_bundle(evidence, out_path)
        with pytest.raises(ContinualAdaptationCampaignError, match="already exists"):
            write_evidence_bundle(evidence, out_path)

    def test_write_promotion_manifest(self, tmp_path: Path) -> None:
        """A prepared protocol fixture can be serialized as deterministic YAML."""
        manifest = _manifest()
        evidence = _build_evidence(
            manifest,
            nominal_uri="runs/nominal.json",
            shift_uri="runs/shift.json",
            forgetting_uri="runs/forgetting.json",
            evidence_bundle_uri="evidence/bundle.yaml",
            evidence_bundle_identifier="evidence_v1",
        )
        promoted = prepare_promotion_manifest(manifest, evidence)
        out_path = tmp_path / "promoted.yaml"
        path = write_promotion_manifest(promoted, out_path)
        assert path.exists()
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert loaded["promotion_decision"]["decision"] == "promote"

    def test_write_promotion_manifest_no_overwrite(self, tmp_path: Path) -> None:
        """The default manifest writer uses exclusive creation semantics."""
        out_path = tmp_path / "promoted.yaml"
        out_path.write_text("existing: true\n", encoding="utf-8")

        with pytest.raises(ContinualAdaptationCampaignError, match="already exists"):
            write_promotion_manifest(_manifest(), out_path)


class TestCampaignCli:
    """Tests for fail-closed ordering in the metadata-only campaign command."""

    def test_validate_rejects_protocol_valid_but_unready_manifest(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The validation command cannot report an experimental manifest as gate-ready."""
        manifest_path = tmp_path / "experimental.yaml"
        manifest_path.write_text(yaml.safe_dump(_manifest()), encoding="utf-8")
        args = SimpleNamespace(
            manifest=manifest_path,
            validate=True,
            artifact_root=tmp_path,
            evidence_out=None,
            promotion_manifest_out=None,
            execution_mode="native",
            overwrite=False,
        )
        monkeypatch.setattr(campaign_cli, "parse_args", lambda: args)

        assert campaign_cli.main() == 1

    def test_blocked_promotion_validation_writes_no_artifacts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Promotion readiness is checked before either requested output is persisted."""
        evidence_dir = tmp_path / "docs/context/evidence/issue_6657_continual_adaptation_campaign"
        evidence_dir.mkdir(parents=True)
        for name in ("nominal", "shift", "forgetting"):
            (evidence_dir / f"{name}_result.json").write_text(
                f'{{"result_type":"{name}"}}\n', encoding="utf-8"
            )
        manifest_path = tmp_path / "experimental.yaml"
        manifest_path.write_text(yaml.safe_dump(_manifest()), encoding="utf-8")
        evidence_out = evidence_dir / "evidence_bundle.yaml"
        promotion_out = tmp_path / "promoted.yaml"
        args = SimpleNamespace(
            manifest=manifest_path,
            validate=False,
            artifact_root=tmp_path,
            evidence_out=evidence_out,
            promotion_manifest_out=promotion_out,
            execution_mode="native",
            overwrite=False,
        )
        blocked = SimpleNamespace(is_promotion_ready=False, blockers=["synthetic blocker"])
        monkeypatch.setattr(campaign_cli, "parse_args", lambda: args)
        monkeypatch.setattr(campaign_cli, "validate_promotion_readiness", lambda *_a, **_k: blocked)

        assert campaign_cli.main() == 1
        assert not evidence_out.exists()
        assert not promotion_out.exists()
