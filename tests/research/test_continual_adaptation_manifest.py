"""Tests for the PPO continual-adaptation manifest generator (issue #6656).

The generator is a metadata-only helper that wires BEHIND the merged protocol
validator (:mod:`robot_sf.research.continual_adaptation_protocol`) and its
``continual_adaptation_run.v1`` schema. It must assemble a manifest that validates
with ``protocol_status='valid'`` and an initial ``promotion_decision='experimental'``
for the reviewed SB3 PPO backend, and it must never launch training, write a
checkpoint, mutate the safety wrapper, run an evaluation, or promote a policy.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
import yaml

from robot_sf.research import continual_adaptation_manifest as cam
from robot_sf.research.continual_adaptation_protocol import (
    CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY,
    CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION,
    PROTOCOL_STATUS_VALID,
    ContinualAdaptationProtocolError,
    check_continual_adaptation_run,
    load_continual_adaptation_run,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_MANIFEST_PATH = (
    REPO_ROOT / "configs" / "training" / "continual_adaptation_run_issue_6655.yaml"
)

# Literal dotted-prefix vocabulary the schema permits (no wildcards/pattern syntax).
_LITERAL_PREFIX_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*\.?$")
_WILDCARD_CHARACTERS = set("*?[]/{}|\\")


def test_builder_manifest_is_valid_experimental_and_not_promotable() -> None:
    """The assembled PPO manifest validates as 'experimental', never 'promote'."""
    manifest = cam.build_ppo_continual_adaptation_manifest()
    report = check_continual_adaptation_run(manifest)
    assert report.protocol_status == PROTOCOL_STATUS_VALID
    assert report.blockers == []
    assert report.promotion_decision == "experimental"
    assert report.promotion_ready is False
    assert report.evidence_boundary == CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY
    assert report.derived_adapted_policy_identifier != report.baseline_policy_identifier
    assert report.safety_wrapper_mutation_permitted is False
    assert report.experience_budget_bounded is True
    assert report.adaptation_evaluation_disjoint is True


def test_builder_declares_reviewed_ppo_backend_identity() -> None:
    """The manifest pins the reviewed PPO baseline and immutable safety wrapper."""
    manifest = cam.build_ppo_continual_adaptation_manifest()
    assert manifest["schema_version"] == CONTINUAL_ADAPTATION_RUN_SCHEMA_VERSION
    assert manifest["baseline_policy"]["identifier"] == cam.PPO_BASELINE_IDENTIFIER
    assert manifest["baseline_policy"]["identifier"] == "ppo_ammv_baseline_v3"
    assert manifest["safety_wrapper"]["identifier"] == cam.PPO_SAFETY_WRAPPER_IDENTIFIER
    assert manifest["safety_wrapper"]["identifier"] == "robot_sf.gym_env.safety_vel_controller"
    assert manifest["safety_wrapper"]["mutation_permitted"] is False
    assert manifest["baseline_policy"]["checksum"]["algorithm"] == "sha256"
    assert manifest["safety_wrapper"]["checksum"]["algorithm"] == "sha256"


def test_builder_mutable_parameters_are_literal_and_disjoint_from_wrapper() -> None:
    """Mutable parameter declarations are literal prefixes, never wildcards."""
    manifest = cam.build_ppo_continual_adaptation_manifest()
    allowed = manifest["adaptation"]["allowed_parameters"]
    assert allowed, "at least one mutable parameter prefix is required"
    wrapper_namespace = manifest["safety_wrapper"]["identifier"].rstrip(".")
    for prefix in allowed:
        assert isinstance(prefix, str) and prefix
        assert _LITERAL_PREFIX_PATTERN.fullmatch(prefix), f"non-literal prefix {prefix!r}"
        assert not (_WILDCARD_CHARACTERS & set(prefix)), f"wildcard in prefix {prefix!r}"
        normalized = prefix.rstrip(".")
        assert normalized != wrapper_namespace
        assert not wrapper_namespace.startswith(f"{normalized}.")
        assert not normalized.startswith(f"{wrapper_namespace}.")


def test_builder_budget_is_bounded_finite_and_positive() -> None:
    """The declared experience budget is bounded with a finite positive step count."""
    manifest = cam.build_ppo_continual_adaptation_manifest()
    budget = manifest["adaptation"]["experience_budget"]
    assert budget["bounded"] is True
    assert isinstance(budget["steps"], int) and not isinstance(budget["steps"], bool)
    assert budget["steps"] > 0
    assert math.isfinite(budget["steps"])
    assert budget["units"]


def test_builder_scenarios_are_disjoint_and_nonempty() -> None:
    """Adaptation and evaluation scenario IDs are non-empty and disjoint."""
    manifest = cam.build_ppo_continual_adaptation_manifest()
    adaptation = manifest["scenarios"]["adaptation"]
    evaluation = manifest["scenarios"]["evaluation"]
    assert adaptation
    assert evaluation
    assert set(adaptation).isdisjoint(evaluation)


def test_builder_declares_at_least_one_synthetic_shift() -> None:
    """At least one well-formed synthetic shift is declared for revalidation."""
    manifest = cam.build_ppo_continual_adaptation_manifest()
    shifts = manifest["shifts"]
    assert len(shifts) >= 1
    for shift in shifts:
        assert shift["id"]
        assert shift["kind"] in {"friction", "payload", "latency", "pedestrian", "other"}
        assert shift["description"]


def test_builder_thresholds_are_finite() -> None:
    """Nominal/shift/forgetting threshold bounds are finite numbers."""
    manifest = cam.build_ppo_continual_adaptation_manifest()
    thresholds = manifest["thresholds"]
    assert set(thresholds) == {"nominal", "shift", "forgetting"}
    for threshold in thresholds.values():
        bound = threshold["bound"]
        assert isinstance(bound, int | float) and not isinstance(bound, bool)
        assert math.isfinite(bound)
        assert threshold["metric"]
        assert threshold["direction"] in {"at_most", "at_least"}


def test_builder_returns_independent_copies() -> None:
    """Repeated builds do not share mutable nested state."""
    first = cam.build_ppo_continual_adaptation_manifest()
    second = cam.build_ppo_continual_adaptation_manifest()
    first["shifts"][0]["parameters"]["friction_coefficient"] = 0.9
    first["thresholds"]["nominal"]["bound"] = -0.5
    assert second["shifts"][0]["parameters"]["friction_coefficient"] == 0.4
    assert second["thresholds"]["nominal"]["bound"] == -0.02


def test_writer_roundtrip_validates(tmp_path: Path) -> None:
    """A written manifest loads and validates as 'experimental'."""
    out = tmp_path / "manifest.yaml"
    written = cam.write_ppo_continual_adaptation_manifest(out)
    assert written == out
    loaded = load_continual_adaptation_run(out)
    report = check_continual_adaptation_run(loaded, source=out)
    assert report.protocol_status == PROTOCOL_STATUS_VALID
    assert report.promotion_decision == "experimental"
    assert report.blockers == []


def test_writer_fails_closed_on_invalid_digest(tmp_path: Path) -> None:
    """A non-hex baseline digest fails schema validation and writes nothing."""
    out = tmp_path / "manifest.yaml"
    with pytest.raises(ContinualAdaptationProtocolError):
        cam.write_ppo_continual_adaptation_manifest(out, baseline_checksum_digest="not-hex")
    assert not out.exists()


def test_writer_fails_closed_on_semantic_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A schema-valid but semantic-invalid manifest fails closed and writes nothing."""
    bad = cam.build_ppo_continual_adaptation_manifest()
    bad["safety_wrapper"]["mutation_permitted"] = True
    monkeypatch.setattr(cam, "build_ppo_continual_adaptation_manifest", lambda **_: bad)
    out = tmp_path / "manifest.yaml"
    with pytest.raises(ContinualAdaptationProtocolError, match="safety wrapper"):
        cam.write_ppo_continual_adaptation_manifest(out)
    assert not out.exists()


def test_shipped_manifest_is_valid_experimental_and_not_promotable() -> None:
    """The shipped PPO manifest validates as 'experimental', not 'promote'."""
    manifest = load_continual_adaptation_run(SHIPPED_MANIFEST_PATH)
    report = check_continual_adaptation_run(manifest, source=SHIPPED_MANIFEST_PATH)
    assert report.protocol_status == PROTOCOL_STATUS_VALID
    assert report.blockers == []
    assert report.promotion_decision == "experimental"
    assert report.promotion_ready is False
    assert report.derived_adapted_policy_identifier != report.baseline_policy_identifier
    assert manifest["promotion_decision"]["decision"] == "experimental"


def test_shipped_manifest_matches_builder_recipe() -> None:
    """The shipped artifact is exactly the helper's output (reproducibility link)."""
    shipped = yaml.safe_load(SHIPPED_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert shipped == cam.build_ppo_continual_adaptation_manifest()
