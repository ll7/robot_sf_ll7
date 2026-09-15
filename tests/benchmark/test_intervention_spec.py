"""Focused tests for the diagnostic-only issue #9308 intervention contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import TYPE_CHECKING

import pytest
import yaml

from robot_sf.benchmark.intervention_spec import (
    CLAIM_BOUNDARY,
    EVIDENCE_TIER,
    INTERVENTION_SPEC_ISSUE,
    INTERVENTION_SPEC_SCHEMA_VERSION,
    STOP_CONDITIONS,
    InterventionSpecValidationError,
    compute_intervention_spec_digest,
    load_intervention_spec,
    load_intervention_spec_schema,
    validate_intervention_spec,
)

if TYPE_CHECKING:
    from pathlib import Path

_BASE_COMMIT = "33823ae01264181acd0bdbff4e4d86c00244a9be"
_DIGEST = "a" * 64


def _payload() -> dict[str, object]:
    """Return a valid matched-start stage-1 payload."""

    return {
        "schema_version": INTERVENTION_SPEC_SCHEMA_VERSION,
        "spec_id": "issue-9308-visibility-001",
        "issue": INTERVENTION_SPEC_ISSUE,
        "evidence_tier": EVIDENCE_TIER,
        "claim_boundary": CLAIM_BOUNDARY,
        "status": "specification_only",
        "hypothesis": {
            "mechanism": "occlusion_exposure",
            "statement": "Reduced visibility delays the planner response at the selected near miss.",
        },
        "factor": {
            "name": "visibility",
            "path": "observation.visibility",
            "unit": "category",
            "baseline": "occluded",
            "intervention": "visible",
        },
        "held_fixed": ["initial_state", "planner_id", "scenario_id", "seed"],
        "known_unfixable": ["pedestrian_response"],
        "comparison": {
            "classification": "matched_start_replay",
            "match_basis": ["initial_state", "scenario_id", "seed"],
            "required_shared_prefix_steps": 0,
            "verification_status": "not_verified",
        },
        "negative_control": {
            "id": "visibility-no-op",
            "factor_path": "observation.visibility",
            "value": "occluded",
            "expected": "no_factor_activation",
            "rationale": "A no-op arm checks that the harness does not activate the selected factor.",
        },
        "stop_rule": {
            "action": "stop",
            "outcome": "not_available",
            "conditions": list(STOP_CONDITIONS),
            "no_substitution": True,
        },
        "provenance": {
            "claim_boundary": CLAIM_BOUNDARY,
            "execution_status": "not_executed",
            "source_identity": {
                "scenario_id": "classic_doorway_medium",
                "planner_id": "ppo",
                "episode_id": "classic_doorway_medium--113--fixture",
                "seed": 113,
                "source_kind": "existing_diagnostic_trace_or_dossier",
                "source_refs": [
                    {
                        "path": "docs/case_workbench.md",
                        "sha256": _DIGEST,
                        "role": "case_dossier",
                    }
                ],
            },
            "config_identity": {
                "config_id": "issue-9308-local-stage-1",
                "path": "configs/analysis/case_workbench.v1.yaml",
                "sha256": _DIGEST,
            },
            "contract_identity": {
                "owner": "robot_sf.benchmark.intervention_spec",
                "schema_version": INTERVENTION_SPEC_SCHEMA_VERSION,
                "base_commit": _BASE_COMMIT,
            },
        },
    }


def _init_git_checkout(repo_root: Path) -> str:
    """Create a minimal local Git checkout and return its committed HEAD."""

    subprocess.run(["git", "init", "--quiet", str(repo_root)], check=True)
    subprocess.run(
        ["git", "-C", str(repo_root), "config", "user.email", "tests@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_root), "config", "user.name", "Robot SF tests"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo_root), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo_root), "commit", "--quiet", "-m", "fixture"],
        check=True,
    )
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _bound_payload(
    repo_root: Path, base_commit: str, *, source_path: str = "source.json"
) -> dict[str, object]:
    """Return a payload bound to the fixture source and config files."""

    source = repo_root / source_path
    config = repo_root / "config.yaml"
    payload = _payload()
    payload["provenance"]["source_identity"]["source_refs"] = [
        {
            "path": source_path,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "role": "mechanism_trace",
        }
    ]
    payload["provenance"]["config_identity"].update(
        {"path": "config.yaml", "sha256": hashlib.sha256(config.read_bytes()).hexdigest()}
    )
    payload["provenance"]["contract_identity"]["base_commit"] = base_commit
    return payload


def _write_binding_fixture(repo_root: Path, marker: str) -> None:
    """Create source/config files and a marker that differentiates fixture commits."""

    repo_root.mkdir()
    (repo_root / "source.json").write_text("source\n", encoding="utf-8")
    (repo_root / "config.yaml").write_text("config: true\n", encoding="utf-8")
    (repo_root / "marker.txt").write_text(marker, encoding="utf-8")


def test_schema_loads_and_valid_payload_is_normalized() -> None:
    """The schema is valid and declared sets have deterministic ordering."""

    schema = load_intervention_spec_schema()
    assert schema["properties"]["schema_version"]["const"] == INTERVENTION_SPEC_SCHEMA_VERSION

    payload = _payload()
    payload["held_fixed"] = ["seed", "initial_state", "scenario_id", "planner_id"]
    normalized = validate_intervention_spec(payload)

    assert normalized["held_fixed"] == ["initial_state", "planner_id", "scenario_id", "seed"]
    assert normalized["stop_rule"]["conditions"] == sorted(STOP_CONDITIONS)
    assert payload["held_fixed"] == ["seed", "initial_state", "scenario_id", "planner_id"]


def test_digest_is_stable_for_mapping_order() -> None:
    """Canonical digesting must not depend on YAML/dict insertion order."""

    first = _payload()
    second = json.loads(json.dumps(first, sort_keys=True))
    assert compute_intervention_spec_digest(first) == compute_intervention_spec_digest(second)


def test_yaml_loader_is_validation_only(tmp_path: Path) -> None:
    """Loading a spec parses and validates it without invoking any execution path."""

    path = tmp_path / "intervention.yaml"
    path.write_text(yaml.safe_dump(_payload(), sort_keys=False), encoding="utf-8")

    loaded = load_intervention_spec(path)
    assert loaded["status"] == "specification_only"
    assert loaded["provenance"]["execution_status"] == "not_executed"


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_loader_rejects_duplicate_mapping_keys(tmp_path: Path, suffix: str) -> None:
    """Duplicate input keys cannot silently replace a contract field last-wins."""

    if suffix == ".json":
        text = json.dumps(_payload())
        text = text.replace(
            '"status": "specification_only"',
            '"status": "specification_only", "status": "specification_only"',
            1,
        )
    else:
        text = yaml.safe_dump(_payload(), sort_keys=False)
        text = text.replace(
            "status: specification_only\n",
            "status: specification_only\nstatus: specification_only\n",
            1,
        )

    path = tmp_path / f"duplicate{suffix}"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(InterventionSpecValidationError, match="duplicate"):
        load_intervention_spec(path)


def test_recursive_yaml_alias_fails_with_validation_error(tmp_path: Path) -> None:
    """Recursive aliases must be rejected as validation errors, not leak RecursionError."""

    text = yaml.safe_dump(_payload(), sort_keys=False)
    assert "baseline: occluded\n" in text
    path = tmp_path / "recursive.yaml"
    path.write_text(
        text.replace("baseline: occluded\n", "baseline: &cycle {self: *cycle}\n", 1),
        encoding="utf-8",
    )

    with pytest.raises(InterventionSpecValidationError, match="recursive"):
        load_intervention_spec(path)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("factor", {"name": "visibility", "path": "observation.visibility"}),
        ("provenance", {"claim_boundary": CLAIM_BOUNDARY}),
    ],
)
def test_missing_required_sections_fail_closed(field: str, replacement: object) -> None:
    """Partial sections cannot be accepted as an implicit default."""

    payload = _payload()
    payload[field] = replacement
    with pytest.raises(InterventionSpecValidationError):
        validate_intervention_spec(payload)


def test_factor_must_change_and_must_not_be_declared_fixed() -> None:
    """The one factor cannot be a no-op or appear in a fixed/unfixable set."""

    unchanged = _payload()
    unchanged["factor"]["intervention"] = "occluded"
    with pytest.raises(InterventionSpecValidationError, match="changed value"):
        validate_intervention_spec(unchanged)

    overlap = _payload()
    overlap["held_fixed"].append("observation.visibility")
    with pytest.raises(InterventionSpecValidationError, match="factor.path"):
        validate_intervention_spec(overlap)

    missing_identity = _payload()
    missing_identity["held_fixed"] = ["scenario_id", "seed", "initial_state"]
    with pytest.raises(InterventionSpecValidationError, match="planner_id"):
        validate_intervention_spec(missing_identity)


def test_numeric_factor_values_are_finite_json_numbers() -> None:
    """Numeric factor values validate without relying on a float instance method."""

    payload = _payload()
    payload["factor"] = {
        **payload["factor"],
        "unit": "m/s",
        "baseline": 0.5,
        "intervention": 1.0,
    }
    payload["negative_control"] = {
        **payload["negative_control"],
        "value": 0.5,
    }

    normalized = validate_intervention_spec(payload)
    assert normalized["factor"]["baseline"] == 0.5

    payload["factor"]["intervention"] = float("inf")
    with pytest.raises(InterventionSpecValidationError, match="finite JSON numbers"):
        validate_intervention_spec(payload)


@pytest.mark.parametrize(("baseline", "intervention"), [(1, 1.0), (-0.0, 0.0)])
def test_numeric_factor_spellings_compare_by_json_number_value(
    baseline: int | float, intervention: int | float
) -> None:
    """Integral and floating JSON number spellings cannot disguise a no-op factor."""

    payload = _payload()
    payload["factor"] = {
        **payload["factor"],
        "unit": "m/s",
        "baseline": baseline,
        "intervention": intervention,
    }
    payload["negative_control"] = {**payload["negative_control"], "value": baseline}

    with pytest.raises(InterventionSpecValidationError, match="changed value"):
        validate_intervention_spec(payload)


def test_factor_json_type_difference_remains_observable() -> None:
    """A JSON number and string with the same text remain distinct values."""

    payload = _payload()
    payload["factor"] = {
        **payload["factor"],
        "unit": "arbitrary",
        "baseline": 1,
        "intervention": "1",
    }
    payload["negative_control"] = {**payload["negative_control"], "value": 1}

    assert validate_intervention_spec(payload)["factor"]["intervention"] == "1"


def test_negative_control_accepts_equivalent_numeric_spelling() -> None:
    """A no-op control may use a different JSON number spelling for the baseline."""

    payload = _payload()
    payload["factor"] = {
        **payload["factor"],
        "unit": "m/s",
        "baseline": 1.0,
        "intervention": 2.0,
    }
    payload["negative_control"] = {**payload["negative_control"], "value": 1}

    validate_intervention_spec(payload)


@pytest.mark.parametrize(
    "integer_path",
    [
        "factor",
        "negative_control",
        "source_identity.seed",
        "comparison.required_shared_prefix_steps",
    ],
)
def test_programmatic_huge_integer_fails_as_bounded_validation_error(integer_path: str) -> None:
    """Every schema-approved integer path must fail before canonical serialization."""

    payload = _payload()
    huge = 10**4301
    if integer_path == "factor":
        payload["factor"] = {
            **payload["factor"],
            "baseline": huge,
            "intervention": huge + 1,
        }
    elif integer_path == "negative_control":
        payload["negative_control"] = {**payload["negative_control"], "value": huge}
    elif integer_path == "source_identity.seed":
        payload["provenance"]["source_identity"]["seed"] = huge
    elif integer_path == "comparison.required_shared_prefix_steps":
        payload["comparison"] = {
            **payload["comparison"],
            "classification": "genuine_shared_prefix",
            "required_shared_prefix_steps": huge,
        }
    else:  # pragma: no cover - guarded by the parameter list
        raise AssertionError(f"unsupported integer path: {integer_path}")

    with pytest.raises(InterventionSpecValidationError, match="bounded JSON integer"):
        validate_intervention_spec(payload)
    with pytest.raises(InterventionSpecValidationError, match="bounded JSON integer"):
        compute_intervention_spec_digest(payload)


def test_reasonable_seed_and_shared_prefix_values_remain_valid() -> None:
    """The integer cap preserves ordinary wide seeds and replay step counts."""

    payload = _payload()
    payload["provenance"]["source_identity"]["seed"] = 2**63 - 1
    payload["comparison"] = {
        **payload["comparison"],
        "classification": "genuine_shared_prefix",
        "required_shared_prefix_steps": 2**32,
    }

    normalized = validate_intervention_spec(payload)

    assert normalized["provenance"]["source_identity"]["seed"] == 2**63 - 1
    assert normalized["comparison"]["required_shared_prefix_steps"] == 2**32
    assert len(compute_intervention_spec_digest(payload)) == 64


@pytest.mark.parametrize(
    ("identity_group", "field", "value"),
    [
        ("source_identity", "scenario_id", "N/A"),
        ("source_identity", "planner_id", "fallback"),
        ("source_identity", "episode_id", "unknown_id"),
        ("config_identity", "config_id", "degraded"),
    ],
)
def test_placeholder_identities_are_rejected(identity_group: str, field: str, value: str) -> None:
    """Fallback and unavailable identifiers cannot pass source admission."""

    payload = _payload()
    payload["provenance"][identity_group][field] = value

    with pytest.raises(InterventionSpecValidationError, match="fallback identity"):
        validate_intervention_spec(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [("spec_id", "fallback"), ("negative_control.id", "unknown_id")],
)
def test_contract_identifiers_reject_placeholder_identities(field: str, value: str) -> None:
    """Top-level and negative-control IDs use the same fallback sentinel policy."""

    payload = _payload()
    if field == "spec_id":
        payload["spec_id"] = value
    else:
        payload["negative_control"]["id"] = value

    with pytest.raises(InterventionSpecValidationError, match="fallback identity"):
        validate_intervention_spec(payload)


def test_commit_identity_rejects_zero_and_missing_commits(tmp_path: Path) -> None:
    """A local binding requires a real Git worktree and an existing commit object."""

    zero = _payload()
    zero["provenance"]["contract_identity"]["base_commit"] = "0" * 40
    with pytest.raises(InterventionSpecValidationError, match="all-zero"):
        validate_intervention_spec(zero)

    non_git = tmp_path / "non-git"
    non_git.mkdir()
    with pytest.raises(InterventionSpecValidationError, match="Git worktree"):
        validate_intervention_spec(_payload(), repo_root=non_git)

    missing = tmp_path / "missing-commit"
    missing.mkdir()
    (missing / "marker.txt").write_text("fixture\n", encoding="utf-8")
    _init_git_checkout(missing)
    payload = _payload()
    payload["provenance"]["contract_identity"]["base_commit"] = "1" * 40
    with pytest.raises(InterventionSpecValidationError, match="not a commit in repo_root"):
        validate_intervention_spec(payload, repo_root=missing)


@pytest.mark.parametrize("identity_group", ["source_identity", "config_identity"])
@pytest.mark.parametrize("with_root", [False, True])
def test_embedded_nul_paths_fail_closed(
    tmp_path: Path, identity_group: str, with_root: bool
) -> None:
    """Embedded NUL path bytes are rejected before any Path or Git binding call."""

    payload = _payload()
    if identity_group == "source_identity":
        payload["provenance"][identity_group]["source_refs"][0]["path"] = "safe\x00name"
    else:
        payload["provenance"][identity_group]["path"] = "safe\x00name"
    root = tmp_path if with_root else None

    with pytest.raises(InterventionSpecValidationError, match="embedded NUL"):
        validate_intervention_spec(payload, repo_root=root)


def test_comparison_classification_carries_no_unverified_shared_prefix() -> None:
    """Shared-prefix wording is a declared design, never an observed result."""

    genuine = _payload()
    genuine["comparison"].update(
        {
            "classification": "genuine_shared_prefix",
            "required_shared_prefix_steps": 3,
        }
    )
    normalized = validate_intervention_spec(genuine)
    assert normalized["comparison"]["verification_status"] == "not_verified"

    invalid = _payload()
    invalid["comparison"]["classification"] = "genuine_shared_prefix"
    with pytest.raises(InterventionSpecValidationError, match="at least one"):
        validate_intervention_spec(invalid)


def test_negative_control_is_explicit_no_op_for_the_selected_factor() -> None:
    """A control that changes the factor or expected behavior is rejected."""

    wrong_value = _payload()
    wrong_value["negative_control"]["value"] = "visible"
    with pytest.raises(InterventionSpecValidationError, match="factor.baseline"):
        validate_intervention_spec(wrong_value)

    wrong_expectation = _payload()
    wrong_expectation["negative_control"]["expected"] = "no_outcome_change"
    with pytest.raises(InterventionSpecValidationError):
        validate_intervention_spec(wrong_expectation)


def test_stop_rule_is_fixed_and_complete() -> None:
    """The contract has no adaptive substitute for a failed guard."""

    payload = _payload()
    payload["stop_rule"]["conditions"] = list(STOP_CONDITIONS[:-1])
    with pytest.raises(InterventionSpecValidationError, match="exactly"):
        validate_intervention_spec(payload)


def test_bound_files_require_matching_hashes(tmp_path: Path) -> None:
    """Local binding verifies the declared config and current source bytes."""

    source = tmp_path / "source.json"
    config = tmp_path / "config.yaml"
    source.write_text("source\n", encoding="utf-8")
    config.write_text("config: true\n", encoding="utf-8")

    payload = _payload()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    payload["provenance"]["source_identity"]["source_refs"] = [
        {"path": "source.json", "sha256": source_hash, "role": "mechanism_trace"}
    ]
    payload["provenance"]["config_identity"].update({"path": "config.yaml", "sha256": config_hash})
    payload["provenance"]["contract_identity"]["base_commit"] = _init_git_checkout(tmp_path)

    assert validate_intervention_spec(payload, repo_root=tmp_path)["spec_id"] == payload["spec_id"]
    source.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(InterventionSpecValidationError, match="does not match source bytes"):
        validate_intervention_spec(payload, repo_root=tmp_path)


def test_bound_files_require_historical_blob_match(tmp_path: Path) -> None:
    """A current file cannot pass by changing its declared SHA-256 after the base commit."""

    source = tmp_path / "source.json"
    config = tmp_path / "config.yaml"
    source.write_text("source\n", encoding="utf-8")
    config.write_text("config: true\n", encoding="utf-8")
    base_commit = _init_git_checkout(tmp_path)

    payload = _payload()
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    payload["provenance"]["source_identity"]["source_refs"] = [
        {
            "path": "source.json",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "role": "mechanism_trace",
        }
    ]
    payload["provenance"]["config_identity"].update({"path": "config.yaml", "sha256": config_hash})
    payload["provenance"]["contract_identity"]["base_commit"] = base_commit

    source.write_text("tampered\n", encoding="utf-8")
    payload["provenance"]["source_identity"]["source_refs"][0]["sha256"] = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()

    with pytest.raises(InterventionSpecValidationError, match="do not match the base_commit"):
        validate_intervention_spec(payload, repo_root=tmp_path)


def test_bound_files_require_tracked_historical_blobs(tmp_path: Path) -> None:
    """A current untracked file cannot satisfy a source binding."""

    config = tmp_path / "config.yaml"
    marker = tmp_path / "tracked.txt"
    config.write_text("config: true\n", encoding="utf-8")
    marker.write_text("tracked\n", encoding="utf-8")
    base_commit = _init_git_checkout(tmp_path)

    source = tmp_path / "untracked.json"
    source.write_text("source\n", encoding="utf-8")
    payload = _payload()
    payload["provenance"]["source_identity"]["source_refs"] = [
        {
            "path": "untracked.json",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "role": "mechanism_trace",
        }
    ]
    payload["provenance"]["config_identity"].update(
        {"path": "config.yaml", "sha256": hashlib.sha256(config.read_bytes()).hexdigest()}
    )
    payload["provenance"]["contract_identity"]["base_commit"] = base_commit

    with pytest.raises(InterventionSpecValidationError, match="not tracked at base_commit"):
        validate_intervention_spec(payload, repo_root=tmp_path)


def test_bound_files_ignore_inherited_git_repository_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ambient Git directory variables cannot redirect a local provenance probe."""

    target = tmp_path / "target"
    unrelated = tmp_path / "unrelated"
    _write_binding_fixture(target, "target\n")
    _write_binding_fixture(unrelated, "unrelated\n")
    _init_git_checkout(target)
    unrelated_commit = _init_git_checkout(unrelated)

    payload = _bound_payload(target, unrelated_commit)
    monkeypatch.setenv("GIT_DIR", str(unrelated / ".git"))
    monkeypatch.setenv("GIT_COMMON_DIR", str(unrelated / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(target))

    with pytest.raises(InterventionSpecValidationError, match="not a commit in repo_root"):
        validate_intervention_spec(payload, repo_root=target)


def test_bound_files_ignore_inherited_alternate_object_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An inherited alternate object store cannot supply a foreign base commit."""

    target = tmp_path / "target"
    unrelated = tmp_path / "unrelated"
    _write_binding_fixture(target, "target\n")
    _write_binding_fixture(unrelated, "unrelated\n")
    _init_git_checkout(target)
    unrelated_commit = _init_git_checkout(unrelated)

    payload = _bound_payload(target, unrelated_commit)
    monkeypatch.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", str(unrelated / ".git" / "objects"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(unrelated / ".git" / "objects"))

    with pytest.raises(InterventionSpecValidationError, match="not a commit in repo_root"):
        validate_intervention_spec(payload, repo_root=target)


def test_bound_files_disable_git_replacement_refs(tmp_path: Path) -> None:
    """A replacement ref cannot make changed current bytes look historical."""

    source = tmp_path / "source.json"
    config = tmp_path / "config.yaml"
    source.write_text("source\n", encoding="utf-8")
    config.write_text("config: true\n", encoding="utf-8")
    base_commit = _init_git_checkout(tmp_path)

    old_blob = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", f"{base_commit}:source.json"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    replacement_blob = (
        subprocess.run(
            ["git", "-C", str(tmp_path), "hash-object", "-w", "--stdin"],
            check=True,
            input=b"replacement\n",
            capture_output=True,
        )
        .stdout.strip()
        .decode("ascii")
    )
    subprocess.run(["git", "-C", str(tmp_path), "replace", old_blob, replacement_blob], check=True)

    source.write_text("replacement\n", encoding="utf-8")
    payload = _bound_payload(tmp_path, base_commit)

    with pytest.raises(InterventionSpecValidationError, match="do not match the base_commit"):
        validate_intervention_spec(payload, repo_root=tmp_path)


def test_bound_files_reject_internal_symlink_path_components(tmp_path: Path) -> None:
    """A symlinked parent cannot redirect a path declared as a repository file."""

    alias = tmp_path / "alias"
    alias.mkdir()
    (alias / "source.json").write_text("source\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("config: true\n", encoding="utf-8")
    base_commit = _init_git_checkout(tmp_path)

    real = tmp_path / "real"
    real.mkdir()
    (real / "source.json").write_text("source\n", encoding="utf-8")
    (alias / "source.json").unlink()
    alias.rmdir()
    alias.symlink_to(real, target_is_directory=True)
    payload = _bound_payload(tmp_path, base_commit, source_path="alias/source.json")

    with pytest.raises(InterventionSpecValidationError, match="symlink component"):
        validate_intervention_spec(payload, repo_root=tmp_path)
