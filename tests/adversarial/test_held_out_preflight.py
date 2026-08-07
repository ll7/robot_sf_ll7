"""Tests for the issue #6104 / #3275 held-out preflight packet materialization.

These tests prove the deterministic, content-addressed preflight packet:

- is fit on exactly the six frozen group-crossing/social-force records and excludes
  the five held-out cross-trap/goal records;
- generates a 64-candidate pool with identical frozen 12-per-arm budgets under the
  frozen disjoint_by_candidate policy;
- predeclares step-3 execution seeds disjoint from archive-certification seeds, the
  candidate-pool seed, and candidate scenario seeds;
- reproduces byte-for-byte and hash-for-hash from the recorded command and revision;
- never executes a planner and never reads, imports, or inspects an outcome;
- matches the main comparison runner's pool, arm membership, and candidate-manifest
  hashes, so the external v2 binding is the exact step-3 admission contract.
"""

# evidence-writer-exempt: tests write preflight packets only under pytest tmp_path;
# they do not generate or modify repository evidence artifacts.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.adversarial.config import SearchSpaceConfig
from robot_sf.adversarial.held_out_preflight import (
    EXECUTION_SEED_BASE,
    build_held_out_preflight,
    certify_structural_eligibility,
    compose_preflight_packet_files,
    execution_seeds_for_candidate,
    generate_candidate_pool,
    materialize_preflight_packet,
    verify_preflight_packet,
)
from robot_sf.adversarial.proposal_model import (
    FailureArchiveProposalModel,
    load_issue_3275_contract,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONTRACT = _REPO_ROOT / "configs/adversarial/issue_3275_same_planner_contract.json"
_PACKET_DIR = _REPO_ROOT / "docs/context/evidence/issue_3275_same_planner_held_out"


def _contract() -> dict[str, Any]:
    """Load the frozen #3275 contract."""
    return load_issue_3275_contract(_CONTRACT)


def _build() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the preflight packet and pool manifest for a test revision."""
    return build_held_out_preflight(_CONTRACT, repo_root=_REPO_ROOT, code_revision="a" * 40)


# --- Frozen fit / exclusion discipline ---------------------------------------


def test_preflight_fit_set_is_exactly_the_six_frozen_fit_ids() -> None:
    """The model sees only the six frozen group-crossing/social-force fit anchors."""
    from robot_sf.adversarial.held_out_preflight import build_held_out_preflight

    _, pool_manifest, _ = build_held_out_preflight(
        _CONTRACT, repo_root=_REPO_ROOT, code_revision="a" * 40
    )
    contract = _contract()
    model, _provenance = FailureArchiveProposalModel.from_frozen_contract(
        _CONTRACT, repo_root=_REPO_ROOT
    )
    assert model.state == "active"
    assert len(model.entries) == 6
    model_ids = {entry["archive_id"] for entry in model.entries}
    assert model_ids == set(contract["fit"]["entry_ids"])
    assert all("classic_group_crossing_medium" in archive_id for archive_id in model_ids)
    assert not any("classic_cross_trap_medium" in archive_id for archive_id in model_ids)
    assert model_ids.isdisjoint(set(contract["exclusions"]["entry_ids"]))
    # The packet records the frozen fit hash, not just the count.
    assert pool_manifest["this_issue"] == 6104


def test_preflight_rejects_fit_anchor_drift_when_contract_changes_fit_ids(
    tmp_path: Path,
) -> None:
    """The preflight derives the fit set from the frozen contract and fails on drift."""
    drifted = json.loads(_CONTRACT.read_text(encoding="utf-8"))
    drifted["fit"]["entry_ids_sha256"] = "0" * 64
    drifted_path = tmp_path / "drifted_contract.json"
    drifted_path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="fit"):
        build_held_out_preflight(drifted_path, repo_root=_REPO_ROOT, code_revision="a" * 40)


def test_all_candidates_are_structurally_eligible_under_frozen_contract() -> None:
    """Every pool candidate passes the outcome-free structural eligibility certifier."""
    contract = _contract()
    search_space = SearchSpaceConfig.from_file(
        _REPO_ROOT / contract["evaluation"]["search_space_path"]
    )
    pool = generate_candidate_pool(
        search_space,
        pool_size=contract["budget"]["candidate_pool_size"],
        pool_seed=contract["budget"]["candidate_pool_seed"],
    )
    for candidate in pool:
        verdict = certify_structural_eligibility(
            candidate,
            search_space,
            scenario_family=contract["evaluation"]["scenario_family"],
            target_planner=contract["target_planner"]["id"],
        )
        assert verdict["eligible"] is True
        assert verdict["errors"] == []
        assert verdict["family_matches"] is True
        assert verdict["planner_matches"] is True


# --- Deterministic pool, arms, budgets, duplicates ---------------------------


def test_pool_generation_is_deterministic_and_seed_sensitive() -> None:
    """Same seed -> identical pool; different seed -> different pool."""
    contract = _contract()
    search_space = SearchSpaceConfig.from_file(
        _REPO_ROOT / contract["evaluation"]["search_space_path"]
    )
    pool_seed = contract["budget"]["candidate_pool_seed"]
    pool_a = generate_candidate_pool(search_space, pool_size=64, pool_seed=pool_seed)
    pool_b = generate_candidate_pool(search_space, pool_size=64, pool_seed=pool_seed)
    assert [candidate.to_json() for candidate in pool_a] == [
        candidate.to_json() for candidate in pool_b
    ]
    pool_c = generate_candidate_pool(search_space, pool_size=64, pool_seed=pool_seed + 1)
    assert [candidate.to_json() for candidate in pool_a] != [
        candidate.to_json() for candidate in pool_c
    ]


def test_arm_budgets_equal_frozen_and_disjoint_by_candidate() -> None:
    """Proposal and random arms both equal the frozen 12 and share no candidate."""
    packet, pool_manifest, _ = _build()
    assert packet["arm_budget_equality"] == {
        "proposal": 12,
        "random": 12,
        "frozen_budget_per_arm": 12,
        "equal_and_frozen": True,
    }
    assert packet["candidate_pool"] == {
        "size": 64,
        "seed": 42,
        "budget_per_arm": 12,
        "identical_budget_both_arms": True,
    }
    assert packet["arm_overlap_policy"]["name"] == "disjoint_by_candidate"
    assert packet["arm_overlap_policy"]["overlap_ids"] == []
    proposal_ids = {
        record["candidate_manifest_id"]
        for record in pool_manifest["candidates"]
        if record["arm"] == "proposal"
    }
    random_ids = {
        record["candidate_manifest_id"]
        for record in pool_manifest["candidates"]
        if record["arm"] == "random"
    }
    assert len(proposal_ids) == 12
    assert len(random_ids) == 12
    assert proposal_ids.isdisjoint(random_ids)


def test_duplicate_accounting_is_explicit_and_machine_checked() -> None:
    """No candidate repeats a normalized control hash; duplicates are accounted."""
    packet, pool_manifest, _ = _build()
    records = pool_manifest["candidates"]
    control_hashes = [record["normalized_control_hash"] for record in records]
    assert len(set(control_hashes)) == len(control_hashes) == 64
    assert packet["duplicate_accounting"] == {
        "unique_normalized_control_hashes": 64,
        "duplicate_normalized_control_hashes": [],
        "duplicate_count": 0,
    }
    # Every selected candidate carries a 1-based selection_rank within its arm.
    for record in records:
        if record["arm"] in ("proposal", "random"):
            assert isinstance(record["selection_rank"], int) and record["selection_rank"] >= 1


# --- Seed provenance and disjointness ----------------------------------------


def test_declared_execution_seeds_are_disjoint_and_unique() -> None:
    """Execution seeds are unique within/across candidates and outside archive seeds."""
    contract = _contract()
    search_space = SearchSpaceConfig.from_file(
        _REPO_ROOT / contract["evaluation"]["search_space_path"]
    )
    pool = generate_candidate_pool(
        search_space,
        pool_size=contract["budget"]["candidate_pool_size"],
        pool_seed=contract["budget"]["candidate_pool_seed"],
    )
    all_seeds: list[int] = []
    for index, candidate in enumerate(pool):
        seeds = execution_seeds_for_candidate(index)
        assert len(seeds) == 5
        assert len(set(seeds)) == 5
        all_seeds.extend(seeds)
        assert all(seed != contract["budget"]["candidate_pool_seed"] for seed in seeds)
        assert all(seed != int(candidate.scenario_seed) for seed in seeds)
    assert len(set(all_seeds)) == len(all_seeds)
    assert all(seed >= EXECUTION_SEED_BASE for seed in all_seeds)


def test_seed_disjointness_checks_pass_against_archive_certification_seeds() -> None:
    """Pool seed, scenario seeds, and execution seeds are disjoint from archive seeds."""
    packet, _, _ = _build()
    checks = packet["seed_provenance"]["disjointness_checks"]
    assert checks["candidate_pool_seed_vs_archive"]["disjoint"] is True
    assert checks["candidate_scenario_seeds_vs_archive"]["disjoint"] is True
    assert checks["execution_seeds_vs_archive"]["disjoint"] is True
    assert checks["execution_seeds_vs_candidate_scenario_seeds"]["disjoint"] is True
    assert all(entry["overlap"] == [] for entry in checks.values())


def test_fit_ranker_is_seed_free() -> None:
    """The nearest-neighbor family-invariant ranker declares no fit seed domain."""
    packet, _, _ = _build()
    assert packet["seed_provenance"]["fit_algorithm_seed"] is None
    assert "seed-free" in packet["seed_provenance"]["fit_algorithm_seed_note"]


# --- Content addressing and reproducibility ----------------------------------


def test_compose_produces_acyclic_reproducible_hashes() -> None:
    """Composing twice yields byte-identical files and identical SHA-256 values."""
    files_a = compose_preflight_packet_files(
        _CONTRACT, repo_root=_REPO_ROOT, code_revision="a" * 40
    )
    files_b = compose_preflight_packet_files(
        _CONTRACT, repo_root=_REPO_ROOT, code_revision="a" * 40
    )
    assert set(files_a) == {
        "README.md",
        "SHA256SUMS",
        "candidate_manifest_bindings.v2.json",
        "candidate_pool_manifest.json",
        "preflight_packet.json",
        "proposal_arm_manifest.json",
        "random_arm_manifest.json",
        "step3_run_plan.json",
    }
    assert files_a == files_b
    for name, content in files_a.items():
        assert hashlib.sha256(content).hexdigest() == hashlib.sha256(files_b[name]).hexdigest()
    # The aggregate packet's generated_files match the recomputed raw digests.
    packet = json.loads(files_a["preflight_packet.json"])
    generated = {entry["path"]: entry["file_sha256"] for entry in packet["generated_files"]}
    for name, digest in generated.items():
        assert hashlib.sha256(files_a[name]).hexdigest() == digest


def test_materialize_and_verify_round_trip(tmp_path: Path) -> None:
    """Materialize writes all files and the repeatable null/check-only verify passes."""
    out = tmp_path / "packet"
    report = materialize_preflight_packet(
        out, contract_path=_CONTRACT, repo_root=_REPO_ROOT, code_revision="b" * 40
    )
    assert report["status"] == "pass"
    assert report["failures"] == []
    assert report["planner_runs"] == 0
    assert report["outcome_reads"] == 0
    assert report["arm_budget_equality"]["equal_and_frozen"] is True
    assert len(report["files_checked"]) == 8
    verify = verify_preflight_packet(out, contract_path=_CONTRACT, repo_root=_REPO_ROOT)
    assert verify["status"] == "pass"
    assert verify["failures"] == []
    # Every file's raw SHA-256 matches its SHA256SUMS entry and recomputes.
    for name in report["checks"]:
        assert report["checks"][name]["byte_identical"] is True


def test_verify_detects_byte_drift(tmp_path: Path) -> None:
    """Verification fails closed when any committed packet file is modified."""
    out = tmp_path / "packet"
    materialize_preflight_packet(
        out, contract_path=_CONTRACT, repo_root=_REPO_ROOT, code_revision="c" * 40
    )
    pool_path = out / "candidate_pool_manifest.json"
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    payload["candidates"][0]["scenario_seed"] += 1
    pool_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report = verify_preflight_packet(out, contract_path=_CONTRACT, repo_root=_REPO_ROOT)
    assert report["status"] == "fail"
    assert any("byte drift" in failure for failure in report["failures"])


def test_committed_packet_reproduces_byte_for_byte() -> None:
    """The committed evidence packet matches a fresh recomposition at its recorded revision."""
    packet_path = _PACKET_DIR / "preflight_packet.json"
    if not packet_path.is_file():
        pytest.skip("committed preflight packet not present in this checkout")
    revision = json.loads(packet_path.read_text(encoding="utf-8"))["code_revision"]
    recomputed = compose_preflight_packet_files(
        _CONTRACT, repo_root=_REPO_ROOT, code_revision=revision
    )
    for name, content in recomputed.items():
        on_disk = (_PACKET_DIR / name).read_bytes()
        assert on_disk == content, f"committed {name} drifted from recomputation"


# --- Zero planner / zero outcome proof ---------------------------------------


def test_packet_declares_zero_planner_runs_and_zero_outcome_reads() -> None:
    """Preflight evidence boundary: no planner execution and no outcome access."""
    packet, _, _ = _build()
    assert packet["executed_planners"] == 0
    assert packet["outcome_reads"] == 0
    assert packet["claim_boundary"].startswith("preflight_evidence_only")
    assert packet["evidence_status"] == "tracked-compact-evidence"
    assert "no planner was executed" in packet["planner_execution_proof"]


def test_preflight_modes_never_invoke_outcome_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Materialize/verify modes never call the independent-outcome loader."""
    import robot_sf.adversarial.independent_outcomes as outcome_mod
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import main as script_main

    def _forbid(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("preflight must never load independent outcomes")

    monkeypatch.setattr(outcome_mod, "load_independent_outcomes", _forbid)

    out = tmp_path / "packet"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proposal_vs_random_issue_2921.py",
            "--contract",
            _CONTRACT.as_posix(),
            "--materialize-preflight",
            out.as_posix(),
        ],
    )
    assert script_main() == 0

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proposal_vs_random_issue_2921.py",
            "--contract",
            _CONTRACT.as_posix(),
            "--verify-preflight",
            out.as_posix(),
        ],
    )
    assert script_main() == 0


# --- Cross-check against the main comparison runner --------------------------


def test_preflight_binding_matches_main_runner_arms_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The external v2 binding equals the normal runner's draw, so step 3 admits rows."""
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import main as script_main

    report_path = tmp_path / "main_report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proposal_vs_random_issue_2921.py",
            "--contract",
            _CONTRACT.as_posix(),
            "--seed",
            "42",
            "--output",
            report_path.as_posix(),
        ],
    )
    assert script_main() == 0
    main_report = json.loads(report_path.read_text("utf-8"))

    files = compose_preflight_packet_files(_CONTRACT, repo_root=_REPO_ROOT, code_revision="a" * 40)
    bindings = json.loads(files["candidate_manifest_bindings.v2.json"])

    assert (
        bindings["candidate_manifest_ids_by_arm"]["proposal"]
        == main_report["arm_manifest_ids_by_arm"]["proposal"]
    )
    assert (
        bindings["candidate_manifest_ids_by_arm"]["random"]
        == main_report["arm_manifest_ids_by_arm"]["random"]
    )
    for manifest_id, digest in main_report["arm_manifest_sha256_by_id"].items():
        assert bindings["candidate_manifest_sha256_by_id"][manifest_id] == digest
    assert bindings["candidate_pool_seed"] == 42
    assert bindings["schema_version"] == "adversarial_candidate_manifest_bindings.v2"


def test_v2_bindings_validate_through_script_loader(tmp_path: Path) -> None:
    """The generated v2 binding is accepted by the runner's external-binding loader."""
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import (
        load_expected_candidate_manifest_binding,
    )

    files = compose_preflight_packet_files(_CONTRACT, repo_root=_REPO_ROOT, code_revision="a" * 40)
    binding_path = tmp_path / "bindings.json"
    binding_path.write_bytes(files["candidate_manifest_bindings.v2.json"])
    payload, reason = load_expected_candidate_manifest_binding(binding_path)
    assert payload is not None, reason
    assert reason == "ok"
    assert len(payload["candidate_manifest_ids_by_arm"]["proposal"]) == 12
    assert len(payload["candidate_manifest_ids_by_arm"]["random"]) == 12
    assert set(payload["candidate_manifest_ids_by_arm"]["proposal"]).isdisjoint(
        set(payload["candidate_manifest_ids_by_arm"]["random"])
    )


# --- CLI-level smoke tests ---------------------------------------------------


def test_cli_materialize_and_verify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The materialize CLI writes the packet and the verify CLI passes on it."""
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import main as script_main

    out = tmp_path / "packet"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proposal_vs_random_issue_2921.py",
            "--contract",
            _CONTRACT.as_posix(),
            "--materialize-preflight",
            out.as_posix(),
        ],
    )
    assert script_main() == 0
    assert (out / "candidate_pool_manifest.json").is_file()
    assert (out / "proposal_arm_manifest.json").is_file()
    assert (out / "random_arm_manifest.json").is_file()
    assert (out / "candidate_manifest_bindings.v2.json").is_file()
    assert (out / "SHA256SUMS").is_file()

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proposal_vs_random_issue_2921.py",
            "--contract",
            _CONTRACT.as_posix(),
            "--verify-preflight",
            out.as_posix(),
        ],
    )
    assert script_main() == 0


def test_cli_preflight_requires_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preflight modes fail closed without the frozen --contract."""
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import main as script_main

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proposal_vs_random_issue_2921.py",
            "--materialize-preflight",
            "/tmp/should_not_write",
        ],
    )
    assert script_main() == 2


def test_cli_preflight_modes_are_mutually_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Materialize and verify flags cannot be combined."""
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import main as script_main

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proposal_vs_random_issue_2921.py",
            "--contract",
            _CONTRACT.as_posix(),
            "--materialize-preflight",
            "/tmp/a",
            "--verify-preflight",
            "/tmp/b",
        ],
    )
    assert script_main() == 2
