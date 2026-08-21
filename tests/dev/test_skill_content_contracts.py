"""Tests for declarative skill content contracts (issue #7661)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "scripts" / "dev" / "check_skills.py"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "dev"))
import skill_content_contracts as scc  # noqa: E402

MIGRATED_SKILLS = ("goal-autopilot", "goal-pr-review", "goal-issue-implementation")

VALID_CONTRACT = {
    "version": 1,
    "skill": "sample-skill",
    "requirements": [
        {
            "id": "must-mention-output",
            "description": "keep output guidance",
            "scope": "lowercase",
            "operator": "all_of",
            "values": ["result.json"],
        }
    ],
}


def _write_contract(repo: Path, skill: str, raw) -> Path:
    import yaml

    contracts = repo / ".agents" / "skills" / "tests" / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    path = contracts / f"{skill}.content-contract.v1.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _repo(tmp_path: Path) -> Path:
    return tmp_path


def test_valid_contract_loads_and_evaluates(tmp_path: Path):
    """A schema-valid contract loads and passes matching text."""
    _write_contract(_repo(tmp_path), "sample-skill", VALID_CONTRACT)
    contract = scc.load_contract(tmp_path, "sample-skill")
    assert contract["skill"] == "sample-skill"
    assert scc.evaluate_contract(contract, "see result.json for details") == []


def test_unknown_top_level_field_rejected(tmp_path: Path):
    """Unknown top-level fields fail closed instead of being ignored."""
    raw = dict(VALID_CONTRACT)
    raw["extra_section"] = {"oops": True}
    _write_contract(_repo(tmp_path), "sample-skill", raw)
    with pytest.raises(scc.ContractError, match="unknown top-level field"):
        scc.load_contract(tmp_path, "sample-skill")


def test_unknown_requirement_field_rejected(tmp_path: Path):
    """Unknown requirement fields fail closed."""
    raw = {
        **VALID_CONTRACT,
        "requirements": [dict(VALID_CONTRACT["requirements"][0], phrase="x")],
    }
    _write_contract(_repo(tmp_path), "sample-skill", raw)
    with pytest.raises(scc.ContractError, match="unknown field"):
        scc.load_contract(tmp_path, "sample-skill")


@pytest.mark.parametrize("operator", ["any", "ALL_OF", "one_of", 1])
def test_malformed_operator_rejected(tmp_path: Path, operator):
    """Operators outside all_of/any_of are rejected at load time."""
    raw = {
        **VALID_CONTRACT,
        "requirements": [dict(VALID_CONTRACT["requirements"][0], operator=operator)],
    }
    _write_contract(_repo(tmp_path), "sample-skill", raw)
    with pytest.raises(scc.ContractError, match="unknown operator"):
        scc.load_contract(tmp_path, "sample-skill")


def test_missing_required_concept_fails_with_contract_id(tmp_path: Path):
    """Missing all_of values produce one error per value naming the contract id."""
    _write_contract(_repo(tmp_path), "sample-skill", VALID_CONTRACT)
    contract = scc.load_contract(tmp_path, "sample-skill")
    errors = scc.evaluate_contract(contract, "nothing relevant here")
    assert len(errors) == 1
    assert "contract 'must-mention-output'" in errors[0]
    assert "'result.json'" in errors[0]
    assert "keep output guidance" in errors[0]


def test_any_of_operator_passes_on_first_match(tmp_path: Path):
    """any_of requirements accept any single listed value."""
    raw = {
        **VALID_CONTRACT,
        "requirements": [
            dict(
                VALID_CONTRACT["requirements"][0],
                id="queue-reference",
                operator="any_of",
                values=["a.py", "b.py"],
            )
        ],
    }
    _write_contract(_repo(tmp_path), "sample-skill", raw)
    contract = scc.load_contract(tmp_path, "sample-skill")
    assert scc.evaluate_contract(contract, "use b.py today") == []
    errors = scc.evaluate_contract(contract, "neither c.py nor d.py")
    assert len(errors) == 1
    assert "missing one of" in errors[0]


def test_normalized_scope_collapses_whitespace(tmp_path: Path):
    """normalized scope matches phrases across line breaks and casing."""
    raw = {
        **VALID_CONTRACT,
        "requirements": [
            dict(
                VALID_CONTRACT["requirements"][0],
                id="routing-pointer",
                scope="normalized",
                values=["shared model-routing pointer"],
            )
        ],
    }
    _write_contract(_repo(tmp_path), "sample-skill", raw)
    contract = scc.load_contract(tmp_path, "sample-skill")
    text = "Use the SHARED\n  model-routing   pointer before dispatch."
    assert scc.evaluate_contract(contract, text) == []


def test_skill_name_must_match_file_stem(tmp_path: Path):
    """A fixture whose skill field disagrees with its filename is invalid."""
    raw = {**VALID_CONTRACT, "skill": "other-skill"}
    _write_contract(_repo(tmp_path), "sample-skill", raw)
    with pytest.raises(scc.ContractError, match="does not match file stem"):
        scc.load_contract(tmp_path, "sample-skill")


def test_duplicate_requirement_ids_rejected(tmp_path: Path):
    """Duplicate requirement ids make failure attribution ambiguous."""
    dup = VALID_CONTRACT["requirements"][0]
    raw = {**VALID_CONTRACT, "requirements": [dup, dict(dup)]}
    _write_contract(_repo(tmp_path), "sample-skill", raw)
    with pytest.raises(scc.ContractError, match="duplicate requirement id"):
        scc.load_contract(tmp_path, "sample-skill")


@pytest.mark.parametrize("skill", MIGRATED_SKILLS)
def test_migrated_contracts_exist_and_match_real_skills(skill: str):
    """Parity gate: every migrated skill's real SKILL.md satisfies its contract."""
    contract = scc.load_contract(REPO_ROOT, skill)
    text = (REPO_ROOT / ".agents" / "skills" / skill / "SKILL.md").read_text(encoding="utf-8")
    assert scc.evaluate_contract(contract, text) == []


def test_check_skills_cli_exit_status_stable():
    """The checker keeps exit code 0 and its summary line on a valid registry."""
    proc = subprocess.run(
        [sys.executable, str(CHECKER)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    assert "Validated" in proc.stdout


def test_no_hard_coded_skill_phrase_constants_remain():
    """The migrated phrase tuples must be gone from check_skills.py."""
    source = CHECKER.read_text(encoding="utf-8")
    for constant in (
        "GOAL_AUTOPILOT_LEDGER_REQUIRED_PHRASES",
        "GOAL_AUTOPILOT_SHARED_ROUTING_REQUIRED_PHRASES",
        "GOAL_PR_REVIEW_REQUIRED_PHRASES",
        "ROUTED_FAILURE_REQUIRED_PHRASES",
        "WORKER_OUTPUT_REQUIRED_PHRASES",
        "ARTIFACT_FIRST_REQUIRED_PHRASES",
        "ARTIFACT_FIRST_REQUIRED_FILES",
    ):
        assert constant not in source
