"""Tests for the disjoint-modulo PR-gate scope contract (issue #5059).

The headline regression is the issue's own scenario: an older gate holding an
open-ended range scope can cross into a newer gate's range on a re-list pass.
The contract must (a) reject the non-immutable range scopes and (b) certify the
residue-class fix as disjoint.
"""

from __future__ import annotations

import json

import pytest

from scripts.tools.pr_gate_scopes import (
    Gate,
    GateScopeError,
    RangeScope,
    ResidueScope,
    load_gates,
    main,
    residue_scopes_disjoint,
    validate_active_gates,
)

# --- ResidueScope construction / ownership ---------------------------------


def test_residue_scope_owns_by_modulo():
    """A residue scope claims exactly the PR numbers matching its residue."""
    even = ResidueScope(modulus=2, residue=0)
    assert even.owns(5044) and even.owns(5058)
    assert not even.owns(5057)


@pytest.mark.parametrize(
    "modulus, residue",
    [(0, 0), (-2, 0), (2, 2), (2, -1), (3, 3)],
)
def test_residue_scope_rejects_bad_params(modulus, residue):
    """Out-of-range or non-positive modulus/residue values are rejected."""
    with pytest.raises(GateScopeError):
        ResidueScope(modulus=modulus, residue=residue)


def test_residue_scope_rejects_bool_params():
    """bool is an int subclass; guard against True/False sneaking in as modulus."""
    with pytest.raises(GateScopeError):
        ResidueScope(modulus=True, residue=0)


def test_residue_scope_is_immutable():
    """Frozen dataclass: the scope cannot be mutated after construction."""
    scope = ResidueScope(modulus=2, residue=1)
    with pytest.raises(Exception):
        scope.modulus = 4  # type: ignore[misc]


# --- residue disjointness arithmetic ---------------------------------------


def test_same_modulus_distinct_residues_disjoint():
    """Distinct residues under a shared modulus never share a PR."""
    assert residue_scopes_disjoint(ResidueScope(2, 0), ResidueScope(2, 1))


def test_same_modulus_same_residue_overlaps():
    """Identical residue classes are not disjoint."""
    assert not residue_scopes_disjoint(ResidueScope(2, 0), ResidueScope(2, 0))


def test_coprime_moduli_always_overlap():
    """2 and 3 are coprime => CRT guarantees a common solution for any residues."""
    assert not residue_scopes_disjoint(ResidueScope(2, 0), ResidueScope(3, 1))


def test_shared_factor_disjoint_when_residues_incompatible():
    """Shared factor, incompatible residues (gcd rule fails) => disjoint.

    mod 4 residue 0 (0,4,8,...) vs mod 6 residue 3 (3,9,15,...): gcd=2,
    (0-3) % 2 == 1 != 0 => disjoint.
    """
    assert residue_scopes_disjoint(ResidueScope(4, 0), ResidueScope(6, 3))


def test_shared_factor_overlap_when_residues_compatible():
    """Shared factor, compatible residues => overlap.

    mod 4 residue 2 (2,6,10,...) vs mod 6 residue 0 (0,6,12,...): both hit 6.
    """
    a, b = ResidueScope(4, 2), ResidueScope(6, 0)
    assert not residue_scopes_disjoint(a, b)


# --- the issue #5059 scenario ----------------------------------------------


def test_issue_5059_open_ended_range_gates_rejected_and_crossing():
    """Reproduce the hazard: an older open-ended gate + a newer bounded gate.

    Gate 999997 processes newly appearing PRs (open-ended range from #5037);
    gate 999996 owns #5047-#5057. They look disjoint in a snapshot but the open
    range crosses into the newer gate's range.
    """
    gates = [
        Gate("999997", RangeScope(low=5037, high=None)),
        Gate("999996", RangeScope(low=5047, high=5057)),
    ]
    report = validate_active_gates(gates)
    assert report.ok is False
    kinds = {v.kind for v in report.violations}
    # Both non-residue scopes are flagged, and the open range overlaps the newer one.
    assert "non_residue_scope" in kinds
    overlaps = [v for v in report.violations if v.kind == "overlap"]
    assert overlaps, "open-ended range must be caught crossing the newer gate"
    assert overlaps[0].witness_pr == 5047


def test_issue_5059_residue_fix_is_disjoint_and_passes():
    """The fix: express both gates as residue classes under a shared modulus."""
    gates = [
        Gate("999997", ResidueScope(modulus=2, residue=0)),  # even PRs
        Gate("999996", ResidueScope(modulus=2, residue=1)),  # odd PRs
    ]
    report = validate_active_gates(gates)
    assert report.ok is True
    assert report.violations == []
    # And they genuinely never share a PR across the disputed range.
    even, odd = gates[0].scope, gates[1].scope
    for pr in range(5037, 5058):
        assert not (even.owns(pr) and odd.owns(pr))


def test_two_residue_gates_same_residue_overlap_reported_with_witness():
    """Two gates on the same residue class report an overlap with a witness PR."""
    gates = [
        Gate("a", ResidueScope(3, 1)),
        Gate("b", ResidueScope(3, 1)),
    ]
    report = validate_active_gates(gates)
    assert not report.ok
    overlap = next(v for v in report.violations if v.kind == "overlap")
    assert overlap.witness_pr == 1
    assert set(overlap.gate_ids) == {"a", "b"}


def test_disjoint_range_gates_pass_when_ranges_allowed():
    """With require_residue=False, non-overlapping range gates are accepted."""
    gates = [
        Gate("a", RangeScope(low=5000, high=5010)),
        Gate("b", RangeScope(low=5011, high=5020)),
    ]
    report = validate_active_gates(gates, require_residue=False)
    assert report.ok


def test_overlapping_range_gates_fail_even_when_ranges_allowed():
    """Overlap is checked even when range scopes are permitted."""
    gates = [
        Gate("a", RangeScope(low=5000, high=5010)),
        Gate("b", RangeScope(low=5010, high=5020)),
    ]
    report = validate_active_gates(gates, require_residue=False)
    assert not report.ok
    assert report.violations[0].witness_pr == 5010


def test_duplicate_gate_id_rejected():
    """A cohort with a repeated gate_id is rejected outright."""
    with pytest.raises(GateScopeError):
        validate_active_gates([Gate("x", ResidueScope(2, 0)), Gate("x", ResidueScope(2, 1))])


# --- manifest loading + CLI ------------------------------------------------


def test_load_gates_residue_and_range():
    """Manifest loader builds residue and range scopes and accepts 'id' alias."""
    payload = {
        "gates": [
            {"gate_id": "999996", "scope": {"modulus": 2, "residue": 1}},
            {"id": "999997", "scope": {"range": [5037]}},
        ]
    }
    gates = load_gates(payload)
    assert isinstance(gates[0].scope, ResidueScope)
    assert isinstance(gates[1].scope, RangeScope)
    assert gates[1].scope.open_ended


def test_load_gates_requires_scope():
    """A gate entry with no scope object is rejected."""
    with pytest.raises(GateScopeError):
        load_gates([{"gate_id": "a"}])


def test_cli_passes_on_disjoint_residue_manifest(tmp_path, capsys):
    """CLI exits 0 on a disjoint residue-scope manifest."""
    manifest = tmp_path / "gates.json"
    manifest.write_text(
        json.dumps(
            [
                {"gate_id": "999997", "scope": {"modulus": 2, "residue": 0}},
                {"gate_id": "999996", "scope": {"modulus": 2, "residue": 1}},
            ]
        ),
        encoding="utf-8",
    )
    rc = main(["--gates", str(manifest)])
    assert rc == 0
    assert "disjoint residue scopes" in capsys.readouterr().out


def test_cli_fails_on_crossing_range_manifest(tmp_path, capsys):
    """CLI exits 1 and explains the violation on the issue #5059 range manifest."""
    manifest = tmp_path / "gates.json"
    manifest.write_text(
        json.dumps(
            [
                {"gate_id": "999997", "scope": {"low": 5037}},
                {"gate_id": "999996", "scope": {"low": 5047, "high": 5057}},
            ]
        ),
        encoding="utf-8",
    )
    rc = main(["--gates", str(manifest)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "non_residue_scope" in err or "overlap" in err


def test_cli_json_output(tmp_path, capsys):
    """CLI --json emits the schema-tagged report on stdout."""
    manifest = tmp_path / "gates.json"
    manifest.write_text(
        json.dumps([{"gate_id": "a", "scope": {"modulus": 1, "residue": 0}}]),
        encoding="utf-8",
    )
    rc = main(["--gates", str(manifest), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "pr_gate_scope_disjointness_report.v1"
    assert payload["ok"] is True


def test_cli_missing_file_fails_closed(tmp_path, capsys):
    """CLI exits 2 (error) when the manifest file is missing."""
    rc = main(["--gates", str(tmp_path / "nope.json")])
    assert rc == 2
    assert "ERROR" in capsys.readouterr().err
