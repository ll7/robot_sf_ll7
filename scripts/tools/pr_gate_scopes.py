#!/usr/bin/env python3
"""Disjoint modulo scopes for concurrent PR gates (issue #5059).

Background
----------
The autonomous PR-gate orchestrator can run more than one *gate* at a time. Each
gate is a worker prompt that claims a set of pull requests to shepherd. When
gates claim PRs by an ad-hoc *range* (for example "gate A owns #5044/#5037" and
"gate B owns #5047-#5057"), the ranges can look disjoint in one snapshot yet
*cross* on a later re-list pass: a gate told to "process newly appearing PRs"
will happily pick up a number that a newer gate already owns. Two gates
shepherding the same PR is a double-dispatch bug.

The dispatch contract this module enforces is simple and immutable: an active
gate owns a **residue class** ``pr_number % modulus == residue``. Residue classes
are stable regardless of which PRs currently exist, so a gate can re-list as
often as it likes and never reach into another gate's residue. Two residue
classes are disjoint iff no integer satisfies both congruences; that is decided
exactly by the classic CRT criterion ``(r_a - r_b) % gcd(N_a, N_b) != 0``.

What this module provides
-------------------------
* :class:`ResidueScope` - the canonical, immutable gate scope.
* :class:`RangeScope` - the legacy non-modulo scope, modelled only so the checker
  can *reject* it (and, for an open-ended range, prove it must eventually cross
  any residue scope).
* :func:`validate_active_gates` - the fail-closed regression check: it refuses a
  gate cohort in which any two live gates could own the same PR, and (by default)
  refuses any non-residue scope because such a scope is not immutable.
* A small CLI (``python -m scripts.tools.pr_gate_scopes --gates gates.json``) so
  the orchestrator can run the check against a live gate manifest before it
  admits a second gate.

Nothing here mutates state or talks to GitHub; it is pure arithmetic over gate
descriptors so it is cheap to unit-test and safe to call inline in a dispatcher.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Any, Union

REPORT_SCHEMA = "pr_gate_scope_disjointness_report.v1"


class GateScopeError(ValueError):
    """A gate scope descriptor is malformed or violates the dispatch contract."""


@dataclass(frozen=True)
class ResidueScope:
    """Canonical immutable gate scope: owns ``pr`` iff ``pr % modulus == residue``.

    A residue class does not depend on which PR numbers currently exist, so a
    gate bound to one cannot drift into another gate's PRs on a re-list pass.
    """

    modulus: int
    residue: int

    def __post_init__(self) -> None:
        """Validate that modulus and residue form a well-formed residue class."""
        if not isinstance(self.modulus, int) or isinstance(self.modulus, bool):
            raise GateScopeError(f"modulus must be an int, got {self.modulus!r}")
        if not isinstance(self.residue, int) or isinstance(self.residue, bool):
            raise GateScopeError(f"residue must be an int, got {self.residue!r}")
        if self.modulus < 1:
            raise GateScopeError(f"modulus must be >= 1, got {self.modulus}")
        if not 0 <= self.residue < self.modulus:
            raise GateScopeError(
                f"residue {self.residue} out of range for modulus {self.modulus} "
                f"(expected 0 <= residue < {self.modulus})"
            )

    def owns(self, pr_number: int) -> bool:
        """Return whether this residue class claims ``pr_number``."""
        return pr_number % self.modulus == self.residue

    def describe(self) -> str:
        """Human-readable one-line rendering of the scope predicate."""
        return f"pr % {self.modulus} == {self.residue}"


@dataclass(frozen=True)
class RangeScope:
    """Legacy non-modulo scope: owns ``pr`` iff ``low <= pr <= high``.

    ``high is None`` models an *open-ended* range - a gate told to keep picking
    up newly appearing PRs. Range scopes are not immutable (their membership
    depends on which numbers exist) and are rejected by the contract; they are
    modelled here only so the checker can explain why and produce a crossing
    witness.
    """

    low: int
    high: Union[int, None] = None

    def __post_init__(self) -> None:
        """Validate that ``low``/``high`` are ints (or ``None``) and ordered."""
        for name, value in (("low", self.low), ("high", self.high)):
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise GateScopeError(f"{name} must be an int or null, got {value!r}")
        if self.high is not None and self.high < self.low:
            raise GateScopeError(f"high {self.high} must be >= low {self.low}")

    def owns(self, pr_number: int) -> bool:
        """Return whether ``pr_number`` falls inside the (possibly open) range."""
        if pr_number < self.low:
            return False
        return self.high is None or pr_number <= self.high

    @property
    def open_ended(self) -> bool:
        """Whether the range has no upper bound (claims all future PRs)."""
        return self.high is None

    def describe(self) -> str:
        """Human-readable one-line rendering of the scope predicate."""
        hi = "inf" if self.high is None else str(self.high)
        return f"{self.low} <= pr <= {hi}"


Scope = Union[ResidueScope, RangeScope]


@dataclass(frozen=True)
class Gate:
    """An active PR gate: an identifier plus the scope of PRs it shepherds."""

    gate_id: str
    scope: Scope

    def describe(self) -> str:
        """Human-readable ``gate <id> [<scope>]`` rendering for messages."""
        return f"gate {self.gate_id} [{self.scope.describe()}]"


@dataclass
class Violation:
    """A single contract breach found by :func:`validate_active_gates`."""

    kind: str  # "non_residue_scope" | "overlap"
    gate_ids: list[str]
    message: str
    witness_pr: Union[int, None] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the violation to a plain JSON-friendly dict."""
        return {
            "kind": self.kind,
            "gate_ids": list(self.gate_ids),
            "message": self.message,
            "witness_pr": self.witness_pr,
        }


@dataclass
class DisjointnessReport:
    """Result of validating an active gate cohort."""

    ok: bool
    gate_count: int
    violations: list[Violation] = field(default_factory=list)
    schema: str = REPORT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report (and its violations) to a plain dict."""
        return {
            "schema": self.schema,
            "ok": self.ok,
            "gate_count": self.gate_count,
            "violations": [v.to_dict() for v in self.violations],
        }


def residue_scopes_disjoint(a: ResidueScope, b: ResidueScope) -> bool:
    """Return ``True`` iff no integer belongs to both residue classes.

    Two congruences ``x == r_a (mod N_a)`` and ``x == r_b (mod N_b)`` have a
    common solution iff ``(r_a - r_b)`` is divisible by ``gcd(N_a, N_b)`` (the
    Chinese Remainder Theorem solvability condition). They are therefore disjoint
    iff that divisibility fails.
    """

    return (a.residue - b.residue) % gcd(a.modulus, b.modulus) != 0


def _shared_residue_witness(a: ResidueScope, b: ResidueScope) -> Union[int, None]:
    """Smallest non-negative PR number owned by both residue classes, if any."""

    if residue_scopes_disjoint(a, b):
        return None
    lcm = a.modulus // gcd(a.modulus, b.modulus) * b.modulus
    for candidate in range(lcm):
        if a.owns(candidate) and b.owns(candidate):
            return candidate
    return None  # pragma: no cover - unreachable when a solution exists


def _range_residue_witness(rng: RangeScope, res: ResidueScope) -> Union[int, None]:
    """Smallest PR number owned by both a range and a residue class, if any.

    For an open-ended range this always exists (a residue class is unbounded), so
    an open range provably crosses every residue scope - which is exactly the
    re-list hazard from issue #5059.
    """

    start = rng.low
    first = start + ((res.residue - start) % res.modulus)
    if rng.high is not None and first > rng.high:
        return None
    return first


def _range_ranges_witness(a: RangeScope, b: RangeScope) -> Union[int, None]:
    low = max(a.low, b.low)
    highs = [h for h in (a.high, b.high) if h is not None]
    high = min(highs) if highs else None
    if high is None or low <= high:
        return low
    return None


def _scopes_overlap_witness(a: Scope, b: Scope) -> Union[int, None]:
    """Smallest PR number owned by both scopes, or ``None`` if disjoint."""

    if isinstance(a, ResidueScope) and isinstance(b, ResidueScope):
        return _shared_residue_witness(a, b)
    if isinstance(a, RangeScope) and isinstance(b, ResidueScope):
        return _range_residue_witness(a, b)
    if isinstance(a, ResidueScope) and isinstance(b, RangeScope):
        return _range_residue_witness(b, a)
    if isinstance(a, RangeScope) and isinstance(b, RangeScope):
        return _range_ranges_witness(a, b)
    raise GateScopeError(f"unsupported scope types: {type(a)!r}, {type(b)!r}")


def validate_active_gates(gates: list[Gate], *, require_residue: bool = True) -> DisjointnessReport:
    """Fail-closed regression check for a concurrent gate cohort.

    Two rules, both required by issue #5059:

    1. **Immutable residue scopes.** With ``require_residue=True`` (default) every
       active gate must use a :class:`ResidueScope`. A :class:`RangeScope` is not
       immutable - its membership shifts as PRs appear - so it is reported as a
       ``non_residue_scope`` violation and must be converted to a residue class
       (or the gate revoked) before another gate is admitted.
    2. **Pairwise disjoint.** No two active gates may own the same PR. Any pair
       that shares a PR number is reported as an ``overlap`` violation with the
       smallest crossing PR as a witness.

    The returned report's ``ok`` is ``True`` only when no violation is found.
    """

    seen_ids: set[str] = set()
    for gate in gates:
        if gate.gate_id in seen_ids:
            raise GateScopeError(f"duplicate gate_id {gate.gate_id!r} in cohort")
        seen_ids.add(gate.gate_id)

    violations: list[Violation] = []

    if require_residue:
        for gate in gates:
            if not isinstance(gate.scope, ResidueScope):
                reason = (
                    "open-ended range crosses every residue scope on re-list"
                    if isinstance(gate.scope, RangeScope) and gate.scope.open_ended
                    else "non-modulo scope is not immutable and can drift on re-list"
                )
                violations.append(
                    Violation(
                        kind="non_residue_scope",
                        gate_ids=[gate.gate_id],
                        message=(
                            f"{gate.describe()} is not a residue scope: {reason}. "
                            f"Convert to `pr % N == residue` or revoke the gate."
                        ),
                    )
                )

    for i in range(len(gates)):
        for j in range(i + 1, len(gates)):
            g_a, g_b = gates[i], gates[j]
            witness = _scopes_overlap_witness(g_a.scope, g_b.scope)
            if witness is not None:
                violations.append(
                    Violation(
                        kind="overlap",
                        gate_ids=[g_a.gate_id, g_b.gate_id],
                        message=(
                            f"{g_a.describe()} and {g_b.describe()} can both own "
                            f"PR #{witness}; active gate scopes must be disjoint."
                        ),
                        witness_pr=witness,
                    )
                )

    return DisjointnessReport(ok=not violations, gate_count=len(gates), violations=violations)


def _scope_from_dict(data: dict[str, Any]) -> Scope:
    if "modulus" in data or "residue" in data:
        try:
            return ResidueScope(modulus=data["modulus"], residue=data["residue"])
        except KeyError as exc:
            raise GateScopeError(
                f"residue scope requires 'modulus' and 'residue': {data!r}"
            ) from exc
    if "low" in data or "high" in data or "range" in data:
        if "range" in data:
            rng = data["range"]
            low, high = rng[0], (rng[1] if len(rng) > 1 else None)
        else:
            low, high = data.get("low"), data.get("high")
        if low is None:
            raise GateScopeError(f"range scope requires 'low': {data!r}")
        return RangeScope(low=low, high=high)
    raise GateScopeError(
        f"scope must declare a residue (modulus/residue) or range (low/high): {data!r}"
    )


def load_gates(payload: Any) -> list[Gate]:
    """Build :class:`Gate` objects from a decoded manifest.

    Accepts either a list of gate objects or an object with a ``gates`` list.
    Each gate object needs ``gate_id`` (or ``id``) and a ``scope`` object.
    """

    if isinstance(payload, dict) and "gates" in payload:
        entries = payload["gates"]
    else:
        entries = payload
    if not isinstance(entries, list):
        raise GateScopeError("gate manifest must be a list or have a 'gates' list")

    gates: list[Gate] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise GateScopeError(f"gate entry must be an object, got {entry!r}")
        gate_id = entry.get("gate_id", entry.get("id"))
        if gate_id is None:
            raise GateScopeError(f"gate entry requires 'gate_id': {entry!r}")
        scope_data = entry.get("scope")
        if not isinstance(scope_data, dict):
            raise GateScopeError(f"gate {gate_id!r} requires a 'scope' object")
        gates.append(Gate(gate_id=str(gate_id), scope=_scope_from_dict(scope_data)))
    return gates


def build_argparser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the disjointness checker."""
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed check that concurrent PR gates own disjoint, immutable "
            "residue scopes (issue #5059)."
        )
    )
    parser.add_argument(
        "--gates",
        type=Path,
        required=True,
        help="Path to a JSON gate manifest (list of {gate_id, scope}).",
    )
    parser.add_argument(
        "--allow-range-scopes",
        action="store_true",
        help=(
            "Do not reject non-residue (range) scopes. Overlap is still checked. "
            "Use only for diagnosing a legacy cohort mid-migration."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON on stdout.",
    )
    return parser


def main(argv: Union[list[str], None] = None) -> int:
    """Validate a gate manifest file; exit 0 (ok), 1 (violation), or 2 (error)."""
    args = build_argparser().parse_args(argv)
    try:
        payload = json.loads(args.gates.read_text(encoding="utf-8"))
        gates = load_gates(payload)
        report = validate_active_gates(gates, require_residue=not args.allow_range_scopes)
    except (GateScopeError, json.JSONDecodeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    elif report.ok:
        print(f"OK: {report.gate_count} active gate(s) own disjoint residue scopes.")
    else:
        print(
            f"FAIL: {len(report.violations)} violation(s) across "
            f"{report.gate_count} active gate(s):",
            file=sys.stderr,
        )
        for violation in report.violations:
            print(f"  - [{violation.kind}] {violation.message}", file=sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
