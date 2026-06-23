#!/usr/bin/env python3
"""CPU-only pre-submit evidence-contract conformance check.

SLURM jobs have burned GPU hours and then failed *closed* because the public
commit did not emit a required evidence-contract field (e.g. issue #1475 /
SLURM job 12913: the ORCA-residual smoke summary was missing
``residual_clipping_rate`` and the other required diagnostics). All existing
evidence tooling is *post*-run. This script is the cheap *pre-submit* gate: it
builds the evidence block from a representative row using the **same canonical
builders the real run uses** and blocks (non-zero exit) if any required field
would be missing.

It is a thin orchestrator: it **composes** existing contract owners and never
redefines the required-field list.

Registry shape
--------------
Each entry in :data:`_CONTRACT_REGISTRY` maps a ``contract_id`` to a
:class:`_ContractSpec`:

* ``required_fields`` — imported from the **canonical owner** module (no second copy).
* ``build_evidence`` — callable ``(row) -> dict`` reusing the production builder;
  returns the evidence block carrying ``missing_required_fields``.
* ``representative_row`` — callable ``() -> dict`` producing a synthetic row that
  mirrors the real current-main row shape (built via the real adapter, not fabricated).
* ``owner`` — human-readable pointer to the module to fix when the check fails.

Adding a second contract is a **one-entry change**: append a ``_ContractSpec`` to
the registry. Nothing else in this file is contract-specific.

Usage
-----
    preflight_evidence_contract.py <contract-id> [--row rows.json] [--json]

Exit code ``0`` means the contract conforms (safe to submit); any non-zero exit
means the public commit would emit an incomplete contract and the job must not
be submitted. CPU-only: no GPU, no SLURM, no network, no training.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Make the repo root importable when run as a bare script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Canonical contract owner: the required-field list lives here, not in this script.
from robot_sf.training.orca_residual_lineage_packet import (  # noqa: E402
    REQUIRED_ORCA_RESIDUAL_SMOKE_FIELDS,
)


@dataclass(frozen=True)
class _ContractSpec:
    """Declarative description of one pre-submit evidence contract.

    Attributes:
        contract_id: Stable identifier used on the command line.
        required_fields: Required-field tuple imported from the canonical owner.
        owner: Human-readable pointer to the module to fix on failure.
        build_evidence: Builds the evidence block (with ``missing_required_fields``)
            from a representative row, reusing the production builder.
        representative_row: Returns a synthetic row mirroring the real on-main shape.
    """

    contract_id: str
    required_fields: tuple[str, ...]
    owner: str
    build_evidence: Callable[[Mapping[str, Any]], dict[str, Any]]
    representative_row: Callable[[], dict[str, Any]]


def _orca_residual_smoke_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build the ORCA-residual smoke evidence block from one representative row.

    Reuses the production builder ``_attach_orca_residual_smoke_evidence`` so the
    pre-submit check evaluates exactly what the real run would emit.
    """
    # Imported lazily: the builder lives in a sibling script, not an importable package.
    from scripts.validation.run_policy_search_candidate import (
        _attach_orca_residual_smoke_evidence,
    )

    summary: dict[str, Any] = {}
    # An in-memory jsonl marker keeps the pointer-status classifier on its
    # "present" branch without touching disk during the cheap CPU gate.
    _attach_orca_residual_smoke_evidence(
        summary,
        [row],
        Path("preflight-representative-row.jsonl"),
        missing_jsonl=False,
    )
    evidence = summary.get("orca_residual_smoke_evidence")
    if not isinstance(evidence, dict):  # pragma: no cover - defensive
        raise RuntimeError("smoke builder did not attach an evidence block")
    return evidence


def _orca_residual_representative_row() -> dict[str, Any]:
    """Return a synthetic GuardedPPO decision row mirroring the post-#1475 shape.

    The row is constructed through the **real** ``GuardedPPOAdapter`` and the
    production ``update_shield_stats`` helper, so it carries the same
    ``algorithm_metadata.shield_stats.last_decision.action_adaptation`` payload a
    live rollout writes — including the post-#1475 ``residual_clipped`` signal.
    Nothing about the contract fields is hand-fabricated to pass.
    """
    from robot_sf.planner.guarded_ppo import (
        GuardedPPOAdapter,
        build_guarded_ppo_config,
    )
    from robot_sf.planner.safety_shield import new_shield_stats, update_shield_stats

    observation = {
        "robot": {"position": [0.0, 0.0], "heading": [0.0]},
        "goal": {"current": [10.0, 0.0], "next": [12.0, 0.0]},
        "pedestrians": {"positions": [], "velocities": [], "count": [0]},
    }
    config = build_guarded_ppo_config(
        {
            "prior_residual_mode": True,
            "prior_residual_max_linear_delta": 0.35,
            "prior_residual_max_angular_delta": 0.35,
            "prior_policy": "none",
        }
    )
    adapter = GuardedPPOAdapter(config=config, fallback_adapter=None, prior_adapter=None)
    decision = adapter.choose_command_decision(observation, (0.8, 0.0))

    shield_stats = new_shield_stats()
    update_shield_stats(shield_stats, decision)
    shield_stats["last_decision"] = decision.to_metadata()

    return {
        "status": "completed",
        "termination_reason": "max_steps",
        "metrics": {"shield_intervention_rate": 0.0},
        "algorithm_metadata": {"shield_stats": shield_stats},
    }


# Registry: contract_id -> spec. Seed with one contract; adding another is a 1-entry change.
_CONTRACT_REGISTRY: dict[str, _ContractSpec] = {
    "orca_residual_smoke": _ContractSpec(
        contract_id="orca_residual_smoke",
        required_fields=REQUIRED_ORCA_RESIDUAL_SMOKE_FIELDS,
        owner=(
            "robot_sf/training/orca_residual_lineage_packet.py "
            "(REQUIRED_ORCA_RESIDUAL_SMOKE_FIELDS) + "
            "scripts/validation/run_policy_search_candidate.py "
            "(_attach_orca_residual_smoke_evidence)"
        ),
        build_evidence=_orca_residual_smoke_evidence,
        representative_row=_orca_residual_representative_row,
    ),
}


def _git_hash() -> str | None:
    """Read the current git HEAD SHA for provenance, or ``None`` if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _evaluate(spec: _ContractSpec, row: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one contract against one row and return a structured report."""
    evidence = spec.build_evidence(row)
    builder_missing = list(evidence.get("missing_required_fields", []))
    # Independent assertion: every required field is present and non-null in the block.
    field_missing = [
        field
        for field in spec.required_fields
        if evidence.get(field) is None or evidence.get(field) == ""
    ]
    missing = sorted(set(builder_missing) | set(field_missing))
    conforms = not missing
    return {
        "contract_id": spec.contract_id,
        "conforms": conforms,
        "required_fields": list(spec.required_fields),
        "missing_required_fields": missing,
        "owner": spec.owner,
        "evidence": evidence,
        "git_head": _git_hash(),
    }


def _render_human(report: dict[str, Any]) -> str:
    """Render a human-readable report block."""
    lines = [
        f"contract: {report['contract_id']}",
        f"git HEAD: {report.get('git_head') or 'unknown'}",
        f"required fields: {', '.join(report['required_fields'])}",
    ]
    if report["conforms"]:
        lines.append("result: PASS — contract conforms; safe to submit.")
    else:
        lines.append("result: BLOCK — public commit would emit an incomplete contract.")
        lines.append(f"missing fields: {', '.join(report['missing_required_fields'])}")
        lines.append(f"fix at owner: {report['owner']}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="preflight_evidence_contract.py",
        description=(
            "CPU-only pre-submit check that the current public commit would emit a "
            "contract's required evidence fields. Exit 0 = safe to submit; "
            "non-zero = block (job would fail closed on bookkeeping)."
        ),
    )
    parser.add_argument(
        "contract_id",
        help=f"Contract to check. Known: {', '.join(sorted(_CONTRACT_REGISTRY))}",
    )
    parser.add_argument(
        "--row",
        type=Path,
        default=None,
        help=(
            "Optional JSON file with a representative row (object) or a list of rows; "
            "the first object is used. Defaults to a built-in synthetic on-main row."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the machine-readable JSON report to stdout.",
    )
    return parser.parse_args(argv)


def _load_row(path: Path) -> Mapping[str, Any]:
    """Load a representative row from a JSON file (object or non-empty list)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"{path}: row list is empty")
        payload = payload[0]
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected a JSON object row")
    return payload


def main(argv: list[str] | None = None) -> int:
    """Run the pre-submit conformance check. Returns the process exit code."""
    args = _parse_args(argv)
    spec = _CONTRACT_REGISTRY.get(args.contract_id)
    if spec is None:
        known = ", ".join(sorted(_CONTRACT_REGISTRY))
        message = f"unknown contract id {args.contract_id!r}; known contracts: {known}"
        if args.json:
            print(json.dumps({"error": message, "known_contracts": sorted(_CONTRACT_REGISTRY)}))
        else:
            print(f"error: {message}", file=sys.stderr)
        return 2

    try:
        if args.row is not None:
            row = _load_row(args.row)
        else:
            row = spec.representative_row()
        report = _evaluate(spec, row)
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
        message = f"preflight evaluation failed for {args.contract_id!r}: {exc}"
        if args.json:
            print(json.dumps({"contract_id": args.contract_id, "error": message}))
        else:
            print(f"error: {message}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render_human(report))
    return 0 if report["conforms"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
