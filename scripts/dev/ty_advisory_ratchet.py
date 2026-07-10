#!/usr/bin/env python3
"""ty advisory diagnostic baseline + per-module downward ratchet (issue #5004).

This helper turns the existing *advisory* ``ty check`` scan (run in CI via
``uvx ty check . --exit-zero``) into a **monotone downward ratchet**: the total
``ty`` finding count per module may stay the same or decrease, but never increase.

What this owns (issue #5004)
----------------------------
The *general* ``ty`` advisory baseline + per-module ratchet over the ~2.7k
pre-existing findings. It is the type-safety counterpart to the security baseline
(``scripts/validation/check_broad_exceptions.py`` style ratchet).

Scope boundary — what is intentionally EXCLUDED
-----------------------------------------------
To avoid overlap with three sibling issues, the **optional-dependency import
resolution** subset of ``ty`` findings is recorded in the baseline but EXCLUDED
from the ratchet gate:

* ``#4990`` owns the optional-import *guard inventory* (AST ratchet on
  ``except ImportError`` spellings).
* ``#4995`` owns standardizing those guard spellings.
* ``#4988`` owns benchmark-CLI typed-error contracts.

Concretely, a finding is treated as the **optional-import category** (excluded
from the gate, tracked informationally) when::

    check_name == "unresolved-import"
    and description starts with "unresolved-import: Cannot resolve imported module"

i.e. a module that cannot be resolved at all (``GPUtil``, ``cairosvg``,
``adjustText``, ``rvo2``, ``ompl``, vendored relative imports, ...). Genuine
first-party *member-resolution* errors (``unresolved-import: Module X has no
member Y``) are KEPT in the general bucket because they are real type errors,
not optional-dependency noise. See ``exclusion`` in the baseline JSON.

Ratchet contract
----------------
* A **clean module** (general baseline count == 0) that gains any finding -> FAIL.
* A **tracked module** whose general count *increases* beyond its baseline -> FAIL.
* A **decrease** never fails; the helper prints a "ratchet opportunity" notice so
  the baseline can be refreshed to lock in the improvement (``--write-baseline``).
* A module NOT present in the baseline is treated as clean (baseline 0), so a
  brand-new module with findings fails until it is explicitly baselined.

Exit codes
----------
* ``0`` — ratchet holds (no per-module increase; clean modules stayed clean).
* ``1`` — a clean module regressed, or a tracked module's count increased.
* ``2`` — ``ty`` could not be run / produced unparseable output (infra error).

Usage
-----
::

    # Re-run ty and check against the committed baseline (CI / local gate).
    uv run python scripts/dev/ty_advisory_ratchet.py --check

    # Refresh the baseline after intentionally reducing findings.
    uv run python scripts/dev/ty_advisory_ratchet.py --write-baseline

    # Parse a pre-rendered ty gitlab-JSON report (offline / test / no-network).
    uv run python scripts/dev/ty_advisory_ratchet.py --check --ty-output report.json

    # Reconstruct the host-independent baseline-reproduction fixture from the
    # committed baseline (no live ty run); see issue #5070.
    uv run python scripts/dev/ty_advisory_ratchet.py --emit-baseline-fixture \
        --fixture scripts/validation/ty_advisory_findings_fixture.json

The committed baseline lives at ``scripts/validation/ty_advisory_baseline.json``.
The deterministic baseline-reproduction fixture lives at
``scripts/validation/ty_advisory_findings_fixture.json``; the baseline-reproduction
test parses it instead of re-running live ``ty`` so reproduction is host-independent
(issue #5070; the live scan remains a separate advisory diagnostic).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_BASELINE = Path("scripts/validation/ty_advisory_baseline.json")
# Deterministic, host-independent raw-findings fixture reconstructed from the
# committed baseline. The baseline-reproduction test parses THIS file (never a
# live ty run) so reproduction holds on every clean worktree regardless of host
# dependency-resolution state. See issue #5070.
DEFAULT_FIXTURE = Path("scripts/validation/ty_advisory_findings_fixture.json")
SCHEMA_VERSION = 1

# A finding is the "optional-import category" (excluded from the ratchet gate,
# owned by #4990/#4995) when it is an unresolved whole-module import. Member
# resolution errors ("Module X has no member Y") are NOT optional-import noise
# and stay in the general bucket.
OPTIONAL_IMPORT_RULE = "unresolved-import"
OPTIONAL_IMPORT_DESC_PREFIX = "unresolved-import: Cannot resolve imported module"

# Sibling issues that own the excluded categories. Recorded in the baseline so
# future contributors know where to take cross-cutting reductions.
OWNED_BY_OTHER_ISSUES = ["#4990", "#4995", "#4988"]


def module_of(path: str) -> str:
    """Return the per-module key for a repository-relative ``path``.

    ``robot_sf`` is grouped two levels deep (``robot_sf/<subpkg>``) because that
    matches how the package is reviewed and how the worked example in #5004 is
    scoped. ``robot_sf_carla_bridge`` and top-level dirs use one level.
    """
    parts = path.split("/")
    if parts[0] == "robot_sf" and len(parts) > 2:
        return "/".join(parts[:2])
    return parts[0]


def classify_finding(finding: dict[str, Any]) -> str:
    """Return ``"optional_import"`` or ``"general"`` for one ty gitlab finding.

    The optional-import category is excluded from the ratchet gate to avoid
    overlap with #4990/#4995 (see module docstring).
    """
    if finding.get("check_name") == OPTIONAL_IMPORT_RULE and str(
        finding.get("description", "")
    ).startswith(OPTIONAL_IMPORT_DESC_PREFIX):
        return "optional_import"
    return "general"


def run_ty(repo_root: Path) -> list[dict[str, Any]]:
    """Run ``ty check`` in gitlab-JSON mode and return parsed findings.

    Uses ``--exit-zero`` so advisory findings never fail the ty invocation
    itself; only this ratchet decides pass/fail. Raises ``RuntimeError`` on a
    non-zero ty exit that is not advisory, or on unparseable output.
    """
    cmd = ["uvx", "ty", "check", ".", "--output-format", "gitlab", "--exit-zero"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:  # e.g. uvx not installed / not on PATH
        raise RuntimeError(f"Could not invoke '{' '.join(cmd)}': {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"ty exited {proc.returncode} (expected 0 with --exit-zero).\n"
            f"stderr:\n{proc.stderr[:2000]}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Could not parse ty gitlab JSON output: {exc}\nstdout head:\n{proc.stdout[:1000]}"
        ) from exc


def load_ty_output(path: Path) -> list[dict[str, Any]]:
    """Load a pre-rendered ty gitlab-JSON report from ``path``."""
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate raw findings into baseline modules + rule summary.

    Returns a dict with ``modules`` (per-module general/optional/total counts),
    ``rules`` (per-rule counts), and scalar summaries.
    """
    modules: dict[str, dict[str, int]] = {}
    rules: Counter[str] = Counter()
    general_total = 0
    optional_total = 0
    for finding in findings:
        path = (finding.get("location") or {}).get("path", "<unknown>")
        mod = module_of(path)
        rules[str(finding.get("check_name", "unknown"))] += 1
        category = classify_finding(finding)
        entry = modules.setdefault(mod, {"general": 0, "optional_import_excluded": 0, "total": 0})
        entry["total"] += 1
        if category == "optional_import":
            entry["optional_import_excluded"] += 1
            optional_total += 1
        else:
            entry["general"] += 1
            general_total += 1
    # Stable ordering for reviewable diffs.
    modules_sorted = {k: modules[k] for k in sorted(modules)}
    rules_sorted = {k: rules[k] for k in sorted(rules)}
    return {
        "modules": modules_sorted,
        "rules": rules_sorted,
        "general_total": general_total,
        "optional_import_total": optional_total,
        "total": len(findings),
    }


def materialize_findings_from_baseline(baseline: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct deterministic raw ty gitlab-JSON findings from a baseline.

    This is the host-independent inverse of :func:`aggregate` used to make the
    baseline-reproduction test deterministic (issue #5070): the live ``ty`` scan
    is non-reproducible across hosts because findings depend on each host's
    dependency-resolution state, so a committed fixture reconstructed from the
    baseline's own per-module counts is used instead.

    For every module in ``baseline["modules"]`` this emits:

    * ``general`` findings with a general ``check_name``
      (``invalid-argument-type``) so they land in the ratcheted general bucket; and
    * ``optional_import_excluded`` findings with the optional-import
      ``check_name``/description prefix so they land in the excluded bucket.

    The result is **stable** (sorted by module then index) so regenerating the
    fixture always produces byte-identical output for reviewable diffs, and
    :func:`aggregate` over the result reproduces the baseline's per-module
    ``general``/``optional_import_excluded``/``total`` counts exactly.
    """
    modules = baseline.get("modules", {})
    findings: list[dict[str, Any]] = []
    for mod in sorted(modules):
        counts = modules[mod]
        general_n = int(counts.get("general", 0))
        optional_n = int(counts.get("optional_import_excluded", 0))
        # A representative .py path inside the module. ``module_of`` maps a
        # ``robot_sf/<sub>/...`` path back to ``robot_sf/<sub>``, and any other
        # top-level dir back to itself, so the materialized path re-keying is
        # consistent with how real ty findings are bucketed.
        base_path = f"{mod}/_ty_baseline_fixture.py"
        for i in range(general_n):
            findings.append(
                {
                    "check_name": "invalid-argument-type",
                    "description": (
                        f"invalid-argument-type: baseline-reproduction fixture #{i} "
                        f"for module '{mod}'"
                    ),
                    "severity": "major",
                    "fingerprint": f"{base_path}:{i + 1}:general",
                    "location": {
                        "path": base_path,
                        "positions": {"begin": {"line": i + 1, "column": 1}},
                    },
                }
            )
        for j in range(optional_n):
            findings.append(
                {
                    "check_name": OPTIONAL_IMPORT_RULE,
                    "description": (
                        f"{OPTIONAL_IMPORT_DESC_PREFIX} `_ty_optional_{j}` "
                        f"(baseline-reproduction fixture for module '{mod}')"
                    ),
                    "severity": "minor",
                    "fingerprint": f"{base_path}:{general_n + j + 1}:optional",
                    "location": {
                        "path": base_path,
                        "positions": {"begin": {"line": general_n + j + 1, "column": 1}},
                    },
                }
            )
    return findings


def build_baseline_payload(
    findings: list[dict[str, Any]],
    *,
    ty_version: str | None,
) -> dict[str, Any]:
    """Build the versioned baseline JSON payload from raw findings."""
    agg = aggregate(findings)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ty_version": ty_version,
        "description": (
            "ty advisory diagnostic baseline + per-module downward ratchet "
            "(issue #5004). The ratchet gates on the 'general' bucket; the "
            "'optional_import_excluded' bucket is recorded for visibility but "
            "owned by #4990/#4995. Refresh with "
            "`scripts/dev/ty_advisory_ratchet.py --write-baseline` after "
            "intentionally reducing findings."
        ),
        "exclusion": {
            "rule": (
                f"Findings with check_name='{OPTIONAL_IMPORT_RULE}' and "
                f"description prefix '{OPTIONAL_IMPORT_DESC_PREFIX}...' are "
                "treated as the optional-import category: recorded but EXCLUDED "
                "from the ratchet gate (owned by #4990/#4995)."
            ),
            "owned_by_other_issues": OWNED_BY_OTHER_ISSUES,
        },
        "summary": {
            "total_findings": agg["total"],
            "general_findings": agg["general_total"],
            "optional_import_findings_excluded": agg["optional_import_total"],
            "module_count": len(agg["modules"]),
        },
        "modules": agg["modules"],
        "rules": agg["rules"],
    }


def load_baseline(path: Path) -> dict[str, Any]:
    """Load and minimally validate a baseline file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Baseline {path} must be a JSON object, got {type(data).__name__}.")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported baseline schema_version in {path}: "
            f"got {data.get('schema_version')}, expected {SCHEMA_VERSION}"
        )
    if not isinstance(data.get("modules"), dict):
        raise ValueError(f"Baseline {path} is missing a valid 'modules' mapping.")
    return data


def _detect_ty_version(repo_root: Path) -> str | None:
    """Best-effort detect the ty version for baseline provenance."""
    try:
        proc = subprocess.run(
            ["uvx", "ty", "--version"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        # Best-effort provenance only; uvx may be absent in offline/test modes.
        return None
    if proc.returncode == 0:
        return proc.stdout.strip() or None
    return None


def check_against_baseline(
    current_findings: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Return (failures, notices) for the downward ratchet.

    ``failures`` is non-empty -> the ratchet is broken (exit 1). ``notices`` are
    informational ratchet-opportunity hints (counts decreased) and are always
    advisory.
    """
    current_agg = aggregate(current_findings)
    current_modules = current_agg["modules"]
    baseline_modules = baseline.get("modules", {})

    failures: list[str] = []
    notices: list[str] = []

    all_modules = sorted(set(current_modules) | set(baseline_modules))
    for mod in all_modules:
        base_general = int(baseline_modules.get(mod, {}).get("general", 0))
        cur_general = int(current_modules.get(mod, {}).get("general", 0))
        if cur_general > base_general:
            if base_general == 0:
                failures.append(
                    f"clean module regressed: '{mod}' went from 0 to "
                    f"{cur_general} general ty findings."
                )
            else:
                failures.append(
                    f"module '{mod}' general ty findings increased from "
                    f"{base_general} to {cur_general} (downward ratchet)."
                )
        elif cur_general < base_general:
            notices.append(
                f"ratchet opportunity: '{mod}' dropped from {base_general} to "
                f"{cur_general} general findings; refresh the baseline to lock "
                f"in the improvement."
            )

    # Overall monotonicity summary (advisory; the per-module gate is authoritative).
    base_general_total = sum(int(m.get("general", 0)) for m in baseline_modules.values())
    if current_agg["general_total"] > base_general_total:
        failures.append(
            f"total general findings increased from {base_general_total} to "
            f"{current_agg['general_total']}."
        )
    return failures, notices


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    """Write stable, reviewable, sort-keyed JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repo_root() -> Path:
    """Return the current Git repository root."""
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("Could not determine git repository root.")
    return Path(proc.stdout.strip())


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Run the ratchet gate.")
    mode.add_argument(
        "--write-baseline",
        action="store_true",
        help="Recompute findings and (re)write the baseline file.",
    )
    mode.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Print the aggregate (per-module counts) without reading/writing a baseline.",
    )
    mode.add_argument(
        "--emit-baseline-fixture",
        action="store_true",
        help=(
            "Reconstruct a deterministic raw-findings fixture from the committed "
            "baseline (NOT a live ty run). Use with --fixture PATH. "
            "Host-independent baseline reproduction (#5070)."
        ),
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root (defaults to git toplevel).",
    )
    parser.add_argument(
        "--ty-output",
        type=Path,
        default=None,
        help=(
            "Path to a pre-rendered ty gitlab-JSON report. When set, ty is NOT "
            "re-run; the report is parsed instead (offline / test mode)."
        ),
    )
    parser.add_argument(
        "--emit-findings",
        type=Path,
        default=None,
        help="Optional path to write the raw ty findings (gitlab JSON) for offline reuse.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE,
        help=(
            "Output path for --emit-baseline-fixture (the deterministic "
            "host-independent raw-findings fixture). "
            f"Default: {DEFAULT_FIXTURE}."
        ),
    )
    return parser.parse_args(argv)


def _gather_findings(args: argparse.Namespace, repo_root: Path) -> list[dict[str, Any]]:
    """Resolve raw findings either by running ty or by parsing --ty-output."""
    if args.ty_output is not None:
        return load_ty_output(args.ty_output)
    return run_ty(repo_root)


def _emit_baseline_fixture(args: argparse.Namespace, repo_root: Path, baseline_path: Path) -> int:
    """Reconstruct + write the deterministic baseline-reproduction fixture (#5070).

    Does NOT run ty; the fixture is reconstructed from the committed baseline so
    reproduction is host-independent and offline.
    """
    if not baseline_path.exists():
        print(
            f"ERROR: baseline not found at {baseline_path}. "
            f"Generate it with --write-baseline first.",
            file=sys.stderr,
        )
        return 2
    baseline = load_baseline(baseline_path)
    fixture = materialize_findings_from_baseline(baseline)
    fixture_path = args.fixture if args.fixture.is_absolute() else repo_root / args.fixture
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(fixture_path, fixture)
    agg = aggregate(fixture)
    print(
        f"Wrote deterministic ty findings fixture to {fixture_path}: "
        f"{agg['general_total']} general / {agg['total']} total findings "
        f"across {len(agg['modules'])} modules (reconstructed from baseline, "
        f"no live ty run)."
    )
    return 0


def _print_aggregate(findings: list[dict[str, Any]]) -> None:
    """Print the per-module aggregate report for ``--aggregate-only``."""
    agg = aggregate(findings)
    print(
        f"total={agg['total']} general={agg['general_total']} "
        f"optional_import_excluded={agg['optional_import_total']} "
        f"modules={len(agg['modules'])}"
    )
    for mod, counts in agg["modules"].items():
        print(f"  {counts['general']:5d} general  ({counts['total']:5d} total)  {mod}")


def _report_check(
    payload: dict[str, Any], baseline: dict[str, Any], failures: list[str], notices: list[str]
) -> int:
    """Print the ``--check`` ratchet result and return the exit code."""
    print(
        f"ty advisory ratchet: general={payload['summary']['general_findings']} "
        f"(baseline general={sum(int(m.get('general', 0)) for m in baseline.get('modules', {}).values())}), "
        f"optional_import_excluded={payload['summary']['optional_import_findings_excluded']}."
    )
    for notice in notices:
        print(f"NOTICE: {notice}")
    if failures:
        print("\nty advisory ratchet FAILED (downward ratchet violated):", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\nFix the new findings, or refresh the baseline with "
            "`scripts/dev/ty_advisory_ratchet.py --write-baseline` if the "
            "increase is intentional and reviewed.",
            file=sys.stderr,
        )
        return 1
    print("ty advisory ratchet passed: no per-module increase; clean modules stayed clean.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the ratchet gate, baseline refresh, or aggregate report."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = args.root.resolve() if args.root is not None else _repo_root()
    baseline_path = args.baseline if args.baseline.is_absolute() else repo_root / args.baseline

    # --emit-baseline-fixture reconstructs a deterministic fixture from the
    # committed baseline WITHOUT running ty, so it is fully host-independent
    # and offline. See issue #5070.
    if args.emit_baseline_fixture:
        return _emit_baseline_fixture(args, repo_root, baseline_path)

    try:
        findings = _gather_findings(args, repo_root)
    except RuntimeError as exc:
        print(f"ERROR: could not obtain ty findings: {exc}", file=sys.stderr)
        return 2

    if args.emit_findings is not None:
        write_json(args.emit_findings, findings)

    if args.aggregate_only:
        _print_aggregate(findings)
        return 0

    payload = build_baseline_payload(findings, ty_version=_detect_ty_version(repo_root))

    if args.write_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(baseline_path, payload)
        print(
            f"Wrote ty baseline to {baseline_path}: "
            f"{payload['summary']['general_findings']} general / "
            f"{payload['summary']['total_findings']} total findings across "
            f"{payload['summary']['module_count']} modules."
        )
        return 0

    # --check
    if not baseline_path.exists():
        print(
            f"ERROR: baseline not found at {baseline_path}. "
            f"Generate it with --write-baseline first.",
            file=sys.stderr,
        )
        return 2
    baseline = load_baseline(baseline_path)
    failures, notices = check_against_baseline(findings, baseline)
    return _report_check(payload, baseline, failures, notices)


if __name__ == "__main__":
    raise SystemExit(main())
