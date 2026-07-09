#!/usr/bin/env python3
"""schema_version writer/reader coverage audit — Issue #4909.

Produces a machine-readable inventory classifying every distinct schema_version
value emitted by Python source into governance buckets:

  named-const-backed        – value equals a named *SCHEMA* constant
  test-asserted             – value checked via == in prod code or tests
  jsonschema-validated      – a .json schema file requires it with a const constraint
  provenance-write-only-by-design – value is intentional provenance, not consumed
  genuinely-orphaned-needs-validator – none of the above

Output: output/schema_version_audit_inventory.json
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXCLUDE_DIRS = {".venv", ".claude", "output", "__pycache__", ".git"}

# ── helpers ──────────────────────────────────────────────────────────────────


def _rg(pattern: str, globs: list[str] | None = None, cwd: Path = REPO_ROOT) -> str:
    """Run ripgrep and return stdout."""
    cmd = ["rg", "--no-heading", "-n", pattern]
    if globs:
        cmd += globs
    for d in EXCLUDE_DIRS:
        cmd += ["--glob", f"!{d}/*"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, check=False)
    return result.stdout


# ── Phase 1: collect all inline schema_version string-literal writers ────────

WRITER_RE = re.compile(r"""["']schema_version["']\s*:\s*["']([^"']+)["']""")
KWARG_RE = re.compile(r"""schema_version\s*=\s*["']([^"']+)["']""")
ASSIGN_RE = re.compile(r"""(?:self\.)?schema_version\s*=\s*["']([^"']+)["']""")

NEGATIVE_TEST_VALUES = {
    "wrong",
    "bogus",
    "invalid",
    "unexpected",
    "bad",
    "other",
    "x",
    "WrongLedger.v1",
    "v999",
    "invalid-schema",
    "bogus.v9",
    "example.v1",
    "test.v1",
    "wrong.v0",
}

# Additional patterns that indicate negative test fixtures (regex)
NEGATIVE_TEST_PATTERNS = [
    re.compile(r"\.v999$"),  # wrong-version tests
    re.compile(r"\.v0$"),  # wrong-version tests on v1 schemas
    re.compile(r"^fixture"),  # test fixtures
    re.compile(r"^camera-ready$"),  # non-schema test values
    re.compile(r"^smoke\.v"),  # smoke test fixtures
    re.compile(r"^track-v"),  # track test fixtures
]


def _is_negative_test_value(val: str) -> bool:
    if val in NEGATIVE_TEST_VALUES:
        return True
    return any(p.search(val) for p in NEGATIVE_TEST_PATTERNS)


def collect_writers() -> dict[str, list[dict]]:
    """Return {value: [{file, line, context}]} for all inline writers."""
    writers: dict[str, list[dict]] = defaultdict(list)

    # Pattern 1: "schema_version": "value"
    raw = _rg(r'"schema_version"\s*:\s*"', ["--glob", "*.py"])
    for m in re.finditer(r'^(.+?):(\d+):.*"schema_version"\s*:\s*"([^"]+)"', raw, re.MULTILINE):
        fpath, lineno, val = m.group(1), int(m.group(2)), m.group(3)
        if _is_negative_test_value(val):
            continue
        writers[val].append({"file": fpath, "line": lineno, "pattern": "dict-key"})

    # Pattern 2: schema_version="value" (kwarg)
    raw2 = _rg(r'schema_version\s*=\s*"', ["--glob", "*.py"])
    for m in re.finditer(r'^(.+?):(\d+):.*schema_version\s*=\s*"([^"]+)"', raw2, re.MULTILINE):
        fpath, lineno, val = m.group(1), int(m.group(2)), m.group(3)
        if _is_negative_test_value(val):
            continue
        writers[val].append({"file": fpath, "line": lineno, "pattern": "kwarg"})

    # Pattern 3: self.schema_version = "value" or schema_version = "value"
    raw3 = _rg(r'(?:self\.)?schema_version\s*=\s*"', ["--glob", "*.py"])
    for m in re.finditer(
        r'^(.+?):(\d+):.*(?:self\.)?schema_version\s*=\s*"([^"]+)"', raw3, re.MULTILINE
    ):
        fpath, lineno, val = m.group(1), int(m.group(2)), m.group(3)
        if _is_negative_test_value(val):
            continue
        writers[val].append({"file": fpath, "line": lineno, "pattern": "assign"})

    # Patterns 2 (kwarg) and 3 (assign) match the same physical lines
    # (``schema_version = "x"`` and ``schema_version="x"`` satisfy both regexes),
    # so every kwarg/assign writer site is recorded twice. Deduplicate by
    # (file, line) — keeping the first record — so the inventory's writer lists
    # and site counts are not inflated. Classification is per distinct value and
    # uses ``all``/``any`` over writes, so dedup cannot change any bucket.
    for val, writes in writers.items():
        seen: set[tuple[str, int]] = set()
        deduped: list[dict] = []
        for w in writes:
            key = (w["file"], w["line"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(w)
        writers[val] = deduped

    return writers


# ── Phase 2: collect named SCHEMA constants ─────────────────────────────────

# Tolerates an optional type annotation between the constant name and ``=``
# (e.g. ``EPISODE_SCHEMA_VERSION: str = "v1"``); without this, type-annotated
# constants are missed and their values get misclassified as ungoverned.
CONST_RE = re.compile(
    r'^(\S+\.py):(\d+):(\w*(?:SCHEMA|VERSION)\w*)\s*(?::[^=\n]*)?=\s*"([^"]+)"',
    re.MULTILINE,
)


def collect_named_constants() -> dict[str, list[dict]]:
    """Return {value: [{file, line, constant_name}]}."""
    constants: dict[str, list[dict]] = defaultdict(list)
    raw = _rg(r'(?:SCHEMA|VERSION).*=\s*"', ["--glob", "*.py"])
    for m in CONST_RE.finditer(raw):
        fpath, lineno, name, val = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        if "SCHEMA" in name or "VERSION" in name:
            constants[val].append(
                {
                    "file": fpath,
                    "line": lineno,
                    "constant": name,
                }
            )
    return constants


# ── Phase 3: collect equality-check readers ──────────────────────────────────


def collect_readers() -> dict[str, list[dict]]:
    """Return {value: [{file, line}]} for == checks against schema_version."""
    readers: dict[str, list[dict]] = defaultdict(list)
    raw = _rg(r'schema_version\s*==\s*"', ["--glob", "*.py"])
    for m in re.finditer(r'^(.+?):(\d+):.*schema_version\s*==\s*"([^"]+)"', raw, re.MULTILINE):
        fpath, lineno, val = m.group(1), int(m.group(2)), m.group(3)
        readers[val].append({"file": fpath, "line": lineno})
    # Also check for != with SCHEMA_VERSION constants (value comparison)
    raw2 = _rg(r'schema_version\s*!=\s*"', ["--glob", "*.py"])
    for m in re.finditer(r'^(.+?):(\d+):.*schema_version\s*!=\s*"([^"]+)"', raw2, re.MULTILINE):
        fpath, lineno, val = m.group(1), int(m.group(2)), m.group(3)
        readers[val].append({"file": fpath, "line": lineno})
    return readers


# ── Phase 4: collect jsonschema const constraints ────────────────────────────


def collect_jsonschema_constraints() -> dict[str, list[dict]]:
    """Return {value: [{file, required}]} for jsonschema const-constrained values."""
    constraints: dict[str, list[dict]] = defaultdict(list)

    # Find all JSON schema files (limit to schema directories for performance)
    schema_dirs = [
        d
        for d in REPO_ROOT.rglob("schemas")
        if d.is_dir() and not any(ex in str(d) for ex in EXCLUDE_DIRS)
    ]
    schema_files: list[Path] = []
    for d in schema_dirs:
        schema_files.extend(d.glob("*.json"))
    # Also check specs/contracts/
    specs_dir = REPO_ROOT / "specs"
    if specs_dir.exists():
        schema_files.extend(specs_dir.rglob("*.schema.json"))
        schema_files.extend(specs_dir.rglob("*.json"))

    for sf in schema_files:
        try:
            text = sf.read_text()
            data = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        # Walk the schema looking for "schema_version" property with "const"
        _walk_schema(data, str(sf.relative_to(REPO_ROOT)), constraints)

    return constraints


def _walk_schema(obj: object, fpath: str, out: dict[str, list[dict]]) -> None:
    """Recursively find schema_version const constraints."""
    if isinstance(obj, dict):
        props = obj.get("properties", {})
        if isinstance(props, dict) and "schema_version" in props:
            sv_prop = props["schema_version"]
            if isinstance(sv_prop, dict) and "const" in sv_prop:
                val = sv_prop["const"]
                if isinstance(val, str):
                    req = "schema_version" in (obj.get("required") or [])
                    out[val].append({"file": fpath, "required": req})
        for v in obj.values():
            _walk_schema(v, fpath, out)
    elif isinstance(obj, list):
        for item in obj:
            _walk_schema(item, fpath, out)


# ── Phase 5: provenance markers ─────────────────────────────────────────────

PROVENANCE_INDICATORS = [
    "provenance",
    "evidence",
    "audit",
    "report",
    "manifest",
    "claim",
    "packet",
    "preflight",
    "smoke",
    "diagnostic",
    "readiness",
    "comparison",
    "analysis",
    "screening",
    "sensitivity",
    "summary",
    "card",
    "record",
    "registry",
    "catalog",
    "profile",
]

# Values that are inherently write-only provenance (diagnostic/observational data)
# These are emitted for post-hoc analysis and never consumed by runtime code.
PROVENANCE_VALUE_PATTERNS = [
    re.compile(
        r"(report|summary|manifest|claim|card|packet|preflight|diagnostic|readiness|evidence|audit)\.v\d"
    ),
    re.compile(r"(comparison|analysis|screening|sensitivity|profile|registry|catalog)\.v\d"),
]


def is_provenance_context(fpath: str) -> bool:
    """Heuristic: files with provenance-like names."""
    lower = fpath.lower()
    return any(ind in lower for ind in PROVENANCE_INDICATORS)


def is_only_in_test_files(writers_list: list[dict]) -> bool:
    """Check if all writers are in test files."""
    return all("tests/" in w["file"] for w in writers_list)


# ── Phase 6: classify ────────────────────────────────────────────────────────


@dataclass
class SchemaVersionEntry:
    """Single schema_version value with its writers and classification."""

    value: str
    writers: list[dict] = field(default_factory=list)
    classification: str = "genuinely-orphaned-needs-validator"
    constant_backing: list[dict] = field(default_factory=list)
    readers: list[dict] = field(default_factory=list)
    jsonschema_backing: list[dict] = field(default_factory=list)
    recommended_action: str = ""


def classify_all(
    writers: dict[str, list[dict]],
    constants: dict[str, list[dict]],
    readers: dict[str, list[dict]],
    jsonschema: dict[str, list[dict]],
) -> list[SchemaVersionEntry]:
    """Classify each distinct schema_version value into a governance bucket."""
    entries: list[SchemaVersionEntry] = []
    for val, writes in sorted(writers.items()):
        e = SchemaVersionEntry(value=val, writers=writes)
        e.constant_backing = constants.get(val, [])
        e.readers = readers.get(val, [])
        e.jsonschema_backing = jsonschema.get(val, [])

        # Classification priority:
        # 1. jsonschema const-constraint (strongest governance)
        if e.jsonschema_backing:
            e.classification = "jsonschema-validated"
        # 2. named constant backing
        elif e.constant_backing:
            e.classification = "named-const-backed"
        # 3. equality check in prod or tests
        elif e.readers:
            e.classification = "test-asserted"
        # 4. provenance-only scripts (write-only by design)
        elif all(is_provenance_context(w["file"]) for w in writes):
            e.classification = "provenance-write-only-by-design"
            e.recommended_action = "document as provenance-only; no reader needed"
        # 5. test-only writers are likely fixtures, not genuinely orphaned prod code
        elif is_only_in_test_files(writes):
            e.classification = "provenance-write-only-by-design"
            e.recommended_action = (
                "test fixture; no reader needed unless prod code emits same value"
            )
        # 6. values whose name matches provenance patterns
        elif any(p.search(val) for p in PROVENANCE_VALUE_PATTERNS):
            e.classification = "provenance-write-only-by-design"
            e.recommended_action = "document as provenance-only; no reader needed"
        # 7. genuinely orphaned
        else:
            e.classification = "genuinely-orphaned-needs-validator"
            # Suggest action based on context
            if any("robot_sf/" in w["file"] for w in writes):
                e.recommended_action = "promote to named constant + add contract test or jsonschema"
            else:
                e.recommended_action = "promote to named constant or document as provenance-only"

        entries.append(e)
    return entries


# ── Phase 7: write inventory ─────────────────────────────────────────────────


def _build_inventory(entries: list[SchemaVersionEntry]) -> dict:
    """Build the JSON-serializable inventory dict from classified entries."""
    counts: dict[str, int] = defaultdict(int)
    for e in entries:
        counts[e.classification] += 1

    inventory: dict = {
        "issue": 4909,
        "title": "schema_version writer/reader coverage audit",
        "total_distinct_values": len(entries),
        "classification_counts": dict(counts),
        "genuinely_orphaned_count": counts.get("genuinely-orphaned-needs-validator", 0),
        "entries": [],
        "excluded_negative_test_fixtures": sorted(NEGATIVE_TEST_VALUES),
    }

    for e in entries:
        entry: dict = {
            "value": e.value,
            "classification": e.classification,
            "writers": e.writers,
        }
        if e.constant_backing:
            entry["constant_backing"] = e.constant_backing
        if e.readers:
            entry["readers"] = e.readers
        if e.jsonschema_backing:
            entry["jsonschema_backing"] = e.jsonschema_backing
        if e.recommended_action:
            entry["recommended_action"] = e.recommended_action
        inventory["entries"].append(entry)

    return inventory


def _print_summary(inventory: dict, entries: list[SchemaVersionEntry]) -> None:
    """Print human-readable summary to stdout."""
    counts = inventory["classification_counts"]
    print("\n=== Classification Summary ===")
    for bucket in [
        "jsonschema-validated",
        "named-const-backed",
        "test-asserted",
        "provenance-write-only-by-design",
        "genuinely-orphaned-needs-validator",
    ]:
        print(f"  {bucket}: {counts.get(bucket, 0)}")
    print(f"  TOTAL: {inventory['total_distinct_values']}")

    orphaned = [e for e in entries if e.classification == "genuinely-orphaned-needs-validator"]
    if orphaned:
        print(f"\n=== Genuinely Orphaned ({len(orphaned)}) ===")
        for e in orphaned:
            print(f"  {e.value}")
            for w in e.writers:
                print(f"    -> {w['file']}:{w['line']}")
            print(f"    Action: {e.recommended_action}")


def main() -> None:
    """Run the schema_version audit and write inventory to output/."""
    print("Phase 1: collecting writers...")
    writers = collect_writers()
    print(
        f"  Found {len(writers)} distinct schema_version values from {sum(len(v) for v in writers.values())} writer sites"
    )

    print("Phase 2: collecting named constants...")
    constants = collect_named_constants()
    print(f"  Found {len(constants)} distinct constant-backed values")

    print("Phase 3: collecting readers (equality checks)...")
    readers = collect_readers()
    print(f"  Found {len(readers)} distinct reader-checked values")

    print("Phase 4: collecting jsonschema constraints...")
    jsonschema_constraints = collect_jsonschema_constraints()
    print(f"  Found {len(jsonschema_constraints)} distinct jsonschema-constrained values")

    print("Phase 5: classifying...")
    entries = classify_all(writers, constants, readers, jsonschema_constraints)

    inventory = _build_inventory(entries)
    out_path = REPO_ROOT / "output" / "schema_version_audit_inventory.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(inventory, indent=2))
    print(f"\nInventory written to: {out_path}")

    _print_summary(inventory, entries)


if __name__ == "__main__":
    main()
