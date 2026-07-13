#!/usr/bin/env python3
"""Regenerate the optional-import guard snapshot (issue #4990).

Walks ``robot_sf/`` with the same AST collector used by
``tests/test_optional_import_guard_inventory.py`` and rewrites
``tests/fixtures/optional_import_guards.json`` in place.

Usage::

    uv run python scripts/dev/generate_optional_import_snapshot.py

Use this after intentionally adding/removing/blessing an optional-import guard.
The PR must explain the delta (new dependency, justified broad catch, or a
migration onto ``robot_sf.common.optional_import.try_import``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "optional_import_guards.json"
TEST_MODULE = REPO_ROOT / "tests" / "test_optional_import_guard_inventory.py"

# Human-readable justification for each observed spelling. New broad catches
# must be added here when blessed, so the snapshot explains *why* the broad
# catch is legitimate rather than a swallowed bug.
NOTES = {
    "ImportError": (
        "Pure optional-import. Prefer "
        "robot_sf.common.optional_import.try_import for new plain cases."
    ),
    "ModuleNotFoundError": (
        "Pure optional-import (ModuleNotFoundError spelling). Prefer "
        "robot_sf.common.optional_import.try_import for new plain cases."
    ),
    "ImportError+ModuleNotFoundError": (
        "Optional-import covering both ImportError spellings; acceptable. "
        "Prefer try_import for new plain cases."
    ),
    "ImportError+RuntimeError": (
        "Blessed broad catch: guarded plotting / native-lib load may raise "
        "RuntimeError. Do not collapse to ImportError."
    ),
    "ImportError+OSError": (
        "Blessed broad catch: native-lib / filesystem load may raise OSError. "
        "campaign_runtime_preflight.py is a deliberate import PROBE (issue #5300): it "
        "exists to catch and report arm-dependency import failures with a remediation "
        "message, so try_import (which discards the exception) is the wrong tool there."
    ),
    "ImportError+SyntaxError": (
        "Blessed broad catch: parser/extension load may surface SyntaxError."
    ),
    "AttributeError+ImportError": (
        "Blessed broad catch: compiled-backend entry point may raise "
        "AttributeError when a native wheel component is absent."
    ),
    "AttributeError+ImportError+ModuleNotFoundError": (
        "Blessed broad catch: registry/compiled-backend boundary."
    ),
    "AttributeError+ImportError+OSError": ("Blessed broad catch: optional registry boundary."),
    "AttributeError+ImportError+RuntimeError": ("Blessed broad catch: optional native boundary."),
    "AttributeError+ImportError+TypeError": ("Blessed broad catch: optional dispatch boundary."),
    "AttributeError+ImportError+OSError+RuntimeError+TypeError+ValueError": (
        "Blessed broad catches: defensive visualization ingest and "
        "authorization-gated episode execution boundaries."
    ),
    "ImportError+OSError+RuntimeError+TypeError+ValueError": (
        "Blessed broad catch: defensive JSON/config ingest boundary. Review "
        "carefully before adding more."
    ),
    "ImportError+OSError+RuntimeError+TypeError+ValueError+json.JSONDecodeError": (
        "Blessed broad catch: defensive resource-lifecycle ingest boundary. "
        "Review carefully before adding more."
    ),
    "ImportError+OSError+RuntimeError+ValueError": (
        "Blessed broad catch: defensive ingest boundary. Review carefully."
    ),
    "AttributeError+ImportError+IndexError+KeyError+OSError+RuntimeError+TypeError+ValueError": (
        "Blessed broad catch: defensive best-effort ingest boundary. Very broad; "
        "do not extend further without strong justification."
    ),
    "AttributeError+ImportError+ModuleNotFoundError+OSError+RuntimeError+SyntaxError+TypeError+ValueError": (
        "Blessed broad catch: defensive best-effort boundary. Very broad; do not "
        "extend further without strong justification."
    ),
}


def _load_collector() -> tuple[Path, object]:
    """Import the collector from the test module without a package context."""
    spec = importlib.util.spec_from_file_location("_optional_import_ratchet_collector", TEST_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load collector from {TEST_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return REPO_ROOT / "robot_sf", module.collect_optional_import_guards


def main() -> int:
    """Regenerate the optional-import guard snapshot fixture from robot_sf/."""
    scanned_root, collect = _load_collector()
    inventory = collect(scanned_root)

    spellings = {}
    for key, info in sorted(inventory.items()):
        spellings[key] = {
            "count_ceiling": info["count"],
            "has_as_count": info["has_as_count"],
            "pragma_no_cover_count": info["pragma_no_cover_count"],
            "blessed": True,
            "note": NOTES.get(
                key,
                "Blessed broad catch. Add a justification note in "
                "scripts/dev/generate_optional_import_snapshot.py when blessing.",
            ),
            "samples": info["samples"],
        }

    document = {
        "description": (
            "AST inventory snapshot of optional-import guards in robot_sf/ "
            "(issue #4990). Enforced by "
            "tests/test_optional_import_guard_inventory.py. Regenerate with "
            "scripts/dev/generate_optional_import_snapshot.py. Each 'count_ceiling' "
            "is pinned to the exact observed count; growth fails the ratchet."
        ),
        "scanned_root": "robot_sf",
        "tracked_exceptions": ["ImportError", "ModuleNotFoundError"],
        "spelling_key": (
            "Caught exception type names, sorted and joined by '+'. The "
            "'as <alias>' name is intentionally NOT part of the key."
        ),
        "total_guards": sum(int(v["count_ceiling"]) for v in spellings.values()),
        "spellings": spellings,
    }

    FIXTURE.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"Wrote {FIXTURE.relative_to(REPO_ROOT)}")
    print(f"  total guards: {document['total_guards']}")
    print(f"  distinct spellings: {len(spellings)}")
    for key, info in sorted(spellings.items()):
        print(f"    {int(info['count_ceiling']):3d}  {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
