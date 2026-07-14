"""AST inventory ratchet for optional-import guards (issue #4990).

This test walks ``robot_sf/`` and collects every ``except`` handler whose caught
exception types include ``ImportError`` or ``ModuleNotFoundError``. It then
compares the observed inventory against a committed allow-list snapshot at
``tests/fixtures/optional_import_guards.json``.

Why an inventory ratchet instead of a mass rewrite?
---------------------------------------------------
A blanket rewrite to a single ``except ImportError`` is **not** safe: several
guards intentionally also catch ``RuntimeError``/``OSError``/``AttributeError``
for real reasons (guarded plotting, native-lib load failures). So this test is a
*ratchet*: it characterizes today's tree and fails only when a **new, un-blessed
spelling** appears or when the count of an existing spelling **grows**, forcing a
reviewer to either adopt :func:`robot_sf.common.optional_import.try_import` or
explicitly bless a justified broad catch in the snapshot.

Acceptance contract (issue #4990)
---------------------------------
* Introducing a new optional-import guard spelling not in the allow-list makes
  the test fail with a clear message.
* Existing intentional broad catches remain unchanged and blessed in the
  snapshot.
* No runtime behavior change (this is a static AST check).
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED_ROOT = REPO_ROOT / "robot_sf"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "optional_import_guards.json"

# Exception type names this ratchet tracks. ``ModuleNotFoundError`` is a
# subclass of ``ImportError``, but the codebase spells both; we track either.
_TRACKED = {"ImportError", "ModuleNotFoundError"}


def _exception_name(node: ast.expr) -> str | None:
    """Return a dotted exception name for a simple AST exception expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _exception_name(node.value)
        return f"{parent}.{node.attr}" if parent is not None else None
    return None


def _caught_type_names(node: ast.ExceptHandler) -> list[str]:
    """Return the sorted, de-duplicated names of exceptions a handler catches.

    Returns an empty list for handlers whose catch clause is not a simple name,
    dotted attribute, or tuple of them (e.g. ``except SomeError()``); those are
    out of scope.
    """
    t = node.type
    if t is None:
        return []
    if isinstance(t, (ast.Name, ast.Attribute)):
        name = _exception_name(t)
        names = [name] if name is not None else []
    elif isinstance(t, ast.Tuple):
        names = []
        for element in t.elts:
            name = _exception_name(element)
            if name is not None:
                names.append(name)
    else:
        return []
    return sorted(set(names))


def _spelling_key(names: list[str]) -> str:
    """Normalize a caught-type-set into a stable allow-list key.

    The alias name (``as exc`` vs ``as e``) is deliberately *not* part of the
    key: it is cosmetic and does not change which exceptions are swallowed. The
    meaningful axis -- "does this guard swallow too much?" -- is the set of
    caught types.
    """
    return "+".join(names)


def _has_pragma(line: str) -> bool:
    """True if a source line carries a ``# pragma: no cover`` marker."""
    return bool(re.search(r"#\s*pragma:\s*no cover", line))


def collect_optional_import_guards(root: Path) -> dict[str, dict[str, object]]:
    """Walk ``root`` and return the optional-import guard inventory.

    The returned mapping is keyed by spelling (caught-type-set) and records the
    current count plus one sample ``file:line`` location per occurrence.
    """
    occurrences: dict[str, list[str]] = defaultdict(list)
    has_as_counts: Counter[str] = Counter()
    pragma_counts: Counter[str] = Counter()

    for path in sorted(root.rglob("*.py")):
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(src, filename=str(path))
        except SyntaxError:
            continue
        lines = src.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            names = _caught_type_names(node)
            if not (_TRACKED & set(names)):
                continue
            key = _spelling_key(names)
            rel = path.relative_to(REPO_ROOT).as_posix()
            occurrences[key].append(f"{rel}:{node.lineno}")
            if node.name:
                has_as_counts[key] += 1
            handler_line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
            if _has_pragma(handler_line):
                pragma_counts[key] += 1

    inventory: dict[str, dict[str, object]] = {}
    for key, locs in sorted(occurrences.items()):
        inventory[key] = {
            "count": len(locs),
            "has_as_count": has_as_counts[key],
            "pragma_no_cover_count": pragma_counts[key],
            "samples": locs,
        }
    return inventory


def _load_fixture() -> dict[str, object]:
    if not FIXTURE.exists():
        pytest.fail(
            f"Snapshot fixture missing: {FIXTURE.relative_to(REPO_ROOT)}.\n"
            "Generate it with: "
            "`uv run python scripts/dev/generate_optional_import_snapshot.py`."
        )
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class TestOptionalImportGuardInventory:
    """Characterization ratchet over optional-import guards in ``robot_sf/``."""

    @pytest.mark.base_sensitive
    def test_collector_preserves_dotted_exception_names(self) -> None:
        """A qualified broad exception must not be collapsed into ImportError."""
        tree = ast.parse(
            "try:\n"
            "    import optional_dep\n"
            "except (ImportError, native_loader.LoadError):\n"
            "    optional_dep = None\n"
        )
        handler = next(node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler))

        assert _caught_type_names(handler) == ["ImportError", "native_loader.LoadError"]

    @pytest.mark.base_sensitive
    def test_no_new_unblessed_spelling_and_no_count_growth(self) -> None:
        """The inventory must stay within the blessed snapshot envelope.

        Fails (with an actionable message) when:
          * a caught-type-set appears that is not in the allow-list, or
          * the count of an existing spelling exceeds its blessed ceiling.

        Decreasing a count is allowed (the ratchet only blocks regression); the
        message will note the lower observed number so the snapshot can be
        tightened at leisure.
        """
        fixture = _load_fixture()
        spellings = fixture.get("spellings", {})
        if not isinstance(spellings, dict) or not spellings:
            pytest.fail(
                f"Snapshot fixture {FIXTURE.relative_to(REPO_ROOT)} has no "
                "'spellings' mapping; regenerate it."
            )

        observed = collect_optional_import_guards(SCANNED_ROOT)

        problems: list[str] = []

        # 1. New or grown spellings -> fail.
        for key, info in sorted(observed.items()):
            observed_count = int(info["count"])  # type: ignore[arg-type]
            if key not in spellings:
                problems.append(
                    f"NEW unblessed optional-import spelling '{key}' "
                    f"({observed_count} occurrence(s)). Either rewrite the plain "
                    "optional-import case(s) using "
                    "`robot_sf.common.optional_import.try_import`, or bless this "
                    "spelling in tests/fixtures/optional_import_guards.json with a "
                    "one-line justification. Locations:\n    "
                    + "\n    ".join(str(x) for x in info["samples"])  # type: ignore[arg-type]
                )
                continue
            blessed = spellings[key]
            ceiling = int(blessed.get("count_ceiling", blessed.get("count", 0)))
            if observed_count > ceiling:
                problems.append(
                    f"RATCHET VIOLATION: spelling '{key}' grew from ceiling "
                    f"{ceiling} to {observed_count}. Prefer "
                    "`robot_sf.common.optional_import.try_import` for new plain "
                    "optional imports, or raise the ceiling with justification. "
                    "Locations:\n    " + "\n    ".join(str(x) for x in info["samples"])  # type: ignore[arg-type]
                )

        # 2. Stale fixture entries (spelling no longer present) and lower counts
        #    are informational only: they never fail CI so removing guards is
        #    always safe. Surface them via the assertion message when something
        #    else already failed, to help tighten the snapshot.
        shrink_notes: list[str] = []
        for key, blessed in sorted(spellings.items()):
            ceiling = int(blessed.get("count_ceiling", blessed.get("count", 0)))
            if key not in observed:
                shrink_notes.append(
                    f"  - spelling '{key}' no longer present in robot_sf/ "
                    f"(ceiling was {ceiling}); safe to remove from snapshot."
                )
            elif int(observed[key]["count"]) < ceiling:  # type: ignore[index,arg-type]
                shrink_notes.append(
                    f"  - spelling '{key}' shrank from ceiling {ceiling} to "
                    f"{observed[key]['count']}; safe to lower the ceiling."  # type: ignore[index]
                )

        if problems:
            msg = "\n".join(problems)
            if shrink_notes:
                msg += "\n\nShrink opportunities (not failures):\n" + "\n".join(shrink_notes)
            pytest.fail(msg)

    @pytest.mark.base_sensitive
    def test_no_first_party_import_wrapped_in_broad_catch(self) -> None:
        """Regression guard for issue #5287.

        A first-party import of ``robot_sf.common.artifact_paths.get_repository_root``
        was previously wrapped in a broad ``except (ImportError, RuntimeError,
        OSError)`` guard inside
        ``robot_sf/benchmark/predictive_checkpoint_schema_audit.py``. That guard
        could not actually fire: ``get_repository_root`` is a first-party, pure-path
        helper with no optional or native dependencies, and 20+ other modules import
        it unguarded. The broad catch was therefore unjustified and tripped this
        inventory ratchet as a new ``ImportError+OSError+RuntimeError`` spelling.

        This test pins the fix: the exact 3-element spelling must never reappear in
        ``robot_sf/``. If a future change genuinely needs that broad catch, it must
        come with strong justification and be blessed in the snapshot instead of
        silently reintroducing the regression.
        """
        observed = collect_optional_import_guards(SCANNED_ROOT)
        assert "ImportError+OSError+RuntimeError" not in observed, (
            "The 'ImportError+OSError+RuntimeError' spelling reappeared in robot_sf/. "
            "This exact spelling was removed in issue #5287 because it guarded a "
            "first-party pure-path import that cannot raise these exceptions. "
            "Either drop the unjustified broad catch or, if genuinely required, bless "
            "a *specific* spelling in tests/fixtures/optional_import_guards.json with "
            "justification. Locations:\n    "
            + "\n    ".join(str(x) for x in observed["ImportError+OSError+RuntimeError"]["samples"])
        )

    @pytest.mark.base_sensitive
    def test_snapshot_matches_collector_output(self) -> None:
        """The committed counts must equal what the collector reports today.

        This keeps the snapshot honest: every ceiling is pinned to the exact
        current count (a true characterization), so any single addition or
        removal surfaces as a diff the author must consciously update. When the
        change is intentional, regenerate the fixture with
        ``scripts/dev/generate_optional_import_snapshot.py`` and explain the
        delta in the PR.
        """
        fixture = _load_fixture()
        spellings = fixture.get("spellings", {})
        observed = collect_optional_import_guards(SCANNED_ROOT)

        mismatches: list[str] = []
        all_keys = sorted(set(spellings) | set(observed))
        for key in all_keys:
            observed_count = int(observed.get(key, {}).get("count", 0))  # type: ignore[union-attr]
            if key in spellings:
                ceiling = int(spellings[key].get("count_ceiling", spellings[key].get("count", 0)))
            else:
                ceiling = -1
            if observed_count != ceiling:
                mismatches.append(f"  - '{key}': snapshot={ceiling} observed={observed_count}")

        if mismatches:
            pytest.fail(
                "Optional-import guard snapshot is out of sync with robot_sf/.\n"
                "If this change is intentional, regenerate and commit:\n"
                "  uv run python scripts/dev/generate_optional_import_snapshot.py\n"
                "Delta (snapshot_ceiling -> observed_count):\n" + "\n".join(mismatches)
            )

    @pytest.mark.base_sensitive
    def test_snapshot_notes_match_generator_notes(self) -> None:
        """Each committed note must equal the generator's NOTES value for that key.

        This prevents the generator from silently overwriting hand-edited fixture
        notes with stale values. When a note is improved in the fixture, the
        generator's NOTES dict must be updated to match, or the generator must be
        run to regenerate the fixture (which will fail this test until NOTES is
        updated). See issue #5475.
        """
        import importlib.util
        import sys

        generator_path = REPO_ROOT / "scripts" / "dev" / "generate_optional_import_snapshot.py"
        spec = importlib.util.spec_from_file_location("_generator", generator_path)
        if spec is None or spec.loader is None:
            pytest.fail(f"Could not load generator from {generator_path}")
        generator_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = generator_module
        spec.loader.exec_module(generator_module)
        generator_notes = getattr(generator_module, "NOTES", {})

        fixture = _load_fixture()
        spellings = fixture.get("spellings", {})

        mismatches: list[str] = []
        for key in sorted(spellings.keys()):
            fixture_note = spellings.get(key, {}).get("note", "")
            generator_note = generator_notes.get(key, "")
            if fixture_note != generator_note:
                if not fixture_note:
                    mismatches.append(
                        f"  - '{key}': fixture has no note, generator has: {generator_note!r}"
                    )
                elif not generator_note:
                    mismatches.append(
                        f"  - '{key}': generator has no note, fixture has: {fixture_note!r}"
                    )
                else:
                    mismatches.append(
                        f"  - '{key}': fixture note={fixture_note!r} != generator note={generator_note!r}"
                    )

        if mismatches:
            pytest.fail(
                "Optional-import guard snapshot notes have drifted from the generator's NOTES.\n"
                "To fix: update NOTES in scripts/dev/generate_optional_import_snapshot.py to "
                "match the fixture notes, then regenerate the fixture to confirm idempotence.\n"
                "Drift:\n" + "\n".join(mismatches)
            )


class TestTryImportHelper:
    """Direct contract tests for the canonical helper."""

    def test_returns_module_when_available(self) -> None:
        from robot_sf.common.optional_import import try_import

        module = try_import("json")
        import json as stdlib_json

        assert module is stdlib_json

    def test_returns_none_when_missing(self) -> None:
        from robot_sf.common.optional_import import try_import

        result = try_import("definitely_not_a_real_module_zzz_4990")
        assert result is None

    def test_returns_submodule_when_available(self) -> None:
        import json as stdlib_json

        from robot_sf.common.optional_import import try_import

        decoder = try_import("json.decoder")
        assert decoder is stdlib_json.decoder

    def test_does_not_swallow_runtime_error(self) -> None:
        """A broken-but-importable dependency must surface, not be masked.

        This is the core safety property: ``try_import`` catches only
        ``ImportError`` and never broader failures.
        """
        import sys

        from robot_sf.common.optional_import import try_import

        # Inject a fake module that raises RuntimeError on attribute access to
        # simulate a broken native dependency, then prove RuntimeError survives.
        fake_name = "_fake_broken_optional_dep_4990"

        class _BrokenLoader:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == fake_name:
                    from importlib.machinery import ModuleSpec

                    return ModuleSpec(fake_name, self)
                return None

            def create_module(self, spec):
                raise RuntimeError("simulated broken native dependency")

            def exec_module(self, module):
                raise RuntimeError("simulated broken native dependency")

        loader = _BrokenLoader()
        sys.meta_path.insert(0, loader)
        try:
            with pytest.raises(RuntimeError, match="simulated broken native dependency"):
                try_import(fake_name)
        finally:
            sys.meta_path.remove(loader)
            sys.modules.pop(fake_name, None)
