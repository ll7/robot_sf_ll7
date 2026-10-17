"""Release decision records must remain complete and unambiguous.

Historical records retain structural validation against their frozen source.
Future records additionally resolve qualified Python test enforcement references;
this does not establish that a recorded choice is implemented or that a test passes.
"""

import ast
import re
from pathlib import Path

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[2] / "docs" / "release"
REQUIRED_FIELDS = (
    "Question",
    "Choice",
    "Decided by",
    "Reason",
    "Alternatives",
    "Evidence",
    "Enforced by",
)
HEADING = re.compile(r"^### (D-\d{3,}): .+$", re.MULTILINE)
DECISION_HEADING = re.compile(r"^#{2,4}\s*D-\d+[^\n]*", re.MULTILINE)
FIELD = re.compile(r"^- \*\*([^*]+):\*\*\s*(.*?)(?=^- \*\*|^#{1,3} |\Z)", re.MULTILINE | re.DOTALL)


def validate_ledger_structure(text: str) -> None:
    """Reject heading drift, duplicate IDs or absent/empty required fields."""
    for candidate in DECISION_HEADING.finditer(text):
        assert HEADING.fullmatch(candidate.group()), (
            f"invalid decision heading: {candidate.group()}"
        )
    headings = list(HEADING.finditer(text))
    assert headings, "no decision entries"
    ids = [heading.group(1) for heading in headings]
    assert len(ids) == len(set(ids)), "duplicate decision id"
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        entry = text[heading.end() : end]
        fields = {match.group(1): match.group(2).strip() for match in FIELD.finditer(entry)}
        for field in REQUIRED_FIELDS:
            assert fields.get(field), f"{heading.group(1)} missing field: {field}"


def validate_ledger(text: str) -> None:
    """Validate a future record's structure and qualified Python test references."""
    validate_ledger_structure(text)
    root = RELEASE_ROOT.parents[1]
    reference = re.compile(r"(tests/[\w./-]+\.py)((?:::[\w]+)*)")
    for field in FIELD.finditer(text):
        if field.group(1) != "Enforced by":
            continue
        for match in reference.finditer(field.group(2)):
            source = (root / match.group(1)).resolve()
            assert source.is_relative_to(root) and source.is_file(), (
                f"unresolved enforcement test: {match.group()}"
            )
            node_path = match.group(2).split("::")[1:]
            if not node_path:
                continue
            nodes = ast.parse(source.read_text(encoding="utf-8")).body
            for name in node_path:
                node = next(
                    (
                        n
                        for n in nodes
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                        and n.name == name
                    ),
                    None,
                )
                assert node is not None, f"unresolved enforcement test: {match.group()}"
                nodes = node.body


@pytest.mark.parametrize(
    "ledger",
    [
        p
        for p in sorted(RELEASE_ROOT.glob("*/decisions.md"))
        if re.fullmatch(r"\d+\.\d+\.\d+", p.parent.name)
        and tuple(map(int, p.parent.name.split("."))) >= (0, 1, 0)
    ],
    ids=lambda p: p.parent.name,
)
def test_future_release_enforcement_references_resolve(ledger: Path) -> None:
    """Future ledgers bind enforcement references to current source, without running tests."""
    validate_ledger(ledger.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "ledger", sorted(RELEASE_ROOT.glob("*/decisions.md")), ids=lambda p: p.parent.name
)
def test_release_decision_entries_have_required_fields_and_unique_ids(ledger: Path) -> None:
    """Inspect the committed ledgers, including new versions discovered later."""
    validate_ledger_structure(ledger.read_text(encoding="utf-8"))


@pytest.mark.parametrize("version", ["0.0.8", "0.1.0"])
def test_duplicate_id_mutant_is_rejected(version: str) -> None:
    """Duplicating a real record ID must fail the same ledger validator."""
    text = (RELEASE_ROOT / version / "decisions.md").read_text(encoding="utf-8")
    headings = list(HEADING.finditer(text))
    second = headings[1]
    mutant = text[: second.start()] + text[second.start() :].replace(
        second.group(1), headings[0].group(1), 1
    )
    with pytest.raises(AssertionError, match="duplicate decision id"):
        validate_ledger_structure(mutant)


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
@pytest.mark.parametrize("version", ["0.0.8", "0.1.0"])
def test_missing_field_mutant_is_rejected(version: str, field: str) -> None:
    """Removing each required field from a real entry must be detected."""
    text = (RELEASE_ROOT / version / "decisions.md").read_text(encoding="utf-8")
    mutant, count = re.subn(
        rf"^- \*\*{re.escape(field)}:\*\*.*?(?=^- \*\*|^#|\Z)",
        "",
        text,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert count == 1, f"mutation did not remove {field}"
    with pytest.raises(AssertionError, match=rf"D-001 missing field: {re.escape(field)}"):
        validate_ledger_structure(mutant)


@pytest.mark.parametrize("version", ["0.0.8", "0.1.0"])
@pytest.mark.parametrize(
    ("heading", "reason"),
    [
        ("## D-081 — x", "invalid decision heading"),
        ("### D-081 - x", "invalid decision heading"),
        ("#### D-081: x", "invalid decision heading"),
        ("## D-080 — dup", "invalid decision heading"),
        ("### D-1000: x", "D-1000 missing field: Question"),
    ],
)
def test_heading_mutants_are_rejected(version: str, heading: str, reason: str) -> None:
    """A drifted heading cannot hide an incomplete or duplicated decision."""
    text = (RELEASE_ROOT / version / "decisions.md").read_text(encoding="utf-8")
    mutant = text + "\n" + heading + "\n- **Choice:** only\n"
    with pytest.raises(AssertionError, match=reason):
        validate_ledger_structure(mutant)


@pytest.mark.parametrize("version", ["0.0.8", "0.1.0"])
def test_complete_four_digit_decision_is_accepted(version: str) -> None:
    """Growing past D-999 retains the same required fields and heading form."""
    text = (RELEASE_ROOT / version / "decisions.md").read_text(encoding="utf-8")
    entry = "\n### D-1000: Future decision\n" + "\n".join(
        f"- **{field}:** Present" for field in REQUIRED_FIELDS
    )
    validate_ledger_structure(text + entry)


@pytest.mark.parametrize("version", ["0.0.8", "0.1.0"])
@pytest.mark.parametrize(
    ("replacement", "missing"),
    [
        ("- **Choice:**\n", "Choice"),
        ("- **Decided By:** Present\n", "Decided by"),
        ("  - **Decided by:** Present\n", "Decided by"),
    ],
)
def test_empty_or_malformed_field_mutants_are_rejected(
    version: str, replacement: str, missing: str
) -> None:
    """Empty values, changed labels and nested fields cannot satisfy a record."""
    text = (RELEASE_ROOT / version / "decisions.md").read_text(encoding="utf-8")
    mutant, count = re.subn(
        rf"^- \*\*{re.escape(missing)}:\*\*.*?(?=^- \*\*|^#|\Z)",
        replacement,
        text,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert count == 1, f"mutation did not change {missing}"
    with pytest.raises(AssertionError, match=rf"D-001 missing field: {missing}"):
        validate_ledger_structure(mutant)


def test_deleted_enforcement_test_cannot_pass_a_future_record() -> None:
    """A complete record must not claim enforcement by a deleted test."""
    entry = "\n### D-1000: Future enforcement\n" + "\n".join(
        f"- **{field}:** "
        + (
            "`tests/benchmark/test_campaign_horizon_authority.py::test_historical_runner_cap_matches_main_oracle`"
            if field == "Enforced by"
            else "Present"
        )
        for field in REQUIRED_FIELDS
    )
    with pytest.raises(AssertionError, match="unresolved enforcement test"):
        validate_ledger(entry)


def test_existing_paired_enforcement_reference_is_accepted() -> None:
    """The paired horizon test is a live enforcer, though not an absolute drift alarm."""
    text = "\n### D-1000: Live enforcement\n" + "\n".join(
        f"- **{field}:** "
        + (
            "`tests/benchmark/test_campaign_horizon_authority.py::test_historical_runner_cap_matches_paired_horizon_path`"
            if field == "Enforced by"
            else "Present"
        )
        for field in REQUIRED_FIELDS
    )
    validate_ledger(text)


@pytest.mark.parametrize(
    "reference",
    [
        "tests/benchmark/no_such_enforcement_file.py",
        "tests/benchmark/test_campaign_horizon_authority.py::MissingClass::test_missing",
    ],
)
def test_missing_enforcement_file_or_class_is_rejected(reference: str) -> None:
    """A qualified pointer must resolve each file and nested symbol."""
    text = "\n### D-1000: Missing enforcement\n" + "\n".join(
        f"- **{field}:** " + (f"`{reference}`" if field == "Enforced by" else "Present")
        for field in REQUIRED_FIELDS
    )
    with pytest.raises(AssertionError, match="unresolved enforcement test"):
        validate_ledger(text)
