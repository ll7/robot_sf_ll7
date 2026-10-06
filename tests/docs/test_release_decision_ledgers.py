"""Release decision records must remain complete and unambiguous.

This checks record structure, not whether any recorded choice is implemented.
"""

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


def validate_ledger(text: str) -> None:
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


@pytest.mark.parametrize(
    "ledger", sorted(RELEASE_ROOT.glob("*/decisions.md")), ids=lambda p: p.parent.name
)
def test_release_decision_entries_have_required_fields_and_unique_ids(ledger: Path) -> None:
    """Inspect the committed ledgers, including new versions discovered later."""
    validate_ledger(ledger.read_text(encoding="utf-8"))


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
        validate_ledger(mutant)


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
        validate_ledger(mutant)


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
        validate_ledger(mutant)


@pytest.mark.parametrize("version", ["0.0.8", "0.1.0"])
def test_complete_four_digit_decision_is_accepted(version: str) -> None:
    """Growing past D-999 retains the same required fields and heading form."""
    text = (RELEASE_ROOT / version / "decisions.md").read_text(encoding="utf-8")
    entry = "\n### D-1000: Future decision\n" + "\n".join(
        f"- **{field}:** Present" for field in REQUIRED_FIELDS
    )
    validate_ledger(text + entry)


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
        validate_ledger(mutant)
