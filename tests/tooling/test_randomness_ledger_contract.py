"""Focused contract tests for the non-normative randomness-ledger prototype sidecar.

These tests pin the PROTOTYPE JSON Schema at
``docs/contracts/randomness_ledger.prototype.schema.json`` and the checked-in fixture
under ``tests/data/randomness_ledger/``. The prototype is explicitly non-normative and
records seed-sensitivity provenance only; it does NOT support causal seed attribution
(see parent design-gap issue #5617). No runtime producer, recorder, replay, or random
stream refactoring is covered or implied here.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING

import jsonschema
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "docs" / "contracts" / "randomness_ledger.prototype.schema.json"
FIXTURE_DIR = REPO_ROOT / "tests" / "data" / "randomness_ledger"
FIXTURE_PATH = FIXTURE_DIR / "episode_seed_23.prototype.ledger.json"

PROTOTYPE_SCHEMA_VERSION = "randomness_ledger.prototype.v1"
PROTOTYPE_SCHEMA_ID = "https://robot-sf.dev/contracts/randomness_ledger.prototype.v1.json"
PROTOTYPE_DRAFT = "http://json-schema.org/draft-07/schema#"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def schema() -> dict:
    """Load the prototype schema once for the whole module."""

    assert SCHEMA_PATH.is_file(), f"missing prototype schema: {SCHEMA_PATH}"
    return _load_json(SCHEMA_PATH)


@pytest.fixture(scope="module")
def valid_ledger() -> dict:
    """Load the checked-in valid prototype fixture once for the whole module."""

    assert FIXTURE_PATH.is_file(), f"missing prototype fixture: {FIXTURE_PATH}"
    return _load_json(FIXTURE_PATH)


def test_schema_declares_stable_prototype_identifier(schema: dict) -> None:
    """The prototype must declare a stable, versioned draft-07 identifier."""

    assert schema["$schema"] == PROTOTYPE_DRAFT
    assert schema["$id"] == PROTOTYPE_SCHEMA_ID
    schema_version = schema["properties"]["schema_version"]
    assert schema_version.get("const") == PROTOTYPE_SCHEMA_VERSION


def test_checked_in_fixture_is_valid(schema: dict, valid_ledger: dict) -> None:
    """The checked-in fixture must validate against the prototype schema and self-identify."""

    assert valid_ledger["schema_version"] == PROTOTYPE_SCHEMA_VERSION
    # Raises jsonschema.ValidationError on any contract violation; that is the proof.
    jsonschema.validate(instance=valid_ledger, schema=schema)


def test_fixture_directory_is_populated() -> None:
    """The fixture directory must contain at least the canonical prototype fixture."""

    assert FIXTURE_DIR.is_dir(), f"missing fixture directory: {FIXTURE_DIR}"
    entries = sorted(p.name for p in FIXTURE_DIR.iterdir())
    assert FIXTURE_PATH.name in entries
    assert len(entries) >= 1


@pytest.mark.parametrize(
    ("malform", "expected_field"),
    [
        pytest.param(
            lambda doc: doc["factors"][0].pop("factor_name"),
            "factor_name",
            id="missing_factor_name",
        ),
        pytest.param(
            lambda doc: doc["factors"][0].pop("stream_identity"),
            "stream_identity",
            id="missing_stream_identity",
        ),
        pytest.param(
            lambda doc: doc["factors"][0].__setitem__("draw_count", -1),
            "draw_count",
            id="negative_draw_count",
        ),
    ],
)
def test_required_malformed_ledgers_are_rejected(
    schema: dict,
    valid_ledger: dict,
    malform: Callable[[dict], object],
    expected_field: str,
) -> None:
    """Each required malformed case must fail validation for its intended contract reason.

    Cases (per issue #6466): missing factor name, missing stream identity, and a
    negative draw count. Each is built from the valid fixture by a single mutation so
    that a failure proves the targeted contract, not an unrelated coincidence.
    """

    invalid = copy.deepcopy(valid_ledger)
    malform(invalid)

    with pytest.raises(jsonschema.ValidationError) as exc_info:
        jsonschema.validate(instance=invalid, schema=schema)

    detail = f"{exc_info.value.json_path}: {exc_info.value.message}"
    assert expected_field in detail, (
        f"expected the '{expected_field}' contract to be named in the error, got: {detail!r}"
    )
