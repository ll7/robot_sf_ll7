"""Reusable assertions for shared-helper migration compatibility tests.

Use ``SharedHelperContract`` once for every migrated call site.  It keeps the
otherwise easy-to-miss behavior checks (return shape, missing input, malformed
input, import footprint, reading strategy, and ordering) together with the call
site that needs them. Callers whose input format has no meaningful malformed
form should say so in ``not_applicable`` rather than silently omitting the check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class ErrorExpectation:
    """Expected exception type and required message fragments for one input class."""

    exception_type: type[BaseException]
    message_fragments: tuple[str, ...] = ()


@dataclass(frozen=True)
class SharedHelperContract:
    """Compatibility obligations for one real shared-helper migration call site."""

    call_site: str
    helper_name: str
    valid_call: Callable[[], object]
    expected_return_type: type | tuple[type, ...]
    validate_result: Callable[[object], None]
    missing_call: Callable[[], object] | None = None
    missing_error: ErrorExpectation | None = None
    malformed_call: Callable[[], object] | None = None
    malformed_error: ErrorExpectation | None = None
    validate_import_footprint: Callable[[], None] | None = None
    validate_read_strategy: Callable[[], None] | None = None
    validate_output_ordering: Callable[[], None] | None = None
    not_applicable: tuple[str, ...] = ()


def _assert_error(call: Callable[[], object], expectation: ErrorExpectation) -> None:
    """Assert an error contract while keeping type and message failures local."""

    with pytest.raises(expectation.exception_type) as excinfo:
        call()
    message = str(excinfo.value)
    for fragment in expectation.message_fragments:
        assert fragment in message


def assert_shared_helper_contract(contract: SharedHelperContract) -> None:
    """Assert the declared compatibility contract for a migrated helper call site.

    The declaration is deliberately explicit: a migration test must either
    exercise each applicable behavior or name why the behavior is not an input
    to that helper.  This makes mechanical replacement reviewable as a contract
    change instead of a textual substitution.
    """

    result = contract.valid_call()
    assert isinstance(result, contract.expected_return_type), (
        f"{contract.call_site} changed {contract.helper_name} return type: "
        f"expected {contract.expected_return_type}, got {type(result)}"
    )
    contract.validate_result(result)

    _assert_optional_error(contract.missing_call, contract.missing_error, "missing", contract)
    _assert_optional_error(
        contract.malformed_call,
        contract.malformed_error,
        "malformed",
        contract,
    )

    _assert_optional_validation(contract.validate_import_footprint, "import", contract)
    _assert_optional_validation(contract.validate_read_strategy, "read_strategy", contract)
    _assert_optional_validation(contract.validate_output_ordering, "ordering", contract)


def _assert_optional_error(
    call: Callable[[], object] | None,
    expectation: ErrorExpectation | None,
    category: str,
    contract: SharedHelperContract,
) -> None:
    """Require a complete error declaration or an explicit inapplicability note."""

    if call is not None or expectation is not None:
        assert call is not None and expectation is not None, (
            f"{contract.call_site} must provide both {category} call and expectation"
        )
        _assert_error(call, expectation)
        return
    assert any(note.startswith(f"{category}:") for note in contract.not_applicable), (
        f"{contract.call_site} must test {category} input or document why it is inapplicable"
    )


def _assert_optional_validation(
    validation: Callable[[], None] | None,
    category: str,
    contract: SharedHelperContract,
) -> None:
    """Require a behavior check or an explicit reason it cannot apply."""

    if validation is not None:
        validation()
        return
    assert any(note.startswith(f"{category}:") for note in contract.not_applicable), (
        f"{contract.call_site} must test {category} behavior or document why it is inapplicable"
    )
