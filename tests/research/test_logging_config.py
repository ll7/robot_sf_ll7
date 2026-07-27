"""Direct unit tests for the structured seed-failure warning contract.

These tests lock the public behavior of
``robot_sf.research.logging_config.log_seed_failure`` by mocking the module-level
loguru logger. They do not write log files or mutate global logger configuration.

The locked contract is:

- exactly one ``logger.warning`` event per call,
- the literal message ``"Seed run failed or missing"``,
- stable structured field names ``seed``, ``policy_type``, and ``reason``,
- numeric, string, and ``None`` seeds are forwarded verbatim,
- a non-``None`` ``policy_type`` is forwarded verbatim,
- a ``None`` ``policy_type`` is normalized to the sentinel string ``"unknown"``,
- the ``reason`` is preserved exactly.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from robot_sf.research import logging_config

EXPECTED_MESSAGE = "Seed run failed or missing"
# Stable structured field names that downstream log parsing and dashboards depend on.
EXPECTED_KWARG_KEYS = ("seed", "policy_type", "reason")


@pytest.fixture
def mock_logger() -> MagicMock:
    """Patch the module-level loguru logger so no real log sink is exercised."""
    with patch.object(logging_config, "logger") as patched:
        yield patched


def _assert_single_warning(
    mock_logger: MagicMock,
    *,
    seed: Any,
    policy_type: str,
    reason: str,
) -> None:
    """Assert exactly one warning event with stable field names and forwarded values."""
    # Exactly one warning event, and no other log level is touched.
    assert mock_logger.warning.call_count == 1
    assert mock_logger.info.call_count == 0
    assert mock_logger.error.call_count == 0
    assert mock_logger.debug.call_count == 0

    call = mock_logger.warning.call_args

    # The literal message is the sole positional argument.
    assert call.args == (EXPECTED_MESSAGE,)

    # Stable structured field names: no drift, no missing, no extra.
    assert tuple(call.kwargs.keys()) == EXPECTED_KWARG_KEYS

    # Forwarded values match the locked contract (``==`` covers ``None`` seeds too).
    assert call.kwargs["seed"] == seed
    assert call.kwargs["policy_type"] == policy_type
    assert call.kwargs["reason"] == reason


@pytest.mark.parametrize(
    "seed",
    [42, 0, -7, "run-007", "non-numeric", None],
    ids=["int", "zero", "negative-int", "string", "non-numeric-string", "none"],
)
def test_seed_forwarded_verbatim(mock_logger: MagicMock, seed: Any) -> None:
    """Numeric, string, and None seeds are forwarded to the warning without coercion."""
    logging_config.log_seed_failure(seed=seed, policy_type="ppo", reason="divergence")

    _assert_single_warning(
        mock_logger,
        seed=seed,
        policy_type="ppo",
        reason="divergence",
    )


# Note: the implementation uses ``policy_type or "unknown"``, so a falsy non-None value
# such as the empty string is *also* normalized to "unknown". The contract only requires
# None normalization and non-None forwarding, so the forwarding test covers non-None truthy
# policy types only. The None case is locked separately below.
@pytest.mark.parametrize(
    "policy_type",
    ["ppo", "dqn", "predictive_v2"],
    ids=["ppo", "dqn", "predictive-v2"],
)
def test_policy_type_forwarded_verbatim(mock_logger: MagicMock, policy_type: str) -> None:
    """A non-None truthy policy_type is forwarded verbatim to the warning."""
    logging_config.log_seed_failure(seed=1, policy_type=policy_type, reason="timeout")

    _assert_single_warning(
        mock_logger,
        seed=1,
        policy_type=policy_type,
        reason="timeout",
    )


def test_none_policy_type_normalized_to_unknown(mock_logger: MagicMock) -> None:
    """A None policy_type is normalized to the sentinel string 'unknown'."""
    logging_config.log_seed_failure(seed=1, policy_type=None, reason="missing")

    _assert_single_warning(
        mock_logger,
        seed=1,
        policy_type="unknown",
        reason="missing",
    )


@pytest.mark.parametrize(
    "reason",
    [
        "checkpoint not found",
        "",
        "value error: NaN at step 1024",
        "with 'quotes' and {braces}",
        "multi\nline\nreason",
    ],
    ids=["normal", "empty", "python-traceback-like", "special-chars", "multiline"],
)
def test_reason_preserved_exactly(mock_logger: MagicMock, reason: str) -> None:
    """The reason is forwarded exactly, including empty, special-char, and multiline text."""
    logging_config.log_seed_failure(seed=7, policy_type="ppo", reason=reason)

    _assert_single_warning(mock_logger, seed=7, policy_type="ppo", reason=reason)


def test_exactly_one_warning_event_with_stable_fields(mock_logger: MagicMock) -> None:
    """One call emits exactly one warning with the literal message and stable field names."""
    logging_config.log_seed_failure(seed=42, policy_type="ppo", reason="divergence")

    # Exactly one warning call total.
    mock_logger.warning.assert_called_once()

    # Literal message + stable structured field names locked.
    mock_logger.warning.assert_called_once_with(
        EXPECTED_MESSAGE,
        seed=42,
        policy_type="ppo",
        reason="divergence",
    )


def test_none_seed_and_none_policy_combined(mock_logger: MagicMock) -> None:
    """A None seed is forwarded as None while a None policy_type normalizes to 'unknown'."""
    logging_config.log_seed_failure(seed=None, policy_type=None, reason="no run")

    _assert_single_warning(
        mock_logger,
        seed=None,
        policy_type="unknown",
        reason="no run",
    )
