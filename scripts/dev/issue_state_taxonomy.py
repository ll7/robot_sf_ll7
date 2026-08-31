#!/usr/bin/env python3
"""Shared issue lifecycle state taxonomy for audit and admission helpers."""

from __future__ import annotations

STATE_PREFIX = "state:"

# These labels are mutually exclusive execution states. An issue may carry exactly one
# execution state while also carrying one or more known state qualifiers below.
EXECUTION_STATE_LABELS = frozenset(
    {
        "state:blocked-external-input",
        "state:blocked",
        "state:hold",
        "state:running",
        "state:ready",
    }
)
STATE_PRIORITY = (
    "state:blocked-external-input",
    "state:blocked",
    "state:hold",
    "state:running",
    "state:ready",
)

# Known composable state labels that qualify routing or downstream handling without
# replacing the execution state.
STATE_QUALIFIER_LABELS = frozenset(
    {
        "state:author-decision",
        "state:blocked-human-decision",
        "state:blocked-no-code-slice",
        "state:deferred",
        "state:needs-artifact-promotion",
        "state:parked",
        "state:review",
        "state:working",
    }
)
KNOWN_STATE_LABELS = EXECUTION_STATE_LABELS | STATE_QUALIFIER_LABELS


def state_labels(labels: set[str]) -> list[str]:
    """Return all known and unknown ``state:*`` labels in deterministic order."""
    return sorted(label for label in labels if label.startswith(STATE_PREFIX))


def execution_state_labels(labels: set[str]) -> list[str]:
    """Return mutually exclusive execution-state labels in deterministic order."""
    return sorted(labels & EXECUTION_STATE_LABELS)


def state_qualifier_labels(labels: set[str]) -> list[str]:
    """Return known composable ``state:*`` qualifier labels in deterministic order."""
    return sorted(labels & STATE_QUALIFIER_LABELS)


def unknown_state_labels(labels: set[str]) -> list[str]:
    """Return ``state:*`` labels that have not been classified into the shared taxonomy."""
    return sorted(
        label
        for label in labels
        if label.startswith(STATE_PREFIX) and label not in KNOWN_STATE_LABELS
    )
