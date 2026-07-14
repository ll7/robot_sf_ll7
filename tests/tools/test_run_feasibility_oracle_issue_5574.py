"""Tests for the issue #5574 candidate-cell oracle CLI."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from scripts.tools import run_feasibility_oracle_issue_5574 as cli

if TYPE_CHECKING:
    from pathlib import Path


def test_resolve_envelope_radii_uses_repository_defaults() -> None:
    """Omitting probes preserves the canonical nominal/reduced pair."""
    assert cli._resolve_envelope_radii(None, None, (1.0, 0.5)) == (1.0, 0.5)


def test_resolve_envelope_radii_requires_probe_for_custom_nominal() -> None:
    """A custom nominal radius cannot silently reuse the default reduced probe."""
    with pytest.raises(ValueError, match="reduced-envelope-radius"):
        cli._resolve_envelope_radii(0.75, None, (1.0, 0.5))


def test_main_writes_report_without_importing_a_campaign_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI serializes the builder result to the requested path."""
    output = tmp_path / "report.json"
    expected = {
        "schema_version": "issue_5574_feasibility_oracle_report.v1",
        "cells": [],
    }
    monkeypatch.setattr(
        "robot_sf.scenario_certification.feasibility_oracle.build_issue_5574_feasibility_report",
        lambda *_args, **_kwargs: expected,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_feasibility_oracle_issue_5574.py", "--output", str(output)],
    )

    assert cli.main() == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected


def test_parser_accepts_repeated_reduced_probes() -> None:
    """The CLI exposes more than one reduced sensitivity probe when requested."""
    args = cli._build_parser().parse_args(
        [
            "--nominal-envelope-radius",
            "1.0",
            "--reduced-envelope-radius",
            "0.5",
            "--reduced-envelope-radius",
            "0.25",
        ]
    )

    assert args.nominal_envelope_radius == 1.0
    assert args.reduced_envelope_radii == [0.5, 0.25]
