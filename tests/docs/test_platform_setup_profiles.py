"""Contract tests for the platform setup profiles (issue #8719)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "docs" / "quickstart_platforms.md"

PROFILE_CONTRACTS = {
    "Linux": {
        "setup": (
            "scripts/dev/check_runtime_requirements.sh",
            "uv sync --all-extras",
        ),
        "verify": "uv run robot-sf doctor --skip-env-smoke --skip-quickstart-smoke",
        "smoke": "uv run robot-sf examples run quickstart/01_basic_robot --fast",
        "paths": (
            "pyproject.toml",
            "third_party/python-rvo2",
            "scripts/dev/check_runtime_requirements.sh",
            "examples/quickstart/01_basic_robot.py",
        ),
    },
    "macOS": {
        "setup": (
            "scripts/dev/check_runtime_requirements.sh",
            "uv sync --all-extras",
        ),
        "verify": "uv run robot-sf doctor --skip-env-smoke --skip-quickstart-smoke",
        "smoke": "uv run robot-sf examples run quickstart/01_basic_robot --fast",
        "paths": (
            "pyproject.toml",
            "third_party/python-rvo2",
            "scripts/dev/check_runtime_requirements.sh",
            "examples/quickstart/01_basic_robot.py",
        ),
    },
    "Headless (servers and CI)": {
        "setup": (
            "scripts/dev/check_runtime_requirements.sh",
            "uv sync --all-extras",
        ),
        "verify": "uv run robot-sf doctor --skip-env-smoke --skip-quickstart-smoke",
        "smoke": "uv run robot-sf examples run quickstart/01_basic_robot --fast",
        "paths": (
            "pyproject.toml",
            "third_party/python-rvo2",
            "scripts/dev/check_runtime_requirements.sh",
            "examples/quickstart/01_basic_robot.py",
            "docs/dev_runtime_requirements.md",
        ),
    },
}


def _text() -> str:
    return PAGE.read_text(encoding="utf-8")


def _profile_section(text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\n(?P<section>.*?)(?=^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, heading
    return match.group("section")


def test_platform_page_exists_with_three_profiles() -> None:
    text = _text()
    for heading in PROFILE_CONTRACTS:
        assert f"## {heading}" in text


@pytest.mark.parametrize("heading", PROFILE_CONTRACTS)
def test_each_profile_has_setup_verification_and_smoke_contract(heading: str) -> None:
    section = _profile_section(_text(), heading)
    contract = PROFILE_CONTRACTS[heading]
    for command in contract["setup"]:
        assert command in section, heading
    assert contract["verify"] in section, heading
    assert contract["smoke"] in section, heading


@pytest.mark.parametrize("heading", PROFILE_CONTRACTS)
def test_each_profile_references_existing_repository_paths(heading: str) -> None:
    contract = PROFILE_CONTRACTS[heading]
    for relative_path in contract["paths"]:
        path = REPO_ROOT / relative_path
        assert path.is_dir() if relative_path == "third_party/python-rvo2" else path.is_file(), (
            relative_path
        )
        assert relative_path in _text(), relative_path
    runtime_check = REPO_ROOT / "scripts" / "dev" / "check_runtime_requirements.sh"
    assert os.access(runtime_check, os.X_OK)


def test_headless_caveats_match_runtime_requirements() -> None:
    text = _profile_section(_text(), "Headless (servers and CI)")
    for variable in ("DISPLAY=", "MPLBACKEND=Agg", "SDL_VIDEODRIVER=dummy"):
        assert variable in text


def test_windows_presented_as_unverified_only() -> None:
    text = _text()
    assert re.search(r"[Ww]indows is \*\*unverified\*\*", text) is not None
    assert "Windows is supported" not in text
    assert "Windows:" not in text.replace("Windows is **unverified**", "")


def test_linked_from_onboarding_pages() -> None:
    for page in ("README.md", "docs/adoption_path.md", "docs/dev_runtime_requirements.md"):
        assert "quickstart_platforms" in (REPO_ROOT / page).read_text(encoding="utf-8"), page
