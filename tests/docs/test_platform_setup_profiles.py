"""Contract tests for the platform setup profiles (issue #8719)."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "docs" / "quickstart_platforms.md"


def _text() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_platform_page_exists_with_three_profiles() -> None:
    text = _text()
    assert "## Linux" in text
    assert "## macOS" in text
    assert "## Headless" in text


def test_referenced_commands_and_paths_exist() -> None:
    text = _text()
    assert (REPO_ROOT / "scripts" / "dev" / "check_runtime_requirements.sh").is_file()
    assert (REPO_ROOT / "docs" / "adoption_path.md").is_file()
    assert (REPO_ROOT / "docs" / "dev_runtime_requirements.md").is_file()
    assert "uv sync --all-extras" in text
    assert "uv run robot-sf doctor --skip-env-smoke --skip-quickstart-smoke" in text


def test_headless_caveats_match_runtime_requirements() -> None:
    text = _text()
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
