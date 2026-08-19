"""Keep the powered #5303 checker as the only current promotion authority."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
V1_INVOCATIONS = (
    "scripts/tools/check_issue_5303_search_promotion_contract.py",
    "scripts/tools/check_issue_5303_search_promotion_preregistration.py",
)
HISTORICAL_PATHS = (
    "configs/adversarial/issue_5303_search_promotion_contract.yaml",
    "docs/context/evidence/issue_5303_search_promotion_preregistration/",
    "robot_sf/benchmark/issue_5303_search_promotion_preregistration.py",
    "scripts/tools/check_issue_5303_search_promotion_contract.py",
    "scripts/tools/check_issue_5303_search_promotion_preregistration.py",
    "tests/adversarial/test_issue_5303_search_promotion_preregistration.py",
    "tests/adversarial/test_issue_5303_checker_authority.py",
)


def _tracked_paths() -> list[Path]:
    """Return tracked text files without following generated or untracked trees."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    paths: list[Path] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        path = Path(raw.decode("utf-8"))
        if path.suffix.lower() in {".json", ".md", ".py", ".yaml", ".yml"}:
            paths.append(path)
    return paths


def _is_historical(path: Path) -> bool:
    """Return whether a path is an immutable historical contract surface."""
    value = path.as_posix()
    return any(value == prefix or value.startswith(prefix) for prefix in HISTORICAL_PATHS)


def test_only_v2_checker_is_referenced_by_current_surfaces() -> None:
    """Historical v1 commands must not leak into current operational documentation/code."""
    offenders: list[str] = []
    for relative_path in _tracked_paths():
        if _is_historical(relative_path):
            continue
        path = REPO_ROOT / relative_path
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for invocation in V1_INVOCATIONS:
            if invocation in text:
                offenders.append(f"{relative_path}: {invocation}")

    assert not offenders, "current #5303 surfaces must use the powered v2 checker: " + "; ".join(
        offenders
    )


def test_powered_checker_is_the_documented_current_entrypoint() -> None:
    """The current guide names the v2 check CLI and its side-effect-free identity mode."""
    guide = (REPO_ROOT / "docs/dev_guide.md").read_text(encoding="utf-8")
    assert "scripts/tools/check_issue_5303_search_promotion_contract_v2.py" in guide
    assert "--identities" in guide
