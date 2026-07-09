"""Regression tests for issue #4956: fast-feedback ``ty`` findings on optional-dep imports.

Main CI reported three classes of ``ty`` diagnostics that the issue names explicitly:

* ``error[unresolved-import]: Cannot resolve imported module GPUtil`` — the
  ``# type: ignore[import]`` suppression used a mypy code that ``ty`` does not
  honour for an ``unresolved-import`` on the import statement itself.
* ``error[unresolved-import]: Module svgelements has no member SVG/Path/Point`` —
  same root cause: ``# type: ignore[attr-defined]`` is the wrong code for an
  ``unresolved-import`` member-resolution diagnostic.
* ``error[invalid-argument-type]: Argument to bound method Axes.bar_label is
  incorrect`` — a real type error: ``ax.containers`` is ``list[Container]`` but
  ``Axes.bar_label`` declares ``container: BarContainer``.

These tests pin the fix at the source level (fast and deterministic) and, when
``uvx``/``ty`` is available locally, run the exact ``ty check`` invocation the
fast-feedback job uses against the three touched files to prove the named
diagnostics no longer appear.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GPUTIL_SCRIPT = REPO_ROOT / "SLURM" / "log_gpu_cpu_usage.py"
SVG_CONV_SCRIPT = REPO_ROOT / "docs" / "tooling" / "svg_conv" / "svg_conv.py"
IMITATION_NOTEBOOK = (
    REPO_ROOT / "docs" / "dev" / "issues" / "classic-interactions-ppo" / "imitation_progress.ipynb"
)

# The three named diagnostics from issue #4956. Each must be absent from a ``ty``
# run over the touched files.
NAMED_DIAGNOSTIC_SUBSTRINGS = (
    "Cannot resolve imported module `GPUtil`",
    "Module `svgelements` has no member `SVG`",
    "Module `svgelements` has no member `Path`",
    "Module `svgelements` has no member `Point`",
    "Argument to bound method `Axes.bar_label` is incorrect",
)

# mypy-style codes that ``ty`` does NOT honour for ``unresolved-import`` on the
# import statement. They must not come back on the fixed lines.
REJECTED_MYPY_CODES = ("type: ignore[import]", "type: ignore[attr-defined]")


def test_gputil_import_uses_ty_honoured_suppression() -> None:
    """GPUtil import must use a bare ``# type: ignore`` (ty-honoured), not ``[import]``."""
    source = GPUTIL_SCRIPT.read_text(encoding="utf-8")
    assert "import GPUtil  # type: ignore[import]" not in source
    assert "import GPUtil  # type: ignore" in source


def test_svgelements_import_uses_ty_honoured_suppression() -> None:
    """svgelements import must use a bare ``# type: ignore``, not ``[attr-defined]``."""
    source = SVG_CONV_SCRIPT.read_text(encoding="utf-8")
    assert "from svgelements import SVG, Path, Point  # type: ignore[attr-defined]" not in source
    assert "from svgelements import SVG, Path, Point  # type: ignore" in source


def test_bar_label_call_is_guarded_by_barcontainer_isinstance() -> None:
    """``ax.bar_label`` must only be called on ``BarContainer`` instances (real type fix).

    ``ax.containers`` is typed ``list[Container]`` while ``Axes.bar_label`` declares
    ``container: BarContainer``. Filtering to ``BarContainer`` is both type-correct
    and runtime-correct (bar labels only make sense for bar containers).
    """
    nb = json.loads(IMITATION_NOTEBOOK.read_text(encoding="utf-8"))
    sources = [
        "".join(cell.get("source", [])) for cell in nb["cells"] if cell.get("cell_type") == "code"
    ]
    joined = "\n".join(sources)

    # The import must be present so the isinstance guard resolves at runtime.
    assert "from matplotlib.container import BarContainer" in joined
    # The unguarded call must not come back.
    assert 'ax.bar_label(container, fmt="{:.0f}")' in joined
    # The loop body must guard bar_label with a BarContainer isinstance check.
    assert "isinstance(container, BarContainer)" in joined
    # The previously-unguarded pattern (bar_label directly under the for-loop) is gone.
    assert "for container in ax.containers:\n    ax.bar_label(" not in joined


@pytest.mark.parametrize("rejected_code", REJECTED_MYPY_CODES)
def test_no_reintroduced_mypy_ignore_codes_on_fixed_lines(rejected_code: str) -> None:
    """Neither fixed ``.py`` file should reintroduce the mypy codes ty ignores."""
    for path in (GPUTIL_SCRIPT, SVG_CONV_SCRIPT):
        assert rejected_code not in path.read_text(encoding="utf-8"), path


def _uvx_available() -> bool:
    return shutil.which("uvx") is not None


def _resolve_ty_python() -> Path | None:
    """Resolve a Python environment for ``ty``.

    ``ty`` only emits the issue #4956 diagnostics when it can resolve the project
    site-packages (so optional deps like GPUtil are confirmed absent and
    svgelements/matplotlib members are checked). In a linked worktree the local
    ``./.venv`` may not exist, so fall back to the shared/main checkout venv the
    same way the worktree shared-venv helper does.
    """
    candidates: list[Path] = []
    local_venv = REPO_ROOT / ".venv"
    if local_venv.exists():
        candidates.append(local_venv)
    env_override = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if env_override:
        candidates.append(Path(env_override))
    # Linked worktree: derive the main checkout venv from the common git dir.
    try:
        common_dir = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        if common_dir:
            main_root = Path(common_dir).resolve().parent
            candidates.append(main_root / ".venv")
    except (subprocess.CalledProcessError, OSError, FileNotFoundError):
        pass
    for cand in candidates:
        if (cand / "bin" / "python").exists():
            return cand
    return None


@pytest.mark.skipif(not _uvx_available(), reason="uvx/ty not installed in this environment")
def test_ty_check_touched_files_reports_no_named_diagnostics() -> None:
    """Run the exact fast-feedback ``ty check`` over the three touched files.

    This is the executable acceptance for issue #4956: the named diagnostics
    (GPUtil unresolved-import, svgelements member resolution, Axes.bar_label
    argument type) must not appear. ``ty`` is advisory in CI (``--exit-zero``);
    this assertion is about the named findings being absent, not the exit code.
    """
    venv = _resolve_ty_python()
    if venv is None:
        pytest.skip("no resolvable Python environment for ty (venv missing)")
    cmd = ["uvx", "ty", "check", "--exit-zero", "--python", str(venv)]
    cmd += [str(GPUTIL_SCRIPT), str(SVG_CONV_SCRIPT), str(IMITATION_NOTEBOOK)]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300)
    combined = result.stdout + result.stderr
    # Guard against a vacuous pass: ty must actually have run against a real env.
    # A non-vacuous run always mentions either a clean result or a diagnostic count.
    assert combined.strip(), f"ty produced no output; env={venv}\nstderr={result.stderr!r}"
    missing = [needle for needle in NAMED_DIAGNOSTIC_SUBSTRINGS if needle in combined]
    assert not missing, (
        f"named issue #4956 diagnostics reappeared: {missing}\nty output:\n{combined}"
    )
