"""Regression tests for hatch-vcs / setuptools-scm package-version derivation.

Guards issue #6328. After the repository published the non-release tag
``artifact/legacy-models-2026-07-registry-v1``, a checkout whose nearest
reachable tag was that artifact tag could no longer build editable: hatch-vcs
delegates to setuptools-scm / vcs_versioning, whose default ``git describe``
match glob ``*[0-9]*`` matched the artifact tag (it contains digits), and the
``tag_regex`` could not parse a PEP 440 version from it, so ``uv sync`` failed
with::

    ValueError: Error getting the version from source `vcs`:
    Can't parse version from tag 'artifact/legacy-models-2026-07-registry-v1'

The fix lives entirely in ``pyproject.toml`` ``[tool.hatch.version.raw-options]``:

* ``git_describe_command`` restricts ``git describe`` to dotted-numeric tags via
  ``--match *[0-9]*.*[0-9]*`` (vcs_versioning's ``tag.strict = true``
  equivalent), so an artifact tag without a dotted version is never selected;
* ``tag_regex`` maps release (``X.Y.Z``), ``v``-prefixed (``vX.Y.Z``) and
  release-candidate (``rcX.Y.Z``) tags to the numeric ``X.Y.Z``;
* ``fallback_version`` is reached only when *no* tag matches the glob, so a
  checkout whose sole reachable tag is an artifact tag still derives a valid
  PEP 440 non-release version and the editable build never breaks.

These tests read the *live* config from ``pyproject.toml`` (so an accidental
reversion fails here) and drive the *real* configured ``git describe`` command
plus the *real* ``tag_regex`` against tiny throwaway git repositories. The
end-to-end editable build (``uv sync`` + ``import robot_sf``) is exercised
separately as issue #6328 acceptance evidence; this module guards the
configuration surface that can regress.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

# The exact non-release artifact tag that broke CI (issue #6328). It contains
# digits but no dotted ``X.Y.Z`` version, which is the whole point of the fix.
ARTIFACT_TAG = "artifact/legacy-models-2026-07-registry-v1"

# A non-release / non-release-line sentinel: the historical model-registry tag.
# Also digit-bearing but version-free, so it must be rejected by the strict glob.
OTHER_ARTIFACT_TAG = "artifact/models-2026-05-registry-v1"


def _raw_options() -> Mapping[str, str]:
    """Return the live ``[tool.hatch.version.raw-options]`` table."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return data["tool"]["hatch"]["version"]["raw-options"]


def _git_env() -> Mapping[str, str]:
    """Return an env with a deterministic git identity for throwaway repos."""
    env = dict(os.environ)
    env.update(
        {
            "GIT_AUTHOR_NAME": "robot-sf-tests",
            "GIT_AUTHOR_EMAIL": "robot-sf-tests@example.com",
            "GIT_COMMITTER_NAME": "robot-sf-tests",
            "GIT_COMMITTER_EMAIL": "robot-sf-tests@example.com",
        }
    )
    return env


def _init_repo(root: Path) -> None:
    """Create a fresh git repo at *root* with one committed file."""
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    (root / "file.txt").write_text("initial\n", encoding="utf-8")
    subprocess.run(["git", "add", "file.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=root, check=True, env=_git_env())


def _add_commit(root: Path, content: str) -> None:
    """Append a commit so older tags stay reachable but no longer sit on HEAD."""
    (root / "file.txt").write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "file.txt"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", f"change-{content}"],
        cwd=root,
        check=True,
        env=_git_env(),
    )


def _tag_head(root: Path, tag: str) -> None:
    """Tag the current HEAD of *root* with *tag* (lightweight)."""
    subprocess.run(["git", "tag", tag], cwd=root, check=True)


def _run_configured_describe(root: Path) -> subprocess.CompletedProcess[str]:
    """Run the live configured ``git_describe_command`` in *root*.

    This is exactly what hatch-vcs / vcs_versioning executes (the string is
    ``shlex.split`` and run as a list, mirroring
    ``vcs_versioning._backends._git.version_from_describe``).
    """
    cmd = shlex.split(_raw_options()["git_describe_command"])
    return subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def _describe_match_glob() -> str:
    """Return the ``--match`` glob from the configured describe command."""
    cmd = shlex.split(_raw_options()["git_describe_command"])
    match_idx = cmd.index("--match")
    return cmd[match_idx + 1]


def _compiled_tag_regex() -> re.Pattern[str]:
    """Return the live ``tag_regex`` compiled, exactly as vcs_versioning does."""
    return re.compile(_raw_options()["tag_regex"])


def _extract_version(tag: str) -> str | None:
    """Mirror vcs_versioning ``tag_to_version``: pull the ``version`` group."""
    match = _compiled_tag_regex().match(tag)
    if match is None:
        return None
    groups = match.groupdict()
    key = "version" if "version" in groups else 1
    return match.group(key)


def _git_parse_describe(output: str) -> tuple[str, int]:
    """Mirror vcs_versioning ``_git_parse_describe``: return (tag, distance)."""
    if output.endswith("-dirty"):
        output = output[: -len("-dirty")]
    split = output.rsplit("-", 2)
    if len(split) < 3:  # bare tag (e.g. .git_archival.txt)
        return output, 0
    tag, number, _node = split
    return tag, int(number)


def _derive_version(repo: Path) -> str:
    """Mirror vcs_versioning's documented derivation using the live config.

    * Run the configured ``git_describe_command``.
    * If it fails (no tag matches the strict glob), vcs_versioning falls back to
      ``fallback_version`` (see ``_backends._git._git_parse_inner``); we return
      it verbatim, which is sufficient to assert the build survives with a
      valid non-release version.
    * Otherwise split the describe output (``_git_parse_describe``) and extract
      the ``version`` group (``tag_to_version``). The release / rc test cases
      place the tag directly on HEAD (distance 0), so the derived version equals
      the extracted version without needing the dev-version guessing scheme.
    """
    raw = _raw_options()
    result = _run_configured_describe(repo)
    if result.returncode != 0:
        return raw["fallback_version"]
    tag, distance = _git_parse_describe(result.stdout.strip())
    extracted = _extract_version(tag)
    assert extracted is not None, f"configured tag_regex could not parse selected tag {tag!r}"
    assert distance == 0, f"unexpected distance {distance} from describe output {result.stdout!r}"
    return extracted


def _is_non_release(version: str) -> bool:
    """Return True for a PEP 440 version that is not a final release."""
    return bool(re.search(r"\.dev\d+|(a|b|rc)\d", version))


# ---------------------------------------------------------------------------
# 1. Live configuration is present and structurally sound.
# ---------------------------------------------------------------------------


def test_pyproject_version_source_resists_non_release_artifact_tags() -> None:
    """The fix's three config knobs must all be present with the right shape.

    An accidental reversion (dropping the strict glob, loosening the regex, or
    removing the fallback) fails here before any git work runs.
    """
    raw = _raw_options()

    # The describe glob must require a digit, a literal dot, and another digit
    # region, i.e. be at least as strict as vcs_versioning's ``tag.strict`` glob
    # (``*[0-9]*.*[0-9]*``). The default glob ``*[0-9]*`` has no literal dot, so
    # any digit-bearing tag -- including the artifact tag -- matched it.
    glob = _describe_match_glob()
    assert "[0-9]" in glob, glob
    assert "." in glob, glob

    # The tag regex must capture a named ``version`` group.
    assert "version" in _compiled_tag_regex().groupindex

    # A fallback must exist and be a valid non-release PEP 440 version so an
    # artifact-tag-only checkout still builds.
    fallback = raw["fallback_version"]
    assert fallback, "fallback_version must be set"
    assert re.match(r"^\d+(\.\d+)+", fallback), fallback
    assert _is_non_release(fallback), f"fallback_version must be non-release: {fallback!r}"


# ---------------------------------------------------------------------------
# 2. The strict glob never selects an artifact tag.
# ---------------------------------------------------------------------------


def test_strict_glob_has_no_match_when_only_an_artifact_tag_is_reachable(tmp_path: Path) -> None:
    """A checkout whose sole reachable tag is the artifact tag must not select it.

    With the strict glob, ``git describe`` finds no matching tag and exits
    non-zero; vcs_versioning then uses ``fallback_version`` (a valid non-release
    version) and the editable build survives.
    """
    repo = tmp_path / "artifact-only"
    _init_repo(repo)
    _tag_head(repo, ARTIFACT_TAG)

    result = _run_configured_describe(repo)
    assert result.returncode != 0, (
        f"strict glob should reject {ARTIFACT_TAG!r}; describe output: {result.stdout!r}"
    )
    assert (
        "No names found" in result.stderr
        or "no tag" in result.stderr.lower()
        or result.stderr == ""
    )


@pytest.mark.parametrize("artifact_tag", [ARTIFACT_TAG, OTHER_ARTIFACT_TAG])
def test_strict_glob_ignores_artifact_tag_and_recovers_release_tag(
    tmp_path: Path, artifact_tag: str
) -> None:
    """Even when an artifact tag is the *nearest* tag, the strict glob skips it.

    This is the exact race that broke CI: the artifact tag sat closer to HEAD
    than the release tag, so the default glob selected it. The strict glob must
    walk past the artifact tag and recover the reachable release tag instead.
    """
    repo = tmp_path / "race"
    _init_repo(repo)
    _tag_head(repo, "0.0.2")  # release tag on the parent commit
    _add_commit(repo, "after-release")  # advance HEAD so the release tag is now behind
    _tag_head(repo, artifact_tag)  # artifact tag is now the nearest tag

    # Negative control: the *default* (pre-fix) glob DOES select the artifact
    # tag, which is precisely what raised ``Can't parse version from tag``.
    lax = subprocess.run(
        ["git", "describe", "--dirty", "--tags", "--long", "--match", "*[0-9]*"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert lax.returncode == 0, lax.stderr
    selected_lax, _ = _git_parse_describe(lax.stdout.strip())
    assert selected_lax == artifact_tag, (
        f"control: lax glob should have selected the artifact tag; got {selected_lax!r}"
    )

    # The fix: the configured (strict) glob skips the artifact tag and recovers
    # the reachable release tag instead.
    fixed = _run_configured_describe(repo)
    assert fixed.returncode == 0, fixed.stderr
    selected_fixed, _ = _git_parse_describe(fixed.stdout.strip())
    assert selected_fixed == "0.0.2", (
        f"strict glob should recover release tag 0.0.2; got {selected_fixed!r}"
    )


# ---------------------------------------------------------------------------
# 3. tag_regex maps release / candidate / prefixed tags to numeric X.Y.Z.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("0.0.2", "0.0.2"),
        ("0.0.3", "0.0.3"),
        ("rc0.0.3", "0.0.3"),
        ("v1.2.3", "1.2.3"),
        ("12.4.9", "12.4.9"),
        ("camera-ready-v0.0.1a", "0.0.1a"),
    ],
)
def test_tag_regex_extracts_numeric_version(tag: str, expected: str) -> None:
    """Release-line tags map to their numeric X.Y.Z (issue #6328 acceptance #2)."""
    assert _extract_version(tag) == expected


@pytest.mark.parametrize("tag", [ARTIFACT_TAG, OTHER_ARTIFACT_TAG, "main", ""])
def test_tag_regex_does_not_parse_non_release_tags(tag: str) -> None:
    """Non-release tags must not yield a version (defense in depth for the glob)."""
    assert _extract_version(tag) is None


# ---------------------------------------------------------------------------
# 4. End-to-end derivation (configured describe + tag_regex composition).
# ---------------------------------------------------------------------------


def test_release_tag_derives_its_version(tmp_path: Path) -> None:
    """A checkout tagged ``0.0.2`` derives package version ``0.0.2``.

    Issue #6328 acceptance #2: release-tag semantics are preserved unchanged.
    """
    repo = tmp_path / "release"
    _init_repo(repo)
    _tag_head(repo, "0.0.2")
    assert _derive_version(repo) == "0.0.2"


def test_release_candidate_tag_derives_base_version(tmp_path: Path) -> None:
    """A checkout tagged ``rc0.0.3`` derives package version ``0.0.3``.

    Issue #6328 acceptance #2: the ``rc`` prefix is stripped, the numeric
    release is kept.
    """
    repo = tmp_path / "candidate"
    _init_repo(repo)
    _tag_head(repo, "rc0.0.3")
    assert _derive_version(repo) == "0.0.3"


def test_artifact_only_checkout_derives_valid_non_release_fallback(tmp_path: Path) -> None:
    """A checkout whose only reachable tag is an artifact tag still derives.

    Issue #6328 acceptance #1 (derivation level): the configured strict glob
    rejects the artifact tag, describe finds no match, and vcs_versioning's
    fallback path yields the configured ``fallback_version``. The full editable
    build (``uv sync``) is acceptance evidence; this asserts the derivation
    survives and produces a valid non-release version.
    """
    repo = tmp_path / "artifact"
    _init_repo(repo)
    _tag_head(repo, ARTIFACT_TAG)
    derived = _derive_version(repo)
    assert derived == _raw_options()["fallback_version"]
    assert re.match(r"^\d+(\.\d+)+", derived), derived
    assert _is_non_release(derived), f"derived version must be non-release: {derived!r}"
