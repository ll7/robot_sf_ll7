#!/usr/bin/env python3
"""Guard against Ruff version drift across four independent version axes.

Checks that the Ruff version is consistent across:

1. The ``astral-sh/ruff-pre-commit`` ``rev:`` in ``.pre-commit-config.yaml``
2. ``[tool.ruff]`` ``required-version`` in ``pyproject.toml``
3. The ``ruff==`` pin in ``[dependency-groups].dev`` in ``pyproject.toml``
4. Every step's ``run`` text mentioning Ruff in ``.github/workflows/*.yml``/``*.yaml``

Axis 4 matters because the CI lint step launches Ruff through ``uvx --from
ruff==<pin>`` (see ``.github/workflows/ci.yml``) while ``required-version``
refuses to run any Ruff whose version disagrees with it. A bump of only axis 3
therefore leaves the CI lint step unable to start, even though the first three
axes still agree with each other.

Exits non-zero when the versions disagree. It is wired into the development
validation surface as a gating local pre-commit hook and a gating CI lint check
so that future drift fails fast. ``--advisory`` remains available for explicitly
non-blocking diagnostic invocations.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRE_COMMIT = REPO_ROOT / ".pre-commit-config.yaml"
DEFAULT_PYPROJECT = REPO_ROOT / "pyproject.toml"
DEFAULT_WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
RUFF_PRE_COMMIT_REPO = "https://github.com/astral-sh/ruff-pre-commit"
RUFF_PIN_PATTERN = re.compile(r"^ruff\s*==\s*(?P<version>\S+)$")
# Deliberately small, whole-line grammar, not a shell parser. Any Ruff mention
# outside this grammar fails closed under every shell. Arguments are literal
# words only (optionally quoted): no expansion, command chaining, comments or
# redirection. Quoted literal config overrides preserve ``--config 'lint.ignore=[]'``
# in the security workflow without mistaking the override for a config-file path.
# New supported forms require an explicit policy change with regression tests.
RUFF_WORKFLOW_PIN_PATTERN = re.compile(
    r"[ \t]*uvx[ \t]+--from[ \t]+ruff==(?P<version>[0-9]+\.[0-9]+\.[0-9]+)"
    r"[ \t]+ruff[ \t]+(?:check|format)"
    r"""(?P<args>(?:[ \t]+(?:[-a-zA-Z0-9_./=,\[\]]+|'[-a-zA-Z0-9_./=,\[\]]*'|"[-a-zA-Z0-9_./=,\[\]]*"))*)[ \t]*"""
)


def normalize_version(raw: str) -> str:
    """Normalize a version string by stripping common prefixes.

    Strips, in order:
    * surrounding whitespace and quote characters,
    * a ``ruff==`` prefix,
    * a ``==`` prefix,
    * a leading ``v``.

    Examples:
        ``v0.16.0`` -> ``0.16.0``
        ``==0.16.0`` -> ``0.16.0``
        ``ruff==0.16.0`` -> ``0.16.0``
    """
    raw = raw.strip().strip("\"'")
    raw = re.sub(r"^ruff\s*==\s*", "", raw)
    raw = re.sub(r"^==\s*", "", raw)
    raw = re.sub(r"^v", "", raw)
    return raw


def read_pre_commit_version(path: Path) -> str:
    """Return the Ruff rev from ``.pre-commit-config.yaml``.

    Raises:
        ValueError: If the ruff-pre-commit repo or its rev is not found.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    repos = data.get("repos", [])
    for repo in repos:
        repo_url = str(repo.get("repo", "")).removesuffix(".git")
        if repo_url == RUFF_PRE_COMMIT_REPO:
            rev = repo.get("rev")
            if rev is not None:
                return str(rev)
            raise ValueError(f"ruff-pre-commit entry in {path} has no 'rev'")
    raise ValueError(f"no astral-sh/ruff-pre-commit repo found in {path}")


def read_pyproject_required_version(path: Path) -> str:
    """Return ``required-version`` from ``[tool.ruff]`` in ``pyproject.toml``.

    Raises:
        ValueError: If the field is missing or not found.
    """
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    tool = data.get("tool", {})
    ruff = tool.get("ruff", {})
    version = ruff.get("required-version")
    if version is not None:
        return str(version)
    raise ValueError(f"no [tool.ruff] required-version found in {path}")


def read_pyproject_dev_dep_pin(path: Path) -> str:
    """Return the exact ``ruff==X.Y.Z`` pin from ``[dependency-groups].dev``.

    Raises:
        ValueError: If the development group is malformed or lacks an exact Ruff pin.
    """
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    dep_groups = data.get("dependency-groups", {})
    dev_dependencies = dep_groups.get("dev", [])
    if not isinstance(dev_dependencies, list):
        raise ValueError(f"[dependency-groups].dev in {path} must be a list")
    for dependency in dev_dependencies:
        if isinstance(dependency, str) and RUFF_PIN_PATTERN.fullmatch(dependency.strip()):
            return dependency.strip()
    raise ValueError(f"no exact ruff== pin found in [dependency-groups].dev in {path}")


def _read_run_ruff_pins(run: str, location: str) -> list[tuple[str, str]]:
    """Account for every Ruff mention before returning any pins from a step.

    Deliberate indirection (variables, eval, command substitution, called scripts)
    is out of scope and tracked in #10433.
    """
    pins: list[tuple[str, str]] = []
    for line in run.replace("\\\n", "").splitlines():
        # Detect literal shell spellings, then apply the original launcher grammar.
        mention_line = re.sub(r"""\$''|\$""|['"\\]""", "", line)
        if "ruff" not in mention_line.lower():
            continue
        match = RUFF_WORKFLOW_PIN_PATTERN.fullmatch(line)
        if match is None or "ruff" in match.group("args").lower():
            raise ValueError(
                f"{location}: unrecognised Ruff mention: {line.strip()!r}; "
                "use a standalone pinned uvx --from ruff==X.Y.Z ruff "
                "check|format launcher with literal arguments"
            )
        pins.append((location, match.group("version")))
    return pins


def read_workflow_ruff_pins(workflows_dir: Path) -> list[tuple[str, str]]:
    """Validate every Ruff mention and return recognised CI launcher pins.

    Loads ``*.yml``/``*.yaml`` directly inside ``workflows_dir`` and scans
    every step's parsed ``run`` string. YAML scalar folding/quoting happens
    before shell backslash-newline continuations are joined. Locations name
    the workflow, job and zero-based step index rather than physical YAML lines.
    Each logical line mentioning ``ruff`` (case-insensitively, even in comments
    or filenames) must fully match the pinned ``uvx --from ruff==X.Y.Z ruff
    check|format`` launcher grammar. Other shells' continuation forms are
    unrecognised and rejected, rather than silently skipped. YAML metadata
    outside ``run`` is not scanned. There are no implicit allowlist exceptions.

    Returns:
        Ordered ``(location, raw_version)`` pairs; empty when no launcher
        pin exists (the caller treats that as fail-closed drift).

    Raises:
        ValueError: If YAML cannot be parsed or any Ruff mention is unrecognised.
    """
    pins: list[tuple[str, str]] = []
    if not workflows_dir.is_dir():
        return pins
    for candidate in sorted(workflows_dir.iterdir()):
        if not candidate.is_file() or candidate.suffix not in {".yml", ".yaml"}:
            continue
        try:
            workflow = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"cannot parse workflow {candidate}: {exc}") from exc
        for job_id, job in (workflow or {}).get("jobs", {}).items():
            for step_index, step in enumerate(job.get("steps", [])):
                run = step.get("run")
                if not isinstance(run, str):
                    continue
                location = f"{candidate.name}:jobs.{job_id}.steps[{step_index}].run"
                pins.extend(_read_run_ruff_pins(run, location))
    return pins


def evaluate(
    pre_commit_version: str,
    required_version: str,
    dev_pin: str,
    workflow_pins: tuple[tuple[str, str], ...] | list[tuple[str, str]] = (),
) -> list[str]:
    """Compare four Ruff version axes and return a list of mismatches.

    Args:
        pre_commit_version: Raw ``rev:`` from ``.pre-commit-config.yaml``.
        required_version: Raw ``required-version`` from ``pyproject.toml``.
        dev_pin: Raw ``ruff==`` pin from ``[dependency-groups].dev``.
        workflow_pins: ``(location, raw_version)`` pairs from
            :func:`read_workflow_ruff_pins`; every pin must agree with the
            other three axes. An empty collection means no launcher pin was
            found and is reported as drift so a silent scan can never pass.

    Returns:
        Human-readable problem strings; empty means all four agree.
    """
    problems: list[str] = []

    norm_pre = normalize_version(pre_commit_version)
    norm_req = normalize_version(required_version)
    norm_dev = normalize_version(dev_pin)

    if norm_pre != norm_req:
        problems.append(
            f".pre-commit-config.yaml rev {pre_commit_version!r} "
            f"(normalized {norm_pre!r}) != "
            f"pyproject.toml [tool.ruff] required-version "
            f"{required_version!r} (normalized {norm_req!r})"
        )
    if norm_pre != norm_dev:
        problems.append(
            f".pre-commit-config.yaml rev {pre_commit_version!r} "
            f"(normalized {norm_pre!r}) != "
            f"pyproject.toml [dependency-groups] ruff pin "
            f"{dev_pin!r} (normalized {norm_dev!r})"
        )
    if norm_req != norm_dev:
        problems.append(
            f"pyproject.toml [tool.ruff] required-version "
            f"{required_version!r} (normalized {norm_req!r}) != "
            f"pyproject.toml [dependency-groups] ruff pin "
            f"{dev_pin!r} (normalized {norm_dev!r})"
        )

    if not workflow_pins:
        problems.append(
            "no uvx --from ruff== launcher pin found in .github/workflows/*.yml: "
            "either the CI lint launchers were removed (update this guard) or the "
            "workflows directory was not scanned"
        )
    for location, raw_version in workflow_pins:
        norm_workflow = normalize_version(raw_version)
        if norm_workflow != norm_dev:
            problems.append(
                f".github/workflows/{location} ruff=={raw_version} "
                f"(normalized {norm_workflow!r}) != "
                f"pyproject.toml [dependency-groups] ruff pin "
                f"{dev_pin!r} (normalized {norm_dev!r})"
            )

    return problems


def main(argv: list[str] | None = None) -> int:
    """Run the Ruff version parity check.

    Returns:
        Exit code 0 when all versions agree, 1 on mismatch (or 0 in advisory).
        Unparseable workflows and unrecognised Ruff mentions always fail,
        including in advisory mode.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pre-commit-config",
        type=Path,
        default=DEFAULT_PRE_COMMIT,
        help="Path to .pre-commit-config.yaml (default: repo root)",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=DEFAULT_PYPROJECT,
        help="Path to pyproject.toml (default: repo root)",
    )
    parser.add_argument(
        "--workflows-dir",
        type=Path,
        default=DEFAULT_WORKFLOWS_DIR,
        help="Directory of CI workflow files scanned for uvx ruff pins"
        " (default: <repo>/.github/workflows)",
    )
    parser.add_argument(
        "--advisory",
        action="store_true",
        help="Report version drift without failing; invalid or unrecognised workflows still fail.",
    )
    args = parser.parse_args(argv)

    pre_commit_version = read_pre_commit_version(args.pre_commit_config)
    required_version = read_pyproject_required_version(args.pyproject)
    dev_pin = read_pyproject_dev_dep_pin(args.pyproject)
    try:
        workflow_pins = read_workflow_ruff_pins(args.workflows_dir)
    except ValueError as exc:
        print(f"Ruff workflow guard failed: {exc}", file=sys.stderr)
        return 1

    problems = evaluate(pre_commit_version, required_version, dev_pin, tuple(workflow_pins))

    print(f".pre-commit-config.yaml rev:       {pre_commit_version!r}")
    print(f"pyproject.toml required-version:   {required_version!r}")
    print(f"pyproject.toml dev ruff pin:       {dev_pin!r}")
    if workflow_pins:
        print("workflow uvx ruff pins:")
        for location, raw_version in workflow_pins:
            print(f"  - .github/workflows/{location}: ruff=={raw_version}")
    else:
        print("workflow uvx ruff pins:          (none found)")

    if not problems:
        print("OK: all four Ruff version axes agree.")
        return 0

    print("\nRuff version drift detected:")
    for problem in problems:
        print(f"  - {problem}")
    if args.advisory:
        print("\n(advisory mode: not failing this step)")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
