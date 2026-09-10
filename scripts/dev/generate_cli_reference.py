#!/usr/bin/env python3
"""Generate and source-check the installed entry-point reference.

Usage:
    uv run python scripts/dev/generate_cli_reference.py [--check]

Reads every ``[project.scripts]`` entry from ``pyproject.toml`` in deterministic
order, runs a bounded isolated ``--help`` smoke for each import-safe callable,
and renders ``docs/cli_reference.md`` from ``docs/cli_reference_meta.yaml``
plus live parser help. With ``--check``, exits 1 when the committed reference
differs from the deterministic render instead of writing it.

The ``--help`` smoke is local-only: it never touches the network, a simulator,
a scheduler, or repository artifacts beyond reading help text.
"""

from __future__ import annotations

import argparse
import difflib
import importlib
import inspect
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[2]
PYPROJECT_REL = Path("pyproject.toml")
META_REL = Path("docs/cli_reference_meta.yaml")
OUTPUT_REL = Path("docs/cli_reference.md")

ALLOWED_PROFILES = ("core", "benchmark", "carla")
HELP_TIMEOUT_S = 15
HELP_COLUMNS = 80

PARSER_GETTERS = ("get_parser", "_build_parser", "_configure_parser", "build_parser")

SUBCOMMAND_BRACE_RE = re.compile(r"\{([A-Za-z0-9_][A-Za-z0-9_\-, ]*)\}")


class CliReferenceError(ValueError):
    """Raised when metadata or project script inventory is malformed."""


@dataclass
class ProbeResult:
    """Isolated ``--help`` outcome for one console script."""

    script: str
    spec: str
    import_ok: bool = False
    import_error: str = ""
    help_ok: bool = False
    help_error: str = ""
    help_text: str = ""
    synopsis: str = ""
    subcommands: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """Return the compact help status label."""
        if self.help_ok:
            return "available"
        return f"unavailable: {self.help_error or self.import_error or 'unknown'}"


def load_project_scripts(pyproject_path: Path) -> dict[str, str]:
    """Load ``[project.scripts]`` entries in deterministic sorted order.

    Args:
        pyproject_path: Path to ``pyproject.toml``.

    Returns:
        Mapping of script name to ``module:attr`` spec, sorted by script name.

    Raises:
        CliReferenceError: When the file or the scripts table is missing.
    """
    if not pyproject_path.is_file():
        raise CliReferenceError(f"pyproject.toml not found: {pyproject_path}")
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    try:
        raw = data["project"]["scripts"]
    except KeyError as exc:
        raise CliReferenceError(f"[project.scripts] missing in {pyproject_path}") from exc
    if not isinstance(raw, dict) or not raw:
        raise CliReferenceError(f"[project.scripts] must be a non-empty mapping: {pyproject_path}")
    scripts: dict[str, str] = {}
    for name in sorted(raw):
        spec = raw[name]
        if not isinstance(spec, str) or ":" not in spec or not spec.strip():
            raise CliReferenceError(f"script {name!r}: spec must look like 'module:attr'")
        scripts[name] = spec.strip()
    return scripts


def load_metadata(meta_path: Path) -> dict:
    """Load the compact CLI reference metadata file.

    Args:
        meta_path: Path to ``docs/cli_reference_meta.yaml``.

    Returns:
        Raw parsed YAML mapping.

    Raises:
        CliReferenceError: When the file is missing or malformed.
    """
    if not meta_path.is_file():
        raise CliReferenceError(f"metadata file not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as handle:
        try:
            raw = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise CliReferenceError(f"metadata YAML parse error: {exc}") from exc
    if not isinstance(raw, dict):
        raise CliReferenceError("metadata root must be a mapping")
    if raw.get("version") != 1:
        raise CliReferenceError(f"unsupported metadata version: {raw.get('version')!r}")
    entries = raw.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise CliReferenceError("'entries' must be a non-empty mapping")
    return raw


def _require_non_empty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CliReferenceError(f"{label} must be a non-empty string")
    return " ".join(value.split())


def _normalize_subcommands(
    name: str, raw_subs: object, repo_root: Path, errors: list[str]
) -> dict[str, str] | None:
    """Normalize one entry's subcommand guides or record errors."""
    if not isinstance(raw_subs, dict):
        errors.append(f"entry {name!r}: 'subcommands' must be a mapping")
        return None
    subcommands: dict[str, str] = {}
    ok = True
    for sub, sub_guide in raw_subs.items():
        if isinstance(sub_guide, dict):
            sub_guide = sub_guide.get("guide")
        if not isinstance(sub_guide, str) or not sub_guide.strip():
            errors.append(f"entry {name!r} subcommand {sub!r}: missing 'guide'")
            ok = False
            continue
        cleaned = sub_guide.strip()
        if not (repo_root / cleaned).is_file():
            errors.append(f"entry {name!r} subcommand {sub!r}: guide missing: {cleaned}")
            ok = False
            continue
        subcommands[sub] = cleaned
    return subcommands if ok else None


def _normalize_single_entry(
    name: str, entry: object, scripts: dict[str, str], repo_root: Path, errors: list[str]
) -> dict | None:
    """Normalize one metadata entry or record fail-closed errors."""
    if name not in scripts:
        errors.append(f"stale documented entry not in [project.scripts]: {name!r}")
        return None
    if not isinstance(entry, dict):
        errors.append(f"entry {name!r}: must be a mapping")
        return None
    try:
        purpose = _require_non_empty(entry.get("purpose"), f"entry {name!r} 'purpose'")
        profile = _require_non_empty(entry.get("profile"), f"entry {name!r} 'profile'")
        availability = _require_non_empty(
            entry.get("availability"), f"entry {name!r} 'availability'"
        )
        guide = _require_non_empty(entry.get("guide"), f"entry {name!r} 'guide'")
    except CliReferenceError as exc:
        errors.append(str(exc))
        return None
    if profile not in ALLOWED_PROFILES:
        errors.append(
            f"entry {name!r}: unknown profile {profile!r} (allowed: {', '.join(ALLOWED_PROFILES)})"
        )
        return None
    if not (repo_root / guide).is_file():
        errors.append(f"entry {name!r}: guide does not exist: {guide}")
        return None
    subcommands = _normalize_subcommands(
        name, entry.get("subcommands", {}) or {}, repo_root, errors
    )
    if subcommands is None:
        return None
    return {
        "purpose": purpose,
        "profile": profile,
        "availability": availability,
        "guide": guide,
        "subcommands": subcommands,
    }


def validate_metadata_entries(
    scripts: dict[str, str], raw: dict, repo_root: Path
) -> tuple[dict[str, dict], list[str]]:
    """Validate metadata against the declared scripts and existing guides.

    Args:
        scripts: Sorted script inventory from ``pyproject.toml``.
        raw: Raw metadata mapping from :func:`load_metadata`.
        repo_root: Repository root for guide existence checks.

    Returns:
        Tuple of normalized entries and fail-closed error strings.
    """
    errors: list[str] = []
    entries: dict[str, dict] = {}
    raw_entries = raw.get("entries", {})
    for name, entry in raw_entries.items():
        normalized = _normalize_single_entry(name, entry, scripts, repo_root, errors)
        if normalized is not None:
            entries[name] = normalized
    for name in sorted(scripts):
        if name not in raw_entries:
            errors.append(f"missing documentation owner for declared script: {name!r}")
    return entries, errors


def _parser_subcommands(parser: object) -> list[str]:
    """Extract sorted subcommand names from a live argparse parser."""
    subcommands: set[str] = set()
    actions = getattr(parser, "_actions", [])
    for action in actions:
        choices = getattr(action, "choices", None)
        action_type = type(action).__name__
        if action_type == "_SubParsersAction" and isinstance(choices, dict):
            for key in choices:
                if isinstance(key, str) and key.strip():
                    subcommands.add(key.strip())
    return sorted(subcommands)


def get_live_parser_info(module_name: str) -> tuple[str, list[str]]:
    """Return ``(description, subcommands)`` from an import-safe parser getter.

    Args:
        module_name: Dotted module owning the console script callable.

    Returns:
        Parser description (possibly empty) and sorted subcommand names. Both
        are empty when no known parser getter exists.
    """
    try:
        module = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 - probe path; import errors handled by help smoke
        return "", []
    for getter in PARSER_GETTERS:
        factory = getattr(module, getter, None)
        if not callable(factory):
            continue
        try:
            parser = factory()
        except Exception:  # noqa: BLE001 - factory may need args; fall back to help text
            continue
        description = str(getattr(parser, "description", "") or "").strip()
        description = " ".join(description.split())
        return description, _parser_subcommands(parser)
    # robot_sf.cli exposes only _build_parser (covered above); keep the direct
    # fallback for modules that build the parser under a private name.
    return "", []


def extract_subcommands_from_help(help_text: str) -> list[str]:
    """Extract candidate subcommand names from ``--help`` usage braces."""
    found: set[str] = set()
    for match in SUBCOMMAND_BRACE_RE.finditer(help_text):
        for token in match.group(1).split(","):
            token = token.strip()
            if token and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-]*", token):
                found.add(token)
    return sorted(found)


def _skip_usage_block(lines: list[str]) -> int:
    """Return the index just after the usage block and blank lines."""
    idx = 0
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("usage:"):
            idx = i + 1
            while idx < len(lines) and lines[idx].strip():
                if not lines[idx].startswith((" ", "\t")) and idx > i + 2:
                    break
                idx += 1
            break
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    return idx


def _collect_description_paragraph(lines: list[str], idx: int) -> tuple[list[str], int]:
    """Collect wrapped description lines until a blank line or section header."""
    paragraph: list[str] = []
    section_headers = ("positional arguments:", "options:", "optional arguments:", "arguments:")
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped or stripped.lower() in section_headers:
            break
        paragraph.append(stripped)
        idx += 1
    return paragraph, idx


def extract_synopsis(help_text: str, fallback_description: str = "") -> str:
    """Extract the one-line help synopsis after the usage block."""
    lines = help_text.splitlines()
    idx = _skip_usage_block(lines)
    paragraph, _ = _collect_description_paragraph(lines, idx)
    if paragraph:
        return " ".join(" ".join(paragraph).split())[:500]
    if fallback_description:
        return fallback_description[:500]
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.lower().startswith("usage:"):
            return " ".join(stripped.split())[:500]
    return ""


def _help_call_code(script: str, module_name: str, attr: str, takes_argv: bool) -> str:
    """Build the isolated ``--help`` snippet for a subprocess."""
    if takes_argv:
        return (
            "import sys; sys.argv=[sys.argv[1]]; "
            f"from {module_name} import {attr} as _fn; "
            "raise SystemExit(_fn(['--help']))"
        )
    return (
        "import sys; sys.argv=[sys.argv[1], '--help']; "
        f"from {module_name} import {attr} as _fn; "
        "raise SystemExit(_fn())"
    )


def probe_help(
    script: str,
    spec: str,
    repo_root: Path,
    timeout_s: int = HELP_TIMEOUT_S,
) -> ProbeResult:
    """Run a bounded isolated ``--help`` smoke for one console script.

    Args:
        script: Console script name (used as ``sys.argv[0]`` for stable help).
        spec: ``module:attr`` callable spec from ``pyproject.toml``.
        repo_root: Repository root (subprocess working directory).
        timeout_s: Subprocess timeout in seconds.

    Returns:
        Populated :class:`ProbeResult` with normalized help text, synopsis,
        and subcommands when available.
    """
    result = ProbeResult(script=script, spec=spec)
    if ":" not in spec:
        result.import_error = f"malformed spec {spec!r}"
        return result
    module_name, attr = spec.split(":", 1)
    module_name, attr = module_name.strip(), attr.strip()
    try:
        module = importlib.import_module(module_name)
        target = getattr(module, attr)
    except Exception as exc:  # noqa: BLE001 - fail-closed probe records the reason
        result.import_error = f"{type(exc).__name__}: {exc}".strip()[:300]
        return result
    result.import_ok = True

    try:
        takes_argv = len(inspect.signature(target).parameters) > 0
    except (TypeError, ValueError):
        takes_argv = True
    code = _help_call_code(script, module_name, attr, takes_argv)
    env = dict(os.environ)
    env["COLUMNS"] = str(HELP_COLUMNS)
    env["LINES"] = "24"
    # Keep the smoke read-only: --help must never need credentials or remotes.
    env.pop("CARLA_HOST", None)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code, script],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        result.help_error = f"--help timed out after {timeout_s}s"
        return result
    except (OSError, ValueError) as exc:  # subprocess launch failures only
        result.help_error = f"--help launch failed: {exc}".strip()[:300]
        return result
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        first = " ".join(detail[0].split()) if detail else "exit nonzero"
        result.help_error = f"--help exit {completed.returncode}: {first}"[:300]
        return result
    stdout = completed.stdout or ""
    # Normalize for byte-stability: strip trailing spaces, enforce LF ending.
    normalized = "\n".join(line.rstrip() for line in stdout.splitlines()).strip() + "\n"
    if not normalized.strip():
        result.help_error = "--help produced no output"
        return result
    result.help_ok = True
    result.help_text = normalized
    live_description, live_subs = get_live_parser_info(module_name)
    result.synopsis = extract_synopsis(normalized, live_description)
    # Prefer the import-safe parser subcommands when available; help-text braces
    # also match option choices (e.g. --log-level {CRITICAL,...}), so text
    # extraction is only a fallback for parsers without an exposed getter.
    if live_subs:
        result.subcommands = live_subs
    else:
        result.subcommands = extract_subcommands_from_help(normalized)
    return result


def _check_single_script(
    name: str, entries: dict[str, dict], probes: dict[str, ProbeResult]
) -> list[str]:
    """Validate one script's owner, import, help, and subcommand coverage."""
    if name not in entries:
        return [f"missing documentation owner for declared script: {name!r}"]
    probe = probes.get(name)
    if probe is None:
        return [f"missing --help probe for declared script: {name!r}"]
    profile = entries[name]["profile"]
    if not probe.import_ok:
        if profile == "carla":
            return []
        return [f"callable cannot import for {name!r}: {probe.import_error}"]
    if not probe.help_ok:
        if profile == "carla":
            return []
        return [f"--help failed for {name!r}: {probe.help_error}"]
    if name != "robot-sf":
        return []
    want = set(probe.subcommands)
    have = set(entries[name].get("subcommands", {}))
    sub_errors = [
        f"missing documentation owner for subcommand: 'robot-sf {s}'" for s in sorted(want - have)
    ]
    sub_errors += [
        f"stale documented subcommand not in live parser: 'robot-sf {s}'"
        for s in sorted(have - want)
    ]
    return sub_errors


def check_entry_points(
    scripts: dict[str, str],
    entries: dict[str, dict],
    probes: dict[str, ProbeResult],
) -> list[str]:
    """Fail-closed validation over scripts, metadata owners, and help smokes.

    Args:
        scripts: Sorted script inventory from ``pyproject.toml``.
        entries: Validated metadata entries keyed by script name.
        probes: Help probe results keyed by script name.

    Returns:
        Error strings; an empty list means the reference is complete.
    """
    errors: list[str] = []
    for name in sorted(scripts):
        errors.extend(_check_single_script(name, entries, probes))
    for name in sorted(entries):
        if name not in scripts:
            errors.append(f"stale documented entry not in [project.scripts]: {name!r}")
    return errors


def _md_link(guide: str) -> str:
    """Return a docs-relative markdown link for a repo-relative guide path."""
    target = guide
    if target.startswith("docs/"):
        target = target[len("docs/") :]
    return f"[{target}]({target})"


def render_markdown(
    scripts: dict[str, str],
    entries: dict[str, dict],
    probes: dict[str, ProbeResult],
) -> str:
    """Render the byte-stable CLI reference markdown.

    Args:
        scripts: Sorted script inventory from ``pyproject.toml``.
        entries: Validated metadata entries keyed by script name.
        probes: Help probe results keyed by script name.

    Returns:
        Deterministic markdown text ending with exactly one newline.
    """
    lines = [
        "# CLI Reference (Installed Entry Points)",
        "",
        "Plain-language summary: this is the complete list of installed commands,",
        "generated from `pyproject.toml` and live `--help` so the docs cannot drift",
        "from the packaged entry points. See the [Glossary](glossary.md) for",
        "acronyms and project terms.",
        "",
        "> Generated file — do not edit by hand. Regenerate with",
        "> `uv run python scripts/dev/generate_cli_reference.py`.",
        "> Sources: `pyproject.toml [project.scripts]` plus",
        "> `docs/cli_reference_meta.yaml` plus live `--help` (local-only, no network,",
        "> simulator, scheduler, or artifact mutation).",
        "",
        "## Overview",
        "",
        "| Command | Purpose | Profile | Availability | Guide | Help |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in sorted(scripts):
        meta = entries.get(name, {})
        probe = probes.get(name)
        purpose = meta.get("purpose", "-")
        profile = meta.get("profile", "-")
        availability = meta.get("availability", "-")
        guide = meta.get("guide", "-")
        guide_cell = _md_link(guide) if guide != "-" else "-"
        help_cell = probe.status if probe is not None else "unknown"
        # Keep table cells single-line for byte-stability.
        purpose_cell = " ".join(str(purpose).split())
        availability_cell = " ".join(str(availability).split())
        lines.append(
            f"| `{name}` | {purpose_cell} | {profile} | {availability_cell} "
            f"| {guide_cell} | {help_cell} |"
        )
    lines += ["", "## Commands", ""]
    for name in sorted(scripts):
        meta = entries.get(name, {})
        probe = probes.get(name)
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append(f"- Callable: `{scripts[name]}`")
        lines.append(f"- Profile: `{meta.get('profile', '-')}`")
        lines.append(f"- Availability: {meta.get('availability', '-')}")
        guide = meta.get("guide", "-")
        lines.append(f"- Guide: {_md_link(guide) if guide != '-' else '-'}")
        if probe is not None:
            lines.append(f"- Help: {probe.status}")
            if probe.synopsis:
                lines.append(f"- Synopsis: {probe.synopsis}")
        lines.append("")
        subcommands = list(probe.subcommands) if probe is not None else []
        if name == "robot-sf" and subcommands:
            lines.append(
                "Nested `robot-sf` subcommands (summary only; see each task guide for flags):"
            )
            lines.append("")
            lines.append("| Subcommand | Help | Guide |")
            lines.append("| --- | --- | --- |")
            sub_help = _robot_sf_subcommand_help()
            sub_guides: dict[str, str] = meta.get("subcommands", {})
            for sub in sorted(subcommands):
                help_text = sub_help.get(sub, "-")
                guide_path = sub_guides.get(sub, meta.get("guide", "-"))
                guide_cell = _md_link(guide_path) if guide_path != "-" else "-"
                lines.append(f"| `{sub}` | {help_text} | {guide_cell} |")
            lines.append("")
        elif subcommands:
            entry_guide = meta.get("guide", "-")
            lines.append("Subcommands (summary only; see the task guide for flags):")
            lines.append("")
            lines.append("| Subcommand | Guide |")
            lines.append("| --- | --- |")
            for sub in sorted(subcommands):
                guide_cell = _md_link(entry_guide) if entry_guide != "-" else "-"
                lines.append(f"| `{sub}` | {guide_cell} |")
            lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("- Script order is sorted from `[project.scripts]` for determinism.")
    lines.append("- `--help` runs in an isolated subprocess with a fixed width and timeout.")
    lines.append("- No network, simulator, scheduler, or artifact mutation occurs during checks.")
    lines.append("- CI fails on drift: run the generator without `--check` to refresh this file.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _robot_sf_subcommand_help() -> dict[str, str]:
    """Return one-line help for each live ``robot-sf`` subcommand."""
    try:
        from robot_sf import cli as robot_cli
    except Exception:  # noqa: BLE001 - fall back to empty; probe still records subcommands
        return {}
    try:
        parser = robot_cli._build_parser()
    except Exception:  # noqa: BLE001 - defensive; help smoke remains authoritative
        return {}
    helps: dict[str, str] = {}
    for action in getattr(parser, "_actions", []):
        if type(action).__name__ != "_SubParsersAction":
            continue
        choices = getattr(action, "choices", {}) or {}
        for key, subparser in choices.items():
            if not isinstance(key, str):
                continue
            text = str(getattr(subparser, "description", "") or getattr(action, "help", ""))
            # Prefer the per-subcommand help registered on the subparsers action.
            help_text = ""
            for sub_action in getattr(action, "_choices_actions", []):
                if getattr(sub_action, "dest", None) == key or sub_action.metavar == key:
                    help_text = str(getattr(sub_action, "help", "") or "")
                    break
            candidate = help_text or text
            helps[key] = " ".join(candidate.split())[:200] or "-"
    return helps


def build_parser() -> argparse.ArgumentParser:
    """Return the generator argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument("--pyproject", type=Path, default=None)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--timeout",
        type=int,
        default=HELP_TIMEOUT_S,
        help="Per-command --help subprocess timeout in seconds",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when the committed reference differs from the deterministic render",
    )
    return parser


def generate(
    repo_root: Path, pyproject: Path, meta_path: Path, timeout_s: int
) -> tuple[str, list[str], dict[str, str], dict[str, dict], dict[str, ProbeResult]]:
    """Load, probe, validate, and render the CLI reference.

    Args:
        repo_root: Repository root.
        pyproject: Path to ``pyproject.toml``.
        meta_path: Path to the compact metadata YAML.
        timeout_s: Per-command help timeout.

    Returns:
        Tuple of rendered markdown, error list, scripts, entries, and probes.
    """
    scripts = load_project_scripts(pyproject)
    raw = load_metadata(meta_path)
    entries, meta_errors = validate_metadata_entries(scripts, raw, repo_root)
    probes: dict[str, ProbeResult] = {}
    for name in sorted(scripts):
        probes[name] = probe_help(name, scripts[name], repo_root, timeout_s)
    errors = list(meta_errors)
    errors.extend(check_entry_points(scripts, entries, probes))
    rendered = render_markdown(scripts, entries, probes)
    return rendered, errors, scripts, entries, probes


def main(argv: list[str] | None = None) -> int:
    """Render or source-check ``docs/cli_reference.md``."""
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    pyproject = Path(args.pyproject).resolve() if args.pyproject else repo_root / PYPROJECT_REL
    meta_path = Path(args.meta).resolve() if args.meta else repo_root / META_REL
    output_path = Path(args.output).resolve() if args.output else repo_root / OUTPUT_REL
    try:
        rendered, errors, scripts, _entries, _probes = generate(
            repo_root, pyproject, meta_path, args.timeout
        )
    except CliReferenceError as exc:
        print(f"cli reference invalid: {exc}", file=sys.stderr)
        return 2
    if args.check:
        if errors:
            for error in errors:
                print(f"cli reference error: {error}")
            return 1
        try:
            current = output_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"cli reference error: cannot read {output_path}: {exc}")
            return 1
        if rendered == current:
            print(f"cli reference up to date: {len(scripts)} entry points")
            return 0
        diff = difflib.unified_diff(
            current.splitlines(),
            rendered.splitlines(),
            fromfile=str(output_path),
            tofile=f"{output_path} (rendered)",
            lineterm="",
        )
        snippet = "\n".join(list(diff)[:40])
        print(f"cli reference drift detected; first diff lines:\n{snippet}")
        return 1
    if errors:
        for error in errors:
            print(f"cli reference error: {error}")
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"rendered {output_path} ({len(scripts)} entry points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
