#!/usr/bin/env python3
"""Prove staged workloads use no untracked, sibling-checkout, or leaked code (#8858).

Validates that a staged command executes only from its intended immutable source
bundle and declared companion inputs, without importing from ambient PYTHONPATH,
user-site packages, sibling worktrees, untracked/dirty source checkouts, stale
editable installations, or .pth-injected paths.

Outputs deterministic sanitized JSON or text receipts under
``staged_source_isolation_receipt.v1`` without exposing raw private host paths.
Exit codes:
  0 - source isolation verified (status: passed)
  1 - isolation violation detected (status: blocked)
  2 - malformed packet or usage error
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA = "staged_source_isolation_receipt.v1"
PACKET_SCHEMA = "staged_source_isolation_packet.v1"
CLAIM_BOUNDARY = (
    "Operational staging verification only. Confirms runtime source isolation "
    "and environment boundaries before workload submission; contains no benchmark "
    "or scientific result evidence."
)
DEFAULT_FIRST_PARTY = ["robot_sf"]


@dataclass(frozen=True)
class IsolationProblem:
    """A deterministic fail-closed violation of source isolation."""

    code: str
    path_class: str
    sanitized_path: str
    message: str


def _match_relative(target: Path, base: Path, cls: str, tag: str) -> tuple[str, str] | None:
    try:
        return cls, f"<{tag}>/{target.relative_to(base).as_posix()}"
    except ValueError:
        return None


def _classify_external_fallback(res: Path, home_dir: Path | None) -> tuple[str, str]:
    parts = res.parts
    if any(".worktrees" in p or p.endswith(".worktree") for p in parts):
        return "sibling_worktree", f"<sibling_worktree>/{res.name}"
    if ("python" in res.as_posix().lower() and ("lib" in parts or "stdlib" in parts)) or (
        sys.base_prefix and str(res).startswith(sys.base_prefix)
    ):
        return "standard_library", f"<stdlib>/{res.name}"
    if home_dir and _match_relative(res, home_dir.resolve(), "user_home", "user_home"):
        top = res.relative_to(home_dir.resolve()).parts
        return "user_home", f"<user_home>/{top[0] if top else ''}/..."
    return "external", f"<external>/{res.name}"


def sanitize_path(
    path: Path | str,
    *,
    source_root: Path,
    declared_companions: Mapping[str, Path] | None = None,
    venv_root: Path | None = None,
    user_site_dir: Path | None = None,
    home_dir: Path | None = None,
) -> tuple[str, str]:
    """Classify and sanitize a path so private machine locators are never emitted."""
    try:
        res = Path(path).resolve()
    except (OSError, RuntimeError, ValueError):
        res = Path(str(path))

    if m := _match_relative(res, source_root.resolve(), "staged_source", "staged_source"):
        return m
    if declared_companions:
        for name, comp_root in declared_companions.items():
            if m := _match_relative(
                res, comp_root.resolve(), "declared_companion", f"companion:{name}"
            ):
                return m
    if venv_root and (m := _match_relative(res, venv_root.resolve(), "virtualenv", "venv")):
        return m
    if user_site_dir and (
        m := _match_relative(res, user_site_dir.resolve(), "user_site", "user_site")
    ):
        return m
    return _classify_external_fallback(res, home_dir)


def inspect_git_state(source_root: Path) -> dict[str, Any]:
    """Inspect git working tree cleanliness and identity without mutations."""

    def _git(*args: str) -> str | None:
        p = subprocess.run(
            ["git", "-C", str(source_root), *args], capture_output=True, text=True, check=False
        )
        return p.stdout.strip() if p.returncode == 0 else None

    status_out = _git("status", "--porcelain=v1") or ""
    untracked, modified = [], []
    for line in status_out.splitlines():
        (untracked if line.startswith("??") else modified).append(line[3:].strip())

    return {
        "commit": _git("rev-parse", "HEAD"),
        "tree": _git("rev-parse", "HEAD^{tree}"),
        "dirty": bool(modified),
        "untracked_count": len(untracked),
        "modified_count": len(modified),
        "untracked_files": untracked,
        "modified_files": modified,
    }


PROBE_SUBPROCESS_SCRIPT = """
import importlib, importlib.metadata, json, os, site, sys
from pathlib import Path
pkgs = json.loads(os.environ.get("ROBOT_SF_FIRST_PARTY", "[]"))
res = {
    "sys_path": sys.path,
    "user_site_enabled": getattr(site, "ENABLE_USER_SITE", False),
    "user_site_dir": site.getusersitepackages() if hasattr(site, "getusersitepackages") else None,
    "pth_files": [], "distributions": [], "first_party_imports": {},
}
for e in sys.path:
    p = Path(e)
    if p.is_dir():
        for f in p.glob("*.pth"):
            try:
                res["pth_files"].append({"file": str(f), "lines": [l.strip() for l in f.read_text("utf-8", "replace").splitlines() if l.strip() and not l.startswith("#")]})
            except Exception: pass
for d in importlib.metadata.distributions():
    u = d.read_text("direct_url.json")
    res["distributions"].append({"name": d.metadata.get("Name") or "", "version": d.version, "direct_url": json.loads(u) if u else None})
for pkg in pkgs:
    entry = {"imported": False, "file": None, "path": None, "error": None, "native_extensions": []}
    try:
        m = importlib.import_module(pkg)
        entry.update({"imported": True, "file": getattr(m, "__file__", None), "path": list(getattr(m, "__path__", []))})
        for mn, mod in list(sys.modules.items()):
            if (mn == pkg or mn.startswith(f"{pkg}.")) and getattr(mod, "__file__", None):
                if Path(mod.__file__).suffix.lower() in {".so", ".pyd", ".dylib"}:
                    entry["native_extensions"].append(mod.__file__)
    except Exception as exc:
        entry["error"] = str(exc)
    res["first_party_imports"][pkg] = entry
sys.stdout.write(json.dumps(res))
"""


def run_environment_probe(
    *,
    source_root: Path,
    first_party_packages: list[str],
    python_executable: str = sys.executable,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run a bounded non-executing import probe in a clean subprocess."""
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    env["ROBOT_SF_FIRST_PARTY"] = json.dumps(first_party_packages)
    proc = subprocess.run(
        [python_executable, "-c", PROBE_SUBPROCESS_SCRIPT],
        cwd=str(source_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Probe failed code {proc.returncode}: {proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Malformed probe output: {proc.stdout}") from exc


def run_command_help_probe(
    command_tokens: list[str], *, working_directory: Path, timeout: int = 15
) -> dict[str, Any]:
    """Verify command preflight/help runs from the staged source identity."""
    if not command_tokens:
        return {"checked": False, "verified": True, "message": "No command tokens"}
    try:
        res = subprocess.run(
            [*command_tokens, "--help"],
            cwd=str(working_directory),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        return {
            "checked": True,
            "verified": res.returncode == 0,
            "message": "OK" if res.returncode == 0 else res.stderr[:200],
        }
    except (subprocess.TimeoutExpired, OSError, RuntimeError) as exc:
        return {"checked": True, "verified": False, "message": f"Preflight error: {exc}"}


def _check_git_cleanliness(
    git_state: Mapping[str, Any] | None,
    source_root: Path,
    kwargs: Mapping[str, Any],
    pkgs: list[str],
) -> list[IsolationProblem]:
    if not git_state:
        return []
    problems = []
    for f in git_state.get("untracked_files", []):
        cls, san = sanitize_path(source_root / f, **kwargs)
        if any(f.startswith(p) or f.startswith("src/") for p in pkgs):
            problems.append(IsolationProblem("untracked_source", cls, san, f"Untracked: {san}"))
    for f in git_state.get("modified_files", []):
        cls, san = sanitize_path(source_root / f, **kwargs)
        problems.append(IsolationProblem("dirty_generated_files", cls, san, f"Dirty: {san}"))
    return problems


def _check_sys_path_and_user_site(
    probe: Mapping[str, Any], kwargs: Mapping[str, Any], user_site_dir: Path | None
) -> list[IsolationProblem]:
    problems = []
    sys_paths = [Path(p).resolve() for p in probe.get("sys_path", [])]
    if probe.get("user_site_enabled") and user_site_dir and user_site_dir in sys_paths:
        _, san = sanitize_path(user_site_dir, **kwargs)
        problems.append(
            IsolationProblem("user_site_leakage", "user_site", san, "User site in path")
        )
    for p in sys_paths:
        cls, san = sanitize_path(p, **kwargs)
        if cls == "sibling_worktree":
            problems.append(
                IsolationProblem("sibling_worktree_import", cls, san, f"Sibling: {san}")
            )
    return problems


def _check_pth_injections(
    probe: Mapping[str, Any], kwargs: Mapping[str, Any]
) -> list[IsolationProblem]:
    problems = []
    allowed = {"staged_source", "declared_companion", "virtualenv", "standard_library"}
    for pth in probe.get("pth_files", []):
        _, s_pth = sanitize_path(Path(pth["file"]), **kwargs)
        for line in pth.get("lines", []):
            cls, san = sanitize_path(Path(line).resolve(), **kwargs)
            if cls not in allowed:
                problems.append(
                    IsolationProblem("pth_injection", cls, san, f"{s_pth} injects {san}")
                )
    return problems


def _check_distributions(
    probe: Mapping[str, Any], kwargs: Mapping[str, Any], pkgs: list[str]
) -> list[IsolationProblem]:
    problems = []
    slugs = {pkg.replace("_", "-").lower() for pkg in pkgs}
    for dist in probe.get("distributions", []):
        if dist.get("name", "").lower() in slugs:
            url_info = dist.get("direct_url") or {}
            if isinstance(url_info, Mapping) and url_info.get("dir_info", {}).get("editable"):
                url = url_info.get("url", "")
                target = Path(url[7:] if url.startswith("file://") else url).resolve()
                cls, san = sanitize_path(target, **kwargs)
                if cls != "staged_source":
                    problems.append(
                        IsolationProblem(
                            "stale_editable_installation", cls, san, f"Editable outside: {san}"
                        )
                    )
    return problems


def _check_first_party_imports(
    probe: Mapping[str, Any], kwargs: Mapping[str, Any]
) -> tuple[dict[str, Any], list[IsolationProblem]]:
    problems, sanitized_imports = [], {}
    for pkg, info in probe.get("first_party_imports", {}).items():
        mod_file = info.get("file")
        if not info.get("imported") or not mod_file:
            problems.append(
                IsolationProblem(
                    "first_party_import_failed",
                    "external",
                    f"<module:{pkg}>",
                    f"Import error: {info.get('error')}",
                )
            )
            continue
        p_class, s_file = sanitize_path(Path(mod_file), **kwargs)
        sanitized_imports[pkg] = {
            "path_class": p_class,
            "sanitized_path": s_file,
            "native_extensions": [],
        }
        if p_class not in {"staged_source", "declared_companion"}:
            code = (
                "sibling_worktree_import"
                if p_class == "sibling_worktree"
                else (
                    "user_site_leakage" if p_class == "user_site" else "first_party_import_escaped"
                )
            )
            problems.append(IsolationProblem(code, p_class, s_file, f"{pkg} from {s_file}"))
        for ext in info.get("native_extensions", []):
            ext_class, s_ext = sanitize_path(Path(ext), **kwargs)
            sanitized_imports[pkg]["native_extensions"].append(
                {"path_class": ext_class, "sanitized_path": s_ext}
            )
            if ext_class not in {"staged_source", "declared_companion"}:
                problems.append(
                    IsolationProblem(
                        "unexpected_native_extension_origin", ext_class, s_ext, f"Extension {s_ext}"
                    )
                )
    return sanitized_imports, problems


def evaluate_source_isolation(
    *,
    source_root: Path,
    declared_companions: Mapping[str, Path] | None,
    first_party_packages: list[str],
    command_tokens: list[str] | None,
    probe_data: Mapping[str, Any],
    git_state: Mapping[str, Any] | None,
    command_probe_result: Mapping[str, Any] | None = None,
    venv_root: Path | None = None,
) -> tuple[dict[str, Any], list[IsolationProblem]]:
    """Evaluate source isolation invariants against probe observations."""
    source_root = source_root.resolve()
    companions = {k: v.resolve() for k, v in (declared_companions or {}).items()}
    home_dir = Path.home().resolve()
    site_raw = probe_data.get("user_site_dir")
    user_site_dir = Path(site_raw).resolve() if site_raw else None
    if venv_root is None and sys.prefix:
        venv_root = Path(sys.prefix).resolve()

    kwargs = {
        "source_root": source_root,
        "declared_companions": companions,
        "venv_root": venv_root,
        "user_site_dir": user_site_dir,
        "home_dir": home_dir,
    }
    problems: list[IsolationProblem] = []
    problems.extend(_check_git_cleanliness(git_state, source_root, kwargs, first_party_packages))
    problems.extend(_check_sys_path_and_user_site(probe_data, kwargs, user_site_dir))
    problems.extend(_check_pth_injections(probe_data, kwargs))
    problems.extend(_check_distributions(probe_data, kwargs, first_party_packages))
    sanitized_imports, imp_probs = _check_first_party_imports(probe_data, kwargs)
    problems.extend(imp_probs)

    if (
        command_probe_result
        and command_probe_result.get("checked")
        and not command_probe_result.get("verified")
    ):
        problems.append(
            IsolationProblem(
                "command_startup_failed",
                "staged_source",
                "<command_tokens>",
                str(command_probe_result.get("message")),
            )
        )

    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "passed" if not problems else "blocked",
        "staged_source": {"path_class": "staged_source", "sanitized_path": "<staged_source>"},
        "git_state": git_state,
        "first_party_imports": sanitized_imports,
        "problems": [asdict(p) for p in problems],
        "problem_codes": sorted({p.code for p in problems}),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return receipt, problems


def load_packet(packet_path: Path) -> dict[str, Any]:
    """Load and validate an input packet or staging bundle."""
    try:
        data = json.loads(packet_path.read_text("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Packet root must be a JSON object")
        return data
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Failed to read packet file {packet_path}: {exc}") from exc


def _resolve_cli_inputs(
    args: argparse.Namespace, packet_data: dict[str, Any]
) -> tuple[Path, dict[str, Path], list[str], list[str] | None]:
    if args.source_root:
        source_root = args.source_root.resolve()
    elif "source_root" in packet_data:
        source_root = Path(packet_data["source_root"]).resolve()
    elif (
        "source" in packet_data
        and isinstance(packet_data["source"], dict)
        and "root" in packet_data["source"]
    ):
        source_root = Path(packet_data["source"]["root"]).resolve()
    else:
        source_root = Path.cwd().resolve()

    companions: dict[str, Path] = {}
    for comp in args.companions:
        if "=" in comp:
            k, v = comp.split("=", 1)
            companions[k.strip()] = Path(v.strip()).resolve()
    if isinstance(packet_data.get("declared_companions"), dict):
        for k, v in packet_data["declared_companions"].items():
            companions[k] = Path(v).resolve()

    first_party = args.first_party or packet_data.get("first_party_packages") or DEFAULT_FIRST_PARTY
    return source_root, companions, first_party, packet_data.get("command_tokens")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for staged source isolation verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", default=True)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--companion", action="append", dest="companions", default=[])
    parser.add_argument("--first-party", action="append", dest="first_party", default=[])
    parser.add_argument("--format", choices=["json", "text"], default="json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    packet_data: dict[str, Any] = {}
    if args.packet:
        try:
            packet_data = load_packet(args.packet)
        except ValueError as exc:
            sys.stderr.write(f"Error loading packet: {exc}\n")
            return 2

    source_root, companions, first_party, command_tokens = _resolve_cli_inputs(args, packet_data)
    if not source_root.is_dir():
        sys.stderr.write(f"Source root does not exist: {source_root}\n")
        return 2

    try:
        probe_data = run_environment_probe(
            source_root=source_root, first_party_packages=first_party
        )
    except RuntimeError as exc:
        sys.stderr.write(f"Probe execution error: {exc}\n")
        return 2

    command_probe = (
        run_command_help_probe(command_tokens, working_directory=source_root)
        if command_tokens
        else None
    )
    receipt, problems = evaluate_source_isolation(
        source_root=source_root,
        declared_companions=companions,
        first_party_packages=first_party,
        command_tokens=command_tokens,
        probe_data=probe_data,
        git_state=inspect_git_state(source_root),
        command_probe_result=command_probe,
    )
    if args.format == "json":
        out_text = json.dumps(receipt, indent=2, sort_keys=True)
    else:
        lines = [f"Status: {receipt['status'].upper()}", f"Problems: {receipt['problem_codes']}"]
        lines.extend(f"  - [{p.code}] {p.sanitized_path}: {p.message}" for p in problems)
        out_text = "\n".join(lines)

    print(out_text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out_text, "utf-8")
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
