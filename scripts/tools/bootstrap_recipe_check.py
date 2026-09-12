#!/usr/bin/env python3
"""Validate frozen bootstrap recipes per execution class (#8853).

Recipes freeze the ordered setup, probe, and cleanup sequence that activated one
execution environment, with source/lock identity, immutable container and module
identities, private substitutions, and verification status. The checker is
report-only; ``--execute-safe-checks`` runs only ``safe_check`` probes in an
isolated temporary root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

RECIPE_SCHEMA = "bootstrap_recipe.v1"
OVERLAY_SCHEMA = "bootstrap_recipe_private_overlay.v1"
REPORT_SCHEMA = "bootstrap_recipe_report.v1"

EXECUTION_CLASSES = ("cpu_batch", "gpu_training", "carla_platform", "local_analysis")
PHASES = ("setup", "probe", "cleanup")
PHASE_INDEX = {phase: index for index, phase in enumerate(PHASES)}
PUBLIC_PLACEHOLDERS = frozenset({"RECIPE_ROOT", "PROJECT_ROOT"})

CREDENTIAL_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|AWS_SECRET_ACCESS_KEY|bearer\s+[A-Za-z0-9_\-\.]+"
    r"|['\"]?(?:password|passwd|api_key|secret_key)['\"]?\s*[:=]\s*['\"][^'\"]{8,}['\"]",
    re.IGNORECASE,
)
PRIVATE_PATH_RE = re.compile(
    r"(?:^|[\s\"'=])(/(?:private|root/\.ssh|etc/(?:shadow|passwd)|var/run/secrets|opt/secrets)/)"
    r"|(?:[a-zA-Z]:[/\\]secrets)",
    re.IGNORECASE,
)
STALE_PATH_RE = re.compile(r"^/(?:home|tmp|var/tmp|scratch|work)/", re.IGNORECASE)
SOURCE_HOST_ACCESS_RE = re.compile(
    r"\b(?:ssh|scp|rsync|sftp)://|\bgit@[a-zA-Z0-9_\-\.]+:", re.IGNORECASE
)
HIDDEN_ENV_RE = re.compile(
    r"\bPYTHONPATH\b|\b--user\b|site-packages|sys\.path\.(?:insert|append)|\.pth\b", re.IGNORECASE
)
SHELL_METACHAR_RE = re.compile(r"&&|\|\||;|\||>|<|`|\$\(")
PLACEHOLDER_RE = re.compile(r"\$\{?([A-Z][A-Z0-9_]*)\}?")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
DESTRUCTIVE_PROGRAMS = frozenset({"dd", "mkfs", "shred", "wipefs", "fdisk", "parted"})
SENSITIVE_ROOTS = frozenset({"/", "/home", "/root", "/etc", "/usr", "/var", "/boot", "/opt"})
BLOCKING_REASONS = frozenset(
    "credential_leak private_path_leak source_host_access_attempt stale_source_path "
    "unresolved_placeholder mutable_container_alias unpinned_module_alias "
    "destructive_cleanup unsafe_safe_check shell_string_step".split()
)


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        return None, str(exc)
    return (data, None) if isinstance(data, dict) else (None, "content is not a JSON object")


def _strings(*values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        if isinstance(value, (str, int, float)):
            out.append(str(value))
        elif isinstance(value, dict):
            for key, item in value.items():
                out.append(str(key))
                out.extend(_strings(item))
        elif isinstance(value, list):
            out.extend(_strings(*value))
    return out


def _scan_public_safety(recipe: dict[str, Any], disc: list[str], reasons: list[str]) -> None:
    scans = (
        (CREDENTIAL_RE, "credential_leak"),
        (PRIVATE_PATH_RE, "private_path_leak"),
        (SOURCE_HOST_ACCESS_RE, "source_host_access_attempt"),
    )
    for text in _strings(recipe):
        for pattern, tag in scans:
            if pattern.search(text) and tag not in reasons:
                disc.append(f"{tag}: {text[:80]}")
                reasons.append(tag)
    for step in recipe.get("steps", []):
        if not isinstance(step, dict):
            continue
        tokens = _strings(step.get("argv", []), step.get("workdir", ""))
        for token in tokens:
            if STALE_PATH_RE.search(token) and "stale_source_path" not in reasons:
                disc.append(f"stale_source_path: {token[:80]}")
                reasons.append("stale_source_path")


def _check_identity_bindings(recipe: dict[str, Any], disc: list[str], reasons: list[str]) -> None:
    def flag(reason: str, name: Any) -> None:
        if reason not in reasons:
            disc.append(f"{reason}: {name}")
            reasons.append(reason)

    for binding in recipe.get("identity_bindings", []):
        if not isinstance(binding, dict):
            continue
        kind = binding.get("kind")
        identity = str(binding.get("identity", ""))
        if kind == "container" and not DIGEST_RE.fullmatch(str(binding.get("digest", ""))):
            flag("mutable_container_alias", binding.get("name"))
        elif kind == "module" and (
            not identity
            or identity.lower() in {"latest", "default", "unversioned"}
            or not re.search(r"\d", identity)
        ):
            flag("unpinned_module_alias", binding.get("name"))
        elif kind == "python-package" and not binding.get("version"):
            flag("unpinned_package", binding.get("name"))


def _check_step_shape(
    step: dict[str, Any], disc: list[str], reasons: list[str]
) -> list[str] | None:
    argv = step.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        disc.append(f"malformed_argv: {step.get('id', '?')}")
        reasons.append("malformed_step")
        return None
    is_shell = len(argv) == 1 and bool(SHELL_METACHAR_RE.search(argv[0]))
    if is_shell or (argv[0] in {"bash", "sh", "zsh"} and "-c" in argv):
        disc.append(f"shell_string_step: {step.get('id', '?')}")
        reasons.append("shell_string_step")
    return argv


def _check_step_safety(
    step: dict[str, Any], phase: str, argv: list[str], disc: list[str], reasons: list[str]
) -> None:
    if step.get("safe_check") and (phase != "probe" or step.get("expect_exit_code", 0) != 0):
        if "unsafe_safe_check" not in reasons:
            disc.append(f"unsafe_safe_check: {step.get('id', '?')} (phase {phase})")
            reasons.append("unsafe_safe_check")
    text = " ".join(argv) + " " + " ".join(_strings(step.get("env", {})))
    if HIDDEN_ENV_RE.search(text) and "hidden_environment_dependence" not in reasons:
        disc.append(f"hidden_environment_dependence: {step.get('id', '?')}")
        reasons.append("hidden_environment_dependence")
    for token in argv:
        destructive = token in SENSITIVE_ROOTS or Path(token).name in DESTRUCTIVE_PROGRAMS
        if destructive and "destructive_cleanup" not in reasons:
            disc.append(f"destructive_cleanup: {step.get('id', '?')} ({token})")
            reasons.append("destructive_cleanup")


def _check_steps(recipe: dict[str, Any], disc: list[str], reasons: list[str]) -> None:
    raw = recipe.get("steps", [])
    if not isinstance(raw, list):
        disc.append("malformed_steps")
        reasons.append("malformed_step")
        return
    steps = [step for step in raw if isinstance(step, dict)]
    if len(steps) != len(raw):
        disc.append("malformed_step_entry")
        reasons.append("malformed_step")
    order: list[int] = []
    for step in steps:
        argv = _check_step_shape(step, disc, reasons)
        phase = step.get("phase")
        if argv is None:
            continue
        if phase not in PHASE_INDEX:
            disc.append(f"unknown_step_phase: {step.get('id', '?')}")
            reasons.append("malformed_step")
            continue
        order.append(PHASE_INDEX[phase])
        _check_step_safety(step, phase, argv, disc, reasons)
    if order != sorted(order):
        disc.append("steps_not_ordered: setup -> probe -> cleanup required")
        reasons.append("wrong_step_order")
    if recipe.get("verification_status") == "verified":
        phases = {step.get("phase") for step in steps}
        for phase in ("setup", "probe"):
            if phase not in phases:
                disc.append(f"missing_{phase}_step")
                reasons.append("missing_step_contract")


def _check_recipe_header(recipe: dict[str, Any], disc: list[str], reasons: list[str]) -> None:
    if recipe.get("schema") != RECIPE_SCHEMA:
        disc.append(f"unsupported_schema: {recipe.get('schema')}")
        reasons.append("unsupported_schema")
    if recipe.get("execution_class") not in EXECUTION_CLASSES:
        disc.append(f"unknown_execution_class: {recipe.get('execution_class')}")
        reasons.append("unknown_execution_class")
    for field in ("recipe_id", "title", "source_identity"):
        if not recipe.get(field):
            disc.append(f"missing_field: {field}")
            reasons.append("missing_field")


def _check_source_identity(
    recipe: dict[str, Any], project_root: Path | None, disc: list[str], reasons: list[str]
) -> None:
    source = recipe.get("source_identity", {})
    if not isinstance(source, dict):
        return
    commit = str(source.get("recorded_from_commit", ""))
    if commit and not COMMIT_RE.fullmatch(commit):
        disc.append(f"invalid_source_commit: {commit}")
        reasons.append("invalid_source_commit")
    lockfile = str(source.get("lockfile", ""))
    checksum = str(source.get("lockfile_sha256", ""))
    lock = project_root / lockfile if project_root is not None and lockfile else None
    if lock is not None and lock.is_file() and checksum:
        if hashlib.sha256(lock.read_bytes()).hexdigest() != checksum:
            disc.append(f"lockfile_drift: {lockfile}")
            reasons.append("lockfile_drift")


def _check_verification_status(recipe: dict[str, Any], disc: list[str], reasons: list[str]) -> None:
    status = recipe.get("verification_status")
    if status not in {"verified", "unavailable"}:
        disc.append(f"invalid_verification_status: {status}")
        reasons.append("invalid_verification_status")
    elif status == "verified":
        verification = recipe.get("verification")
        if not isinstance(verification, dict) or not verification.get("mode"):
            disc.append("verified_recipe_missing_verification_record")
            reasons.append("missing_verification_record")
    elif not recipe.get("unavailable_reason"):
        disc.append("unavailable_recipe_missing_reason")
        reasons.append("missing_unavailable_reason")


def _declared_placeholders(
    recipe: dict[str, Any], disc: list[str], reasons: list[str]
) -> tuple[set[str], list[dict[str, Any]]]:
    declared = set(PUBLIC_PLACEHOLDERS)
    substitutions = recipe.get("private_substitutions", [])
    if not isinstance(substitutions, list):
        disc.append("malformed_private_substitutions")
        reasons.append("malformed_step")
        return declared, []
    valid = [
        entry
        for entry in substitutions
        if isinstance(entry, dict) and entry.get("placeholder") and entry.get("capability_class")
    ]
    if len(valid) != len(substitutions):
        disc.append("malformed_private_substitution_entry")
        reasons.append("malformed_step")
    declared.update(str(entry["placeholder"]).lstrip("$") for entry in valid)
    return declared, valid


def _recipe_tokens(recipe: dict[str, Any]) -> list[str]:
    steps = [step for step in recipe.get("steps", []) if isinstance(step, dict)]
    return _strings(
        [step.get("argv", []) for step in steps],
        [step.get("workdir", "") for step in steps],
        [step.get("env", {}) for step in steps],
        recipe.get("outputs", []),
    )


def _check_recipe(recipe: dict[str, Any], project_root: Path | None) -> dict[str, Any]:
    disc: list[str] = []
    reasons: list[str] = []
    _check_recipe_header(recipe, disc, reasons)
    _check_source_identity(recipe, project_root, disc, reasons)
    _check_verification_status(recipe, disc, reasons)
    declared, substitutions = _declared_placeholders(recipe, disc, reasons)
    _scan_public_safety(recipe, disc, reasons)
    _check_identity_bindings(recipe, disc, reasons)
    _check_steps(recipe, disc, reasons)
    found = {name for token in _recipe_tokens(recipe) for name in PLACEHOLDER_RE.findall(token)}
    for name in sorted(found - declared):
        disc.append(f"unresolved_placeholder: ${name}")
        reasons.append("unresolved_placeholder")
    status = recipe.get("verification_status")
    if any(reason in BLOCKING_REASONS for reason in reasons):
        entry_status = "blocked"
    elif reasons:
        entry_status = "invalid"
    elif status == "unavailable":
        entry_status = "unavailable"
    else:
        entry_status = "verified"
    return {
        "recipe_id": str(recipe.get("recipe_id", "unknown")),
        "execution_class": recipe.get("execution_class"),
        "verification_status": status,
        "status": entry_status,
        "reasons": sorted(set(reasons)),
        "discrepancies": disc,
        "private_substitutions": [entry["placeholder"] for entry in substitutions],
    }


def _substitution_values(
    recipe: dict[str, Any], temp_root: Path, project_root: Path
) -> dict[str, str]:
    values = {"RECIPE_ROOT": str(temp_root), "PROJECT_ROOT": str(project_root)}
    for entry in recipe.get("private_substitutions", []):
        if isinstance(entry, dict):
            value = entry.get("value") or entry.get("default") or ""
            values[str(entry.get("placeholder", "")).lstrip("$")] = value
    return values


def _run_safe_checks(
    recipe: dict[str, Any], temp_root: Path, project_root: Path, timeout: int
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    values = _substitution_values(recipe, temp_root, project_root)
    for step in recipe.get("steps", []):
        if not (isinstance(step, dict) and step.get("phase") == "probe" and step.get("safe_check")):
            continue
        tokens = _strings(step.get("argv", []))
        unresolved = [
            n for token in tokens for n in PLACEHOLDER_RE.findall(token) if not values.get(n)
        ]
        if unresolved:
            results.append({"step_id": step.get("id"), "status": "skipped_private_substitution"})
            continue
        for name, value in values.items():
            tokens = [t.replace(f"${{{name}}}", value).replace(f"${name}", value) for t in tokens]
        if not shutil.which(tokens[0]):
            results.append({"step_id": step.get("id"), "status": "skipped_missing_program"})
            continue
        env = os.environ.copy()
        env.update(values)
        env.update(step.get("env", {}))
        cwd = Path(step.get("workdir", "$RECIPE_ROOT").replace("$RECIPE_ROOT", str(temp_root)))
        cwd.mkdir(parents=True, exist_ok=True)
        expected = step.get("expect_exit_code", 0)
        try:
            completed = subprocess.run(
                tokens, cwd=cwd, env=env, capture_output=True, timeout=timeout, check=False
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            results.append({"step_id": step.get("id"), "status": "error", "error": str(exc)[:120]})
            continue
        results.append(
            {
                "step_id": step.get("id"),
                "status": "passed" if completed.returncode == expected else "failed",
                "exit_code": completed.returncode,
                "expected_exit_code": expected,
            }
        )
    return results


def _load_recipes(path: Path) -> tuple[list[tuple[Path, dict[str, Any]]], list[str]]:
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    recipes: list[tuple[Path, dict[str, Any]]] = []
    errors: list[str] = []
    for file in files:
        data, err = _read_json(file)
        if err or data is None:
            errors.append(f"{file.name}: {err}")
        elif data.get("schema") != OVERLAY_SCHEMA:
            recipes.append((file, data))
    return recipes, errors


def _check_overlay(overlay_path: Path) -> dict[str, Any]:
    overlay, err = _read_json(overlay_path)
    if err or overlay is None or overlay.get("schema") != OVERLAY_SCHEMA:
        return {
            "path": str(overlay_path),
            "ok": False,
            "error": err or "unsupported overlay schema",
        }
    placeholders = overlay.get("placeholders", {})
    missing = [
        key
        for key, value in placeholders.items()
        if not isinstance(value, dict) or not value.get("capability_class")
    ]
    return {
        "path": str(overlay_path),
        "ok": not missing,
        "placeholder_count": len(placeholders),
        "missing_capability_class": sorted(missing),
    }


def check_recipes(
    recipes_path: Path,
    overlay_path: Path | None = None,
    project_root: Path | None = None,
    execute_safe_checks: bool = False,
    require_verified: bool = False,
    safe_check_timeout: int = 30,
) -> dict[str, Any]:
    """Validate bootstrap recipes and return a deterministic report dictionary."""
    recipes, load_errors = _load_recipes(recipes_path)
    entries: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="bootstrap_recipe_") as tmp:
        for file, recipe in recipes:
            entry = _check_recipe(recipe, project_root)
            entry["path"] = file.name
            if execute_safe_checks and entry["status"] in {"verified", "unavailable"}:
                entry["safe_checks"] = _run_safe_checks(
                    recipe, Path(tmp), project_root or Path.cwd(), safe_check_timeout
                )
                if any(c.get("status") in {"failed", "error"} for c in entry["safe_checks"]):
                    entry["status"] = "invalid"
                    entry["reasons"] = sorted({*entry["reasons"], "safe_check_failed"})
            entries.append(entry)
    entries.sort(key=lambda item: (str(item.get("recipe_id")), str(item.get("path"))))
    class_status: dict[str, str] = {}
    for entry in entries:
        cls = entry.get("execution_class")
        if isinstance(cls, str) and cls and class_status.get(cls) != "blocked":
            class_status[cls] = entry["status"]
    missing = sorted(cls for cls in EXECUTION_CLASSES if class_status.get(cls) != "verified")
    blocked = any(entry["status"] == "blocked" for entry in entries) or bool(load_errors)
    invalid = any(entry["status"] == "invalid" for entry in entries)
    verdict = "blocked" if blocked else ("fail" if invalid else "pass")
    if verdict == "pass" and require_verified and missing:
        verdict = "fail"
    return {
        "schema": REPORT_SCHEMA,
        "verdict": verdict,
        "mode": "structural+safe_checks" if execute_safe_checks else "structural",
        "recipe_count": len(entries),
        "require_verified": require_verified,
        "execution_classes": class_status,
        "missing_verified_classes": missing if require_verified else [],
        "recipes": entries,
        "load_errors": load_errors,
        "private_overlay": _check_overlay(overlay_path) if overlay_path else None,
        "host_mutation": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a deterministic Markdown summary of a recipe check report."""
    classes = ", ".join(
        f"{cls}={report['execution_classes'].get(cls, 'missing')}" for cls in EXECUTION_CLASSES
    )
    lines = [
        "# Bootstrap recipe check",
        "",
        f"Verdict: {report['verdict'].upper()}",
        f"Mode: {report['mode']}",
        "",
        "| Recipe | Class | Status | Reasons |",
        "| --- | --- | --- | --- |",
        *(
            f"| {entry['recipe_id']} | {entry['execution_class']} | {entry['status']} "
            f"| {', '.join(entry['reasons']) or '-'} |"
            for entry in report["recipes"]
        ),
        "",
        f"Execution classes: {classes}",
    ]
    if report["missing_verified_classes"]:
        lines.append("Missing verified classes: " + ", ".join(report["missing_verified_classes"]))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns 0 for pass, 1 for fail, 2 for blocked."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Run the read-only structural check.")
    parser.add_argument("--recipes", required=True, type=Path, help="Recipe file or directory.")
    parser.add_argument("--overlay", type=Path, default=None, help="Private overlay example path.")
    parser.add_argument(
        "--project-root", type=Path, default=None, help="Checkout for lock identity."
    )
    parser.add_argument("--format", choices=["text", "json", "markdown"], default="text")
    parser.add_argument("--safe-check-timeout", type=int, default=30)
    parser.add_argument(
        "--execute-safe-checks", action="store_true", help="Run safe_check probes in a temp root."
    )
    parser.add_argument(
        "--require-verified", action="store_true", help="Fail without a verified class recipe."
    )
    args = parser.parse_args(argv)
    report = check_recipes(
        args.recipes,
        args.overlay,
        args.project_root,
        args.execute_safe_checks,
        args.require_verified,
        args.safe_check_timeout,
    )
    if args.format == "json":
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    elif args.format == "markdown":
        sys.stdout.write(render_markdown(report))
    else:
        sys.stdout.write(f"Verdict: {report['verdict'].upper()} [{report['mode']}]\n")
        for entry in report["recipes"]:
            detail = ", ".join(entry["reasons"]) or "none"
            sys.stdout.write(f"- {entry['recipe_id']}: {entry['status']} ({detail})\n")
    return {"pass": 0, "fail": 1}.get(report["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
