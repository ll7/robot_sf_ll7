"""Fail-closed preflight for proxy-based checkpoint selection inputs (issue #3204).

The proxy-vs-ADE checkpoint-selection analyzer
(``scripts/research/analyze_predictive_checkpoint_proxy.py``, merged via #3307) can only
produce a conclusive result when its *inputs* are present:

1. a held-out hard-seed fixture to score proxy success on;
2. ">= 6 checkpoints from >= 1 real training run" whose registry ``local_path`` actually
   resolve on this machine; and
3. a training summary whose ``proxy.history`` records non-degenerate spread in hard-seed
   ``success_rate`` across enough proxy epochs.

Issue #3204 is currently *blocked* precisely because (2) and (3) are not satisfied: every
``predictive_*`` registry ``local_path`` points at a non-durable ``output/tmp/...`` path that
does not exist, and the one real probe run records ``success_rate = 0.0`` at every epoch (no
spread -> the analyzer returns ``inconclusive``). This tool turns that manual diagnostic into a
reusable, read-only preflight that fails closed (``status: blocked``) when the inputs are
absent or degenerate.

It selects no checkpoint, runs no training, submits no jobs, downloads nothing, and asserts no
benchmark result. It only maps declared inputs to their on-disk readiness and reports
``ready``/``blocked`` with exit code ``0``/``2``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_BLOCKED = "blocked"

DEFAULT_CONFIG = Path("configs/research/predictive_checkpoint_proxy_v1.yaml")
DEFAULT_REGISTRY = Path("model/registry.yaml")


def _load_analyzer() -> Any:
    """Load the merged proxy-vs-ADE analyzer module by path (no package install needed)."""
    tool = _REPO_ROOT / "scripts" / "research" / "analyze_predictive_checkpoint_proxy.py"
    spec = importlib.util.spec_from_file_location("analyze_predictive_checkpoint_proxy", tool)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Could not load analyzer module from {tool}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_yaml(path: Path) -> Any:
    """Load a YAML file, returning None on read/parse failure."""
    try:
        with path.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return None


def _check_config(config: dict | None, config_path: Path) -> tuple[str, list[str]]:
    """Verify the readiness contract exists and declares the required keys."""
    if config is None:
        return STATUS_BLOCKED, [f"readiness config not found or unreadable: {config_path}"]
    if not isinstance(config, dict):
        return STATUS_FAILED, ["readiness config must be a mapping"]
    errors: list[str] = []
    hard_seed = config.get("hard_seed_fixture")
    if not isinstance(hard_seed, str) or not hard_seed:
        errors.append("readiness config missing or invalid hard_seed_fixture")
    selector = config.get("checkpoint_selector")
    if not isinstance(selector, dict):
        errors.append("readiness config missing checkpoint_selector mapping")
    else:
        if not selector.get("registry_tag"):
            errors.append("checkpoint_selector missing registry_tag")
        if not isinstance(selector.get("min_resolvable_checkpoints"), int):
            errors.append("checkpoint_selector missing integer min_resolvable_checkpoints")
    summary_contract = config.get("proxy_summary_contract")
    if summary_contract is not None and not isinstance(summary_contract, dict):
        errors.append("proxy_summary_contract must be a mapping when provided")
    if errors:
        return STATUS_FAILED, errors
    return STATUS_PASSED, []


def _check_hard_seed_fixture(fixture_path: Path) -> tuple[str, list[str]]:
    """Verify the hard-seed fixture exists and parses to a non-empty mapping/list."""
    if not fixture_path.is_file():
        return STATUS_BLOCKED, [f"hard-seed fixture not found or is not a file: {fixture_path}"]
    data = _load_yaml(fixture_path)
    if not data:
        return STATUS_FAILED, [f"hard-seed fixture empty or unreadable: {fixture_path}"]
    return STATUS_PASSED, []


def _resolve_local_path(local_path: Any, repo_root: Path) -> Path | None:
    """Resolve a registry ``local_path`` against ``repo_root`` without downloading."""
    if not isinstance(local_path, str) or not local_path:
        return None
    candidate = Path(local_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate


def _check_checkpoint_artifacts(
    registry: dict[str, dict[str, Any]],
    *,
    registry_tag: str,
    min_resolvable: int,
    repo_root: Path,
) -> tuple[str, list[str], dict[str, Any]]:
    """Map tagged checkpoint candidates to on-disk presence and gate on the minimum count.

    Returns:
        tuple[str, list[str], dict[str, Any]]: status, messages, and a mapping payload
        recording, per candidate, its ``local_path`` and resolved presence.
    """
    tag = registry_tag.strip().lower()
    candidates: list[dict[str, Any]] = []
    for model_id, entry in registry.items():
        if not isinstance(entry, dict):
            continue
        tags = {str(t).strip().lower() for t in (entry.get("tags") or [])}
        if tag not in tags:
            continue
        local_path = entry.get("local_path")
        resolved = _resolve_local_path(local_path, repo_root)
        present = bool(resolved and resolved.exists())
        candidates.append(
            {
                "model_id": model_id,
                "local_path": local_path,
                "present": present,
            }
        )

    candidates.sort(key=lambda item: str(item["model_id"]))
    resolvable = [c for c in candidates if c["present"]]
    mapping = {
        "registry_tag": registry_tag,
        "min_resolvable_checkpoints": min_resolvable,
        "candidate_count": len(candidates),
        "resolvable_count": len(resolvable),
        "candidates": candidates,
    }

    if not candidates:
        return (
            STATUS_BLOCKED,
            [f"no registry entries carry tag '{registry_tag}'"],
            mapping,
        )
    if len(resolvable) < min_resolvable:
        absent = [c["model_id"] for c in candidates if not c["present"]]
        return (
            STATUS_BLOCKED,
            [
                f"only {len(resolvable)} of {len(candidates)} '{registry_tag}' checkpoints resolve "
                f"locally; need >= {min_resolvable}. Absent local_path for: {', '.join(absent)}"
            ],
            mapping,
        )
    return STATUS_PASSED, [], mapping


def _check_proxy_summary(
    summary_path: Path,
    *,
    analyzer: Any,
    require_enabled: bool,
    min_proxy_epochs: int,
) -> tuple[str, list[str], dict[str, Any]]:
    """Judge whether a training summary would yield a conclusive proxy-vs-ADE comparison.

    Reuses the merged analyzer so the blocked/ready boundary stays identical to the tool that
    consumes the summary. An ``inconclusive`` verdict (e.g. the all-zero probe with no
    hard-success spread) fails closed.

    Returns:
        tuple[str, list[str], dict[str, Any]]: status, messages, and a compact summary payload.
    """
    if not summary_path.is_file():
        return STATUS_BLOCKED, [f"training summary not found or is not a file: {summary_path}"], {}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return STATUS_FAILED, [f"training summary unreadable: {exc}"], {}
    if not isinstance(summary, dict):
        return STATUS_FAILED, ["training summary must be a JSON object/mapping"], {}

    report = analyzer.analyze_summary(summary)
    verdict = report.get("verdict")
    n_epochs = int(report.get("n_proxy_epochs") or 0)
    payload = {
        "verdict": verdict,
        "proxy_enabled": report.get("proxy_enabled"),
        "n_proxy_epochs": n_epochs,
        "success_spread": report.get("success_spread"),
    }

    errors: list[str] = []
    if require_enabled and not report.get("proxy_enabled"):
        errors.append("training summary has proxy.enabled != true")
    if verdict == "inconclusive":
        errors.append(
            f"analyzer verdict is inconclusive ({report.get('reason')}); "
            "proxy and ADE selection are indistinguishable"
        )
    if n_epochs < min_proxy_epochs:
        errors.append(
            f"only {n_epochs} usable proxy epochs; claim contract needs >= {min_proxy_epochs}"
        )

    if errors:
        return STATUS_BLOCKED, errors, payload
    return STATUS_PASSED, [], payload


def check_readiness(
    *,
    config_path: Path,
    registry_path: Path,
    repo_root: Path,
    training_summary: Path | None = None,
) -> dict[str, Any]:
    """Run all proxy-checkpoint-selection readiness checks and return a structured report."""
    config = _load_yaml(config_path)
    config_status, config_messages = _check_config(
        config if isinstance(config, dict) else None, config_path
    )

    prerequisites: dict[str, dict[str, Any]] = {
        "readiness_config": {"status": config_status, "messages": config_messages}
    }

    # Only run input checks the config is well-formed enough to parameterize.
    cfg = config if isinstance(config, dict) and config_status != STATUS_BLOCKED else {}

    fixture_rel = cfg.get("hard_seed_fixture") if cfg else None
    if isinstance(fixture_rel, str) and fixture_rel:
        fixture_path = Path(fixture_rel)
        if not fixture_path.is_absolute():
            fixture_path = repo_root / fixture_path
        fixture_status, fixture_messages = _check_hard_seed_fixture(fixture_path)
    else:
        fixture_status, fixture_messages = STATUS_BLOCKED, ["hard_seed_fixture not declared"]
    prerequisites["hard_seed_fixture"] = {
        "status": fixture_status,
        "messages": fixture_messages,
    }

    selector = cfg.get("checkpoint_selector") if cfg else None
    if isinstance(selector, dict) and registry_path.exists():
        registry_module = _load_registry_module()
        registry = registry_module.load_registry(registry_path)
        ckpt_status, ckpt_messages, ckpt_mapping = _check_checkpoint_artifacts(
            registry,
            registry_tag=str(selector.get("registry_tag", "")),
            min_resolvable=int(selector.get("min_resolvable_checkpoints", 0) or 0),
            repo_root=repo_root,
        )
    elif not registry_path.exists():
        ckpt_status, ckpt_messages, ckpt_mapping = (
            STATUS_BLOCKED,
            [f"model registry not found: {registry_path}"],
            {},
        )
    else:
        ckpt_status, ckpt_messages, ckpt_mapping = (
            STATUS_BLOCKED,
            ["checkpoint_selector not usable from config"],
            {},
        )
    prerequisites["checkpoint_artifacts"] = {
        "status": ckpt_status,
        "messages": ckpt_messages,
        "mapping": ckpt_mapping,
    }

    # The training-summary check is optional: it only runs when a candidate summary is supplied,
    # because the conclusive proxy run is itself run-gated (resource:slurm).
    if training_summary is not None:
        summary_contract = (cfg.get("proxy_summary_contract") or {}) if cfg else {}
        analyzer = _load_analyzer()
        summary_status, summary_messages, summary_payload = _check_proxy_summary(
            training_summary,
            analyzer=analyzer,
            require_enabled=bool(summary_contract.get("require_enabled", True)),
            min_proxy_epochs=int(summary_contract.get("min_proxy_epochs", 6) or 6),
        )
        prerequisites["proxy_training_summary"] = {
            "status": summary_status,
            "messages": summary_messages,
            "summary": summary_payload,
        }

    errors: list[str] = []
    for payload in prerequisites.values():
        if payload["status"] != STATUS_PASSED:
            errors.extend(payload["messages"])

    status = "ready" if not errors else "blocked"
    return {
        "schema_version": "predictive-checkpoint-proxy-readiness-report.v1",
        "status": status,
        "errors": errors,
        "checked": {
            "readiness_config": str(config_path),
            "registry": str(registry_path),
            "training_summary": str(training_summary) if training_summary else None,
        },
        "prerequisites": prerequisites,
        "claim_boundary": (
            "Diagnostic readiness/preflight only. Reports whether proxy-based checkpoint "
            "selection inputs resolve locally; selects no checkpoint and asserts no benchmark "
            "result. A 'ready' status means inputs are present, not that a proxy beats ADE."
        ),
    }


def _load_registry_module() -> Any:
    """Import the canonical registry loader, adding the repo root to ``sys.path`` if needed."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from robot_sf.models import registry as registry_module

    return registry_module


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the proxy-checkpoint readiness contract YAML.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="Path to the model registry YAML.",
    )
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=None,
        help="Optional training summary JSON to gate proxy.history spread/epochs.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help="Repository root for resolving relative paths.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON readiness report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the readiness preflight and return a shell-friendly exit code (0 ready, 2 blocked)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()

    def _resolve(path: Path | None) -> Path | None:
        if path is None:
            return None
        return path if path.is_absolute() else repo_root / path

    report = check_readiness(
        config_path=_resolve(args.config) or args.config,
        registry_path=_resolve(args.registry) or args.registry,
        repo_root=repo_root,
        training_summary=_resolve(args.training_summary),
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        label = "READY" if report["status"] == "ready" else "BLOCKED"
        print(f"predictive-checkpoint-proxy readiness: {label}")
        for key, payload in report["prerequisites"].items():
            print(f" - {key}: {payload['status']}")
            for message in payload["messages"]:
                print(f"   * {message}")

    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
