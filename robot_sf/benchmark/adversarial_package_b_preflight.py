"""Fail-closed preflight for the issue #3079 package-B comparison manifest.

The checker validates only local readiness metadata for the budget-matched
adversarial falsification package-B plan. It never runs sampler comparisons,
submits jobs, or interprets falsification yield.
"""

from __future__ import annotations

import ast
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "adversarial-package-b-preflight.v1"
MANIFEST_SCHEMA_VERSION = "adversarial-package-b-comparison.v1"
EXPECTED_ISSUE = 3079
EXPECTED_BUDGETS = (16, 32, 64)
EXPECTED_SAMPLERS = ("random", "coordinate", "optuna")
EXPECTED_POLICY = "goal"
EXPECTED_OBJECTIVE = "worst_case_snqi"
EXPECTED_REPORTING_FIELDS = frozenset(
    {
        "first_failure_iteration",
        "best_valid_objective",
        "invalid_candidate_rate",
        "replay_success_rate",
        "certified_valid_failure_count",
        "replayable_valid_failure_count",
        "fallback_candidate_count",
        "degraded_candidate_count",
        "held_out_family_yield",
    }
)
EXPECTED_OUTPUT_PREFIX = Path("output/adversarial/issue_3079_package_b")
RESEARCH_PACKAGE_REGISTRY = Path("configs/research/research_package_registry_issue_3057.yaml")
FORBIDDEN_COMMAND_TOKENS = frozenset({"sbatch", "srun", "slurm", "wandb", "paper", "dissertation"})


@dataclass(frozen=True)
class PackageBPreflightResult:
    """Structured package-B readiness result."""

    manifest_path: str
    ready: bool
    blocked: bool
    checks: dict[str, bool]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    metadata: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        """Return a stable JSON payload for CLI and tests."""
        return {
            "schema_version": SCHEMA_VERSION,
            "manifest_path": self.manifest_path,
            "ready": self.ready,
            "blocked": self.blocked,
            "checks": self.checks,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _as_int_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    values: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return ()
        values.append(item)
    return tuple(values)


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    values: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return ()
        values.append(item.strip())
    return tuple(values)


def _resolve_repo_path(repo_root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def _extract_output_paths(
    command: str, repo_root: Path
) -> tuple[Path | None, Path | None, list[str]]:
    warnings: list[str] = []
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        warnings.append(f"example_command could not be parsed with shlex: {exc}")
        return None, None, warnings

    output_dir: Path | None = None
    out_json: Path | None = None
    for index, token in enumerate(tokens):
        if token == "--output-dir" and index + 1 < len(tokens):
            output_dir = _resolve_repo_path(repo_root, tokens[index + 1])
        if token == "--out-json" and index + 1 < len(tokens):
            out_json = _resolve_repo_path(repo_root, tokens[index + 1])
    return output_dir, out_json, warnings


def _extract_repeated_seed_args(command: str) -> tuple[tuple[int, ...], list[str]]:
    warnings: list[str] = []
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        warnings.append(f"example_command could not shlex: {exc}")
        return (), warnings

    seed_args: list[int] = []
    for index, token in enumerate(tokens):
        if token == "--seed":
            if index + 1 >= len(tokens):
                warnings.append("example_command has --seed without value")
                continue
            seed_value = tokens[index + 1]
        elif token.startswith("--seed="):
            seed_value = token.removeprefix("--seed=")
        else:
            continue

        try:
            seed_args.append(int(seed_value))
        except ValueError:
            warnings.append(f"example_command has non-integer --seed value: {seed_value!r}")
    return tuple(seed_args), warnings


def _extract_option_values(command: str, option: str) -> tuple[tuple[str, ...], list[str]]:
    """Return all values passed to a repeatable argparse option."""
    warnings: list[str] = []
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        warnings.append(f"example_command could not be parsed by shlex: {exc}")
        return (), warnings

    values: list[str] = []
    prefix = f"{option}="
    for index, token in enumerate(tokens):
        if token == option:
            if index + 1 >= len(tokens):
                warnings.append(f"example_command has {option} without value")
                continue
            values.append(tokens[index + 1])
        elif token.startswith(prefix):
            values.append(token.removeprefix(prefix))
    return tuple(values), warnings


def _under_output_prefix(path: Path | None, repo_root: Path) -> bool:
    if path is None:
        return False
    try:
        rel = path.resolve().relative_to((repo_root / EXPECTED_OUTPUT_PREFIX).resolve())
    except ValueError:
        return False
    return rel == Path(".") or bool(rel.parts)


def _same_resolved_path(left: Path | None, right: Path | None) -> bool:
    """Return true when both optional paths resolve to the same filesystem path."""
    if left is None or right is None:
        return False
    return left.resolve() == right.resolve()


def _report_path_inside_output_dir(report_json: Path | None, output_dir: Path | None) -> bool:
    """Return true when report_json is a future or file-valued path inside output_dir."""
    if report_json is None or output_dir is None:
        return False
    if report_json.is_symlink():
        return False
    if report_json.exists() and not report_json.is_file():
        return False
    resolved_report = report_json.resolve(strict=False)
    resolved_dir = output_dir.resolve(strict=False)
    return resolved_report != resolved_dir and resolved_report.is_relative_to(resolved_dir)


def _output_dir_is_directory_or_future(output_dir: Path | None) -> bool:
    """Return true when output_dir is either absent or an existing directory."""
    if output_dir is None:
        return False
    return not output_dir.exists() or output_dir.is_dir()


def _runner_row_fields(runner_path: Path | None) -> frozenset[str]:
    """Return static SamplerComparisonRow fields without importing or running runner."""
    if runner_path is None:
        raise FileNotFoundError("runner output schema target is not declared")
    if not runner_path.is_file():
        raise FileNotFoundError(f"runner output schema target is missing: {runner_path}")

    module = ast.parse(runner_path.read_text(encoding="utf-8"), filename=str(runner_path))

    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "SamplerComparisonRow":
            fields = {
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            }
            return frozenset(fields)

    raise ValueError("runner output schema missing SamplerComparisonRow dataclass")


def _registry_includes_package_b_manifest(
    registry_path: Path, manifest_path: Path, repo_root: Path
) -> bool:
    """Return true when the research package registry points at this manifest."""
    if not registry_path.is_file():
        return False

    try:
        payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(payload, dict):
        return False

    expected_manifest = _repo_relative(manifest_path, repo_root)
    packages = payload.get("packages")
    if not isinstance(packages, list):
        return False

    for package in packages:
        if not isinstance(package, dict):
            continue
        if package.get("id") != "package_b_adversarial":
            continue
        if package.get("issue") != EXPECTED_ISSUE:
            return False
        artifacts = _as_str_tuple(package.get("required_artifacts"))
        return expected_manifest in artifacts

    return False


def preflight_package_b_manifest(  # noqa: C901, PLR0912, PLR0915
    manifest_path: Path = Path("configs/adversarial/issue_3079_package_b_budget_matched.yaml"),
    *,
    repo_root: Path | None = None,
) -> PackageBPreflightResult:
    """Validate the package-B manifest readiness contract without executing it.

    Returns:
        Fail-closed readiness result with per-check booleans and blockers.
    """
    root = (repo_root or Path.cwd()).resolve()
    manifest_path = manifest_path if manifest_path.is_absolute() else root / manifest_path
    blockers: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {_repo_relative(manifest_path, root)}")

    try:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        payload = None
        blockers.append(f"manifest YAML could not be parsed: {exc}")

    if not isinstance(payload, dict):
        blockers.append("manifest payload must be a mapping")
        payload = {}

    checks["schema_version"] = payload.get("schema_version") == MANIFEST_SCHEMA_VERSION
    if not checks["schema_version"]:
        blockers.append(
            f"schema_version must be {MANIFEST_SCHEMA_VERSION!r}, "
            f"got {payload.get('schema_version')!r}"
        )

    checks["issue"] = payload.get("issue") == EXPECTED_ISSUE
    if not checks["issue"]:
        blockers.append(f"issue must be {EXPECTED_ISSUE}")

    registry_path = root / RESEARCH_PACKAGE_REGISTRY
    checks["research_package_registry_includes_manifest"] = _registry_includes_package_b_manifest(
        registry_path, manifest_path, root
    )
    if not checks["research_package_registry_includes_manifest"]:
        blockers.append(
            "research package registry must list the Package B manifest as a required artifact"
        )

    runner_path = _resolve_repo_path(root, payload.get("runner"))
    checks["runner_exists"] = bool(runner_path and runner_path.is_file())
    if not checks["runner_exists"]:
        blockers.append("runner must point at an existing repository file")

    base_config = payload.get("base_config")
    checks["base_config_mapping"] = isinstance(base_config, dict)
    if not isinstance(base_config, dict):
        blockers.append("base_config must be a mapping")
        base_config = {}

    for key in ("scenario_template", "search_space"):
        path = _resolve_repo_path(root, base_config.get(key))
        checks[f"{key}_exists"] = bool(path and path.is_file())
        if not checks[f"{key}_exists"]:
            blockers.append(f"base_config.{key} must point at an existing repository file")

    checks["policy_expected"] = base_config.get("policy") == EXPECTED_POLICY
    if not checks["policy_expected"]:
        blockers.append(f"base_config.policy must be {EXPECTED_POLICY!r}")

    checks["objective_expected"] = base_config.get("objective") == EXPECTED_OBJECTIVE
    if not checks["objective_expected"]:
        blockers.append(f"base_config.objective must be {EXPECTED_OBJECTIVE!r}")

    budgets = _as_int_tuple(payload.get("budget_grid"))
    checks["budget_grid"] = budgets == EXPECTED_BUDGETS
    if not checks["budget_grid"]:
        blockers.append(f"budget_grid must exactly match {list(EXPECTED_BUDGETS)}")

    seeds = _as_int_tuple(payload.get("repeated_seeds"))
    checks["repeated_seeds"] = (
        bool(seeds) and len(set(seeds)) == len(seeds) and all(seed >= 0 for seed in seeds)
    )
    if not checks["repeated_seeds"]:
        blockers.append("repeated_seeds must be non-empty, unique, non-negative integers")

    samplers = _as_str_tuple(payload.get("samplers"))
    checks["samplers"] = samplers == EXPECTED_SAMPLERS
    if not checks["samplers"]:
        blockers.append(f"samplers must exactly match {list(EXPECTED_SAMPLERS)}")

    reporting_contract = frozenset(_as_str_tuple(payload.get("reporting_contract")))
    missing_reporting = sorted(EXPECTED_REPORTING_FIELDS - reporting_contract)
    checks["reporting_contract"] = not missing_reporting
    if missing_reporting:
        blockers.append(f"reporting_contract missing required fields: {missing_reporting}")

    try:
        runner_row_fields = _runner_row_fields(runner_path)
    except (OSError, SyntaxError, ValueError) as exc:
        runner_row_fields = frozenset()
        warnings.append(f"runner output schema could not be parsed: {exc}")
    missing_runner_fields = sorted(reporting_contract - runner_row_fields)
    checks["runner_emits_reporting_contract"] = not missing_runner_fields
    if missing_runner_fields:
        blockers.append(
            "runner SamplerComparisonRow missing reporting_contract fields: "
            f"{missing_runner_fields}"
        )

    exclusions = payload.get("explicit_exclusions")
    checks["explicit_exclusions_mapping"] = isinstance(exclusions, dict)
    if not isinstance(exclusions, dict):
        blockers.append("explicit_exclusions must be a mapping")
        exclusions = {}

    exclusion_checks = {
        "learned_failure_proposal_issue_2921": "stretch_out_of_scope",
        "held_out_family_yield": "not_evaluated",
        "paper_facing_success_claims": "forbidden",
    }
    for key, required_fragment in exclusion_checks.items():
        value = str(exclusions.get(key, ""))
        checks[f"exclusion_{key}"] = required_fragment in value
        if not checks[f"exclusion_{key}"]:
            blockers.append(f"explicit_exclusions.{key} must include {required_fragment!r} caveat")

    checks["claim_scope_not_paper_facing"] = (
        payload.get("claim_scope") == "not_paper_facing_benchmark_evidence"
    )
    if not checks["claim_scope_not_paper_facing"]:
        blockers.append("claim_scope must stay not_paper_facing_benchmark_evidence")

    checks["status_diagnostic"] = payload.get("status") == "diagnostic_local_nominal"
    if not checks["status_diagnostic"]:
        blockers.append("status must stay diagnostic_local_nominal")

    output_artifacts = payload.get("output_artifacts")
    checks["output_artifacts_mapping"] = isinstance(output_artifacts, dict)
    if not isinstance(output_artifacts, dict):
        blockers.append("output_artifacts must be a mapping")
        output_artifacts = {}

    artifact_output_dir = _resolve_repo_path(root, output_artifacts.get("output_dir"))
    artifact_report_json = _resolve_repo_path(root, output_artifacts.get("report_json"))

    checks["output_artifacts_output_dir_declared"] = artifact_output_dir is not None
    if not checks["output_artifacts_output_dir_declared"]:
        blockers.append("output_artifacts.output_dir must be a non-empty path")

    checks["output_artifacts_output_dir_directory_or_future"] = _output_dir_is_directory_or_future(
        artifact_output_dir
    )
    if not checks["output_artifacts_output_dir_directory_or_future"]:
        blockers.append("output_artifacts.output_dir must be a directory or future path")

    checks["output_artifacts_report_json_declared"] = artifact_report_json is not None
    if not checks["output_artifacts_report_json_declared"]:
        blockers.append("output_artifacts.report_json must be a non-empty path")

    checks["output_artifacts_output_dir_under_issue_path"] = _under_output_prefix(
        artifact_output_dir, root
    )
    if not checks["output_artifacts_output_dir_under_issue_path"]:
        blockers.append(
            f"output_artifacts.output_dir must be under {EXPECTED_OUTPUT_PREFIX.as_posix()}"
        )

    checks["output_artifacts_report_json_under_issue_path"] = _under_output_prefix(
        artifact_report_json, root
    )
    if not checks["output_artifacts_report_json_under_issue_path"]:
        blockers.append(
            f"output_artifacts.report_json must be under {EXPECTED_OUTPUT_PREFIX.as_posix()}"
        )

    checks["output_artifacts_report_json_in_output_dir"] = _report_path_inside_output_dir(
        artifact_report_json, artifact_output_dir
    )
    if not checks["output_artifacts_report_json_in_output_dir"]:
        blockers.append("output_artifacts.report_json must be inside output_artifacts.output_dir")

    example_command = payload.get("example_command")
    checks["example_command_declared"] = isinstance(example_command, str) and bool(
        example_command.strip()
    )
    if not checks["example_command_declared"]:
        blockers.append("example_command must be a non-empty string")
        example_command = ""

    lower_command = example_command.lower()
    forbidden_hits = sorted(token for token in FORBIDDEN_COMMAND_TOKENS if token in lower_command)
    checks["example_command_no_forbidden_actions"] = not forbidden_hits
    if forbidden_hits:
        blockers.append(f"example_command includes forbidden action tokens: {forbidden_hits}")

    checks["example_command_uses_package_b_grid"] = "--package-b-budget-grid" in example_command
    if not checks["example_command_uses_package_b_grid"]:
        blockers.append("example_command must use --package-b-budget-grid")

    for seed in seeds:
        checks[f"example_command_seed_{seed}"] = (
            f"--seed {seed}" in example_command or f"--seed={seed}" in example_command
        )
        if not checks[f"example_command_seed_{seed}"]:
            blockers.append(f"example_command missing --seed {seed}")

    command_seeds, seed_warnings = _extract_repeated_seed_args(example_command)
    warnings.extend(seed_warnings)
    checks["example_command_repeated_seeds"] = command_seeds == seeds
    if not checks["example_command_repeated_seeds"]:
        blockers.append(
            "example_command --seed values must exactly match repeated_seeds "
            f"{list(seeds)}, got {list(command_seeds)}"
        )

    command_samplers, sampler_warnings = _extract_option_values(example_command, "--sampler")
    warnings.extend(sampler_warnings)
    checks["example_command_sampler_set"] = not command_samplers or command_samplers == samplers
    if not checks["example_command_sampler_set"]:
        blockers.append(
            "example_command --sampler values must be absent or exactly match samplers "
            f"{list(samplers)}, got {list(command_samplers)}"
        )

    command_scenario_templates, scenario_warnings = _extract_option_values(
        example_command, "--scenario-template"
    )
    warnings.extend(scenario_warnings)
    checks["example_command_scenario_template"] = not command_scenario_templates or tuple(
        _resolve_repo_path(root, value) for value in command_scenario_templates
    ) == (_resolve_repo_path(root, base_config.get("scenario_template")),)
    if not checks["example_command_scenario_template"]:
        blockers.append(
            "example_command --scenario-template must be absent or match "
            "base_config.scenario_template"
        )

    command_search_spaces, search_space_warnings = _extract_option_values(
        example_command, "--search-space"
    )
    warnings.extend(search_space_warnings)
    checks["example_command_search_space"] = not command_search_spaces or tuple(
        _resolve_repo_path(root, value) for value in command_search_spaces
    ) == (_resolve_repo_path(root, base_config.get("search_space")),)
    if not checks["example_command_search_space"]:
        blockers.append(
            "example_command --search-space must be absent or match base_config.search_space"
        )

    command_policies, policy_warnings = _extract_option_values(example_command, "--policy")
    warnings.extend(policy_warnings)
    checks["example_command_policy"] = not command_policies or command_policies == (
        str(base_config.get("policy")),
    )
    if not checks["example_command_policy"]:
        blockers.append("example_command --policy must be absent or match base_config.policy")

    command_objectives, objective_warnings = _extract_option_values(example_command, "--objective")
    warnings.extend(objective_warnings)
    checks["example_command_objective"] = not command_objectives or command_objectives == (
        str(base_config.get("objective")),
    )
    if not checks["example_command_objective"]:
        blockers.append("example_command --objective must be absent or match base_config.objective")

    output_dir, out_json, output_warnings = _extract_output_paths(example_command, root)
    warnings.extend(output_warnings)
    checks["output_dir_under_issue_path"] = _under_output_prefix(output_dir, root)
    if not checks["output_dir_under_issue_path"]:
        blockers.append(f"--output-dir must stay under {EXPECTED_OUTPUT_PREFIX.as_posix()}")
    checks["out_json_under_issue_path"] = _under_output_prefix(out_json, root)
    if not checks["out_json_under_issue_path"]:
        blockers.append(f"--out-json must stay under {EXPECTED_OUTPUT_PREFIX.as_posix()}")

    checks["example_command_matches_output_artifacts"] = _same_resolved_path(
        output_dir, artifact_output_dir
    ) and _same_resolved_path(out_json, artifact_report_json)
    if not checks["example_command_matches_output_artifacts"]:
        blockers.append("example_command output paths must match output_artifacts metadata")

    metadata = {
        "issue": EXPECTED_ISSUE,
        "budget_grid": list(budgets),
        "repeated_seeds": list(seeds),
        "example_command_repeated_seeds": list(command_seeds),
        "samplers": list(samplers),
        "base_config": {
            "policy": base_config.get("policy"),
            "objective": base_config.get("objective"),
        },
        "research_package_registry": _repo_relative(registry_path, root),
        "runner": _repo_relative(runner_path, root) if runner_path else None,
        "runner_reporting_fields": sorted(runner_row_fields),
        "output_dir": _repo_relative(output_dir, root) if output_dir else None,
        "out_json": _repo_relative(out_json, root) if out_json else None,
        "output_artifacts": {
            "output_dir": _repo_relative(artifact_output_dir, root)
            if artifact_output_dir
            else None,
            "report_json": _repo_relative(artifact_report_json, root)
            if artifact_report_json
            else None,
        },
        "claim_scope": payload.get("claim_scope"),
        "status": payload.get("status"),
        "does_not_execute_benchmark": True,
        "does_not_submit_slurm_or_gpu": True,
        "does_not_edit_paper_claims": True,
    }
    ready = not blockers
    return PackageBPreflightResult(
        manifest_path=_repo_relative(manifest_path, root),
        ready=ready,
        blocked=not ready,
        checks=checks,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        metadata=metadata,
    )


def dump_preflight_payload(result: PackageBPreflightResult, output: Path | None) -> None:
    """Write preflight payload to disk when requested."""
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
