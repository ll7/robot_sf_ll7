"""Small v1 helpers for draft scenario authoring and validation."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import yaml
from loguru import logger

from robot_sf.benchmark.scenario_generator import (
    PARAMETERIZED_SCENARIO_PARAMS_SCHEMA_VERSION,
    SCENARIO_GENERATION_PARAMS_SCHEMA_VERSION,
    derive_generation_parameters_from_physical_slice,
    estimate_initial_difficulty,
    normalize_generation_parameters,
    normalize_parameterized_scenario_parameters,
    select_map_id_for_parameterized_scenario,
)

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

AUTHORING_SCHEMA_VERSION = "robot_sf.scenario_authoring.v1"
SCENARIO_MATRIX_SCHEMA_VERSION = "robot_sf.scenario_matrix.v1"
DEFAULT_SOURCE_ISSUE = "#1891"
DEFAULT_SEEDS = (101, 102, 103)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """Actionable validation issue for an authored scenario YAML file."""

    path: str
    message: str
    hint: str

    def format(self) -> str:
        """Return a stable, human-readable validation message."""

        return f"{self.path}: {self.message} Hint: {self.hint}"


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Validation result for a scenario authoring YAML file."""

    scenario_path: Path
    scenario_count: int
    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        """Whether the authored scenario file passed v1 checks."""

        return not self.issues


def available_templates() -> tuple[str, ...]:
    """Return supported deterministic draft template names."""

    return ("bottleneck", "parameterized")


def configure_authoring_tool_logging(*, verbose: bool = False) -> None:
    """Keep authoring CLI output focused unless loader details are requested."""

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "ERROR")


def build_scenario_payload(
    *,
    template: str,
    name: str,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    source_issue: str = DEFAULT_SOURCE_ISSUE,
    generation_profile: dict[str, Any] | None = None,
    parameterized_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, reviewable scenario YAML payload.

    The generated payload is intentionally marked as draft and not benchmark
    evidence. It is a starting point for review and later certification.
    """

    normalized_template = template.strip().lower()
    if normalized_template not in available_templates():
        raise ValueError(
            f"Unknown scenario template '{template}'. "
            f"Available templates: {', '.join(available_templates())}."
        )
    scenario_name = _validate_output_name(name)
    seed_values = _validate_seed_tuple(seeds)
    physical_params = None
    if normalized_template == "parameterized":
        physical_params = normalize_parameterized_scenario_parameters(parameterized_profile or {})
        generation = derive_generation_parameters_from_physical_slice(physical_params)
        generation = normalize_generation_parameters(
            {**generation, "id": scenario_name, "repeats": 1}
        )
        map_id = select_map_id_for_parameterized_scenario(physical_params)
        purpose = (
            "Draft parameterized scenario review local smoke validation for sidewalk width, "
            "density, bottleneck, crossing angle, and occlusion knobs."
        )
    else:
        generation = normalize_generation_parameters(
            {**(generation_profile or {}), "id": scenario_name, "repeats": 1}
        )
        map_id = "classic_bottleneck"
        purpose = "Draft bottleneck scenario for review local smoke validation."
    initial_difficulty = estimate_initial_difficulty(generation)
    seed_signature = ",".join(str(seed) for seed in seed_values)
    return {
        "schema_version": SCENARIO_MATRIX_SCHEMA_VERSION,
        "authoring_schema_version": AUTHORING_SCHEMA_VERSION,
        "scenarios": [
            {
                "name": scenario_name,
                "map_id": map_id,
                "id": scenario_name,
                "density": generation["density"],
                "flow": generation["flow"],
                "obstacle": generation["obstacle"],
                "groups": generation["groups"],
                "speed_var": generation["speed_var"],
                "goal_topology": generation["goal_topology"],
                "robot_context": generation["robot_context"],
                "repeats": generation["repeats"],
                "simulation_config": {
                    "max_episode_steps": 300,
                    "ped_density": (
                        physical_params["pedestrian_density"]
                        if physical_params is not None
                        else 0.0
                    ),
                },
                "robot_config": {},
                "metadata": {
                    "archetype": normalized_template,
                    "generation_profile": {
                        "schema_version": SCENARIO_GENERATION_PARAMS_SCHEMA_VERSION,
                        "seed_signature": seed_signature,
                        "parameters": dict(generation),
                    },
                    **(
                        {
                            "parameterized_profile": {
                                "schema_version": PARAMETERIZED_SCENARIO_PARAMS_SCHEMA_VERSION,
                                "seed_signature": seed_signature,
                                "parameters": physical_params,
                            }
                        }
                        if physical_params is not None
                        else {}
                    ),
                    "initial_difficulty": initial_difficulty,
                    "density": "draft_low",
                    "flow": "bi",
                    "purpose": purpose,
                    "authoring": {
                        "status": "draft",
                        "template": normalized_template,
                        "template_version": AUTHORING_SCHEMA_VERSION,
                        "source_issue": source_issue,
                        "generated_by": "scripts/tools/create_scenario.py",
                        "benchmark_evidence": False,
                        "promotion_note": (
                            "Not benchmark evidence until separately reviewed, certified, "
                            "and executed through the benchmark workflow."
                        ),
                    },
                },
                "seeds": list(seed_values),
            },
        ],
    }


def dump_scenario_yaml(payload: dict[str, Any]) -> str:
    """Serialize a scenario payload with stable key order and block lists."""

    return yaml.dump(
        payload,
        sort_keys=False,
        width=100,
        allow_unicode=False,
        Dumper=_IndentedSafeDumper,
    )


def write_scenario_yaml(path: Path, payload: dict[str, Any], *, overwrite: bool = False) -> None:
    """Write a scenario payload, refusing accidental overwrites by default."""

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass --overwrite to replace it intentionally."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_scenario_yaml(payload), encoding="utf-8")


def validate_scenario_file(
    path: Path, *, require_authoring_metadata: bool = True
) -> ValidationReport:
    """Validate a draft scenario YAML file for authoring-time mistakes."""

    from robot_sf.training.scenario_loader import (
        build_robot_config_from_scenario,
        load_scenarios,
    )

    issues: list[ValidationIssue] = []
    raw = _load_raw_yaml(path, issues)
    raw_scenarios = _extract_raw_scenarios(raw, path=path, issues=issues)
    for index, scenario in enumerate(raw_scenarios):
        _validate_raw_scenario(
            scenario,
            index=index,
            issues=issues,
            require_authoring_metadata=require_authoring_metadata,
        )

    normalized_scenarios: list[dict[str, Any]] = []
    if not issues:
        try:
            normalized_scenarios = [dict(item) for item in load_scenarios(path)]
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    path="$",
                    message=f"scenario loader rejected the file: {exc}",
                    hint="Fix the YAML shape, includes, map_id, map_file, or map_search_paths.",
                )
            )

    for index, scenario in enumerate(normalized_scenarios):
        scenario_name = str(scenario.get("name") or scenario.get("scenario_id") or index)
        try:
            config = build_robot_config_from_scenario(scenario, scenario_path=path)
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    path=f"/scenarios/{index}",
                    message=f"scenario '{scenario_name}' could not build a robot config: {exc}",
                    hint="Check map references, route overrides, robot_config, and simulation_config.",
                )
            )
            continue
        if not getattr(config, "map_pool", None) or not config.map_pool.map_defs:
            issues.append(
                ValidationIssue(
                    path=f"/scenarios/{index}/map_file",
                    message=f"scenario '{scenario_name}' resolved no map definitions",
                    hint="Use a valid map_id from maps/registry.yaml or an existing map_file path.",
                )
            )

    return ValidationReport(
        scenario_path=path,
        scenario_count=len(raw_scenarios),
        issues=tuple(issues),
    )


def parse_seed_args(raw_seeds: list[str] | None) -> tuple[int, ...]:
    """Parse repeated or comma-separated seed CLI values."""

    if not raw_seeds:
        return DEFAULT_SEEDS
    parsed: list[int] = []
    for raw in raw_seeds:
        for item in raw.split(","):
            value = item.strip()
            if not value:
                continue
            parsed.append(_parse_seed(value))
    return _validate_seed_tuple(tuple(parsed))


def add_common_seed_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared seed argument used by authoring CLIs."""

    parser.add_argument(
        "--seeds",
        nargs="+",
        help="Draft seed list, either space-separated or comma-separated. Defaults to 101 102 103.",
    )


class _IndentedSafeDumper(yaml.SafeDumper):
    """PyYAML dumper that keeps nested sequences easy to review."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        """Indent nested lists under their mapping key."""

        return super().increase_indent(flow, False)


def _validate_output_name(name: str) -> str:
    """Validate a scenario name supplied to the generator."""

    normalized = name.strip()
    if not normalized:
        raise ValueError("Scenario name must be non-empty.")
    return normalized


def _parse_seed(raw: str) -> int:
    """Parse one CLI seed, rejecting booleans and floats."""

    try:
        return int(raw, 10)
    except ValueError as exc:
        raise ValueError(f"Seed '{raw}' must be an integer.") from exc


def _validate_seed_tuple(seeds: tuple[int, ...]) -> tuple[int, ...]:
    """Validate deterministic seed metadata for generated scenarios."""

    if not seeds:
        raise ValueError("At least one seed is required.")
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Seeds must contain integers.")
        if seed < 0:
            raise ValueError("Seeds must be non-negative integers.")
    return seeds


def _load_raw_yaml(path: Path, issues: list[ValidationIssue]) -> Any:
    """Load YAML while converting parser errors into validation issues."""

    if not path.exists():
        issues.append(
            ValidationIssue(
                path="$",
                message=f"scenario file does not exist: {path}",
                hint="Pass a path to a generated or hand-authored scenario YAML file.",
            )
        )
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        issues.append(
            ValidationIssue(
                path="$",
                message=f"invalid YAML: {exc}",
                hint="Fix the YAML syntax before running scenario validation again.",
            )
        )
        return None


def _extract_raw_scenarios(
    raw: Any,
    *,
    path: Path,
    issues: list[ValidationIssue],
) -> list[Any]:
    """Extract raw scenario entries for strict authoring validation."""

    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        scenarios = raw.get("scenarios")
        if isinstance(scenarios, list):
            if not scenarios:
                issues.append(
                    ValidationIssue(
                        path="/scenarios",
                        message=f"{path} must contain a non-empty scenarios list",
                        hint="Generate a skeleton with scripts/tools/create_scenario.py.",
                    )
                )
            return scenarios
        issues.append(
            ValidationIssue(
                path="/scenarios",
                message=f"{path} must contain a non-empty scenarios list",
                hint="Generate a skeleton with scripts/tools/create_scenario.py or add scenarios: [...].",
            )
        )
        return []
    issues.append(
        ValidationIssue(
            path="$",
            message=f"{path} must contain a mapping with scenarios or a scenario list",
            hint="Use the robot_sf.scenario_matrix.v1 YAML shape.",
        )
    )
    return []


def _validate_raw_scenario(
    scenario: Any,
    *,
    index: int,
    issues: list[ValidationIssue],
    require_authoring_metadata: bool,
) -> None:
    """Validate one raw scenario before invoking heavier repository loaders."""

    prefix = f"/scenarios/{index}"
    if not isinstance(scenario, dict):
        issues.append(
            ValidationIssue(
                path=prefix,
                message="scenario entry must be a mapping",
                hint="Use key/value YAML fields such as name, map_id, simulation_config, metadata, seeds.",
            )
        )
        return

    _require_non_empty_string(scenario, "name", path=prefix, issues=issues)
    _validate_map_reference(scenario, path=prefix, issues=issues)
    _require_mapping(scenario, "simulation_config", path=prefix, issues=issues)
    _require_mapping(scenario, "robot_config", path=prefix, issues=issues)
    _validate_metadata(
        scenario.get("metadata"),
        path=f"{prefix}/metadata",
        issues=issues,
        require_authoring_metadata=require_authoring_metadata,
    )
    _validate_seeds(scenario.get("seeds"), path=f"{prefix}/seeds", issues=issues)


def _require_non_empty_string(
    mapping: dict[str, Any],
    key: str,
    *,
    path: str,
    issues: list[ValidationIssue],
) -> None:
    """Require a non-empty string field."""

    value = mapping.get(key)
    if isinstance(value, str) and value.strip():
        return
    issues.append(
        ValidationIssue(
            path=f"{path}/{key}",
            message=f"{key} is required and must be a non-empty string",
            hint=f"Add {key}: <stable_identifier>.",
        )
    )


def _require_mapping(
    mapping: dict[str, Any],
    key: str,
    *,
    path: str,
    issues: list[ValidationIssue],
) -> None:
    """Require a mapping field, accepting an intentionally empty mapping."""

    if isinstance(mapping.get(key), dict):
        return
    issues.append(
        ValidationIssue(
            path=f"{path}/{key}",
            message=f"{key} is required and must be a mapping",
            hint=f"Add {key}: {{}} or the specific override fields for this scenario.",
        )
    )


def _validate_map_reference(
    scenario: dict[str, Any],
    *,
    path: str,
    issues: list[ValidationIssue],
) -> None:
    """Require exactly one non-empty map reference field."""

    map_id = scenario.get("map_id")
    map_file = scenario.get("map_file")
    has_map_id = isinstance(map_id, str) and bool(map_id.strip())
    has_map_file = isinstance(map_file, str) and bool(map_file.strip())
    if has_map_id ^ has_map_file:
        return
    if has_map_id and has_map_file:
        issues.append(
            ValidationIssue(
                path=path,
                message="scenario must set only one of map_id or map_file",
                hint="Prefer map_id for portable authored scenarios; remove the duplicate map reference.",
            )
        )
        return
    issues.append(
        ValidationIssue(
            path=path,
            message="scenario must set map_id or map_file",
            hint="Use map_id from maps/registry.yaml, for example classic_bottleneck.",
        )
    )


def _validate_metadata(
    metadata: Any,
    *,
    path: str,
    issues: list[ValidationIssue],
    require_authoring_metadata: bool,
) -> None:
    """Validate required draft metadata and benchmark-evidence caveats."""

    if not isinstance(metadata, dict):
        issues.append(
            ValidationIssue(
                path=path,
                message="metadata is required and must be a mapping",
                hint="Include purpose plus metadata.authoring with draft status and benchmark_evidence: false.",
            )
        )
        return
    _metadata_string(metadata, "purpose", path=path, issues=issues)
    if require_authoring_metadata or "generation_profile" in metadata:
        _validate_generation_profile(
            metadata.get("generation_profile"),
            path=f"{path}/generation_profile",
            issues=issues,
        )
    if not require_authoring_metadata:
        return
    authoring = metadata.get("authoring")
    if not isinstance(authoring, dict):
        issues.append(
            ValidationIssue(
                path=f"{path}/authoring",
                message="metadata.authoring is required for draft authoring validation",
                hint="Generate a skeleton or add status, template, source_issue, and benchmark_evidence.",
            )
        )
        return
    _metadata_string(authoring, "status", path=f"{path}/authoring", issues=issues)
    _metadata_string(authoring, "template", path=f"{path}/authoring", issues=issues)
    _metadata_string(authoring, "source_issue", path=f"{path}/authoring", issues=issues)
    if authoring.get("benchmark_evidence") is not False:
        issues.append(
            ValidationIssue(
                path=f"{path}/authoring/benchmark_evidence",
                message="benchmark_evidence must be false for generated draft scenarios",
                hint="Do not mark authored drafts as benchmark evidence; use certification and benchmark runs.",
            )
        )


def _metadata_string(
    metadata: dict[str, Any],
    key: str,
    *,
    path: str,
    issues: list[ValidationIssue],
) -> None:
    """Require a non-empty metadata string."""

    value = metadata.get(key)
    if isinstance(value, str) and value.strip():
        return
    issues.append(
        ValidationIssue(
            path=f"{path}/{key}",
            message=f"{key} is required and must be a non-empty string",
            hint=f"Add {key}: <reviewable text>.",
        )
    )


def _validate_generation_profile(
    generation_profile: Any,
    *,
    path: str,
    issues: list[ValidationIssue],
) -> None:
    """Validate generation profile metadata for reproducibility."""

    if not isinstance(generation_profile, dict):
        issues.append(
            ValidationIssue(
                path=path,
                message="metadata.generation_profile is required for draft authoring validation",
                hint=(
                    "Generate using create_scenario.py so generation_profile and "
                    "initial_difficulty are both present."
                ),
            )
        )
        return

    schema_version = generation_profile.get("schema_version")
    if schema_version != SCENARIO_GENERATION_PARAMS_SCHEMA_VERSION:
        issues.append(
            ValidationIssue(
                path=f"{path}/schema_version",
                message=(
                    f"metadata.generation_profile.schema_version must equal "
                    f"{SCENARIO_GENERATION_PARAMS_SCHEMA_VERSION}"
                ),
                hint="Use the scenario_generator profile schema version from constants.",
            )
        )

    parameters = generation_profile.get("parameters")
    if not isinstance(parameters, dict):
        issues.append(
            ValidationIssue(
                path=f"{path}/parameters",
                message="metadata.generation_profile.parameters is required and must be a mapping",
                hint="Include density, flow, obstacle, groups, speed_var, goal_topology.",
            )
        )
        return
    try:
        normalize_generation_parameters(parameters)
    except ValueError as exc:
        issues.append(
            ValidationIssue(
                path=path,
                message=f"metadata.generation_profile.parameters invalid: {exc}",
                hint="Fix invalid generation values before running the authoring validator.",
            )
        )


def _validate_seeds(raw_seeds: Any, *, path: str, issues: list[ValidationIssue]) -> None:
    """Require deterministic non-empty integer seed metadata."""

    if not isinstance(raw_seeds, list) or not raw_seeds:
        issues.append(
            ValidationIssue(
                path=path,
                message="seeds is required and must be a non-empty list",
                hint="Add deterministic seeds such as seeds: [101, 102, 103].",
            )
        )
        return
    invalid = [seed for seed in raw_seeds if isinstance(seed, bool) or not isinstance(seed, int)]
    if invalid:
        issues.append(
            ValidationIssue(
                path=path,
                message=f"seeds must contain integers; invalid entries: {invalid}",
                hint="Use integer YAML values, not strings, floats, or booleans.",
            )
        )
        return
    negative = [seed for seed in raw_seeds if seed < 0]
    if negative:
        issues.append(
            ValidationIssue(
                path=path,
                message=f"seeds must be non-negative integers; invalid entries: {negative}",
                hint="Use stable non-negative seed values.",
            )
        )
