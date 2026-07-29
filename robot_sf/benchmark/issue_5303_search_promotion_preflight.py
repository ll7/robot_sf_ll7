"""Side-effect-free preflight for the issue #5303 search-promotion frozen contract.

This checker freezes and reproduces the promotion contract for the existing
adversarial v1 search (Optuna/TPE versus random) on the held-out
``classic_group_crossing_medium`` family for ``scenario_adaptive_hybrid_orca_v2_collision_guard``.

What this module deliberately does NOT do
-----------------------------------------
It never executes planners, never runs a search/replay/confirmation campaign, never
submits Slurm jobs, and never reads evaluation outcomes. Concretely it does not import
or call any of ``robot_sf.adversarial`` execution surfaces (``samplers``, ``search``,
``runtime``, ``qd``, ``warm_start``, ``transfer_matrix``, or any campaign/replay/
benchmark-runner module). It only reads the frozen contract config, the issue #6139
recertification receipt, preregistration manifest, and statically parsed handoff source
files. It recomputes SHA-256 hashes and asserts the frozen fields, outcome schema,
runner/objective support, and power analysis.

The companion test ``tests/adversarial/test_issue_5303_search_promotion_preflight.py``
AST-scans this module's source to prove the side-effect-free contract holds.
"""

# evidence-writer-exempt: dump_preflight_payload writes only an optional caller-selected
# diagnostic path; the frozen preflight and default checker remain read-only and never
# target the tracked evidence tree.

from __future__ import annotations

import ast
import hashlib
import json
import math
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "issue_5303_search_promotion_preflight.v1"
CONTRACT_SCHEMA_VERSION = "issue_5303_search_promotion_contract.v1"
MANIFEST_SCHEMA_VERSION = "issue_5303_search_promotion_manifest.v1"

EXPECTED_ISSUE = 5303
EXPECTED_STEP = 2
EXPECTED_PARENT = 5303

DEFAULT_CONTRACT_PATH = Path("configs/adversarial/issue_5303_search_promotion_contract.yaml")
DEFAULT_RECEIPT_PATH = Path(
    "docs/context/evidence/issue_5305_certified_archive/recertification_issue_6139.json"
)
DEFAULT_MANIFEST_PATH = Path(
    "docs/context/evidence/issue_5303_search_promotion_preregistration/contract_frozen.json"
)

EXPECTED_ELIGIBLE_COUNT = 8
EXPECTED_RECORD_COUNT = 17
EXPECTED_ELIGIBLE_FLOOR = 2
EXPECTED_FIT_FAMILY = "classic_cross_trap_medium"
EXPECTED_FRESH_FAMILY = "classic_group_crossing_medium"
EXPECTED_METHODS: tuple[str, ...] = ("optuna", "random")
EXPECTED_CANDIDATE_BUDGET = 64
EXPECTED_SEEDS_PER_METHOD = 3
EXPECTED_SEARCH_SEEDS: tuple[int, ...] = (530301, 530302, 530303)
EXPECTED_TOTAL_CANDIDATES_PER_METHOD = 192
EXPECTED_HORIZON_STEPS = 100
EXPECTED_DT_S = 0.1
EXPECTED_TIME_CAP_S = 10.0
EXPECTED_DOORWAY_SEEDS: tuple[int, ...] = (128, 130)
EXPECTED_WARM_START_RECORD_IDS: tuple[str, ...] = (
    "issue5305_classic_cross_trap_medium_fbbd96687d61",
    "issue5305_classic_cross_trap_medium_fe24f0ff86a1",
)
EXPECTED_NEGATIVE_CONTROL_FAMILY = "francis2023_blind_corner"
EXPECTED_DIAGNOSTIC_EXECUTION_MODE = "adapter"
EXPECTED_NULL_TEST_COUNT = 2
EXPECTED_NULL_TEST_NAMES: tuple[str, ...] = (
    "shuffled_outcome_seed_permutation",
    "ranking_permutation_seed",
)
EXPECTED_COUNTED_GATE_COUNT = 7
EXPECTED_COUNTED_GATE_NAMES: tuple[str, ...] = (
    "corrected_scenario_path_certification",
    "deterministic_replay",
    "target_failure_in_at_least_4_of_5_seeds_no_retries",
    "same_primary_mechanism_in_at_least_4_of_5_seeds",
    "neutral_reference_planner_succeeds_in_at_least_4_of_5_seeds",
    "shortlist_passes_threshold_in_second_execution_context",
    "no_excluded_row_class",
)
EXPECTED_EXCLUDED_ROW_CLASSES = (
    "fallback",
    "degraded",
    "unavailable",
    "geometry_artifact",
    "knife_edge",
    "stress_only",
    "duplicate",
)
NULL_THRESHOLD = 0.05
DIAGNOSTIC_DECLARATION = "diagnostic_inconclusive"
OUTCOME_ROW_SCHEMA_VERSION = "issue_5303_search_promotion_outcome_row.v1"

EXPECTED_PROVENANCE_INPUT_IDS = frozenset(
    {
        "target_planner_config",
        "neutral_reference_planner_config",
        "fit_family_config",
        "fresh_family_config",
        "diagnostic_scenario_template",
        "search_space",
        "certification_runner",
        "objective_registry",
        "diagnostic_runner",
        "adversarial_search_runner",
        "promotion_analysis_module",
        "promotion_analysis_cli",
        "preflight_module",
        "contract_check_cli",
    }
)
EXPECTED_PROVENANCE_PATHS = {
    "target_planner_config": (
        "configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_collision_guard.yaml"
    ),
    "neutral_reference_planner_config": "configs/policy_search/candidates/scenario_adaptive_orca_v1.yaml",
    "fit_family_config": "configs/scenarios/archetypes/classic_cross_trap.yaml",
    "fresh_family_config": "configs/scenarios/archetypes/classic_group_crossing.yaml",
    "diagnostic_scenario_template": (
        "configs/adversarial/issue_5303_classic_group_crossing_medium.yaml"
    ),
    "search_space": "configs/adversarial/issue_5303_search_promotion_space.yaml",
    "certification_runner": "robot_sf/adversarial/certification.py",
    "objective_registry": "robot_sf/adversarial/objectives.py",
    "diagnostic_runner": "scripts/tools/compare_adversarial_samplers.py",
    "adversarial_search_runner": "robot_sf/adversarial/search.py",
    "promotion_analysis_module": "robot_sf/benchmark/issue_5303_search_promotion_analysis.py",
    "promotion_analysis_cli": "scripts/tools/analyze_issue_5303_search_promotion.py",
    "preflight_module": "robot_sf/benchmark/issue_5303_search_promotion_preflight.py",
    "contract_check_cli": "scripts/tools/check_issue_5303_search_promotion_contract.py",
}
EXPECTED_ENTRY_GATE_BINDINGS = {
    "blocking_issue": 6139,
    "blocking_issue_state": "merged",
    "recertification_receipt_path": DEFAULT_RECEIPT_PATH.as_posix(),
    "certified_archive_path": "docs/context/evidence/issue_5305_certified_archive/archive.json",
    "record_count": EXPECTED_RECORD_COUNT,
    "eligible_count": EXPECTED_ELIGIBLE_COUNT,
    "eligible_floor": EXPECTED_ELIGIBLE_FLOOR,
}
EXPECTED_REJECTION_CONTROLS = [
    {
        "family": "doorway",
        "config": "configs/scenarios/archetypes/classic_doorway.yaml",
        "seeds": list(EXPECTED_DOORWAY_SEEDS),
        "role": "must_not_yield_discoveries_counted_as_weak_points",
    }
]
EXPECTED_NEGATIVE_CONTROL = {
    "family": EXPECTED_NEGATIVE_CONTROL_FAMILY,
    "config": "configs/scenarios/single/francis2023_blind_corner.yaml",
    "map": "maps/svg_maps/francis2023/francis2023_blind_corner.svg",
    "role": "certifier_must_reject_never_a_candidate_or_denominator_row",
}
EXPECTED_METHOD_ENTRIES = [
    {
        "name": "optuna",
        "role": "existing_tpe_tree_structured_parzen_estimator",
        "sampler_class": "robot_sf.adversarial.samplers.OptunaCandidateSampler",
        "builder": "robot_sf.adversarial.samplers.build_sampler",
    },
    {
        "name": "random",
        "role": "existing_dependency_light_random_search",
        "sampler_class": "robot_sf.adversarial.samplers.RandomCandidateSampler",
        "builder": "robot_sf.adversarial.samplers.build_sampler",
    },
]
EXPECTED_FEASIBILITY_RULES = [
    "corrected_swept_envelope_full_polyline_clearance",
    "runtime_simulator_obstacle_collision",
    "search_space_validation",
]
EXPECTED_OBJECTIVE_TIERS = [
    {
        "tier": 1,
        "name": "collision_or_severe_intrusion",
        "kind": "hard_constraint",
        "veto": True,
    },
    {
        "tier": 2,
        "name": "liveness_or_goal_completion",
        "kind": "liveness",
        "no_soft_compensation_when_tier_1_fails": True,
    },
    {
        "tier": 3,
        "name": "comfort_and_efficiency",
        "kind": "soft",
        "compensates_hard_constraint_or_zero_goal_completion": False,
    },
]
EXPECTED_COUNTED_GATE_ENTRIES = [
    {
        "id": 1,
        "name": "corrected_scenario_path_certification",
        "rule": "candidate passes corrected swept-envelope and runtime simulator-collision certification",
    },
    {
        "id": 2,
        "name": "deterministic_replay",
        "rule": "exact deterministic-replay signature agreement",
    },
    {
        "id": 3,
        "name": "target_failure_in_at_least_4_of_5_seeds_no_retries",
        "rule": "target planner fails in at least 4 of 5 fresh confirmation seeds with no retries",
    },
    {
        "id": 4,
        "name": "same_primary_mechanism_in_at_least_4_of_5_seeds",
        "rule": "the same primary failure mechanism reproduces in at least 4 of 5 seeds",
    },
    {
        "id": 5,
        "name": "neutral_reference_planner_succeeds_in_at_least_4_of_5_seeds",
        "rule": "the neutral reference planner succeeds in at least 4 of 5 of the same seeds",
    },
    {
        "id": 6,
        "name": "shortlist_passes_threshold_in_second_execution_context",
        "rule": "the shortlist passes the same threshold in a second recorded execution context",
    },
    {
        "id": 7,
        "name": "no_excluded_row_class",
        "rule": "no fallback, degraded, unavailable, geometry_artifact, knife_edge, stress_only, or duplicate classification",
    },
]
EXPECTED_FORBIDDEN_ACTIONS = [
    "planner_execution",
    "adversarial_search_campaign",
    "replay_or_confirmation_run",
    "slurm_or_sbatch_or_srun_submission",
    "evaluation_outcome_import_or_read",
]
EXPECTED_OUTCOME_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "row_id",
        "arm",
        "method",
        "search_seed",
        "candidate_index",
        "normalized_candidate_config_sha256",
        "candidate",
        "scenario_family",
        "scenario_config_path",
        "scenario_config_sha256",
        "search_space_path",
        "search_space_sha256",
        "target_planner_config_path",
        "target_planner_config_sha256",
        "neutral_reference_planner_config_path",
        "neutral_reference_planner_config_sha256",
        "execution_stage",
        "execution_seed",
        "seed_lineage",
        "execution_mode",
        "readiness_status",
        "availability_status",
        "constraints_first_outcome",
        "objective",
        "objective_value",
        "primary_failure_mechanism",
        "stable_attribution_evidence",
        "certification",
        "recertification_lineage",
        "deterministic_replay",
        "confirmation_target",
        "confirmation_reference",
        "second_execution_context",
        "execution_commit",
        "execution_context_label",
        "admission_decision",
        "exclusion_reason",
        "immutable_record_sha256",
    }
)
REQUIRED_DIAGNOSTIC_RUNNER_OPTIONS = frozenset(
    {
        "--scenario-template",
        "--scenario-family",
        "--contract",
        "--search-space",
        "--policy",
        "--algo-config",
        "--reference-algo-config",
        "--objective",
        "--output-dir",
        "--budget",
        "--seed",
        "--horizon",
        "--dt",
        "--require-certification",
        "--benchmark-profile",
        "--sampler",
        "--out-json",
        "--out-md",
        "--outcomes-jsonl",
        "--issue-5303-diagnostic-only",
        "--execution-context-label",
        "--warm-start-archive",
        "--warm-start-record",
    }
)


@dataclass(frozen=True)
class Issue5303PreflightResult:
    """Structured promotion-contract preflight result."""

    contract_path: str
    ready: bool
    blocked: bool
    checks: dict[str, bool]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Return a stable JSON payload for CLI and tests."""
        return {
            "schema_version": SCHEMA_VERSION,
            "contract_path": self.contract_path,
            "ready": self.ready,
            "blocked": self.blocked,
            "checks": self.checks,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file's raw bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve(repo_root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _read_python_ast(path: Path) -> ast.Module | None:
    """Parse a source file statically, returning ``None`` for unreadable syntax.

    Returns:
        Parsed module, or ``None`` if the path cannot be parsed without importing it.
    """
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None


def _static_dict_keys(path: Path, *, assignment_name: str) -> set[str]:
    """Return literal string keys from a module-level mapping assignment."""
    tree = _read_python_ast(path)
    if tree is None:
        return set()
    for node in tree.body:
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == assignment_name for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == assignment_name
        ):
            value = node.value
        if isinstance(value, ast.Dict):
            return {
                key.value
                for key in value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    return set()


def _function_names(path: Path) -> set[str]:
    """Return top-level function names without importing the target module."""
    tree = _read_python_ast(path)
    if tree is None:
        return set()
    return {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _parser_options(path: Path) -> set[str]:
    """Return literal ``add_argument`` options from a runner source file."""
    tree = _read_python_ast(path)
    if tree is None:
        return set()
    options: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                options.add(arg.value)
    return options


def _algorithm_default_execution_mode(path: Path, algorithm: str) -> str | None:
    """Read an algorithm's declared production execution mode without importing it.

    Returns:
        Declared execution mode, or ``None`` when the static profile is unavailable.
    """
    tree = _read_python_ast(path)
    if tree is None:
        return None
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id != "_KINEMATICS_PROFILE_BY_CANONICAL" or not isinstance(
            node.value, ast.Dict
        ):
            continue
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            if not (isinstance(key, ast.Constant) and key.value == algorithm):
                continue
            if not isinstance(value, ast.Dict):
                return None
            for profile_key, profile_value in zip(value.keys, value.values, strict=True):
                if (
                    isinstance(profile_key, ast.Constant)
                    and profile_key.value == "default_execution_mode"
                    and isinstance(profile_value, ast.Constant)
                    and isinstance(profile_value.value, str)
                ):
                    return profile_value.value
    return None


def _function_row_keys(path: Path, *, function_name: str) -> set[str]:
    """Return literal keys assigned to the ``row`` mapping in one function."""
    tree = _read_python_ast(path)
    if tree is None:
        return set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != function_name:
            continue
        keys: set[str] = set()
        for nested in ast.walk(node):
            if isinstance(nested, ast.Assign):
                if any(
                    isinstance(target, ast.Name) and target.id == "row" for target in nested.targets
                ) and isinstance(nested.value, ast.Dict):
                    keys.update(
                        key.value
                        for key in nested.value.keys
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)
                    )
                for target in nested.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "row"
                        and isinstance(target.slice, ast.Constant)
                        and isinstance(target.slice.value, str)
                    ):
                        keys.add(target.slice.value)
        return keys
    return set()


def _command_options(command: Any) -> tuple[dict[str, list[str | None]], str | None]:
    """Parse a frozen command without executing it, preserving repeated options.

    Returns:
        Parsed option occurrences and an error string when static parsing fails.
    """
    if not isinstance(command, str) or not command.strip():
        return {}, "command must be a non-empty string"
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return {}, f"command cannot be parsed: {exc}"
    if not tokens:
        return {}, "command must not be empty"
    options: dict[str, list[str | None]] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        if token in {"--require-certification", "--issue-5303-diagnostic-only"}:
            options.setdefault(token, []).append(None)
            index += 1
            continue
        if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
            return options, f"{token} requires a value"
        options.setdefault(token, []).append(tokens[index + 1])
        index += 2
    return options, None


def _command_entrypoint_matches(command: Any, expected_script: str) -> bool:
    """Return whether a frozen command invokes the expected script entry point."""
    if not isinstance(command, str) or not command.strip():
        return False
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    return tokens[:4] == ["uv", "run", "python", expected_script]


def _warm_start_space_errors(  # noqa: C901, PLR0912, PLR0915
    *, archive_path: Path, record_ids: tuple[str, ...], search_space_path: Path
) -> list[str]:
    """Validate frozen archive warm starts against the declared search-space bounds.

    This mirrors the input checks that ``SearchConfig.validate`` performs without importing
    adversarial execution modules into the side-effect-free preflight.

    Returns:
        A list of compatibility errors; an empty list means all selected warm starts fit.
    """
    errors: list[str] = []
    try:
        archive = json.loads(archive_path.read_text(encoding="utf-8"))
        search_space = yaml.safe_load(search_space_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return [f"warm-start compatibility inputs could not be loaded: {exc}"]
    entries = archive.get("entries") if isinstance(archive, dict) else None
    variables = search_space.get("variables") if isinstance(search_space, dict) else None
    constraints = search_space.get("constraints", {}) if isinstance(search_space, dict) else {}
    if not isinstance(entries, list):
        return ["warm-start archive must contain an entries list"]
    if not isinstance(variables, dict) or not isinstance(constraints, dict):
        return ["warm-start search space must contain variables and constraints mappings"]
    by_id = {
        entry.get("archive_id"): entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("archive_id"), str)
    }
    for record_id in record_ids:
        entry = by_id.get(record_id)
        candidate = entry.get("candidate") if isinstance(entry, dict) else None
        if not isinstance(candidate, dict):
            errors.append(f"warm-start record {record_id!r} has no candidate mapping")
            continue
        start = candidate.get("start")
        goal = candidate.get("goal")
        if not isinstance(start, dict) or not isinstance(goal, dict):
            errors.append(f"warm-start record {record_id!r} has invalid start/goal poses")
            continue
        values = {
            "start_x": start.get("x"),
            "start_y": start.get("y"),
            "goal_x": goal.get("x"),
            "goal_y": goal.get("y"),
            "spawn_time_s": candidate.get("spawn_time_s"),
            "pedestrian_speed_mps": candidate.get("pedestrian_speed_mps"),
            "pedestrian_delay_s": candidate.get("pedestrian_delay_s"),
            "scenario_seed": candidate.get("scenario_seed"),
        }
        numeric_values: dict[str, float] = {}
        for name, raw_value in values.items():
            try:
                parsed = float(raw_value)
            except (TypeError, ValueError):
                errors.append(f"warm-start {record_id!r} has non-numeric {name}")
                continue
            if not math.isfinite(parsed):
                errors.append(f"warm-start {record_id!r} has non-finite {name}")
                continue
            numeric_values[name] = parsed
            bound = variables.get(name)
            if not isinstance(bound, dict):
                errors.append(f"warm-start search space is missing bounds for {name}")
                continue
            try:
                lower = float(bound["min"])
                upper = float(bound["max"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"warm-start search space has invalid bounds for {name}")
                continue
            if not math.isfinite(lower) or not math.isfinite(upper) or not lower <= parsed <= upper:
                errors.append(
                    f"warm-start {record_id!r} {name}={parsed} is outside [{lower}, {upper}]"
                )
        seed = numeric_values.get("scenario_seed")
        if seed is not None and not seed.is_integer():
            errors.append(f"warm-start {record_id!r} scenario_seed must be an integer")
        if numeric_values.get("spawn_time_s", 0.0) < 0.0:
            errors.append(f"warm-start {record_id!r} spawn_time_s must be non-negative")
        if numeric_values.get("pedestrian_speed_mps", 1.0) <= 0.0:
            errors.append(f"warm-start {record_id!r} pedestrian_speed_mps must be positive")
        if numeric_values.get("pedestrian_delay_s", 0.0) < 0.0:
            errors.append(f"warm-start {record_id!r} pedestrian_delay_s must be non-negative")
        min_distance = constraints.get("min_start_goal_distance_m", 0.25)
        try:
            min_distance_value = float(min_distance)
            distance = math.hypot(
                numeric_values["goal_x"] - numeric_values["start_x"],
                numeric_values["goal_y"] - numeric_values["start_y"],
            )
        except (KeyError, TypeError, ValueError):
            continue
        if distance < min_distance_value:
            errors.append(
                f"warm-start {record_id!r} start/goal distance {distance:.6f} "
                f"is below {min_distance_value:.6f}"
            )
    return errors


def _single_command_value(options: dict[str, list[str | None]], option: str) -> str | None:
    """Return a single required command value, or ``None`` when ambiguous/missing."""
    values = options.get(option, [])
    if len(values) != 1 or values[0] is None:
        return None
    return values[0]


def _check_input_provenance(
    provenance: Any,
    *,
    repo_root: Path,
    checks: dict[str, bool],
    blockers: list[str],
    metadata: dict[str, Any],
) -> None:
    """Check every executable handoff input exists and matches its frozen raw hash."""
    checks["input_provenance_algorithm"] = (
        isinstance(provenance, dict) and provenance.get("algorithm") == "sha256_raw_file_bytes"
    )
    if not checks["input_provenance_algorithm"]:
        blockers.append("input_provenance.algorithm must be sha256_raw_file_bytes")
    entries = provenance.get("required_inputs") if isinstance(provenance, dict) else None
    if not isinstance(entries, list):
        checks["input_provenance_complete"] = False
        checks["input_provenance_hashes"] = False
        blockers.append("input_provenance.required_inputs must be a list")
        return
    indexed = {
        entry.get("id"): entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    checks["input_provenance_complete"] = set(indexed) == EXPECTED_PROVENANCE_INPUT_IDS and len(
        indexed
    ) == len(entries)
    if not checks["input_provenance_complete"]:
        blockers.append(
            "input provenance IDs must exactly cover the target/reference/family/search/runner/"
            "analysis handoff inputs"
        )
    observed_hashes: dict[str, str] = {}
    all_hashes_match = True
    for input_id, expected_path in EXPECTED_PROVENANCE_PATHS.items():
        entry = indexed.get(input_id)
        raw_path = entry.get("path") if isinstance(entry, dict) else None
        expected_hash = entry.get("sha256") if isinstance(entry, dict) else None
        path = _resolve(repo_root, raw_path)
        if raw_path != expected_path or path is None or not path.is_file():
            all_hashes_match = False
            blockers.append(
                f"input provenance {input_id!r} must reference existing {expected_path!r}"
            )
            continue
        actual_hash = sha256_file(path)
        observed_hashes[input_id] = actual_hash
        if not isinstance(expected_hash, str) or actual_hash != expected_hash:
            all_hashes_match = False
            blockers.append(
                f"input provenance SHA-256 mismatch for {input_id!r} "
                f"(contract={expected_hash!r}, recomputed={actual_hash!r})"
            )
    checks["input_provenance_hashes"] = all_hashes_match
    metadata["input_provenance_sha256"] = observed_hashes


def _eligible_records_by_family(receipt: dict[str, Any]) -> dict[str, list[str]]:
    """Group eligible archive IDs by family using the receipt's corrected verdicts.

    Returns:
        Mapping of scenario family to the sorted eligible archive IDs recorded by
        the corrected #6139 recertification.
    """
    grouped: dict[str, list[str]] = {}
    records = receipt.get("records", [])
    if not isinstance(records, list):
        return grouped
    for record in records:
        if not isinstance(record, dict):
            continue
        after = record.get("after") if isinstance(record.get("after"), dict) else {}
        eligibility = after.get("benchmark_eligibility")
        if eligibility != "eligible":
            continue
        family = record.get("scenario_family")
        archive_id = record.get("archive_id")
        if isinstance(family, str) and isinstance(archive_id, str):
            grouped.setdefault(family, []).append(archive_id)
    for family in grouped:
        grouped[family] = sorted(grouped[family])
    return grouped


def _as_int_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return ()
        out.append(item)
    return tuple(out)


def _approx_equal(left: Any, right: float, tol: float = 1e-12) -> bool:
    try:
        return abs(float(left) - float(right)) <= tol
    except (TypeError, ValueError):
        return False


def _min_permutation_p_values(seeds_per_method: int) -> tuple[int, float, float]:
    """Return (arrangements, min one-sided p, min two-sided p) for a seed-label permutation.

    With ``seeds_per_method`` seeds per method, the permutation null relabels
    ``2 * seeds_per_method`` seed observations. The number of distinct label
    assignments is ``C(2n, n)``; the most extreme observed statistic is one
    arrangement, so the minimum one-sided p is ``1 / C(2n, n)`` and the minimum
    two-sided p is ``2 / C(2n, n)``.
    """
    total = 2 * seeds_per_method
    arrangements = math.comb(total, seeds_per_method)
    min_one_sided = 1.0 / arrangements
    min_two_sided = 2.0 / arrangements
    return arrangements, min_one_sided, min_two_sided


def preflight_issue_5303_contract(  # noqa: C901, PLR0912, PLR0915
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    *,
    receipt_path: Path | None = None,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    repo_root: Path | None = None,
) -> Issue5303PreflightResult:
    """Validate the frozen promotion contract without executing anything.

    Returns:
        Fail-closed result with per-check booleans and blockers.
    """
    root = (repo_root or Path.cwd()).resolve()
    contract_path = contract_path if contract_path.is_absolute() else root / contract_path
    if receipt_path is not None and not receipt_path.is_absolute():
        receipt_path = root / receipt_path
    manifest_path = manifest_path if manifest_path.is_absolute() else root / manifest_path

    checks: dict[str, bool] = {}
    blockers: list[str] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {}

    # ---- Load the frozen contract -------------------------------------------------
    checks["contract_exists"] = contract_path.is_file()
    if not checks["contract_exists"]:
        blockers.append(f"contract not found: {_repo_relative(contract_path, root)}")
        return Issue5303PreflightResult(
            contract_path=_repo_relative(contract_path, root),
            ready=False,
            blocked=True,
            checks=checks,
            blockers=tuple(blockers),
            warnings=tuple(warnings),
        )

    try:
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        blockers.append(f"contract YAML could not be parsed: {exc}")
        return Issue5303PreflightResult(
            contract_path=_repo_relative(contract_path, root),
            ready=False,
            blocked=True,
            checks=checks,
            blockers=tuple(blockers),
            warnings=tuple(warnings),
        )
    if not isinstance(contract, dict):
        blockers.append("contract payload must be a mapping")
        contract = {}

    checks["contract_schema_version"] = contract.get("schema_version") == CONTRACT_SCHEMA_VERSION
    if not checks["contract_schema_version"]:
        blockers.append(
            f"schema_version must be {CONTRACT_SCHEMA_VERSION!r}, "
            f"got {contract.get('schema_version')!r}"
        )

    checks["contract_issue"] = contract.get("issue") == EXPECTED_ISSUE
    if not checks["contract_issue"]:
        blockers.append(f"issue must be {EXPECTED_ISSUE}")
    checks["contract_step"] = contract.get("step") == EXPECTED_STEP
    if not checks["contract_step"]:
        blockers.append(f"step must be {EXPECTED_STEP}")
    checks["contract_parent"] = contract.get("parent_issue") == EXPECTED_PARENT
    if not checks["contract_parent"]:
        blockers.append(f"parent_issue must be {EXPECTED_PARENT}")

    checks["evidence_boundary_proposal_only"] = (
        contract.get("evidence_boundary") == "proposal_preflight_only"
    )
    if not checks["evidence_boundary_proposal_only"]:
        blockers.append("evidence_boundary must stay proposal_preflight_only")

    # ---- Manifest + contract hash (reproduce the frozen contract hash) ------------
    checks["manifest_exists"] = manifest_path.is_file()
    contract_hash = sha256_file(contract_path)
    metadata["contract_file_sha256"] = contract_hash
    manifest_hash: str | None = None
    if checks["manifest_exists"]:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            manifest = {}
            warnings.append(f"manifest JSON could not be parsed: {exc}")
        manifest_hash = manifest.get("contract_sha256") if isinstance(manifest, dict) else None
        checks["manifest_schema_version"] = (
            isinstance(manifest, dict) and manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION
        )
        if not checks["manifest_schema_version"]:
            blockers.append(f"manifest schema_version must be {MANIFEST_SCHEMA_VERSION!r}")
        checks["contract_hash_matches_manifest"] = manifest_hash == contract_hash
        if not checks["contract_hash_matches_manifest"]:
            blockers.append(
                "contract SHA-256 does not match the frozen manifest hash "
                f"(manifest={manifest_hash!r}, recomputed={contract_hash!r}); "
                "the contract was changed after freezing and must be re-preregistered"
            )
    else:
        checks["manifest_schema_version"] = False
        checks["contract_hash_matches_manifest"] = False
        blockers.append(f"manifest not found: {_repo_relative(manifest_path, root)}")

    # ---- Receipt and archive hashes (contract <-> entry-gate tamper-evidence) -----
    entry_gate = contract.get("entry_gate") if isinstance(contract.get("entry_gate"), dict) else {}
    receipt_resolved = (
        receipt_path
        or _resolve(root, entry_gate.get("recertification_receipt_path"))
        or root / DEFAULT_RECEIPT_PATH
    )
    certified_archive_resolved = _resolve(root, entry_gate.get("certified_archive_path"))
    checks["certified_archive_exists"] = bool(
        certified_archive_resolved and certified_archive_resolved.is_file()
    )
    certified_archive_file_hash: str | None = None
    if checks["certified_archive_exists"]:
        assert certified_archive_resolved is not None
        certified_archive_file_hash = sha256_file(certified_archive_resolved)
        metadata["certified_archive_file_sha256"] = certified_archive_file_hash
        checks["certified_archive_file_hash_matches_contract"] = (
            certified_archive_file_hash == entry_gate.get("certified_archive_sha256")
        )
        if not checks["certified_archive_file_hash_matches_contract"]:
            blockers.append(
                "certified archive file SHA-256 does not match the contract's frozen value "
                f"(contract={entry_gate.get('certified_archive_sha256')!r}, "
                f"recomputed={certified_archive_file_hash!r})"
            )
    else:
        checks["certified_archive_file_hash_matches_contract"] = False
        if certified_archive_resolved is None:
            blockers.append("entry_gate.certified_archive_path must be a non-empty path")
        else:
            blockers.append(
                f"certified archive not found: {_repo_relative(certified_archive_resolved, root)}"
            )
    checks["receipt_exists"] = receipt_resolved.is_file()
    receipt: dict[str, Any] = {}
    if checks["receipt_exists"]:
        try:
            receipt_payload = json.loads(receipt_resolved.read_text(encoding="utf-8"))
            if isinstance(receipt_payload, dict):
                receipt = receipt_payload
            else:
                blockers.append("receipt payload must be a mapping")
        except (OSError, ValueError) as exc:
            blockers.append(f"receipt JSON could not be parsed: {exc}")
            receipt = {}
        receipt_file_hash = sha256_file(receipt_resolved)
        metadata["receipt_file_sha256"] = receipt_file_hash
        checks["receipt_file_hash_matches_contract"] = receipt_file_hash == entry_gate.get(
            "recertification_receipt_file_sha256"
        )
        if not checks["receipt_file_hash_matches_contract"]:
            blockers.append(
                "recertification receipt file SHA-256 does not match the contract's "
                f"frozen value (contract={entry_gate.get('recertification_receipt_file_sha256')!r}, "
                f"recomputed={receipt_file_hash!r})"
            )
        receipt_self_declared = receipt.get("recertification_sha256")
        checks["receipt_self_declared_hash_matches_contract"] = (
            receipt_self_declared == entry_gate.get("recertification_self_declared_sha256")
        )
        if not checks["receipt_self_declared_hash_matches_contract"]:
            blockers.append(
                "receipt self-declared recertification_sha256 does not match the contract"
            )
    else:
        checks["receipt_file_hash_matches_contract"] = False
        checks["receipt_self_declared_hash_matches_contract"] = False
        blockers.append(
            f"recertification receipt not found: {_repo_relative(receipt_resolved, root)}"
        )
    receipt_records = receipt.get("records") if isinstance(receipt, dict) else None
    checks["receipt_records_shape"] = isinstance(receipt_records, list) and all(
        isinstance(record, dict) for record in receipt_records
    )
    if not checks["receipt_records_shape"]:
        blockers.append("receipt.records must be a list of mappings")
    normalized_receipt_records = receipt_records if checks["receipt_records_shape"] else []
    checks["receipt_record_archive_ids_shape"] = all(
        isinstance(record.get("archive_id"), str) for record in normalized_receipt_records
    )
    if not checks["receipt_record_archive_ids_shape"]:
        blockers.append("every receipt record must declare a string archive_id")
    checks["archive_hash_consistent"] = receipt.get("archive_sha256") == entry_gate.get(
        "certified_archive_sha256"
    ) and certified_archive_file_hash == entry_gate.get("certified_archive_sha256")
    if not checks["archive_hash_consistent"]:
        blockers.append(
            "receipt and certified archive file SHA-256 values must both match the contract's "
            "certified_archive_sha256"
        )

    # ---- Entry-gate eligible counts ----------------------------------------------
    eligible_by_family = _eligible_records_by_family(receipt) if receipt else {}
    receipt_eligible_total = sum(len(ids) for ids in eligible_by_family.values())
    receipt_record_count = len(normalized_receipt_records)
    checks["entry_gate_record_count"] = receipt_record_count == EXPECTED_RECORD_COUNT
    if not checks["entry_gate_record_count"]:
        blockers.append(
            f"receipt must re-certify {EXPECTED_RECORD_COUNT} records, got {receipt_record_count}"
        )
    checks["entry_gate_eligible_count"] = receipt_eligible_total == EXPECTED_ELIGIBLE_COUNT
    if not checks["entry_gate_eligible_count"]:
        blockers.append(
            f"corrected recertification must leave {EXPECTED_ELIGIBLE_COUNT} eligible "
            f"records, got {receipt_eligible_total}"
        )
    checks["entry_gate_meets_floor"] = receipt_eligible_total >= EXPECTED_ELIGIBLE_FLOOR
    if not checks["entry_gate_meets_floor"]:
        blockers.append(
            "entry gate requires at least two eligible candidates; stop the promotion sequence"
        )
    checks["entry_gate_satisfied"] = entry_gate.get("entry_gate_satisfied") is True and (
        receipt_eligible_total >= EXPECTED_ELIGIBLE_FLOOR
    )
    if not checks["entry_gate_satisfied"]:
        blockers.append("entry_gate.entry_gate_satisfied must be true with >= 2 eligible records")
    checks["entry_gate_bindings_frozen"] = all(
        entry_gate.get(field) == expected
        for field, expected in EXPECTED_ENTRY_GATE_BINDINGS.items()
    )
    if not checks["entry_gate_bindings_frozen"]:
        blockers.append(
            "entry_gate must retain the merged #6139 blocker, receipt/archive paths, and "
            "the frozen 17-record/8-eligible/2-floor counts"
        )

    # ---- Target and neutral reference planner configs exist ----------------------
    target = (
        contract.get("target_planner") if isinstance(contract.get("target_planner"), dict) else {}
    )
    target_cfg = _resolve(root, target.get("config_path"))
    checks["target_planner_name"] = (
        target.get("name") == "scenario_adaptive_hybrid_orca_v2_collision_guard"
    )
    if not checks["target_planner_name"]:
        blockers.append(
            "target_planner.name must be scenario_adaptive_hybrid_orca_v2_collision_guard"
        )
    checks["target_planner_config_exists"] = bool(target_cfg and target_cfg.is_file())
    if not checks["target_planner_config_exists"]:
        blockers.append("target_planner.config_path must point at an existing planner config")
    checks["target_planner_config_frozen"] = (
        target.get("config_path") == EXPECTED_PROVENANCE_PATHS["target_planner_config"]
        and target.get("base_algo") == "hybrid_rule_local_planner"
    )
    if not checks["target_planner_config_frozen"]:
        blockers.append(
            "target planner must keep the frozen hybrid_rule_local_planner config binding"
        )

    neutral = (
        contract.get("neutral_reference_planner")
        if isinstance(contract.get("neutral_reference_planner"), dict)
        else {}
    )
    neutral_cfg = _resolve(root, neutral.get("config_path"))
    checks["neutral_reference_config_exists"] = bool(neutral_cfg and neutral_cfg.is_file())
    if not checks["neutral_reference_config_exists"]:
        blockers.append("neutral_reference_planner.config_path must point at an existing config")
    checks["neutral_reference_not_target"] = (
        neutral.get("name") == "scenario_adaptive_orca_v1"
        and neutral.get("name") != target.get("name")
        and neutral.get("config_path")
        == EXPECTED_PROVENANCE_PATHS["neutral_reference_planner_config"]
        and neutral.get("base_algo") == "orca"
    )
    if not checks["neutral_reference_not_target"]:
        blockers.append("neutral reference planner must keep its frozen distinct ORCA binding")

    # ---- Family split matches the receipt exactly --------------------------------
    family_split = (
        contract.get("family_split") if isinstance(contract.get("family_split"), dict) else {}
    )
    fit_family = family_split.get("fit_tuning_warm_start_family")
    fresh_family = family_split.get("fresh_outcome_family")
    checks["family_split_disjoint"] = (
        fit_family == EXPECTED_FIT_FAMILY
        and fresh_family == EXPECTED_FRESH_FAMILY
        and fit_family != fresh_family
    )
    if not checks["family_split_disjoint"]:
        blockers.append(
            "family_split must be classic_cross_trap_medium (fit) and "
            "classic_group_crossing_medium (fresh outcome), family-disjoint"
        )
    checks["family_split_inputs_frozen"] = (
        family_split.get("fit_family_config") == EXPECTED_PROVENANCE_PATHS["fit_family_config"]
        and family_split.get("fresh_outcome_family_config")
        == EXPECTED_PROVENANCE_PATHS["fresh_family_config"]
        and family_split.get("disjointness") == "family_disjoint_with_no_seed_or_archive_id_overlap"
        and family_split.get("do_not_reuse_issue_3275_outcomes") is True
    )
    if not checks["family_split_inputs_frozen"]:
        blockers.append(
            "family split must retain the frozen source configs, disjointness policy, and "
            "issue #3275 outcome exclusion"
        )

    contract_fit_ids_raw = family_split.get("fit_family_eligible_records", [])
    contract_fresh_ids_raw = family_split.get("fresh_outcome_family_eligible_records", [])
    fit_ids_shape = isinstance(contract_fit_ids_raw, list) and all(
        isinstance(record_id, str) for record_id in contract_fit_ids_raw
    )
    fresh_ids_shape = isinstance(contract_fresh_ids_raw, list) and all(
        isinstance(record_id, str) for record_id in contract_fresh_ids_raw
    )
    fit_ids_values = contract_fit_ids_raw if isinstance(contract_fit_ids_raw, list) else []
    fresh_ids_values = contract_fresh_ids_raw if isinstance(contract_fresh_ids_raw, list) else []
    contract_fit_ids = sorted(
        record_id for record_id in fit_ids_values if isinstance(record_id, str)
    )
    contract_fresh_ids = sorted(
        record_id for record_id in fresh_ids_values if isinstance(record_id, str)
    )
    receipt_fit_ids = eligible_by_family.get(EXPECTED_FIT_FAMILY, [])
    receipt_fresh_ids = eligible_by_family.get(EXPECTED_FRESH_FAMILY, [])
    checks["fit_family_eligible_ids_match_receipt"] = fit_ids_shape and (
        contract_fit_ids == receipt_fit_ids
    )
    if not checks["fit_family_eligible_ids_match_receipt"]:
        blockers.append(
            "fit_family_eligible_records must match the receipt's eligible "
            f"classic_cross_trap_medium IDs (contract={contract_fit_ids}, receipt={receipt_fit_ids})"
        )
    checks["fresh_family_eligible_ids_match_receipt"] = fresh_ids_shape and (
        contract_fresh_ids == receipt_fresh_ids
    )
    if not checks["fresh_family_eligible_ids_match_receipt"]:
        blockers.append(
            "fresh_outcome_family_eligible_records must match the receipt's eligible "
            f"classic_group_crossing_medium IDs (contract={contract_fresh_ids}, receipt={receipt_fresh_ids})"
        )

    all_eligible_ids = set(contract_fit_ids) | set(contract_fresh_ids)
    receipt_excluded_ids = {
        record["archive_id"]
        for record in normalized_receipt_records
        if isinstance(record.get("archive_id"), str)
        if isinstance(record.get("after"), dict)
        and record.get("after", {}).get("benchmark_eligibility") != "eligible"
    }
    checks["no_excluded_ids_in_eligible_sets"] = all_eligible_ids.isdisjoint(receipt_excluded_ids)
    if not checks["no_excluded_ids_in_eligible_sets"]:
        blockers.append(
            "an excluded (stress_only/knife_edge) record appears in an eligible set; "
            "excluded rows may never be discoveries, although scheduled attempts remain in "
            "the primary denominator"
        )
    checks["excluded_records_count"] = family_split.get("excluded_records_count") == (
        EXPECTED_RECORD_COUNT - EXPECTED_ELIGIBLE_COUNT
    )
    if not checks["excluded_records_count"]:
        blockers.append(
            f"excluded_records_count must be {EXPECTED_RECORD_COUNT - EXPECTED_ELIGIBLE_COUNT}"
        )

    # ---- Executable input provenance ---------------------------------------------
    _check_input_provenance(
        contract.get("input_provenance"),
        repo_root=root,
        checks=checks,
        blockers=blockers,
        metadata=metadata,
    )

    # ---- Controls ----------------------------------------------------------------
    controls = contract.get("controls") if isinstance(contract.get("controls"), dict) else {}
    rejection_controls = controls.get("rejection_controls", [])
    if not isinstance(rejection_controls, list):
        rejection_controls = []
    doorway_seed_sets = [
        tuple(_as_int_tuple(item.get("seeds")))
        for item in rejection_controls
        if isinstance(item, dict) and item.get("family") == "doorway"
    ]
    checks["doorway_rejection_seeds"] = EXPECTED_DOORWAY_SEEDS in doorway_seed_sets
    if not checks["doorway_rejection_seeds"]:
        blockers.append(f"doorway rejection-control seeds must be {list(EXPECTED_DOORWAY_SEEDS)}")
    negative_control = controls.get("certifier_negative_control")
    checks["certifier_negative_control"] = (
        isinstance(negative_control, dict)
        and negative_control.get("family") == EXPECTED_NEGATIVE_CONTROL_FAMILY
    )
    if not checks["certifier_negative_control"]:
        blockers.append(
            "certifier_negative_control must be francis2023_blind_corner "
            "(certifier negative control only, never a candidate/denominator)"
        )
    checks["controls_frozen"] = (
        rejection_controls == EXPECTED_REJECTION_CONTROLS
        and negative_control == EXPECTED_NEGATIVE_CONTROL
    )
    if not checks["controls_frozen"]:
        blockers.append(
            "rejection and certifier-negative controls must retain their frozen config, map, "
            "seed, and exclusion-role bindings"
        )

    # ---- Methods, budget, simulator-time cap -------------------------------------
    methods = contract.get("methods") if isinstance(contract.get("methods"), dict) else {}
    method_entries = methods.get("entries", [])
    if not isinstance(method_entries, list):
        method_entries = []
    method_names = tuple(
        entry.get("name")
        for entry in method_entries
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    )
    checks["methods_exactly_optuna_and_random"] = method_names == EXPECTED_METHODS
    if not checks["methods_exactly_optuna_and_random"]:
        blockers.append(
            f"methods must be exactly {list(EXPECTED_METHODS)}, got {list(method_names)}"
        )
    checks["warm_start_fit_family_only"] = methods.get("warm_start_source") in (
        "fit_family_eligible_records_only",
    )
    if not checks["warm_start_fit_family_only"]:
        blockers.append("warm_start must come from fit-family eligible records only")
    checks["methods_and_warm_start_frozen"] = (
        methods.get("allowed_methods_only") == list(EXPECTED_METHODS)
        and methods.get("identical_warm_start_for_both_methods") is True
    )
    if not checks["methods_and_warm_start_frozen"]:
        blockers.append("both frozen sampler arms must retain the same fit-family warm starts")
    checks["method_entries_frozen"] = method_entries == EXPECTED_METHOD_ENTRIES
    if not checks["method_entries_frozen"]:
        blockers.append(
            "method entries must retain the existing Optuna/TPE and random sampler classes "
            "and shared builder"
        )

    budget = contract.get("budget") if isinstance(contract.get("budget"), dict) else {}
    checks["candidate_budget_64_per_seed"] = (
        budget.get("candidate_budget_per_search_seed_per_method") == EXPECTED_CANDIDATE_BUDGET
    )
    if not checks["candidate_budget_64_per_seed"]:
        blockers.append("candidate budget must be exactly 64 per search seed per method")
    search_seeds = _as_int_tuple(budget.get("search_seeds"))
    checks["search_seeds_exactly_three"] = (
        budget.get("search_seeds_per_method") == EXPECTED_SEEDS_PER_METHOD
        and search_seeds == EXPECTED_SEARCH_SEEDS
        and len(set(search_seeds)) == EXPECTED_SEEDS_PER_METHOD
    )
    if not checks["search_seeds_exactly_three"]:
        blockers.append(f"search seeds must be exactly {list(EXPECTED_SEARCH_SEEDS)} per method")
    checks["total_candidates_per_method_frozen"] = (
        budget.get("total_candidates_per_method") == EXPECTED_TOTAL_CANDIDATES_PER_METHOD
    )
    if not checks["total_candidates_per_method_frozen"]:
        blockers.append(
            f"total_candidates_per_method must be {EXPECTED_TOTAL_CANDIDATES_PER_METHOD}"
        )

    time_cap = (
        contract.get("simulator_time_cap")
        if isinstance(contract.get("simulator_time_cap"), dict)
        else {}
    )
    checks["simulator_time_cap_frozen"] = (
        time_cap.get("horizon_steps") == EXPECTED_HORIZON_STEPS
        and _approx_equal(time_cap.get("dt_s"), EXPECTED_DT_S)
        and _approx_equal(time_cap.get("simulator_time_cap_s"), EXPECTED_TIME_CAP_S)
    )
    if not checks["simulator_time_cap_frozen"]:
        blockers.append("simulator_time_cap must be horizon=100, dt=0.1, simulator_time_cap_s=10.0")

    # ---- Candidate space and feasibility ------------------------------------------
    candidate_space = (
        contract.get("candidate_space_and_feasibility")
        if isinstance(contract.get("candidate_space_and_feasibility"), dict)
        else {}
    )
    checks["candidate_space_and_feasibility_frozen"] = (
        candidate_space.get("search_space_schema") == "adversarial-search-space.v1"
        and candidate_space.get("candidate_space_structure_ref")
        == EXPECTED_PROVENANCE_PATHS["search_space"]
        and candidate_space.get("identical_for_both_methods") is True
        and candidate_space.get("feasibility_rules") == EXPECTED_FEASIBILITY_RULES
        and candidate_space.get("require_certification") is True
        and candidate_space.get("certifier") == "corrected_scenario_cert_v1_from_merged_issue_6139"
        and candidate_space.get("normalization")
        == "canonical_json_sorted_key_sha256_per_candidate_before_accounting"
        and candidate_space.get("duplicate_handling")
        == "global_within_arm_normalized_config_hash_for_unique_candidate_endpoint"
        and candidate_space.get(
            "identical_candidate_space_feasibility_and_simulator_time_cap_for_both_methods"
        )
        is True
        and candidate_space.get("duplicate_accounting")
        == (
            "Every scheduled attempt remains in its method's intention-to-search denominator. "
            "Identical normalized configurations are collapsed only for the secondary "
            "unique-candidate endpoint, globally within an arm across all three search seeds; "
            "they are never silently removed from the primary denominator."
            "\n"
        )
    )
    if not checks["candidate_space_and_feasibility_frozen"]:
        blockers.append(
            "candidate_space_and_feasibility must retain the frozen search-space, certification, "
            "normalization, duplicate, and matched-feasibility bindings"
        )

    # ---- Objective ordering -------------------------------------------------------
    objective = contract.get("objective") if isinstance(contract.get("objective"), dict) else {}
    objective_tiers = objective.get("tiers", [])
    if not isinstance(objective_tiers, list):
        objective_tiers = []
    tier_names = [
        tier.get("name")
        for tier in objective_tiers
        if isinstance(tier, dict) and isinstance(tier.get("name"), str)
    ]
    checks["objective_constraints_first"] = tier_names == [
        "collision_or_severe_intrusion",
        "liveness_or_goal_completion",
        "comfort_and_efficiency",
    ]
    if not checks["objective_constraints_first"]:
        blockers.append(
            "objective tiers must be constraints-first: "
            "collision_or_severe_intrusion, liveness_or_goal_completion, comfort_and_efficiency"
        )
    tiers = objective.get("tiers") if isinstance(objective.get("tiers"), list) else []
    checks["objective_hard_constraint_veto"] = (
        objective.get("ordering") == "constraints_first_lexicographic"
        and len(tiers) == 3
        and all(isinstance(tier, dict) for tier in tiers)
        and tiers[0].get("kind") == "hard_constraint"
        and tiers[0].get("veto") is True
        and tiers[1].get("kind") == "liveness"
        and tiers[1].get("no_soft_compensation_when_tier_1_fails") is True
        and tiers[2].get("kind") == "soft"
        and tiers[2].get("compensates_hard_constraint_or_zero_goal_completion") is False
    )
    if not checks["objective_hard_constraint_veto"]:
        blockers.append(
            "objective must retain the constraints-first ordering, tier-1 hard veto, and "
            "no soft compensation for failed constraints or liveness"
        )
    checks["objective_definition_frozen"] = (
        objective.get("name") == "constraints_first_lexicographic_v1"
        and objective.get("ordering") == "constraints_first_lexicographic"
        and objective.get("tiers") == EXPECTED_OBJECTIVE_TIERS
        and objective.get("scalar_diagnostic_only") == "worst_case_snqi"
        and objective.get("rule")
        == (
            "no_weighted_comfort_or_snqi_improvement_compensates_for_collision_or_zero_goal_completion"
        )
    )
    if not checks["objective_definition_frozen"]:
        blockers.append(
            "the constraints-first objective name, tiers, scalar diagnostic, and "
            "no-compensation rule must remain frozen"
        )
    objective_registry_path = root / EXPECTED_PROVENANCE_PATHS["objective_registry"]
    checks["objective_runner_registered"] = (
        objective.get("name") == "constraints_first_lexicographic_v1"
        and "constraints_first_lexicographic_v1"
        in _static_dict_keys(objective_registry_path, assignment_name="_OBJECTIVES")
    )
    if not checks["objective_runner_registered"]:
        blockers.append(
            "constraints_first_lexicographic_v1 must be statically registered in "
            "robot_sf/adversarial/objectives.py"
        )

    # ---- Counted weak-point gates -------------------------------------------------
    gates_block = (
        contract.get("counted_weak_point_gates")
        if isinstance(contract.get("counted_weak_point_gates"), dict)
        else {}
    )
    gates = gates_block.get("gates")
    gate_entries = gates if isinstance(gates, list) else []
    gate_ids = sorted(
        gate.get("id")
        for gate in gate_entries
        if isinstance(gate, dict) and isinstance(gate.get("id"), int)
    )
    checks["counted_weak_point_gates_all_seven"] = gate_ids == list(
        range(1, EXPECTED_COUNTED_GATE_COUNT + 1)
    )
    if not checks["counted_weak_point_gates_all_seven"]:
        blockers.append(
            f"counted_weak_point_gates must include all {EXPECTED_COUNTED_GATE_COUNT} gates, "
            f"got ids {gate_ids}"
        )
    checks["gates_fail_closed"] = gates_block.get("fail_closed") is True
    if not checks["gates_fail_closed"]:
        blockers.append("counted_weak_point_gates.fail_closed must be true")
    confirmation = (
        gates_block.get("confirmation") if isinstance(gates_block.get("confirmation"), dict) else {}
    )
    checks["confirmation_no_retries"] = confirmation.get("no_retries") is True
    if not checks["confirmation_no_retries"]:
        blockers.append("confirmation.no_retries must be true")
    gate_names = tuple(
        gate.get("name")
        for gate in gate_entries
        if isinstance(gate, dict) and isinstance(gate.get("name"), str)
    )
    checks["counted_weak_point_gate_semantics_frozen"] = (
        isinstance(gates, list)
        and len(gates) == EXPECTED_COUNTED_GATE_COUNT
        and gate_names == EXPECTED_COUNTED_GATE_NAMES
        and gate_entries == EXPECTED_COUNTED_GATE_ENTRIES
        and confirmation.get("fresh_confirmation_seeds") == 5
        and confirmation.get("seeds_distinct_from_search_seeds") is True
        and confirmation.get("mechanism_threshold_seeds") == "4_of_5"
        and confirmation.get("neutral_reference_threshold_seeds") == "4_of_5"
        and confirmation.get("second_recorded_execution_context_required") is True
        and gates_block.get("excluded_row_classes_never_discoveries_but_remain_primary_denominator")
        == list(EXPECTED_EXCLUDED_ROW_CLASSES)
    )
    if not checks["counted_weak_point_gate_semantics_frozen"]:
        blockers.append(
            "counted weak-point gates must retain their frozen names, 4-of-5 confirmation "
            "thresholds, second-context requirement, and excluded-row classes"
        )

    # ---- Estimand / uncertainty / null tests -------------------------------------
    estimand = contract.get("estimand") if isinstance(contract.get("estimand"), dict) else {}
    checks["estimand_frozen"] = (
        estimand.get("clustering") == "candidate_level_clustering_across_search_seeds"
        and estimand.get("independent_unit") == "search_seed"
    )
    if not checks["estimand_frozen"]:
        blockers.append(
            "estimand must cluster candidates across seeds (independent unit = search seed)"
        )
    checks["intention_to_search_primary_denominator"] = (
        estimand.get("denominator") == "intention_to_search_192_scheduled_attempts_per_method"
        and estimand.get("primary_denominator_policy")
        == "all_scheduled_attempts_including_missing_invalid_and_attrition"
        and estimand.get("unique_endpoint_deduplication")
        == "global_within_arm_normalized_config_hash_across_all_search_seeds"
    )
    if not checks["intention_to_search_primary_denominator"]:
        blockers.append(
            "the primary estimand must retain all 192 scheduled attempts per arm and deduplicate "
            "normalized candidates globally within an arm only for the unique endpoint"
        )
    checks["estimand_definition_frozen"] = (
        estimand.get("primary")
        == "tpe_minus_random_difference_in_unique_fully_admitted_weak_points"
        and estimand.get("unit") == "unique_fully_admitted_weak_point"
        and estimand.get("counted_only_when_all_seven_gates_pass") is True
    )
    if not checks["estimand_definition_frozen"]:
        blockers.append(
            "the estimand must retain its TPE-minus-random endpoint, unit, and all-seven-gates "
            "admission rule"
        )

    uncertainty = (
        contract.get("uncertainty") if isinstance(contract.get("uncertainty"), dict) else {}
    )
    checks["uncertainty_seed_clustered"] = (
        uncertainty.get("cluster_unit") == "search_seed"
        and uncertainty.get("clusters_per_method") == EXPECTED_SEEDS_PER_METHOD
    )
    if not checks["uncertainty_seed_clustered"]:
        blockers.append("uncertainty must bootstrap over search-seed clusters (3 per method)")
    checks["uncertainty_definition_frozen"] = (
        uncertainty.get("method") == "nonparametric_bootstrap_over_seed_clusters"
        and uncertainty.get("cluster_unit") == "search_seed"
        and uncertainty.get("clusters_per_method") == EXPECTED_SEEDS_PER_METHOD
        and uncertainty.get("interval") == "95_percent_percentile_bootstrap"
        and _approx_equal(uncertainty.get("confidence_level"), 0.95)
        and uncertainty.get("resamples") == 10000
    )
    if not checks["uncertainty_definition_frozen"]:
        blockers.append(
            "uncertainty must retain the 10,000-resample 95% percentile bootstrap over "
            "three search-seed clusters"
        )

    null_tests = contract.get("null_tests") if isinstance(contract.get("null_tests"), dict) else {}
    null_test_entries = null_tests.get("tests")
    null_test_names = (
        tuple(
            test.get("name")
            for test in null_test_entries
            if isinstance(test, dict) and isinstance(test.get("name"), str)
        )
        if isinstance(null_test_entries, list)
        else ()
    )
    checks["null_tests_two_seed_permutations"] = (
        isinstance(null_test_entries, list)
        and len(null_test_entries) == EXPECTED_NULL_TEST_COUNT
        and null_test_names == EXPECTED_NULL_TEST_NAMES
        and all(
            isinstance(test, dict) and test.get("unit") == "search_seed"
            for test in null_test_entries
        )
    )
    if not checks["null_tests_two_seed_permutations"]:
        blockers.append(
            "null_tests must be exactly the shuffled-outcome and ranking seed permutations "
            "(unit = search_seed)"
        )
    checks["null_tests_two_sided_threshold"] = null_tests.get(
        "sidedness"
    ) == "two_sided" and _approx_equal(null_tests.get("threshold_p"), NULL_THRESHOLD)
    if not checks["null_tests_two_sided_threshold"]:
        blockers.append("null_tests must be two-sided at p <= 0.05")
    checks["null_tests_both_required"] = null_tests.get("both_required") is True
    if not checks["null_tests_both_required"]:
        blockers.append("both preregistered seed-permutation null tests must be required")

    attrition = (
        contract.get("missing_invalid_attrition")
        if isinstance(contract.get("missing_invalid_attrition"), dict)
        else {}
    )
    checks["missing_invalid_stay_primary_denominator"] = (
        attrition.get("handling") == "fail_closed"
        and attrition.get("included_in_primary_denominator") is True
        and attrition.get("excluded_from_primary_denominator") is False
        and attrition.get("complete_case_analysis") == "secondary_sensitivity_only_never_primary"
    )
    if not checks["missing_invalid_stay_primary_denominator"]:
        blockers.append(
            "missing/invalid/attrition rows must remain in the primary intention-to-search "
            "denominator; complete-case analysis is sensitivity-only"
        )
    checks["missing_invalid_attrition_policy_frozen"] = (
        attrition.get("reason_recorded") is True
        and attrition.get("no_optional_seeds") is True
        and attrition.get("no_outcome_dependent_replacement_or_exclusion") is True
    )
    if not checks["missing_invalid_attrition_policy_frozen"]:
        blockers.append(
            "missing/invalid attrition must record a reason and forbid optional seeds or "
            "outcome-dependent replacement/exclusion"
        )

    outcome_schema = (
        contract.get("outcome_row_schema")
        if isinstance(contract.get("outcome_row_schema"), dict)
        else {}
    )
    schema_fields = outcome_schema.get("required_fields")
    checks["outcome_row_schema_complete"] = (
        outcome_schema.get("schema_version") == OUTCOME_ROW_SCHEMA_VERSION
        and isinstance(schema_fields, list)
        and all(isinstance(field, str) for field in schema_fields)
        and len(schema_fields) == len(set(schema_fields))
        and set(schema_fields) == EXPECTED_OUTCOME_ROW_FIELDS
    )
    if not checks["outcome_row_schema_complete"]:
        blockers.append(
            "outcome_row_schema must exactly declare the complete frozen per-attempt record fields"
        )

    mii = (
        contract.get("minimally_important_improvement")
        if isinstance(contract.get("minimally_important_improvement"), dict)
        else {}
    )
    checks["minimally_important_improvement_frozen"] = (
        mii.get("unit") == "additional_unique_fully_admitted_weak_point" and mii.get("value") == 1
    )
    if not checks["minimally_important_improvement_frozen"]:
        blockers.append("minimally_important_improvement must be frozen at one unique weak point")

    decision = (
        contract.get("decision_rule") if isinstance(contract.get("decision_rule"), dict) else {}
    )
    decision_outcomes = decision.get("outcomes", [])
    decision_outcomes_shape = isinstance(decision_outcomes, list) and all(
        isinstance(outcome, str) for outcome in decision_outcomes
    )
    decision_outcome_values = decision_outcomes if decision_outcomes_shape else []
    checks["decision_rule_three_outcomes"] = (
        decision_outcomes_shape
        and sorted(decision_outcome_values)
        == [
            "inconclusive",
            "promote",
            "stop",
        ]
        and decision.get("promote_requires_all_positive_gate_conditions") is True
    )
    if not checks["decision_rule_three_outcomes"]:
        blockers.append(
            "decision_rule must retain promote | stop | inconclusive and require every "
            "positive-gate condition before promotion"
        )

    # ---- Positive gate kept frozen (NOT weakened) --------------------------------
    positive_gate = (
        contract.get("positive_gate") if isinstance(contract.get("positive_gate"), dict) else {}
    )
    checks["positive_gate_thresholds_kept"] = (
        positive_gate.get("admitted_weak_points_floor") == 2
        and positive_gate.get("tpe_minus_random_difference") == "positive"
        and positive_gate.get("ci_95_excludes_zero") is True
        and positive_gate.get("both_null_tests_p_le_0_05") is True
    )
    if not checks["positive_gate_thresholds_kept"]:
        blockers.append("positive_gate thresholds must be kept as proposed (not weakened)")
    checks["positive_gate_not_weakened"] = positive_gate.get("thresholds_weakened") is False
    if not checks["positive_gate_not_weakened"]:
        blockers.append("positive_gate.thresholds_weakened must be false")

    # ---- Power analysis recomputed and cross-checked -----------------------------
    power = (
        contract.get("power_analysis") if isinstance(contract.get("power_analysis"), dict) else {}
    )
    seeds_per_method = EXPECTED_SEEDS_PER_METHOD
    arrangements, min_one_sided, min_two_sided = _min_permutation_p_values(seeds_per_method)
    metadata["power_analysis"] = {
        "permutation_arrangements": arrangements,
        "min_one_sided_permutation_p": min_one_sided,
        "min_two_sided_permutation_p": min_two_sided,
        "null_test_threshold": NULL_THRESHOLD,
    }
    checks["power_arrangements_C_6_3"] = (
        power.get("permutation_arrangements_C_6_3") == arrangements == 20
    )
    if not checks["power_arrangements_C_6_3"]:
        blockers.append(
            f"power_analysis.permutation_arrangements_C_6_3 must equal C(6,3)={arrangements}"
        )
    checks["power_min_two_sided_p"] = _approx_equal(
        power.get("min_two_sided_permutation_p"), min_two_sided
    ) and _approx_equal(min_two_sided, 0.10)
    if not checks["power_min_two_sided_p"]:
        blockers.append(
            f"power_analysis.min_two_sided_permutation_p must equal 2/C(6,3)={min_two_sided:.2f}"
        )
    checks["power_min_one_sided_p"] = _approx_equal(
        power.get("min_one_sided_permutation_p"), min_one_sided
    )
    if not checks["power_min_one_sided_p"]:
        blockers.append(
            f"power_analysis.min_one_sided_permutation_p must equal 1/C(6,3)={min_one_sided:.2f}"
        )
    checks["power_two_sided_cannot_reject"] = (
        power.get("two_sided_can_reject_at_threshold") is False and min_two_sided > NULL_THRESHOLD
    )
    if not checks["power_two_sided_cannot_reject"]:
        blockers.append(
            "power_analysis must record that the two-sided null cannot reject at p<=0.05 "
            f"(min two-sided p = {min_two_sided:.2f} > {NULL_THRESHOLD})"
        )
    checks["power_declaration_frozen"] = (
        power.get("n_seeds_per_method") == EXPECTED_SEEDS_PER_METHOD
        and power.get("total_labeled_seeds") == 2 * EXPECTED_SEEDS_PER_METHOD
        and _approx_equal(power.get("null_test_threshold"), NULL_THRESHOLD)
        and power.get("one_sided_only_at_single_most_extreme_arrangement") is True
        and power.get("bootstrap_ci_clusters_per_method") == EXPECTED_SEEDS_PER_METHOD
        and power.get("ci_robustly_excludes_zero") is False
    )
    if not checks["power_declaration_frozen"]:
        blockers.append(
            "power_analysis must retain the six-seed permutation framing, 0.05 threshold, "
            "three-cluster bootstrap, and non-robust CI declaration"
        )

    # ---- The honest diagnostic declaration (the crux) ----------------------------
    checks["positive_gate_not_robustly_testable"] = (
        power.get("positive_gate_robustly_testable") is False and min_two_sided > NULL_THRESHOLD
    )
    if not checks["positive_gate_not_robustly_testable"]:
        blockers.append(
            "power_analysis.positive_gate_robustly_testable must be false given the "
            "recomputed minimum two-sided permutation p exceeds 0.05"
        )

    future_run = (
        contract.get("future_run_declaration")
        if isinstance(contract.get("future_run_declaration"), dict)
        else {}
    )
    checks["future_run_diagnostic_inconclusive"] = (
        future_run.get("status") == DIAGNOSTIC_DECLARATION
    )
    if not checks["future_run_diagnostic_inconclusive"]:
        blockers.append(
            "future_run_declaration.status must be diagnostic_inconclusive because the "
            "positive gate is not robustly testable under three search seeds"
        )
    checks["diagnostic_thresholds_not_weakened"] = (
        future_run.get("thresholds_not_weakened") is True
        and positive_gate.get("thresholds_weakened") is False
    )
    if not checks["diagnostic_thresholds_not_weakened"]:
        blockers.append(
            "the diagnostic declaration must keep thresholds frozen (thresholds_not_weakened=true)"
        )
    checks["diagnostic_declared_before_outcomes"] = (
        future_run.get("declare_before_outcomes") is True
    )
    if not checks["diagnostic_declared_before_outcomes"]:
        blockers.append("the diagnostic declaration must be made before any outcomes")
    checks["re_preregistration_required_for_promote"] = (
        future_run.get("re_preregistration_required_to_claim_promote") is True
    )
    if not checks["re_preregistration_required_for_promote"]:
        blockers.append("claiming promote later requires re-preregistration with more seeds")
    diagnostic_run = future_run.get("separately_justified_diagnostic_search_run")
    checks["promotion_campaign_stopped"] = (
        future_run.get("promotion_sequence_status")
        == "stopped_before_evidence_grade_promotion_campaign"
        and future_run.get("evidence_grade_step3_authorized") is False
    )
    if not checks["promotion_campaign_stopped"]:
        blockers.append(
            "the evidence-grade promotion campaign must be explicitly stopped before a "
            "three-seed diagnostic run is authorized"
        )
    never_authorizes = (
        diagnostic_run.get("never_authorizes") if isinstance(diagnostic_run, dict) else None
    )
    checks["diagnostic_run_separately_justified"] = (
        isinstance(diagnostic_run, dict)
        and diagnostic_run.get("authorized") is True
        and diagnostic_run.get("fixed_decision") == "inconclusive"
        and diagnostic_run.get("required_exclusion_reason")
        == "diagnostic_only_no_replay_reference_or_second_context"
        and isinstance(never_authorizes, list)
        and all(isinstance(item, str) for item in never_authorizes)
        and set(never_authorizes) == {"promote", "transfer_claim", "evidence_grade_comparison"}
        and isinstance(diagnostic_run.get("stop_rule"), str)
        and bool(diagnostic_run["stop_rule"].strip())
    )
    if not checks["diagnostic_run_separately_justified"]:
        blockers.append(
            "a separately justified diagnostic run must have an explicit inconclusive-only "
            "decision, exclusion reason, non-promotion boundary, and stop rule"
        )
    checks["future_run_boundary_frozen"] = (
        future_run.get("reason")
        == "positive_gate_not_robustly_testable_under_exactly_three_search_seeds"
        and future_run.get("thresholds_not_weakened") is True
        and future_run.get("re_preregistration_required_to_claim_promote") is True
        and future_run.get("declare_before_outcomes") is True
        and future_run.get("promotion_sequence_status")
        == "stopped_before_evidence_grade_promotion_campaign"
        and future_run.get("evidence_grade_step3_authorized") is False
    )
    if not checks["future_run_boundary_frozen"]:
        blockers.append(
            "future_run_declaration must retain the pre-outcome diagnostic reason, stopped "
            "promotion sequence, and re-preregistration boundary"
        )

    # ---- Static executable diagnostic handoff ------------------------------------
    step3 = (
        contract.get("step3_execution") if isinstance(contract.get("step3_execution"), dict) else {}
    )
    runner_path = _resolve(root, step3.get("runner_ref"))
    analysis_path = _resolve(root, step3.get("analysis_ref"))
    execution_mode = _algorithm_default_execution_mode(
        root / "robot_sf/benchmark/algorithm_metadata.py", "hybrid_rule_local_planner"
    )
    checks["step3_execution_declared_diagnostic_only"] = (
        step3.get("execution_kind") == "separately_justified_diagnostic_search_stage_only"
        and step3.get("promotion_campaign_status") == "stopped"
        and step3.get("diagnostic_objective") == "constraints_first_lexicographic_v1"
        and step3.get("required_execution_mode") == EXPECTED_DIAGNOSTIC_EXECUTION_MODE
        and runner_path == root / EXPECTED_PROVENANCE_PATHS["diagnostic_runner"]
        and analysis_path == root / EXPECTED_PROVENANCE_PATHS["promotion_analysis_cli"]
    )
    if not checks["step3_execution_declared_diagnostic_only"]:
        blockers.append(
            "step3_execution must identify the runner and analysis CLI plus the frozen objective "
            "and the declared execution mode for a stopped-promotion diagnostic-only handoff"
        )
    checks["step3_execution_mode_matches_production_metadata"] = (
        execution_mode == EXPECTED_DIAGNOSTIC_EXECUTION_MODE
        and step3.get("required_execution_mode") == execution_mode
    )
    if not checks["step3_execution_mode_matches_production_metadata"]:
        blockers.append(
            "the diagnostic execution mode must match hybrid_rule_local_planner's production "
            "algorithm_metadata default_execution_mode"
        )
    checks["step3_runner_static_support"] = bool(runner_path and runner_path.is_file()) and (
        REQUIRED_DIAGNOSTIC_RUNNER_OPTIONS <= _parser_options(runner_path)
    )
    if not checks["step3_runner_static_support"]:
        blockers.append(
            "the frozen runner must statically support every required diagnostic command option"
        )
    runner_functions = _function_names(runner_path) if runner_path else set()
    checks["step3_runner_outcome_writer_support"] = {
        "build_issue_5303_search_outcome_rows",
        "write_issue_5303_search_outcome_rows",
    } <= runner_functions
    if not checks["step3_runner_outcome_writer_support"]:
        blockers.append(
            "the frozen runner must provide the issue #5303 complete outcome-row writer"
        )
    checks["step3_runner_row_schema_matches_contract"] = (
        _function_row_keys(runner_path, function_name="build_issue_5303_search_outcome_rows")
        == EXPECTED_OUTCOME_ROW_FIELDS
        if runner_path
        else False
    )
    if not checks["step3_runner_row_schema_matches_contract"]:
        blockers.append(
            "the frozen runner's static outcome-row keys must exactly match outcome_row_schema"
        )
    checks["step3_analysis_static_support"] = bool(analysis_path and analysis_path.is_file()) and (
        "analyze_issue_5303_search_promotion"
        in _function_names(root / EXPECTED_PROVENANCE_PATHS["promotion_analysis_module"])
    )
    if not checks["step3_analysis_static_support"]:
        blockers.append("the frozen analysis module must expose the diagnostic accounting analyzer")

    command_options, command_error = _command_options(step3.get("diagnostic_search_command"))
    checks["step3_diagnostic_command_entrypoint"] = _command_entrypoint_matches(
        step3.get("diagnostic_search_command"), EXPECTED_PROVENANCE_PATHS["diagnostic_runner"]
    )
    if not checks["step3_diagnostic_command_entrypoint"]:
        blockers.append(
            "the frozen diagnostic command must invoke the pinned comparison runner through "
            "uv run python"
        )
    expected_artifacts = (
        step3.get("expected_artifacts") if isinstance(step3.get("expected_artifacts"), dict) else {}
    )
    required_command_values = {
        "--contract": DEFAULT_CONTRACT_PATH.as_posix(),
        "--policy": "hybrid_rule_local_planner",
        "--algo-config": EXPECTED_PROVENANCE_PATHS["target_planner_config"],
        "--reference-algo-config": EXPECTED_PROVENANCE_PATHS["neutral_reference_planner_config"],
        "--scenario-template": EXPECTED_PROVENANCE_PATHS["diagnostic_scenario_template"],
        "--scenario-family": EXPECTED_FRESH_FAMILY,
        "--search-space": EXPECTED_PROVENANCE_PATHS["search_space"],
        "--budget": str(EXPECTED_CANDIDATE_BUDGET),
        "--objective": step3.get("diagnostic_objective"),
        "--horizon": str(EXPECTED_HORIZON_STEPS),
        "--dt": str(EXPECTED_DT_S),
        "--benchmark-profile": "experimental",
        "--execution-context-label": "diagnostic_adapter_context_a",
        "--warm-start-archive": ("docs/context/evidence/issue_5305_certified_archive/archive.json"),
        "--output-dir": "output/adversarial/issue_5303_search_promotion",
        "--out-json": "output/adversarial/issue_5303_search_promotion/report.json",
        "--out-md": "output/adversarial/issue_5303_search_promotion/comparison_table.md",
        "--outcomes-jsonl": "output/adversarial/issue_5303_search_promotion/outcomes.jsonl",
    }
    command_values_match = all(
        _single_command_value(command_options, option) == value
        for option, value in required_command_values.items()
    )
    command_matrix_match = (
        command_options.get("--sampler") == ["random", "optuna"]
        and command_options.get("--seed") == ["530301", "530302", "530303"]
        and "--require-certification" in command_options
        and "--issue-5303-diagnostic-only" in command_options
        and "--synthetic" not in command_options
        and "--empirical" not in command_options
        and command_options.get("--warm-start-record") == list(EXPECTED_WARM_START_RECORD_IDS)
    )
    artifact_paths_match = expected_artifacts == {
        "output_dir": "output/adversarial/issue_5303_search_promotion",
        "report_json": "output/adversarial/issue_5303_search_promotion/report.json",
        "comparison_table_md": "output/adversarial/issue_5303_search_promotion/comparison_table.md",
        "outcomes_jsonl": "output/adversarial/issue_5303_search_promotion/outcomes.jsonl",
        "analysis_json": "output/adversarial/issue_5303_search_promotion/analysis.json",
    }
    checks["step3_command_parses"] = command_error is None
    checks["step3_execution_command_complete"] = (
        command_error is None
        and checks["step3_diagnostic_command_entrypoint"]
        and command_values_match
        and command_matrix_match
        and artifact_paths_match
    )
    if not checks["step3_execution_command_complete"]:
        blockers.append(
            "the frozen diagnostic command must bind target/reference/configuration/certification "
            "inputs, all seeds/arms, and report/outcome artifact paths"
        )
    warm_start_archive = certified_archive_resolved
    warm_start_search_space = _resolve(root, EXPECTED_PROVENANCE_PATHS["search_space"])
    warm_start_errors = (
        _warm_start_space_errors(
            archive_path=warm_start_archive,
            record_ids=EXPECTED_WARM_START_RECORD_IDS,
            search_space_path=warm_start_search_space,
        )
        if warm_start_archive is not None and warm_start_search_space is not None
        else ["warm-start archive or search-space path is unavailable"]
    )
    metadata["warm_start_compatibility_errors"] = warm_start_errors
    checks["step3_warm_start_search_space_compatible"] = (
        checks["step3_execution_command_complete"] and not warm_start_errors
    )
    if not checks["step3_warm_start_search_space_compatible"]:
        blockers.append(
            "the frozen fit-family warm starts must exist and validate against the declared "
            "diagnostic search space: " + "; ".join(warm_start_errors)
        )
    checks["step3_warm_start_wiring"] = (
        "_load_archive_warm_starts" in runner_functions
        and "warm_start" in _function_names(runner_path)
        if runner_path
        else False
    )
    # The command and parser checks above prove the frozen IDs reach the runner; this source
    # check makes the SearchConfig handoff explicit without importing execution surfaces.
    if runner_path and runner_path.is_file():
        try:
            runner_source = runner_path.read_text(encoding="utf-8")
        except OSError:
            runner_source = ""
        checks["step3_warm_start_wiring"] = checks["step3_warm_start_wiring"] or (
            "_load_archive_warm_starts" in runner_source
            and "warm_start=warm_starts" in runner_source
        )
    if not checks["step3_warm_start_wiring"]:
        blockers.append(
            "the frozen fit-family warm-start archive IDs must be wired into SearchConfig"
        )

    analysis_options, analysis_error = _command_options(step3.get("analysis_command"))
    checks["step3_analysis_command_entrypoint"] = _command_entrypoint_matches(
        step3.get("analysis_command"), EXPECTED_PROVENANCE_PATHS["promotion_analysis_cli"]
    )
    if not checks["step3_analysis_command_entrypoint"]:
        blockers.append(
            "the frozen analysis command must invoke the pinned analysis CLI through uv run python"
        )
    checks["step3_analysis_command_complete"] = (
        analysis_error is None
        and checks["step3_analysis_command_entrypoint"]
        and _single_command_value(analysis_options, "--contract")
        == DEFAULT_CONTRACT_PATH.as_posix()
        and _single_command_value(analysis_options, "--outcomes")
        == expected_artifacts.get("outcomes_jsonl")
        and _single_command_value(analysis_options, "--output")
        == expected_artifacts.get("analysis_json")
    )
    if not checks["step3_analysis_command_complete"]:
        blockers.append(
            "the frozen analysis command must bind the contract, complete outcomes JSONL, and "
            "analysis artifact path"
        )

    # ---- Forbidden actions declared ----------------------------------------------
    forbidden = contract.get("forbidden_in_this_step", [])
    checks["forbidden_actions_declared"] = forbidden == EXPECTED_FORBIDDEN_ACTIONS
    if not checks["forbidden_actions_declared"]:
        blockers.append("forbidden_in_this_step must declare planner/Slurm/outcome-read bans")

    metadata["eligible_by_family"] = eligible_by_family
    metadata["receipt_eligible_total"] = receipt_eligible_total
    ready = not blockers
    return Issue5303PreflightResult(
        contract_path=_repo_relative(contract_path, root),
        ready=ready,
        blocked=not ready,
        checks=checks,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        metadata=metadata,
    )


def dump_preflight_payload(result: Issue5303PreflightResult, output: Path | None) -> None:
    """Write preflight payload to disk when requested."""
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
