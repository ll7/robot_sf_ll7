"""Side-effect-free preflight for the powered six-seed issue #5303 promotion contract (v2).

This checker freezes and reproduces the powered successor of the historical three-seed
#5303 search-promotion preregistration: existing Optuna/TPE versus existing random search,
six independent search seeds per method, 64 scheduled candidates per seed per method
(384 per method, 768 total), for ``scenario_adaptive_hybrid_orca_v2_collision_guard`` on
the held-out ``classic_group_crossing_medium`` family, under the merged PR #6586
runtime-effective candidate space.

What this module deliberately does NOT do
-----------------------------------------
It never executes planners, never runs a search/replay/confirmation campaign, never
submits Slurm jobs, and never reads evaluation outcomes. It only reads the frozen v2
contract config, the issue #6139 recertification receipt, the preregistration manifest,
the scheduled-identity manifest, and statically parsed handoff source files. It
recomputes SHA-256 hashes and asserts the frozen design, the exact cluster-level
inference, the outcome-free sensitivity analysis, and the machine-readable #6145
terminal result schema. It also rejects any use of the historical v1 contract for
promotion-capable execution and proves, through the existing side-effect-free #6586
preflight, that the frozen timing dimensions are runtime-effective while the historical
inert/no-pedestrian mode is rejected.

The only adversarial-package imports arrive through
:mod:`robot_sf.benchmark.issue_5303_search_promotion_preflight`, which materializes
in-memory candidate payloads and hashes them; no sampler, search loop, runtime, QD,
warm-start, transfer-matrix, campaign, replay, or benchmark-runner surface is imported.

The companion test ``tests/adversarial/test_issue_5303_search_promotion_contract_v2.py``
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
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.issue_5303_search_promotion_preflight import (
    SearchPromotionPreflightError,
    evaluate_preflight_from_files,
)

SCHEMA_VERSION = "issue_5303_search_promotion_preflight.v2"
CONTRACT_SCHEMA_VERSION = "issue_5303_search_promotion_contract.v2"
MANIFEST_SCHEMA_VERSION = "issue_5303_search_promotion_manifest.v2"
RESULT_SCHEMA_VERSION = "issue_5303_search_promotion_result.v2"
IDENTITY_SCHEMA_VERSION = "issue_5303_search_promotion_identity.v2"
IDENTITY_MANIFEST_SCHEMA_VERSION = "issue_5303_search_promotion_identity_manifest.v2"
HISTORICAL_CONTRACT_SCHEMA_VERSION = "issue_5303_search_promotion_contract.v1"

EXPECTED_ISSUE = 5303
EXPECTED_STEP = "2b"
EXPECTED_PARENT = 5303
EXPECTED_TASK_ID = "issue-5303-step-2b-powered-search-promotion-preregistration"
EXPECTED_BASE_COMMIT = "2b3e3c199f1f0d283ffeed0e0bac55710d8efccc"
EXPECTED_PEDESTRIAN_ID = "issue_5303_powered_promotion_candidate"
FULL_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

DEFAULT_CONTRACT_PATH = Path("configs/adversarial/issue_5303_search_promotion_contract_v2.yaml")
DEFAULT_RECEIPT_PATH = Path(
    "docs/context/evidence/issue_5305_certified_archive/recertification_issue_6139.json"
)
DEFAULT_MANIFEST_PATH = Path(
    "docs/context/evidence/issue_5303_search_promotion_powered_preregistration/contract_frozen.json"
)
DEFAULT_IDENTITY_MANIFEST_PATH = Path(
    "docs/context/evidence/issue_5303_search_promotion_powered_preregistration/"
    "scheduled_search_identities.json"
)
HISTORICAL_CONTRACT_PATH = Path("configs/adversarial/issue_5303_search_promotion_contract.yaml")
POWERED_SPACE_PATH = Path("configs/adversarial/issue_5303_search_promotion_space_v2.yaml")
POWERED_TEMPLATE_PATH = Path("configs/adversarial/issue_5303_classic_group_crossing_medium_v2.yaml")
HISTORICAL_SPACE_PATH = Path("configs/adversarial/issue_5303_search_promotion_space.yaml")
HISTORICAL_TEMPLATE_PATH = Path("configs/adversarial/issue_5303_classic_group_crossing_medium.yaml")

EXPECTED_ELIGIBLE_COUNT = 8
EXPECTED_RECORD_COUNT = 17
EXPECTED_ELIGIBLE_FLOOR = 2
EXPECTED_FIT_FAMILY = "classic_cross_trap_medium"
EXPECTED_FRESH_FAMILY = "classic_group_crossing_medium"
EXPECTED_METHODS: tuple[str, ...] = ("optuna", "random")
#: Order in which the frozen execution command arms the samplers and the identity manifest
#: lists methods (historical runner order retained).
EXPECTED_METHOD_COMMAND_ORDER: tuple[str, ...] = ("random", "optuna")
EXPECTED_CANDIDATE_BUDGET = 64
EXPECTED_SEEDS_PER_METHOD = 6
EXPECTED_SEARCH_SEEDS: tuple[int, ...] = (530301, 530302, 530303, 530304, 530305, 530306)
EXPECTED_TOTAL_CANDIDATES_PER_METHOD = EXPECTED_CANDIDATE_BUDGET * EXPECTED_SEEDS_PER_METHOD
EXPECTED_TOTAL_SCHEDULED_ATTEMPTS = 2 * EXPECTED_TOTAL_CANDIDATES_PER_METHOD
EXPECTED_HORIZON_STEPS = 100
EXPECTED_DT_S = 0.1
EXPECTED_TIME_CAP_S = 10.0
EXPECTED_DOORWAY_SEEDS: tuple[int, ...] = (128, 130)
EXPECTED_WARM_START_RECORD_IDS: tuple[str, ...] = (
    "issue5305_classic_cross_trap_medium_fbbd96687d61",
    "issue5305_classic_cross_trap_medium_fe24f0ff86a1",
)
EXPECTED_NEGATIVE_CONTROL_FAMILY = "francis2023_blind_corner"
EXPECTED_EXECUTION_MODE = "adapter"
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
PROMOTION_DECLARATION = "promotion_capable_preregistered"
OUTCOME_ROW_SCHEMA_VERSION = "issue_5303_search_promotion_outcome_row.v2"
DOWNSTREAM_ACTIVATION_ISSUE = 6146
DOWNSTREAM_MIN_ADMITTED_CANDIDATES = 5
PROMOTION_TIMING_STATUS_REQUIRED = "promotion_timing_ready"
HISTORICAL_INERT_STATUS_REQUIRED = "blocked_no_pedestrian"

EXPECTED_PROVENANCE_INPUT_IDS = frozenset(
    {
        "target_planner_config",
        "neutral_reference_planner_config",
        "fit_family_config",
        "fresh_family_config",
        "certification_runner",
        "objective_registry",
        "execution_runner",
        "adversarial_search_runner",
        "adversarial_bundle",
        "adversarial_config",
        "timing_preflight_module",
        "algorithm_metadata",
        "powered_search_space",
        "powered_scenario_template",
        "powered_preflight_module",
        "powered_contract_check_cli",
        "historical_contract",
        "historical_search_space",
        "historical_scenario_template",
    }
)
EXPECTED_PROVENANCE_PATHS = {
    "target_planner_config": (
        "configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_collision_guard.yaml"
    ),
    "neutral_reference_planner_config": "configs/policy_search/candidates/scenario_adaptive_orca_v1.yaml",
    "fit_family_config": "configs/scenarios/archetypes/classic_cross_trap.yaml",
    "fresh_family_config": "configs/scenarios/archetypes/classic_group_crossing.yaml",
    "certification_runner": "robot_sf/adversarial/certification.py",
    "objective_registry": "robot_sf/adversarial/objectives.py",
    "execution_runner": "scripts/tools/compare_adversarial_samplers.py",
    "adversarial_search_runner": "robot_sf/adversarial/search.py",
    "adversarial_bundle": "robot_sf/adversarial/bundle.py",
    "adversarial_config": "robot_sf/adversarial/config.py",
    "timing_preflight_module": "robot_sf/benchmark/issue_5303_search_promotion_preflight.py",
    "algorithm_metadata": "robot_sf/benchmark/algorithm_metadata.py",
    "powered_search_space": POWERED_SPACE_PATH.as_posix(),
    "powered_scenario_template": POWERED_TEMPLATE_PATH.as_posix(),
    "powered_preflight_module": (
        "robot_sf/benchmark/issue_5303_search_promotion_preregistration_v2.py"
    ),
    "powered_contract_check_cli": (
        "scripts/tools/check_issue_5303_search_promotion_contract_v2.py"
    ),
    "historical_contract": HISTORICAL_CONTRACT_PATH.as_posix(),
    "historical_search_space": HISTORICAL_SPACE_PATH.as_posix(),
    "historical_scenario_template": HISTORICAL_TEMPLATE_PATH.as_posix(),
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
EXPECTED_RUNTIME_EFFECTIVE_REJECTION_CLASSES = [
    "missing",
    "metadata_only",
    "unbound",
    "inert",
]
EXPECTED_RESULT_REQUIRED_FIELDS = [
    "schema_version",
    "decision",
    "contract_sha256",
    "execution_commit",
    "admitted_candidate_count",
    "candidate_manifest_sha256",
    "evidence_packet_sha256",
]
EXPECTED_RESULT_DECISION_VALUES = ["promote", "stop", "inconclusive"]
EXPECTED_EXECUTION_STAGE_REQUIREMENTS = [
    (
        "the runner's issue #5303 preflight hook must invoke this v2 preflight before any "
        "scheduled attempt; the historical v1 hook only authorizes the stopped v1 diagnostic"
    ),
    (
        "the analysis surface implementing the frozen decision function must hash-pin this "
        "contract and the issue_5303_search_promotion_result.v2 schema in its own execution "
        "preflight before any outcome is produced"
    ),
    (
        "execution happens only under campaign issue 6145 with separately reviewed "
        "authorization and a recorded execution_commit"
    ),
    (
        "no scheduled search seed or attempt may be added, replaced, retried, or stopped "
        "after the first outcome of either arm"
    ),
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
REQUIRED_EXECUTION_RUNNER_OPTIONS = frozenset(
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
        "--execution-context-label",
        "--warm-start-archive",
        "--warm-start-record",
    }
)


@dataclass(frozen=True)
class Issue5303PoweredPreflightResult:
    """Structured powered-promotion-contract preflight result."""

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


def _command_options(
    command: Any, *, allowed_options: frozenset[str]
) -> tuple[dict[str, list[str | None]], str | None]:
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
    if tokens[:3] != ["uv", "run", "python"] or len(tokens) < 4:
        return {}, "command must start with 'uv run python <script>'"
    options: dict[str, list[str | None]] = {}
    index = 4
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            return options, f"unexpected positional command token: {token!r}"
        if token not in allowed_options:
            return options, f"unsupported command option: {token!r}"
        if token == "--require-certification":
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


def _single_command_value(options: dict[str, list[str | None]], option: str) -> str | None:
    """Return a single required command value, or ``None`` when ambiguous/missing."""
    values = options.get(option, [])
    if len(values) != 1 or values[0] is None:
        return None
    return values[0]


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

    With ``seeds_per_method`` seeds per method, the exact permutation null relabels
    ``2 * seeds_per_method`` seed-cluster units. The number of distinct arm-label
    assignments is ``C(2n, n)``; the most extreme observed statistic is one arrangement,
    so the minimum one-sided p is ``1 / C(2n, n)`` and the minimum two-sided p is
    ``2 / C(2n, n)``.
    """
    total = 2 * seeds_per_method
    arrangements = math.comb(total, seeds_per_method)
    min_one_sided = 1.0 / arrangements
    min_two_sided = 2.0 / arrangements
    return arrangements, min_one_sided, min_two_sided


def _two_sided_rejection_region_capacity(arrangements: int, threshold: float) -> int:
    """Return the maximum number of assignments at a two-sided rejection region.

    The exact two-sided p-value is ``k / arrangements`` where ``k`` counts assignments at
    least as extreme as observed. The frozen threshold is representable by any region with
    ``k <= floor(threshold * arrangements)`` assignments.
    """
    return math.floor(threshold * arrangements)


def scheduled_search_identities() -> list[dict[str, Any]]:
    """Return the deterministic 768 scheduled search identities, outcome-free.

    The identities enumerate method x search seed x candidate index for the frozen budget.
    They depend only on frozen constants (task id, seed roster, budget) and never on any
    outcome, sampled candidate, or execution state. Each identity carries a canonical
    SHA-256 so downstream manifests can be tamper-checked without re-derivation.

    Returns:
        Exactly 768 identity records in the frozen (seed, method, index) order.
    """
    identities: list[dict[str, Any]] = []
    for search_seed in EXPECTED_SEARCH_SEEDS:
        for method in EXPECTED_METHOD_COMMAND_ORDER:
            for candidate_index in range(EXPECTED_CANDIDATE_BUDGET):
                canonical = json.dumps(
                    {
                        "schema_version": IDENTITY_SCHEMA_VERSION,
                        "task_id": EXPECTED_TASK_ID,
                        "method": method,
                        "search_seed": int(search_seed),
                        "candidate_index": int(candidate_index),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                identities.append(
                    {
                        "schema_version": IDENTITY_SCHEMA_VERSION,
                        "task_id": EXPECTED_TASK_ID,
                        "method": method,
                        "search_seed": int(search_seed),
                        "candidate_index": int(candidate_index),
                        "identity_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                    }
                )
    return identities


def identity_manifest_bytes() -> bytes:
    """Return the canonical serialized scheduled-identity manifest.

    The byte string is a pure function of the frozen constants, so the committed evidence
    file and every recomputation must agree exactly.

    Returns:
        Canonical UTF-8 JSON bytes of the 768-identity manifest.
    """
    payload = {
        "schema_version": IDENTITY_MANIFEST_SCHEMA_VERSION,
        "task_id": EXPECTED_TASK_ID,
        "scheduled_attempt_count": len(scheduled_search_identities()),
        "identities": scheduled_search_identities(),
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def identity_manifest_sha256() -> str:
    """Return the SHA-256 of the canonical scheduled-identity manifest bytes."""
    return hashlib.sha256(identity_manifest_bytes()).hexdigest()


def validate_promotion_execution_contract(payload: Any) -> list[str]:  # noqa: C901
    """Return fail-closed errors when ``payload`` is not promotion-capable.

    The historical v1 contract is a three-seed diagnostic handoff whose positive gate is
    not robustly testable; it can never authorize a promotion-capable execution. Only the
    v2 contract, with the frozen six-seed budget, exact inference, and result schema, is
    promotion-capable.

    Returns:
        Error strings; an empty list means the contract is promotion-capable.
    """
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["promotion execution contract must be a mapping"]
    schema_version = payload.get("schema_version")
    if schema_version == HISTORICAL_CONTRACT_SCHEMA_VERSION:
        errors.append(
            "the historical v1 contract is a diagnostic-only handoff and is rejected for "
            "promotion-capable execution"
        )
        return errors
    if schema_version != CONTRACT_SCHEMA_VERSION:
        errors.append(
            f"promotion-capable execution requires schema_version {CONTRACT_SCHEMA_VERSION!r}, "
            f"got {schema_version!r}"
        )
    if payload.get("task_id") != EXPECTED_TASK_ID:
        errors.append(f"promotion-capable execution requires task_id {EXPECTED_TASK_ID!r}")
    budget = payload.get("budget") if isinstance(payload.get("budget"), dict) else {}
    if budget.get("search_seeds_per_method") != EXPECTED_SEEDS_PER_METHOD:
        errors.append("promotion-capable execution requires six search seeds per method")
    if _as_int_tuple(budget.get("search_seeds")) != EXPECTED_SEARCH_SEEDS:
        errors.append("promotion-capable execution requires the frozen six-seed roster")
    if budget.get("candidate_budget_per_search_seed_per_method") != EXPECTED_CANDIDATE_BUDGET:
        errors.append("promotion-capable execution requires 64 candidates per seed per method")
    if budget.get("total_scheduled_attempts") != EXPECTED_TOTAL_SCHEDULED_ATTEMPTS:
        errors.append(
            "promotion-capable execution requires 768 scheduled attempts with complete "
            "intention-to-search accounting"
        )
    decision = (
        payload.get("decision_rule") if isinstance(payload.get("decision_rule"), dict) else {}
    )
    if sorted(
        decision.get("outcomes", []) if isinstance(decision.get("outcomes"), list) else []
    ) != [
        "inconclusive",
        "promote",
        "stop",
    ]:
        errors.append("promotion-capable execution requires the promote|stop|inconclusive rule")
    result_handoff = (
        payload.get("result_handoff") if isinstance(payload.get("result_handoff"), dict) else {}
    )
    if result_handoff.get("schema_version") != RESULT_SCHEMA_VERSION:
        errors.append(
            f"promotion-capable execution requires result schema {RESULT_SCHEMA_VERSION!r}"
        )
    future_run = (
        payload.get("future_run_declaration")
        if isinstance(payload.get("future_run_declaration"), dict)
        else {}
    )
    if future_run.get("status") != PROMOTION_DECLARATION:
        errors.append(
            "promotion-capable execution requires the promotion_capable_preregistered declaration"
        )
    return errors


def validate_terminal_result(  # noqa: C901
    payload: Any, *, expected_contract_sha256: str | None = None
) -> list[str]:
    """Return fail-closed schema errors for a #6145 terminal result payload.

    Returns:
        Error strings; an empty list means the payload satisfies the frozen
        ``issue_5303_search_promotion_result.v2`` schema.
    """
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["terminal result must be a mapping"]
    for field_name in EXPECTED_RESULT_REQUIRED_FIELDS:
        if field_name not in payload:
            errors.append(f"terminal result is missing required field {field_name!r}")
    if errors:
        return errors
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        errors.append(
            f"terminal result schema_version must be {RESULT_SCHEMA_VERSION!r}, "
            f"got {payload.get('schema_version')!r}"
        )
    decision = payload.get("decision")
    if decision not in EXPECTED_RESULT_DECISION_VALUES:
        errors.append(
            f"terminal result decision must be one of {EXPECTED_RESULT_DECISION_VALUES}, "
            f"got {decision!r}"
        )
    for hash_field in ("contract_sha256", "candidate_manifest_sha256", "evidence_packet_sha256"):
        value = payload.get(hash_field)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            errors.append(f"terminal result {hash_field} must be a 64-hex SHA-256 string")
    contract_sha256 = payload.get("contract_sha256")
    if (
        expected_contract_sha256 is not None
        and isinstance(contract_sha256, str)
        and SHA256_RE.fullmatch(contract_sha256)
        and contract_sha256 != expected_contract_sha256
    ):
        errors.append(
            "terminal result contract_sha256 does not match the frozen powered contract hash"
        )
    execution_commit = payload.get("execution_commit")
    if not isinstance(execution_commit, str) or not FULL_COMMIT_RE.fullmatch(execution_commit):
        errors.append("terminal result execution_commit must be a 40-hex git SHA")
    admitted = payload.get("admitted_candidate_count")
    if isinstance(admitted, bool) or not isinstance(admitted, int) or admitted < 0:
        errors.append("terminal result admitted_candidate_count must be a non-negative integer")
    return errors


def downstream_activation_errors(
    payload: Any, *, expected_contract_sha256: str | None = None
) -> list[str]:
    """Return fail-closed errors when a terminal result cannot activate downstream work.

    Downstream #6146 activation is valid only when the decision is ``promote``, at least
    five candidates were admitted, and the frozen result schema verifies (every referenced
    hash is well-formed and, when the frozen contract hash is supplied, matches it). Actual
    hash verification against the referenced artifacts happens at activation time; this gate
    freezes the structural conditions.

    Returns:
        Error strings; an empty list means the structural activation gates pass.
    """
    errors = validate_terminal_result(payload, expected_contract_sha256=expected_contract_sha256)
    if errors:
        return errors
    if not isinstance(payload, dict):
        return ["terminal result must be a mapping"]
    if payload.get("decision") != "promote":
        errors.append(
            f"downstream activation requires decision 'promote', got {payload.get('decision')!r}"
        )
    admitted = payload.get("admitted_candidate_count")
    if isinstance(admitted, int) and not isinstance(admitted, bool):
        if admitted < DOWNSTREAM_MIN_ADMITTED_CANDIDATES:
            errors.append(
                "downstream activation requires admitted_candidate_count >= "
                f"{DOWNSTREAM_MIN_ADMITTED_CANDIDATES}, got {admitted}"
            )
    return errors


def _warm_start_space_errors(  # noqa: C901, PLR0912, PLR0915
    *, archive_path: Path, record_ids: tuple[str, ...], search_space_path: Path
) -> list[str]:
    """Validate frozen archive warm starts against the declared powered search-space bounds.

    Mirrors the historical preflight's input checks without importing the v1 module: the
    powered space keeps identical bounds, so the same two fit-family warm starts must fit.

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
            "input provenance IDs must exactly cover the powered execution and historical "
            "boundary handoff inputs"
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


def _check_timing_runtime_effectiveness(
    *,
    root: Path,
    checks: dict[str, bool],
    blockers: list[str],
    metadata: dict[str, Any],
) -> None:
    """Prove the powered space is runtime-effective and the historical mode is rejected.

    Uses the existing side-effect-free #6586 preflight (in-memory materialization and
    hashing only); no planner, search, replay, campaign, or outcome surface is touched.
    """
    powered_status: str | None = None
    try:
        powered = evaluate_preflight_from_files(
            search_space_path=root / POWERED_SPACE_PATH,
            scenario_template_path=root / POWERED_TEMPLATE_PATH,
        )
        powered_status = powered.status
        metadata["timing_preflight_pedestrian_id"] = powered.materialized_pedestrian_id
        metadata["timing_preflight_dimensions"] = [
            {"name": probe.name, "status": probe.status} for probe in powered.dimensions
        ]
    except SearchPromotionPreflightError as exc:
        blockers.append(f"powered timing preflight could not be evaluated: {exc}")
    checks["timing_dimensions_runtime_effective"] = (
        powered_status == PROMOTION_TIMING_STATUS_REQUIRED
    )
    if not checks["timing_dimensions_runtime_effective"]:
        blockers.append(
            "the powered search space and scenario template must reach "
            f"{PROMOTION_TIMING_STATUS_REQUIRED} (spawn_time_s and pedestrian_delay_s "
            f"runtime-effective); observed status: {powered_status!r}"
        )

    historical_status: str | None = None
    try:
        historical = evaluate_preflight_from_files(
            search_space_path=root / HISTORICAL_SPACE_PATH,
            scenario_template_path=root / HISTORICAL_TEMPLATE_PATH,
        )
        historical_status = historical.status
    except SearchPromotionPreflightError as exc:
        blockers.append(f"historical timing preflight could not be evaluated: {exc}")
    checks["historical_inert_mode_rejected"] = historical_status == HISTORICAL_INERT_STATUS_REQUIRED
    if not checks["historical_inert_mode_rejected"]:
        blockers.append(
            "the historical no-pedestrian space/template pair must stay rejected as "
            f"{HISTORICAL_INERT_STATUS_REQUIRED}; observed status: {historical_status!r}"
        )


def preflight_issue_5303_powered_contract(  # noqa: C901, PLR0912, PLR0915
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    *,
    receipt_path: Path | None = None,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    repo_root: Path | None = None,
) -> Issue5303PoweredPreflightResult:
    """Validate the frozen powered promotion contract without executing anything.

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
        return Issue5303PoweredPreflightResult(
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
        return Issue5303PoweredPreflightResult(
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
        blockers.append(f"step must be {EXPECTED_STEP!r}")
    checks["contract_parent"] = contract.get("parent_issue") == EXPECTED_PARENT
    if not checks["contract_parent"]:
        blockers.append(f"parent_issue must be {EXPECTED_PARENT}")
    checks["contract_task_id"] = contract.get("task_id") == EXPECTED_TASK_ID
    if not checks["contract_task_id"]:
        blockers.append(f"task_id must be {EXPECTED_TASK_ID!r}")

    checks["evidence_boundary_proposal_only"] = (
        contract.get("evidence_boundary") == "proposal_preflight_only"
    )
    if not checks["evidence_boundary_proposal_only"]:
        blockers.append("evidence_boundary must stay proposal_preflight_only")

    base_commit = contract.get("base_commit")
    checks["base_commit_recorded"] = isinstance(base_commit, str) and bool(
        FULL_COMMIT_RE.fullmatch(base_commit)
    )
    if not checks["base_commit_recorded"]:
        blockers.append("base_commit must record the exact 40-hex green origin/main commit")
    elif base_commit != EXPECTED_BASE_COMMIT:
        warnings.append(
            f"base_commit {base_commit} differs from the originally frozen green base "
            f"{EXPECTED_BASE_COMMIT}; every input hash must have been re-frozen on it"
        )
    checks["frozen_on_recorded"] = isinstance(contract.get("frozen_on"), str) and bool(
        str(contract.get("frozen_on")).strip()
    )
    if not checks["frozen_on_recorded"]:
        blockers.append("frozen_on must record the outcome-free freeze date")

    # ---- Manifest + contract hash (reproduce the frozen contract hash) ------------
    checks["manifest_exists"] = manifest_path.is_file()
    contract_hash = sha256_file(contract_path)
    metadata["contract_file_sha256"] = contract_hash
    manifest: dict[str, Any] = {}
    if checks["manifest_exists"]:
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(manifest_payload, dict):
                manifest = manifest_payload
            else:
                blockers.append("manifest payload must be a mapping")
        except (OSError, ValueError) as exc:
            warnings.append(f"manifest JSON could not be parsed: {exc}")
        manifest_hash = manifest.get("contract_sha256")
        checks["manifest_schema_version"] = manifest.get("schema_version") == (
            MANIFEST_SCHEMA_VERSION
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
    checks["certified_archive_exists"] = (
        certified_archive_resolved is not None and certified_archive_resolved.is_file()
    )
    certified_archive_file_hash: str | None = None
    if certified_archive_resolved is not None and certified_archive_resolved.is_file():
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
        checks["certified_archive_exists"] = False
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
        entry_gate.get(field_name) == expected
        for field_name, expected in EXPECTED_ENTRY_GATE_BINDINGS.items()
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
            f"classic_group_crossing_medium IDs (contract={contract_fresh_ids}, "
            f"receipt={receipt_fresh_ids})"
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
            "rejection and certifier-negative controls must retain their frozen family, "
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
    checks["search_seeds_exactly_six"] = (
        budget.get("search_seeds_per_method") == EXPECTED_SEEDS_PER_METHOD
        and search_seeds == EXPECTED_SEARCH_SEEDS
        and len(set(search_seeds)) == EXPECTED_SEEDS_PER_METHOD
    )
    if not checks["search_seeds_exactly_six"]:
        blockers.append(f"search seeds must be exactly {list(EXPECTED_SEARCH_SEEDS)} per method")
    checks["seed_contract_outcome_independent"] = (
        budget.get("seed_derivation") == "explicit_listed_set_outcome_independent_530300_plus_index"
        and budget.get("no_post_outcome_seed_addition_replacement_retry_or_stopping") is True
    )
    if not checks["seed_contract_outcome_independent"]:
        blockers.append(
            "the seed contract must be an explicit outcome-independent listed set with no "
            "post-outcome seed addition, replacement, retry, or stopping"
        )
    checks["total_candidates_per_method_frozen"] = (
        budget.get("total_candidates_per_method") == EXPECTED_TOTAL_CANDIDATES_PER_METHOD
    )
    if not checks["total_candidates_per_method_frozen"]:
        blockers.append(
            f"total_candidates_per_method must be {EXPECTED_TOTAL_CANDIDATES_PER_METHOD}"
        )
    checks["total_scheduled_attempts_frozen"] = (
        budget.get("total_scheduled_attempts") == EXPECTED_TOTAL_SCHEDULED_ATTEMPTS
    )
    if not checks["total_scheduled_attempts_frozen"]:
        blockers.append(
            f"total_scheduled_attempts must be {EXPECTED_TOTAL_SCHEDULED_ATTEMPTS} with "
            "complete intention-to-search accounting"
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

    # ---- Candidate space, feasibility, and runtime effectiveness ------------------
    candidate_space = (
        contract.get("candidate_space_and_feasibility")
        if isinstance(contract.get("candidate_space_and_feasibility"), dict)
        else {}
    )
    checks["candidate_space_and_feasibility_frozen"] = (
        candidate_space.get("search_space_schema") == "adversarial-search-space.v1"
        and candidate_space.get("candidate_space_structure_ref")
        == EXPECTED_PROVENANCE_PATHS["powered_search_space"]
        and candidate_space.get("scenario_template_ref")
        == EXPECTED_PROVENANCE_PATHS["powered_scenario_template"]
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
            "unique-candidate endpoint, globally within an arm across all six search seeds; "
            "they are never silently removed from the primary denominator."
            "\n"
        )
    )
    if not checks["candidate_space_and_feasibility_frozen"]:
        blockers.append(
            "candidate_space_and_feasibility must retain the frozen powered search-space, "
            "scenario template, certification, normalization, duplicate, and matched-feasibility "
            "bindings"
        )
    runtime_effective = (
        candidate_space.get("runtime_effective_candidate_space")
        if isinstance(candidate_space.get("runtime_effective_candidate_space"), dict)
        else {}
    )
    checks["runtime_effective_space_declared"] = (
        runtime_effective.get("source_pr") == 6586
        and runtime_effective.get("source_issue") == 6475
        and runtime_effective.get("non_null_pedestrian_identity") is True
        and runtime_effective.get("pedestrian_route_and_single_pedestrians_populated") is True
        and runtime_effective.get("runtime_effective_dimensions")
        == ["spawn_time_s", "pedestrian_delay_s"]
        and runtime_effective.get(
            "canonical_effective_scenario_hash_excludes_provenance_only_metadata"
        )
        is True
        and runtime_effective.get("fail_closed_rejection_classes")
        == EXPECTED_RUNTIME_EFFECTIVE_REJECTION_CLASSES
        and runtime_effective.get("preflight_status_required") == PROMOTION_TIMING_STATUS_REQUIRED
    )
    if not checks["runtime_effective_space_declared"]:
        blockers.append(
            "the merged PR #6586 runtime-effective candidate-space contract must be declared "
            "with fail-closed rejection of missing/metadata-only/unbound/inert dimensions"
        )
    checks["runtime_effective_pedestrian_id_matches_space"] = (
        runtime_effective.get("pedestrian_identity") == EXPECTED_PEDESTRIAN_ID
        and _declared_space_pedestrian_id(root) == EXPECTED_PEDESTRIAN_ID
    )
    if not checks["runtime_effective_pedestrian_id_matches_space"]:
        blockers.append(
            f"the frozen pedestrian identity {EXPECTED_PEDESTRIAN_ID!r} must match the "
            "powered search-space pedestrian.id declaration"
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
        estimand.get("denominator") == "intention_to_search_384_scheduled_attempts_per_method"
        and estimand.get("primary_denominator_policy")
        == "all_scheduled_attempts_including_missing_invalid_and_attrition"
        and estimand.get("unique_endpoint_deduplication")
        == "global_within_arm_normalized_config_hash_across_all_search_seeds"
    )
    if not checks["intention_to_search_primary_denominator"]:
        blockers.append(
            "the primary estimand must retain all 384 scheduled attempts per arm and deduplicate "
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
    checks["uncertainty_exact_cluster_level"] = (
        uncertainty.get("cluster_unit") == "search_seed"
        and uncertainty.get("clusters_per_method") == EXPECTED_SEEDS_PER_METHOD
        and uncertainty.get("total_labeled_seed_units") == 2 * EXPECTED_SEEDS_PER_METHOD
    )
    if not checks["uncertainty_exact_cluster_level"]:
        blockers.append(
            "uncertainty must cluster over search seeds: six clusters per method, twelve "
            "labeled seed units total"
        )
    checks["uncertainty_definition_frozen"] = (
        uncertainty.get("method") == "exact_cluster_level_permutation_distribution"
        and uncertainty.get("arm_label_assignments_C_12_6") == 924
        and uncertainty.get("interval")
        == "exact_95_percent_cluster_level_interval_over_all_assignments"
        and _approx_equal(uncertainty.get("confidence_level"), 0.95)
        and uncertainty.get("secondary_diagnostic") == "nonparametric_bootstrap_over_seed_clusters"
        and uncertainty.get("secondary_diagnostic_resamples") == 10000
    )
    if not checks["uncertainty_definition_frozen"]:
        blockers.append(
            "uncertainty must retain the exact 924-assignment cluster-level interval with the "
            "10,000-resample bootstrap as secondary diagnostic only"
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
    checks["null_tests_exact_enumeration"] = (
        null_tests.get("exact_enumeration") == "all_924_arm_label_assignments"
    )
    if not checks["null_tests_exact_enumeration"]:
        blockers.append("null_tests must enumerate all C(12,6)=924 arm-label assignments")

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
        and all(isinstance(field_name, str) for field_name in schema_fields)
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
    checks["decision_rule_exactly_one_function"] = (
        decision.get("exactly_one_decision_function") is True
    )
    if not checks["decision_rule_exactly_one_function"]:
        blockers.append("exactly one promote|stop|inconclusive decision function must be frozen")

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

    # ---- Power/sensitivity analysis recomputed and cross-checked ------------------
    power = (
        contract.get("power_analysis") if isinstance(contract.get("power_analysis"), dict) else {}
    )
    arrangements, min_one_sided, min_two_sided = _min_permutation_p_values(
        EXPECTED_SEEDS_PER_METHOD
    )
    rejection_capacity = _two_sided_rejection_region_capacity(arrangements, NULL_THRESHOLD)
    metadata["power_analysis"] = {
        "permutation_arrangements": arrangements,
        "min_one_sided_permutation_p": min_one_sided,
        "min_two_sided_permutation_p": min_two_sided,
        "null_test_threshold": NULL_THRESHOLD,
        "two_sided_rejection_region_max_arrangements": rejection_capacity,
    }
    checks["power_arrangements_C_12_6"] = (
        power.get("permutation_arrangements_C_12_6") == arrangements == 924
    )
    if not checks["power_arrangements_C_12_6"]:
        blockers.append(
            f"power_analysis.permutation_arrangements_C_12_6 must equal C(12,6)={arrangements}"
        )
    checks["power_min_two_sided_p"] = _approx_equal(
        power.get("min_two_sided_permutation_p"), min_two_sided
    )
    if not checks["power_min_two_sided_p"]:
        blockers.append(
            f"power_analysis.min_two_sided_permutation_p must equal 2/C(12,6)={min_two_sided}"
        )
    checks["power_min_one_sided_p"] = _approx_equal(
        power.get("min_one_sided_permutation_p"), min_one_sided
    )
    if not checks["power_min_one_sided_p"]:
        blockers.append(
            f"power_analysis.min_one_sided_permutation_p must equal 1/C(12,6)={min_one_sided}"
        )
    checks["power_two_sided_can_reject"] = (
        power.get("two_sided_can_reject_at_threshold") is True and min_two_sided <= NULL_THRESHOLD
    )
    if not checks["power_two_sided_can_reject"]:
        blockers.append(
            "power_analysis must record that the two-sided null can reject at p<=0.05 "
            f"(min two-sided p = {min_two_sided} <= {NULL_THRESHOLD})"
        )
    checks["power_rejection_region_capacity"] = (
        power.get("two_sided_rejection_region_max_arrangements") == rejection_capacity == 46
    )
    if not checks["power_rejection_region_capacity"]:
        blockers.append(
            "power_analysis.two_sided_rejection_region_max_arrangements must equal "
            f"floor(0.05 * 924) = {rejection_capacity}"
        )
    checks["power_declaration_frozen"] = (
        power.get("n_seeds_per_method") == EXPECTED_SEEDS_PER_METHOD
        and power.get("total_labeled_seeds") == 2 * EXPECTED_SEEDS_PER_METHOD
        and _approx_equal(power.get("null_test_threshold"), NULL_THRESHOLD)
        and power.get("bootstrap_ci_clusters_per_method") == EXPECTED_SEEDS_PER_METHOD
    )
    if not checks["power_declaration_frozen"]:
        blockers.append(
            "power_analysis must retain the twelve-seed permutation framing, 0.05 threshold, "
            "and six-cluster secondary bootstrap declaration"
        )
    distinction = (
        power.get("attainable_significance_vs_power")
        if isinstance(power.get("attainable_significance_vs_power"), dict)
        else {}
    )
    checks["power_significance_vs_power_distinction"] = distinction.get(
        "attainable_significance"
    ) == (
        "the exact enumeration of C(12,6)=924 arm-label assignments can represent the "
        "two-sided p<=0.05 boundary; the minimum attainable two-sided p is 2/924"
    ) and distinction.get("power_or_sensitivity") == (
        "no outcome-free power claim is made against an unspecified alternative; "
        "sensitivity is characterized by the frozen rejection-region boundary (at most 46 "
        "of 924 assignments at least as extreme as the observed statistic) and the "
        "minimally important improvement of one unique fully admitted weak point"
    )
    if not checks["power_significance_vs_power_distinction"]:
        blockers.append(
            "power_analysis must distinguish attainable significance from power/sensitivity "
            "with the frozen boundary statements"
        )

    # ---- Promotion-capable declaration (the crux) ---------------------------------
    future_run = (
        contract.get("future_run_declaration")
        if isinstance(contract.get("future_run_declaration"), dict)
        else {}
    )
    checks["future_run_promotion_capable_preregistered"] = (
        future_run.get("status") == PROMOTION_DECLARATION
    )
    if not checks["future_run_promotion_capable_preregistered"]:
        blockers.append(
            f"future_run_declaration.status must be {PROMOTION_DECLARATION!r} now that the "
            "exact six-seed inference can represent the two-sided decision boundary"
        )
    checks["future_run_execution_not_authorized_here"] = (
        future_run.get("execution_authorized_here") is False
        and future_run.get("promotion_campaign_issue") == 6145
    )
    if not checks["future_run_execution_not_authorized_here"]:
        blockers.append(
            "this preregistration does not authorize #6145 execution; the campaign remains "
            "separately authorized"
        )
    checks["future_run_thresholds_not_weakened"] = (
        future_run.get("thresholds_not_weakened") is True
        and positive_gate.get("thresholds_weakened") is False
    )
    if not checks["future_run_thresholds_not_weakened"]:
        blockers.append("the powered declaration must keep thresholds frozen")
    checks["future_run_declared_before_outcomes"] = (
        future_run.get("declare_before_outcomes") is True
    )
    if not checks["future_run_declared_before_outcomes"]:
        blockers.append("the powered declaration must be made before any outcomes")
    checks["future_run_no_transfer_before_promote"] = (
        future_run.get("no_transfer_before_hash_bound_promote") is True
    )
    if not checks["future_run_no_transfer_before_promote"]:
        blockers.append("no transfer work is permitted before a hash-bound promote result")

    # ---- Machine-readable terminal result schema ----------------------------------
    result_handoff = (
        contract.get("result_handoff") if isinstance(contract.get("result_handoff"), dict) else {}
    )
    checks["result_schema_frozen"] = (
        result_handoff.get("schema_version") == RESULT_SCHEMA_VERSION
        and result_handoff.get("required_fields") == EXPECTED_RESULT_REQUIRED_FIELDS
        and result_handoff.get("decision_values") == EXPECTED_RESULT_DECISION_VALUES
        and result_handoff.get("contract_sha256_binds_this_contract") is True
    )
    if not checks["result_schema_frozen"]:
        blockers.append(
            f"result_handoff must freeze the {RESULT_SCHEMA_VERSION!r} schema binding "
            "decision, contract_sha256, execution_commit, admitted_candidate_count, "
            "candidate_manifest_sha256, and evidence_packet_sha256"
        )
    activation = (
        result_handoff.get("downstream_activation")
        if isinstance(result_handoff.get("downstream_activation"), dict)
        else {}
    )
    checks["result_downstream_activation_frozen"] = (
        activation.get("downstream_issue") == DOWNSTREAM_ACTIVATION_ISSUE
        and activation.get("requires_decision_promote") is True
        and activation.get("min_admitted_candidate_count") == DOWNSTREAM_MIN_ADMITTED_CANDIDATES
        and activation.get("all_referenced_hashes_must_verify") is True
        and activation.get("admitted_candidates_pass_frozen_eligibility_and_lineage_gates") is True
        and activation.get("activation_on_issue_closure_alone") is False
    )
    if not checks["result_downstream_activation_frozen"]:
        blockers.append(
            "downstream activation must require decision==promote, at least five admitted "
            "candidates, verified hashes, and frozen eligibility/lineage gates; issue closure "
            "alone never activates downstream work"
        )

    # ---- Static executable execution-stage handoff --------------------------------
    step3 = (
        contract.get("step3_execution") if isinstance(contract.get("step3_execution"), dict) else {}
    )
    runner_path = _resolve(root, step3.get("runner_ref"))
    checks["step3_execution_binding_declared"] = (
        step3.get("campaign_issue") == 6145
        and step3.get("execution_kind")
        == "evidence_grade_promotion_campaign_pending_separate_authorization"
        and step3.get("promotion_capability")
        == "promote_stop_or_inconclusive_per_frozen_decision_rule"
        and step3.get("required_execution_mode") == EXPECTED_EXECUTION_MODE
        and step3.get("execution_stage_requirements") == EXPECTED_EXECUTION_STAGE_REQUIREMENTS
    )
    if not checks["step3_execution_binding_declared"]:
        blockers.append(
            "step3_execution must declare the #6145 campaign binding, the frozen execution "
            "mode, and the execution-stage requirements"
        )
    execution_mode = _algorithm_default_execution_mode(
        root / EXPECTED_PROVENANCE_PATHS["algorithm_metadata"], "hybrid_rule_local_planner"
    )
    checks["step3_execution_mode_matches_production_metadata"] = (
        execution_mode == EXPECTED_EXECUTION_MODE
        and step3.get("required_execution_mode") == execution_mode
    )
    if not checks["step3_execution_mode_matches_production_metadata"]:
        blockers.append(
            "the powered execution mode must match hybrid_rule_local_planner's production "
            "algorithm_metadata default_execution_mode"
        )
    checks["step3_runner_static_support"] = bool(runner_path and runner_path.is_file()) and (
        REQUIRED_EXECUTION_RUNNER_OPTIONS <= _parser_options(runner_path)
    )
    if not checks["step3_runner_static_support"]:
        blockers.append(
            "the frozen runner must statically support every required execution command option"
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
            "the frozen runner's static outcome-row keys must exactly cover outcome_row_schema"
        )

    command_options, command_error = _command_options(
        step3.get("execution_command"),
        allowed_options=REQUIRED_EXECUTION_RUNNER_OPTIONS,
    )
    checks["step3_command_parses"] = command_error is None
    if command_error is not None:
        blockers.append(f"the frozen execution command does not parse: {command_error}")
    checks["step3_execution_command_entrypoint"] = _command_entrypoint_matches(
        step3.get("execution_command"), EXPECTED_PROVENANCE_PATHS["execution_runner"]
    )
    if not checks["step3_execution_command_entrypoint"]:
        blockers.append(
            "the frozen execution command must invoke the pinned comparison runner through "
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
        "--scenario-template": EXPECTED_PROVENANCE_PATHS["powered_scenario_template"],
        "--scenario-family": EXPECTED_FRESH_FAMILY,
        "--search-space": EXPECTED_PROVENANCE_PATHS["powered_search_space"],
        "--budget": str(EXPECTED_CANDIDATE_BUDGET),
        "--objective": "constraints_first_lexicographic_v1",
        "--horizon": str(EXPECTED_HORIZON_STEPS),
        "--dt": str(EXPECTED_DT_S),
        "--benchmark-profile": "experimental",
        "--execution-context-label": "powered_promotion_context_a",
        "--warm-start-archive": "docs/context/evidence/issue_5305_certified_archive/archive.json",
        "--output-dir": "output/adversarial/issue_5303_search_promotion_v2",
        "--out-json": "output/adversarial/issue_5303_search_promotion_v2/report.json",
        "--out-md": "output/adversarial/issue_5303_search_promotion_v2/comparison_table.md",
        "--outcomes-jsonl": "output/adversarial/issue_5303_search_promotion_v2/outcomes.jsonl",
    }
    command_values_match = all(
        _single_command_value(command_options, option) == value
        for option, value in required_command_values.items()
    )
    command_matrix_match = (
        command_options.get("--sampler") == list(EXPECTED_METHOD_COMMAND_ORDER)
        and command_options.get("--seed") == [str(seed) for seed in EXPECTED_SEARCH_SEEDS]
        and "--require-certification" in command_options
        and command_options.get("--warm-start-record") == list(EXPECTED_WARM_START_RECORD_IDS)
    )
    artifact_paths_match = expected_artifacts == {
        "output_dir": "output/adversarial/issue_5303_search_promotion_v2",
        "report_json": "output/adversarial/issue_5303_search_promotion_v2/report.json",
        "comparison_table_md": (
            "output/adversarial/issue_5303_search_promotion_v2/comparison_table.md"
        ),
        "outcomes_jsonl": "output/adversarial/issue_5303_search_promotion_v2/outcomes.jsonl",
    }
    checks["step3_execution_command_complete"] = (
        command_error is None
        and checks["step3_execution_command_entrypoint"]
        and command_values_match
        and command_matrix_match
        and artifact_paths_match
    )
    if not checks["step3_execution_command_complete"]:
        blockers.append(
            "the frozen execution command must bind target/reference/configuration/"
            "certification inputs, all six seeds for both arms, and report/outcome artifact "
            "paths without any diagnostic-only flag"
        )
    warm_start_archive = certified_archive_resolved
    warm_start_search_space = _resolve(root, EXPECTED_PROVENANCE_PATHS["powered_search_space"])
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
            "powered search space: " + "; ".join(warm_start_errors)
        )

    # ---- Historical v1 boundary ----------------------------------------------------
    historical_contract_path = root / HISTORICAL_CONTRACT_PATH
    historical_contract: dict[str, Any] = {}
    if historical_contract_path.is_file():
        try:
            historical_payload = yaml.safe_load(
                historical_contract_path.read_text(encoding="utf-8")
            )
            if isinstance(historical_payload, dict):
                historical_contract = historical_payload
            else:
                blockers.append("historical v1 contract payload must be a mapping")
        except yaml.YAMLError as exc:
            blockers.append(f"historical v1 contract YAML could not be parsed: {exc}")
    else:
        blockers.append(f"historical v1 contract not found: {HISTORICAL_CONTRACT_PATH.as_posix()}")
    historical_future_run = (
        historical_contract.get("future_run_declaration")
        if isinstance(historical_contract.get("future_run_declaration"), dict)
        else {}
    )
    checks["historical_contract_immutable_diagnostic"] = (
        historical_contract.get("schema_version") == HISTORICAL_CONTRACT_SCHEMA_VERSION
        and historical_contract.get("task_id")
        == "issue-5303-step-2-search-promotion-preregistration"
        and historical_future_run.get("status") == "diagnostic_inconclusive"
        and historical_future_run.get("thresholds_not_weakened") is True
    )
    if not checks["historical_contract_immutable_diagnostic"]:
        blockers.append(
            "the historical v1 contract must remain the unchanged diagnostic_inconclusive "
            "three-seed handoff"
        )
    v1_rejection_errors = validate_promotion_execution_contract(historical_contract)
    checks["historical_contract_rejected_for_promotion"] = bool(v1_rejection_errors) and any(
        "rejected for promotion-capable execution" in error for error in v1_rejection_errors
    )
    if not checks["historical_contract_rejected_for_promotion"]:
        blockers.append(
            "the checker must reject the historical v1 contract for promotion-capable execution"
        )
    checks["powered_contract_promotion_capable"] = not validate_promotion_execution_contract(
        contract
    )
    if not checks["powered_contract_promotion_capable"]:
        blockers.append("the v2 contract itself must satisfy the promotion-capability gate")

    # ---- Runtime-effective timing dimensions ---------------------------------------
    _check_timing_runtime_effectiveness(
        root=root, checks=checks, blockers=blockers, metadata=metadata
    )

    # ---- Scheduled search identities ------------------------------------------------
    identities = scheduled_search_identities()
    checks["scheduled_identities_complete"] = (
        len(identities) == EXPECTED_TOTAL_SCHEDULED_ATTEMPTS
        and len({identity["identity_sha256"] for identity in identities})
        == EXPECTED_TOTAL_SCHEDULED_ATTEMPTS
    )
    if not checks["scheduled_identities_complete"]:
        blockers.append(
            f"the scheduled identity manifest must contain exactly "
            f"{EXPECTED_TOTAL_SCHEDULED_ATTEMPTS} unique identities"
        )
    manifest_identity_hash = manifest.get("scheduled_identity_manifest_sha256")
    recomputed_identity_hash = identity_manifest_sha256()
    metadata["scheduled_identity_manifest_sha256"] = recomputed_identity_hash
    checks["scheduled_identity_manifest_hash_frozen"] = (
        manifest_identity_hash == recomputed_identity_hash
    )
    if not checks["scheduled_identity_manifest_hash_frozen"]:
        blockers.append(
            "the frozen manifest's scheduled_identity_manifest_sha256 does not match the "
            f"recomputed deterministic 768-identity manifest ({manifest_identity_hash!r} vs "
            f"{recomputed_identity_hash!r})"
        )
    checks["scheduled_identity_manifest_count"] = (
        manifest.get("scheduled_identity_count") == EXPECTED_TOTAL_SCHEDULED_ATTEMPTS
    )
    if not checks["scheduled_identity_manifest_count"]:
        blockers.append(
            f"the frozen manifest must record {EXPECTED_TOTAL_SCHEDULED_ATTEMPTS} scheduled "
            "identities"
        )

    # ---- Manifest cross-checks -------------------------------------------------------
    checks["manifest_base_commit"] = manifest.get("base_commit") == contract.get("base_commit")
    if not checks["manifest_base_commit"]:
        blockers.append(
            "the frozen manifest must record the same green base commit as the contract"
        )
    checks["manifest_budget_bindings"] = (
        manifest.get("search_seeds_per_method") == EXPECTED_SEEDS_PER_METHOD
        and manifest.get("candidate_budget_per_search_seed_per_method") == EXPECTED_CANDIDATE_BUDGET
        and manifest.get("total_candidates_per_method") == EXPECTED_TOTAL_CANDIDATES_PER_METHOD
        and manifest.get("total_scheduled_attempts") == EXPECTED_TOTAL_SCHEDULED_ATTEMPTS
        and manifest.get("search_seeds") == list(EXPECTED_SEARCH_SEEDS)
    )
    if not checks["manifest_budget_bindings"]:
        blockers.append("the frozen manifest must retain the six-seed 64-candidate budget bindings")
    checks["manifest_result_schema"] = manifest.get("result_schema_version") == (
        RESULT_SCHEMA_VERSION
    )
    if not checks["manifest_result_schema"]:
        blockers.append(f"the frozen manifest must record result schema {RESULT_SCHEMA_VERSION!r}")
    checks["manifest_thresholds_not_weakened"] = manifest.get("thresholds_weakened") is False
    if not checks["manifest_thresholds_not_weakened"]:
        blockers.append("the frozen manifest must record thresholds_weakened=false")
    checks["manifest_declaration"] = manifest.get("future_run_declaration") == (
        PROMOTION_DECLARATION
    )
    if not checks["manifest_declaration"]:
        blockers.append(
            f"the frozen manifest must record the {PROMOTION_DECLARATION!r} declaration"
        )

    # ---- Forbidden actions declared ----------------------------------------------
    forbidden = contract.get("forbidden_in_this_step", [])
    checks["forbidden_actions_declared"] = forbidden == EXPECTED_FORBIDDEN_ACTIONS
    if not checks["forbidden_actions_declared"]:
        blockers.append("forbidden_in_this_step must declare planner/Slurm/outcome-read bans")

    metadata["eligible_by_family"] = eligible_by_family
    metadata["receipt_eligible_total"] = receipt_eligible_total
    metadata["scheduled_identity_count"] = len(identities)
    ready = not blockers
    return Issue5303PoweredPreflightResult(
        contract_path=_repo_relative(contract_path, root),
        ready=ready,
        blocked=not ready,
        checks=checks,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        metadata=metadata,
    )


def _declared_space_pedestrian_id(root: Path) -> str | None:
    """Read the powered search-space pedestrian.id declaration without importing the loader.

    Returns:
        The declared pedestrian identity, or ``None`` when the space or declaration is absent.
    """
    space_path = root / POWERED_SPACE_PATH
    if not space_path.is_file():
        return None
    try:
        payload = yaml.safe_load(space_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    pedestrian = payload.get("pedestrian") if isinstance(payload, dict) else None
    if not isinstance(pedestrian, dict):
        return None
    declared = pedestrian.get("id")
    return str(declared).strip() or None


def dump_preflight_payload(result: Issue5303PoweredPreflightResult, output: Path | None) -> None:
    """Write preflight payload to disk when requested."""
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
