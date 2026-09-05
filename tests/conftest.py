"""Minimal pytest configuration: headless graphics + slow test timing.

Rewritten to purge legacy pytest_sessionfinish hook with invalid signature.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    import triton  # noqa: F401
except ImportError:
    pass

import pytest

from robot_sf.common.artifact_paths import ensure_canonical_tree
from robot_sf.common.seed import _set_torch_deterministic_algorithms
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition
from robot_sf.planner import ClassicGlobalPlanner, ClassicPlannerConfig
from tests.support.process_teardown import reap_matching_descendants
from tests.support.pytest_temproot import (
    clean_stale_process_roots,
    is_pid_running,
)
from tests.support.pytest_temproot import pytest_process_root as _pytest_process_root

try:
    from tests.perf_utils.policy import PerformanceBudgetPolicy
    from tests.perf_utils.reporting import SlowTestSample, format_report, generate_report
except Exception:  # pragma: no cover - perf utils optional in some contexts
    PerformanceBudgetPolicy = None  # type: ignore[assignment]
    SlowTestSample = None  # type: ignore[assignment]
    format_report = None  # type: ignore[assignment]
    generate_report = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from collections.abc import Generator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
root_str = str(PROJECT_ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


_is_pid_running = is_pid_running


def _clean_stale_proc_dirs(wt_dir: Path) -> None:
    """Clean up process temproot directories whose PID is no longer running."""
    clean_stale_process_roots(wt_dir)


def _setup_isolated_pytest_temproot() -> None:
    """Isolate pytest temporary directory root per worktree and process.

    Prevents cross-session temporary directory cleanup collisions and PytestWarning (rm_rf)
    cleanup warnings when multiple pytest sessions run concurrently.
    """
    configured_temproot = os.environ.get("PYTEST_DEBUG_TEMPROOT")
    if configured_temproot:
        _clean_stale_proc_dirs(Path(configured_temproot).parent)
        return
    for arg in sys.argv:
        if arg.startswith("--basetemp"):
            return

    temproot = _pytest_process_root(PROJECT_ROOT, os.getpid())
    _clean_stale_proc_dirs(temproot.parent)
    temproot.mkdir(parents=True, exist_ok=True)
    os.environ["PYTEST_DEBUG_TEMPROOT"] = str(temproot)


_setup_isolated_pytest_temproot()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Reap leaked nested pytest children and clean up process temproot."""
    del session, exitstatus
    enabled = (
        os.environ.get("ROBOT_SF_TEST_ENV") == "licca"
        or os.environ.get("ROBOT_SF_REAP_LEAKED_PYTEST_CHILDREN") == "1"
    )
    if enabled:
        reaped = reap_matching_descendants(command_substrings=("pytest",), grace_seconds=2.0)
        if reaped:
            print(
                "robot_sf pytest teardown reaped leaked child processes: "
                + ", ".join(str(pid) for pid in reaped),
                file=sys.stderr,
            )

    temproot_env = os.environ.get("PYTEST_DEBUG_TEMPROOT")
    if temproot_env and Path(temproot_env) == _pytest_process_root(PROJECT_ROOT, os.getpid()):
        temproot_path = Path(temproot_env)
        if temproot_path.exists():
            shutil.rmtree(temproot_path, ignore_errors=True)


def _import_torch_optional():
    """TODO docstring. Document this function."""
    try:
        return importlib.import_module("torch")  # type: ignore
    except Exception:  # pragma: no cover - torch optional in some envs
        return None


def _snapshot_torch_determinism(torch_module):
    """TODO docstring. Document this function.

    Args:
        torch_module: TODO docstring.
    """
    state: dict[str, object | None] = {
        "algos": None,
        "cudnn_backend": None,
        "cudnn_det": None,
        "cudnn_bench": None,
    }
    if hasattr(torch_module, "are_deterministic_algorithms_enabled"):
        try:
            state["algos"] = bool(torch_module.are_deterministic_algorithms_enabled())
        except Exception:  # pragma: no cover - defensive capture
            state["algos"] = None
    cudnn_backend = getattr(getattr(torch_module, "backends", None), "cudnn", None)
    state["cudnn_backend"] = cudnn_backend
    if cudnn_backend is not None:
        state["cudnn_det"] = getattr(cudnn_backend, "deterministic", None)
        state["cudnn_bench"] = getattr(cudnn_backend, "benchmark", None)
    return state


def _apply_nondeterministic(torch_module, cudnn_backend):
    """TODO docstring. Document this function.

    Args:
        torch_module: TODO docstring.
        cudnn_backend: TODO docstring.
    """
    try:
        _set_torch_deterministic_algorithms(torch_module, False)
        if cudnn_backend is not None:
            cudnn_backend.deterministic = False  # type: ignore[attr-defined]
            cudnn_backend.benchmark = True  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - best effort guard
        pass


def _restore_torch_determinism(torch_module, state):
    """TODO docstring. Document this function.

    Args:
        torch_module: TODO docstring.
        state: TODO docstring.
    """
    try:
        prev_algos = state.get("algos")
        cudnn_backend = state.get("cudnn_backend")
        if prev_algos is not None:
            _set_torch_deterministic_algorithms(torch_module, bool(prev_algos))
        if cudnn_backend is not None:
            prev_det = state.get("cudnn_det")
            prev_bench = state.get("cudnn_bench")
            if prev_det is not None:
                cudnn_backend.deterministic = prev_det  # type: ignore[attr-defined]
            if prev_bench is not None:
                cudnn_backend.benchmark = prev_bench  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - best effort restore
        pass


@pytest.fixture(scope="session", autouse=True)
def headless_pygame_environment() -> Generator[None, None, None]:
    """Force pygame/matplotlib to run headlessly for the duration of the test session."""
    originals: dict[str, str | None] = {
        "DISPLAY": os.environ.get("DISPLAY"),
        "SDL_VIDEODRIVER": os.environ.get("SDL_VIDEODRIVER"),
        "MPLBACKEND": os.environ.get("MPLBACKEND"),
        "SDL_AUDIODRIVER": os.environ.get("SDL_AUDIODRIVER"),
        "PYGAME_HIDE_SUPPORT_PROMPT": os.environ.get("PYGAME_HIDE_SUPPORT_PROMPT"),
    }
    os.environ.update(
        {
            "DISPLAY": "",
            "SDL_VIDEODRIVER": "dummy",
            "MPLBACKEND": "Agg",
            "SDL_AUDIODRIVER": "dummy",
            "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
        },
    )
    yield
    for k, v in originals.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="session", autouse=True)
def reroute_artifact_root(tmp_path_factory: pytest.TempPathFactory) -> Generator[None, None, None]:
    """Override ROBOT_SF_ARTIFACT_ROOT so tests keep the repo tree clean.

    Args:
        tmp_path_factory: Pytest factory used to create a persistent temp directory.
    """
    original = os.environ.get("ROBOT_SF_ARTIFACT_ROOT")
    if original:
        ensure_canonical_tree(original)
        yield
        return

    override_dir = tmp_path_factory.mktemp("robot_sf_artifacts")
    os.environ["ROBOT_SF_ARTIFACT_ROOT"] = str(override_dir)
    ensure_canonical_tree(override_dir)
    try:
        yield
    finally:
        os.environ.pop("ROBOT_SF_ARTIFACT_ROOT", None)


@pytest.fixture(scope="session", autouse=True)
def writable_headless_caches(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[None, None, None]:
    """Provide writable cache directories for matplotlib/fontconfig-backed imports."""
    cache_root = tmp_path_factory.mktemp("robot_sf_test_cache")
    mpl_dir = cache_root / "mplconfig"
    xdg_dir = cache_root / "xdg_cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir.mkdir(parents=True, exist_ok=True)

    originals: dict[str, str | None] = {
        "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR"),
        "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME"),
    }
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    os.environ["XDG_CACHE_HOME"] = str(xdg_dir)
    try:
        yield
    finally:
        for key, value in originals.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(autouse=True)
def torch_nondeterministic_guard():  # type: ignore[missing-return-type-doc]
    """Ensure torch deterministic algorithms aren't forced across the suite."""

    torch_module = _import_torch_optional()
    if torch_module is None:
        yield
        return

    state = _snapshot_torch_determinism(torch_module)
    _apply_nondeterministic(torch_module, state.get("cudnn_backend"))

    try:
        yield
    finally:
        _restore_torch_determinism(torch_module, state)


@pytest.fixture(scope="session")
def perf_policy():  # type: ignore[missing-return-type-doc]
    """TODO docstring. Document this function."""
    if PerformanceBudgetPolicy is not None:
        try:
            return PerformanceBudgetPolicy()
        except Exception:  # pragma: no cover
            pass

    class _Fallback:  # pragma: no cover - only used when perf utils missing
        """Fallback performance policy used only when ``tests.perf_utils`` is unavailable.

        Mirrors the public ``PerformanceBudgetPolicy`` contract (soft/hard envelopes,
        CI vs local thresholds, and xdist contention handling) so the slow-test report
        still works in minimal environments.
        """

        soft_threshold_seconds = 20.0
        hard_timeout_seconds = 60.0
        report_count = 10
        relax_env_var = "ROBOT_SF_PERF_RELAX"
        xdist_contention_multiplier = 3.0

        def is_under_xdist(self) -> bool:
            """Return whether the suite runs under pytest-xdist parallel workers.

            Returns:
                True when ``PYTEST_XDIST_WORKER`` is set to a non-empty worker id.
            """
            worker = os.environ.get("PYTEST_XDIST_WORKER")
            return bool(worker) and worker.strip() != ""

        def effective_soft_threshold(self, *, ci: bool = False) -> float:
            """Compute the soft performance threshold, widened under xdist contention.

            In CI the base soft threshold is used as-is; locally it is halved so the
            minimal off-CI runs stay fast. When running under pytest-xdist the soft
            threshold is multiplied by ``xdist_contention_multiplier`` (capped at 90
            percent of the hard timeout) to absorb parallel-worker scheduling jitter.

            Args:
                ci: When True, use the full ``soft_threshold_seconds`` instead of the
                    halved local soft threshold.

            Returns:
                The effective soft threshold in seconds.
            """
            base = self.soft_threshold_seconds if ci else (self.soft_threshold_seconds / 2.0)
            if self.is_under_xdist():
                return min(base * self.xdist_contention_multiplier, self.hard_timeout_seconds * 0.9)
            return base

        def classify(self, duration_seconds: float):
            """Classify a measured test duration against the soft/hard envelopes.

            Args:
                duration_seconds: Measured wall-clock duration of a single test in
                    seconds.

            Returns:
                ``"hard"`` when the duration reaches or exceeds the hard timeout,
                ``"soft"`` when it reaches or exceeds the soft threshold, otherwise
                ``"none"``.
            """
            if duration_seconds >= self.hard_timeout_seconds:
                return "hard"
            if duration_seconds >= self.soft_threshold_seconds:
                return "soft"
            return "none"

    return _Fallback()


_SLOW_SAMPLES: list[tuple[str, float]] = []


_FAST_PATH_FRAGMENTS = (
    "tests/common/",
    "tests/contract/",
    "tests/factories/",
    "tests/guard/",
    "tests/scenario_certification/",
    "tests/sensor/",
    "tests/sim/",
    "tests/training/",
    "tests/unit/",
)
_FAST_FILE_PREFIXES = (
    "test_action_adapters",
    "test_campaign_arm_admission",
    "test_config_validation",
    "test_environment_factory_signatures",
    "test_error_policy",
    "test_map_runner",
    "test_planner",
    "test_range_sensor",
    "test_seed_utils",
    "test_types",
)
_FAST_FILES = {
    "map_test.py",
    "navigation_test.py",
    "ped_grouping_test.py",
    "test_compare_coverage_cli.py",
    "test_open_coverage_report.py",
    "test_pytest_config_contract.py",
    # Canonical logging helper tests provide changed-line coverage when the
    # deprecated research logging shim is retired (issue #8378).
    "test_logging_config.py",
    # Algorithm metadata facade tests are deterministic contract coverage for
    # the changed benchmark producer; keep them in the exact-head fast lane.
    "test_algorithm_metadata.py",
    # The declarative algorithm-contract and readiness tests are deterministic
    # schema/compatibility coverage for changed benchmark producers.
    "test_algorithm_contract_registry.py",
    "test_algorithm_readiness_contract.py",
    # CUDA/NVML runtime classification tests are deterministic unit contracts
    # (importlib stubs, no real GPU) that cover the changed telemetry producer
    # in the exact-head changed-coverage gate.
    "test_check_cuda_runtime.py",
    "test_gpu_telemetry.py",
    # Collision-pressure report tests are deterministic schema and materialization
    # contracts for the changed benchmark producer.
    "test_collision_pressure_report.py",
    # Goal-marker pixels require the optional pygame extra, but the focused
    # regression is deterministic and covers the renderer in PR fast shards.
    "test_sim_view_goal_marker.py",
    # SVG geometry contract tests provide changed-line coverage for the
    # parser's explicit legacy/corrected transform paths (issue #8314).
    "test_svg_transform_contract.py",
    # The real-manifest contract tests use mocked candidate evaluation only;
    # keep their coverage available to pull-request shards without promoting
    # the broader adversarial-search suite into the fast lane.
    "test_feasibility_first_real.py",
    # The preparation-only adversarial search harness uses deterministic data
    # fixtures only; keep its contract coverage in pull-request fast shards.
    "test_search_harness.py",
    # Matched-compute runtime and canary tests use injected seams and deterministic
    # receipt fixtures; keep their accounting coverage in pull-request fast shards.
    "test_matched_compute_runtime.py",
    "test_matched_compute_production_canary.py",
    # Occupancy-grid rasterization helpers are pure unit contracts with
    # numpy fixtures; keep their coverage in fast pull-request shards for the
    # exact-head changed-coverage gate (see #7282).
    "test_occupancy_grid.py",
    "test_occupancy_grid_helpers.py",
    # Published-release audit tests are deterministic pure-contract coverage
    # for the changed release audit module; keep them in fast shards for the
    # exact-head changed-coverage gate (issue #7936).
    "test_published_release_audit.py",
    # Effective algorithm-branch tests are deterministic pure-contract coverage
    # for the changed release branch module; keep them in fast shards for the
    # exact-head changed-coverage gate (issue #7937).
    "test_effective_algorithm_branches.py",
    # Release tag/SHA identity tests are deterministic pure-contract coverage
    # for the changed release identity module; keep them in fast shards for the
    # exact-head changed-coverage gate (issue #7938).
    "test_release_tag_identity.py",
    # ORCA preflight tests are deterministic contract coverage for the changed
    # benchmark preflight module; keep them in fast PR shards (issue #8021).
    "test_orca_preflight.py",
    # Predictive multimodal forecast types tests are deterministic pure-contract
    # coverage for the changed predictive types module (issue #8049).
    "test_predictive_types.py",
    # Goal-belief contract tests are deterministic schema/lineage coverage for
    # the changed actor-observation producer; keep them in fast PR shards
    # (issue #8063).
    "test_goal_belief_contract.py",
    "test_oracle_transition_trace_contract.py",
    # Tracker-to-goal-belief adapter tests are deterministic observation-contract
    # coverage; keep them in fast PR shards for the exact-head changed-coverage gate.
    "test_tracker_goal_belief_adapter.py",
    # One-frame heading posterior and actor/oracle boundary tests are deterministic
    # contract coverage for the observation-only research baseline (issue #8068).
    "test_goal_intention.py",
    "test_goal_posterior_actor_oracle_boundary.py",
    "test_goal_posterior_planner_input.py",
    "test_goal_posterior_actor_smoke_issue_8068.py",
    "test_goal_posterior_planner_input_smoke_issue_4164.py",
    # Public goal-candidate generation and candidate-coverage tests are
    # deterministic schema/lineage contracts for the research provider;
    # keep them in fast PR shards for the exact-head changed-coverage gate
    # (issue #8073).
    "test_goal_candidate_provider.py",
    "test_goal_candidate_coverage.py",
    # Inverse goal-force estimator tests are deterministic actor/oracle-boundary
    # contracts; keep their implementation-integrity coverage in PR fast shards.
    "test_goal_force_inverse_dynamics.py",
    # Core simulation-state contract tests are deterministic alias, schema, and
    # import-boundary coverage for the additive core package (issue #8243).
    "test_contract_v1.py",
    # The pre-submit launch-manifest tests cover deterministic release-input
    # binding and the public Slurm preflight consumer contract.
    "test_generate_slurm_launch_manifest.py",
    # Topology-guided local-policy tests are deterministic planner-contract
    # coverage for the changed adapter; keep them in fast PR shards for the
    # exact-head changed-coverage gate.
    "test_topology_guided_local_policy.py",
    # Route-side/homotopy observability tests are deterministic pure-metric
    # contracts with numpy fixtures; keep them in fast pull-request shards for
    # the exact-head changed-coverage gate (issue #7890).
    "test_route_choice_observability.py",
    # Anisotropic Gaussian cost tests are deterministic planner-contract
    # coverage; keep them in PR fast shards for the changed-coverage gate (issue #7603).
    "test_anisotropic_gaussian_cost.py",
    # Biased route generator tests are deterministic navigation contracts with
    # geometric fixtures; keep them in PR fast shards (issue #8033).
    "test_biased_route_generator.py",
    # Force-coupled potential-field comparator tests are deterministic
    # benchmark-diagnostic contracts; keep them in PR fast shards (issue #8015).
    "test_force_coupled_comparator.py",
    # The trace-dossier contract tests exercise lightweight, diagnostic-only
    # export/render paths required by the exact-head changed-coverage gate.
    "test_export_trace_dossier.py",
    # Release admission, staging, doctor, and publisher suites are deterministic
    # provenance/schema contracts for the benchmark-data release lane.
    # The direct source/test pairs are required by the changed-coverage router
    # when the release lane is evaluated against current main.
    "test_artifact_publication.py",
    "test_camera_ready_checkpoint_submit_preflight.py",
    "test_camera_ready_subprocess_isolation.py",
    "test_post_execution_release_doctor.py",
    # Checkpoint provenance and Predictive MPPI adapter tests are deterministic
    # contract coverage for release-smoke producer changes; keep them in the
    # exact-head changed-coverage shards.
    "test_checkpoint_provenance_issue_4970.py",
    "test_checkpoint_staging_receipt.py",
    "test_fallback_policy.py",
    # The hybrid stress suites are deterministic filesystem/schema contracts
    # and carry the direct fail-closed branch coverage for release acceptance
    # and release-protocol changes; keep them in hosted pull-request shards.
    "test_hybrid_stress_acceptance_hardening.py",
    "test_hybrid_stress_smoke_contract.py",
    "test_hybrid_rule_local_planner.py",
    "test_release_stress_smoke_acceptance.py",
    "test_release_acceptance.py",
    "test_release_admission_edge_cases.py",
    "test_release_assurance_case.py",
    "test_release_cli_edge_cases.py",
    "test_release_doctor.py",
    "test_release_doctor_edge_cases.py",
    # Erratum tests are deterministic publication/provenance contracts for the
    # derived-only successor path. The revalidation suite exercises the real
    # build/export/cold-audit orchestration with only external seams mocked.
    "test_release_erratum.py",
    "test_revalidate_benchmark_release.py",
    "test_release_protocol.py",
    "test_release_protocol_edge_cases.py",
    # Resolved release-identity tests are deterministic contract coverage for
    # the source-freeze implementation; keep them in the changed-coverage lane.
    "test_release_resolved_identity.py",
    "test_release_resume_admission.py",
    "test_runtime_smoke_admission.py",
    "test_predictive_mppi_planner.py",
    "test_s30_h600_runtime_smoke_contract.py",
    "test_submit_release_single_node_contract.py",
    "test_trace_dossier_package.py",
    "test_trace_dossier_renderer.py",
    "test_run_benchmark_release.py",
    "test_zenodo_manifest_binding.py",
    "test_zenodo_publisher.py",
    "test_zenodo_publisher_edge_cases.py",
    # SocNav runtime-adapter tests cover the changed fallback diagnostics in
    # the planner modules without requiring an external benchmark dependency.
    "test_socnav_observation.py",
    "test_socnav.py",
    "test_socnav_prediction_module.py",
    # Termination reason tests are deterministic unit contracts for canonical outcome flags.
    "test_termination_reason.py",
    # Benchmark metric characterization tests are deterministic pure-metric
    # contracts; keep them in fast shards for changed metrics coverage.
    "test_metrics.py",
    "test_aggregated_time_cooperative.py",
    # Classic planner adapter tests are deterministic planner-contract tests for
    # the changed classic_planner_adapter.py producer; keep in fast shards for
    # the exact-head changed-coverage gate.
    "test_classic_planner_adapter.py",
    # RecurrentPPO learned adapter tests are deterministic planner-contract coverage
    # for the changed recurrent_ppo_learned_adapter.py producer in PR fast shards.
    "test_recurrent_ppo_learned_adapter.py",
    # Both CLI test owners exercise deterministic release command contracts.
    "test_cli.py",
    # The release-publication contract is deterministic schema/CLI coverage for
    # the changed release_publication_contract.py producer.
    "test_release_publication_contract.py",
    # Radius rank-stability schema tests exercise the changed benchmark producer;
    # keep their deterministic contract coverage in pull-request fast shards.
    "test_radius_rank_stability.py",
    # The mechanism-boundary atlas tests are deterministic schema/lineage checks;
    # keep coverage for the changed producer in pull-request fast shards.
    "test_mechanism_boundary_atlas.py",
    # The anisotropic human-cost adapter is covered by deterministic planner
    # contract tests; keep that focused file in the fast lane for exact-head
    # changed-coverage admission.
    "test_predictive_human_cost.py",
    # Forecast-preparation packet tests are deterministic schema/provenance
    # contracts; keep their changed producer covered by the fast lane.
    "test_forecast_preparation.py",
    # Figure-interpretation replay tests are provider-free deterministic contracts;
    # keep their exact-head mutation and provenance coverage in fast shards.
    "test_agent_figure_interpretation_eval.py",
    # Result-interpretation packet tests are deterministic schema, digest, and
    # CLI contracts for the changed benchmark producer.
    "test_result_interpretation_packet.py",
    # The shared DWA diagnostic harness tests are deterministic contract tests;
    # keep their changed-module coverage in pull-request fast shards.
    "test_dwa_diagnostic_harness.py",
    # Research orchestration tests are deterministic manifest/report contracts;
    # keep coverage for the benchmark orchestrator basename match in fast shards.
    "test_orchestrator.py",
    # Full Classic phase-boundary tests are deterministic scheduler/context/finalizer
    # contracts; keep extracted orchestration coverage in exact-head fast shards.
    "test_orchestration_boundaries.py",
    # The CALF/LegNav comparator tests exercise deterministic schema, manifest,
    # materialization, and provenance contracts for the changed benchmark module.
    "test_calf_legnav_comparator.py",
    # These smoke/fixture scenario tests run short deterministic simulation
    # episodes through the map-runner trace recorder; they are the only
    # fast-lane coverage for changed trace paths in the exact-head
    # changed-coverage gate (see #7578). Each file runs in ~30s locally.
    "test_issue_2526_cyclist_vru_smoke.py",
    "test_issue_2527_waiting_crossing_fixture.py",
    "test_issue_2727_fast_bicycle_actor.py",
    "test_issue_3977_public_requirement_smoke.py",
    "sim_config_test.py",
    "unicycle_drive_test.py",
    "zone_sampling_test.py",
    # Incident-to-scenario provenance contract tests are deterministic schema
    # coverage for the changed benchmark producer; keep them in the exact-head
    # fast lane for changed-coverage admission.
    "test_incident_scenario_provenance.py",
    # SNQI inventory CLI tests use deterministic monkeypatched data and are
    # required to cover the machine-readable diagnostic contract in PR shards.
    "test_snqi_weight_inventory_cli.py",
    # Force-coupled potential-field planner tests are deterministic contract
    # coverage for the changed planner producer; keep them in the exact-head
    # fast lane for changed-coverage admission.
    "test_force_coupled_potential_field.py",
    # Versioned obstacle-force dispatch tests are deterministic contract
    # coverage for the planner, simulator, and wrapper seams; keep their
    # top-level modules in PR shards so changed coverage cannot exclude them as
    # auto-marked slow tests.
    "test_socnav_planner_adapter.py",
    "test_fast_pysf_wrapper.py",
    "test_simulator_init_factory.py",
    # Reset metadata and JSONL recording tests provide deterministic contract
    # coverage for persisted obstacle-force runtime metadata.
    "test_reset_metadata.py",
    "test_jsonl_recording.py",
    # Pedestrian reset compatibility covers the simulator metadata forwarding
    # branch in the changed environment reset path.
    "test_pedestrian_env_compat.py",
    # The evaluator-only oracle channel test is a one-step deterministic contract
    # for changed environment plumbing; keep it in PR fast shards so changed
    # coverage proves the info-only branch.
    "test_oracle_force_trace_channel.py",
    # Release preflight and camera-ready checkpoint tests are deterministic
    # admission-contract coverage for changed release producers.
    "test__legacy_campaign_facade.py",
    "test_campaign_checkpoint_preflight.py",
    # Interaction-conditioned realism validation and its contract tests are
    # deterministic schema/segmenter contracts for the changed benchmark
    # producer; keep them in fast shards for the exact-head changed-coverage
    # gate (issue #8246).
    "test_pedestrian_realism_validation.py",
    "test_realism_validation_contract.py",
    "test_realism_segmenter.py",
    # Public API facade tests must run in PR fast shards so their coverage
    # proves the changed lazy exports, facade, and episode record round-trip.
    "test_public_api.py",
    # Planner-worker exception classification is a deterministic benchmark
    # control-plane contract; keep it in PR fast shards so changed runner lines
    # are covered without admitting any benchmark result.
    "test_runner_exception_logging.py",
    # Recording-save policy tests are deterministic branch contracts (uninitialized
    # shells, no environments) for the changed recording-save path; keep them in
    # PR fast shards for the exact-head changed-coverage gate (issue #8422).
    "test_recording_save_policy.py",
}
_SLOW_FILE_OVERRIDES = {
    "test_edge_cases_recording.py",
    "test_runner_video.py",
}
_TEST_LANE_ENV = "ROBOT_SF_TEST_LANE"
_TEST_LANE_ALL = "all"
_TEST_LANE_CORE = "core"
_TEST_LANE_OPTIONAL = "optional"
_OPTIONAL_ALLOWLIST_PATH = Path(__file__).parent / "support" / "optional_test_allowlist.txt"


def _load_optional_allowlist() -> tuple[set[str], set[str]]:
    """Load the optional test allowlist from the single source of truth.

    Returns:
        A tuple of (directory_patterns, file_paths) where:
        - directory_patterns: set of directory patterns (with trailing /)
        - file_paths: set of specific file paths
    """
    allowlist_path = _OPTIONAL_ALLOWLIST_PATH
    if not allowlist_path.is_file():
        raise FileNotFoundError(f"Optional test allowlist file not found: {allowlist_path}")

    directory_patterns = set()
    file_paths = set()

    with open(allowlist_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Directory patterns end with /
            if line.endswith("/"):
                directory_patterns.add(line)
            else:
                file_paths.add(line)

    return directory_patterns, file_paths


# Load the allowlist once at module load time
_OPTIONAL_TEST_PATH_FRAGMENTS, _OPTIONAL_TEST_FILES = _load_optional_allowlist()


@pytest.fixture
def test_map() -> MapDefinition:
    """Provide a compact deterministic map for adversarial-route unit tests."""
    robot_spawn_zone = ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0))
    robot_goal_zone = ((8.0, 8.0), (9.0, 8.0), (8.0, 9.0))
    ped_spawn_zone = ((1.0, 8.0), (2.0, 8.0), (1.0, 9.0))
    ped_goal_zone = ((8.0, 1.0), (9.0, 1.0), (8.0, 2.0))
    bounds = [
        (0.0, 10.0, 0.0, 0.0),
        (0.0, 10.0, 10.0, 10.0),
        (0.0, 0.0, 0.0, 10.0),
        (10.0, 10.0, 0.0, 10.0),
    ]
    robot_route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.5, 1.5), (8.5, 8.5)],
        spawn_zone=robot_spawn_zone,
        goal_zone=robot_goal_zone,
    )
    ped_route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.5, 8.5), (8.5, 1.5)],
        spawn_zone=ped_spawn_zone,
        goal_zone=ped_goal_zone,
    )
    return MapDefinition(
        width=10.0,
        height=10.0,
        obstacles=[],
        robot_spawn_zones=[robot_spawn_zone],
        ped_spawn_zones=[ped_spawn_zone],
        robot_goal_zones=[robot_goal_zone],
        bounds=bounds,
        robot_routes=[robot_route],
        ped_goal_zones=[ped_goal_zone],
        ped_crowded_zones=[],
        ped_routes=[ped_route],
        single_pedestrians=[],
    )


@pytest.fixture
def test_planner(test_map: MapDefinition) -> ClassicGlobalPlanner:
    """Provide deterministic classic planner for adversarial-route unit tests."""
    return ClassicGlobalPlanner(
        test_map,
        ClassicPlannerConfig(
            cells_per_meter=1.0,
            inflate_radius_meters=0.0,
            add_boundary_obstacles=False,
            algorithm="a_star",
        ),
    )


def _should_auto_mark_slow(path_str: str) -> bool:
    """Return True when a test path should be auto-marked as slow."""
    normalized = path_str.replace("\\", "/")
    filename = Path(normalized).name
    if filename in _SLOW_FILE_OVERRIDES:
        return True
    if filename in _FAST_FILES:
        return False
    if any(fragment in normalized for fragment in _FAST_PATH_FRAGMENTS):
        return False
    if any(filename.startswith(prefix) for prefix in _FAST_FILE_PREFIXES):
        return False
    return True


def _configured_test_lane() -> str:
    """Return the active pytest collection lane."""
    lane = os.environ.get(_TEST_LANE_ENV, _TEST_LANE_ALL).strip().lower()
    if lane in {_TEST_LANE_ALL, _TEST_LANE_CORE, _TEST_LANE_OPTIONAL}:
        return lane
    return _TEST_LANE_ALL


def _is_optional_readiness_test_path(path_str: str) -> bool:
    """Return whether a test path needs optional-extra readiness dependencies."""
    normalized = path_str.replace("\\", "/").split("::", maxsplit=1)[0]
    if "/fast-pysf/tests/" in normalized:
        rel = normalized.split("/fast-pysf/tests/", maxsplit=1)[1]
        normalized = f"fast-pysf/tests/{rel}"
    elif "/tests/" in normalized:
        rel = normalized.rsplit("/tests/", maxsplit=1)[1]
        normalized = f"tests/{rel}"
    return normalized in _OPTIONAL_TEST_FILES or any(
        fragment in normalized for fragment in _OPTIONAL_TEST_PATH_FRAGMENTS
    )


def _should_collect_in_lane(path_str: str, lane: str) -> bool:
    """Return whether a path belongs in the requested pytest lane."""
    is_optional = _is_optional_readiness_test_path(path_str)
    if lane == _TEST_LANE_CORE:
        return not is_optional
    if lane == _TEST_LANE_OPTIONAL:
        return is_optional
    return True


def pytest_ignore_collect(collection_path, path=None, config=None):  # type: ignore[missing-type-doc]
    """Skip fast or slow files before import when a lane is explicitly selected."""
    del path
    del config
    lane = _configured_test_lane()
    if lane == _TEST_LANE_ALL:
        return False

    path_obj = Path(str(collection_path))
    if path_obj.is_dir():
        return False
    return not _should_collect_in_lane(path_obj.as_posix(), lane)


def pytest_collection_modifyitems(config, items):  # type: ignore[missing-type-doc]
    """Auto-mark non-core tests as slow to keep fast unit runs small."""
    del config
    for item in items:
        path_str = str(item.fspath)
        if _should_auto_mark_slow(path_str):
            item.add_marker(pytest.mark.slow)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):  # type: ignore[missing-type-doc]
    """TODO docstring. Document this function.

    Args:
        item: TODO docstring.
    """
    start = time.perf_counter()
    try:
        outcome = yield
    finally:
        # Always record duration, even if the test raised/failed (ensures slow report completeness).
        _SLOW_SAMPLES.append((item.nodeid, time.perf_counter() - start))
    return outcome


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):  # type: ignore[missing-type-doc]
    """Emit slow test report and optionally enforce soft breaches.

    Enforcement logic (feature 124):
      * If ROBOT_SF_PERF_ENFORCE=1 and any soft breach occurs, convert to failure.
      * Hard breaches are expected to be handled by individual test timeouts / assertions.
    """
    del exitstatus, config
    if not _SLOW_SAMPLES:
        return
    if PerformanceBudgetPolicy is None or SlowTestSample is None:
        return
    policy = PerformanceBudgetPolicy()
    # Optional environment overrides (not part of public API; used for enforcement test)
    try:
        soft_override = os.environ.get("ROBOT_SF_PERF_SOFT")
        hard_override = os.environ.get("ROBOT_SF_PERF_HARD")
        if soft_override:
            policy.soft_threshold_seconds = float(soft_override)  # type: ignore[misc]
        if hard_override:
            policy.hard_timeout_seconds = float(hard_override)  # type: ignore[misc]
    except Exception:  # pragma: no cover
        pass
    samples = [SlowTestSample(test_identifier=n, duration_seconds=d) for n, d in _SLOW_SAMPLES]
    records = generate_report(samples, policy)
    if not records:
        return
    relax = os.environ.get(policy.relax_env_var) == "1"
    enforce = os.environ.get(getattr(policy, "enforce_env_var", "ROBOT_SF_PERF_ENFORCE")) == "1"
    terminalreporter.write_line(
        "\n"
        + format_report(records, policy)
        + ("\n(relax mode active)" if relax else "")
        + ("\n(enforce mode)" if enforce else "")
        + "\n",
    )
    if enforce and not relax:
        # Treat any soft or hard breach as failure. Hard breaches ideally already handled
        # by per-test timeout markers, but we enforce here for determinism in minimal runs.
        if any(r.breach_type in {"soft", "hard"} for r in records):
            msg = "Performance breach (soft/hard) detected under enforce mode"
            terminalreporter.write_line(msg)
            # Use pytest's exit API to set return code cleanly (avoids raw SystemExit trace noise)
            pytest.exit(msg, returncode=pytest.ExitCode.TESTS_FAILED)


# Coverage-specific fixtures for test isolation


@pytest.fixture
def sample_coverage_data():
    """
    Provide sample coverage data for testing coverage tools.

    Returns a minimal but complete coverage.json structure
    for unit testing formatters, analyzers, and comparators.
    """
    return {
        "meta": {
            "version": "7.0.0",
            "timestamp": "2025-10-23T12:00:00",
            "branch_coverage": False,
            "show_contexts": False,
        },
        "files": {
            "robot_sf/gym_env/environment.py": {
                "executed_lines": [1, 2, 3, 10, 11, 12, 20],
                "summary": {
                    "covered_lines": 7,
                    "num_statements": 10,
                    "percent_covered": 70.0,
                    "missing_lines": 3,
                    "excluded_lines": 0,
                },
                "missing_lines": [4, 5, 6],
            },
            "robot_sf/sim/simulator.py": {
                "executed_lines": [1, 2, 3, 4, 5],
                "summary": {
                    "covered_lines": 5,
                    "num_statements": 8,
                    "percent_covered": 62.5,
                    "missing_lines": 3,
                    "excluded_lines": 0,
                },
                "missing_lines": [10, 11, 12],
            },
        },
        "totals": {
            "covered_lines": 12,
            "num_statements": 18,
            "percent_covered": 66.67,
            "missing_lines": 6,
            "excluded_lines": 0,
        },
    }


@pytest.fixture
def sample_gap_data():
    """Provide sample gap analysis data for testing."""
    return {
        "gaps": [
            {
                "file": "robot_sf/gym_env/environment.py",
                "coverage_percent": 70.0,
                "uncovered_lines": 3,
                "priority_score": 4.5,
                "recommendation": "Add integration tests for reset() method",
            },
            {
                "file": "robot_sf/sim/simulator.py",
                "coverage_percent": 62.5,
                "uncovered_lines": 3,
                "priority_score": 4.5,
                "recommendation": "Add unit tests for step() method",
            },
        ],
        "summary": {
            "total_gaps": 2,
            "total_uncovered_lines": 6,
            "average_coverage": 66.25,
        },
    }


@pytest.fixture
def sample_trend_data():
    """Provide sample trend analysis data for testing."""
    return {
        "direction": "improving",
        "rate_per_week": 0.5,
        "current_coverage": 66.67,
        "oldest_coverage": 60.0,
        "snapshot_count": 10,
        "date_range": {
            "start": "2025-10-01",
            "end": "2025-10-23",
        },
    }


@pytest.fixture
def sample_baseline_data():
    """Provide sample baseline comparison data for testing."""
    return {
        "current_coverage": 66.67,
        "baseline_coverage": 70.0,
        "delta": -3.33,
        "threshold": 1.0,
        "changed_files": [
            {
                "file": "robot_sf/gym_env/environment.py",
                "current": 70.0,
                "baseline": 75.0,
                "delta": -5.0,
            },
        ],
    }


# ============================================================================
# Occupancy Grid Fixtures
# ============================================================================


@pytest.fixture
def simple_grid_config():
    """Basic 10x10m grid with 0.1m resolution (100x100 cells)."""
    from robot_sf.nav.occupancy_grid import GridChannel, GridConfig

    return GridConfig(
        resolution=0.1,
        width=10.0,
        height=10.0,
        channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS],
    )


@pytest.fixture
def large_grid_config():
    """Larger 20x20m grid with 0.1m resolution (200x200 cells)."""
    from robot_sf.nav.occupancy_grid import GridChannel, GridConfig

    return GridConfig(
        resolution=0.1,
        width=20.0,
        height=20.0,
        channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS, GridChannel.ROBOT],
    )


@pytest.fixture
def coarse_grid_config():
    """Coarse 10x10m grid with 0.5m resolution (20x20 cells)."""
    from robot_sf.nav.occupancy_grid import GridChannel, GridConfig

    return GridConfig(
        resolution=0.5,
        width=10.0,
        height=10.0,
        channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS],
    )


@pytest.fixture
def single_channel_config():
    """Grid with only obstacles channel."""
    from robot_sf.nav.occupancy_grid import GridChannel, GridConfig

    return GridConfig(
        resolution=0.1,
        width=10.0,
        height=10.0,
        channels=[GridChannel.OBSTACLES],
    )


@pytest.fixture
def occupancy_grid(simple_grid_config):
    """Instantiated OccupancyGrid with simple config."""
    from robot_sf.nav.occupancy_grid import OccupancyGrid

    return OccupancyGrid(config=simple_grid_config)


@pytest.fixture
def robot_pose_center():
    """Robot at center of a 10x10m grid (world frame origin)."""
    return ((5.0, 5.0), 0.0)


@pytest.fixture
def robot_pose_corner():
    """Robot at corner of grid."""
    return ((1.0, 1.0), 0.0)


@pytest.fixture
def robot_pose_rotated():
    """Robot at center with 45° rotation."""
    import numpy as np

    return ((5.0, 5.0), np.pi / 4)


@pytest.fixture
def simple_obstacles():
    """Simple obstacle layout: two horizontal walls."""
    return [
        ((1.0, 3.0), (9.0, 3.0)),  # Horizontal wall at Y=3
        ((1.0, 7.0), (9.0, 7.0)),  # Horizontal wall at Y=7
    ]


@pytest.fixture
def complex_obstacles():
    """More complex obstacle layout: rectangular room with interior walls."""
    return [
        # Outer walls
        ((0.5, 0.5), (9.5, 0.5)),  # Bottom
        ((0.5, 9.5), (9.5, 9.5)),  # Top
        ((0.5, 0.5), (0.5, 9.5)),  # Left
        ((9.5, 0.5), (9.5, 9.5)),  # Right
        # Interior walls
        ((3.0, 2.0), (3.0, 8.0)),  # Vertical divider
        ((7.0, 2.0), (7.0, 8.0)),  # Vertical divider
    ]


@pytest.fixture
def simple_pedestrians():
    """Simple pedestrian layout: two pedestrians."""
    return [
        ((3.0, 5.0), 0.3),  # Pedestrian at (3, 5)
        ((7.0, 5.0), 0.3),  # Pedestrian at (7, 5)
    ]


@pytest.fixture
def crowded_pedestrians():
    """Crowded layout: 5 pedestrians in middle of grid."""
    return [
        ((4.5, 4.5), 0.3),
        ((5.5, 4.5), 0.3),
        ((5.0, 5.5), 0.3),
        ((4.5, 5.5), 0.3),
        ((5.5, 5.5), 0.3),
    ]


@pytest.fixture
def empty_pedestrians():
    """Empty pedestrian list."""
    return []


@pytest.fixture
def pre_generated_grid(occupancy_grid, simple_obstacles, simple_pedestrians, robot_pose_center):
    """Pre-generated grid with simple layout."""
    grid = occupancy_grid
    grid.generate(
        obstacles=simple_obstacles,
        pedestrians=simple_pedestrians,
        robot_pose=robot_pose_center,
        ego_frame=False,
    )
    return grid


# ============================================================================
# Shared Subprocess Mock Fixture
# ============================================================================


def _build_matcher_predicate(
    matcher: list[str] | tuple[str, ...] | str | Callable[[list[str]], bool],
) -> Callable[[list[str], dict[str, Any]], bool]:
    def predicate(cmd: list[str], kwargs: dict[str, Any]) -> bool:
        del kwargs
        if callable(matcher):
            return bool(matcher(cmd))
        if isinstance(matcher, (list, tuple)):
            matcher_list = [str(x) for x in matcher]
            return cmd == matcher_list or cmd[: len(matcher_list)] == matcher_list
        if isinstance(matcher, str):
            return bool(cmd and cmd[0] == matcher)
        return False

    return predicate


def _make_exception_handler(exc: Exception) -> Callable[..., Any]:
    def exc_handler(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        del cmd, args, kwargs
        raise exc

    return exc_handler


def _make_completed_process_handler(
    result: Any,
    stdout: str | None,
    stderr: str,
    returncode: int,
) -> Callable[..., Any]:
    if stdout is not None or returncode != 0 or stderr != "":
        out = stdout or ""
        code = returncode
    elif isinstance(result, str):
        out = result
        code = returncode
    elif isinstance(result, int):
        out = ""
        code = result
    elif result is not None:

        def obj_handler(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
            del cmd, args, kwargs
            return result

        return obj_handler
    else:
        out = stdout or ""
        code = returncode

    def proc_handler(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return subprocess.CompletedProcess(cmd, code, stdout=out, stderr=stderr)

    return proc_handler


def _build_result_handler(
    result: Any,
    stdout: str | None,
    stderr: str,
    returncode: int,
) -> Callable[..., Any]:
    if isinstance(result, Exception):
        return _make_exception_handler(result)
    if callable(result):
        return result
    if isinstance(result, subprocess.CompletedProcess):

        def pass_proc(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
            del cmd, args, kwargs
            return result

        return pass_proc

    return _make_completed_process_handler(result, stdout, stderr, returncode)


class FakeSubprocess:
    """Configurable subprocess mock for test assertions and deterministic returns."""

    def __init__(self) -> None:
        """Initialize an empty FakeSubprocess recorder."""
        self.calls: list[list[str]] = []
        self.kwargs_history: list[dict[str, Any]] = []
        self._handlers: list[
            tuple[
                Callable[[list[str], dict[str, Any]], bool],
                Callable[..., Any],
            ]
        ] = []
        self._default_handler: Callable[..., Any] | None = None

    def register(
        self,
        matcher: list[str] | tuple[str, ...] | str | Callable[[list[str]], bool],
        result: Any = None,
        *,
        stdout: str | None = None,
        stderr: str = "",
        returncode: int = 0,
    ) -> FakeSubprocess:
        """Register a handler for matching command invocations."""
        predicate = _build_matcher_predicate(matcher)
        handler = _build_result_handler(result, stdout, stderr, returncode)
        self._handlers.append((predicate, handler))
        return self

    def set_default(
        self,
        result: Any = None,
        *,
        stdout: str | None = None,
        stderr: str = "",
        returncode: int = 0,
    ) -> FakeSubprocess:
        """Set fallback result when no registered matchers match."""
        self._default_handler = _build_result_handler(result, stdout, stderr, returncode)
        return self

    def called(
        self, matcher: list[str] | tuple[str, ...] | str | Callable[[list[str]], bool]
    ) -> bool:
        """Check if any recorded call matches the matcher."""
        for cmd in self.calls:
            if callable(matcher) and matcher(cmd):
                return True
            if isinstance(matcher, (list, tuple)):
                matcher_list = [str(x) for x in matcher]
                if cmd == matcher_list or cmd[: len(matcher_list)] == matcher_list:
                    return True
            if isinstance(matcher, str) and cmd and cmd[0] == matcher:
                return True
        return False

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_call(self) -> list[str] | None:
        return self.calls[-1] if self.calls else None

    @property
    def last_kwargs(self) -> dict[str, Any] | None:
        return self.kwargs_history[-1] if self.kwargs_history else None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            cmd_arg = args[0]
            rest_args = args[1:]
        elif "args" in kwargs:
            cmd_arg = kwargs["args"]
            rest_args = ()
        elif "command" in kwargs:
            cmd_arg = kwargs["command"]
            rest_args = ()
        elif "cmd" in kwargs:
            cmd_arg = kwargs["cmd"]
            rest_args = ()
        else:
            raise TypeError("fake_subprocess requires command arguments")

        if isinstance(cmd_arg, str):
            cmd = [cmd_arg]
        elif isinstance(cmd_arg, (list, tuple)):
            cmd = [str(x) for x in cmd_arg]
        else:
            cmd = [str(cmd_arg)]

        self.calls.append(cmd)
        self.kwargs_history.append(kwargs)

        for predicate, handler in self._handlers:
            if predicate(cmd, kwargs):
                return handler(cmd, *rest_args, **kwargs)

        if self._default_handler is not None:
            return self._default_handler(cmd, *rest_args, **kwargs)

        raise AssertionError(f"Unexpected subprocess command: {cmd} with kwargs={kwargs}")


@pytest.fixture
def fake_subprocess() -> FakeSubprocess:
    """Provide a configurable FakeSubprocess instance for mocking subprocess calls."""
    return FakeSubprocess()
