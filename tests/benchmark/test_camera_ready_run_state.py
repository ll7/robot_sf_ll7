"""Focused unit tests for camera-ready run-state and provenance helpers.

These tests target the pure helpers in
``robot_sf.benchmark.camera_ready._run_state``, emphasizing corner cases
(coercion of falsy identifiers, heterogeneous-seed ordering, success-basis
anchoring, fail-closed observation-noise resolution, credential scrubbing, and
rollup error aggregation) that are not already asserted directly by the broader
campaign tests in ``test_camera_ready_campaign.py``.

This is the companion slice to #6080 (which covered ``_reporting``): it adds
direct, focused coverage for the run-state helpers extracted for issue #3385.

Inputs are minimal in-memory dicts/lists plus a ``tmp_path`` for the helpers
that touch disk; no real benchmark artifacts are read.

Refs: #6081
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from robot_sf.benchmark.camera_ready._config import _sanitize_name
from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec
from robot_sf.benchmark.camera_ready._run_state import (
    _build_arm_rollup,
    _campaign_id,
    _campaign_success_counters,
    _coerce_identifier,
    _episode_identity,
    _episode_identity_sort_key,
    _expected_episode_identities,
    _git_context,
    _integrity_blocker,
    _resolve_campaign_id,
    _resolve_execution_mode,
    _resolve_observation_noise,
    _resolve_path,
    _sanitize_git_remote,
)

# Anchor provenance commit length used by structural git-context assertions.
_FULL_GIT_HASH_LENGTH = 40

# Canonical timestamp suffix shape produced by ``_campaign_id``.
_TIMESTAMP_SUFFIX_RE = re.compile(r"_\d{8}_\d{6}$")

# Maximum length retained for a per-arm first-error string in the rollup.
_MAX_FIRST_ERROR_LEN = 200


def _minimal_campaign_config(*, name: str = "test_campaign") -> CampaignConfig:
    """Return a minimal ``CampaignConfig`` carrying only a name for id helpers.

    The id helpers only read ``cfg.name``; the matrix path and planners are
    never dereferenced, so placeholder values are safe.
    """
    return CampaignConfig(
        name=name,
        scenario_matrix_path=Path("scenarios.yaml"),
        planners=(PlannerSpec(key="goal", algo="goal"),),
    )


# ---------------------------------------------------------------------------
# _coerce_identifier
# ---------------------------------------------------------------------------


class TestCoerceIdentifier:
    """``_coerce_identifier`` normalizes a logical id without collapsing valid falsy values."""

    def test_none_returns_none(self) -> None:
        assert _coerce_identifier(None) is None

    def test_int_is_returned_unchanged(self) -> None:
        assert _coerce_identifier(5) == 5
        assert isinstance(_coerce_identifier(5), int)

    def test_numeric_string_is_coerced_to_int(self) -> None:
        assert _coerce_identifier("5") == 5
        assert isinstance(_coerce_identifier("5"), int)

    def test_float_is_truncated_to_int(self) -> None:
        # ``int(3.5)`` truncates toward zero without raising; this documents the
        # coercion rather than rounding.
        assert _coerce_identifier(3.5) == 3

    def test_decimal_string_falls_back_to_string(self) -> None:
        # ``int("3.5")`` raises ValueError, so the value is kept as a string.
        assert _coerce_identifier("3.5") == "3.5"

    def test_non_numeric_string_is_returned_unchanged(self) -> None:
        assert _coerce_identifier("abc") == "abc"

    def test_bool_is_coerced_to_int_one(self) -> None:
        # bool is a subclass of int; ``int(True) == 1``.
        assert _coerce_identifier(True) == 1

    def test_list_falls_back_to_string(self) -> None:
        assert _coerce_identifier([1, 2]) == "[1, 2]"

    def test_dict_falls_back_to_string(self) -> None:
        assert _coerce_identifier({"a": 1}) == "{'a': 1}"


# ---------------------------------------------------------------------------
# _episode_identity
# ---------------------------------------------------------------------------


class TestEpisodeIdentity:
    """``_episode_identity`` returns the stable logical identity of a record."""

    def test_returns_scenario_id_and_seed(self) -> None:
        assert _episode_identity({"scenario_id": "corridor", "seed": 5}) == ("corridor", 5)

    def test_strips_whitespace_from_scenario_id(self) -> None:
        assert _episode_identity({"scenario_id": "  hallway  ", "seed": 5}) == ("hallway", 5)

    def test_empty_record_returns_empty_string_and_none(self) -> None:
        assert _episode_identity({}) == ("", None)

    def test_missing_seed_returns_none(self) -> None:
        assert _episode_identity({"scenario_id": "corridor"}) == ("corridor", None)

    def test_string_seed_is_coerced_to_string(self) -> None:
        # A non-numeric seed string stays a string in the identity tuple.
        assert _episode_identity({"seed": "abc"}) == ("", "abc")

    def test_numeric_string_seed_is_coerced_to_int(self) -> None:
        assert _episode_identity({"scenario_id": "s1", "seed": "9"}) == ("s1", 9)

    def test_none_scenario_id_becomes_string_none(self) -> None:
        # A present-but-None scenario_id is stringified rather than defaulted;
        # this documents the explicit ``str()`` coercion behavior.
        assert _episode_identity({"scenario_id": None}) == ("None", None)


# ---------------------------------------------------------------------------
# _episode_identity_sort_key
# ---------------------------------------------------------------------------


class TestEpisodeIdentitySortKey:
    """``_episode_identity_sort_key`` gives a total order to heterogeneous seeds."""

    def test_none_seed_sorts_before_int_and_string(self) -> None:
        identities = [("a", 5), ("a", "x"), ("a", None)]
        ordered = sorted(identities, key=_episode_identity_sort_key)
        assert ordered == [("a", None), ("a", 5), ("a", "x")]

    def test_orders_by_scenario_id_first(self) -> None:
        identities = [("z", None), ("a", "x"), ("m", 5)]
        ordered = sorted(identities, key=_episode_identity_sort_key)
        assert [scenario for scenario, _seed in ordered] == ["a", "m", "z"]

    def test_int_seeds_sort_numerically_as_strings(self) -> None:
        # The secondary key is the stringified int, so lexicographic order
        # applies (e.g. "10" before "2").
        ordered = sorted([("a", 2), ("a", 10)], key=_episode_identity_sort_key)
        assert ordered == [("a", 10), ("a", 2)]

    def test_string_seed_tier_is_two(self) -> None:
        # Explicitly checks the tier tuple shape for the three identity kinds.
        assert _episode_identity_sort_key(("a", None))[1] == (0, "")
        assert _episode_identity_sort_key(("a", 5))[1] == (1, "5")
        assert _episode_identity_sort_key(("a", "x"))[1] == (2, "x")

    def test_full_sort_across_scenario_and_seed(self) -> None:
        identities = [("z", None), ("a", "x"), ("a", 5), ("a", None), ("b", 2)]
        ordered = sorted(identities, key=_episode_identity_sort_key)
        assert ordered == [("a", None), ("a", 5), ("a", "x"), ("b", 2), ("z", None)]


# ---------------------------------------------------------------------------
# _expected_episode_identities
# ---------------------------------------------------------------------------


class TestExpectedEpisodeIdentities:
    """``_expected_episode_identities`` builds the scenario/seed denominator."""

    def test_uses_scenario_owned_seeds_when_present(self) -> None:
        assert _expected_episode_identities([{"id": "s1", "seeds": [1, 2]}], []) == {
            ("s1", 1),
            ("s1", 2),
        }

    def test_falls_back_to_resolved_seeds_when_scenario_has_none(self) -> None:
        assert _expected_episode_identities([{"id": "s1"}], [10, 20]) == {
            ("s1", 10),
            ("s1", 20),
        }

    def test_scenario_seeds_take_precedence_over_fallback_seeds(self) -> None:
        # When a scenario owns seeds, the fallback (resolved_seeds) is ignored.
        result = _expected_episode_identities([{"id": "s1", "seeds": [1]}], [99])
        assert result == {("s1", 1)}

    def test_id_takes_precedence_over_scenario_id_and_name(self) -> None:
        result = _expected_episode_identities(
            [{"id": "i", "scenario_id": "sc", "name": "nm", "seeds": [1]}], []
        )
        assert result == {("i", 1)}

    def test_uses_name_when_id_and_scenario_id_absent(self) -> None:
        result = _expected_episode_identities([{"name": "nm", "seeds": [1]}], [])
        assert result == {("nm", 1)}

    def test_empty_scenario_id_when_no_resolvable_field(self) -> None:
        assert _expected_episode_identities([{"seeds": [1]}], []) == {("", 1)}

    def test_empty_inputs_produce_empty_set(self) -> None:
        assert _expected_episode_identities([], [1, 2]) == set()

    def test_non_int_seeds_are_skipped(self) -> None:
        result = _expected_episode_identities([{"id": "s1", "seeds": [1, "bad", 3]}], [])
        assert result == {("s1", 1), ("s1", 3)}

    def test_string_seeds_treated_as_empty_and_use_fallback(self) -> None:
        # A str seeds value is not a Sequence-of-seeds here, so fallback applies.
        result = _expected_episode_identities([{"id": "s1", "seeds": "12"}], [4])
        assert result == {("s1", 4)}

    def test_tuple_seeds_are_accepted(self) -> None:
        result = _expected_episode_identities([{"id": "s1", "seeds": (7, 8)}], [])
        assert result == {("s1", 7), ("s1", 8)}


# ---------------------------------------------------------------------------
# _integrity_blocker
# ---------------------------------------------------------------------------


class TestIntegrityBlocker:
    """``_integrity_blocker`` builds one deterministic blocker payload."""

    def test_returns_blocker_with_details(self) -> None:
        blocker = _integrity_blocker("arm1", "missing_episode_artifact", count=3, path="x")
        assert blocker == {
            "arm": "arm1",
            "invariant": "missing_episode_artifact",
            "details": {"count": 3, "path": "x"},
        }

    def test_details_empty_when_no_kwargs(self) -> None:
        blocker = _integrity_blocker("arm1", "missing_episode_artifact")
        assert blocker["details"] == {}

    def test_is_json_serializable(self) -> None:
        import json

        blocker = _integrity_blocker("arm", "inv", identities=[["s", 1]])
        # Blockers are machine-readable; must round-trip through JSON.
        assert json.loads(json.dumps(blocker)) == blocker


# ---------------------------------------------------------------------------
# _campaign_success_counters
# ---------------------------------------------------------------------------


class TestCampaignSuccessCounters:
    """``_campaign_success_counters`` anchors success on planners present.

    The core-ignores-experimental happy path is already covered by
    ``test_camera_ready_campaign.py``; these cover the remaining basis/mismatch
    corner cases.
    """

    def test_empty_entries_fails_closed_on_all_basis(self) -> None:
        counters = _campaign_success_counters([])
        assert counters == {
            "benchmark_success": False,
            "benchmark_success_basis": "all",
            "total_runs": 0,
            "successful_runs": 0,
            "core_total_runs": 0,
            "core_successful_runs": 0,
        }

    def test_all_runs_success_basis_when_no_core_planners(self) -> None:
        counters = _campaign_success_counters([{"status": "ok"}, {"status": "ok"}])
        assert counters["benchmark_success"] is True
        assert counters["benchmark_success_basis"] == "all"
        assert counters["total_runs"] == 2
        assert counters["successful_runs"] == 2
        assert counters["core_total_runs"] == 0

    def test_all_runs_basis_fails_when_any_run_failed_and_no_core(self) -> None:
        counters = _campaign_success_counters([{"status": "ok"}, {"status": "failed"}])
        assert counters["benchmark_success"] is False
        assert counters["benchmark_success_basis"] == "all"
        assert counters["successful_runs"] == 1

    def test_expected_core_runs_mismatch_fails_even_when_all_core_succeed(self) -> None:
        counters = _campaign_success_counters(
            [{"status": "ok", "planner": {"planner_group": "core"}}], expected_core_runs=3
        )
        assert counters["benchmark_success"] is False
        assert counters["benchmark_success_basis"] == "core"
        assert counters["core_total_runs"] == 1
        assert counters["core_successful_runs"] == 1

    def test_expected_core_runs_none_defaults_to_observed_core_total(self) -> None:
        counters = _campaign_success_counters(
            [{"status": "ok", "planner": {"planner_group": "core"}}]
        )
        assert counters["benchmark_success"] is True
        assert counters["benchmark_success_basis"] == "core"

    def test_core_planner_failure_drives_overall_failure_regardless_of_others(self) -> None:
        # A failed core planner fails the benchmark even with other ok runs.
        counters = _campaign_success_counters(
            [
                {"status": "failed", "planner": {"planner_group": "core"}},
                {"status": "ok", "planner": {"planner_group": "experimental"}},
            ]
        )
        assert counters["benchmark_success"] is False
        assert counters["benchmark_success_basis"] == "core"
        assert counters["core_successful_runs"] == 0

    def test_planner_group_is_case_insensitive(self) -> None:
        counters = _campaign_success_counters(
            [{"status": "ok", "planner": {"planner_group": " CORE "}}]
        )
        assert counters["core_total_runs"] == 1
        assert counters["benchmark_success_basis"] == "core"

    def test_status_comparison_is_case_sensitive(self) -> None:
        # Only the exact string ``"ok"`` counts as success.
        counters = _campaign_success_counters(
            [{"status": "OK", "planner": {"planner_group": "core"}}]
        )
        assert counters["successful_runs"] == 0
        assert counters["benchmark_success"] is False

    def test_missing_planner_block_is_treated_as_non_core(self) -> None:
        counters = _campaign_success_counters([{"status": "ok"}])
        assert counters["core_total_runs"] == 0
        assert counters["benchmark_success_basis"] == "all"


# ---------------------------------------------------------------------------
# _sanitize_git_remote
# ---------------------------------------------------------------------------


class TestSanitizeGitRemote:
    """``_sanitize_git_remote`` strips credentials and preserves the rest.

    The basic credential-strip case is already covered by
    ``test_camera_ready_campaign.py``; these cover the remaining edge cases.
    """

    def test_clean_url_is_returned_unchanged(self) -> None:
        assert _sanitize_git_remote("https://example.com/org/repo.git") == (
            "https://example.com/org/repo.git"
        )

    def test_strips_credentials_and_preserves_port(self) -> None:
        assert _sanitize_git_remote("https://user:tok@example.com:8080/org/repo.git") == (
            "https://example.com:8080/org/repo.git"
        )

    def test_empty_string_returned_as_is(self) -> None:
        assert _sanitize_git_remote("") == ""

    def test_none_returned_as_is(self) -> None:
        assert _sanitize_git_remote(None) is None

    def test_scp_style_remote_without_scheme_returned_as_is(self) -> None:
        # No ``://`` separator -> the helper returns the input untouched.
        assert _sanitize_git_remote("git@host:org/repo.git") == "git@host:org/repo.git"

    def test_file_scheme_without_hostname_returned_as_is(self) -> None:
        # ``file:///path`` parses but has no hostname, so it is returned as-is.
        assert _sanitize_git_remote("file:///tmp/x") == "file:///tmp/x"


# ---------------------------------------------------------------------------
# _resolve_execution_mode
# ---------------------------------------------------------------------------


class TestResolveExecutionMode:
    """``_resolve_execution_mode`` resolves mode from metadata with legacy fallbacks."""

    def test_none_returns_unknown(self) -> None:
        assert _resolve_execution_mode(None) == "unknown"

    def test_non_dict_returns_unknown(self) -> None:
        assert _resolve_execution_mode("not-a-dict") == "unknown"

    def test_empty_dict_returns_unknown(self) -> None:
        assert _resolve_execution_mode({}) == "unknown"

    def test_reads_planner_kinematics_execution_mode(self) -> None:
        assert (
            _resolve_execution_mode({"planner_kinematics": {"execution_mode": "native"}})
            == "native"
        )

    def test_reads_top_level_execution_mode(self) -> None:
        assert _resolve_execution_mode({"execution_mode": "adapter"}) == "adapter"

    def test_reads_adapter_impact_execution_mode(self) -> None:
        assert (
            _resolve_execution_mode({"adapter_impact": {"execution_mode": "fallback"}})
            == "fallback"
        )

    def test_planner_kinematics_takes_precedence_over_top_level(self) -> None:
        result = _resolve_execution_mode(
            {"planner_kinematics": {"execution_mode": "native"}, "execution_mode": "adapter"}
        )
        assert result == "native"

    def test_non_string_execution_mode_is_stringified(self) -> None:
        # The value is coerced via ``str()``, so an int mode becomes its string form.
        assert _resolve_execution_mode({"execution_mode": 1}) == "1"


# ---------------------------------------------------------------------------
# _build_arm_rollup
# ---------------------------------------------------------------------------


class TestBuildArmRollup:
    """``_build_arm_rollup`` aggregates per-arm status, counts, and first error."""

    def test_empty_entries_returns_empty_list(self) -> None:
        assert _build_arm_rollup([]) == []

    def test_ok_arm_has_no_error_fields(self) -> None:
        rollup = _build_arm_rollup(
            [
                {
                    "status": "ok",
                    "planner": {"key": "p1", "algo": "orca", "kinematics": "diff"},
                    "summary": {"written": 10, "failed_jobs": 0},
                }
            ]
        )
        assert rollup == [
            {
                "planner_key": "p1",
                "algo": "orca",
                "kinematics": "diff",
                "status": "ok",
                "episodes_written": 10,
                "episodes_failed": 0,
            }
        ]
        assert "first_error" not in rollup[0]
        assert "distinct_error_count" not in rollup[0]

    def test_not_available_arm_treated_like_ok_for_errors(self) -> None:
        rollup = _build_arm_rollup([{"status": "not_available", "planner": {}}])
        assert rollup[0]["status"] == "not_available"
        assert "first_error" not in rollup[0]

    def test_failed_arm_records_first_error_and_distinct_count(self) -> None:
        rollup = _build_arm_rollup(
            [
                {
                    "status": "failed",
                    "summary": {"failures": [{"error": "boom"}, {"error": "kaboom"}]},
                }
            ]
        )
        assert rollup[0]["first_error"] == "boom"
        assert rollup[0]["distinct_error_count"] == 2

    def test_distinct_count_deduplicates_identical_errors(self) -> None:
        rollup = _build_arm_rollup(
            [
                {
                    "status": "failed",
                    "summary": {"failures": [{"error": "boom"}, {"error": "boom"}]},
                }
            ]
        )
        assert rollup[0]["first_error"] == "boom"
        assert rollup[0]["distinct_error_count"] == 1

    def test_summary_error_is_used_when_no_failure_errors(self) -> None:
        rollup = _build_arm_rollup([{"status": "failed", "summary": {"error": "summary_err"}}])
        assert rollup[0]["first_error"] == "summary_err"
        # Distinct count stays zero because no per-failure signatures were seen.
        assert rollup[0]["distinct_error_count"] == 0

    def test_failed_arm_with_no_errors_has_no_error_fields(self) -> None:
        rollup = _build_arm_rollup([{"status": "failed", "summary": {}}])
        assert "first_error" not in rollup[0]
        assert "distinct_error_count" not in rollup[0]

    def test_first_error_is_truncated_to_max_length(self) -> None:
        long_error = "x" * (_MAX_FIRST_ERROR_LEN + 50)
        rollup = _build_arm_rollup(
            [{"status": "failed", "summary": {"failures": [{"error": long_error}]}}]
        )
        assert len(rollup[0]["first_error"]) == _MAX_FIRST_ERROR_LEN

    def test_episodes_written_falls_back_to_episodes_total(self) -> None:
        rollup = _build_arm_rollup([{"status": "ok", "summary": {"episodes_total": 7}}])
        assert rollup[0]["episodes_written"] == 7

    def test_episodes_written_zero_when_absent(self) -> None:
        rollup = _build_arm_rollup([{"status": "ok", "summary": {}}])
        assert rollup[0]["episodes_written"] == 0

    def test_episodes_failed_from_failed_jobs(self) -> None:
        rollup = _build_arm_rollup([{"status": "ok", "summary": {"failed_jobs": 3}}])
        assert rollup[0]["episodes_failed"] == 3

    def test_missing_planner_and_status_default_to_unknown(self) -> None:
        rollup = _build_arm_rollup([{}])
        assert rollup[0]["planner_key"] == "unknown"
        assert rollup[0]["algo"] == "unknown"
        assert rollup[0]["kinematics"] == "unknown"
        assert rollup[0]["status"] == "unknown"

    def test_failure_with_none_error_is_ignored(self) -> None:
        rollup = _build_arm_rollup(
            [{"status": "failed", "summary": {"failures": [{"error": None}]}}]
        )
        # None errors produce no signature; falls back to summary error (absent).
        assert "first_error" not in rollup[0]


# ---------------------------------------------------------------------------
# _resolve_path
# ---------------------------------------------------------------------------


class TestResolvePath:
    """``_resolve_path`` resolves relative paths against a base directory."""

    def test_none_returns_none(self, tmp_path: Path) -> None:
        assert _resolve_path(None, base_dir=tmp_path) is None

    def test_empty_string_returns_none(self, tmp_path: Path) -> None:
        assert _resolve_path("", base_dir=tmp_path) is None

    def test_absolute_path_returned_as_is(self, tmp_path: Path) -> None:
        result = _resolve_path("/tmp/some_file.yaml", base_dir=tmp_path)
        assert result == Path("/tmp/some_file.yaml")

    def test_existing_relative_path_resolved_under_base_dir(self, tmp_path: Path) -> None:
        (tmp_path / "sub.yaml").write_text("x", encoding="utf-8")
        result = _resolve_path("sub.yaml", base_dir=tmp_path)
        assert result == (tmp_path / "sub.yaml").resolve()
        assert result.exists()

    def test_nonexistent_relative_path_returns_base_dir_candidate(self, tmp_path: Path) -> None:
        result = _resolve_path("nope.yaml", base_dir=tmp_path)
        assert result == (tmp_path / "nope.yaml").resolve()
        assert not result.exists()


# ---------------------------------------------------------------------------
# _resolve_observation_noise
# ---------------------------------------------------------------------------


class TestResolveObservationNoise:
    """``_resolve_observation_noise`` resolves inline/file/absent noise configs fail-closed."""

    def test_none_returns_none(self, tmp_path: Path) -> None:
        assert _resolve_observation_noise(None, base_dir=tmp_path) is None

    def test_dict_is_normalized(self, tmp_path: Path) -> None:
        result = _resolve_observation_noise({}, base_dir=tmp_path)
        assert isinstance(result, dict)
        assert "enabled" in result

    def test_nonexistent_file_raises_filenotfound(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="missing.yaml"):
            _resolve_observation_noise("missing.yaml", base_dir=tmp_path)

    def test_empty_string_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="observation_noise"):
            _resolve_observation_noise("", base_dir=tmp_path)

    def test_whitespace_only_string_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="observation_noise"):
            _resolve_observation_noise("   ", base_dir=tmp_path)

    def test_non_string_non_dict_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="observation_noise"):
            _resolve_observation_noise(42, base_dir=tmp_path)  # type: ignore[arg-type]

    def test_existing_file_is_loaded(self, tmp_path: Path) -> None:
        (tmp_path / "noise.yaml").write_text("pose_noise_std_m: 0.1\n", encoding="utf-8")
        result = _resolve_observation_noise("noise.yaml", base_dir=tmp_path)
        assert result["pose_noise_std_m"] == pytest.approx(0.1)
        assert result["enabled"] is True


# ---------------------------------------------------------------------------
# _campaign_id / _resolve_campaign_id
# ---------------------------------------------------------------------------


class TestCampaignId:
    """``_campaign_id`` builds a timestamped identifier from the config name."""

    def test_starts_with_sanitized_name_and_timestamp(self) -> None:
        cfg = _minimal_campaign_config(name="My Campaign!")
        campaign_id = _campaign_id(cfg)
        assert campaign_id.startswith("my_campaign_")
        assert _TIMESTAMP_SUFFIX_RE.search(campaign_id) is not None

    def test_label_is_sanitized_and_inserted_before_timestamp(self) -> None:
        cfg = _minimal_campaign_config(name="demo")
        campaign_id = _campaign_id(cfg, label="Re Run")
        assert campaign_id.startswith("demo_re_run_")
        assert _TIMESTAMP_SUFFIX_RE.search(campaign_id) is not None


class TestResolveCampaignId:
    """``_resolve_campaign_id`` prefers an explicit id and sanitizes it."""

    def test_explicit_id_is_sanitized_and_returned(self) -> None:
        cfg = _minimal_campaign_config()
        assert _resolve_campaign_id(cfg, campaign_id="My Campaign!") == "my_campaign"

    def test_explicit_id_matches_sanitize_name_directly(self) -> None:
        # The helper delegates to ``_sanitize_name``; document the equivalence.
        cfg = _minimal_campaign_config()
        assert _resolve_campaign_id(cfg, campaign_id="A B") == _sanitize_name("A B")

    def test_punctuation_only_id_collapses_to_default_campaign(self) -> None:
        # ``_sanitize_name`` always returns at least ``"campaign"``, so the
        # empty-guard ``ValueError`` is unreachable for a punctuation-only id.
        cfg = _minimal_campaign_config()
        assert _resolve_campaign_id(cfg, campaign_id="!!!") == "campaign"

    def test_whitespace_only_id_collapses_to_default_campaign(self) -> None:
        cfg = _minimal_campaign_config()
        assert _resolve_campaign_id(cfg, campaign_id="   ") == "campaign"

    def test_no_explicit_id_falls_back_to_timestamped_form(self) -> None:
        cfg = _minimal_campaign_config(name="derived")
        campaign_id = _resolve_campaign_id(cfg)
        assert campaign_id.startswith("derived_")
        assert _TIMESTAMP_SUFFIX_RE.search(campaign_id) is not None


# ---------------------------------------------------------------------------
# _git_context (structural smoke; no specific values asserted)
# ---------------------------------------------------------------------------


class TestGitContext:
    """``_git_context`` collects commit/branch/remote provenance strings."""

    def test_returns_three_string_fields(self) -> None:
        context = _git_context()
        assert set(context) == {"commit", "branch", "remote"}
        for value in context.values():
            assert isinstance(value, str)

    def test_commit_is_a_full_hash_in_a_real_repo(self) -> None:
        # The worktree is a real git checkout, so the commit degrades to a
        # 40-char hash rather than the ``"unknown"`` fallback.
        context = _git_context()
        assert len(context["commit"]) == _FULL_GIT_HASH_LENGTH
