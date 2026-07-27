"""Dependency-free pure-contract tests for the CrowdNav PredAttnG adapter helpers.

Issue #6372: the staged-checkout smoke (``test_crowdnav_pred_attng.py``) is skipped
unless the external CrowdNav repository has been staged. These tests lock the pure
helpers in ``robot_sf/planner/crowdnav_pred_attng.py`` so config construction,
the temporary import context, the observation-space layout, XY normalization, and
preferred-speed clipping are covered without ever cloning, staging, or importing
the external CrowdNav repository.

Only ``tmp_path`` and ``monkeypatch`` are used to bound ``sys.path`` /
``sys.modules`` for the import-context contract; no model load, environment run,
or external dependency fetch is performed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from robot_sf.planner.crowdnav_pred_attng import (
    CrowdNavPredAttnGraphConfig,
    _build_args_namespace,
    _build_observation_space,
    _clip_holonomic_to_v_pref,
    _default_args_namespace,
    _pred_attng_import_context,
    _require_xy,
    _xy_rows,
    build_crowdnav_pred_attng_config,
)

# Defaults frozen by the pinned upstream commit / bundled config (see module
# constants). Asserted here so a silent default drift is caught locally.
DEFAULT_REPO_ROOT = Path("third_party/external_repos/crowdnav_pred_attng")
DEFAULT_MODEL_SUBDIR = Path("trained_models/GST_predictor_non_rand")
DEFAULT_CHECKPOINT_NAME = "41200.pt"
DEFAULT_HUMAN_NUM = 20
DEFAULT_TIME_STEP = 0.25
DEFAULT_V_PREF = 1.0
DEFAULT_SENSOR_RANGE = 5.0
DEFAULT_ROBOT_RADIUS = 0.3
DEFAULT_PREDICT_STEPS = 5


# --------------------------------------------------------------------------- #
# build_crowdnav_pred_attng_config: config-namespace construction             #
# --------------------------------------------------------------------------- #
class TestBuildConfig:
    """``build_crowdnav_pred_attng_config`` namespace construction contracts."""

    def test_none_payload_yields_smoke_defaults(self) -> None:
        """A ``None`` payload must resolve to the documented smoke defaults."""
        cfg = build_crowdnav_pred_attng_config(None)
        assert cfg.repo_root == DEFAULT_REPO_ROOT
        assert cfg.model_subdir == DEFAULT_MODEL_SUBDIR
        assert cfg.checkpoint_name == DEFAULT_CHECKPOINT_NAME
        assert cfg.device == "cpu"
        assert cfg.human_num == DEFAULT_HUMAN_NUM
        assert cfg.time_step == pytest.approx(DEFAULT_TIME_STEP)
        assert cfg.v_pref == pytest.approx(DEFAULT_V_PREF)
        assert cfg.sensor_range == pytest.approx(DEFAULT_SENSOR_RANGE)
        assert cfg.robot_radius == pytest.approx(DEFAULT_ROBOT_RADIUS)
        assert cfg.predict_steps == DEFAULT_PREDICT_STEPS

    def test_empty_payload_matches_frozen_dataclass_defaults(self) -> None:
        """An empty payload must match the frozen dataclass field defaults."""
        cfg = build_crowdnav_pred_attng_config({})
        default = CrowdNavPredAttnGraphConfig()
        assert cfg == default

    def test_overrides_propagate_into_namespace(self) -> None:
        """Explicit overrides must flow through into the resolved config."""
        custom_root = Path("/tmp/custom_crowdnav")
        cfg = build_crowdnav_pred_attng_config(
            {
                "repo_root": str(custom_root),
                "model_subdir": "trained_models/alt",
                "checkpoint_name": "99999.pt",
                "device": "cuda",
                "human_num": 9,
                "time_step": 0.5,
                "v_pref": 1.5,
                "sensor_range": 7.0,
                "robot_radius": 0.35,
                "predict_steps": 3,
            }
        )
        assert cfg.repo_root == custom_root
        assert cfg.model_subdir == Path("trained_models/alt")
        assert cfg.checkpoint_name == "99999.pt"
        assert cfg.device == "cuda"
        assert cfg.human_num == 9
        assert cfg.time_step == pytest.approx(0.5)
        assert cfg.v_pref == pytest.approx(1.5)
        assert cfg.sensor_range == pytest.approx(7.0)
        assert cfg.robot_radius == pytest.approx(0.35)
        assert cfg.predict_steps == 3

    def test_device_payload_is_stripped_and_blank_falls_back_to_cpu(self) -> None:
        """Whitespace device strings are stripped; blank/whitespace falls back to cpu."""
        assert build_crowdnav_pred_attng_config({"device": "  cuda  "}).device == "cuda"
        assert build_crowdnav_pred_attng_config({"device": ""}).device == "cpu"
        assert build_crowdnav_pred_attng_config({"device": "   "}).device == "cpu"

    def test_numeric_payloads_are_coerced(self) -> None:
        """Numeric string/int payloads must coerce to the declared field types."""
        cfg = build_crowdnav_pred_attng_config(
            {"human_num": "12", "time_step": "0.25", "v_pref": "1", "predict_steps": "5"}
        )
        assert cfg.human_num == 12
        assert cfg.time_step == pytest.approx(0.25)
        assert cfg.v_pref == pytest.approx(1.0)
        assert cfg.predict_steps == 5

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            ({"human_num": 0}, "human_num"),
            ({"human_num": -3}, "human_num"),
            ({"time_step": 0.0}, "time_step"),
            ({"time_step": -0.1}, "time_step"),
            ({"v_pref": 0.0}, "v_pref"),
            ({"v_pref": -2.0}, "v_pref"),
        ],
    )
    def test_non_positive_contract_fields_raise(self, payload: dict, match: str) -> None:
        """human_num / time_step / v_pref must be strictly positive."""
        with pytest.raises(ValueError, match=match):
            build_crowdnav_pred_attng_config(payload)

    def test_config_namespace_is_frozen(self) -> None:
        """The resolved config must be an immutable (frozen) dataclass instance."""
        cfg = build_crowdnav_pred_attng_config({})
        with pytest.raises((AttributeError, Exception)):
            cfg.human_num = 99  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# _default_args_namespace / _build_args_namespace                             #
# --------------------------------------------------------------------------- #
class TestArgsNamespace:
    """Upstream-compatible ``args`` namespace construction contracts."""

    def test_default_namespace_matches_upstream_contract(self) -> None:
        """The default args namespace must reconstruct the selfAttn_merge_SRNN args."""
        args = _default_args_namespace()
        # env_name selects SpatialEdgeSelfAttn.input_size=12 for the 41200.pt weights.
        assert args.env_name == "CrowdSimPred-v0"
        assert args.env_type == "crowd_sim"
        assert args.use_self_attn is True
        assert args.use_hr_attn is True
        assert args.sort_humans is True
        assert args.no_cuda is True
        assert args.cuda is False
        # Recurrent graph + embedding sizes frozen by the bundled arguments.py.
        assert args.human_node_rnn_size == 128
        assert args.human_human_edge_rnn_size == 256
        assert args.human_node_output_size == 256
        assert args.human_node_input_size == 3
        assert args.human_human_edge_input_size == 2
        assert args.human_node_embedding_size == 64
        assert args.human_human_edge_embedding_size == 64
        assert args.attention_size == 64
        # Rollout bookkeeping.
        assert args.seq_length == 30
        assert args.num_processes == 1
        assert args.num_mini_batch == 2

    def test_default_namespace_is_a_simple_namespace(self) -> None:
        """The default args namespace is a ``types.SimpleNamespace`` (upstream shape)."""
        assert isinstance(_default_args_namespace(), SimpleNamespace)

    def test_default_namespace_is_lru_cached(self) -> None:
        """``_default_args_namespace`` returns one shared cached namespace object."""
        assert _default_args_namespace() is _default_args_namespace()

    def test_build_args_namespace_overrides_propagate(self) -> None:
        """Explicit overrides for the variadic fields must land on the namespace."""
        args = _build_args_namespace(
            env_name="CrowdSimVarNum-v0",
            use_self_attn=False,
            use_hr_attn=False,
            sort_humans=False,
            no_cuda=False,
            num_processes=4,
        )
        assert args.env_name == "CrowdSimVarNum-v0"
        assert args.use_self_attn is False
        assert args.use_hr_attn is False
        assert args.sort_humans is False
        assert args.no_cuda is False
        assert args.num_processes == 4

    def test_build_args_namespace_keeps_cuda_false_regardless_of_no_cuda(self) -> None:
        """``cuda`` stays False even when ``no_cuda`` is False (smoke runs on CPU)."""
        args = _build_args_namespace(
            env_name="x",
            use_self_attn=True,
            use_hr_attn=True,
            sort_humans=True,
            no_cuda=False,
            num_processes=1,
        )
        assert args.no_cuda is False
        assert args.cuda is False

    def test_build_args_namespace_preserves_frozen_sizes(self) -> None:
        """The non-variadic architecture sizes are constant across overrides."""
        base = _default_args_namespace()
        overridden = _build_args_namespace(
            env_name="other",
            use_self_attn=False,
            use_hr_attn=False,
            sort_humans=False,
            no_cuda=False,
            num_processes=8,
        )
        for field in (
            "env_type",
            "human_node_rnn_size",
            "human_human_edge_rnn_size",
            "human_node_output_size",
            "human_node_input_size",
            "human_human_edge_input_size",
            "human_node_embedding_size",
            "human_human_edge_embedding_size",
            "attention_size",
            "seq_length",
            "num_mini_batch",
        ):
            assert getattr(overridden, field) == getattr(base, field)


# --------------------------------------------------------------------------- #
# _pred_attng_import_context: sys.path / sys.modules restoration              #
# --------------------------------------------------------------------------- #
class TestImportContext:
    """``_pred_attng_import_context`` path/module restoration contracts."""

    def test_context_exposes_repo_on_path_and_envs_stub(self, tmp_path: Path) -> None:
        """Inside the block the repo root leads ``sys.path`` and the envs stub is live."""
        repo = tmp_path / "fake_crowdnav"
        path_before = list(sys.path)
        modules_before = set(sys.modules)
        try:
            with _pred_attng_import_context(repo):
                # The repo root is inserted at the front so upstream imports resolve.
                assert sys.path[0] == str(repo)
                assert str(repo) in sys.path
                # A minimal rl.networks.envs stub with VecNormalize is injected.
                assert "rl.networks.envs" in sys.modules
                stub = sys.modules["rl.networks.envs"]
                assert isinstance(stub, ModuleType)
                assert hasattr(stub, "VecNormalize")
        finally:
            # State must be fully restored even on the success path.
            assert sys.path == path_before
            assert set(sys.modules) == modules_before

    def test_context_restores_path_and_removes_stub_on_clean_exit(self, tmp_path: Path) -> None:
        """A clean exit restores ``sys.path`` content and drops the injected stub."""
        repo = tmp_path / "fake_crowdnav"
        path_before = list(sys.path)
        assert "rl.networks.envs" not in sys.modules

        with _pred_attng_import_context(repo):
            assert sys.path[0] == str(repo)
            assert "rl.networks.envs" in sys.modules

        assert sys.path == path_before
        assert str(repo) not in sys.path
        assert "rl.networks.envs" not in sys.modules

    def test_context_restores_state_and_propagates_exception(self, tmp_path: Path) -> None:
        """An exception inside the block must still restore state and then propagate."""
        repo = tmp_path / "fake_crowdnav"
        path_before = list(sys.path)
        modules_before = set(sys.modules)

        class _SimulatedFailure(RuntimeError):
            pass

        with pytest.raises(_SimulatedFailure):
            with _pred_attng_import_context(repo):
                assert sys.path[0] == str(repo)
                assert "rl.networks.envs" in sys.modules
                raise _SimulatedFailure("boom inside context")

        # The finally block must have run despite the exception.
        assert sys.path == path_before
        assert str(repo) not in sys.path
        assert set(sys.modules) == modules_before
        assert "rl.networks.envs" not in sys.modules

    def test_context_restores_a_pre_existing_envs_module(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pre-existing ``rl.networks.envs`` module is restored by identity on exit."""
        repo = tmp_path / "fake_crowdnav"
        original = ModuleType("rl.networks.envs")
        original.VecNormalize = "ORIGINAL_MARKER"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "rl.networks.envs", original)

        with _pred_attng_import_context(repo):
            inside = sys.modules["rl.networks.envs"]
            # The stub temporarily shadows the original module.
            assert inside is not original
            assert inside.VecNormalize.__name__ == "_VecNormalize"

        # The original module is restored by identity, not replaced with a copy.
        assert sys.modules["rl.networks.envs"] is original
        assert sys.modules["rl.networks.envs"].VecNormalize == "ORIGINAL_MARKER"


# --------------------------------------------------------------------------- #
# _build_observation_space: dimension contracts                               #
# --------------------------------------------------------------------------- #
class TestObservationSpace:
    """``_build_observation_space`` upstream dict-layout dimension contracts."""

    def test_returns_gymnasium_dict_space(self) -> None:
        """The observation space is a Gymnasium ``Dict`` over the upstream keys."""
        space = _build_observation_space(DEFAULT_HUMAN_NUM, DEFAULT_PREDICT_STEPS)
        assert isinstance(space, spaces.Dict)
        assert set(space.spaces) == {
            "robot_node",
            "temporal_edges",
            "spatial_edges",
            "detected_human_num",
            "visible_masks",
        }

    def test_default_dimensions_match_upstream_checkpoint_contract(self) -> None:
        """Each node matches the 41200.pt attention-graph input contract."""
        space = _build_observation_space(DEFAULT_HUMAN_NUM, DEFAULT_PREDICT_STEPS)
        assert space.spaces["robot_node"].shape == (1, 7)
        assert space.spaces["temporal_edges"].shape == (1, 2)
        # spatial_edges: human_num rows, width 2*(predict_steps+1) = 12.
        assert space.spaces["spatial_edges"].shape == (DEFAULT_HUMAN_NUM, 12)
        assert space.spaces["detected_human_num"].shape == (1,)
        assert space.spaces["visible_masks"].shape == (DEFAULT_HUMAN_NUM,)

    def test_all_node_spaces_are_unbounded_float32_boxes(self) -> None:
        """Every node is an unbounded (-inf, inf) float32 Box, mirroring upstream."""
        space = _build_observation_space(DEFAULT_HUMAN_NUM, DEFAULT_PREDICT_STEPS)
        for box in space.spaces.values():
            assert isinstance(box, spaces.Box)
            assert box.dtype == np.float32
            assert np.isneginf(box.low).all()
            assert np.isposinf(box.high).all()

    @pytest.mark.parametrize(
        ("human_num", "predict_steps", "expected_edge_width"),
        [
            (20, 5, 12),
            (7, 3, 8),
            (1, 0, 2),
        ],
    )
    def test_spatial_and_mask_dimensions_parameterize_correctly(
        self, human_num: int, predict_steps: int, expected_edge_width: int
    ) -> None:
        """spatial_edges/visible_masks scale with human_num and predict_steps."""
        space = _build_observation_space(human_num, predict_steps)
        assert space.spaces["robot_node"].shape == (1, 7)
        assert space.spaces["temporal_edges"].shape == (1, 2)
        assert space.spaces["spatial_edges"].shape == (human_num, expected_edge_width)
        assert space.spaces["detected_human_num"].shape == (1,)
        assert space.spaces["visible_masks"].shape == (human_num,)


# --------------------------------------------------------------------------- #
# _xy_rows: scalar/flat/matrix XY normalization                               #
# --------------------------------------------------------------------------- #
class TestXyRows:
    """``_xy_rows`` normalization of arbitrary XY payloads to ``(N, 2)`` arrays."""

    @pytest.mark.parametrize("value", [None, [], [[]]])
    def test_empty_or_missing_payloads_yield_empty_two_column_array(self, value: object) -> None:
        """None / empty / zero-width payloads normalize to a (0, 2) float array."""
        result = _xy_rows(value)
        assert result.shape == (0, 2)
        assert result.dtype == np.float64

    @pytest.mark.parametrize("value", [5.0, [1.0, 2.0], [1.0, 2.0, 3.0, 4.0]])
    def test_scalar_and_flat_payloads_yield_empty_rows(self, value: object) -> None:
        """A scalar or a flat 1-D list is not a row of XY pairs, so it yields (0, 2).

        The helper reshapes flat arrays to ``(N, 1)`` which has fewer than two
        columns; only 2-D ``[[x, y], ...]`` matrices carry XY rows.
        """
        result = _xy_rows(value)
        assert result.shape == (0, 2)

    def test_single_row_matrix_normalizes_to_one_row(self) -> None:
        """A ``[[x, y]]`` matrix yields a single ``(1, 2)`` row."""
        result = _xy_rows([[1.0, 2.0]])
        assert result.shape == (1, 2)
        assert result.tolist() == [[1.0, 2.0]]

    def test_multi_row_matrix_preserves_each_xy_row(self) -> None:
        """A multi-row matrix returns each XY row in order."""
        result = _xy_rows([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert result.shape == (3, 2)
        assert result.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    def test_extra_columns_are_truncated_to_xy(self) -> None:
        """Columns beyond XY (e.g. radius/z) are dropped, keeping only the first two."""
        result = _xy_rows([[1.0, 2.0, 9.0], [3.0, 4.0, 9.0]])
        assert result.shape == (2, 2)
        assert result.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_integer_matrix_is_coerced_to_float(self) -> None:
        """Integer-coordinate matrices are coerced to a float array."""
        result = _xy_rows([[1, 2], [3, 4]])
        assert result.dtype == np.float64
        assert result.tolist() == [[1.0, 2.0], [3.0, 4.0]]


# --------------------------------------------------------------------------- #
# _require_xy: required 2-vector field errors                                  #
# --------------------------------------------------------------------------- #
class TestRequireXy:
    """``_require_xy`` required-field validation and 2-vector extraction."""

    @pytest.mark.parametrize(
        ("value", "label"),
        [
            (None, "None payload"),
            ([], "empty list"),
            (5.0, "scalar"),
            ([1.0], "single element"),
        ],
    )
    def test_malformed_required_fields_raise_with_field_name(
        self, value: object, label: str
    ) -> None:
        """Missing / scalar / sub-two-element payloads raise ``ValueError`` naming the field."""
        with pytest.raises(ValueError, match="Missing or malformed required field: robot.position"):
            _require_xy(value, "robot.position")

    def test_flat_xy_payload_returns_two_element_array(self) -> None:
        """A flat ``[x, y]`` payload resolves to a two-element float array."""
        result = _require_xy([1.0, 2.0], "robot.position")
        assert result.shape == (2,)
        assert result.tolist() == [1.0, 2.0]

    def test_nested_xy_payload_flattens_to_two_element_array(self) -> None:
        """A nested ``[[x, y]]`` payload flattens to the same two-element array."""
        result = _require_xy([[1.0, 2.0]], "robot.position")
        assert result.shape == (2,)
        assert result.tolist() == [1.0, 2.0]

    def test_extra_elements_are_truncated_to_first_two(self) -> None:
        """Payloads longer than two elements keep only the first two components."""
        result = _require_xy([1.0, 2.0, 3.0, 4.0], "goal")
        assert result.tolist() == [1.0, 2.0]

    def test_nested_extra_elements_are_truncated_to_first_two(self) -> None:
        """A nested 3-vector flattens and is truncated to its first two components."""
        result = _require_xy([[1.0, 2.0, 3.0]], "goal")
        assert result.tolist() == [1.0, 2.0]

    def test_field_name_is_rendered_in_error_message(self) -> None:
        """The raised error message interpolates the supplied field name verbatim."""
        with pytest.raises(ValueError, match="Missing or malformed required field: goal_current"):
            _require_xy(None, "goal_current")

    @pytest.mark.parametrize("value", [[np.nan, 1.0], [np.inf, 1.0], [1.0, -np.inf]])
    def test_non_finite_required_fields_raise_with_field_name(self, value: list[float]) -> None:
        """Required XY vectors reject NaN and infinity rather than reaching the policy."""
        with pytest.raises(ValueError, match="Missing or malformed required field: robot.position"):
            _require_xy(value, "robot.position")


# --------------------------------------------------------------------------- #
# _clip_holonomic_to_v_pref: preferred-speed magnitude clipping               #
# --------------------------------------------------------------------------- #
class TestClipHolonomicToVPref:
    """``_clip_holonomic_to_v_pref`` preferred-speed envelope clipping."""

    def test_within_envelope_is_unchanged(self) -> None:
        """A velocity already inside the envelope is returned unchanged."""
        assert _clip_holonomic_to_v_pref(0.3, 0.4, 1.0) == (0.3, 0.4)

    def test_exactly_at_envelope_is_unchanged(self) -> None:
        """A velocity whose magnitude equals ``v_pref`` is left unchanged (no over-clip)."""
        assert _clip_holonomic_to_v_pref(1.0, 0.0, 1.0) == (1.0, 0.0)

    def test_above_envelope_is_clipped_to_v_pref_magnitude(self) -> None:
        """A velocity above the envelope is scaled down to magnitude ``v_pref``."""
        vx, vy = _clip_holonomic_to_v_pref(2.0, 0.0, 1.0)
        assert vx == pytest.approx(1.0)
        assert vy == pytest.approx(0.0)
        assert np.hypot(vx, vy) == pytest.approx(1.0)

    def test_above_envelope_preserves_direction(self) -> None:
        """Clipping scales magnitude without changing the direction (signs retained)."""
        vx, vy = _clip_holonomic_to_v_pref(-3.0, -4.0, 1.0)
        assert np.hypot(vx, vy) == pytest.approx(1.0)
        assert vx < 0.0 and vy < 0.0
        # Direction preserved: unit vector is (-0.6, -0.8).
        assert vx == pytest.approx(-0.6)
        assert vy == pytest.approx(-0.8)

    def test_zero_velocity_is_unchanged_at_nonzero_v_pref(self) -> None:
        """A zero velocity is returned as-is (the norm>0 guard avoids division by zero)."""
        assert _clip_holonomic_to_v_pref(0.0, 0.0, 1.0) == (0.0, 0.0)

    def test_nonzero_velocity_clipped_to_zero_at_zero_v_pref(self) -> None:
        """With ``v_pref == 0`` a nonzero command collapses to the zero vector."""
        vx, vy = _clip_holonomic_to_v_pref(2.0, 0.0, 0.0)
        assert vx == pytest.approx(0.0)
        assert vy == pytest.approx(0.0)
        assert np.hypot(vx, vy) == pytest.approx(0.0)

    def test_zero_velocity_unchanged_at_zero_v_pref(self) -> None:
        """At zero ``v_pref`` a zero velocity stays zero (no NaN from 0/0)."""
        vx, vy = _clip_holonomic_to_v_pref(0.0, 0.0, 0.0)
        assert vx == pytest.approx(0.0)
        assert vy == pytest.approx(0.0)
        assert np.isfinite(vx) and np.isfinite(vy)

    def test_clipped_output_never_exceeds_v_pref(self) -> None:
        """Across a spread of magnitudes the clipped norm never exceeds ``v_pref``."""
        for vx, vy, v_pref in [
            (1.5, 2.0, 1.0),
            (0.1, 0.1, 0.5),
            (10.0, 10.0, 1.0),
            (0.7, 0.7, 0.2),
        ]:
            cx, cy = _clip_holonomic_to_v_pref(vx, vy, v_pref)
            assert np.hypot(cx, cy) <= v_pref + 1e-9
