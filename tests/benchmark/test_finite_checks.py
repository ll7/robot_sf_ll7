"""Contract-parity tests for consolidated finite-value validation helpers."""

from __future__ import annotations

import ast
import inspect
import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robot_sf import common
from robot_sf.benchmark import (
    actuator_feasibility,
    clearance_semantics,
    counterfactual_pair,
    finite_checks,
    heterogeneous_population_metrics,
    pedestrian_flow_validation,
    pedestrian_model_fixture_diagnostics,
    reactivity_ablation,
    safety_predicates,
    trajectory_verifier,
)
from robot_sf.benchmark.scenario_generation import catalog_schema
from robot_sf.common import validation
from robot_sf.nav import occupancy_grid, proxemic_costmap
from robot_sf.planner import stream_gap_gate_calibration
from robot_sf.representation import uncertainty_source_generalization
from robot_sf.research import zanlungo_corridor_acceptance
from robot_sf.training import discrete_action_lattice


@dataclass(frozen=True)
class _ErrorContract:
    value: Any
    exception: type[BaseException]
    message: str


def test_public_scalar_entry_point_and_exports() -> None:
    """New callers receive one strict public scalar API without exporting implementation helpers."""
    assert common.require_finite is validation.require_finite
    assert validation.require_finite("metric", -1, allow_negative=True) == -1.0
    assert validation.require_finite("metric", -0.0, allow_negative=False) == -0.0
    with pytest.raises(ValueError, match=r"^metric must be non-negative: -1\.0$"):
        validation.require_finite("metric", -1, allow_negative=False)

    assert validation.__all__ == [
        "require_finite",
        "require_finite_array",
        "require_finite_fields",
        "require_finite_scalar",
    ]
    assert "_require_finite" not in common.__all__


def test_benchmark_finite_checks_preserve_backward_compatible_reexports() -> None:
    """The historical benchmark import path remains identity-equivalent to the shared owner."""
    assert finite_checks.require_finite_scalar is validation.require_finite_scalar
    assert finite_checks.require_finite_array is validation.require_finite_array
    assert finite_checks.require_finite_fields is validation.require_finite_fields
    assert finite_checks.__all__ == [
        "require_finite_array",
        "require_finite_fields",
        "require_finite_scalar",
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(1, 1.0, id="int"),
        pytest.param(-1.25, -1.25, id="negative-float"),
        pytest.param(True, 1.0, id="bool-is-historical-real"),
        pytest.param(Fraction(1, 4), 0.25, id="fraction"),
        pytest.param(np.float32(1.5), 1.5, id="numpy-real"),
    ],
)
def test_require_finite_scalar_acceptance_and_return_type(value: Any, expected: float) -> None:
    result = validation.require_finite_scalar("metric", value)
    assert type(result) is float
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "contract",
    [
        pytest.param(
            _ErrorContract("1.25", TypeError, "metric must be a real numeric scalar, got str"),
            id="string",
        ),
        pytest.param(
            _ErrorContract(1 + 2j, TypeError, "metric must be a real numeric scalar, got complex"),
            id="complex",
        ),
        pytest.param(
            _ErrorContract(
                Decimal("1.25"), TypeError, "metric must be a real numeric scalar, got Decimal"
            ),
            id="decimal-not-real",
        ),
        pytest.param(
            _ErrorContract(object(), TypeError, "metric must be a real numeric scalar, got object"),
            id="object",
        ),
        pytest.param(_ErrorContract(math.nan, ValueError, "metric is not finite: nan"), id="nan"),
        pytest.param(_ErrorContract(math.inf, ValueError, "metric is not finite: inf"), id="inf"),
        pytest.param(
            _ErrorContract(-math.inf, ValueError, "metric is not finite: -inf"), id="neg-inf"
        ),
    ],
)
def test_require_finite_scalar_rejection_contract(contract: _ErrorContract) -> None:
    with pytest.raises(contract.exception) as caught:
        validation.require_finite_scalar("metric", contract.value)
    assert str(caught.value) == contract.message


def test_require_finite_array_preserves_dtype_shape_and_iterable_contract() -> None:
    matrix = validation.require_finite_array("trace", [[1, 2], [True, False]])
    assert matrix.dtype == np.dtype(np.float64)
    assert matrix.shape == (2, 2)
    assert matrix.tolist() == [[1.0, 2.0], [1.0, 0.0]]

    vector = validation.require_finite_array("trace", range(3))
    assert vector.dtype == np.dtype(np.float64)
    assert vector.shape == (3,)

    scalar = validation.require_finite_array("trace", "1.25")
    assert scalar.dtype == np.dtype(np.float64)
    assert scalar.shape == ()
    assert scalar.item() == pytest.approx(1.25)

    with pytest.raises(TypeError):
        validation.require_finite_array("trace", (value for value in (1.0, 2.0)))
    with pytest.raises(TypeError):
        validation.require_finite_array("trace", [1 + 2j])
    with pytest.raises(TypeError):
        validation.require_finite_array("trace", object())

    for non_finite in (math.nan, math.inf, -math.inf):
        with pytest.raises(
            ValueError,
            match=r"^trace must contain only finite values$",
        ):
            validation.require_finite_array("trace", [0.0, non_finite])


def test_require_finite_fields_accepts_generator_and_preserves_field_path() -> None:
    class Row:
        safe = 0.0
        unsafe = math.inf

    assert validation.require_finite_fields("row", Row(), (name for name in ("safe",))) is None
    with pytest.raises(ValueError, match=r"^row\.unsafe is not finite: inf$"):
        validation.require_finite_fields("row", Row(), ("safe", "unsafe"))
    with pytest.raises(AttributeError, match="missing"):
        validation.require_finite_fields("row", Row(), ("missing",))


def test_local_helper_alias_matrix_covers_every_direct_replacement() -> None:
    """Every removed direct helper remains importable from its historical module path."""
    aliases = [
        (actuator_feasibility._require_finite, validation._require_finite_coerce),
        (
            clearance_semantics._require_finite_non_negative,
            validation._require_finite_non_negative_coerce,
        ),
        (pedestrian_flow_validation._require_finite_number, validation._require_finite_number),
        (
            pedestrian_model_fixture_diagnostics._require_finite_number,
            validation._require_finite_number,
        ),
        (trajectory_verifier._require_finite, validation._require_finite_ndarray),
        (occupancy_grid._require_finite, validation._require_finite),
        (proxemic_costmap._require_finite_non_negative, validation._require_finite_non_negative),
        (
            zanlungo_corridor_acceptance._require_finite_real,
            validation._require_finite_real,
        ),
        (
            discrete_action_lattice._require_finite_bounded_values,
            validation._require_finite_bounded_values,
        ),
    ]
    assert all(local is shared for local, shared in aliases)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(2, 2.0, id="int"),
        pytest.param(True, 1.0, id="bool"),
        pytest.param("1.25", 1.25, id="numeric-string"),
        pytest.param(np.float64(1.5), 1.5, id="numpy-scalar"),
    ],
)
def test_coercing_finite_helper_acceptance(value: Any, expected: float) -> None:
    result = validation._require_finite_coerce(value, key="speed")
    assert type(result) is float
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("value", [1 + 2j, object()])
def test_coercing_finite_helper_wraps_conversion_errors(value: Any) -> None:
    with pytest.raises(ValueError, match=r"^speed must be numeric$") as caught:
        validation._require_finite_coerce(value, key="speed")
    assert isinstance(caught.value.__cause__, TypeError)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        pytest.param(math.nan, "speed must be finite; got nan", id="nan"),
        pytest.param(math.inf, "speed must be finite; got inf", id="inf"),
        pytest.param("nan", "speed must be finite; got nan", id="nan-string"),
    ],
)
def test_coercing_finite_helper_non_finite_message(value: Any, message: str) -> None:
    with pytest.raises(ValueError) as caught:
        validation._require_finite_coerce(value, key="speed")
    assert str(caught.value) == message


def test_non_negative_coercing_helper_contract() -> None:
    assert validation._require_finite_non_negative_coerce("1.25", key="radius") == 1.25
    assert validation._require_finite_non_negative_coerce(True, key="radius") == 1.0
    assert validation._require_finite_non_negative_coerce(-0.0, key="radius") == -0.0

    cases = [
        (1 + 2j, "radius must be numeric"),
        (object(), "radius must be numeric"),
        (math.nan, "radius must be finite"),
        (math.inf, "radius must be finite"),
        (-1, "radius must be non-negative"),
    ]
    for value, message in cases:
        with pytest.raises(ValueError) as caught:
            validation._require_finite_non_negative_coerce(value, key="radius")
        assert str(caught.value) == message


def test_non_coercing_scalar_and_non_negative_contracts() -> None:
    assert validation._require_finite("width", True) is None
    assert validation._require_finite_non_negative("radius", "1.25") is None
    assert validation._require_finite_non_negative("radius", -0.0) is None

    with pytest.raises(ValueError, match=r"^width must be finite, got nan$"):
        validation._require_finite("width", math.nan)
    with pytest.raises(TypeError):
        validation._require_finite("width", "1.25")
    with pytest.raises(TypeError):
        validation._require_finite("width", 1 + 2j)

    for value in (-1, math.nan, math.inf):
        with pytest.raises(ValueError, match=r"^radius must be finite and >= 0$"):
            validation._require_finite_non_negative("radius", value)
    with pytest.raises(TypeError):
        validation._require_finite_non_negative("radius", 1 + 2j)


def test_ndarray_helper_preserves_shape_agnostic_and_complex_acceptance() -> None:
    assert validation._require_finite_ndarray("trajectory", np.array(1.0)) is None
    assert validation._require_finite_ndarray("trajectory", np.ones((2, 1))) is None
    assert validation._require_finite_ndarray("trajectory", np.array([1 + 2j])) is None
    assert validation._require_finite_ndarray("trajectory", [1.0, 2.0]) is None

    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(
            ValueError,
            match=r"^trajectory must contain only finite values \(no NaN or inf\)$",
        ):
            validation._require_finite_ndarray("trajectory", np.array([0.0, value]))
    with pytest.raises(TypeError):
        validation._require_finite_ndarray("trajectory", np.array(["1.0"]))
    with pytest.raises(TypeError):
        validation._require_finite_ndarray("trajectory", object())


def test_run_control_number_helper_preserves_bool_coercion_and_sign_bounds() -> None:
    assert validation._require_finite_number("1.25", "horizon") is None
    with pytest.raises(ValueError, match=r"^horizon must be finite$"):
        validation._require_finite_number(True, "horizon")
    with pytest.raises(ValueError, match=r"^horizon must be finite$"):
        validation._require_finite_number(math.nan, "horizon")
    with pytest.raises(ValueError, match=r"^horizon must be positive$"):
        validation._require_finite_number(0, "horizon", positive=True)
    with pytest.raises(ValueError, match=r"^horizon must be non-negative$"):
        validation._require_finite_number(-1, "horizon", non_negative=True)
    with pytest.raises(ValueError, match=r"^horizon must be positive$"):
        validation._require_finite_number(
            -1,
            "horizon",
            positive=True,
            non_negative=True,
        )
    with pytest.raises(TypeError):
        validation._require_finite_number(1 + 2j, "horizon")
    with pytest.raises(TypeError):
        validation._require_finite_number(object(), "horizon")


def test_bounded_values_helper_preserves_iterable_and_bound_contract() -> None:
    kwargs = {
        "field_name": "linear_values",
        "max_abs_value": 1.0,
        "max_field_name": "max_linear_speed",
    }
    assert validation._require_finite_bounded_values(values=(0.0, 1.0), **kwargs) is None
    assert validation._require_finite_bounded_values(values=[True], **kwargs) is None
    assert (
        validation._require_finite_bounded_values(
            values=(value for value in (0.0, 1.0)),
            **kwargs,
        )
        is None
    )
    # Historical runtime behavior: an empty generator is truthy, so only an empty tuple is rejected.
    assert (
        validation._require_finite_bounded_values(
            values=(value for value in ()),
            **kwargs,
        )
        is None
    )

    with pytest.raises(
        ValueError,
        match=r"^linear_values must contain at least one command value$",
    ):
        validation._require_finite_bounded_values(values=(), **kwargs)
    with pytest.raises(ValueError, match=r"^linear_values must be finite$"):
        validation._require_finite_bounded_values(values=(math.nan,), **kwargs)
    with pytest.raises(ValueError, match=r"^linear_values exceed max_linear_speed$"):
        validation._require_finite_bounded_values(values=(1.01,), **kwargs)
    with pytest.raises(TypeError):
        validation._require_finite_bounded_values(values=("1.0",), **kwargs)
    with pytest.raises(TypeError):
        validation._require_finite_bounded_values(values=(1 + 2j,), **kwargs)


def test_finite_real_helper_preserves_strict_real_and_bool_rejection() -> None:
    assert validation._require_finite_real(Fraction(1, 4), "speed") == 0.25
    assert validation._require_finite_real(np.float64(1.5), "speed") == 1.5
    for value in (True, "1.25", 1 + 2j, Decimal("1.25"), object(), math.nan, math.inf):
        with pytest.raises(
            ValueError,
            match=r"^speed must be a finite real number, not a boolean$",
        ):
            validation._require_finite_real(value, "speed")


def test_position_helper_and_catalog_exception_wrapper_contract() -> None:
    assert validation._require_finite_position([]) is None
    assert validation._require_finite_position([0, 1.5, True]) is None
    assert validation._require_finite_position(value for value in (0.0, 1.0)) is None

    for value in ("1.0", 1 + 2j, object(), math.nan, math.inf, np.int64(1)):
        with pytest.raises(ValueError, match=r"^trace positions must be finite numbers$"):
            validation._require_finite_position([value])

    with pytest.raises(
        catalog_schema.GeneratedScenarioCatalogValidationError,
        match=r"^trace positions must be finite numbers$",
    ) as caught:
        catalog_schema._require_finite_position([math.nan])
    assert isinstance(caught.value.__cause__, ValueError)


def test_preserved_thin_wrappers_keep_none_return_and_field_context() -> None:
    assert heterogeneous_population_metrics._require_finite("metric", 1.0) is None
    with pytest.raises(ValueError, match=r"^metric is not finite: inf$"):
        heterogeneous_population_metrics._require_finite("metric", math.inf)

    setting = stream_gap_gate_calibration.GateSettingResult(
        thresholds={"existence": 0.5},
        unsafe_commit_rate=0.0,
        collision_rate=0.0,
        min_separation_m=1.0,
    )
    assert stream_gap_gate_calibration._require_finite_setting(setting, "setting") is None

    bad_setting = stream_gap_gate_calibration.GateSettingResult(
        thresholds={"existence": 0.5},
        unsafe_commit_rate=math.nan,
        collision_rate=0.0,
        min_separation_m=1.0,
    )
    with pytest.raises(ValueError, match=r"^setting\.unsafe_commit_rate is not finite: nan$"):
        stream_gap_gate_calibration._require_finite_setting(bad_setting, "setting")


def test_stream_gap_classifier_rejects_non_finite_safety_aggregate() -> None:
    baseline = stream_gap_gate_calibration.GateSettingResult(
        thresholds={"existence": 0.0},
        unsafe_commit_rate=0.0,
        collision_rate=0.0,
        min_separation_m=1.0,
    )
    setting = stream_gap_gate_calibration.GateSettingResult(
        thresholds={"existence": 0.5},
        unsafe_commit_rate=math.nan,
        collision_rate=0.0,
        min_separation_m=1.0,
    )
    with pytest.raises(ValueError, match=r"setting\.unsafe_commit_rate"):
        stream_gap_gate_calibration.classify_setting_safety(setting, baseline)


@pytest.mark.parametrize(
    "module",
    [
        counterfactual_pair,
        heterogeneous_population_metrics,
        reactivity_ablation,
        safety_predicates,
        stream_gap_gate_calibration,
        uncertainty_source_generalization,
    ],
)
def test_diagnostic_producers_import_shared_finite_policy(module: object) -> None:
    """Recent diagnostic producers use the shared fail-closed finite policy."""
    source = inspect.getsource(module)
    assert "robot_sf.common.validation" in source or "robot_sf.benchmark.finite_checks" in source


@pytest.mark.parametrize(
    "relative_path",
    [
        "robot_sf/benchmark/actuator_feasibility.py",
        "robot_sf/benchmark/clearance_semantics.py",
        "robot_sf/benchmark/pedestrian_flow_validation.py",
        "robot_sf/benchmark/pedestrian_model_fixture_diagnostics.py",
        "robot_sf/benchmark/scenario_generation/catalog_schema.py",
        "robot_sf/benchmark/trajectory_verifier.py",
        "robot_sf/nav/occupancy_grid.py",
        "robot_sf/nav/proxemic_costmap.py",
        "robot_sf/research/zanlungo_corridor_acceptance.py",
        "robot_sf/training/discrete_action_lattice.py",
    ],
)
def test_shared_validation_imports_precede_runtime_definitions(relative_path: str) -> None:
    """Consolidation imports stay in the normal import layer, never below runtime declarations."""
    path = Path(relative_path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    import_lines = [
        node.lineno
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "robot_sf.common.validation"
    ]
    runtime_lines = [
        node.lineno
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert import_lines
    assert max(import_lines) < min(runtime_lines)
