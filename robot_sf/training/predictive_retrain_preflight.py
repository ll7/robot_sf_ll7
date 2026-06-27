"""Static launch preflight for crossing-conflict predictive retraining configs.

This module provides a cheap, CPU-only validation of a predictive training
pipeline config (the kind consumed by
``scripts/training/run_predictive_training_pipeline.py``) *before* any SLURM/GPU
retraining run is launched. It exists because the pipeline's own guards
(producer-metadata and obstacle-feature preflights) only run mid-pipeline, after
expensive base/hard-case dataset collection. Catching a missing scenario matrix,
a mismatched ego-conditioning width, an invalid weighting rule, or a missing
evaluation block statically saves a wasted launch.

Scope boundary: this is launch/config preflight tooling only. It validates that
a config is internally consistent and that its referenced inputs exist. It does
not collect data, train a model, submit a SLURM job, change augmentation
semantics, or make any predictive model-improvement claim. Related issues:
``#3214`` (weighting spec + tooling) and ``#3254`` (the actual retraining launch).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Mirror the weighting rule accepted by build_predictive_mixed_dataset.py so the
# preflight fails closed on specs the dataset builder would reject downstream.
_SUPPORTED_WEIGHTING_RULE = "repeat_hardcase_rows"

# Data-prerequisite references resolved relative to the config directory.
_REQUIRED_SCENARIO_REFERENCES = (
    "scenario_matrix",
    "hard_seed_manifest",
    "planner_grid",
)

# Evaluation keys the pipeline reads when wiring final eval / hard-seed campaign.
_REQUIRED_EVALUATION_KEYS = ("workers", "horizon", "dt")


class PredictiveRetrainPreflightError(ValueError):
    """Raised when a predictive retraining launch config fails preflight."""


def load_pipeline_config(config_path: Path) -> dict[str, Any]:
    """Load a predictive pipeline config and return its mapping.

    Returns:
        Parsed config mapping.
    """
    if not config_path.is_file():
        raise PredictiveRetrainPreflightError(f"config is not a file: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PredictiveRetrainPreflightError("config must be a YAML mapping")
    return payload


def validate_retrain_preflight(
    config_path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate a crossing-conflict predictive retraining launch config.

    The check is static: it confirms config structure, that referenced data
    prerequisites exist, that base/hard-case feature widths are compatible (the
    checkpoint-lineage contract), that the navigation gate stays separate from
    the trajectory gate, that an evaluation block is present, and that an output
    root is declared. It never executes the pipeline.

    Returns:
        Compact validation report on success.
    """
    root = (repo_root or Path.cwd()).resolve()
    config_path = _resolve_path(config_path, root)
    config = load_pipeline_config(config_path)
    # Most paths inside a pipeline config are config-relative, matching how the
    # pipeline resolves scenario/manifest/weighting references.
    config_dir = config_path.parent
    errors: list[str] = []

    _require_non_empty_string(config, "model_family", errors)
    _validate_experiment(config, errors)
    scenarios = _validate_scenarios(config, config_dir, errors)
    feature_contract = _validate_feature_compatibility(config, errors)
    mixing = _validate_mixing(config, config_dir, errors)
    training = _validate_training(config, errors)
    evaluation = _validate_evaluation(config, errors)
    output_root = _validate_output(config, errors)

    if errors:
        joined = "\n- ".join(errors)
        raise PredictiveRetrainPreflightError(
            f"predictive retrain launch preflight failed:\n- {joined}"
        )

    return {
        "status": "valid",
        "config_path": str(config_path),
        "model_family": config["model_family"],
        "run_id": config["experiment"]["run_id"],
        "scenarios": scenarios,
        "feature_compatibility": feature_contract,
        "mixing": mixing,
        "training": training,
        "evaluation": evaluation,
        "output_root": output_root,
    }


def _resolve_path(path: Path | str, base: Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def _require_non_empty_string(mapping: dict[str, Any], key: str, errors: list[str]) -> Any:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} must be a non-empty string")
        return None
    return value


def _require_mapping(mapping: dict[str, Any], key: str, errors: list[str]) -> dict[str, Any] | None:
    value = mapping.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key} must be a mapping")
        return None
    return value


def _require_positive_int(mapping: dict[str, Any], key: str, label: str, errors: list[str]) -> None:
    value = mapping.get(key)
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        errors.append(f"{label}.{key} must be an integer")
        return
    if as_int < 1:
        errors.append(f"{label}.{key} must be >= 1")


def _validate_experiment(config: dict[str, Any], errors: list[str]) -> None:
    experiment = _require_mapping(config, "experiment", errors)
    if experiment is None:
        return
    _require_non_empty_string(experiment, "run_id", errors)


def _validate_scenarios(
    config: dict[str, Any],
    config_dir: Path,
    errors: list[str],
) -> dict[str, str] | None:
    scenarios = _require_mapping(config, "scenarios", errors)
    if scenarios is None:
        return None
    resolved: dict[str, str] = {}
    for key in _REQUIRED_SCENARIO_REFERENCES:
        value = scenarios.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"scenarios.{key} must be a non-empty path string")
            continue
        path = _resolve_path(value, config_dir)
        if not path.is_file():
            errors.append(f"scenarios.{key} does not exist: {value}")
            continue
        resolved[key] = str(path)
    return resolved


def _validate_feature_compatibility(
    config: dict[str, Any],
    errors: list[str],
) -> dict[str, Any] | None:
    """Validate base/hard-case collections share a feature layout (lineage contract).

    Returns:
        Resolved feature contract, or ``None`` when a collection block is absent.
    """
    base = _require_mapping(config, "base_collection", errors)
    hardcase = _require_mapping(config, "hardcase_collection", errors)
    if base is None or hardcase is None:
        return None
    top_family = config.get("model_family")
    base_family = str(base.get("model_family", top_family))
    hardcase_family = str(hardcase.get("model_family", top_family))
    if base_family != hardcase_family:
        errors.append(
            "base_collection and hardcase_collection model families must match: "
            f"{base_family!r} != {hardcase_family!r}"
        )
    base_ego = bool(base.get("ego_conditioning", False))
    hardcase_ego = bool(hardcase.get("ego_conditioning", False))
    if base_ego != hardcase_ego:
        errors.append(
            "base_collection.ego_conditioning and hardcase_collection.ego_conditioning "
            f"must match for compatible feature widths: {base_ego} != {hardcase_ego}"
        )
    return {
        "model_family": base_family if base_family == hardcase_family else None,
        "ego_conditioning": base_ego if base_ego == hardcase_ego else None,
    }


def _validate_mixing(
    config: dict[str, Any],
    config_dir: Path,
    errors: list[str],
) -> dict[str, Any] | None:
    """Validate the hard-case mixing/weighting prerequisites without altering them.

    Returns:
        Compact mixing summary, or ``None`` when the mixing block is absent.
    """
    mixing = _require_mapping(config, "mixing", errors)
    if mixing is None:
        return None
    spec_ref = mixing.get("weighting_spec")
    summary: dict[str, Any] = {}
    if spec_ref is not None:
        if not isinstance(spec_ref, str) or not spec_ref.strip():
            errors.append("mixing.weighting_spec must be a non-empty path string when set")
            return summary
        spec_path = _resolve_path(spec_ref, config_dir)
        if not spec_path.is_file():
            errors.append(f"mixing.weighting_spec does not exist: {spec_ref}")
            return summary
        summary["weighting_spec"] = str(spec_path)
        summary.update(_validate_weighting_spec(spec_path, errors))
    elif "hardcase_repeat" in mixing:
        # Inline weighting falls back to CLI-style hardcase_repeat/shuffle_seed.
        _require_positive_int(mixing, "hardcase_repeat", "mixing", errors)
        summary["hardcase_repeat"] = mixing.get("hardcase_repeat")
    return summary


def _validate_weighting_spec(spec_path: Path, errors: list[str]) -> dict[str, Any]:
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        errors.append(f"weighting spec must be a YAML mapping: {spec_path}")
        return {}
    weighting = payload.get("weighting")
    if not isinstance(weighting, dict):
        errors.append("weighting spec must contain a 'weighting' mapping")
        return {}
    rule = str(weighting.get("rule") or _SUPPORTED_WEIGHTING_RULE)
    if rule != _SUPPORTED_WEIGHTING_RULE:
        errors.append(
            f"unsupported weighting rule {rule!r}; expected {_SUPPORTED_WEIGHTING_RULE!r}"
        )
    _require_positive_int(weighting, "hardcase_repeat", "weighting", errors)
    family = weighting.get("hardcase_family")
    if not isinstance(family, str) or not family.strip():
        errors.append("weighting.hardcase_family must be a non-empty string")
    return {
        "weighting_rule": rule,
        "hardcase_family": family if isinstance(family, str) else None,
        "hardcase_repeat": weighting.get("hardcase_repeat"),
    }


def _validate_training(config: dict[str, Any], errors: list[str]) -> dict[str, Any] | None:
    """Validate the training block: checkpoint identity + separate quality gates.

    Returns:
        Compact training summary, or ``None`` when the training block is absent.
    """
    training = _require_mapping(config, "training", errors)
    if training is None:
        return None
    model_id = _require_non_empty_string(training, "model_id", errors)
    _require_positive_int(training, "epochs", "training", errors)
    # The navigation gate (closed-loop benchmark) must stay separate from the
    # trajectory gate (val_ade/val_fde); both ADE and FDE ceilings are required.
    for gate_key in ("max_val_ade", "max_val_fde"):
        value = training.get(gate_key)
        try:
            if float(value) <= 0.0:
                errors.append(f"training.{gate_key} must be > 0")
        except (TypeError, ValueError):
            errors.append(f"training.{gate_key} must be a positive number")
    return {"model_id": model_id}


def _validate_evaluation(config: dict[str, Any], errors: list[str]) -> dict[str, Any] | None:
    evaluation = _require_mapping(config, "evaluation", errors)
    if evaluation is None:
        return None
    for key in _REQUIRED_EVALUATION_KEYS:
        if key not in evaluation:
            errors.append(f"evaluation.{key} is required")
    return {"keys": sorted(k for k in _REQUIRED_EVALUATION_KEYS if k in evaluation)}


def _validate_output(config: dict[str, Any], errors: list[str]) -> str | None:
    output = _require_mapping(config, "output", errors)
    if output is None:
        return None
    return _require_non_empty_string(output, "root", errors)


__all__ = [
    "PredictiveRetrainPreflightError",
    "load_pipeline_config",
    "validate_retrain_preflight",
]
