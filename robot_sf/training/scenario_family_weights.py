"""Compile opt-in family multipliers into the canonical scenario sampler weights."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from robot_sf.training.scenario_loader import load_scenarios
from robot_sf.training.scenario_sampling import ScenarioSampler

if TYPE_CHECKING:
    from robot_sf.training.imitation_config import ExpertTrainingConfig

FAMILIES = ("bottleneck", "doorway", "exiting_elevator", "exiting_room", "frontal_approach")


def _scenario_names(raw: object) -> tuple[str, ...]:
    """Validate a config filter before passing it to the canonical sampler.

    Returns:
        Declared scenario identifiers.
    """
    if not isinstance(raw, (list, tuple)) or any(not isinstance(v, str) for v in raw):
        raise ValueError("Scenario filters must be lists of names")
    return tuple(str(v) for v in raw)


def _numeric_weights(raw: object) -> dict[str, float] | None:
    """Validate numeric YAML weights without dropping malformed entries.

    Returns:
        Validated weights, or None when absent.
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("Scenario weights must be a mapping")
    result = {}
    for name, value in raw.items():
        if (
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError("Scenario weight names must be strings and values numeric")
        result[name] = float(value)
    return result


def compile_family_weights(
    scenario_ids: Sequence[str],
    family_weights: Mapping[str, float] | None = None,
    base_weights: Mapping[str, float] | None = None,
) -> dict[str, float] | None:
    """Multiply named families, retaining uniform sampling when the option is off.

    All unweighted scenarios retain weight one. Existing explicit per-scenario
    weights keep their canonical semantics (missing names have weight zero).
    Families match complete underscore-delimited names, including density suffixes.

    Returns:
        Complete unnormalized per-scenario weights, or the unchanged off-state.
    """
    if not family_weights:
        return dict(base_weights) if base_weights is not None else None
    unknown = set(family_weights) - set(FAMILIES)
    if unknown:
        raise ValueError(f"Unknown scenario families: {sorted(unknown)}")
    weights = {
        sid: float(base_weights.get(sid, 0.0)) if base_weights else 1.0 for sid in scenario_ids
    }
    if any(not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("Base scenario weights must be finite and nonnegative")
    for family, raw_multiplier in family_weights.items():
        multiplier = float(raw_multiplier)
        if not math.isfinite(multiplier) or multiplier < 1:
            raise ValueError("Family oversampling multipliers must be finite and >= 1")
        matched = [sid for sid in scenario_ids if re.search(rf"(?:^|_){family}(?:_|$)", sid)]
        if not matched or not any(weights[sid] > 0 for sid in matched):
            raise ValueError(f"Family has no sampleable training scenarios: {family}")
        for sid in matched:
            weights[sid] *= multiplier
    if any(not math.isfinite(w) for w in weights.values()):
        raise ValueError("Resolved scenario weights overflowed")
    return weights


def with_family_weights(config: ExpertTrainingConfig) -> ExpertTrainingConfig:
    """Resolve family options once, before launching canonical trainer workers.

    Returns:
        A new config using the existing sampler; selection/evaluation is unchanged.
    """
    sampling = dict(config.scenario_sampling)
    families = sampling.pop("family_weights", None)
    if families is not None and not isinstance(families, Mapping):
        raise ValueError("scenario_sampling.family_weights must be a mapping")
    if not families:
        return replace(config, scenario_sampling=sampling)
    if config.scenario_id:
        raise ValueError("Family weighting requires multi-scenario training")
    if sampling.get("strategy", "random") != "random":
        raise ValueError("Family weighting requires random scenario sampling")
    protocol = config.multi_map_protocol
    sampler = ScenarioSampler(
        load_scenarios(config.scenario_config),
        include_scenarios=protocol.train_scenarios
        if protocol
        else _scenario_names(sampling.get("include_scenarios", ())),
        exclude_scenarios=tuple(config.evaluation.hold_out_scenarios)
        + _scenario_names(sampling.get("exclude_scenarios", ())),
    )
    sampling["weights"] = compile_family_weights(
        sampler.scenario_ids, _numeric_weights(families), _numeric_weights(sampling.get("weights"))
    )
    return replace(config, scenario_sampling=sampling)
