"""Refuse held-out seeds before production RNG use or episode dispatch."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.benchmark import seed_bands

# Historical execution is admitted only by its immutable source identity.
SEALED_SOURCE_SHA = "66f402ba176b13e45210d0da0b2cf20fcdc0cc02"
SEED_FIELDS = (
    "seed",
    "pedestrian_seed",
    "route_spawn_seed",
    "desired_speed_seed",
    "archetype_seed",
    "response_law_seed",
)


def validate_sealed_authorization(identity_path: Path) -> Any:
    """Validate explicit, materialized authorization; never create a global exemption.

    Main and development checkouts cannot execute the historical sealed campaign.
    The existing release validator binds canonical inputs and runtime source bytes.

    Returns:
        The verified release identity.
    """
    from robot_sf.benchmark.release_protocol import (  # noqa: PLC0415
        sealed_seed_execution_problem,
        verify_resolved_release_identity,
    )

    identity = verify_resolved_release_identity(identity_path)
    if identity.source_sha != SEALED_SOURCE_SHA:
        raise ValueError("sealed authorization requires the immutable freeze source")
    problem = sealed_seed_execution_problem(
        identity,
        tuple(identity.resolved_seeds),
        source_commit=SEALED_SOURCE_SHA,
    )
    if problem is not None:
        raise ValueError(problem)
    return identity


def check_simulation_seed(
    seed: Any, *, boundary: str, authorization: Path | None = None, dev_only: bool = False
) -> None:
    """Check before seeding/reset/step. Environment variables confer no permission."""
    if seed is None:
        if dev_only:
            raise ValueError(f"development seed required at {boundary}; use 1001..1030")
        return
    try:
        if isinstance(seed, bool):
            raise TypeError
        value = operator.index(seed)
    except TypeError as exc:
        raise ValueError(f"simulation seed must be an integer at {boundary}") from exc
    if dev_only and not 1001 <= value <= 1030:
        raise ValueError(f"non-development seed {value} at {boundary}; use 1001..1030")
    if value not in seed_bands.HELD_OUT_SEEDS:
        return
    if authorization is not None:
        identity = validate_sealed_authorization(authorization)
        if value in seed_bands.EVAL_SEEDS_0_0_8 and value in identity.resolved_seeds:
            return
    raise ValueError(
        f"held-out simulation seed {value} at {boundary}; use development seeds 1001..1030"
    )


def check_seed_config(config: Any, *, boundary: str) -> None:
    """Check explicit simulator seeds, including an enclosing environment config."""
    if config is None:
        return
    for candidate in (config, getattr(config, "sim_config", None)):
        for field in SEED_FIELDS:
            check_simulation_seed(getattr(candidate, field, None), boundary=f"{boundary}.{field}")
