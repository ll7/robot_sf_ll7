"""Replay data structures for SimulationView video generation (T020).

These lightweight dataclasses capture the minimal per‑timestep and per‑episode
state required to reconstruct an episode visually inside the SimulationView
renderer. They are intentionally decoupled from the core episode schema to
avoid any changes to existing benchmark contracts (FR-020) while enabling
enhanced video fidelity (FR-001).

Usage flow (future tasks):
1. (T021) During episode execution, optionally record ReplayStep entries when a
   feature flag (e.g., cfg.capture_replay) is enabled.
2. (T022) After run completion, build a list[ReplayEpisode] (adapter) for the
   selected episodes destined for video rendering.
3. (T031+) Feed ReplayEpisode instances into a SimulationView frame generator.

Validation is intentionally loose: we only ensure shapes / lengths align and
required numeric fields exist; deeper semantic checks (e.g., physical
consistency) remain out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass(slots=True)
class ReplayStep:
    """Single timestep state snapshot.

    Fields kept minimal: position (x, y), heading (radians), timestamp (seconds).
    Additional optional fields (speed, goal vectors, etc.) can be appended later
    without breaking existing usage since dataclass provides defaults.
    """

    t: float
    x: float
    y: float
    heading: float
    # Optional extensions (kept None when not captured)
    speed: float | None = None


@dataclass(slots=True)
class ReplayEpisode:
    """Per‑episode replay container.

    Maintains ordered list of ReplayStep items plus identifying metadata used
    for manifest correlation and deterministic selection.
    """

    episode_id: str
    scenario_id: str
    steps: List[ReplayStep] = field(default_factory=list)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.steps)

    def append(self, step: ReplayStep) -> None:
        self.steps.append(step)


@dataclass(slots=True)
class ReplayCapture:
    """Mutable builder used during live episode execution (T021 integration).

    Chosen to separate the capture mutation concern from the immutable-ish
    ReplayEpisode used downstream. Provides a `finalize()` method producing a
    ReplayEpisode suitable for adaptation/encoding.
    """

    episode_id: str
    scenario_id: str
    _steps: List[ReplayStep] = field(default_factory=list)

    def record(
        self, t: float, x: float, y: float, heading: float, speed: float | None = None
    ) -> None:
        self._steps.append(ReplayStep(t=t, x=x, y=y, heading=heading, speed=speed))

    def finalize(self) -> ReplayEpisode:
        return ReplayEpisode(
            episode_id=self.episode_id, scenario_id=self.scenario_id, steps=list(self._steps)
        )


def validate_replay_episode(ep: ReplayEpisode, min_length: int = 2) -> bool:
    """Light validation helper (T020).

    Returns True if episode appears structurally valid for SimulationView:
    - Has at least `min_length` steps
    - All steps contain finite numeric coordinates (basic check)
    - Timestamps non‑decreasing (monotonic tolerance: strictly increasing or equal)
    """
    if len(ep.steps) < min_length:
        return False
    prev_t = ep.steps[0].t
    for s in ep.steps:
        if not all(isinstance(v, (int, float)) for v in (s.t, s.x, s.y, s.heading)):
            return False
        if s.t < prev_t:  # non‑monotonic
            return False
        prev_t = s.t
    return True


def build_replay_episode(
    episode_id: str, scenario_id: str, seq: Sequence[tuple[float, float, float, float]]
) -> ReplayEpisode:
    """Convenience constructor from a sequence of (t, x, y, heading) tuples.

    Useful for tests (e.g., insufficient replay skip) and adapter code. Speed
    left unset to keep interface lean for now.
    """
    steps = [ReplayStep(t=t, x=x, y=y, heading=h) for (t, x, y, h) in seq]
    return ReplayEpisode(episode_id=episode_id, scenario_id=scenario_id, steps=steps)


__all__ = [
    "ReplayStep",
    "ReplayEpisode",
    "ReplayCapture",
    "validate_replay_episode",
    "build_replay_episode",
]
