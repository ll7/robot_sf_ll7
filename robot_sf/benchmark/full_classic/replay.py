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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(slots=True)
class ReplayStep:
    """Single timestep state snapshot (enriched T039).

    Core fields:
        - t: relative time (seconds)
        - x, y: robot position
        - heading: robot orientation (radians)

    Optional enrichment:
        - speed: scalar robot speed
        - ped_positions: list of (x,y) pedestrian positions
        - action: (ax, ay) or (vx, vy) tuple representing last action
    """

    t: float
    x: float
    y: float
    heading: float
    speed: float | None = None
    ped_positions: list[tuple[float, float]] | None = None
    action: tuple[float, float] | None = None


@dataclass(slots=True)
class ReplayEpisode:
    """Per‑episode replay container.

    Maintains ordered list of ReplayStep items plus identifying metadata used
    for manifest correlation and deterministic selection.
    """

    episode_id: str
    scenario_id: str
    steps: list[ReplayStep] = field(default_factory=list)

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
    _steps: list[ReplayStep] = field(default_factory=list)

    def record(
        self,
        t: float,
        x: float,
        y: float,
        heading: float,
        speed: float | None = None,
        ped_positions: list[tuple[float, float]] | None = None,
        action: tuple[float, float] | None = None,
    ) -> None:
        self._steps.append(
            ReplayStep(
                t=t,
                x=x,
                y=y,
                heading=heading,
                speed=speed,
                ped_positions=ped_positions,
                action=action,
            ),
        )

    def finalize(self) -> ReplayEpisode:
        return ReplayEpisode(
            episode_id=self.episode_id,
            scenario_id=self.scenario_id,
            steps=list(self._steps),
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
        if not all(isinstance(v, int | float) for v in (s.t, s.x, s.y, s.heading)):
            return False
        if s.t < prev_t:  # non-monotonic
            return False
        prev_t = s.t
    return True


def build_replay_episode(
    episode_id: str,
    scenario_id: str,
    seq: Sequence[tuple[float, float, float, float]],
    ped_seq: Sequence[list[tuple[float, float]] | None] | None = None,
    action_seq: Sequence[tuple[float, float] | None] | None = None,
) -> ReplayEpisode:
    """Convenience constructor from basic sequences.

    Parameters
    ----------
    seq : sequence of (t, x, y, heading)
        Core robot kinematic samples.
    ped_seq : optional sequence parallel to seq with pedestrian position lists.
    action_seq : optional sequence parallel to seq with action tuples.
    """
    steps: list[ReplayStep] = []
    for i, (t, x, y, h) in enumerate(seq):
        peds = ped_seq[i] if ped_seq and i < len(ped_seq) else None
        act = action_seq[i] if action_seq and i < len(action_seq) else None
        steps.append(ReplayStep(t=t, x=x, y=y, heading=h, ped_positions=peds, action=act))
    return ReplayEpisode(episode_id=episode_id, scenario_id=scenario_id, steps=steps)


__all__ = [
    "ReplayCapture",
    "ReplayEpisode",
    "ReplayStep",
    "build_replay_episode",
    "validate_replay_episode",
]


def extract_replay_episodes(records: list[dict], min_length: int = 2):
    """Extract replay episodes from raw episode records (T022 adapter).

    Returns mapping episode_id -> ReplayEpisode for those with a valid
    `replay_steps` list of (t,x,y,heading) tuples. Invalid or too-short
    sequences are ignored (caller may apply skip logic per FR-008).
    """
    out: dict[str, ReplayEpisode] = {}
    for rec in records:
        ep_id = rec.get("episode_id")
        sc_id = rec.get("scenario_id", "unknown")
        steps_raw = rec.get("replay_steps")
        ped_raw = rec.get("replay_peds")  # parallel list of lists or None
        action_raw = rec.get("replay_actions")  # parallel list of tuples or None
        if not isinstance(ep_id, str) or not isinstance(steps_raw, list):
            continue
        try:
            seq = []
            for tpl in steps_raw:
                # Support legacy 4-tuple or enriched 6-tuple (t,x,y,h, ped_list, action_tuple)
                if not isinstance(tpl, list | tuple) or len(tpl) < 4:
                    raise ValueError
                t, x, y, h = tpl[:4]
                seq.append((float(t), float(x), float(y), float(h)))
            ped_seq = ped_raw if isinstance(ped_raw, list) else None
            action_seq = action_raw if isinstance(action_raw, list) else None
        except (ValueError, TypeError):
            continue
        ep = build_replay_episode(ep_id, sc_id, seq, ped_seq=ped_seq, action_seq=action_seq)
        if validate_replay_episode(ep, min_length=min_length):
            out[ep_id] = ep
    return out


__all__.append("extract_replay_episodes")
