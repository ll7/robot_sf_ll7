#!/usr/bin/env python3
"""Pedestrian-model assumption inventory / preflight for HSFM + TTC force experiments.

This module is the canonical owner for the *assumption and artifact inventory* slice of
issue #3481 ("headed social-force (HSFM) + TTC-predictive pedestrian forces to remove
passive-sliding exploits"). It is a **read-only preflight**: it documents the assumptions
baked into the *current* pedestrian force model, probes that the entry-point surfaces an
HSFM/TTC experiment would touch are importable, and tracks which local experiment surfaces
now exist versus which stronger proof steps (for example seed-controlled benchmark evidence
or external calibration) still remain.

Scope and boundaries (read before extending):

- This is an *inventory*, not an implementation. It does **not** add or alter any force law,
  add heading state, change scenario behavior, run a benchmark, or make any realism claim.
  Per issue #3481 the force-model upgrade itself stays ``evidence_tier: idea`` until durable
  in-repo evidence exists.
- The "current assumptions" below describe the active vendored Social Force core
  (``fast-pysf/pysocialforce/forces.py``). They are stated conservatively and verifiably:
  for example the ped-ped repulsion is velocity-anisotropic (elliptical Helbing-Molnar-Farkas
  form with ``lambda_importance``), but it carries **no** field-of-view view-cone attenuation
  on the general ped-ped interaction and **no** time-to-collision term. These gaps are what
  the issue proposes to close; this module only records them.
- The verdict is *fail-closed on the existing entry-point surfaces only*. If a named force-core
  or pedestrian-NPC surface cannot be imported the preflight fails (an HSFM/TTC experiment
  could not even be wired in). Missing *prerequisites* are reported as planned blockers, but
  they do **not** fail the verdict, because the whole point of the inventory is to enumerate
  what local capability exists and what proof work still remains.

Companions: ``scripts/research/check_ped_model_assumption_inventory.py`` (thin CLI).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

__all__ = [
    "CURRENT_ASSUMPTIONS",
    "ENTRY_POINT_SURFACES",
    "EXPERIMENT_PREREQUISITES",
    "AssumptionSpec",
    "EntryPointSpec",
    "InventoryReport",
    "PrerequisiteSpec",
    "PrerequisiteStatus",
    "SurfaceStatus",
    "build_inventory_report",
    "probe_entry_point",
    "probe_prerequisite",
    "render_markdown",
    "repo_root",
]

# Issue this inventory belongs to; used in report headers and the CLI banner.
ISSUE = 3481


def repo_root() -> Path:
    """Return the repository root inferred from this file's location.

    ``robot_sf/research/ped_model_assumption_inventory.py`` -> repo root is three parents up.
    Used to resolve the declared surface/fixture paths against the active checkout so the
    inventory reflects reality rather than a hard-coded assumption.
    """
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Current pedestrian-model assumptions (informational, verifiable).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssumptionSpec:
    """A documented assumption of the *current* pedestrian force model.

    These describe the baseline an HSFM/TTC upgrade would change. They are descriptive
    breadcrumbs to the owning code, not runtime configuration and not realism claims.
    """

    key: str
    title: str
    description: str
    # Where the assumption is realized in the active force core / NPC layer.
    evidence_path: str
    # What the issue proposes to change about this assumption (for traceability only).
    proposed_change: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the assumption record."""
        return {
            "key": self.key,
            "title": self.title,
            "description": self.description,
            "evidence_path": self.evidence_path,
            "proposed_change": self.proposed_change,
        }


# Conservative, source-backed inventory of the assumptions the issue targets. Each
# ``evidence_path`` points at the owner that realizes the assumption today.
CURRENT_ASSUMPTIONS: tuple[AssumptionSpec, ...] = (
    AssumptionSpec(
        key="no_fov_attenuation",
        title="No field-of-view attenuation on ped-ped repulsion",
        description=(
            "The active ped-ped social force is the velocity-anisotropic elliptical "
            "Helbing-Molnar-Farkas form (lambda_importance / n / n_prime). It has no "
            "view-cone weight that down-weights interactions originating behind a "
            "pedestrian's heading. (fov_phi exists only in GroupGazeForceConfig for "
            "intra-group gaze, not for general ped-ped repulsion.)"
        ),
        evidence_path="fast-pysf/pysocialforce/forces.py",
        proposed_change=("Anisotropic FoV weight: ~full strength in-cone, ~0.1 behind heading."),
    ),
    AssumptionSpec(
        key="heading_coupled_to_velocity",
        title="Heading coupled to instantaneous velocity (no HSFM state)",
        description=(
            "Desired directions are derived from velocity/goal; there is no body-orientation "
            "state phi_i decoupled from instantaneous velocity v_i and no alignment-torque "
            "term in the force core. The heading used in ped_behavior is for scripted-actor "
            "pose, not a force-law body orientation."
        ),
        evidence_path="fast-pysf/pysocialforce/forces.py",
        proposed_change=(
            "HSFM heading state phi_i decoupled from v_i plus a body-alignment torque."
        ),
    ),
    AssumptionSpec(
        key="euclidean_distance_repulsion",
        title="Euclidean-distance repulsion (no predictive TTC term)",
        description=(
            "social_force scales repulsion by current position differences and distances; "
            "there is no time-to-collision tau_ij term, so the model reacts to present "
            "separation rather than predicted collision time."
        ),
        evidence_path="fast-pysf/pysocialforce/forces.py",
        proposed_change=(
            "Opt-in TTC-scaled predictive repulsion F_ij proportional to exp(-tau_ij/tau_0)."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Entry-point surfaces an HSFM/TTC experiment would touch (probed, fail-closed).
# ---------------------------------------------------------------------------


class SurfaceStatus(StrEnum):
    """Classification for an entry-point surface probe."""

    PRESENT = "present"
    """File present, module imports, and all required symbols are exposed."""

    MISSING_MODULE = "missing_module"
    """Declared module cannot be imported (import raised)."""

    MISSING_SYMBOLS = "missing_symbols"
    """Module imports but one or more declared public symbols are absent."""

    MISSING_FILES = "missing_files"
    """Declared file path does not exist on the active checkout."""


@dataclass(frozen=True)
class EntryPointSpec:
    """A surface an HSFM/TTC experiment must be able to import to be wired in.

    ``module`` is an importable dotted path; ``file_path`` is the repo-relative file that
    backs it (verified for presence); ``required_symbols`` are public names the surface must
    still expose. ``required`` surfaces fail the verdict closed when broken.
    """

    key: str
    title: str
    module: str
    file_path: str
    required_symbols: tuple[str, ...] = ()
    required: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the entry-point spec."""
        return {
            "key": self.key,
            "title": self.title,
            "module": self.module,
            "file_path": self.file_path,
            "required_symbols": list(self.required_symbols),
            "required": self.required,
            "note": self.note,
        }


# Canonical entry points named in issue #3481 plus the active force core they depend on.
# These already exist on main; the probe keeps the inventory honest if any are renamed.
ENTRY_POINT_SURFACES: tuple[EntryPointSpec, ...] = (
    EntryPointSpec(
        key="force_core",
        title="Active Social Force core (vendored PySocialForce)",
        module="pysocialforce.forces",
        file_path="fast-pysf/pysocialforce/forces.py",
        required_symbols=("social_force", "SocialForce", "DesiredForce"),
        note="FoV-weight and TTC terms would extend this force evaluation.",
    ),
    EntryPointSpec(
        key="force_config",
        title="Social Force configuration dataclasses",
        module="pysocialforce.config",
        file_path="fast-pysf/pysocialforce/config.py",
        required_symbols=("SocialForceConfig",),
        note="Versioned HSFM/TTC/FoV parameters would be added here.",
    ),
    EntryPointSpec(
        key="ped_behavior",
        title="Pedestrian behavior integration",
        module="robot_sf.ped_npc.ped_behavior",
        file_path="robot_sf/ped_npc/ped_behavior.py",
        required_symbols=("PedestrianBehavior",),
        note="HSFM heading state would surface through behavior stepping.",
    ),
    EntryPointSpec(
        key="ped_population",
        title="Pedestrian population stepping / force aggregation",
        module="robot_sf.ped_npc.ped_population",
        file_path="robot_sf/ped_npc/ped_population.py",
        required_symbols=("populate_simulation",),
        note="Vectorized force aggregation is where an FoV weight would attach.",
    ),
    EntryPointSpec(
        key="ped_archetypes",
        title="Pedestrian archetype composition axis (#3206, kept orthogonal)",
        module="robot_sf.ped_npc.ped_archetypes",
        file_path="robot_sf/ped_npc/ped_archetypes.py",
        required_symbols=("load_archetypes",),
        required=False,
        note="Context only: composition axis is distinct from the force-law axis.",
    ),
)


@dataclass(frozen=True)
class SurfaceProbeResult:
    """Result of probing one entry-point surface against the active checkout."""

    spec: EntryPointSpec
    status: SurfaceStatus
    missing_symbols: tuple[str, ...] = ()
    detail: str = ""

    @property
    def present(self) -> bool:
        """True when the surface imports with all required symbols."""
        return self.status is SurfaceStatus.PRESENT

    @property
    def is_blocker(self) -> bool:
        """True when a *required* surface is not present (fails the verdict closed)."""
        return self.spec.required and not self.present

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the probe result."""
        return {
            "key": self.spec.key,
            "status": self.status.value,
            "required": self.spec.required,
            "present": self.present,
            "is_blocker": self.is_blocker,
            "missing_symbols": list(self.missing_symbols),
            "detail": self.detail,
        }


def probe_entry_point(spec: EntryPointSpec, root: Path | None = None) -> SurfaceProbeResult:
    """Probe one entry-point surface: file presence, import, and required symbols.

    This is read-only introspection. It never executes simulation behavior; it only checks
    that the surface an experiment would extend is reachable on this checkout.

    Args:
        spec: The entry-point surface to probe.
        root: Repository root to resolve ``file_path`` against; defaults to :func:`repo_root`.

    Returns:
        A :class:`SurfaceProbeResult` with the classified status.
    """
    base = root if root is not None else repo_root()
    if not (base / spec.file_path).exists():
        return SurfaceProbeResult(
            spec=spec,
            status=SurfaceStatus.MISSING_FILES,
            detail=f"declared file not found: {spec.file_path}",
        )
    try:
        module = importlib.import_module(spec.module)
    except Exception as exc:  # noqa: BLE001 - report any import failure as a blocker
        return SurfaceProbeResult(
            spec=spec,
            status=SurfaceStatus.MISSING_MODULE,
            detail=f"import failed: {type(exc).__name__}: {exc}",
        )
    missing = tuple(sym for sym in spec.required_symbols if not hasattr(module, sym))
    if missing:
        return SurfaceProbeResult(
            spec=spec,
            status=SurfaceStatus.MISSING_SYMBOLS,
            missing_symbols=missing,
            detail=f"module missing symbols: {', '.join(missing)}",
        )
    return SurfaceProbeResult(spec=spec, status=SurfaceStatus.PRESENT)


# ---------------------------------------------------------------------------
# Experiment prerequisites / blockers (declared, presence-probed, not implemented).
# ---------------------------------------------------------------------------


class PrerequisiteStatus(StrEnum):
    """Whether a prerequisite for the HSFM/TTC experiments exists yet."""

    PRESENT = "present"
    """The prerequisite already exists on this checkout (path probe matched)."""

    ABSENT = "absent"
    """Not yet built. Expected for the design-stage force-law work; a planned blocker."""

    EXTERNAL = "external"
    """Depends on data/assets not in repo; cannot be satisfied by local code alone."""


@dataclass(frozen=True)
class PrerequisiteSpec:
    """A capability or artifact the HSFM/TTC experiments need before they can run.

    ``probe_paths`` are repo-relative paths (or glob patterns) whose presence would indicate
    the prerequisite is satisfied. When empty, the prerequisite is ``EXTERNAL`` and reported
    as a standing blocker rather than something local code can produce.
    """

    key: str
    title: str
    description: str
    blocks: tuple[str, ...]  # which DoD item(s) / experiment(s) this gates
    probe_paths: tuple[str, ...] = ()
    external: bool = False
    related_issues: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the prerequisite spec."""
        return {
            "key": self.key,
            "title": self.title,
            "description": self.description,
            "blocks": list(self.blocks),
            "probe_paths": list(self.probe_paths),
            "external": self.external,
            "related_issues": list(self.related_issues),
        }


# Ordered inventory of the local surfaces and proof blockers around the HSFM/TTC experiments.
# Paths are probed so the inventory flips an item to "present" automatically once the work
# lands, instead of going stale.
EXPERIMENT_PREREQUISITES: tuple[PrerequisiteSpec, ...] = (
    PrerequisiteSpec(
        key="hsfm_heading_state",
        title="HSFM heading state + alignment torque",
        description=(
            "A body-orientation state phi_i decoupled from instantaneous velocity v_i, plus "
            "an alignment-torque term, added to the force core. The current total-force "
            "heading proxy is present, but this fuller body-orientation surface is still "
            "absent."
        ),
        blocks=("DoD: HSFM heading state + alignment-torque term",),
        probe_paths=("robot_sf/ped_npc/hsfm_heading.py", "fast-pysf/pysocialforce/hsfm.py"),
    ),
    PrerequisiteSpec(
        key="fov_attenuation",
        title="Anisotropic field-of-view repulsion weight",
        description=(
            "An opt-in view-cone weight on the vectorized pedestrian-pedestrian force "
            "(full strength in-cone, rear attenuation behind heading)."
        ),
        blocks=("DoD: anisotropic FoV weight prototyped",),
        probe_paths=(
            "robot_sf/sim/pedestrian_model_variants.py",
            "tests/sim/test_hsfm_fov_pairwise_isolation.py",
        ),
    ),
    PrerequisiteSpec(
        key="ttc_predictive_term",
        title="TTC-scaled predictive repulsion term",
        description=(
            "An opt-in time-to-collision repulsion F_ij ~ exp(-tau_ij/tau_0) offered as an "
            "alternative to pure Euclidean-distance repulsion."
        ),
        blocks=("DoD: opt-in TTC-scaled predictive repulsion term",),
        probe_paths=(
            "robot_sf/sim/pedestrian_model_variants.py",
            "tests/sim/test_ttc_predictive_pedestrian_model.py",
        ),
    ),
    PrerequisiteSpec(
        key="narrow_passage_fixture",
        title="Narrow-passage lateral-sliding fixture",
        description=(
            "A seed-controlled narrow-passage fixture/scenario that exercises lateral sliding "
            "so reduced sliding vs isotropic SFM can be measured. The shared-throat local "
            "precursor harness does not satisfy this geometric fixture blocker."
        ),
        blocks=("DoD: fixture proving reduced lateral sliding",),
        probe_paths=(
            "tests/ped_npc/test_hsfm_narrow_passage.py",
            "tests/fixtures/ped_npc/narrow_passage_sliding.yaml",
        ),
        related_issues=(3206,),
    ),
    PrerequisiteSpec(
        key="bottleneck_fixture",
        title="Bottleneck freeze/deadlock fixture",
        description=(
            "A seed-controlled geometric bottleneck fixture/scenario to measure mutual-freeze / "
            "deadlock rate ('freezing robot' anomaly). The shared-throat local precursor "
            "harness does not satisfy this bottleneck-specific blocker."
        ),
        blocks=("DoD: fixture proving reduced mutual-freeze/deadlock rate",),
        probe_paths=(
            "tests/ped_npc/test_hsfm_bottleneck.py",
            "tests/fixtures/ped_npc/bottleneck_freeze.yaml",
        ),
    ),
    PrerequisiteSpec(
        key="versioned_parameters",
        title="Versioned HSFM/TTC parameter set",
        description=(
            "A config-first, versioned parameter set for the HSFM/TTC/FoV terms so runs are "
            "reproducible."
        ),
        blocks=("DoD: HSFM/TTC parameters versioned",),
        probe_paths=("configs/research/hsfm_ttc_predictive_forces_issue_3481.yaml",),
    ),
    PrerequisiteSpec(
        key="design_note",
        title="Scoping/design note for the force-law changes",
        description=(
            "A note describing the HSFM/TTC/FoV force-law changes and their modeling "
            "assumptions (diagnostic/prototype, no calibrated-realism claim)."
        ),
        blocks=("DoD: scoping/design note",),
        probe_paths=("docs/context/issue_3481_hsfm_ttc_predictive_forces.md",),
    ),
    PrerequisiteSpec(
        key="calibration_data",
        title="External calibration data for realism claim",
        description=(
            "Real pedestrian-interaction data needed to calibrate HSFM/TTC parameters before "
            "any realism claim. Out of repo; a standing external blocker, not local work."
        ),
        blocks=("Out of scope: real-world behavioral realism claim",),
        external=True,
        related_issues=(3293,),
    ),
)


@dataclass(frozen=True)
class PrerequisiteProbeResult:
    """Result of probing one experiment prerequisite against the active checkout."""

    spec: PrerequisiteSpec
    status: PrerequisiteStatus
    matched_paths: tuple[str, ...] = ()

    @property
    def satisfied(self) -> bool:
        """True when the prerequisite is already present on this checkout."""
        return self.status is PrerequisiteStatus.PRESENT

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the prerequisite probe result."""
        return {
            "key": self.spec.key,
            "status": self.status.value,
            "satisfied": self.satisfied,
            "external": self.spec.external,
            "blocks": list(self.spec.blocks),
            "matched_paths": list(self.matched_paths),
        }


def probe_prerequisite(spec: PrerequisiteSpec, root: Path | None = None) -> PrerequisiteProbeResult:
    """Classify a prerequisite as present / absent / external by probing its paths.

    Read-only path presence check. Any matching path (literal or glob) marks the prerequisite
    satisfied so the inventory updates itself once the work lands. External prerequisites with
    no local probe path are always reported as standing blockers.

    Args:
        spec: The prerequisite to probe.
        root: Repository root to resolve ``probe_paths`` against; defaults to :func:`repo_root`.

    Returns:
        A :class:`PrerequisiteProbeResult` with the classified status.
    """
    if spec.external and not spec.probe_paths:
        return PrerequisiteProbeResult(spec=spec, status=PrerequisiteStatus.EXTERNAL)
    base = root if root is not None else repo_root()
    matched: list[str] = []
    for pattern in spec.probe_paths:
        if any(ch in pattern for ch in "*?["):
            if list(base.glob(pattern)):
                matched.append(pattern)
        elif (base / pattern).exists():
            matched.append(pattern)
    if matched:
        return PrerequisiteProbeResult(
            spec=spec, status=PrerequisiteStatus.PRESENT, matched_paths=tuple(matched)
        )
    status = PrerequisiteStatus.EXTERNAL if spec.external else PrerequisiteStatus.ABSENT
    return PrerequisiteProbeResult(spec=spec, status=status)


# ---------------------------------------------------------------------------
# Aggregate report.
# ---------------------------------------------------------------------------


@dataclass
class InventoryReport:
    """Aggregate inventory: assumptions, probed surfaces, and probed prerequisites."""

    assumptions: tuple[AssumptionSpec, ...]
    surfaces: tuple[SurfaceProbeResult, ...]
    prerequisites: tuple[PrerequisiteProbeResult, ...]

    @property
    def required_surfaces(self) -> tuple[SurfaceProbeResult, ...]:
        """Probed surfaces marked required (these drive the fail-closed verdict)."""
        return tuple(s for s in self.surfaces if s.spec.required)

    @property
    def surface_blockers(self) -> tuple[SurfaceProbeResult, ...]:
        """Required surfaces that are not present (verdict blockers)."""
        return tuple(s for s in self.surfaces if s.is_blocker)

    @property
    def ok(self) -> bool:
        """Verdict: True iff every *required* entry-point surface is present.

        Missing prerequisites are planned work and do not flip the verdict; only a broken
        required surface (an experiment could not be wired in at all) fails closed.
        """
        return not self.surface_blockers

    @property
    def pending_prerequisites(self) -> tuple[PrerequisiteProbeResult, ...]:
        """Prerequisites not yet satisfied (absent or external standing blockers)."""
        return tuple(p for p in self.prerequisites if not p.satisfied)

    def exit_code(self) -> int:
        """Process exit code for the preflight.

        Returns:
            ``0`` when the verdict is OK, ``1`` otherwise.
        """
        return 0 if self.ok else 1

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the full inventory report."""
        return {
            "issue": ISSUE,
            "ok": self.ok,
            "assumptions": [a.to_dict() for a in self.assumptions],
            "surfaces": [s.to_dict() for s in self.surfaces],
            "prerequisites": [p.to_dict() for p in self.prerequisites],
            "summary": {
                "required_surfaces_present": sum(1 for s in self.required_surfaces if s.present),
                "required_surfaces_total": len(self.required_surfaces),
                "prerequisites_present": sum(1 for p in self.prerequisites if p.satisfied),
                "prerequisites_total": len(self.prerequisites),
                "pending_prerequisites": len(self.pending_prerequisites),
            },
        }


def build_inventory_report(root: Path | None = None) -> InventoryReport:
    """Build the full pedestrian-model assumption inventory for the active checkout.

    Args:
        root: Repository root; defaults to :func:`repo_root`.

    Returns:
        A populated :class:`InventoryReport`.
    """
    base = root if root is not None else repo_root()
    surfaces = tuple(probe_entry_point(spec, base) for spec in ENTRY_POINT_SURFACES)
    prerequisites = tuple(probe_prerequisite(spec, base) for spec in EXPERIMENT_PREREQUISITES)
    return InventoryReport(
        assumptions=CURRENT_ASSUMPTIONS,
        surfaces=surfaces,
        prerequisites=prerequisites,
    )


def render_markdown(report: InventoryReport) -> str:
    """Render the inventory report as a compact Markdown summary for the CLI.

    Args:
        report: The aggregate inventory report to render.

    Returns:
        A Markdown string with the verdict banner and the assumption / surface /
        prerequisite sections.
    """
    verdict = "PASS" if report.ok else "FAIL"
    lines: list[str] = []
    lines.append(f"# Pedestrian-model assumption inventory (#{ISSUE}): {verdict}")
    present_req = sum(1 for s in report.required_surfaces if s.present)
    total_req = len(report.required_surfaces)
    present_pre = sum(1 for p in report.prerequisites if p.satisfied)
    total_pre = len(report.prerequisites)
    lines.append(
        f"Required surfaces present: {present_req}/{total_req} | "
        f"Prerequisites present: {present_pre}/{total_pre} | "
        f"Pending blockers: {len(report.pending_prerequisites)}"
    )

    lines.append("")
    lines.append("## Current force-model assumptions (baseline the upgrade targets)")
    for a in report.assumptions:
        lines.append(f"- **{a.title}** ({a.evidence_path})")
        lines.append(f"  - proposed: {a.proposed_change}")

    lines.append("")
    lines.append("## Entry-point surfaces (fail-closed)")
    for s in report.surfaces:
        tag = "required" if s.spec.required else "optional"
        mark = "ok" if s.present else "BLOCKER" if s.is_blocker else s.status.value
        lines.append(f"- [{mark}] {s.spec.title} ({tag}) — {s.spec.module}")
        if s.detail:
            lines.append(f"  - {s.detail}")

    lines.append("")
    lines.append("## Experiment prerequisites / blockers")
    for p in report.prerequisites:
        mark = "present" if p.satisfied else p.status.value
        lines.append(f"- [{mark}] {p.spec.title}")
        for blocked in p.spec.blocks:
            lines.append(f"  - blocks: {blocked}")

    return "\n".join(lines)
