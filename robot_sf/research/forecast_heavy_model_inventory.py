#!/usr/bin/env python3
"""Heavy forecast-model-family inventory / preflight for the offline prediction study (#2845).

This module is the canonical owner for the *model-family inventory and minimum-experiment
preflight* slice of issue #2845 ("prediction: offline transformer or diffusion benchmark
study"). It is a **read-only, design-stage preflight**: it documents the candidate heavy
predictor families (AgentFormer-like / transformer / CVAE / diffusion) with their planning-stage
compute cost, inference latency, uncertainty quality, and repository integration burden; probes
that the offline-evaluation entry-point surfaces such an experiment would touch are importable;
and reports the still-missing minimum-offline-experiment prerequisites as explicit blockers.

Scope and boundaries (read before extending):

- This is an *inventory*, not an implementation. It does **not** add a model, train anything,
  run inference, add a dependency (PyTorch Geometric, diffusers, etc.), run a benchmark, or make
  any model-quality claim. Per issue #2845 the heavy-model work stays ``evidence_tier: blocked``
  until durable in-repo evidence exists; this module only enumerates the assessment surface.
- The per-family ``compute_cost`` / ``inference_latency`` / ``uncertainty_quality`` /
  ``integration_burden`` fields are *qualitative planning estimates derived from the published
  literature*, not repository measurements. They are ordinal ranks (see :class:`CostTier`) meant
  for relative triage only. Treating them as benchmark evidence would violate the issue's
  "do not treat literature plausibility as repository evidence" non-goal.
- The verdict is *fail-closed on the existing offline-evaluation surfaces only*. If a named
  forecast metric / calibration / dataset surface cannot be imported the preflight fails (an
  offline experiment could not even be wired in). Missing *prerequisites* (a trained adapter, a
  CPU runtime budget, a staged held-out dataset) are expected to be absent today and are reported
  as planned blockers; they do **not** fail the import verdict. The separate
  :attr:`InventoryReport.minimum_experiment_status` rolls those prerequisites up into a
  ``ready`` / ``blocked`` status so callers can see whether the minimum offline experiment could
  run yet (it cannot, today).

Companions: ``scripts/research/check_forecast_heavy_model_inventory.py`` (thin CLI).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

__all__ = [
    "ENTRY_POINT_SURFACES",
    "EXPERIMENT_PREREQUISITES",
    "MODEL_FAMILIES",
    "CostTier",
    "EntryPointSpec",
    "InventoryReport",
    "MinimumExperimentStatus",
    "ModelFamilySpec",
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
ISSUE = 2845


def repo_root() -> Path:
    """Return the repository root inferred from this file's location.

    ``robot_sf/research/forecast_heavy_model_inventory.py`` -> repo root is three parents up.
    Used to resolve declared surface/prerequisite paths against the active checkout so the
    inventory reflects reality rather than a hard-coded assumption.
    """
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Candidate heavy forecast-model families (informational planning estimates).
# ---------------------------------------------------------------------------


class CostTier(StrEnum):
    """Ordinal planning-stage tier for a qualitative cost/quality axis.

    These are *relative* literature-derived ranks for triage, not measured values. ``LOW`` ..
    ``VERY_HIGH`` order from cheapest/weakest to most expensive/strongest on the named axis.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass(frozen=True)
class ModelFamilySpec:
    """A candidate heavy predictor family and its planning-stage tradeoff profile.

    All tiered fields are qualitative literature-derived estimates for relative triage only;
    none is a repository measurement or a model-quality claim. ``offline_use_cases`` records
    where the family *might* help (online prediction, offline scenario generation, adversarial
    stress) so the final continue/revise/stop recommendation can be made per use case.
    """

    key: str
    title: str
    description: str
    # Qualitative, literature-derived planning estimates (NOT measured here).
    compute_cost: CostTier  # training + per-run compute relative to the lightweight baseline
    inference_latency: CostTier  # per-step online inference latency relative to baseline
    uncertainty_quality: CostTier  # expected richness of the predictive distribution
    integration_burden: CostTier  # repo wiring cost (deps, adapter, config, eval glue)
    # New third-party dependencies a real integration would likely pull in.
    new_dependencies: tuple[str, ...]
    # Where the family could plausibly help (used by the per-use-case recommendation).
    offline_use_cases: tuple[str, ...]
    # Literature breadcrumb; explicitly NOT repository evidence.
    literature_reference: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the model-family record."""
        return {
            "key": self.key,
            "title": self.title,
            "description": self.description,
            "compute_cost": self.compute_cost.value,
            "inference_latency": self.inference_latency.value,
            "uncertainty_quality": self.uncertainty_quality.value,
            "integration_burden": self.integration_burden.value,
            "new_dependencies": list(self.new_dependencies),
            "offline_use_cases": list(self.offline_use_cases),
            "literature_reference": self.literature_reference,
        }


# Conservative, literature-derived inventory of the families the issue names. Tiers are
# relative to the in-repo lightweight learned baseline (constant-velocity Gaussian and the
# graph predictor in robot_sf/planner/predictive_model.py), which anchors the LOW end.
MODEL_FAMILIES: tuple[ModelFamilySpec, ...] = (
    ModelFamilySpec(
        key="transformer",
        title="Transformer trajectory predictor (deterministic / regression head)",
        description=(
            "Self-attention over agent histories (and optionally map context) with a "
            "regression or simple mixture head. Strong point-accuracy potential; uncertainty "
            "is limited unless an explicit multi-modal or probabilistic head is added."
        ),
        compute_cost=CostTier.HIGH,
        inference_latency=CostTier.MEDIUM,
        uncertainty_quality=CostTier.MEDIUM,
        integration_burden=CostTier.MEDIUM,
        new_dependencies=("torch",),
        offline_use_cases=("online_prediction",),
        literature_reference=(
            "Trajectory-Transformer / Transformer-based motion forecasting literature "
            "(planning estimate only, not repo-measured)."
        ),
    ),
    ModelFamilySpec(
        key="agentformer",
        title="AgentFormer-like socio-temporal transformer",
        description=(
            "Jointly attends over the time and agent axes to model interactions, typically "
            "paired with a latent-variable (CVAE-style) head for multi-modal futures. Richer "
            "interaction modeling at notably higher compute and integration cost."
        ),
        compute_cost=CostTier.VERY_HIGH,
        inference_latency=CostTier.HIGH,
        uncertainty_quality=CostTier.HIGH,
        integration_burden=CostTier.HIGH,
        new_dependencies=("torch",),
        offline_use_cases=("online_prediction", "offline_scenario_generation"),
        literature_reference=(
            "AgentFormer (Yuan et al., 2021) family (planning estimate only, not repo-measured)."
        ),
    ),
    ModelFamilySpec(
        key="cvae",
        title="CVAE multi-modal trajectory predictor",
        description=(
            "Conditional variational autoencoder with a latent mode variable, sampling several "
            "plausible futures per agent. Moderate compute; sampling gives a usable predictive "
            "distribution but calibration must be verified, not assumed."
        ),
        compute_cost=CostTier.MEDIUM,
        inference_latency=CostTier.MEDIUM,
        uncertainty_quality=CostTier.HIGH,
        integration_burden=CostTier.MEDIUM,
        new_dependencies=("torch",),
        offline_use_cases=("online_prediction", "offline_scenario_generation"),
        literature_reference=(
            "Trajectron++/CVAE multi-modal forecasting literature "
            "(planning estimate only, not repo-measured)."
        ),
    ),
    ModelFamilySpec(
        key="diffusion",
        title="Diffusion / score-based trajectory predictor",
        description=(
            "Iterative denoising generative model producing diverse multi-modal futures. "
            "Highest sample diversity and arguably best for scenario / adversarial generation, "
            "but iterative sampling makes online inference latency the worst of the four and "
            "integration cost the highest (new sampler + dependency surface)."
        ),
        compute_cost=CostTier.VERY_HIGH,
        inference_latency=CostTier.VERY_HIGH,
        uncertainty_quality=CostTier.HIGH,
        integration_burden=CostTier.VERY_HIGH,
        new_dependencies=("torch", "diffusers?"),
        offline_use_cases=("offline_scenario_generation", "adversarial_stress"),
        literature_reference=(
            "MID / LED diffusion trajectory-prediction literature "
            "(planning estimate only, not repo-measured)."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Offline-evaluation entry-point surfaces (probed, fail-closed).
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
    """A surface an offline heavy-model experiment must be able to import to be wired in.

    ``module`` is an importable dotted path; ``file_path`` is the repo-relative file that backs
    it (verified for presence); ``required_symbols`` are public names the surface must still
    expose. ``required`` surfaces fail the verdict closed when broken.
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


# Canonical offline-evaluation surfaces named in issue #2845's agent-exec spec. These already
# exist on main; the probe keeps the inventory honest if any are renamed before the study runs.
ENTRY_POINT_SURFACES: tuple[EntryPointSpec, ...] = (
    EntryPointSpec(
        key="forecast_metrics",
        title="Forecast error metrics (ADE/FDE/miss-rate)",
        module="robot_sf.benchmark.forecast_metrics",
        file_path="robot_sf/benchmark/forecast_metrics.py",
        required_symbols=("evaluate_forecast_batch", "ForecastMetricRow"),
        note="Heavy-model forecasts would be scored through this metric surface.",
    ),
    EntryPointSpec(
        key="forecast_calibration",
        title="Forecast calibration / reliability report",
        module="robot_sf.benchmark.forecast_calibration_report",
        file_path="robot_sf/benchmark/forecast_calibration_report.py",
        required_symbols=("build_forecast_calibration_report",),
        note="Uncertainty-quality claims must pass calibration here, not be assumed.",
    ),
    EntryPointSpec(
        key="forecast_conformal",
        title="Forecast conformal-coverage pilot",
        module="robot_sf.benchmark.forecast_conformal_pilot",
        file_path="robot_sf/benchmark/forecast_conformal_pilot.py",
        required_symbols=("build_forecast_conformal_pilot_report",),
        note="Distribution-free coverage check for any probabilistic predictor.",
    ),
    EntryPointSpec(
        key="forecast_dataset",
        title="Bounded durable forecast dataset recorder",
        module="robot_sf.benchmark.forecast_dataset_recorder",
        file_path="robot_sf/benchmark/forecast_dataset_recorder.py",
        required_symbols=(
            "record_forecast_dataset_from_trace_exports",
            "validate_forecast_dataset_manifest",
        ),
        note="The minimum offline experiment needs a versioned held-out dataset from here.",
    ),
    EntryPointSpec(
        key="forecast_batch",
        title="Forecast batch / actor-forecast data contracts",
        module="robot_sf.benchmark.forecast_batch",
        file_path="robot_sf/benchmark/forecast_batch.py",
        required_symbols=("ForecastBatch", "ActorForecast"),
        note="Any heavy-model adapter must emit ActorForecast/ForecastBatch to be scored.",
    ),
    EntryPointSpec(
        key="forecast_baseline_comparison",
        title="Forecast baseline comparison harness",
        module="robot_sf.benchmark.forecast_baseline_comparison",
        file_path="robot_sf/benchmark/forecast_baseline_comparison.py",
        required_symbols=("compare_forecast_baselines",),
        note="Heavy models must beat this lightweight comparison ladder to be worth adding.",
    ),
    EntryPointSpec(
        key="lightweight_baseline",
        title="Lightweight CV/graph predictor baseline (#2844/#2915 ladder)",
        module="robot_sf.benchmark.pedestrian_forecast",
        file_path="robot_sf/benchmark/pedestrian_forecast.py",
        required_symbols=("constant_velocity_gaussian_baseline", "PedestrianForecast"),
        note="Comparator the heavy families are triaged against (the LOW cost tier anchor).",
    ),
    EntryPointSpec(
        key="graph_predictor",
        title="Learned graph/transformer predictor scaffold (#2844)",
        module="robot_sf.planner.predictive_model",
        file_path="robot_sf/planner/predictive_model.py",
        required_symbols=("PredictiveTrajectoryModel", "compute_ade_fde"),
        required=False,
        note="Context only: an existing learned scaffold a transformer study could extend.",
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

    This is read-only introspection. It never trains, never runs inference, and never executes
    simulation behavior; it only checks that the surface an experiment would extend is reachable
    on this checkout.

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
# Minimum offline-experiment prerequisites (declared, presence-probed, not implemented).
# ---------------------------------------------------------------------------


class PrerequisiteStatus(StrEnum):
    """Whether a prerequisite for the minimum offline heavy-model experiment exists yet."""

    PRESENT = "present"
    """The prerequisite already exists on this checkout (path probe matched)."""

    ABSENT = "absent"
    """Not yet built. Expected for the assessment-stage study; a planned blocker."""

    EXTERNAL = "external"
    """Depends on data/assets/decisions not in repo; cannot be satisfied by local code alone."""


@dataclass(frozen=True)
class PrerequisiteSpec:
    """A capability or artifact the minimum offline experiment needs before it can run.

    ``probe_paths`` are repo-relative paths (or glob patterns) whose presence would indicate the
    prerequisite is satisfied. When empty and ``external`` is set, the prerequisite is reported
    as a standing EXTERNAL blocker rather than something local code can produce.
    """

    key: str
    title: str
    description: str
    blocks: tuple[str, ...]  # which DoD item(s) / experiment step(s) this gates
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


# Ordered inventory of what the minimum offline experiment still needs. Paths are probed so an
# item flips to "present" automatically once the work lands, instead of going stale. Keep this
# aligned with the issue's Definition of Done and the "minimum viable offline experiment" scope.
EXPERIMENT_PREREQUISITES: tuple[PrerequisiteSpec, ...] = (
    PrerequisiteSpec(
        key="staged_holdout_dataset",
        title="Bounded durable held-out forecast dataset + manifest",
        description=(
            "A versioned, split-disjoint forecast dataset (with manifest) recorded via "
            "forecast_dataset_recorder, durable enough to score a heavy model against the "
            "baseline ladder. Not staged in-repo today."
        ),
        blocks=("Definition of Done: minimum viable offline experiment on durable data",),
        probe_paths=(
            "configs/forecast/heavy_model_holdout*.yaml",
            "configs/forecast/heavy_model_holdout*.yml",
        ),
        related_issues=(2844, 3065),
    ),
    PrerequisiteSpec(
        key="heavy_model_adapter",
        title="Heavy-model -> ActorForecast/ForecastBatch adapter",
        description=(
            "An adapter that turns a transformer/CVAE/diffusion model's output into the "
            "ActorForecast/ForecastBatch contract so it can be scored by forecast_metrics. "
            "Absent today (no heavy model is wired into the forecast surfaces)."
        ),
        blocks=("Definition of Done: run heavy model through the offline evaluator",),
        probe_paths=(
            "robot_sf/benchmark/forecast_heavy_model_adapter.py",
            "robot_sf/planner/forecast_heavy_model_adapter.py",
        ),
    ),
    PrerequisiteSpec(
        key="cpu_runtime_budget",
        title="CPU-bounded runtime budget / config for the offline run",
        description=(
            "A config-first, CPU-only runtime budget for the minimum experiment (issue requires "
            "no SLURM/GPU campaign). Absent today; needed before any run records runtime."
        ),
        blocks=("Validation: record runtime for the offline run within a CPU budget",),
        probe_paths=("configs/forecast/heavy_model_cpu_budget*.yaml",),
    ),
    PrerequisiteSpec(
        key="study_report",
        title="Heavy-model study report (tradeoffs + MVP experiment + decision)",
        description=(
            "The docs/context study report capturing model-family tradeoffs, one concrete "
            "minimum viable experiment (or stop recommendation), and a continue/revise/stop "
            "decision boundary. The companion of this inventory; tracked as a prerequisite so "
            "the inventory flags when it has not yet been authored."
        ),
        blocks=("Definition of Done: study report with tradeoffs and a decision",),
        probe_paths=(
            "docs/context/forecast_heavy_model_study_2026-06-20.md",
            "docs/context/forecast_heavy_model_study*.md",
        ),
    ),
    PrerequisiteSpec(
        key="dependency_decision",
        title="Maintainer decision on new heavy-model dependencies",
        description=(
            "An explicit decision to add the heavy-model dependency surface (torch is present, "
            "but diffusion/graph extras and training glue are not approved). The issue forbids "
            "adding dependencies in this slice, so this is an external standing decision."
        ),
        blocks=("Out of scope: adding the heavy-model dependency/training surface",),
        external=True,
        related_issues=(2835, 2843),
    ),
    PrerequisiteSpec(
        key="trained_checkpoint",
        title="Trained heavy-model checkpoint(s)",
        description=(
            "At least one trained transformer/CVAE/diffusion checkpoint to evaluate. Training is "
            "out of scope for this assessment-first slice (no GPU campaign), so this is an "
            "external standing blocker until a training decision is made."
        ),
        blocks=("Out of scope: producing a heavy-model checkpoint to evaluate",),
        external=True,
        related_issues=(2835,),
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
    satisfied so the inventory updates itself once the work lands. External prerequisites with no
    local probe path are always reported as standing blockers.

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


class MinimumExperimentStatus(StrEnum):
    """Roll-up status for whether the minimum offline experiment could run yet."""

    READY = "ready"
    """All entry-point surfaces import and every non-external prerequisite is satisfied."""

    BLOCKED = "blocked"
    """A required surface is missing or a non-external prerequisite is still absent."""


@dataclass
class InventoryReport:
    """Aggregate inventory: model families, probed surfaces, and probed prerequisites."""

    model_families: tuple[ModelFamilySpec, ...]
    surfaces: tuple[SurfaceProbeResult, ...]
    prerequisites: tuple[PrerequisiteProbeResult, ...]

    @property
    def required_surfaces(self) -> tuple[SurfaceProbeResult, ...]:
        """Probed surfaces marked required (these drive the fail-closed import verdict)."""
        return tuple(s for s in self.surfaces if s.spec.required)

    @property
    def surface_blockers(self) -> tuple[SurfaceProbeResult, ...]:
        """Required surfaces that are not present (import-verdict blockers)."""
        return tuple(s for s in self.surfaces if s.is_blocker)

    @property
    def ok(self) -> bool:
        """Import verdict: True iff every *required* entry-point surface is present.

        Missing prerequisites are planned work and do not flip this verdict; only a broken
        required surface (an experiment could not be wired in at all) fails closed. Use
        :attr:`minimum_experiment_status` for the broader ready/blocked roll-up.
        """
        return not self.surface_blockers

    @property
    def pending_prerequisites(self) -> tuple[PrerequisiteProbeResult, ...]:
        """Prerequisites not yet satisfied (absent or external standing blockers)."""
        return tuple(p for p in self.prerequisites if not p.satisfied)

    @property
    def local_pending_prerequisites(self) -> tuple[PrerequisiteProbeResult, ...]:
        """Non-external prerequisites that are still absent (locally closeable blockers)."""
        return tuple(
            p
            for p in self.prerequisites
            if not p.satisfied and p.status is not PrerequisiteStatus.EXTERNAL
        )

    @property
    def minimum_experiment_status(self) -> MinimumExperimentStatus:
        """Whether the minimum offline experiment could run on this checkout.

        ``READY`` only when the import verdict passes *and* no non-external prerequisite is
        still absent. External standing blockers (dependency decision, trained checkpoint) are
        excluded because they are out of this assessment slice's scope; they are still reported,
        but they do not by themselves define local readiness.
        """
        if self.ok and not self.local_pending_prerequisites:
            return MinimumExperimentStatus.READY
        return MinimumExperimentStatus.BLOCKED

    def exit_code(self) -> int:
        """Process exit code for the preflight.

        Returns:
            ``0`` when the import verdict is OK, ``1`` otherwise. The minimum-experiment
            ready/blocked roll-up is reported but does not fail the preflight, because being
            blocked on planned prerequisites is the expected assessment-stage state.
        """
        return 0 if self.ok else 1

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the full inventory report."""
        return {
            "issue": ISSUE,
            "ok": self.ok,
            "minimum_experiment_status": self.minimum_experiment_status.value,
            "model_families": [m.to_dict() for m in self.model_families],
            "surfaces": [s.to_dict() for s in self.surfaces],
            "prerequisites": [p.to_dict() for p in self.prerequisites],
            "summary": {
                "model_families_total": len(self.model_families),
                "required_surfaces_present": sum(1 for s in self.required_surfaces if s.present),
                "required_surfaces_total": len(self.required_surfaces),
                "prerequisites_present": sum(1 for p in self.prerequisites if p.satisfied),
                "prerequisites_total": len(self.prerequisites),
                "pending_prerequisites": len(self.pending_prerequisites),
                "local_pending_prerequisites": len(self.local_pending_prerequisites),
            },
        }


def build_inventory_report(root: Path | None = None) -> InventoryReport:
    """Build the full heavy forecast-model inventory for the active checkout.

    Args:
        root: Repository root; defaults to :func:`repo_root`.

    Returns:
        A populated :class:`InventoryReport`.
    """
    base = root if root is not None else repo_root()
    surfaces = tuple(probe_entry_point(spec, base) for spec in ENTRY_POINT_SURFACES)
    prerequisites = tuple(probe_prerequisite(spec, base) for spec in EXPERIMENT_PREREQUISITES)
    return InventoryReport(
        model_families=MODEL_FAMILIES,
        surfaces=surfaces,
        prerequisites=prerequisites,
    )


def render_markdown(report: InventoryReport) -> str:
    """Render the inventory report as a compact Markdown summary for the CLI.

    Args:
        report: The aggregate inventory report to render.

    Returns:
        A Markdown string with the verdict banner and the model-family / surface /
        prerequisite sections.
    """
    verdict = "PASS" if report.ok else "FAIL"
    lines: list[str] = []
    lines.append(f"# Heavy forecast-model inventory (#{ISSUE}): {verdict}")
    present_req = sum(1 for s in report.required_surfaces if s.present)
    total_req = len(report.required_surfaces)
    present_pre = sum(1 for p in report.prerequisites if p.satisfied)
    total_pre = len(report.prerequisites)
    lines.append(
        f"Minimum offline experiment: {report.minimum_experiment_status.value.upper()} | "
        f"Required surfaces present: {present_req}/{total_req} | "
        f"Prerequisites present: {present_pre}/{total_pre} | "
        f"Pending blockers: {len(report.pending_prerequisites)}"
    )

    lines.append("")
    lines.append(
        "## Candidate model families (literature-derived planning estimates, not measured)"
    )
    for m in report.model_families:
        lines.append(f"- **{m.title}**")
        lines.append(
            f"  - compute={m.compute_cost.value}, latency={m.inference_latency.value}, "
            f"uncertainty={m.uncertainty_quality.value}, integration={m.integration_burden.value}"
        )
        lines.append(f"  - candidate use cases: {', '.join(m.offline_use_cases)}")

    lines.append("")
    lines.append("## Offline-evaluation entry-point surfaces (fail-closed)")
    for s in report.surfaces:
        tag = "required" if s.spec.required else "optional"
        mark = "ok" if s.present else "BLOCKER" if s.is_blocker else s.status.value
        lines.append(f"- [{mark}] {s.spec.title} ({tag}) — {s.spec.module}")
        if s.detail:
            lines.append(f"  - {s.detail}")

    lines.append("")
    lines.append("## Minimum offline-experiment prerequisites / blockers")
    for p in report.prerequisites:
        mark = "present" if p.satisfied else p.status.value
        lines.append(f"- [{mark}] {p.spec.title}")
        for blocked in p.spec.blocks:
            lines.append(f"  - blocks: {blocked}")

    return "\n".join(lines)
