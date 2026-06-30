"""Read-only diagnostic inventory of SNQI weight sets and their provenance.

Motivation (issue #3723)
------------------------
Several weight sets in this repository carry — or imply — the "canonical" SNQI
label while disagreeing *qualitatively* about which term dominates:

* ``recompute_snqi_weights("canonical")`` (code default) is collision-dominant
  on a raw magnitude scale (``w_collisions = 2.0``).
* ``model/snqi_canonical_weights_v1.json`` is jerk-dominant (``w_jerk = 3.0``),
  also on a raw scale.
* ``configs/benchmarks/snqi_weights_camera_ready_v{2,3}.json`` are normalized
  (weights sum ~= 1) and are time- / near-miss-dominant respectively.

Because the same "canonical" label maps to contradictory sets, a user calling
``recompute_snqi_weights("canonical")`` can obtain a *different* planner ranking
than a user loading ``model/snqi_canonical_weights_v1.json`` — with no in-repo
signal that the two differ.

Scope of this module (deliberately narrow)
------------------------------------------
This is a **diagnostic inventory / preflight only**. It:

* discovers all known SNQI weight sets (code default + shipped JSON),
* records each set's weights, dominant term, and numeric scale,
* detects and reports provenance conflicts between sources that claim the
  "canonical" designation,
* offers a **fail-closed** preflight that raises when such a conflict exists.

It does **not** choose a canonical set, re-tune weights, change normalization,
or alter the SNQI metric / scoring in any way. Picking the source of truth is a
maintainer decision (issue #3723 is ``decision-required``); the per-term
scaling split is tracked separately in #3699.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from robot_sf.benchmark.snqi.compute import WEIGHT_NAMES, recompute_snqi_weights
from robot_sf.common.artifact_paths import get_repository_root

if TYPE_CHECKING:
    from pathlib import Path

# Tolerance for "weights sum to 1" (normalized simplex) detection.
_SIMPLEX_TOL = 1e-3
# Tolerance when comparing normalized weight *directions* across sources.
_DIRECTION_TOL = 1e-6

# Bounded shipped-file discovery. Keep this intentionally narrow so generated
# reports or unrelated schemas do not become provenance inputs.
_SHIPPED_WEIGHT_FILE_GLOBS = (
    "model/*snqi*weight*.json",
    "configs/**/*snqi*weight*.json",
)


@dataclass(frozen=True)
class WeightSourceSpec:
    """Static description of a known SNQI weight source.

    Attributes:
        name: Stable identifier for the source.
        kind: ``"code_default"`` or ``"shipped_json"``.
        relpath: Repository-root-relative JSON path (``None`` for the code default).
        declares_canonical: True when the source carries the "canonical"
            designation (by function method, file name, or description) and is
            therefore expected to agree with every other canonical-labeled set.
    """

    name: str
    kind: str
    relpath: str | None
    declares_canonical: bool


# Registry of weight sources known to this repository. Extend here (rather than
# duplicating discovery logic) when a new shipped weight file is introduced.
WEIGHT_SOURCES: tuple[WeightSourceSpec, ...] = (
    WeightSourceSpec(
        name="code_default",
        kind="code_default",
        relpath=None,
        declares_canonical=True,
    ),
    WeightSourceSpec(
        name="model_canonical_v1",
        kind="shipped_json",
        relpath="model/snqi_canonical_weights_v1.json",
        declares_canonical=True,
    ),
    WeightSourceSpec(
        name="camera_ready_v1",
        kind="shipped_json",
        relpath="configs/benchmarks/snqi_weights_camera_ready_v1.json",
        declares_canonical=False,
    ),
    WeightSourceSpec(
        name="camera_ready_v2",
        kind="shipped_json",
        relpath="configs/benchmarks/snqi_weights_camera_ready_v2.json",
        declares_canonical=False,
    ),
    WeightSourceSpec(
        name="camera_ready_v3",
        kind="shipped_json",
        relpath="configs/benchmarks/snqi_weights_camera_ready_v3.json",
        declares_canonical=False,
    ),
)


def _source_name_from_relpath(relpath: str) -> str:
    """Build a stable diagnostic source name for an unregistered shipped file.

    Returns:
        Source name safe for JSON reports and deterministic comparisons.
    """
    stem = relpath.removesuffix(".json")
    return "unregistered_" + re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()


def discover_shipped_weight_source_specs(repo_root: Path) -> list[WeightSourceSpec]:
    """Discover shipped SNQI weight JSON files under bounded source directories.

    Registered files keep their explicit names and canonical declarations. Matching files that are
    not in :data:`WEIGHT_SOURCES` are returned as ``unregistered_shipped_json`` records so the
    inventory reports every discovered shipped weight file instead of silently relying on the static
    registry.

    Returns:
        Shipped JSON source specs, including registered files and any unregistered matches.
    """
    registered_by_path = {spec.relpath: spec for spec in WEIGHT_SOURCES if spec.relpath is not None}
    discovered: dict[str, WeightSourceSpec] = dict(registered_by_path)

    for pattern in _SHIPPED_WEIGHT_FILE_GLOBS:
        for path in repo_root.glob(pattern):
            if not path.is_file():
                continue
            relpath = path.relative_to(repo_root).as_posix()
            if relpath in discovered:
                continue
            discovered[relpath] = WeightSourceSpec(
                name=_source_name_from_relpath(relpath),
                kind="unregistered_shipped_json",
                relpath=relpath,
                declares_canonical="canonical" in relpath.lower(),
            )

    return [discovered[relpath] for relpath in sorted(discovered)]


@dataclass
class WeightSetRecord:
    """A single discovered SNQI weight set with derived provenance facts."""

    name: str
    kind: str
    relpath: str | None
    declares_canonical: bool
    available: bool
    weights: dict[str, float] = field(default_factory=dict)
    weight_sum: float | None = None
    dominant_term: str | None = None
    scale_class: str | None = None
    content_sha256: str | None = None
    load_error: str | None = None

    def normalized_direction(self) -> dict[str, float] | None:
        """Return weights rescaled to sum 1 (the set's *direction*), or None.

        Comparing directions lets us detect qualitative disagreement
        independent of overall scale (raw vs normalized).

        Returns:
            A mapping of weight name -> share of the total, or ``None`` when the
            set is empty or sums to ~0 (no meaningful direction).
        """
        if not self.weights or self.weight_sum is None:
            return None
        total = self.weight_sum
        if abs(total) < _DIRECTION_TOL:
            return None
        return {k: self.weights.get(k, 0.0) / total for k in WEIGHT_NAMES}


@dataclass
class WeightProvenanceConflict:
    """A detected provenance conflict between weight sources."""

    kind: str
    severity: str  # "error" | "warning" | "info"
    sources: list[str]
    detail: str


@dataclass
class WeightSourceComparison:
    """Diagnostic comparison between the code default and a shipped source."""

    source: str
    relpath: str | None
    relationship: str
    code_default_dominant_term: str | None
    source_dominant_term: str | None
    source_scale_class: str | None
    available: bool
    """Whether the direction comparison itself could be computed.

    This is distinct from source-load state: a source may load successfully yet
    still be uncomparable (e.g. empty or zero-sum weights with no meaningful
    direction), in which case ``available`` is False and ``load_error`` carries
    the reason.
    """
    load_error: str | None = None


@dataclass
class WeightInventoryReport:
    """Structured result of an SNQI weight-set inventory pass."""

    records: list[WeightSetRecord]
    conflicts: list[WeightProvenanceConflict]

    @property
    def has_blocking_conflict(self) -> bool:
        """True if any conflict is severe enough to fail a preflight."""
        return any(c.severity == "error" for c in self.conflicts)

    @property
    def source_summary(self) -> dict[str, object]:
        """Summarize discovery/provenance surface for preflight consumers."""
        shipped_records = [r for r in self.records if r.relpath is not None]
        registered_records = [r for r in self.records if r.kind in {"code_default", "shipped_json"}]
        unregistered_records = [r for r in self.records if r.kind == "unregistered_shipped_json"]
        canonical_records = [r for r in self.records if r.declares_canonical]
        unavailable_records = [r for r in self.records if not r.available]
        blocking_conflicts = [c for c in self.conflicts if c.severity == "error"]
        return {
            "registered_source_count": len(WEIGHT_SOURCES),
            "discovered_shipped_source_count": len(shipped_records),
            "unregistered_shipped_source_count": len(unregistered_records),
            "canonical_declaring_source_count": len(canonical_records),
            "unavailable_source_count": len(unavailable_records),
            "blocking_conflict_count": len(blocking_conflicts),
            "registered_sources": [r.name for r in registered_records],
            "discovered_shipped_sources": [r.name for r in shipped_records],
            "unregistered_shipped_sources": [r.name for r in unregistered_records],
            "canonical_declaring_sources": [r.name for r in canonical_records],
            "unavailable_sources": [r.name for r in unavailable_records],
            "blocking_conflict_kinds": [c.kind for c in blocking_conflicts],
            "code_default_shipped_direction_comparisons": [
                {
                    "source": comparison.source,
                    "relpath": comparison.relpath,
                    "relationship": comparison.relationship,
                    "code_default_dominant_term": comparison.code_default_dominant_term,
                    "source_dominant_term": comparison.source_dominant_term,
                    "source_scale_class": comparison.source_scale_class,
                    "available": comparison.available,
                    "load_error": comparison.load_error,
                }
                for comparison in compare_code_default_to_shipped_sources(self.records)
            ],
        }

    def to_dict(self) -> dict:
        """Build a JSON-serializable representation of the report.

        Returns:
            A dict with ``source_summary``, ``records``, ``conflicts``, and
            ``has_blocking_conflict``.
        """
        return {
            "source_summary": self.source_summary,
            "records": [
                {
                    "name": r.name,
                    "kind": r.kind,
                    "relpath": r.relpath,
                    "declares_canonical": r.declares_canonical,
                    "available": r.available,
                    "weights": r.weights,
                    "weight_sum": r.weight_sum,
                    "dominant_term": r.dominant_term,
                    "scale_class": r.scale_class,
                    "content_sha256": r.content_sha256,
                    "load_error": r.load_error,
                }
                for r in self.records
            ],
            "conflicts": [
                {
                    "kind": c.kind,
                    "severity": c.severity,
                    "sources": c.sources,
                    "detail": c.detail,
                }
                for c in self.conflicts
            ],
            "has_blocking_conflict": self.has_blocking_conflict,
        }


class SNQIWeightProvenanceError(RuntimeError):
    """Raised by the fail-closed preflight when a blocking conflict exists."""


def _extract_weights(raw: object) -> dict[str, float]:
    """Pull a flat ``WEIGHT_NAMES`` mapping out of a loaded JSON payload.

    Accepts either a flat ``{w_*: value}`` mapping (camera-ready files) or a
    payload nesting the mapping under a ``"weights"`` key (model artifact).

    Returns:
        A validated ``{weight_name: float}`` mapping over :data:`WEIGHT_NAMES`.

    Raises:
        ValueError: If the payload shape is unrecognized, a required key is
            missing, or a value is non-numeric / non-finite. Callers convert
            this into a recorded ``load_error`` rather than crashing the sweep.
    """
    if isinstance(raw, dict) and "weights" in raw and isinstance(raw["weights"], dict):
        candidate = raw["weights"]
    elif isinstance(raw, dict):
        candidate = raw
    else:
        raise ValueError(f"unexpected JSON top-level type: {type(raw).__name__}")

    missing = [k for k in WEIGHT_NAMES if k not in candidate]
    if missing:
        raise ValueError(f"missing weight keys: {missing}")

    out: dict[str, float] = {}
    for k in WEIGHT_NAMES:
        try:
            fv = float(candidate[k])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric weight for {k}: {candidate[k]!r}") from exc
        if not math.isfinite(fv):
            raise ValueError(f"non-finite weight for {k}: {fv}")
        out[k] = fv
    return out


def _finalize_record(record: WeightSetRecord) -> None:
    """Populate derived fields (sum, dominant term, scale class) in place."""
    record.weight_sum = float(sum(record.weights.values()))
    # Dominant term = key with the largest weight (ties broken by WEIGHT_NAMES order).
    record.dominant_term = max(WEIGHT_NAMES, key=lambda k: record.weights.get(k, 0.0))
    record.scale_class = (
        "normalized_simplex" if abs(record.weight_sum - 1.0) <= _SIMPLEX_TOL else "raw"
    )


def _weights_content_sha256(weights: dict[str, float]) -> str:
    """Return stable fingerprint for synthetic/code-defined weight mappings."""
    payload = json.dumps(weights, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_record(spec: WeightSourceSpec, repo_root: Path) -> WeightSetRecord:
    """Load a single source into a :class:`WeightSetRecord` (never raises).

    Returns:
        The populated record; ``available=False`` with ``load_error`` set when
        the shipped JSON is missing or malformed.
    """
    record = WeightSetRecord(
        name=spec.name,
        kind=spec.kind,
        relpath=spec.relpath,
        declares_canonical=spec.declares_canonical,
        available=False,
    )

    if spec.kind == "code_default":
        # Read the code default through the public API so this inventory tracks
        # whatever recompute_snqi_weights("canonical") actually returns. We pass
        # empty baseline stats because the "canonical" method ignores them.
        weights = dict(recompute_snqi_weights({}, method="canonical").weights)
        record.weights = {k: float(weights[k]) for k in WEIGHT_NAMES}
        record.content_sha256 = _weights_content_sha256(record.weights)
        record.available = True
        _finalize_record(record)
        return record

    assert spec.relpath is not None  # shipped_json always has a path
    path = repo_root / spec.relpath
    if not path.exists():
        record.load_error = "file not found"
        return record
    try:
        raw_bytes = path.read_bytes()
        record.content_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        raw = json.loads(raw_bytes.decode("utf-8"))
        record.weights = _extract_weights(raw)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        record.load_error = str(exc)
        return record
    record.available = True
    _finalize_record(record)
    return record


def inventory_weight_sets(repo_root: Path | None = None) -> list[WeightSetRecord]:
    """Discover and load registered and bounded shipped SNQI weight sets.

    Args:
        repo_root: Repository root used to resolve shipped JSON paths. Defaults
            to :func:`get_repository_root`.

    Returns:
        One :class:`WeightSetRecord` per registered or discovered shipped source, with the
        code-default source first and shipped files in path order. Sources that fail to load are
        returned with
        ``available=False`` and a populated ``load_error`` (fail-closed: a
        missing/broken file is surfaced, not silently skipped).
    """
    root = (repo_root or get_repository_root()).resolve()
    shipped = discover_shipped_weight_source_specs(root)
    specs = [spec for spec in WEIGHT_SOURCES if spec.relpath is None] + shipped
    return [_build_record(spec, root) for spec in specs]


def _directions_disagree(a: dict[str, float], b: dict[str, float]) -> bool:
    """Compare two normalized weight directions.

    Returns:
        True if any component differs beyond :data:`_DIRECTION_TOL`.
    """
    return any(abs(a[k] - b[k]) > _DIRECTION_TOL for k in WEIGHT_NAMES)


def _direction_relationship(a: WeightSetRecord, b: WeightSetRecord) -> str:
    """Classify two records by their scale-independent weight direction.

    Returns:
        ``same_direction``, ``different_direction``, or ``unavailable``.
    """
    if not a.available or not b.available:
        return "unavailable"
    da, db = a.normalized_direction(), b.normalized_direction()
    if da is None or db is None:
        return "unavailable"
    if _directions_disagree(da, db):
        return "different_direction"
    return "same_direction"


def compare_code_default_to_shipped_sources(
    records: list[WeightSetRecord],
) -> list[WeightSourceComparison]:
    """Compare the code-default weights against every discovered shipped JSON source.

    The issue #3723 ambiguity is specifically code default versus shipped JSON. This
    diagnostic matrix makes every shipped source explicit without choosing a canonical set.

    Returns:
        One comparison per shipped JSON source in the inventory records.
    """
    code_default = next((r for r in records if r.name == "code_default"), None)
    if code_default is None:
        return []

    comparisons: list[WeightSourceComparison] = []
    for record in records:
        if record.relpath is None:
            continue
        relationship = _direction_relationship(code_default, record)
        comparison_available = relationship != "unavailable"
        if record.load_error is not None:
            # The source failed to load; surface the underlying load error.
            load_error = record.load_error
        elif not comparison_available:
            # The source loaded but has no comparable direction (empty or
            # zero-sum weights); report a comparison-level reason rather than
            # leaving an unexplained ``unavailable`` relationship.
            load_error = "no_comparable_direction"
        else:
            load_error = None
        comparisons.append(
            WeightSourceComparison(
                source=record.name,
                relpath=record.relpath,
                relationship=relationship,
                code_default_dominant_term=code_default.dominant_term,
                source_dominant_term=record.dominant_term,
                source_scale_class=record.scale_class,
                available=comparison_available,
                load_error=load_error,
            )
        )
    return comparisons


def _canonical_load_error_conflicts(
    canonical: list[WeightSetRecord],
) -> list[WeightProvenanceConflict]:
    """Fail-closed conflicts for canonical sources that did not load.

    Returns:
        One ``canonical_load_error`` (error) conflict per unavailable source.
    """
    return [
        WeightProvenanceConflict(
            kind="canonical_load_error",
            severity="error",
            sources=[r.name],
            detail=(
                f"canonical-declaring source '{r.name}' "
                f"({r.relpath or 'code default'}) failed to load: {r.load_error}"
            ),
        )
        for r in canonical
        if not r.available
    ]


def _canonical_direction_conflicts(
    loaded_canonical: list[WeightSetRecord],
) -> list[WeightProvenanceConflict]:
    """Pairwise direction disagreement among canonical-declaring sources.

    Returns:
        One ``canonical_direction_conflict`` (error) per disagreeing pair.
    """
    conflicts: list[WeightProvenanceConflict] = []
    for i in range(len(loaded_canonical)):
        for j in range(i + 1, len(loaded_canonical)):
            a, b = loaded_canonical[i], loaded_canonical[j]
            da, db = a.normalized_direction(), b.normalized_direction()
            if da is None or db is None or not _directions_disagree(da, db):
                continue
            conflicts.append(
                WeightProvenanceConflict(
                    kind="canonical_direction_conflict",
                    severity="error",
                    sources=[a.name, b.name],
                    detail=(
                        f"'{a.name}' (dominant={a.dominant_term}) and "
                        f"'{b.name}' (dominant={b.dominant_term}) both claim the "
                        "canonical SNQI designation but yield different weight "
                        "directions; loading one vs the other changes the ranking."
                    ),
                )
            )
    return conflicts


def _scale_split_conflicts(loaded: list[WeightSetRecord]) -> list[WeightProvenanceConflict]:
    """Warn when discovered sources mix raw and normalized scales.

    Returns:
        A single ``scale_split`` (warning) conflict, or an empty list.
    """
    scales = {r.scale_class for r in loaded if r.scale_class}
    if len(scales) <= 1:
        return []
    return [
        WeightProvenanceConflict(
            kind="scale_split",
            severity="warning",
            sources=[r.name for r in loaded],
            detail=(
                "discovered weight sources use mixed numeric scales "
                f"({sorted(scales)}); raw vs normalized scaling overlaps with #3699."
            ),
        )
    ]


def _duplicate_label_conflicts(
    loaded: list[WeightSetRecord],
) -> list[WeightProvenanceConflict]:
    """Flag distinct-label sources that ship identical weight directions.

    Returns:
        One ``duplicate_weights_distinct_label`` (info) per matching pair.
    """
    conflicts: list[WeightProvenanceConflict] = []
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            a, b = loaded[i], loaded[j]
            da, db = a.normalized_direction(), b.normalized_direction()
            if da is None or db is None or _directions_disagree(da, db):
                continue
            conflicts.append(
                WeightProvenanceConflict(
                    kind="duplicate_weights_distinct_label",
                    severity="info",
                    sources=[a.name, b.name],
                    detail=(
                        f"'{a.name}' and '{b.name}' ship identical weight "
                        "directions under different labels."
                    ),
                )
            )
    return conflicts


def _unregistered_weight_source_conflicts(
    records: list[WeightSetRecord],
) -> list[WeightProvenanceConflict]:
    """Fail closed when shipped SNQI weight files are absent from the provenance registry.

    Returns:
        Error conflict when unregistered shipped weight sources are present, otherwise empty list.
    """
    unregistered = [r for r in records if r.kind == "unregistered_shipped_json"]
    if not unregistered:
        return []
    return [
        WeightProvenanceConflict(
            kind="unregistered_shipped_weight_source",
            severity="error",
            sources=[r.name for r in unregistered],
            detail=(
                "discovered shipped SNQI weight JSON files not listed in WEIGHT_SOURCES; "
                "register or explicitly exclude them before treating provenance inventory complete."
            ),
        )
    ]


def detect_conflicts(records: list[WeightSetRecord]) -> list[WeightProvenanceConflict]:
    """Detect provenance conflicts among discovered weight sets.

    Conflict taxonomy:

    * ``canonical_load_error`` (error): a canonical-declaring source failed to
      load. Inventory cannot certify provenance, so this fails closed.
    * ``canonical_direction_conflict`` (error): two canonical-declaring sources
      disagree on their (scale-independent) weight direction — i.e. the same
      "canonical" label yields different rankings.
    * ``scale_split`` (warning): discovered sources use different numeric scales
      (raw vs normalized_simplex). Overlaps with #3699.
    * ``duplicate_weights_distinct_label`` (info): two differently-named sources
      ship identical weights (e.g. a camera-ready file duplicating the model
      canonical set), which can mask which name is authoritative.

    Returns:
        Conflicts in deterministic order (errors first, then warnings, then
        info; ties broken by kind then source names).
    """
    canonical = [r for r in records if r.declares_canonical]
    loaded_canonical = [r for r in canonical if r.available]
    loaded = [r for r in records if r.available]

    conflicts: list[WeightProvenanceConflict] = []
    conflicts += _canonical_load_error_conflicts(canonical)
    conflicts += _canonical_direction_conflicts(loaded_canonical)
    conflicts += _unregistered_weight_source_conflicts(records)
    conflicts += _scale_split_conflicts(loaded)
    conflicts += _duplicate_label_conflicts(loaded)

    severity_order = {"error": 0, "warning": 1, "info": 2}
    conflicts.sort(key=lambda c: (severity_order.get(c.severity, 9), c.kind, c.sources))
    return conflicts


def build_inventory_report(repo_root: Path | None = None) -> WeightInventoryReport:
    """Run a full inventory + conflict-detection pass.

    Returns:
        A :class:`WeightInventoryReport` with discovered records and conflicts.
    """
    records = inventory_weight_sets(repo_root)
    conflicts = detect_conflicts(records)
    return WeightInventoryReport(records=records, conflicts=conflicts)


def preflight_snqi_weight_sets(
    repo_root: Path | None = None,
    *,
    strict: bool = True,
) -> WeightInventoryReport:
    """Fail-closed preflight over SNQI weight-set provenance.

    Args:
        repo_root: Optional repository root override.
        strict: When True (default), raise :class:`SNQIWeightProvenanceError`
            if any blocking (``error``-severity) conflict is detected. When
            False, the report is returned regardless so callers can inspect it.

    Returns:
        The :class:`WeightInventoryReport`.

    Raises:
        SNQIWeightProvenanceError: If ``strict`` and a blocking conflict exists.
    """
    report = build_inventory_report(repo_root)
    if strict and report.has_blocking_conflict:
        blocking = [c for c in report.conflicts if c.severity == "error"]
        summary = "; ".join(f"[{c.kind}] {c.detail}" for c in blocking)
        raise SNQIWeightProvenanceError(
            "SNQI weight-set provenance preflight failed (fail-closed): " + summary
        )
    return report


__all__ = [
    "WEIGHT_SOURCES",
    "SNQIWeightProvenanceError",
    "WeightInventoryReport",
    "WeightProvenanceConflict",
    "WeightSetRecord",
    "WeightSourceComparison",
    "WeightSourceSpec",
    "build_inventory_report",
    "compare_code_default_to_shipped_sources",
    "detect_conflicts",
    "discover_shipped_weight_source_specs",
    "inventory_weight_sets",
    "preflight_snqi_weight_sets",
]
