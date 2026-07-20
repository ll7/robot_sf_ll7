#!/usr/bin/env python3
"""Compile a textual regulation/requirement excerpt into a scenario hypothesis.

This is a deterministic prototype (no LLM). It reads a ``regulation-record.v1``
YAML that embeds a textual regulation/requirement excerpt plus minimal
structured hints, compiles the excerpt into a parameterized ``robot_sf_ll7``
scenario config, and reports validity on three distinct axes:

1. ``compilation_validity``  - did deterministic extraction succeed and which
   parameters/warnings it produced. This is about the *compiler*, not the
   scenario.
2. ``schema_validity``       - does the generated scenario matrix satisfy the
   ``robot_sf.scenario_matrix.v1`` JSON Schema via
   ``robot_sf.benchmark.scenario_schema`` (reused, not rewritten). This is about
   the *shape* of the config, not whether the scenario behaves as intended.
3. ``scenario_validity``     - explicitly NOT assessed by this tool. Only
   execution through the benchmark runner plus human review can establish
   scenario validity, so this is always reported as ``not_assessed``.

Outputs are hypotheses: ``required_manual_review: true``,
``benchmark_evidence: false``, and a claim boundary stating the result is not
evidence until executed and reviewed.

Usage::

    uv run python scripts/tools/convert_regulation_to_scenario.py \\
        --record configs/scenarios/contracts/issue_6054_regulation_source_example.yaml \\
        --stdout

    uv run python scripts/tools/convert_regulation_to_scenario.py \\
        --record <regulation-record.v1.yaml> \\
        --output-yaml output/regulation_scenarios/example.scenario.yaml
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from robot_sf.benchmark.scenario_schema import (
    SCENARIO_MATRIX_SCHEMA_VERSION,
    validate_scenario_list,
    validate_scenario_matrix_metadata,
)

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent

REGULATION_RECORD_SCHEMA_VERSION = "regulation-record.v1"
GENERATION_METHOD = "deterministic_regulation_compile_v1"
DEFAULT_SEEDS: tuple[int, ...] = (6054, 6055, 6056)
DEFAULT_MAX_EPISODE_STEPS = 400

# Zone/context keywords -> template + map. The map is a structural approximation
# of the regulation's setting, not a replica (same convention as the
# failure-record converter). Keys are matched as whole-word, case-insensitive
# substrings of the excerpt/hints.
ZONE_TO_TEMPLATE: dict[str, str] = {
    "shared space": "shared_space",
    "shared-space": "shared_space",
    "pedestrian zone": "pedestrian_zone",
    "pedestrian-zone": "pedestrian_zone",
    "crossing": "crossing",
    "intersection": "crossing",
    "corridor": "corridor",
    "hallway": "corridor",
    "narrow passage": "corridor",
    "station": "station_platform",
    "platform": "station_platform",
    "sidewalk": "sidewalk",
    "pavement": "sidewalk",
    "plaza": "shared_space",
    "room": "room",
    "elevator": "room",
    "default": "shared_space",
}

TEMPLATE_TO_MAP: dict[str, Path] = {
    "shared_space": Path("maps/svg_maps/classic_merging.svg"),
    "pedestrian_zone": Path("maps/svg_maps/classic_urban_crossing.svg"),
    "crossing": Path("maps/svg_maps/classic_urban_crossing.svg"),
    "corridor": Path("maps/svg_maps/classic_head_on_corridor.svg"),
    "station_platform": Path("maps/svg_maps/classic_station_platform.svg"),
    "sidewalk": Path("maps/svg_maps/classic_crossing.svg"),
    "room": Path("maps/svg_maps/francis2023/francis2023_entering_room.svg"),
}

# Density wording -> ped_density (pedestrians per square meter of spawnable
# sidewalk area, per the scenario schema description). These are coarse
# deterministic buckets.
DENSITY_WORD_TO_VALUE: dict[str, float] = {
    "sparse": 0.02,
    "low": 0.02,
    "light": 0.02,
    "medium": 0.05,
    "moderate": 0.05,
    "high": 0.08,
    "dense": 0.08,
    "crowded": 0.12,
    "very high": 0.12,
}

DEFAULT_DENSITY_VALUE = 0.05


@dataclass
class CompiledParameters:
    """Deterministically extracted parameters from a regulation excerpt.

    ``extracted`` records each parameter actually pulled from the text (so the
    compiler is auditable). ``unmatched_clauses`` records sentences/clauses that
    the compiler could not interpret, so a human reviewer knows what was dropped.
    """

    zone_template: str = "shared_space"
    max_linear_speed: float | None = None
    clearance_m: float | None = None
    ped_density: float = DEFAULT_DENSITY_VALUE
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS
    extracted: list[dict[str, Any]] = field(default_factory=list)
    unmatched_clauses: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def meaningful_extractions(self) -> list[dict[str, Any]]:
        """Extractions actually pulled from the text (not pure defaults).

        Zone/density entries that were defaulted (no keyword match / source
        ``default``) do not count as real extractions. This keeps the
        compilation-validity axis honest: an excerpt the compiler could not
        interpret at all is reported as compilation-invalid even though the
        compiler still emits default zone and density values.
        """
        meaningful: list[dict[str, Any]] = []
        for entry in self.extracted:
            param = entry.get("parameter")
            if param == "zone_template" and entry.get("matched_keyword") is None:
                continue
            if param == "ped_density" and entry.get("source") == "default":
                continue
            meaningful.append(entry)
        return meaningful


def _map_file_for_output(template: str, *, output_path: Path | None = None) -> str:
    """Return the ``map_file`` path, relative to the output YAML when known.

    When ``output_path`` is None (``--stdout``) the repository-relative path is
    returned so the canonical location is documented.
    """
    repo_rel = TEMPLATE_TO_MAP.get(template, TEMPLATE_TO_MAP["shared_space"])
    if output_path is None:
        return repo_rel.as_posix()
    return os.path.relpath(_REPO_ROOT / repo_rel, start=output_path.parent)


def _resolve_generated_map_file(generated_yaml: Path, map_file: str) -> Path:
    """Resolve a ``map_file`` value against the scenario YAML location."""
    candidate = Path(map_file)
    if not candidate.is_absolute():
        candidate = generated_yaml.parent / candidate
    return candidate.resolve()


def _configure_logging(verbose: bool = False) -> None:
    """Configure loguru for CLI output."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO")


def _split_clauses(text: str) -> list[str]:
    """Split a regulation excerpt into clauses (sentences / list items).

    Newlines and semicolons are treated as clause separators in addition to
    sentence terminators, so multi-line/bulleted regulations decompose cleanly.
    """
    if not text:
        return []
    # Normalize whitespace/newlines into clause boundaries.
    normalized = re.sub(r"[\r\n]+", ". ", text)
    parts = re.split(r"(?i)(?<!\be\.g\.)(?<!\bi\.e\.)(?<=[.;])\s+", normalized)
    clauses: list[str] = []
    for part in parts:
        cleaned = part.strip().strip(";.").strip()
        if cleaned:
            clauses.append(cleaned)
    return clauses


def _find_float_after_keywords(text: str, keywords: tuple[str, ...]) -> float | None:
    """Return the first ``<number>`` that appears right after any keyword.

    Matches forms like ``1.5 m/s``, ``1.5 m``, ``1,5 meter``, ``>= 1.0 m``.
    Returns None if no match is found.
    """
    if not keywords:
        return None
    escaped_keywords = "|".join(re.escape(kw) for kw in keywords)
    pattern = re.compile(
        rf"(?:{escaped_keywords})[^0-9]{{0,20}}(?P<value>\d+(?:[.,]\d+)?)",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        raw = match.group("value").replace(",", ".")
        try:
            return float(raw)
        except ValueError:
            return None
    return None


SPEED_KEYWORDS = (
    "max speed",
    "maximum speed",
    "speed limit",
    "shall not exceed",
    "must not exceed",
    "no faster than",
    "at most",
    "not exceed",
)
SPEED_UNIT_KEYWORDS = ("m/s", "meters per second", "metres per second", "mps")


def _extract_speed(text: str) -> tuple[float | None, list[dict[str, Any]]]:
    """Extract a maximum linear speed (m/s) from the excerpt.

    Looks for speed-limit phrasings. Only the *first* match is used; additional
    matches are recorded as warnings so a reviewer is alerted that the
    regulation may specify more than one speed regime.
    """
    # Prefer a speed keyword followed by a number; fall back to a number
    # immediately preceding a speed unit (e.g. "1.5 m/s").
    value = _find_float_after_keywords(text, SPEED_KEYWORDS)
    if value is None:
        value = _find_float_before_units(text, SPEED_UNIT_KEYWORDS)

    extracted: list[dict[str, Any]] = []
    if value is None:
        return None, extracted

    extracted.append({"parameter": "max_linear_speed", "value_m_s": value})
    warning = _multi_speed_warning(text)
    if warning is not None:
        extracted.append({"parameter": "max_linear_speed", "warning": warning})
    return value, extracted


def _find_float_before_units(text: str, units: tuple[str, ...]) -> float | None:
    """Return the first number immediately preceding one of ``units``."""
    if not units:
        return None
    escaped_units = "|".join(re.escape(unit) for unit in units)
    pattern = re.compile(
        rf"(?P<value>\d+(?:[.,]\d+)?)\s*(?:{escaped_units})",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        try:
            return float(match.group("value").replace(",", "."))
        except ValueError:
            return None
    return None


def _multi_speed_warning(text: str) -> str | None:
    """Return a warning string if more than one distinct speed appears."""
    all_speeds: set[float] = set()
    for unit in SPEED_UNIT_KEYWORDS:
        for m in re.finditer(
            rf"(?P<value>\d+(?:[.,]\d+)?)\s*{re.escape(unit)}",
            text,
            re.IGNORECASE,
        ):
            try:
                all_speeds.add(float(m.group("value").replace(",", ".")))
            except ValueError:
                continue
    for kw in SPEED_KEYWORDS:
        v = _find_float_after_keywords(text, (kw,))
        if v is not None:
            all_speeds.add(v)
    if len(all_speeds) > 1:
        return (
            f"multiple speed values found {sorted(all_speeds)}; "
            "used the first match; review which regime applies"
        )
    return None


def _extract_clearance(text: str) -> tuple[float | None, list[dict[str, Any]]]:
    """Extract a clearance/distance requirement (m) from the excerpt."""
    clearance_keywords = (
        "clearance",
        "distance",
        "separation",
        "keep at least",
        "maintain at least",
        "minimum distance",
        "buffer",
        "gap of",
    )
    value = _find_float_after_keywords(text, clearance_keywords)
    extracted: list[dict[str, Any]] = []
    if value is not None:
        extracted.append({"parameter": "clearance_m", "value_m": value})
    return value, extracted


def _extract_density(text: str) -> tuple[float, list[dict[str, Any]]]:
    """Extract a pedestrian density value (peds/m^2) from the excerpt.

    Looks for an explicit numeric density first, then density wording, then
    falls back to the default.
    """
    extracted: list[dict[str, Any]] = []
    explicit = _find_float_after_keywords(
        text, ("pedestrian density", "density", "crowd density", "peds per", "per m")
    )
    if explicit is not None and 0.0 <= explicit <= 1.0:
        extracted.append({"parameter": "ped_density", "value": explicit, "source": "explicit"})
        return explicit, extracted

    lowered = text.lower()
    for word, value in DENSITY_WORD_TO_VALUE.items():
        if word in lowered:
            extracted.append({"parameter": "ped_density", "value": value, "source": f"word:{word}"})
            return value, extracted

    extracted.append(
        {"parameter": "ped_density", "value": DEFAULT_DENSITY_VALUE, "source": "default"}
    )
    return DEFAULT_DENSITY_VALUE, extracted


def _extract_zone(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Determine the zone template from the excerpt/hints.

    Longer/more-specific zone keys are checked first so that, e.g.,
    "shared space" wins over a generic "space". Returns the template name plus
    an audit record of which keyword matched.
    """
    lowered = text.lower()
    # Order by descending key length so multi-word zones take precedence.
    for key in sorted(ZONE_TO_TEMPLATE, key=len, reverse=True):
        if key == "default":
            continue
        if key in lowered:
            template = ZONE_TO_TEMPLATE[key]
            return template, [
                {"parameter": "zone_template", "value": template, "matched_keyword": key}
            ]
    template = ZONE_TO_TEMPLATE["default"]
    return template, [{"parameter": "zone_template", "value": template, "matched_keyword": None}]


def compile_regulation_excerpt(
    regulation_text: str, *, hints: dict[str, Any] | None = None
) -> CompiledParameters:
    """Deterministically compile a regulation excerpt into scenario parameters.

    This is the core compiler. It performs NO network/LLM calls. Every extracted
    parameter is recorded in ``extracted`` for auditability, and clauses that
    could not be interpreted are recorded in ``unmatched_clauses`` so a reviewer
    knows what was dropped.

    Args:
        regulation_text: The textual regulation/requirement excerpt.
        hints: Optional structured hints merged into the search text (e.g.
            ``context`` from the record). Hints never override an explicit
            extraction; they only broaden the keyword search surface.

    Returns:
        Compiled parameters with audit trails.
    """
    hints = hints or {}
    # Build a combined search surface: excerpt first (authoritative), then
    # hint text appended only to broaden zone/word matching.
    hint_text = " ".join(str(v) for v in (hints.get("context"), hints.get("setting")) if v)
    search_surface = f"{regulation_text} {hint_text}".strip()

    params = CompiledParameters()

    zone, zone_audit = _extract_zone(search_surface)
    params.zone_template = zone
    params.extracted.extend(zone_audit)

    speed, speed_audit = _extract_speed(regulation_text)
    params.max_linear_speed = speed
    params.extracted.extend(speed_audit)

    clearance, clearance_audit = _extract_clearance(regulation_text)
    params.clearance_m = clearance
    params.extracted.extend(clearance_audit)

    density, density_audit = _extract_density(regulation_text)
    params.ped_density = density
    params.extracted.extend(density_audit)

    steps = hints.get("max_episode_steps")
    if isinstance(steps, int) and steps > 0:
        params.max_episode_steps = steps
        params.extracted.append(
            {"parameter": "max_episode_steps", "value": steps, "source": "hint"}
        )

    # Determine unmatched clauses: clauses that contributed no extracted
    # parameter. A clause "contributes" if any extraction keyword fired inside
    # it. This is conservative (a clause mentioning "speed" but with no number
    # still counts as addressed only if a number was found within it).
    keyword_groups = (
        ("max speed", "maximum speed", "speed limit", "m/s", "mps"),
        ("clearance", "distance", "separation", "buffer"),
        ("density", "crowd", "sparse", "dense", "medium", "high", "low"),
    )
    for clause in _split_clauses(regulation_text):
        lowered_clause = clause.lower()
        addressed = False
        for group in keyword_groups:
            if any(kw in lowered_clause for kw in group):
                addressed = True
                break
        if not addressed:
            params.unmatched_clauses.append(clause)

    if params.max_linear_speed is None:
        params.warnings.append(
            "No maximum speed extracted; robot_config.max_linear_speed omitted. "
            "Review whether the regulation implies a speed limit."
        )
    if params.clearance_m is None:
        params.warnings.append(
            "No clearance/separation distance extracted; recorded as metadata only when present."
        )
    if not params.meaningful_extractions:
        params.warnings.append(
            "No parameters could be extracted from the excerpt; output is a "
            "near-empty hypothesis shell with defaulted zone/density."
        )

    return params


def _validate_regulation_record(record: dict[str, Any]) -> list[str]:
    """Validate a regulation record against ``regulation-record.v1``.

    Returns a list of human-readable errors; empty means valid. This validates
    the *input* envelope, separate from compilation/schema/scenario validity.
    """
    if not isinstance(record, dict):
        return ["Top-level record must be a mapping"]

    errors: list[str] = _validate_record_envelope(record)

    reg = record.get("regulation")
    if reg is None:
        errors.append("Missing required field: regulation")
        return errors
    if not isinstance(reg, dict):
        errors.append("regulation must be a mapping")
        return errors

    errors.extend(_validate_regulation_fields(reg))
    return errors


def _validate_record_envelope(record: dict[str, Any]) -> list[str]:
    """Validate the top-level envelope (schema_version only)."""
    errors: list[str] = []
    if "schema_version" not in record:
        errors.append("Missing required field: schema_version")
    elif record["schema_version"] != REGULATION_RECORD_SCHEMA_VERSION:
        errors.append(
            f"Unsupported schema_version: {record['schema_version']!r}; "
            f"expected {REGULATION_RECORD_SCHEMA_VERSION!r}"
        )
    return errors


def _validate_regulation_fields(reg: dict[str, Any]) -> list[str]:
    """Validate the ``regulation`` sub-mapping fields."""
    errors: list[str] = []
    for field_name in ("id", "regulation_text", "required_manual_review", "claim_boundary"):
        if field_name not in reg:
            errors.append(f"Missing required field in regulation: {field_name}")

    text = reg.get("regulation_text")
    if "regulation_text" in reg and not isinstance(text, str):
        errors.append("regulation.regulation_text must be a string")
    elif isinstance(text, str) and not text.strip():
        errors.append("regulation.regulation_text must not be empty")

    if "id" in reg and reg["id"] is None:
        errors.append("regulation.id must not be null")

    if "required_manual_review" in reg and reg["required_manual_review"] is not True:
        errors.append("required_manual_review must be true")

    claim = reg.get("claim_boundary")
    if "claim_boundary" in reg and not isinstance(claim, str):
        errors.append("regulation.claim_boundary must be a string")
    elif isinstance(claim, str) and "not evidence" not in claim.lower():
        errors.append("claim_boundary must state 'not evidence'")

    return errors


def _build_scenario_payload(
    regulation: dict[str, Any],
    params: CompiledParameters,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Build the ``robot_sf.scenario_matrix.v1`` payload from compiled params."""
    reg = copy.deepcopy(regulation)
    raw_id = reg.get("id")
    reg_id = str(raw_id) if raw_id is not None else "unknown"
    map_file = _map_file_for_output(params.zone_template, output_path=output_path)

    robot_config: dict[str, Any] = {"type": "differential_drive", "radius": 0.3}
    if params.max_linear_speed is not None:
        robot_config["max_linear_speed"] = params.max_linear_speed
        # Angular speed is a conservative default tied to the linear limit; it
        # is a hypothesis, not a regulation-derived value.
        robot_config["max_angular_speed"] = round(2.0 * params.max_linear_speed, 3)

    simulation_config: dict[str, Any] = {
        "max_episode_steps": params.max_episode_steps,
        "ped_density": params.ped_density,
    }

    metadata: dict[str, Any] = {
        "generated_from_regulation": reg_id,
        "generation_method": GENERATION_METHOD,
        "hypothesis": True,
        "required_manual_review": True,
        "claim_boundary": (
            "scenario hypothesis compiled from a regulation excerpt; not executed evidence"
        ),
        "compiled_parameters": [copy.deepcopy(e) for e in params.extracted],
        "unmatched_clauses": list(params.unmatched_clauses),
        "compilation_warnings": list(params.warnings),
        "archetype": params.zone_template,
        "flow": "bi",
        "behavior": "regulation_compiled",
        "authoring": {
            "status": "draft",
            "source_issue": "#6054",
            "generated_by": "scripts/tools/convert_regulation_to_scenario.py",
            "benchmark_evidence": False,
            "promotion_note": (
                "Not benchmark evidence until separately reviewed, certified, "
                "and executed through the benchmark workflow."
            ),
        },
        "plausibility": {
            "status": "unverified",
            "verified_on": None,
            "verified_by": None,
            "method": None,
            "notes": (
                "Compiled hypothesis; scenario validity (does it run and behave "
                "as intended) is not assessed by the converter."
            ),
        },
    }

    # Record a clearance requirement as metadata only. The simulator does not
    # enforce a clearance constraint from this field; it is recorded so a
    # reviewer/harness can audit it.
    if params.clearance_m is not None:
        metadata["clearance_requirement_m"] = params.clearance_m
        metadata["clearance_enforcement"] = "metadata_only_not_runtime_enforced"

    # Preserve provenance hints from the input record without trusting them.
    provenance = {
        k: reg[k] for k in ("source", "issuing_body", "context", "setting", "date") if k in reg
    }
    if provenance:
        metadata["regulation_provenance"] = provenance

    return {
        "schema_version": SCENARIO_MATRIX_SCHEMA_VERSION,
        "scenarios": [
            {
                "name": f"regulation_{reg_id}",
                "map_file": map_file,
                "simulation_config": simulation_config,
                "robot_config": robot_config,
                "metadata": metadata,
                "seeds": list(DEFAULT_SEEDS),
            }
        ],
    }


def _assess_schema_validity(payload: dict[str, Any]) -> dict[str, Any]:
    """Assess schema validity by reusing scenario_schema validators.

    Reports metadata (schema_version) errors and per-item errors separately.
    Does NOT touch the filesystem or the runner; this is shape validation only.
    """
    scenarios = payload.get("scenarios", [])
    if not isinstance(scenarios, list):
        return {
            "valid": False,
            "metadata_errors": [{"error": "payload.scenarios must be a list"}],
            "item_errors": [],
        }
    metadata_errors = validate_scenario_matrix_metadata(payload)
    item_errors = validate_scenario_list(scenarios)
    return {
        "valid": not metadata_errors and not item_errors,
        "metadata_errors": metadata_errors,
        "item_errors": item_errors,
    }


def assess_validity(
    *,
    compilation: CompiledParameters,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Report compilation, schema, and scenario validity on separate axes.

    ``scenario_validity`` is always ``not_assessed``: this tool compiles and
    shape-checks only. Establishing that the scenario runs and behaves as
    intended requires execution through the benchmark runner plus human review.
    """
    compilation_errors: list[str] = []
    if not compilation.meaningful_extractions:
        compilation_errors.append(
            "no parameters extracted from excerpt (zone/density are defaults)"
        )
    compilation_valid = {
        "valid": not compilation_errors,
        "extracted_count": len(compilation.extracted),
        "meaningful_extracted_count": len(compilation.meaningful_extractions),
        "unmatched_clause_count": len(compilation.unmatched_clauses),
        "warnings": list(compilation.warnings),
        "errors": compilation_errors,
    }
    schema_valid = _assess_schema_validity(payload)
    scenario_valid = {
        "status": "not_assessed",
        "reason": (
            "Scenario validity requires execution through the benchmark runner "
            "and human review. This converter does not execute scenarios."
        ),
    }
    return {
        "compilation_validity": compilation_valid,
        "schema_validity": schema_valid,
        "scenario_validity": scenario_valid,
    }


def convert_regulation_record(
    input_path: Path,
    output_path: Path | None = None,
    *,
    verbose: bool = False,
) -> dict[str, Any] | None:
    """Compile a regulation record into a scenario hypothesis.

    Args:
        input_path: Path to the input ``regulation-record.v1`` YAML.
        output_path: Optional path for the output scenario YAML.
        verbose: Enable verbose logging.

    Returns:
        A result dict with ``payload`` (scenario config) and ``validity`` (the
        three-axis validity report), or None if input validation fails.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If input-record validation fails.
    """
    _configure_logging(verbose)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Reading regulation record: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        record = yaml.safe_load(f)

    errors = _validate_regulation_record(record)
    if errors:
        logger.error("Input validation failed:")
        for err in errors:
            logger.error(f"  - {err}")
        raise ValueError(f"Invalid regulation record: {'; '.join(errors)}")

    logger.info("Input validation passed")

    regulation = record["regulation"]
    hints = {
        k: regulation[k] for k in ("context", "setting", "max_episode_steps") if k in regulation
    }
    compilation = compile_regulation_excerpt(regulation["regulation_text"], hints=hints)
    logger.info(
        f"Compiled {len(compilation.extracted)} parameter(s); "
        f"{len(compilation.unmatched_clauses)} unmatched clause(s)"
    )

    payload = _build_scenario_payload(regulation, compilation, output_path=output_path)

    validity = assess_validity(compilation=compilation, payload=payload)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            yaml.dump(payload, f, sort_keys=False, width=100, allow_unicode=False)
        logger.info(f"Wrote scenario hypothesis to: {output_path}")

    if not validity["schema_validity"]["valid"]:
        logger.warning("Generated scenario did NOT pass schema validation; see validity report")

    return {"payload": payload, "validity": validity}


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Validity is reported on three axes: compilation (did the compiler "
            "extract parameters), schema (does the config satisfy "
            "robot_sf.scenario_matrix.v1), and scenario (not assessed here). "
            "Outputs are hypotheses pending execution and review."
        ),
    )
    parser.add_argument(
        "--record",
        type=Path,
        required=True,
        help="Path to the input regulation-record.v1 YAML file.",
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        required=False,
        help="Path for the output scenario YAML file.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the generated scenario YAML to stdout instead of writing a file.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help=(
            "Also print the three-axis validity report (compilation/schema/"
            "scenario) as YAML to stderr after the scenario output."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point.

    Exit codes distinguish input failure (2) from a successfully compiled but
    schema-invalid hypothesis (3), so callers can tell compilation/schema
    problems apart. A clean compile + valid schema exits 0; the scenario is
    still a hypothesis either way.
    """
    args = _build_parser().parse_args(argv)

    if not args.output_yaml and not args.stdout:
        print("Error: Must specify --output-yaml or --stdout", file=sys.stderr)
        return 2

    try:
        result = convert_regulation_record(
            args.record,
            args.output_yaml if not args.stdout else None,
            verbose=args.verbose,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if result is None:  # defensive; convert raises instead
        return 2

    if args.stdout:
        print(yaml.dump(result["payload"], sort_keys=False, width=100, allow_unicode=False))

    if args.report:
        print(
            yaml.dump(result["validity"], sort_keys=False, width=100, allow_unicode=False),
            file=sys.stderr,
        )

    if not result["validity"]["schema_validity"]["valid"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
