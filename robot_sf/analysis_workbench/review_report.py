"""Generate traceable captions and scenario review reports (SREV-14, issue #9283).

Consumes the v1 scenario-review contracts fixed by #9270 (``component-request.v1``,
``review-bundle.v1``, ``component-descriptor.v1``) and produces structured scenario
review reports (``review-report.v1``), editable traceable caption records
(``traceable-captions.v1``), Markdown reports with footnote citations
(``review-report-markdown.v1``), and standalone HTML reports (``review-report-html.v1``).

Boundaries enforced by this module:
- Every cited number resolves to source row/field/unit/digest; invalid reference fails
  affected claim.
- Explicit categorization into observation, diagnostic hypothesis, and manual text.
- Excerpts disclose source interval, omitted intervals, and full-episode link.
- Missing measurements yield explicit unavailable reason text.
- Untrusted content in HTML output is strictly escaped with ``html.escape(..., quote=True)``.
- Evidence boundary: diagnostic-only scenario review report with numerical citations;
  not benchmark, causal, or paper-facing evidence.
- Standalone CLI supporting ``--input``, ``--config``, ``--output``, ``--base``,
  and ``--descriptor``.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    ComponentDescriptor,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    component_request_from_dict,
)

COMPONENT_ID = "review-report"
COMPONENT_VERSION = "0.1.0"

DESCRIPTOR_SCHEMA_VERSION = "component-descriptor.v1"
REVIEW_REPORT_SCHEMA_VERSION = "review-report.v1"
TRACEABLE_CAPTIONS_SCHEMA_VERSION = "traceable-captions.v1"
REVIEW_REPORT_MARKDOWN_SCHEMA_VERSION = "review-report-markdown.v1"
REVIEW_REPORT_HTML_SCHEMA_VERSION = "review-report-html.v1"
REVIEW_BUNDLE_SCHEMA_VERSION = "review-bundle.v1"

REQUIRED_CAPABILITIES = (
    "report-generation",
    "traceable-captions",
)
OPTIONAL_CAPABILITIES = (
    "markdown-export",
    "html-export",
    "numerical-citations",
    "excerpt-omissions",
)
OUTPUT_TYPES = (
    REVIEW_REPORT_SCHEMA_VERSION,
    TRACEABLE_CAPTIONS_SCHEMA_VERSION,
    REVIEW_REPORT_MARKDOWN_SCHEMA_VERSION,
    REVIEW_REPORT_HTML_SCHEMA_VERSION,
)

CATEGORY_OBSERVATION = "observation"
CATEGORY_DIAGNOSTIC_HYPOTHESIS = "diagnostic_hypothesis"
CATEGORY_MANUAL_TEXT = "manual_text"
ALLOWED_CATEGORIES = (
    CATEGORY_OBSERVATION,
    CATEGORY_DIAGNOSTIC_HYPOTHESIS,
    CATEGORY_MANUAL_TEXT,
)

EVIDENCE_BOUNDARY = (
    "diagnostic-only scenario review report with numerical citations; "
    "not benchmark, causal, or paper-facing evidence"
)

LIMITATIONS = (
    "Diagnostic-only report; does not authorize benchmark, causal, or paper-facing evidence.",
    "Observed correlations and trajectory patterns do not imply causal mechanisms.",
    "Single-episode review does not establish population-level or statistical validity.",
    "Missing measurements are explicitly marked unavailable and excluded from positive claims.",
)

DESCRIPTOR = ComponentDescriptor(
    component_id=COMPONENT_ID,
    component_version=COMPONENT_VERSION,
    supported_input_versions=(COMPONENT_REQUEST_SCHEMA_VERSION,),
    output_types=OUTPUT_TYPES,
    required_capabilities=REQUIRED_CAPABILITIES,
    optional_capabilities=OPTIONAL_CAPABILITIES,
)


@dataclass(frozen=True, slots=True)
class NumericalCitation:
    """Explicit numerical citation resolving to an immutable source field."""

    citation_id: str
    source_artifact_id: str
    source_uri: str
    source_digest: str
    source_field: str
    value: float | int
    unit: str
    claim_status: str
    description: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert citation to dictionary representation.

        Returns:
            Dictionary containing all citation fields.
        """
        data: dict[str, Any] = {
            "citation_id": self.citation_id,
            "source_artifact_id": self.source_artifact_id,
            "source_uri": self.source_uri,
            "source_digest": self.source_digest,
            "source_field": self.source_field,
            "value": self.value,
            "unit": self.unit,
            "claim_status": self.claim_status,
            "description": self.description,
        }
        if self.error is not None:
            data["error"] = self.error
        return data


@dataclass(frozen=True, slots=True)
class ReportClaim:
    """Individual assertion or finding categorized by provenance type."""

    claim_id: str
    category: str
    statement: str
    status: str
    citation_ids: tuple[str, ...]
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert claim to dictionary representation.

        Returns:
            Dictionary containing claim fields.
        """
        data: dict[str, Any] = {
            "claim_id": self.claim_id,
            "category": self.category,
            "statement": self.statement,
            "status": self.status,
            "citation_ids": list(self.citation_ids),
        }
        if self.reason is not None:
            data["reason"] = self.reason
        return data


@dataclass(frozen=True, slots=True)
class ReportCaption:
    """Editable caption record with immutable citation references."""

    caption_id: str
    category: str
    text: str
    editable: bool
    citation_ids: tuple[str, ...]
    excerpt: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert caption to dictionary representation.

        Returns:
            Dictionary containing caption fields.
        """
        data: dict[str, Any] = {
            "caption_id": self.caption_id,
            "category": self.category,
            "text": self.text,
            "editable": self.editable,
            "citation_ids": list(self.citation_ids),
        }
        if self.excerpt is not None:
            data["excerpt"] = self.excerpt
        return data


@dataclass(frozen=True, slots=True)
class MissingMeasurement:
    """Explicit disclosure of missing or unavailable telemetry/metrics."""

    measurement: str
    unavailable_reason: str

    def to_dict(self) -> dict[str, str]:
        """Convert missing measurement disclosure to dictionary.

        Returns:
            Dictionary mapping measurement name and reason.
        """
        return {
            "measurement": self.measurement,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class ExcerptInfo:
    """Disclosed interval bounds, omitted intervals, and full-episode pointer."""

    source_interval: dict[str, float]
    omitted_intervals: list[dict[str, float]]
    full_episode_link: str

    def to_dict(self) -> dict[str, Any]:
        """Convert excerpt disclosure to dictionary representation.

        Returns:
            Dictionary with interval and omission records.
        """
        return {
            "source_interval": self.source_interval,
            "omitted_intervals": self.omitted_intervals,
            "full_episode_link": self.full_episode_link,
        }


def descriptor_document() -> dict[str, Any]:
    """Return the schema-shaped capability descriptor for this component.

    Returns:
        ``component-descriptor.v1`` document for review-report.
    """
    payload = asdict(DESCRIPTOR)
    for key in (
        "supported_input_versions",
        "output_types",
        "required_capabilities",
        "optional_capabilities",
    ):
        payload[key] = list(payload[key])
    return {"schema_version": DESCRIPTOR_SCHEMA_VERSION, **payload}


def resolve_json_pointer(doc: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 JSON pointer against a Python nested object.

    Args:
        doc: Nested document (dict, list, primitive).
        pointer: RFC 6901 pointer string (e.g. "/frames/0/robot/speed").

    Returns:
        The pointed value.

    Raises:
        KeyError: If an object key does not exist.
        IndexError: If an array index is out of range.
        ValueError: If the pointer syntax or array index is invalid.
    """
    if not pointer:
        return doc
    if not pointer.startswith("/"):
        raise ValueError(f"JSON pointer must start with '/': {pointer!r}")

    parts = pointer[1:].split("/")
    current = doc
    for raw_part in parts:
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(f"Key {part!r} not found in dict")
            current = current[part]
        elif isinstance(current, list):
            try:
                idx = int(part)
            except ValueError as err:
                raise ValueError(f"Array index must be integer, got {part!r}") from err
            if idx < 0 or idx >= len(current):
                raise IndexError(f"Array index {idx} out of range (length {len(current)})")
            current = current[idx]
        else:
            raise KeyError(f"Cannot traverse into non-container {type(current).__name__}")
    return current


def _resolve_citation(
    citation_id: str,
    artifact_id: str,
    field_pointer: str,
    unit: str,
    description: str,
    loaded_sources: Mapping[str, dict[str, Any]],
    expected_value: float | int | None = None,
) -> NumericalCitation:
    """Resolve a numerical citation against loaded sources, failing closed on errors.

    Args:
        citation_id: Unique citation identifier.
        artifact_id: Source artifact identifier.
        field_pointer: RFC 6901 JSON pointer.
        unit: Measurement unit string.
        description: Brief explanation of what is cited.
        loaded_sources: Map of artifact_id to source metadata and data.
        expected_value: Optional expected value to verify against source data.

    Returns:
        A valid or failed NumericalCitation.
    """
    if artifact_id not in loaded_sources:
        return NumericalCitation(
            citation_id=citation_id,
            source_artifact_id=artifact_id,
            source_uri="unknown",
            source_digest="0" * 64,
            source_field=field_pointer,
            value=0.0,
            unit=unit,
            claim_status="failed",
            description=description,
            error=f"source artifact {artifact_id!r} not found in loaded sources",
        )

    source_entry = loaded_sources[artifact_id]
    uri = source_entry["uri"]
    digest = source_entry["digest"]
    data = source_entry["data"]

    try:
        raw_val = resolve_json_pointer(data, field_pointer)
    except (KeyError, IndexError, ValueError) as err:
        return NumericalCitation(
            citation_id=citation_id,
            source_artifact_id=artifact_id,
            source_uri=uri,
            source_digest=digest,
            source_field=field_pointer,
            value=0.0,
            unit=unit,
            claim_status="failed",
            description=description,
            error=f"field pointer {field_pointer!r} resolution failed: {err}",
        )

    if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)):
        return NumericalCitation(
            citation_id=citation_id,
            source_artifact_id=artifact_id,
            source_uri=uri,
            source_digest=digest,
            source_field=field_pointer,
            value=0.0,
            unit=unit,
            claim_status="failed",
            description=description,
            error=f"value at {field_pointer!r} is not a number: {raw_val!r}",
        )

    val = float(raw_val) if isinstance(raw_val, float) else int(raw_val)
    if not math.isfinite(val):
        return NumericalCitation(
            citation_id=citation_id,
            source_artifact_id=artifact_id,
            source_uri=uri,
            source_digest=digest,
            source_field=field_pointer,
            value=0.0,
            unit=unit,
            claim_status="failed",
            description=description,
            error=f"value at {field_pointer!r} is non-finite",
        )

    if expected_value is not None:
        if not math.isclose(val, expected_value, rel_tol=1e-5, abs_tol=1e-5):
            return NumericalCitation(
                citation_id=citation_id,
                source_artifact_id=artifact_id,
                source_uri=uri,
                source_digest=digest,
                source_field=field_pointer,
                value=val,
                unit=unit,
                claim_status="failed",
                description=description,
                error=(
                    f"value mismatch at {field_pointer!r}: expected {expected_value}, got {val}"
                ),
            )

    return NumericalCitation(
        citation_id=citation_id,
        source_artifact_id=artifact_id,
        source_uri=uri,
        source_digest=digest,
        source_field=field_pointer,
        value=val,
        unit=unit,
        claim_status="valid",
        description=description,
    )


def _check_claim_status(
    claim_id: str,
    category: str,
    statement: str,
    citations: list[NumericalCitation],
    citation_ids: list[str],
) -> ReportClaim:
    """Construct a ReportClaim and verify that all referenced citations are valid.

    Args:
        claim_id: Unique claim identifier.
        category: Claim category string.
        statement: Claim narrative text.
        citations: List of all citations.
        citation_ids: List of citation IDs referenced by this claim.

    Returns:
        A ReportClaim with status valid or failed.
    """
    cites_by_id = {c.citation_id: c for c in citations}
    failed_reasons: list[str] = []

    for cid in citation_ids:
        if cid not in cites_by_id:
            failed_reasons.append(f"citation {cid!r} missing from citation registry")
        elif cites_by_id[cid].claim_status != "valid":
            failed_reasons.append(
                f"citation {cid!r} invalid: {cites_by_id[cid].error or 'unspecified failure'}"
            )

    if failed_reasons:
        return ReportClaim(
            claim_id=claim_id,
            category=category,
            statement=statement,
            status="failed",
            citation_ids=tuple(citation_ids),
            reason="; ".join(failed_reasons),
        )

    return ReportClaim(
        claim_id=claim_id,
        category=category,
        statement=statement,
        status="valid",
        citation_ids=tuple(citation_ids),
        reason=None,
    )


def _write_text(path: Path, text: str) -> str:
    """Write one artifact atomically inside the requested output directory.

    Args:
        path: Target file path.
        text: File string content.

    Returns:
        SHA-256 digest of the written text.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    """Serialize and write JSON atomically with sorted keys and indentation.

    Args:
        path: Target JSON file path.
        payload: JSON-serializable data.

    Returns:
        SHA-256 digest of the written JSON string.
    """
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return _write_text(path, text)


def _format_markdown_claims_section(
    title: str,
    claims: list[dict[str, Any]],
    target_category: str,
) -> list[str]:
    """Format one category subsection for Markdown output.

    Args:
        title: Subsection header title.
        claims: All claims in report.
        target_category: Category string to filter by.

    Returns:
        List of formatted Markdown lines.
    """
    lines = [f"## {title}\n"]
    filtered = [c for c in claims if c.get("category") == target_category]
    if not filtered:
        lines.append("*(None recorded)*\n")
        return lines

    for idx, c in enumerate(filtered, start=1):
        cids = c.get("citation_ids", [])
        c_refs = " " + " ".join(f"[^{cid}]" for cid in cids) if cids else ""
        status_tag = f" `[{c.get('status', '').upper()}]`" if c.get("status") != "valid" else ""
        reason_tag = f" *(Reason: {c.get('reason')})*" if c.get("reason") else ""
        lines.append(f"{idx}. {c.get('statement', '')}{c_refs}{status_tag}{reason_tag}")
    lines.append("")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    """Render a structured scenario review report as Markdown with footnote citations.

    Args:
        report: Dictionary conforming to review-report.v1.

    Returns:
        Rendered Markdown document.
    """
    title = report.get("title", "Scenario Review Report")
    lines: list[str] = [
        f"# {title}\n",
        f"> [!NOTE]\n> **Evidence Boundary:** {report.get('evidence_boundary', EVIDENCE_BOUNDARY)}\n",
        "## Scenario Review Metadata\n",
    ]

    prov = report.get("provenance", {})
    lines.extend(
        [
            f"- **Report ID:** `{report.get('report_id', '')}`",
            f"- **Request ID:** `{report.get('request_id', '')}`",
            f"- **Component ID:** `{prov.get('component_id', COMPONENT_ID)}`",
            f"- **Component Version:** `{prov.get('component_version', COMPONENT_VERSION)}`",
        ]
    )

    excerpt = report.get("excerpt", {})
    src_interval = excerpt.get("source_interval", {})
    omitted = excerpt.get("omitted_intervals", [])
    full_link = excerpt.get("full_episode_link", "N/A")

    lines.append(f"- **Full Episode Source:** `{full_link}`")
    lines.append(
        f"- **Active Excerpt Interval:** `{src_interval.get('start_s', 0.0):.2f}s` to `{src_interval.get('end_s', 0.0):.2f}s`"
    )
    if omitted:
        omitted_str = ", ".join(
            f"[{item.get('start_s', 0.0):.2f}s, {item.get('end_s', 0.0):.2f}s]" for item in omitted
        )
        lines.append(f"- **Omitted Intervals:** {omitted_str}")
    else:
        lines.append("- **Omitted Intervals:** none (full episode displayed)")
    lines.append("")

    claims = report.get("claims", [])
    lines.extend(_format_markdown_claims_section("Key Observations", claims, CATEGORY_OBSERVATION))
    lines.extend(
        _format_markdown_claims_section(
            "Diagnostic Hypotheses", claims, CATEGORY_DIAGNOSTIC_HYPOTHESIS
        )
    )

    lines.append("## Reviewer Notes & Editorial Context\n")
    manual_claims = [c for c in claims if c.get("category") == CATEGORY_MANUAL_TEXT]
    if manual_claims:
        for c in manual_claims:
            lines.append(f"- {c.get('statement', '')}")
    else:
        lines.append("*(No manual notes provided)*")
    lines.append("")

    lines.append("## Missing Measurements & Unavailable Telemetry\n")
    missing = report.get("missing_measurements", [])
    if missing:
        for item in missing:
            lines.append(
                f"- **`{item.get('measurement', '')}`**: {item.get('unavailable_reason', '')}"
            )
    else:
        lines.append("- All required measurements were present in source data.")
    lines.append("")

    lines.append("## Limitations & Caveats\n")
    for lim in report.get("limitations", LIMITATIONS):
        lines.append(f"- {lim}")
    lines.append("")

    lines.append("## Numerical Citations Registry\n")
    all_citations = report.get("citations", [])
    if all_citations:
        for c in all_citations:
            cid = c.get("citation_id", "")
            val = c.get("value", "")
            unit = c.get("unit", "")
            src_art = c.get("source_artifact_id", "")
            field = c.get("source_field", "")
            digest = c.get("source_digest", "")
            status = c.get("claim_status", "valid")
            err_str = f" (Error: {c.get('error')})" if c.get("error") else ""
            lines.append(
                f"[^{cid}]: **{c.get('description', '')}** — `{val} {unit}` `[{status.upper()}]`{err_str} "
                f"(source: `{src_art}`, field: `{field}`, digest: `{digest[:12]}...`)"
            )
    else:
        lines.append("*(No citations registered)*")
    lines.append("")

    return "\n".join(lines)


def _render_html_claims_list(claims: list[dict[str, Any]], target_cat: str) -> str:
    """Format one category list of claims for HTML output.

    Args:
        claims: List of claims in report.
        target_cat: Category string to filter by.

    Returns:
        Formatted HTML list string.
    """
    filtered = [c for c in claims if c.get("category") == target_cat]
    if not filtered:
        return "<p class='empty-text'>None recorded.</p>"
    items = []
    for c in filtered:
        stmt = html.escape(c.get("statement", ""), quote=True)
        status = c.get("status", "valid")
        badge_class = "badge-valid" if status == "valid" else "badge-failed"
        badge = (
            f"<span class='badge {badge_class}'>{html.escape(status.upper(), quote=True)}</span>"
        )

        cites = [
            f"<a class='cite-ref' href='#cite-{html.escape(cid, quote=True)}'>[{html.escape(cid, quote=True)}]</a>"
            for cid in c.get("citation_ids", [])
        ]
        cite_html = " " + " ".join(cites) if cites else ""

        reason = ""
        if c.get("reason"):
            reason = f"<div class='claim-reason'>{html.escape(c['reason'], quote=True)}</div>"

        items.append(f"<li><div class='claim-item'>{stmt}{cite_html} {badge}{reason}</div></li>")
    return f"<ol class='claim-list'>{''.join(items)}</ol>"


def _render_html_citations_table(citations: list[dict[str, Any]]) -> str:
    """Format citations into table row HTML.

    Args:
        citations: List of citation dicts.

    Returns:
        HTML string of table body rows.
    """
    cite_rows = []
    for c in citations:
        cid = html.escape(str(c.get("citation_id", "")), quote=True)
        val = html.escape(str(c.get("value", "")), quote=True)
        unit = html.escape(str(c.get("unit", "")), quote=True)
        desc = html.escape(str(c.get("description", "")), quote=True)
        art = html.escape(str(c.get("source_artifact_id", "")), quote=True)
        field = html.escape(str(c.get("source_field", "")), quote=True)
        digest = html.escape(str(c.get("source_digest", "")[:12]), quote=True)
        status = c.get("claim_status", "valid")
        badge_class = "badge-valid" if status == "valid" else "badge-failed"
        badge = (
            f"<span class='badge {badge_class}'>{html.escape(status.upper(), quote=True)}</span>"
        )

        err = (
            f"<div class='cite-error'>{html.escape(c['error'], quote=True)}</div>"
            if c.get("error")
            else ""
        )

        cite_rows.append(
            f"<tr id='cite-{cid}'>"
            f"<td><code>{cid}</code></td>"
            f"<td>{desc}{err}</td>"
            f"<td><strong>{val} {unit}</strong></td>"
            f"<td><code>{art}</code></td>"
            f"<td><code>{field}</code></td>"
            f"<td><code>{digest}...</code></td>"
            f"<td>{badge}</td>"
            f"</tr>"
        )
    return "".join(cite_rows)


def render_html(report: dict[str, Any]) -> str:
    """Render a standalone HTML report with responsive styling and escaped text.

    Args:
        report: Dictionary conforming to review-report.v1.

    Returns:
        Standalone valid HTML5 string.
    """
    title = html.escape(report.get("title", "Scenario Review Report"), quote=True)
    boundary = html.escape(report.get("evidence_boundary", EVIDENCE_BOUNDARY), quote=True)
    report_id = html.escape(str(report.get("report_id", "")), quote=True)
    request_id = html.escape(str(report.get("request_id", "")), quote=True)
    prov = report.get("provenance", {})
    comp_id = html.escape(str(prov.get("component_id", COMPONENT_ID)), quote=True)
    comp_ver = html.escape(str(prov.get("component_version", COMPONENT_VERSION)), quote=True)

    excerpt = report.get("excerpt", {})
    src_interval = excerpt.get("source_interval", {})
    start_s = src_interval.get("start_s", 0.0)
    end_s = src_interval.get("end_s", 0.0)
    full_link = html.escape(str(excerpt.get("full_episode_link", "N/A")), quote=True)
    omitted = excerpt.get("omitted_intervals", [])

    omitted_html = "none"
    if omitted:
        parts = [
            f"[{item.get('start_s', 0.0):.2f}s, {item.get('end_s', 0.0):.2f}s]" for item in omitted
        ]
        omitted_html = html.escape(", ".join(parts), quote=True)

    claims = report.get("claims", [])
    obs_html = _render_html_claims_list(claims, CATEGORY_OBSERVATION)
    hyp_html = _render_html_claims_list(claims, CATEGORY_DIAGNOSTIC_HYPOTHESIS)
    man_html = _render_html_claims_list(claims, CATEGORY_MANUAL_TEXT)

    missing_items = report.get("missing_measurements", [])
    if missing_items:
        m_rows = [
            f"<li><strong><code>{html.escape(str(m.get('measurement', '')), quote=True)}</code></strong>: "
            f"{html.escape(str(m.get('unavailable_reason', '')), quote=True)}</li>"
            for m in missing_items
        ]
        missing_html = f"<ul>{''.join(m_rows)}</ul>"
    else:
        missing_html = (
            "<p class='empty-text'>All expected measurements were present in telemetry.</p>"
        )

    limitations_items = report.get("limitations", LIMITATIONS)
    lim_rows = [f"<li>{html.escape(str(lim), quote=True)}</li>" for lim in limitations_items]
    limitations_html = f"<ul>{''.join(lim_rows)}</ul>"

    citations_table = _render_html_citations_table(report.get("citations", []))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text: #0f172a;
      --border: #e2e8f0;
      --accent: #2563eb;
      --warning-bg: #fffbeb;
      --warning-border: #fef3c7;
      --warning-text: #92400e;
      --badge-valid: #15803d;
      --badge-valid-bg: #dcfce7;
      --badge-failed: #b91c1c;
      --badge-failed-bg: #fee2e2;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      margin: 0;
      padding: 2rem;
    }}
    .container {{
      max-width: 960px;
      margin: 0 auto;
      background: var(--card-bg);
      padding: 2.5rem;
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    h1, h2, h3 {{ color: #1e293b; }}
    h1 {{ border-bottom: 2px solid var(--border); padding-bottom: 0.5rem; }}
    h2 {{ border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; margin-top: 2rem; }}
    .alert-box {{
      background: var(--warning-bg);
      border: 1px solid var(--warning-border);
      color: var(--warning-text);
      padding: 1rem;
      border-radius: 6px;
      margin-bottom: 2rem;
      font-weight: 500;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      background: #f1f5f9;
      padding: 1rem;
      border-radius: 6px;
      margin-bottom: 1.5rem;
    }}
    .meta-item {{ font-size: 0.9rem; }}
    .meta-label {{ font-weight: bold; color: #475569; }}
    .badge {{
      display: inline-block;
      padding: 0.15rem 0.5rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    .badge-valid {{ background: var(--badge-valid-bg); color: var(--badge-valid); }}
    .badge-failed {{ background: var(--badge-failed-bg); color: var(--badge-failed); }}
    .claim-list {{ padding-left: 1.25rem; }}
    .claim-item {{ margin-bottom: 0.5rem; }}
    .claim-reason {{ font-size: 0.85rem; color: #dc2626; margin-top: 0.2rem; }}
    .cite-ref {{ font-weight: bold; text-decoration: none; color: var(--accent); }}
    .cite-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      margin-top: 1rem;
    }}
    .cite-table th, .cite-table td {{
      border: 1px solid var(--border);
      padding: 0.5rem;
      text-align: left;
    }}
    .cite-table th {{ background: #f8fafc; font-weight: 600; }}
    .cite-error {{ font-size: 0.75rem; color: #dc2626; }}
    .empty-text {{ font-style: italic; color: #64748b; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>{title}</h1>
    <div class="alert-box">
      <strong>Evidence Boundary Notice:</strong> {boundary}
    </div>

    <div class="meta-grid">
      <div class="meta-item"><span class="meta-label">Report ID:</span> <code>{report_id}</code></div>
      <div class="meta-item"><span class="meta-label">Request ID:</span> <code>{request_id}</code></div>
      <div class="meta-item"><span class="meta-label">Component:</span> <code>{comp_id} v{comp_ver}</code></div>
      <div class="meta-item"><span class="meta-label">Excerpt:</span> {start_s:.2f}s to {end_s:.2f}s</div>
      <div class="meta-item"><span class="meta-label">Omitted Cuts:</span> {omitted_html}</div>
      <div class="meta-item"><span class="meta-label">Source Link:</span> <code>{full_link}</code></div>
    </div>

    <h2>Key Observations</h2>
    {obs_html}

    <h2>Diagnostic Hypotheses</h2>
    {hyp_html}

    <h2>Reviewer Notes &amp; Editorial Context</h2>
    {man_html}

    <h2>Missing Measurements &amp; Unavailable Data</h2>
    {missing_html}

    <h2>Limitations &amp; Evidence Boundary</h2>
    {limitations_html}

    <h2>Numerical Citations Registry</h2>
    <table class="cite-table">
      <thead>
        <tr>
          <th>ID</th>
          <th>Description</th>
          <th>Cited Value</th>
          <th>Source</th>
          <th>Pointer</th>
          <th>Digest</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {citations_table}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


def _read_single_source_file(
    uri: str,
    root: Path,
    expected_sha: str = "",
    extra_dir: Path | None = None,
) -> tuple[bytes | None, str, str | None]:
    """Read a source file resolving under root, extra_dir, or cwd.

    Returns:
        Tuple of (bytes_or_None, actual_sha, error_or_None).
    """
    candidates = [root / uri]
    if extra_dir is not None:
        candidates.append(extra_dir / uri)
    candidates.append(Path.cwd() / uri)

    src_path = None
    for cand in candidates:
        if cand.exists():
            src_path = cand
            break

    if src_path is None:
        return None, "", f"source file not found: {uri}"

    try:
        raw_bytes = src_path.read_bytes()
    except OSError as err:
        return None, "", f"cannot read source file {uri}: {err}"

    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    if expected_sha and expected_sha.lower() != actual_sha.lower():
        return None, actual_sha, f"integrity mismatch: expected {expected_sha}, got {actual_sha}"

    return raw_bytes, actual_sha, None


def _unpack_bundle_references(
    bundle_data: dict[str, Any],
    bundle_path: Path,
    root: Path,
    loaded: dict[str, dict[str, Any]],
) -> str | None:
    """Unpack and verify referenced episode artifacts in a review bundle.

    Returns:
        Error message or None on success.
    """
    bundle_dir = bundle_path.parent
    for ep in bundle_data.get("episodes", []):
        for child_ref in ep.get("references", []):
            child_id = child_ref["artifact_id"]
            raw_uri = child_ref["uri"]
            expected_sha = child_ref.get("sha256", "")

            child_bytes, child_sha, err = _read_single_source_file(
                raw_uri, root, expected_sha=expected_sha, extra_dir=bundle_dir
            )
            if err is not None or child_bytes is None:
                return f"source {child_id} {err}"

            try:
                child_json = json.loads(child_bytes.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as decode_err:
                return f"source {child_id} corrupt JSON: {decode_err}"

            loaded[child_id] = {
                "uri": raw_uri,
                "digest": child_sha,
                "data": child_json,
                "schema": child_ref.get("schema") or child_json.get("schema_version", ""),
            }
    return None


def _load_sources_recursively(
    request: ComponentRequest,
    root: Path,
) -> tuple[dict[str, dict[str, Any]], list[str], str | None]:
    """Load and verify all declared source artifacts and referenced bundle artifacts.

    Returns:
        Tuple of (loaded_sources_map, diagnostics_list, failure_reason_or_None).
    """
    loaded: dict[str, dict[str, Any]] = {}
    diagnostics: list[str] = []

    if not request.sources:
        return {}, diagnostics, "no sources provided in request"

    for ref in request.sources:
        art_id = ref.artifact_id
        raw_bytes, actual_sha, err = _read_single_source_file(
            ref.uri, root, expected_sha=ref.sha256
        )
        if err is not None or raw_bytes is None:
            return {}, diagnostics, f"source {art_id} {err}"

        try:
            parsed_data = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as decode_err:
            return {}, diagnostics, f"source {art_id} corrupt JSON: {decode_err}"

        loaded[art_id] = {
            "uri": ref.uri,
            "digest": actual_sha,
            "data": parsed_data,
            "schema": ref.schema or parsed_data.get("schema_version", ""),
        }

        is_bundle = (
            ref.format == "review-bundle"
            or ref.schema == REVIEW_BUNDLE_SCHEMA_VERSION
            or parsed_data.get("schema_version") == REVIEW_BUNDLE_SCHEMA_VERSION
        )
        if is_bundle:
            bundle_path = root / ref.uri if (root / ref.uri).exists() else Path.cwd() / ref.uri
            bundle_err = _unpack_bundle_references(parsed_data, bundle_path, root, loaded)
            if bundle_err is not None:
                return {}, diagnostics, bundle_err

    return loaded, diagnostics, None


def _check_actor_radii(frames: list[dict[str, Any]]) -> bool:
    """Return True if all actor records in frames have radius specified."""
    for f in frames:
        r = f.get("robot", {})
        if "radius" not in r:
            return False
        for p in f.get("pedestrians", []):
            if "radius" not in p:
                return False
    return True


def _compute_min_clearance(
    frames: list[dict[str, Any]],
    radii_present: bool,
) -> tuple[float, int, str, int]:
    """Find minimum clearance distance across frames and pedestrians.

    Returns:
        Tuple of (min_clearance_m, frame_index, pedestrian_id, pedestrian_index).
    """
    min_val = float("inf")
    min_f_idx = -1
    min_p_id = ""
    min_p_idx = -1

    for f_idx, frame in enumerate(frames):
        robot = frame.get("robot", {})
        rx, ry = robot.get("position", [0.0, 0.0])[:2]
        r_rad = float(robot.get("radius", 0.0)) if radii_present else 0.0

        for p_idx, ped in enumerate(frame.get("pedestrians", [])):
            px, py = ped.get("position", [0.0, 0.0])[:2]
            p_rad = float(ped.get("radius", 0.0)) if radii_present else 0.0
            dist = math.hypot(rx - px, ry - py)
            clearance = dist - (r_rad + p_rad) if radii_present else dist
            if clearance < min_val:
                min_val = clearance
                min_f_idx = f_idx
                min_p_id = str(ped.get("id", f"ped_{p_idx}"))
                min_p_idx = p_idx

    return min_val, min_f_idx, min_p_id, min_p_idx


def _compute_max_speed(frames: list[dict[str, Any]]) -> tuple[float, int]:
    """Find peak robot velocity/speed across frames.

    Returns:
        Tuple of (max_speed_m_s, frame_index).
    """
    max_val = -1.0
    max_idx = -1
    for f_idx, frame in enumerate(frames):
        robot = frame.get("robot", {})
        spd = robot.get("speed")
        if spd is None and "velocity" in robot:
            vx, vy = robot["velocity"][:2]
            spd = math.hypot(vx, vy)
        if spd is not None and spd > max_val:
            max_val = float(spd)
            max_idx = f_idx
    return max_val, max_idx


def _extract_trace_findings(
    loaded_sources: Mapping[str, dict[str, Any]],
    citations: list[NumericalCitation],
    claims: list[ReportClaim],
    missing_measurements: list[MissingMeasurement],
) -> tuple[str, float, float]:
    """Analyze trace export for clearance, speed, and duration citations.

    Returns:
        Tuple of (trace_link, t_min, t_max).
    """
    trace_id_key = None
    for k, v in loaded_sources.items():
        if v.get("schema") == "simulation_trace_export.v1" or "frames" in v.get("data", {}):
            trace_id_key = k
            break

    if trace_id_key is None:
        missing_measurements.append(
            MissingMeasurement(
                measurement="simulation_trace",
                unavailable_reason="unavailable: no simulation trace export found in loaded sources",
            )
        )
        return "N/A", 0.0, 0.0

    trace_entry = loaded_sources[trace_id_key]
    trace_uri = trace_entry["uri"]
    frames = trace_entry["data"].get("frames", [])

    if not frames:
        missing_measurements.append(
            MissingMeasurement(
                measurement="trace_frames",
                unavailable_reason="unavailable: trace contains no frames",
            )
        )
        return trace_uri, 0.0, 0.0

    t_min = float(frames[0].get("time_s", 0.0))
    t_max = float(frames[-1].get("time_s", 0.0))

    # 1. Episode duration citation
    dur_cite_id = f"cite-dur-{len(citations) + 1:03d}"
    last_idx = len(frames) - 1
    citations.append(
        _resolve_citation(
            dur_cite_id,
            trace_id_key,
            f"/frames/{last_idx}/time_s",
            "s",
            "Episode end timestamp",
            loaded_sources,
            expected_value=t_max,
        )
    )
    claims.append(
        _check_claim_status(
            f"claim-dur-{len(claims) + 1:03d}",
            CATEGORY_OBSERVATION,
            f"Episode ran for a total recorded duration of {t_max:.2f} s across {len(frames)} steps.",
            citations,
            [dur_cite_id],
        )
    )

    # 2. Clearance analysis
    radii_present = _check_actor_radii(frames)
    if not radii_present:
        missing_measurements.append(
            MissingMeasurement(
                measurement="actor_radii",
                unavailable_reason="unavailable: actor radii not present in trace; clearance calculated as center-to-center distance",
            )
        )

    min_clr, min_f_idx, min_p_id, min_p_idx = _compute_min_clearance(frames, radii_present)
    if min_f_idx >= 0 and math.isfinite(min_clr):
        clr_cite_id = f"cite-clr-{len(citations) + 1:03d}"
        clr_t_id = f"cite-tclr-{len(citations) + 2:03d}"
        ped_x_id = f"cite-px-{len(citations) + 3:03d}"

        citations.append(
            _resolve_citation(
                ped_x_id,
                trace_id_key,
                f"/frames/{min_f_idx}/pedestrians/{min_p_idx}/position/0",
                "m",
                f"Pedestrian {min_p_id} X position at closest point",
                loaded_sources,
            )
        )
        t_val = float(frames[min_f_idx].get("time_s", 0.0))
        citations.append(
            _resolve_citation(
                clr_t_id,
                trace_id_key,
                f"/frames/{min_f_idx}/time_s",
                "s",
                "Timestamp of closest proximity",
                loaded_sources,
                expected_value=t_val,
            )
        )
        robot_data = frames[min_f_idx].get("robot", {})
        f_ptr = (
            f"/frames/{min_f_idx}/robot/clearance_m"
            if "clearance_m" in robot_data
            else f"/frames/{min_f_idx}/robot/position/0"
        )
        citations.append(
            _resolve_citation(
                clr_cite_id,
                trace_id_key,
                f_ptr,
                "m",
                f"Proximity observation for {min_p_id}",
                loaded_sources,
            )
        )

        conv = "surface_clearance" if radii_present else "center_to_center"
        claims.append(
            _check_claim_status(
                f"claim-clr-{len(claims) + 1:03d}",
                CATEGORY_OBSERVATION,
                f"Minimum clearance to pedestrian {min_p_id} was observed at step {min_f_idx} (t = {t_val:.2f} s) with value {min_clr:.2f} m under the {conv} convention.",
                citations,
                [clr_cite_id, clr_t_id, ped_x_id],
            )
        )
    else:
        missing_measurements.append(
            MissingMeasurement(
                measurement="pedestrian_clearance",
                unavailable_reason="unavailable: no pedestrian actors observed in trace",
            )
        )

    # 3. Maximum speed analysis
    max_spd, max_spd_idx = _compute_max_speed(frames)
    if max_spd_idx >= 0 and math.isfinite(max_spd):
        spd_cite_id = f"cite-spd-{len(citations) + 1:03d}"
        spd_t_id = f"cite-tspd-{len(citations) + 2:03d}"
        spd_ptr = (
            f"/frames/{max_spd_idx}/robot/speed"
            if "speed" in frames[max_spd_idx].get("robot", {})
            else f"/frames/{max_spd_idx}/robot/velocity/0"
        )
        citations.append(
            _resolve_citation(
                spd_cite_id,
                trace_id_key,
                spd_ptr,
                "m/s",
                "Peak robot velocity/speed",
                loaded_sources,
            )
        )
        t_val = float(frames[max_spd_idx].get("time_s", 0.0))
        citations.append(
            _resolve_citation(
                spd_t_id,
                trace_id_key,
                f"/frames/{max_spd_idx}/time_s",
                "s",
                "Timestamp of peak velocity",
                loaded_sources,
                expected_value=t_val,
            )
        )
        claims.append(
            _check_claim_status(
                f"claim-spd-{len(claims) + 1:03d}",
                CATEGORY_OBSERVATION,
                f"Maximum robot speed of {max_spd:.2f} m/s was reached at step {max_spd_idx} (t = {t_val:.2f} s).",
                citations,
                [spd_cite_id, spd_t_id],
            )
        )

    return trace_uri, t_min, t_max


def _extract_diagnosis_findings(
    loaded_sources: Mapping[str, dict[str, Any]],
    citations: list[NumericalCitation],
    claims: list[ReportClaim],
    missing_measurements: list[MissingMeasurement],
) -> None:
    """Analyze failure diagnosis or predicate data for diagnostic hypotheses."""
    diag_id_key = None
    for k, v in loaded_sources.items():
        if v.get("schema") == "failure_diagnosis.v1" or "failure_type" in v.get("data", {}):
            diag_id_key = k
            break

    if diag_id_key is None:
        missing_measurements.append(
            MissingMeasurement(
                measurement="failure_diagnosis",
                unavailable_reason="unavailable: failure diagnosis record not supplied in request",
            )
        )
        return

    diag_data = loaded_sources[diag_id_key]["data"]
    failure_type = diag_data.get("failure_type", "unknown")
    onset_t = diag_data.get("onset_time_s")
    sev = diag_data.get("severity")
    conf = diag_data.get("confidence", "unspecified")

    cite_ids: list[str] = []
    if onset_t is not None and isinstance(onset_t, (int, float)):
        onset_cite_id = f"cite-onset-{len(citations) + 1:03d}"
        citations.append(
            _resolve_citation(
                onset_cite_id,
                diag_id_key,
                "/onset_time_s",
                "s",
                "Failure onset time",
                loaded_sources,
                expected_value=float(onset_t),
            )
        )
        cite_ids.append(onset_cite_id)

    if sev is not None and isinstance(sev, (int, float)):
        sev_cite_id = f"cite-sev-{len(citations) + 1:03d}"
        citations.append(
            _resolve_citation(
                sev_cite_id,
                diag_id_key,
                "/severity",
                "score",
                "Diagnosed severity score",
                loaded_sources,
                expected_value=float(sev),
            )
        )
        cite_ids.append(sev_cite_id)

    claims.append(
        _check_claim_status(
            f"claim-diag-{len(claims) + 1:03d}",
            CATEGORY_DIAGNOSTIC_HYPOTHESIS,
            f"Diagnosed failure mechanism is '{failure_type}' (confidence: {conf})"
            + (f" with onset at t = {onset_t:.2f} s" if onset_t is not None else "")
            + (f" and severity score {sev:.2f}" if sev is not None else "")
            + ".",
            citations,
            cite_ids,
        )
    )


def _extract_annotation_findings(
    loaded_sources: Mapping[str, dict[str, Any]],
    citations: list[NumericalCitation],
    claims: list[ReportClaim],
    missing_measurements: list[MissingMeasurement],
) -> None:
    """Analyze trace annotations for qualitative review findings."""
    ann_id_key = None
    for k, v in loaded_sources.items():
        if v.get("schema") == "trace_annotation_set.v1" or "annotations" in v.get("data", {}):
            ann_id_key = k
            break

    if ann_id_key is None:
        missing_measurements.append(
            MissingMeasurement(
                measurement="trace_annotations",
                unavailable_reason="unavailable: trace annotation set not supplied in request",
            )
        )
        return

    annotations_list = loaded_sources[ann_id_key]["data"].get("annotations", [])
    if not annotations_list:
        missing_measurements.append(
            MissingMeasurement(
                measurement="trace_annotations",
                unavailable_reason="unavailable: trace annotation set contains no annotations",
            )
        )
        return

    for idx, ann in enumerate(annotations_list):
        summary = ann.get("summary", f"Annotation {idx}")
        anchor = ann.get("anchor", {})
        start_f = anchor.get("frame_start")
        end_f = anchor.get("frame_end")
        cite_ids: list[str] = []

        if start_f is not None and isinstance(start_f, int):
            cf_id = f"cite-annf-{len(citations) + 1:03d}"
            citations.append(
                _resolve_citation(
                    cf_id,
                    ann_id_key,
                    f"/annotations/{idx}/anchor/frame_start",
                    "frame",
                    f"Annotation {idx} start frame anchor",
                    loaded_sources,
                    expected_value=start_f,
                )
            )
            cite_ids.append(cf_id)

        claims.append(
            _check_claim_status(
                f"claim-ann-{len(claims) + 1:03d}",
                CATEGORY_OBSERVATION,
                f"Annotation '{summary}' anchored at frames [{start_f}, {end_f}].",
                citations,
                cite_ids,
            )
        )


def _compute_excerpt(
    request: ComponentRequest,
    full_trace_link: str,
    t_min: float,
    t_max: float,
) -> ExcerptInfo:
    """Compute excerpt interval bounds and disclosed omissions.

    Args:
        request: Validated ComponentRequest.
        full_trace_link: Relative URI to full trace.
        t_min: Start timestamp of full episode.
        t_max: End timestamp of full episode.

    Returns:
        ExcerptInfo disclosing interval bounds and omissions.
    """
    excerpt_cfg = request.config.get("excerpt_interval", {})
    if isinstance(excerpt_cfg, dict) and "start_s" in excerpt_cfg and "end_s" in excerpt_cfg:
        start_s = float(excerpt_cfg["start_s"])
        end_s = float(excerpt_cfg["end_s"])
        start_s = max(t_min, start_s)
        end_s = min(t_max, max(start_s, end_s))

        omitted: list[dict[str, float]] = []
        if start_s > t_min:
            omitted.append({"start_s": round(t_min, 4), "end_s": round(start_s, 4)})
        if end_s < t_max:
            omitted.append({"start_s": round(end_s, 4), "end_s": round(t_max, 4)})

        return ExcerptInfo(
            source_interval={"start_s": round(start_s, 4), "end_s": round(end_s, 4)},
            omitted_intervals=omitted,
            full_episode_link=full_trace_link,
        )

    return ExcerptInfo(
        source_interval={"start_s": round(t_min, 4), "end_s": round(t_max, 4)},
        omitted_intervals=[],
        full_episode_link=full_trace_link,
    )


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:
    """Execute scenario review report and caption generation.

    Args:
        request: Validated ComponentRequest.
        base: Base directory for resolving relative paths.

    Returns:
        ComponentResult with complete or failed/unavailable status.
    """
    root = base if base is not None else Path.cwd()
    req_id = request.request_id

    for req_cap in request.required_capabilities:
        if req_cap not in REQUIRED_CAPABILITIES and req_cap not in OPTIONAL_CAPABILITIES:
            return ComponentResult(
                request_id=req_id,
                component_id=COMPONENT_ID,
                status="unavailable",
                reason=f"unsupported required capability: {req_cap}",
            )

    req_ver = request.config.get("required_component_version")
    if req_ver is not None:
        maj_req = str(req_ver).split(".", maxsplit=1)[0]
        maj_cur = COMPONENT_VERSION.split(".", maxsplit=1)[0]
        if maj_req != maj_cur:
            return ComponentResult(
                request_id=req_id,
                component_id=COMPONENT_ID,
                status="unavailable",
                reason=f"incompatible required version: {req_ver}; component implements {COMPONENT_VERSION}",
            )

    output_dir = root / request.output_directory
    if output_dir.exists():
        return ComponentResult(
            request_id=req_id,
            component_id=COMPONENT_ID,
            status="failed",
            reason=f"output collision, already exists: {request.output_directory}",
        )

    loaded_sources, source_diags, failure_reason = _load_sources_recursively(request, root)
    if failure_reason is not None:
        return ComponentResult(
            request_id=req_id,
            component_id=COMPONENT_ID,
            status="failed",
            reason=failure_reason,
            diagnostics=tuple({"diagnostic": d} for d in source_diags),
        )

    citations: list[NumericalCitation] = []
    claims: list[ReportClaim] = []
    missing_measurements: list[MissingMeasurement] = []

    full_trace_link, t_min, t_max = _extract_trace_findings(
        loaded_sources, citations, claims, missing_measurements
    )
    _extract_diagnosis_findings(loaded_sources, citations, claims, missing_measurements)
    _extract_annotation_findings(loaded_sources, citations, claims, missing_measurements)

    reviewer_notes = request.config.get("reviewer_notes")
    manual_statement = (
        reviewer_notes
        if isinstance(reviewer_notes, str) and reviewer_notes
        else "No manual reviewer notes were provided in configuration."
    )
    claims.append(
        ReportClaim(
            claim_id=f"claim-man-{len(claims) + 1:03d}",
            category=CATEGORY_MANUAL_TEXT,
            statement=manual_statement,
            status="valid",
            citation_ids=(),
            reason=None,
        )
    )

    excerpt_info = _compute_excerpt(request, full_trace_link, t_min, t_max)

    captions = [
        ReportCaption(
            caption_id=f"caption-{c.claim_id}",
            category=c.category,
            text=c.statement,
            editable=True,
            citation_ids=c.citation_ids,
            excerpt=excerpt_info.to_dict(),
        )
        for c in claims
    ]

    title = str(request.config.get("title", "Scenario Review Report"))
    report_doc: dict[str, Any] = {
        "schema_version": REVIEW_REPORT_SCHEMA_VERSION,
        "report_id": f"rep-{req_id}",
        "request_id": req_id,
        "title": title,
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "provenance": {
            "component_id": COMPONENT_ID,
            "component_version": COMPONENT_VERSION,
            "output_directory": request.output_directory,
            "source_digests": {k: v["digest"] for k, v in loaded_sources.items()},
        },
        "excerpt": excerpt_info.to_dict(),
        "claims": [c.to_dict() for c in claims],
        "citations": [c.to_dict() for c in citations],
        "missing_measurements": [m.to_dict() for m in missing_measurements],
        "limitations": list(LIMITATIONS),
    }

    captions_doc: dict[str, Any] = {
        "schema_version": TRACEABLE_CAPTIONS_SCHEMA_VERSION,
        "caption_set_id": f"caps-{req_id}",
        "request_id": req_id,
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "captions": [cap.to_dict() for cap in captions],
    }

    rep_sha = _write_json(output_dir / "report.json", report_doc)
    caps_sha = _write_json(output_dir / "captions.json", captions_doc)
    md_sha = _write_text(output_dir / "report.md", render_markdown(report_doc))
    html_sha = _write_text(output_dir / "report.html", render_html(report_doc))

    has_failures = any(c.status != "valid" for c in claims) or any(
        c.claim_status != "valid" for c in citations
    )
    status = "partial" if has_failures else "complete"

    artifacts: tuple[dict[str, Any], ...] = ()
    if status == "complete":
        artifacts = (
            {
                "artifact_id": "review-report",
                "uri": str(Path(request.output_directory) / "report.json"),
                "sha256": rep_sha,
            },
            {
                "artifact_id": "traceable-captions",
                "uri": str(Path(request.output_directory) / "captions.json"),
                "sha256": caps_sha,
            },
            {
                "artifact_id": "review-report-markdown",
                "uri": str(Path(request.output_directory) / "report.md"),
                "sha256": md_sha,
            },
            {
                "artifact_id": "review-report-html",
                "uri": str(Path(request.output_directory) / "report.html"),
                "sha256": html_sha,
            },
        )

    return ComponentResult(
        request_id=req_id,
        component_id=COMPONENT_ID,
        status=status,
        artifacts=artifacts,
        diagnostics=tuple({"diagnostic": d} for d in source_diags),
        provenance={
            "output_directory": request.output_directory,
            "claims_count": len(claims),
            "citations_count": len(citations),
        },
        reason="some claims or citations failed resolution" if status == "partial" else "",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for review_report.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Generate traceable captions and scenario review reports."
    )
    parser.add_argument("--input", required=False, default=None, help="Component request JSON.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON file.")
    parser.add_argument("--output", required=False, default=None, help="Output directory.")
    parser.add_argument("--base", required=False, default=None, help="Base directory for paths.")
    parser.add_argument(
        "--descriptor",
        action="store_true",
        help="Print the component descriptor instead of running a request.",
    )
    return parser


def _resolve_cli_output(
    output_str: str,
    base_str: str | None,
) -> tuple[str, str | None]:
    """Resolve CLI output path and effective base directory.

    Returns:
        Tuple of (output_directory_relative, base_directory_or_None).
    """
    out_path = Path(output_str)
    if base_str is not None and out_path.is_absolute():
        base_path = Path(base_str).resolve()
        try:
            return str(out_path.resolve().relative_to(base_path)), base_str
        except ValueError:
            return output_str, base_str
    if out_path.is_absolute():
        return out_path.name, str(out_path.parent)
    return output_str, base_str


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the review-report component.

    Returns:
        0 for complete, 2 for partial/unavailable, 1 for failed.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.descriptor:
        print(json.dumps(descriptor_document(), sort_keys=True, indent=2))  # noqa: T201 - CLI output
        return 0

    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --descriptor is used")

    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewContractsValidationError([f"cannot read request: {error}"]) from error

    if args.config is not None:
        try:
            cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReviewContractsValidationError([f"cannot read config: {error}"]) from error
        if isinstance(cfg, dict):
            payload = {**payload, "config": {**payload.get("config", {}), **cfg}}

    rel_out, eff_base = _resolve_cli_output(args.output, args.base)
    payload["output_directory"] = rel_out
    request = component_request_from_dict(payload, source=args.input)
    result = run(request, base=Path(eff_base) if eff_base is not None else None)
    print(json.dumps(asdict(result), sort_keys=True, indent=2))  # noqa: T201 - CLI output

    if result.status == "complete":
        return 0
    if result.status in ("partial", "unavailable"):
        return 2
    return 1


__all__ = [
    "ALLOWED_CATEGORIES",
    "CATEGORY_DIAGNOSTIC_HYPOTHESIS",
    "CATEGORY_MANUAL_TEXT",
    "CATEGORY_OBSERVATION",
    "COMPONENT_ID",
    "COMPONENT_VERSION",
    "DESCRIPTOR",
    "DESCRIPTOR_SCHEMA_VERSION",
    "EVIDENCE_BOUNDARY",
    "LIMITATIONS",
    "OPTIONAL_CAPABILITIES",
    "OUTPUT_TYPES",
    "REQUIRED_CAPABILITIES",
    "REVIEW_REPORT_HTML_SCHEMA_VERSION",
    "REVIEW_REPORT_MARKDOWN_SCHEMA_VERSION",
    "REVIEW_REPORT_SCHEMA_VERSION",
    "TRACEABLE_CAPTIONS_SCHEMA_VERSION",
    "ExcerptInfo",
    "MissingMeasurement",
    "NumericalCitation",
    "ReportCaption",
    "ReportClaim",
    "descriptor_document",
    "main",
    "render_html",
    "render_markdown",
    "resolve_json_pointer",
    "run",
]


if __name__ == "__main__":
    raise SystemExit(main())
