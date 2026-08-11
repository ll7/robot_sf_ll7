"""Tests for issue #6872 cross-context validity / revalidation matrix."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

MATRIX_PATH = Path("configs/benchmarks/cross_context_validity_matrix_v1.yaml")
CONTEXT_NOTE_PATH = Path("docs/context/issue_6872_cross_context_validity_matrix.md")

VALID_EVIDENCE_STATUSES = frozenset(
    {
        "covered",
        "partially_covered",
        "requires_revalidation",
        "not_evidenced",
        "unavailable",
    }
)
VALID_PROTOCOL_PORTABILITY = frozenset(
    {
        "portable",
        "requires_adaptation",
        "not_portable",
        "unknown",
    }
)
VALID_RESULT_PORTABILITY = frozenset(
    {
        "portable",
        "not_portable",
        "unknown",
    }
)
REQUIRED_AXES = frozenset(
    {
        "site_topology",
        "social_cultural",
        "robot_embodiment",
        "observation_perception",
    }
)

_REPOSITORY_LOCAL_PATH_RE = re.compile(
    r"^(?:(?:configs|maps|benchmarks|robot_sf|scripts|tests|docs)/\S+)$"
)
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_matrix() -> dict:
    with open(MATRIX_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_context_note() -> str:
    return CONTEXT_NOTE_PATH.read_text(encoding="utf-8")


# --- Schema validation ---


class TestMatrixSchema:
    """Validate YAML matrix structure and required fields."""

    def test_matrix_file_exists(self) -> None:
        assert MATRIX_PATH.exists(), f"Matrix file not found: {MATRIX_PATH}"

    def test_schema_version_is_correct(self) -> None:
        matrix = _load_matrix()
        assert matrix.get("schema_version") == "cross_context_validity_matrix.v1"

    def test_issue_field_is_6872(self) -> None:
        matrix = _load_matrix()
        assert matrix.get("issue") == 6872

    def test_claim_boundary_exists(self) -> None:
        matrix = _load_matrix()
        boundary = matrix.get("claim_boundary", "")
        assert isinstance(boundary, str) and len(boundary) > 50

    def test_axes_section_has_all_four_axes(self) -> None:
        matrix = _load_matrix()
        axes = matrix.get("axes", {})
        assert isinstance(axes, dict)
        for axis_key in REQUIRED_AXES:
            assert axis_key in axes, f"Missing axis: {axis_key}"

    def test_each_axis_has_values(self) -> None:
        matrix = _load_matrix()
        for axis_key, axis_def in matrix.get("axes", {}).items():
            values = axis_def.get("values", [])
            assert isinstance(values, list) and len(values) > 0, f"Axis {axis_key} has no values"
            for val in values:
                assert "key" in val, f"Axis {axis_key} value missing 'key'"
                assert "display_name" in val, f"Axis {axis_key} value missing 'display_name'"
                assert "provenance" in val, f"Axis {axis_key} value missing 'provenance'"

    def test_cells_section_exists_and_is_nonempty(self) -> None:
        matrix = _load_matrix()
        cells = matrix.get("protocol_portability_matrix", {}).get("cells", [])
        assert isinstance(cells, list) and len(cells) > 0

    def test_canonical_configurations_exist(self) -> None:
        matrix = _load_matrix()
        configs = matrix.get("canonical_configurations", {})
        configs_list = configs.get("configs", []) if isinstance(configs, dict) else configs
        assert isinstance(configs_list, list) and len(configs_list) > 0

    def test_no_overclaim_boundary_exists(self) -> None:
        matrix = _load_matrix()
        boundary = matrix.get("no_overclaim_boundary", {})
        assert isinstance(boundary, dict)
        statements = boundary.get("statements", [])
        assert isinstance(statements, list) and len(statements) >= 5

    def test_status_is_draft(self) -> None:
        matrix = _load_matrix()
        assert matrix.get("status") == "draft", (
            f"Expected status 'draft' for unmerged artifact, got {matrix.get('status')!r}"
        )


# --- Cell content validation ---


class TestMatrixCells:
    """Validate cell content and fail-closed properties."""

    def _load_cells(self) -> list[dict]:
        matrix = _load_matrix()
        return matrix.get("protocol_portability_matrix", {}).get("cells", [])

    def test_all_cells_have_required_axes(self) -> None:
        cells = self._load_cells()
        for i, cell in enumerate(cells):
            axes = cell.get("axes", {})
            for axis_key in REQUIRED_AXES:
                assert axis_key in axes, f"Cell {i} missing axis {axis_key}"

    def test_all_cells_have_evidence_status(self) -> None:
        cells = self._load_cells()
        for i, cell in enumerate(cells):
            status = cell.get("evidence_status")
            assert status in VALID_EVIDENCE_STATUSES, (
                f"Cell {i} has invalid evidence_status: {status}"
            )

    def test_all_cells_have_protocol_portability(self) -> None:
        cells = self._load_cells()
        for i, cell in enumerate(cells):
            pp = cell.get("protocol_portability")
            assert pp in VALID_PROTOCOL_PORTABILITY, (
                f"Cell {i} has invalid protocol_portability: {pp}"
            )

    def test_all_cells_have_result_portability(self) -> None:
        cells = self._load_cells()
        for i, cell in enumerate(cells):
            rp = cell.get("result_portability")
            assert rp in VALID_RESULT_PORTABILITY, f"Cell {i} has invalid result_portability: {rp}"

    def test_all_cells_have_provenance(self) -> None:
        cells = self._load_cells()
        for i, cell in enumerate(cells):
            provenance = cell.get("provenance")
            assert isinstance(provenance, dict), f"Cell {i} missing or invalid provenance"

    def test_all_cells_have_revalidation_action(self) -> None:
        cells = self._load_cells()
        for i, cell in enumerate(cells):
            action = cell.get("revalidation_action")
            assert isinstance(action, str) and len(action) > 10, (
                f"Cell {i} missing or trivial revalidation_action"
            )

    def test_fail_closed_unsupported_cells_are_not_covered(self) -> None:
        """Cells with not_evidenced or requires_revalidation must never
        have evidence_status of covered."""
        cells = self._load_cells()
        for i, cell in enumerate(cells):
            status = cell.get("evidence_status")
            assert status != "covered" or isinstance(cell.get("provenance"), dict), (
                f"Cell {i} is covered but lacks provenance"
            )


# --- Fail-closed property ---


class TestFailClosed:
    """Verify the matrix fails closed against overclaiming."""

    def test_all_result_portability_is_not_portable_or_unknown(self) -> None:
        """No cell asserts result portability without explicit evidence."""
        cells = _load_matrix().get("protocol_portability_matrix", {}).get("cells", [])
        for i, cell in enumerate(cells):
            rp = cell.get("result_portability")
            assert rp in {"not_portable", "unknown"}, (
                f"Cell {i} asserts result_portability={rp}; "
                "this matrix does not assert cross-context result transfer"
            )

    def test_no_claim_boundary_statements_present(self) -> None:
        matrix = _load_matrix()
        statements = matrix.get("no_overclaim_boundary", {}).get("statements", [])
        assert "not benchmark evidence" in " ".join(statements).lower()

    def test_claim_boundary_mentions_protocol_vs_result(self) -> None:
        matrix = _load_matrix()
        boundary = matrix.get("claim_boundary", "").lower()
        assert "protocol" in boundary
        assert "result" in boundary or "empirical" in boundary


# --- Canonical configurations ---


class TestCanonicalConfigurations:
    """Validate canonical configuration inventory."""

    def _load_configs(self) -> list[dict]:
        matrix = _load_matrix()
        return matrix.get("canonical_configurations", {}).get("configs", [])

    def test_each_config_has_key_and_display_name(self) -> None:
        configs = self._load_configs()
        for i, cfg in enumerate(configs):
            assert "key" in cfg, f"Config {i} missing 'key'"
            assert "display_name" in cfg, f"Config {i} missing 'display_name'"

    def test_each_config_has_evidence_status(self) -> None:
        configs = self._load_configs()
        for i, cfg in enumerate(configs):
            status = cfg.get("evidence_status")
            assert status in VALID_EVIDENCE_STATUSES, (
                f"Config {i} has invalid evidence_status: {status}"
            )

    def test_each_config_has_coverage_notes(self) -> None:
        configs = self._load_configs()
        for i, cfg in enumerate(configs):
            notes = cfg.get("coverage_notes", "")
            assert isinstance(notes, str) and len(notes) > 20, (
                f"Config {i} missing or trivial coverage_notes"
            )


# --- Related issues ---


class TestRelatedIssues:
    """Verify linked issues are present."""

    def test_related_issues_include_3207(self) -> None:
        matrix = _load_matrix()
        issues = matrix.get("related_issues", [])
        issue_nums = [iss.get("issue") for iss in issues]
        assert 3207 in issue_nums

    def test_related_issues_include_6472(self) -> None:
        matrix = _load_matrix()
        issues = matrix.get("related_issues", [])
        issue_nums = [iss.get("issue") for iss in issues]
        assert 6472 in issue_nums

    def test_related_issues_include_6473(self) -> None:
        matrix = _load_matrix()
        issues = matrix.get("related_issues", [])
        issue_nums = [iss.get("issue") for iss in issues]
        assert 6473 in issue_nums


# --- Context note validation ---


class TestContextNote:
    """Validate the human-facing markdown document."""

    def test_context_note_exists(self) -> None:
        assert CONTEXT_NOTE_PATH.exists()

    def test_context_note_references_issue_6872(self) -> None:
        content = _load_context_note()
        assert "6872" in content

    def test_context_note_references_issue_3207(self) -> None:
        content = _load_context_note()
        assert "3207" in content

    def test_context_note_references_issue_6472(self) -> None:
        content = _load_context_note()
        assert "6472" in content

    def test_context_note_references_issue_6473(self) -> None:
        content = _load_context_note()
        assert "6473" in content

    def test_context_note_has_no_overclaim_boundary(self) -> None:
        content = _load_context_note()
        assert "no-overclaim" in content.lower() or "overclaim" in content.lower()

    def test_context_note_status_is_draft_candidate(self) -> None:
        content = _load_context_note()
        assert "Status: draft/candidate" in content, (
            "Context note status should be 'draft/candidate' for unmerged artifact"
        )

    def test_context_note_has_validation_section(self) -> None:
        content = _load_context_note()
        assert "pytest" in content
        assert "ruff" in content

    def test_context_note_mentions_protocol_portability(self) -> None:
        content = _load_context_note()
        assert "protocol portability" in content.lower()

    def test_context_note_mentions_result_portability(self) -> None:
        content = _load_context_note()
        assert "result portability" in content.lower()

    def test_context_note_links_to_yaml_matrix(self) -> None:
        content = _load_context_note()
        assert "cross_context_validity_matrix_v1.yaml" in content

    def test_context_note_mentions_evidence_statuses(self) -> None:
        content = _load_context_note()
        for status in ("covered", "partially_covered", "requires_revalidation", "not_evidenced"):
            assert status in content, f"Context note missing evidence status: {status}"


# --- Deterministic provenance / config path existence ---


def _is_repository_local_path(ref: str) -> bool:
    if not _REPOSITORY_LOCAL_PATH_RE.match(ref):
        return False
    resolved = (REPOSITORY_ROOT / ref).resolve()
    return str(resolved).startswith(str(REPOSITORY_ROOT) + "/") or resolved == REPOSITORY_ROOT


def _collect_cell_provenance_paths() -> list[str]:
    """Collect all repository-local provenance references from matrix cells."""
    cells = _load_matrix().get("protocol_portability_matrix", {}).get("cells", [])
    paths: list[str] = []
    for cell in cells:
        provenance = cell.get("provenance", {})
        for ref in provenance.get("references", []):
            if isinstance(ref, str) and _is_repository_local_path(ref):
                paths.append(ref)
    return paths


def _collect_canonical_config_paths() -> list[str]:
    """Collect all config_path values from canonical_configurations."""
    matrix = _load_matrix()
    configs = matrix.get("canonical_configurations", {}).get("configs", [])
    paths: list[str] = []
    for cfg in configs:
        cp = cfg.get("config_path")
        if isinstance(cp, str) and _is_repository_local_path(cp):
            paths.append(cp)
    return paths


def _collect_axis_provenance_paths() -> list[str]:
    """Collect repository-local paths from axis value provenance (configs, maps, source)."""
    matrix = _load_matrix()
    axes = matrix.get("axes", {})
    paths: list[str] = []
    for axis_def in axes.values():
        for val in axis_def.get("values", []):
            prov = val.get("provenance", {})
            for cfg in prov.get("configs", []):
                if isinstance(cfg, str) and _is_repository_local_path(cfg):
                    paths.append(cfg)
            src = prov.get("source")
            if isinstance(src, str) and _is_repository_local_path(src):
                paths.append(src)
    return paths


class TestProvenancePathsExist:
    """Verify every repository-local provenance/config path exists on disk.

    URLs, issue-only references, and non-path strings are skipped.
    """

    def test_cell_provenance_references_exist(self) -> None:
        paths = _collect_cell_provenance_paths()
        assert paths, "No cell provenance paths found to validate"
        missing = [p for p in paths if not Path(p).exists()]
        assert not missing, f"Cell provenance paths missing on disk: {missing}"

    def test_canonical_config_paths_exist(self) -> None:
        paths = _collect_canonical_config_paths()
        assert paths, "No canonical config paths found to validate"
        missing = [p for p in paths if not Path(p).exists()]
        assert not missing, f"Canonical config paths missing on disk: {missing}"

    def test_axis_provenance_paths_exist(self) -> None:
        paths = _collect_axis_provenance_paths()
        assert paths, "No axis provenance paths found to validate"
        missing = [p for p in paths if not Path(p).exists()]
        assert not missing, f"Axis provenance paths missing on disk: {missing}"

    def test_no_urls_in_repository_local_references(self) -> None:
        """Ensure no URL strings are accidentally treated as local paths."""
        all_refs: list[str] = []
        all_refs.extend(_collect_cell_provenance_paths())
        all_refs.extend(_collect_canonical_config_paths())
        all_refs.extend(_collect_axis_provenance_paths())
        urls = [r for r in all_refs if r.startswith("http://") or r.startswith("https://")]
        assert not urls, f"URLs found in local-path references: {urls}"

    def test_traversal_references_are_rejected(self) -> None:
        """Path traversal attempts must not be classified as repository-local."""
        traversal_refs = [
            "configs/../../../etc/passwd",
            "configs/../../etc/shadow",
            "docs/../../../etc/passwd",
            "maps/../../../etc/passwd",
        ]
        for ref in traversal_refs:
            assert not _is_repository_local_path(ref), (
                f"Traversal reference was accepted as local: {ref}"
            )


class TestEverySiteTopologyInMatrix:
    """Every defined site_topology value should appear in at least one cell."""

    def test_all_site_topology_values_covered_by_cells(self) -> None:
        matrix = _load_matrix()
        topo_values = {
            v["key"] for v in matrix.get("axes", {}).get("site_topology", {}).get("values", [])
        }
        cells = matrix.get("protocol_portability_matrix", {}).get("cells", [])
        covered = {
            cell["axes"]["site_topology"]
            for cell in cells
            if "site_topology" in cell.get("axes", {})
        }
        missing = topo_values - covered
        assert not missing, f"site_topology values without any matrix cell: {missing}"


class TestCellAxisMembership:
    """Every cell axis value must belong to the corresponding declared axis value set."""

    def test_all_cell_axis_values_are_declared(self) -> None:
        matrix = _load_matrix()
        axis_value_sets: dict[str, set[str]] = {}
        for axis_key, axis_def in matrix.get("axes", {}).items():
            axis_value_sets[axis_key] = {v["key"] for v in axis_def.get("values", [])}
        cells = matrix.get("protocol_portability_matrix", {}).get("cells", [])
        violations: list[str] = []
        for i, cell in enumerate(cells):
            for axis_key, cell_val in cell.get("axes", {}).items():
                allowed = axis_value_sets.get(axis_key)
                if allowed is None:
                    violations.append(f"Cell {i}: axis {axis_key!r} is not a declared axis")
                elif cell_val not in allowed:
                    violations.append(
                        f"Cell {i}: axis {axis_key} value {cell_val!r} "
                        f"not in declared values {sorted(allowed)}"
                    )
        assert not violations, "Cell axis membership violations:\n" + "\n".join(violations)


class TestCellProvenanceReferences:
    """Every cell provenance block must contain a non-empty references list."""

    def test_all_cells_have_nonempty_references(self) -> None:
        cells = _load_matrix().get("protocol_portability_matrix", {}).get("cells", [])
        violations: list[str] = []
        for i, cell in enumerate(cells):
            prov = cell.get("provenance", {})
            refs = prov.get("references")
            axes = cell.get("axes", {})
            cell_label = (
                f"Cell {i} ({axes.get('site_topology', '?')}/{axes.get('social_cultural', '?')})"
            )
            if not isinstance(refs, list) or len(refs) == 0:
                violations.append(f"{cell_label}: references is empty or missing")
            else:
                for j, ref in enumerate(refs):
                    if not isinstance(ref, str) or not ref.strip():
                        violations.append(
                            f"{cell_label}: references[{j}] is not a non-empty string"
                        )
        assert not violations, "Provenance reference violations:\n" + "\n".join(violations)
