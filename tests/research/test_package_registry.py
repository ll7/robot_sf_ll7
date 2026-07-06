"""Tests for the research-package registry/preflight helper (issue #3057)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_sf.research.package_registry import (
    SCHEMA_VERSION,
    evaluate_registry_preflight,
    load_registry,
    render_markdown,
)

REAL_REGISTRY = Path("configs/research/research_package_registry_issue_3057.yaml")


def _write_registry(tmp_path: Path, packages: list[dict]) -> Path:
    """Write a synthetic registry document and return its path."""
    document = {
        "schema_version": SCHEMA_VERSION,
        "description": "synthetic registry for tests",
        "packages": packages,
    }
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_ready_package_when_all_artifacts_present(tmp_path: Path) -> None:
    """A package with present artifacts and no prerequisites is ready."""
    (tmp_path / "a.yaml").write_text("ok", encoding="utf-8")
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "id": "pkg_a",
                "title": "Package A",
                "issue": 1,
                "resource": "local",
                "required_artifacts": ["a.yaml"],
                "prerequisites": [],
            }
        ],
    )
    report = evaluate_registry_preflight(load_registry(registry_path), repo_root=tmp_path)

    assert report["summary"] == {
        "package_count": 1,
        "ready_count": 1,
        "blocked_count": 0,
        "gap_count": 0,
    }
    package = report["packages"][0]
    assert package["status"] == "ready"
    assert package["artifacts"]["present_count"] == 1
    assert package["gaps"] == []


def test_missing_artifact_blocks_package(tmp_path: Path) -> None:
    """A missing required artifact fails closed to blocked with a gap."""
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "id": "pkg_a",
                "title": "Package A",
                "required_artifacts": ["missing.yaml"],
                "prerequisites": [],
            }
        ],
    )
    report = evaluate_registry_preflight(load_registry(registry_path), repo_root=tmp_path)

    package = report["packages"][0]
    assert package["status"] == "blocked"
    assert package["artifacts"]["missing"] == ["missing.yaml"]
    assert report["gaps"] == [
        {
            "package_id": "pkg_a",
            "gap_type": "missing_artifact",
            "detail": "required artifact not found: missing.yaml",
        }
    ]


def test_blocked_prerequisite_propagates(tmp_path: Path) -> None:
    """A package downstream of a blocked prerequisite is itself blocked."""
    (tmp_path / "b.yaml").write_text("ok", encoding="utf-8")
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "id": "upstream",
                "title": "Upstream",
                "required_artifacts": ["missing.yaml"],  # blocked
                "prerequisites": [],
            },
            {
                "id": "downstream",
                "title": "Downstream",
                "required_artifacts": ["b.yaml"],  # present, but prerequisite is blocked
                "prerequisites": ["upstream"],
            },
        ],
    )
    report = evaluate_registry_preflight(load_registry(registry_path), repo_root=tmp_path)

    statuses = {p["package_id"]: p["status"] for p in report["packages"]}
    assert statuses == {"upstream": "blocked", "downstream": "blocked"}
    downstream = next(p for p in report["packages"] if p["package_id"] == "downstream")
    assert downstream["blocked_prerequisites"] == ["upstream"]
    assert any(g["gap_type"] == "blocked_prerequisite" for g in downstream["gaps"])


def test_missing_flow_stage_blocks_package(tmp_path: Path) -> None:
    """A missing declared vertical-flow stage fails closed."""
    (tmp_path / "manifest.md").write_text("ok", encoding="utf-8")
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "id": "release",
                "title": "Release",
                "required_artifacts": ["manifest.md"],
                "prerequisites": [],
                "flow_stages": [
                    {"id": "campaign_manifest", "artifact": "manifest.md"},
                    {"id": "claim_card", "artifact": "missing_claim_card.yaml"},
                ],
            }
        ],
    )

    report = evaluate_registry_preflight(load_registry(registry_path), repo_root=tmp_path)

    package = report["packages"][0]
    assert package["status"] == "blocked"
    assert package["flow_stages"] == [
        {"stage_id": "campaign_manifest", "artifact": "manifest.md", "present": True},
        {"stage_id": "claim_card", "artifact": "missing_claim_card.yaml", "present": False},
    ]
    assert {
        "package_id": "release",
        "gap_type": "missing_flow_stage",
        "detail": ("vertical flow stage claim_card missing artifact: missing_claim_card.yaml"),
    } in report["gaps"]


def test_malformed_flow_stage_rejected(tmp_path: Path) -> None:
    """Flow stages require stable ids and artifact paths."""
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "id": "release",
                "title": "Release",
                "required_artifacts": [],
                "prerequisites": [],
                "flow_stages": [{"id": "campaign_manifest"}],
            }
        ],
    )

    with pytest.raises(ValueError, match="flow_stages\\[0\\].*artifact"):
        load_registry(registry_path)


def test_dependency_order_independent_of_declaration_order(tmp_path: Path) -> None:
    """Prerequisites resolve transitively regardless of declared order."""
    for name in ("a.yaml", "b.yaml"):
        (tmp_path / name).write_text("ok", encoding="utf-8")
    # Declare the dependent package before its prerequisite on purpose.
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "id": "downstream",
                "title": "Downstream",
                "required_artifacts": ["b.yaml"],
                "prerequisites": ["upstream"],
            },
            {
                "id": "upstream",
                "title": "Upstream",
                "required_artifacts": ["a.yaml"],
                "prerequisites": [],
            },
        ],
    )
    report = evaluate_registry_preflight(load_registry(registry_path), repo_root=tmp_path)
    statuses = {p["package_id"]: p["status"] for p in report["packages"]}
    assert statuses == {"downstream": "ready", "upstream": "ready"}
    # Declared order is preserved in output.
    assert [p["package_id"] for p in report["packages"]] == ["downstream", "upstream"]


def test_unknown_prerequisite_rejected(tmp_path: Path) -> None:
    """A prerequisite referencing an unknown package id is rejected at load time."""
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "id": "pkg_a",
                "title": "Package A",
                "required_artifacts": [],
                "prerequisites": ["does_not_exist"],
            }
        ],
    )
    with pytest.raises(ValueError, match="unknown prerequisite"):
        load_registry(registry_path)


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    """Duplicate package ids are rejected at load time."""
    registry_path = _write_registry(
        tmp_path,
        [
            {"id": "dup", "title": "One", "required_artifacts": [], "prerequisites": []},
            {"id": "dup", "title": "Two", "required_artifacts": [], "prerequisites": []},
        ],
    )
    with pytest.raises(ValueError, match="duplicate package id"):
        load_registry(registry_path)


def test_prerequisite_cycle_rejected(tmp_path: Path) -> None:
    """A prerequisite cycle is rejected at load time."""
    registry_path = _write_registry(
        tmp_path,
        [
            {"id": "a", "title": "A", "required_artifacts": [], "prerequisites": ["b"]},
            {"id": "b", "title": "B", "required_artifacts": [], "prerequisites": ["a"]},
        ],
    )
    with pytest.raises(ValueError, match="cycle"):
        load_registry(registry_path)


def test_unsupported_schema_version_rejected(tmp_path: Path) -> None:
    """An unsupported schema version is rejected."""
    path = tmp_path / "registry.yaml"
    path.write_text(
        yaml.safe_dump({"schema_version": "bogus.v9", "packages": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unsupported registry schema_version"):
        load_registry(path)


def test_non_string_list_entry_rejected(tmp_path: Path) -> None:
    """A non-string artifact/prerequisite entry is rejected instead of coerced."""
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "id": "pkg_a",
                "title": "Package A",
                "required_artifacts": [123],  # not a string
                "prerequisites": [],
            }
        ],
    )
    with pytest.raises(ValueError, match="non-string entry"):
        load_registry(registry_path)


def test_invalid_yaml_raises_value_error(tmp_path: Path) -> None:
    """A syntactically invalid registry surfaces as the documented ValueError."""
    path = tmp_path / "registry.yaml"
    path.write_text("schema_version: [unbalanced\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid YAML"):
        load_registry(path)


def test_render_markdown_contains_summary(tmp_path: Path) -> None:
    """Markdown rendering surfaces the package table and gap section."""
    (tmp_path / "a.yaml").write_text("ok", encoding="utf-8")
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "id": "pkg_a",
                "title": "Package A",
                "issue": 7,
                "resource": "local",
                "required_artifacts": ["a.yaml"],
                "prerequisites": [],
            }
        ],
    )
    report = evaluate_registry_preflight(load_registry(registry_path), repo_root=tmp_path)
    markdown = render_markdown(report)
    assert "# Research Package Registry Preflight" in markdown
    assert "pkg_a" in markdown
    assert "#7" in markdown
    assert "1/1" in markdown


def test_real_registry_loads_and_is_consistent() -> None:
    """The shipped registry parses, and prerequisites reference declared packages."""
    registry = load_registry(REAL_REGISTRY)
    assert registry.schema_version == SCHEMA_VERSION
    known = registry.package_ids()
    assert len(known) == len(registry.packages)  # no duplicate ids
    for package in registry.packages:
        for prerequisite in package.prerequisites:
            assert prerequisite in known, f"{package.package_id} -> unknown {prerequisite}"


def test_real_registry_preflight_against_checkout() -> None:
    """The shipped registry evaluates against the real checkout without error."""
    registry = load_registry(REAL_REGISTRY)
    report = evaluate_registry_preflight(registry, repo_root=Path("."))
    assert report["summary"]["package_count"] == len(registry.packages)
    # Infrastructure-stage artifacts are tracked on main, so those packages are ready.
    statuses = {p["package_id"]: p["status"] for p in report["packages"]}
    assert statuses["scenario_suite_v0"] == "ready"
    assert statuses["package_b_adversarial"] == "ready"
    # The release gate is now ready since issue #3081 has published its release manifest.
    assert statuses["release_july_2026"] == "ready"
    release = next(p for p in report["packages"] if p["package_id"] == "release_july_2026")
    stage_ids = {stage["stage_id"] for stage in release["flow_stages"]}
    assert stage_ids == {
        "campaign_manifest",
        "scenario_seed_expansion",
        "local_or_slurm_execution_packet",
        "episode_rows_result_store",
        "social_compliance_metrics",
        "comparison_report",
        "claim_card_durable_manifest",
    }
    assert not any(gap["package_id"] == "release_july_2026" for gap in report["gaps"])
