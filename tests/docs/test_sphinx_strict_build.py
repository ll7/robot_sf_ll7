"""Tests for the curated strict Sphinx build (issue #8723).

The curated strict build must fail closed on actionable warnings on curated
pages while allowing cross-references from curated pages into repository
documents that the curated site intentionally does not build.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.dev.sphinx_curated_build import (
    build_manifest,
    classify_warnings,
    compute_curated_sources,
    load_manifest,
    strict_build,
)

pytest.importorskip("sphinx")
pytest.importorskip("myst_parser")

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DOCS = REPO_ROOT / "docs"
_REAL_MANIFEST = REAL_DOCS / "sphinx_curated_sources.json"

_MINIMAL_CONF = """\
extensions = ["myst_parser"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build"]
import json, os
from pathlib import Path
if os.environ.get("ROBOT_SF_SPHINX_EXCLUSIONS_FILE"):
    exclude_patterns = json.loads(Path(os.environ["ROBOT_SF_SPHINX_EXCLUSIONS_FILE"]).read_text())
"""


def _make_project(tmp_path: Path, *, page_body: str, extra_page: str | None = None) -> Path:
    """Create a minimal curated docs project and return its docs directory."""

    docs = tmp_path / "docs"
    docs.mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fixture'\n", encoding="utf-8")
    (docs / "conf.py").write_text(_MINIMAL_CONF, encoding="utf-8")
    (docs / "index.rst").write_text(
        "Fixture site\n============\n\n.. toctree::\n\n   page\n", encoding="utf-8"
    )
    (docs / "page.md").write_text(page_body, encoding="utf-8")
    if extra_page is not None:
        (docs / "extra.md").write_text(extra_page, encoding="utf-8")
    (docs / "sphinx_curated_sources.json").write_text(
        json.dumps(build_manifest(compute_curated_sources(docs), docs), indent=2) + "\n",
        encoding="utf-8",
    )
    return docs


def test_clean_curated_project_passes_and_allows_existing_excluded_xrefs(
    tmp_path: Path,
) -> None:
    """A clean page passes, and links to existing uncurated docs are allowed."""

    docs = _make_project(
        tmp_path,
        page_body="# Page\n\nSee [extra](extra.md) and [repo config](../pyproject.toml).\n",
        extra_page="# Extra\n",
    )

    result = strict_build(docs_dir=docs, output_dir=tmp_path / "out")

    assert result.status == "pass", result.blocking_warnings
    assert result.blocking_warnings == ()
    assert result.allowed_crossref_count >= 1


def test_broken_cross_reference_fails_strict_build(tmp_path: Path) -> None:
    """A link to a nonexistent document must fail the strict build."""

    docs = _make_project(
        tmp_path,
        page_body="# Page\n\nSee [missing](does-not-exist.md).\n",
    )

    result = strict_build(docs_dir=docs, output_dir=tmp_path / "out")

    assert result.status == "failed"
    assert any("myst.xref_missing" in line for line in result.blocking_warnings)
    assert any("does-not-exist" in line for line in result.blocking_warnings)


def test_malformed_heading_fails_strict_build(tmp_path: Path) -> None:
    """A heading-level jump must surface as a blocking myst.header warning."""

    docs = _make_project(
        tmp_path,
        page_body="# Page\n\n### Skipped level\n\nBody.\n",
    )

    result = strict_build(docs_dir=docs, output_dir=tmp_path / "out")

    assert result.status == "failed"
    assert any("myst.header" in line for line in result.blocking_warnings)


def test_highlighting_failure_fails_strict_build(tmp_path: Path) -> None:
    """An unknown fence language must surface as a blocking highlighting warning."""

    docs = _make_project(
        tmp_path,
        page_body="# Page\n\n```not-a-real-lexer\ncontent\n```\n",
    )

    result = strict_build(docs_dir=docs, output_dir=tmp_path / "out")

    assert result.status == "failed"
    assert any(
        "highlighting" in line or "lexer" in line.lower() for line in result.blocking_warnings
    ), result.blocking_warnings


def test_manifest_drift_fails_closed(tmp_path: Path) -> None:
    """A stale curated manifest must fail before any build starts."""

    docs = _make_project(tmp_path, page_body="# Page\n\nBody.\n")
    manifest = docs / "sphinx_curated_sources.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "sphinx_curated_sources.v1",
                "generated_from": "fixture",
                "curated": ["index.rst", "page.md", "stale.md"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = strict_build(docs_dir=docs, output_dir=tmp_path / "out")

    assert result.status == "manifest_drift"
    assert result.returncode == 2
    assert any("stale.md" in line for line in result.warning_lines)


def test_classify_warnings_allows_only_resolvable_crossrefs(tmp_path: Path) -> None:
    """The classifier allows resolvable doc links and blocks everything else."""

    docs = _make_project(
        tmp_path,
        page_body="# Page\n\n[extra](extra.md)\n",
        extra_page="# Extra\n",
    )
    lines = (
        f"{docs / 'page.md'}:3: WARNING: Unknown source document 'extra' [myst.xref_missing]",
        f"{docs / 'page.md'}:4: WARNING: Unknown source document 'ghost' [myst.xref_missing]",
        f"{docs / 'page.md'}:5: WARNING: Non-consecutive header level [myst.header]",
    )

    blocking, allowed = classify_warnings(lines, docs)

    assert allowed == 1
    assert len(blocking) == 2
    assert any("ghost" in line for line in blocking)
    assert any("myst.header" in line for line in blocking)


def test_sphinx_subprocess_failure_cannot_report_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nonzero Sphinx return code must fail the build even without warning lines."""

    docs = _make_project(tmp_path, page_body="# Page\n\nBody.\n")
    completed = subprocess.CompletedProcess(
        args=["sphinx"], returncode=1, stdout="", stderr="crash without warnings\n"
    )
    monkeypatch.setattr(
        "scripts.dev.sphinx_curated_build.subprocess.run", lambda *args, **kwargs: completed
    )

    result = strict_build(docs_dir=docs, output_dir=tmp_path / "out")

    assert result.status == "failed"
    assert result.returncode == 1
    assert any("exit code 1" in line for line in result.blocking_warnings)


def test_cli_exits_nonzero_for_sphinx_subprocess_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main() must return nonzero in JSON and non-JSON modes when Sphinx fails."""

    from scripts.dev import sphinx_curated_build as module

    failed = module.CuratedBuildResult(
        status="failed",
        curated_count=1,
        excluded_count=0,
        returncode=1,
        output_dir=str(tmp_path / "out"),
        warning_lines=(),
        blocking_warnings=("Sphinx subprocess failed with exit code 1.",),
    )
    monkeypatch.setattr(module, "strict_build", lambda **kwargs: failed)

    json_exit = module.main(["--builder", "dummy", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert json_exit == 1
    assert payload["status"] == "failed"
    assert payload["returncode"] == 1
    assert payload["sphinx_returncode"] == 1

    assert module.main(["--builder", "dummy"]) == 1


def test_real_curated_manifest_matches_toctree_closure() -> None:
    """The checked-in curated manifest must match the current toctree closure."""

    curated = compute_curated_sources(REAL_DOCS)
    stored = load_manifest(_REAL_MANIFEST)
    assert [path.relative_to(REAL_DOCS).as_posix() for path in curated] == sorted(stored["curated"])


@pytest.mark.slow
def test_real_curated_strict_build_passes() -> None:
    """The real curated site must pass the strict build with no blocking warnings."""

    result = strict_build(docs_dir=REAL_DOCS, builder="dummy")

    assert result.status == "pass", result.blocking_warnings[:10]
    assert result.blocking_warnings == ()


def test_large_exclusion_set_reaches_sphinx_without_oversized_argument(tmp_path: Path) -> None:
    """A real Sphinx build must exclude a corpus larger than Linux's argument limit."""
    docs = _make_project(tmp_path, page_body="# Page\n\nBody.\n")
    expected = []
    for index in range(700):
        name = f"{index:04d}_" + "historical_" * 20 + ".md"
        (docs / name).write_text("# Historical\n\n### Invalid heading\n", encoding="utf-8")
        expected.append(name)
    # The old single -D value cannot be transported even by an otherwise tiny command.
    assert len(("exclude_patterns=" + ",".join(expected)).encode()) > 131072
    (docs / "capture.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "def setup(app):\n"
        "    def capture(app, config):\n"
        "        Path(app.outdir).mkdir(parents=True, exist_ok=True)\n"
        "        Path(app.outdir, 'excluded.json').write_text(json.dumps(config.exclude_patterns))\n"
        "    app.connect('config-inited', capture)\n",
        encoding="utf-8",
    )
    with (docs / "conf.py").open("a", encoding="utf-8") as conf:
        conf.write(
            "\nimport sys\nsys.path.insert(0, str(Path(__file__).parent))\nextensions.append('capture')\n"
        )
    output = tmp_path / "out"
    result = strict_build(docs_dir=docs, output_dir=output)
    assert result.status == "pass", result.blocking_warnings
    assert result.excluded_count == 700
    assert json.loads((output / "excluded.json").read_text()) == sorted(expected)
    assert (output / "page.html").is_file()
    assert {path.name for path in output.glob("*.html")} == {
        "index.html",
        "page.html",
        "search.html",
        "genindex.html",
    }
    assert not any((output / Path(name).with_suffix(".html")).exists() for name in expected)


def test_real_docs_config_receives_exact_exclusion_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real conf.py must replace defaults with the exact curated complement."""
    from sphinx.config import Config
    from sphinx.util.tags import Tags

    curated = set(compute_curated_sources(REAL_DOCS))
    expected = sorted(
        path.relative_to(REAL_DOCS).as_posix()
        for pattern in ("*.md", "*.rst")
        for path in REAL_DOCS.rglob(pattern)
        if "_build" not in path.parts and path.resolve() not in curated
    )
    payload = tmp_path / "excluded.json"
    payload.write_text(json.dumps(expected), encoding="utf-8")
    monkeypatch.setenv("ROBOT_SF_SPHINX_EXCLUSIONS_FILE", str(payload))
    config = Config.read(REAL_DOCS, overrides={}, tags=Tags())
    assert config.exclude_patterns == expected
