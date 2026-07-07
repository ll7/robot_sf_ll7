"""Tests for the Release 0.0.2 Reproduction Note."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPRO_NOTE = ROOT / "docs" / "benchmark_release_0_0_2_reproduction.md"
DOCS_README = ROOT / "docs" / "README.md"
REPRODUCIBILITY_GUIDE = ROOT / "docs" / "benchmark_release_reproducibility.md"
SUMMARY_NOTE = (
    ROOT
    / "docs"
    / "experiments"
    / "publication"
    / "20260414_benchmark_release_0_0_2"
    / "summary.md"
)


def test_release_0_0_2_reproduction_doc_content() -> None:
    """Verify that the reproduction doc contains all necessary instructions and variables."""
    assert REPRO_NOTE.exists()
    text = REPRO_NOTE.read_text(encoding="utf-8")

    # Verify scoped manifest path is present
    assert (
        "configs/benchmarks/releases/paper_experiment_matrix_7planners_v1_release_v0_0_2_scoped.yaml"
        in text
    )

    # Verify ROBOT_SF_PAPER_HANDOFF_BUNDLE environment variable is present
    assert "ROBOT_SF_PAPER_HANDOFF_BUNDLE" in text

    # Verify canonical handoff parity test command is present
    assert (
        "tests/benchmark/test_paper_results_handoff.py::test_canonical_handoff_matches_durable_release_campaign_table"
        in text
    )

    # Verify 0.0.2 release archive name is present
    assert (
        "paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz"
        in text
    )

    # Verify expected SHA-256 checksum is present
    assert "64e8510ab7ba934103c709907f66a783c7b3dd2dd58aa4bd725e762da2734d90" in text

    # Verify explanation of annotated tag and campaign commit difference is present
    assert "cbeaca617" in text
    assert "f7ebdcae2375d085e925213197a75a386e26a79c" in text


def test_release_0_0_2_reproduction_doc_is_linked() -> None:
    """Verify that the reproduction doc is correctly referenced in READMEs and snapshot summaries."""
    note_name = "benchmark_release_0_0_2_reproduction.md"

    assert note_name in DOCS_README.read_text(encoding="utf-8")
    assert note_name in REPRODUCIBILITY_GUIDE.read_text(encoding="utf-8")
    assert note_name in SUMMARY_NOTE.read_text(encoding="utf-8")
