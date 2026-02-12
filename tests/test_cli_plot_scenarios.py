"""CLI integration tests for scenario thumbnail rendering."""

import json
from pathlib import Path


def test_cli_plot_scenarios_png_and_pdf(tmp_path: Path, capsys):
    """Ensure plot-scenarios renders expected thumbnail and montage artifacts."""
    out_dir = tmp_path / "thumbs"
    out_dir.mkdir(parents=True, exist_ok=True)
    from robot_sf.benchmark.cli import cli_main

    rc = cli_main(
        [
            "plot-scenarios",
            "--matrix",
            "configs/baselines/example_matrix.yaml",
            "--out-dir",
            str(out_dir),
            "--pdf",
            "--montage",
            "--cols",
            "2",
        ],
    )
    out = capsys.readouterr().out
    assert rc == 0, f"plot-scenarios failed: {out}"
    payload = json.loads(out)
    # Three unique scenarios in example matrix
    wrote = payload.get("wrote", [])
    assert len(wrote) >= 3
    # Validate some file existence
    for p in wrote:
        assert Path(p).exists(), f"missing thumbnail: {p}"
        pdf = Path(str(p).replace(".png", ".pdf"))
        assert pdf.exists(), f"missing pdf: {pdf}"
    # Montage
    montage = payload.get("montage", {})
    assert "png" in montage
    assert Path(montage["png"]).exists()
    if "pdf" in montage:
        assert Path(montage["pdf"]).exists()


def test_cli_plot_scenarios_name_only_matrix_avoids_overwrite(tmp_path: Path, capsys):
    """Ensure name-based matrices produce unique sanitized filenames.

    Regresses filename collisions when scenarios omit `id` and rely on `name`/`scenario_id`.
    """

    matrix = tmp_path / "francis_style_matrix.yaml"
    matrix.write_text(
        "\n".join(
            [
                '- name: "francis2023/frontal approach"',
                "  density: low",
                '- name: "francis2023:frontal approach"',
                "  density: med",
                '- scenario_id: "francis2023 frontal approach"',
                "  density: high",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "thumbs_names"
    out_dir.mkdir(parents=True, exist_ok=True)

    from robot_sf.benchmark.cli import cli_main

    rc = cli_main(
        [
            "plot-scenarios",
            "--matrix",
            str(matrix),
            "--out-dir",
            str(out_dir),
            "--base-seed",
            "123",
        ],
    )
    out = capsys.readouterr().out
    assert rc == 0, f"plot-scenarios failed: {out}"
    payload = json.loads(out)
    wrote = payload.get("wrote", [])
    assert len(wrote) == 3, "Expected all three scenarios to render without overwrite"

    basenames = [Path(path).name for path in wrote]
    assert len(set(basenames)) == 3
    assert all("/" not in name and ":" not in name and " " not in name for name in basenames)
    assert any("__2.png" in name for name in basenames), "Expected deterministic collision suffix"
