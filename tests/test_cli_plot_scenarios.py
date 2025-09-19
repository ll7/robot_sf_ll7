import json
from pathlib import Path


def test_cli_plot_scenarios_png_and_pdf(tmp_path: Path, capsys):
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
        ]
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
