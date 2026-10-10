"""The real SVG diagnostic identifies early endpoint proximity on loops."""

import json
from pathlib import Path

from scripts.validation.check_scenario_archetype_geometry import main


def test_svg_loop_diagnostic_reports_only_earlier_segments(tmp_path, capsys):
    svg = Path("tests/fixtures/test_maps/simple_corridor.svg").read_text()
    svg = svg.replace(
        "</svg>", '<path inkscape:label="ped_route_0_0" d="M 1 7 L 8 7 L 8 8 L 1 7" /></svg>'
    )
    path = tmp_path / "loop.svg"
    path.write_text(svg)
    assert main(["--map", str(path), "--ped-route-completion"]) == 0
    hits = json.loads(capsys.readouterr().out)[0]["findings"]
    assert [(h["route"], h["segment"]) for h in hits] == [(0, 0)]
    assert hits[0]["distance_m"] == 0
