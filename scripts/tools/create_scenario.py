#!/usr/bin/env python3
"""Create deterministic draft scenario YAML from safe authoring templates."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.tools.scenario_authoring import (
    add_common_seed_argument,
    available_templates,
    build_scenario_payload,
    configure_authoring_tool_logging,
    parse_seed_args,
    validate_scenario_file,
    write_scenario_yaml,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the scenario creation CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        choices=available_templates(),
        default="bottleneck",
        help="Draft scenario template to materialize.",
    )
    parser.add_argument("--name", required=True, help="Stable scenario name to write.")
    parser.add_argument("--output", type=Path, required=True, help="YAML file to create.")
    parser.add_argument(
        "--source-issue",
        default="#1891",
        help="Issue or provenance label to store in metadata.authoring.source_issue.",
    )
    parser.add_argument(
        "--density",
        choices=("low", "med", "high"),
        default="med",
        help="Scenario-generation density level.",
    )
    parser.add_argument(
        "--flow",
        choices=("uni", "bi", "cross", "merge"),
        default="uni",
        help="Scenario-generation flow mode.",
    )
    parser.add_argument(
        "--obstacle",
        choices=("open", "bottleneck", "maze"),
        default="open",
        help="Scenario-generation obstacle profile.",
    )
    parser.add_argument(
        "--groups",
        type=float,
        default=0.0,
        help="Scenario-generation grouped-agent fraction [0.0,1.0].",
    )
    parser.add_argument(
        "--speed-var",
        choices=("low", "high"),
        default="low",
        help="Scenario-generation speed variation setting.",
    )
    parser.add_argument(
        "--goal-topology",
        choices=("point", "swap", "circulate"),
        default="point",
        help="Scenario-generation goal topology.",
    )
    parser.add_argument(
        "--robot-context",
        choices=("ahead", "behind", "embedded"),
        default="embedded",
        help="Scenario-generation robot context.",
    )
    parser.add_argument(
        "--sidewalk-width",
        type=float,
        default=4.0,
        help="Parameterized template sidewalk width in meters.",
    )
    parser.add_argument(
        "--obstacle-density",
        type=float,
        default=0.0,
        help="Parameterized template obstacle density [0.0,1.0].",
    )
    parser.add_argument(
        "--pedestrian-density",
        type=float,
        default=0.06,
        help="Parameterized template pedestrian density.",
    )
    parser.add_argument(
        "--bottleneck-width",
        type=float,
        default=2.0,
        help="Parameterized template bottleneck width in meters.",
    )
    parser.add_argument(
        "--crossing-angle",
        type=float,
        default=90.0,
        help="Parameterized template crossing angle in degrees.",
    )
    parser.add_argument(
        "--occlusion-probability",
        type=float,
        default=0.0,
        help="Parameterized template occlusion probability [0.0,1.0].",
    )
    add_common_seed_argument(parser)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output path if it already exists.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Write the draft without running the local authoring validator.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show loader and map-parser logs during validation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the draft scenario generator."""

    args = _build_parser().parse_args(argv)
    configure_authoring_tool_logging(verbose=args.verbose)
    payload = build_scenario_payload(
        template=args.template,
        name=args.name,
        seeds=parse_seed_args(args.seeds),
        source_issue=args.source_issue,
        generation_profile={
            "density": args.density,
            "flow": args.flow,
            "obstacle": args.obstacle,
            "groups": args.groups,
            "speed_var": args.speed_var,
            "goal_topology": args.goal_topology,
            "robot_context": args.robot_context,
        },
        parameterized_profile={
            "sidewalk_width": args.sidewalk_width,
            "obstacle_density": args.obstacle_density,
            "pedestrian_density": args.pedestrian_density,
            "bottleneck_width": args.bottleneck_width,
            "crossing_angle": args.crossing_angle,
            "occlusion_probability": args.occlusion_probability,
        },
    )
    write_scenario_yaml(args.output, payload, overwrite=args.overwrite)
    if args.skip_validation:
        print(f"Wrote draft scenario: {args.output}")
        return 0

    report = validate_scenario_file(args.output)
    if report.ok:
        print(f"Wrote and validated {report.scenario_count} draft scenario(s): {args.output}")
        return 0

    print(f"Wrote draft scenario but validation found {len(report.issues)} issue(s): {args.output}")
    for issue in report.issues:
        print(f"- {issue.format()}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
