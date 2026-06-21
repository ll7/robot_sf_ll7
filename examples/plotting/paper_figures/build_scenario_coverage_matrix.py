#!/usr/bin/env python3
"""Build a manuscript-facing AMV scenario coverage matrix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

TIER_ORDER = ["low", "medium", "high"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-csv", type=Path, required=True, help="Scenario inventory CSV input")
    parser.add_argument("--out-csv", type=Path, required=True, help="Coverage matrix CSV output")
    parser.add_argument(
        "--out-md", type=Path, required=True, help="Coverage matrix Markdown output"
    )
    parser.add_argument(
        "--out-tex", type=Path, required=True, help="Coverage matrix LaTeX tabular output"
    )
    return parser.parse_args()


def normalize_interaction(flow: str) -> str:
    flow = (flow or "").strip().lower()
    mapping = {
        "bi": "bidirectional",
        "uni": "unidirectional",
        "converging": "converging",
        "cross": "crossing",
        "parallel": "parallel",
        "perpendicular": "perpendicular",
        "crowd": "crowd",
        "enter": "entering",
        "exit": "exiting",
        "circular": "circular",
        "n/a": "unspecified",
        "unknown": "unspecified",
    }
    return mapping.get(flow, flow or "unspecified")


def format_ped_density(value: str) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return value or "n/a"


def build_rows(input_csv: Path) -> list[dict[str, str | int]]:
    buckets: dict[str, dict[str, object]] = {}
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            archetype = row["archetype"].strip()
            entry = buckets.setdefault(
                archetype,
                {
                    "interaction_classes": set(),
                    "density_tiers": set(),
                    "extra_tier_variants": set(),
                    "ped_densities": set(),
                    "total": 0,
                },
            )
            interaction_classes = entry["interaction_classes"]
            density_tiers = entry["density_tiers"]
            extra_tier_variants = entry["extra_tier_variants"]
            ped_densities = entry["ped_densities"]
            assert isinstance(interaction_classes, set)
            assert isinstance(density_tiers, set)
            assert isinstance(extra_tier_variants, set)
            assert isinstance(ped_densities, set)

            interaction_classes.add(normalize_interaction(row["flow"]))

            density = row["density"].strip().lower()
            if density in TIER_ORDER:
                density_tiers.add(density)
                scenario_id = row["scenario_id"].strip()
                canonical_id = f"classic_{archetype}_{density}"
                if scenario_id != canonical_id:
                    extra_label = scenario_id.removeprefix("classic_").replace("_", " ")
                    extra_label = extra_label.replace("realworld", "real-world")
                    extra_tier_variants.add(extra_label)
            else:
                ped_densities.add(format_ped_density(row["ped_density"]))

            entry["total"] = int(entry["total"]) + 1

    rows: list[dict[str, str | int]] = []
    for archetype in sorted(buckets):
        entry = buckets[archetype]
        interaction_classes = sorted(entry["interaction_classes"])
        density_tiers = entry["density_tiers"]
        extra_tier_variants = entry["extra_tier_variants"]
        ped_densities = entry["ped_densities"]
        assert isinstance(density_tiers, set)
        assert isinstance(extra_tier_variants, set)
        assert isinstance(ped_densities, set)

        if density_tiers:
            section = "Classic density-tier archetypes"
            density_coverage = ", ".join(tier for tier in TIER_ORDER if tier in density_tiers)
            if extra_tier_variants:
                extra_label = ", ".join(sorted(extra_tier_variants))
                if int(entry["total"]) == len(extra_tier_variants):
                    density_coverage = extra_label
                else:
                    density_coverage = f"{density_coverage}; plus {extra_label}"
        else:
            section = "Francis singleton motifs"
            if len(ped_densities) == 1:
                density_coverage = f"singleton (ped_density={next(iter(ped_densities))})"
            else:
                density_coverage = "singleton / mixed numeric density"

        rows.append(
            {
                "section": section,
                "archetype": archetype,
                "interaction_class": ", ".join(interaction_classes),
                "density_coverage": density_coverage,
                "total": int(entry["total"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["section", "archetype", "interaction_class", "density_coverage", "total"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Scenario Coverage Matrix", ""]
    for section in ("Classic density-tier archetypes", "Francis singleton motifs"):
        lines.extend(
            [
                f"## {section}",
                "",
                "| Archetype | Interaction Class | Density Coverage | Total |",
                "|---|---|---|---:|",
            ]
        )
        for row in rows:
            if row["section"] != section:
                continue
            lines.append(
                f"| `{row['archetype']}` | `{row['interaction_class']}` | `{row['density_coverage']}` | {row['total']} |"
            )
        subtotal = sum(int(row["total"]) for row in rows if row["section"] == section)
        lines.append(f"| **Subtotal** |  |  | **{subtotal}** |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tex(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabularx}{\\linewidth}{@{}llXr@{}}",
        "\\toprule",
        "Archetype & Interaction class & Density coverage & Total \\\\",
        "\\midrule",
    ]
    for idx, section in enumerate(("Classic density-tier archetypes", "Francis singleton motifs")):
        if idx:
            lines.append("\\addlinespace")
        lines.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textbf{{{section}}}}} \\\\")
        for row in rows:
            if row["section"] != section:
                continue
            archetype = str(row["archetype"]).replace("_", "\\_")
            interaction = str(row["interaction_class"]).replace("_", "\\_")
            density_coverage = str(row["density_coverage"]).replace("_", "\\_")
            lines.append(f"{archetype} & {interaction} & {density_coverage} & {row['total']} \\\\")
        subtotal = sum(int(row["total"]) for row in rows if row["section"] == section)
        lines.append(f"\\textbf{{Subtotal}} &  &  & \\textbf{{{subtotal}}} \\\\")
    total = sum(int(row["total"]) for row in rows)
    lines.extend(["\\addlinespace", f"\\textbf{{Total}} &  &  & \\textbf{{{total}}} \\\\"])
    lines.extend(["\\bottomrule", "\\end{tabularx}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows(args.in_csv)
    write_csv(args.out_csv, rows)
    write_md(args.out_md, rows)
    write_tex(args.out_tex, rows)
    print(f"Wrote {len(rows)} archetype rows to {args.out_csv}")
    print(f"Wrote markdown matrix to {args.out_md}")
    print(f"Wrote latex tabular to {args.out_tex}")


if __name__ == "__main__":
    main()
