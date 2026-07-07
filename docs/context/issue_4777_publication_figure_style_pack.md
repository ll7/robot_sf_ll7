# Issue #4777 Publication Figure Style Pack

Status: current figure-presentation guidance, July 7, 2026.

Related issue: [Issue #4777](https://github.com/ll7/robot_sf_ll7/issues/4777)
(follow-up: [Issue #4790](https://github.com/ll7/robot_sf_ll7/issues/4790)).

## Purpose

This note records the publication figure style pack: an opt-in presentation and
provenance layer for benchmark figures. It is a **non-semantic** layer — it only
changes presentation, output formats, and metadata. It does not change metric
computation, aggregation, planner ordering, or benchmark interpretation. First
slice PR #4786 added the standalone pack; the completing slice wired it into the
existing generators and hardened metadata/caption embedding.

## Modules

| Module | Role |
| --- | --- |
| `robot_sf/benchmark/figures/style.py` | `publication_style()` context manager, `planner_color`/`planner_palette`, `figure_size` presets |
| `robot_sf/benchmark/figures/provenance.py` | `build_provenance`, `write_provenance`, `build_caption_fragment` (LaTeX-escaped), `write_caption_fragment` |
| `robot_sf/benchmark/figures/export.py` | `save_publication_figure` (multi-format save + sidecar + embedded metadata) |

## Supported formats

`pdf`, `png`, `svg`. PDF is the default for publication paths. Existing
non-publication CLI paths keep their old PNG defaults unless explicitly changed.

## Palette policy

A fixed colorblind-safe planner palette (Wong 2011) is centralized in
`style.py`. Known planners map to fixed colors; unknown planners map
deterministically to a fallback palette via a stable SHA-256 hash (never
Python's salted `hash()`), so a planner keeps its color across processes and
figures.

## Size presets

`figure_size("single")` ≈ 3.4 in (single column); `figure_size("double")` ≈ 7 in
(double column). The `publication_style` context restores prior matplotlib
rcParams on exit and does not mutate global state outside the context.

## Provenance sidecar contract

Every publication figure writes a `<name>.provenance.json` sidecar containing:
source artifact paths + SHA-256 hashes, seeds, episode ids, config path/hash,
scenario matrix path/hash, repo commit, generator command, figure formats,
generated file SHA-256 hashes, and a claim boundary. The sidecar is the
canonical machine-readable record.

The same stable provenance subset (source hashes, seeds, config hash, repo
commit, generator command) is embedded **inside** PDF (`Keywords` info-dict
field) and PNG (`Provenance` text chunk) file metadata, best-effort, so the
provenance travels with the file. SVG relies on the sidecar.

## Caption fragment contract

`build_caption_fragment` produces a LaTeX-ready one-liner naming campaign +
scenario + episodes. User-supplied identifiers are LaTeX-escaped
(`_ % & # $ { } ~ ^ \`) so free-form campaign/scenario/episode strings cannot
break LaTeX compilation. `--caption-fragment` / `caption=True` writes a
`<name>.caption.tex` sidecar.

## Generator wiring (opt-in)

The pack is applied to the existing generators without changing data semantics:

- `scripts/generate_figures.py`: `--publication-style`, `--format pdf,png,svg`,
  `--caption-fragment`, `--campaign`, `--figure-size`. In publication mode the
  figure block runs inside `publication_style`; force-field and thumbnails use
  `save_publication_figure`; pareto and distribution figures get provenance
  sidecars (and PDF when `--format` includes `pdf`).
- `robot_sf/benchmark/figures/force_field.py`: `publication=True` renders inside
  `publication_style` and saves via `save_publication_figure`; CLI gains
  `--publication-style`/`--format`/`--caption-fragment`/`--campaign`.
- `robot_sf/benchmark/figures/thumbnails.py` +
  `robot_sf/benchmark/scenario_thumbnails.py`: `publication=True` renders
  thumbnails and montage in publication style with provenance sidecars; a PNG is
  always included for thumbnails so montages can composite it.
- `robot_sf/benchmark/visualization.py`: `generate_benchmark_plots(...,
  publication=True, formats=..., caption_fragment=..., size=...)` renders the
  metrics and scenario-comparison plots in publication style via
  `save_publication_figure`, threaded through the subprocess plot path.

## Non-semantic presentation boundary

This pack is presentation and provenance only. It must not be cited as evidence
of planner quality, benchmark success, or paper readiness. The `claim_boundary`
field in each provenance record marks figures as presentation/diagnostic unless
produced from a camera-ready campaign. Paper-facing claims still require the
durable release or publication-bundle proof described in
[docs/benchmark_artifact_publication.md](../benchmark_artifact_publication.md).
