"""Build a self-contained static scenario/planner gallery HTML page.

Aggregates existing tooling (no new rendering):

- :func:`robot_sf.benchmark.scenario_thumbnails.save_scenario_thumbnails`
  renders deterministic per-scenario thumbnails.
- :func:`robot_sf.benchmark.runner.load_scenario_matrix` loads the scenario
  manifest.
- :func:`robot_sf.benchmark.scenario_generator.resolve_agent_count` derives the
  pedestrian count from a scenario's ``density``.
- :func:`robot_sf.benchmark.scenario_generator.estimate_initial_difficulty`
  derives a deterministic difficulty band shown on each card.

The gallery is a discoverability/inspection artifact, not benchmark evidence.
The per-card "expected runtime" is a deterministic order-of-magnitude estimate
derived from agent count and horizon, and "supported planners" is a documented
constant set of canonical planner names.
"""

from __future__ import annotations

import base64
import html
import json
import shlex
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from robot_sf.benchmark.scenario_generator import (
    estimate_initial_difficulty,
    resolve_agent_count,
)
from robot_sf.benchmark.scenario_thumbnails import (
    resolve_scenario_label,
    sanitize_scenario_label,
    save_scenario_thumbnails,
)

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from collections.abc import Mapping, Sequence

GALLERY_HTML_SCHEMA_VERSION = "scenario_gallery_html.v1"

# A documented constant set of canonical planner families the gallery advertises
# as "supported". This is a discoverability aid, not a per-scenario measured
# capability: the canonical robot_sf benchmark runners can target any of these
# planner families on any abstract scenario. Kept intentionally short and stable
# so the gallery does not imply benchmark evidence for specific planner/scenario
# pairs. See ``robot_sf/planner/`` for the full planner registry.
SUPPORTED_PLANNERS: tuple[str, ...] = (
    "simple_policy",
    "baseline_sf",
    "dwa",
    "orca",
    "classic_global",
)

# Per-step cost is dominated by the pedestrian model. This is a coarse
# order-of-magnitude estimate for a CPU step on a modest machine and is labeled
# as an estimate on every card; it is NOT a benchmark result.
_SECONDS_PER_AGENT_STEP = 0.0006
_DEFAULT_HORIZON_STEPS = 100

_DEFAULT_OUT_DIR = Path("output/gallery")
_DEFAULT_HTML_NAME = "index.html"
_DEFAULT_MANIFEST_NAME = "gallery_manifest.json"


@dataclass
class GalleryCard:
    """One scenario card in the gallery.

    Attributes:
        scenario_id: Resolved, filesystem-safe scenario identifier.
        label: Human-readable scenario label (pre-sanitization).
        map_name: Map name shown on the card (derived from the scenario's
            ``map_id``/``map_file`` when present, otherwise the arena descriptor
            ``flow/obstacle``).
        pedestrian_count: Deterministic pedestrian count derived from density.
        supported_planners: Documented constant set of canonical planner names.
        expected_runtime_seconds: Deterministic order-of-magnitude runtime
            estimate (NOT a benchmark result).
        difficulty_band: Deterministic difficulty band from the generator.
        difficulty_score: Deterministic difficulty score from the generator.
        run_command: A copy-pasteable ``robot_sf_bench run`` command for this
            scenario.
        thumbnail_relpath: Path to the rendered thumbnail relative to the HTML
            page. ``None`` when no thumbnail was rendered.
        sample_rollout_relpath: Optional path to a sample rollout relative to the
            HTML page, when one is discoverable. ``None`` otherwise.
        params: The raw scenario parameter dictionary (echoed for transparency).
    """

    scenario_id: str
    label: str
    map_name: str
    pedestrian_count: int
    supported_planners: tuple[str, ...]
    expected_runtime_seconds: float
    difficulty_band: str
    difficulty_score: float
    run_command: str
    thumbnail_relpath: str | None
    sample_rollout_relpath: str | None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GalleryBuildResult:
    """Outcome of a gallery build.

    Attributes:
        html_path: Absolute path to the written HTML page.
        manifest_path: Absolute path to the written JSON manifest sidecar.
        cards: The gallery cards in render order.
        thumbnail_dir: Absolute path to the directory holding rendered thumbnails.
        matrix_path: Repository-root-relative path of the source scenario matrix.
        schema_version: Schema version stamp for the generated HTML/manifest.
    """

    html_path: Path
    manifest_path: Path
    cards: list[GalleryCard]
    thumbnail_dir: Path
    matrix_path: str
    schema_version: str


def resolve_supported_planners() -> tuple[str, ...]:
    """Return the documented constant set of supported planner names.

    Returns:
        Tuple of canonical planner family names advertised by the gallery.
    """
    return SUPPORTED_PLANNERS


def estimate_expected_runtime_seconds(
    pedestrian_count: int,
    *,
    horizon_steps: int = _DEFAULT_HORIZON_STEPS,
) -> float:
    """Return a deterministic order-of-magnitude CPU runtime estimate.

    The estimate is ``pedestrian_count * horizon_steps * _SECONDS_PER_AGENT_STEP``
    floored at a small constant so empty scenarios still report a nonzero cost.

    This is NOT a benchmark result; it is a coarse discoverability hint shown on
    each card and in the manifest.

    Args:
        pedestrian_count: Number of simulated pedestrians.
        horizon_steps: Number of simulation steps in the default run.

    Returns:
        Estimated wall-clock seconds for a single CPU episode.
    """
    pedestrian_count = max(pedestrian_count, 0)
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be >= 1")
    estimate = max(0.1, pedestrian_count * horizon_steps * _SECONDS_PER_AGENT_STEP)
    return round(estimate, 3)


def _resolve_map_name(scenario: Mapping[str, Any]) -> str:
    """Derive a human-readable map name from a scenario dict.

    Priority: ``map_id`` -> ``map_file`` stem -> ``flow/obstacle`` arena
    descriptor -> ``"abstract"``.

    Args:
        scenario: Scenario parameter mapping.

    Returns:
        Map name string for display.
    """
    map_id = scenario.get("map_id")
    if isinstance(map_id, str) and map_id.strip():
        return map_id.strip()
    map_file = scenario.get("map_file")
    if isinstance(map_file, str) and map_file.strip():
        return Path(map_file.strip()).stem
    flow = scenario.get("flow")
    obstacle = scenario.get("obstacle")
    bits = [str(v) for v in (flow, obstacle) if isinstance(v, str) and v.strip()]
    if bits:
        return "/".join(bits)
    return "abstract"


def _resolve_sample_rollout(
    scenario: Mapping[str, Any],
    *,
    sample_rollout_root: Path | None,
) -> Path | None:
    """Discover an optional sample rollout file for a scenario.

    Looks for ``<sample_rollout_root>/<scenario_id>.(mp4|webm|jsonl|html)``. Returns
    the first match or ``None``.

    Args:
        scenario: Scenario parameter mapping (used for the id).
        sample_rollout_root: Directory to search, or ``None``.

    Returns:
        Path to a discovered sample rollout, or ``None``.
    """
    if sample_rollout_root is None or not sample_rollout_root.is_dir():
        return None
    sid = sanitize_scenario_label(resolve_scenario_label(scenario))
    for ext in ("mp4", "webm", "jsonl", "html"):
        candidate = sample_rollout_root / f"{sid}.{ext}"
        if candidate.is_file():
            return candidate
    return None


def _build_run_command(matrix_path: str, scenario_id: str, horizon_steps: int) -> str:
    """Build a copy-pasteable ``robot_sf_bench run`` command for one scenario.

    Args:
        matrix_path: Repository-root-relative scenario matrix path.
        scenario_id: Resolved scenario identifier.
        horizon_steps: Horizon used for the runtime estimate and runner command.

    Returns:
        Shell command string selecting a single scenario from the matrix.
    """
    output_path = f"output/gallery/runs/{scenario_id}.jsonl"
    return (
        f"uv run robot_sf_bench run --matrix {shlex.quote(matrix_path)} "
        f"--out {shlex.quote(output_path)} --scenario-id {shlex.quote(scenario_id)} "
        f"--algo simple_policy --horizon {horizon_steps}"
    )


def _embed_thumbnail_base64(thumbnail_path: Path) -> str | None:
    """Return a base64 data URI for a thumbnail PNG, or ``None``.

    Args:
        thumbnail_path: Path to a PNG thumbnail.

    Returns:
        ``data:image/png;base64,...`` URI, or ``None`` if unreadable.
    """
    try:
        data = thumbnail_path.read_bytes()
    except OSError:  # pragma: no cover - filesystem boundary
        return None
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _escape(text: object) -> str:
    """HTML-escape a value for safe inline display.

    Args:
        text: Value to escape.

    Returns:
        HTML-escaped string.
    """
    return html.escape(str(text))


def _stage_sample_rollout(
    rollout_path: Path,
    *,
    out_dir_path: Path,
    scenario_id: str,
) -> str | None:
    """Copy a discovered rollout beneath the gallery and return a relative link.

    Returns:
        Relative staged path, or ``None`` when the source cannot be copied.
    """
    out_dir_abs = out_dir_path.resolve()
    source = rollout_path.resolve()
    target_dir = out_dir_abs / "rollouts"
    target = target_dir / f"{scenario_id}{source.suffix.lower()}"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        if source != target.resolve():
            shutil.copy2(source, target)
    except OSError:
        logger.exception("Could not stage sample rollout {}", source)
        return None
    return str(target.relative_to(out_dir_abs))


def _thumbnail_render_state(
    *,
    requested: bool,
    rendered: int,
    total: int,
) -> tuple[str, list[str]]:
    """Summarize thumbnail rendering as an honest manifest status.

    Returns:
        Tuple of status (``disabled``, ``rendered``, or ``degraded``) and
        human-readable caveats.
    """
    if not requested:
        return "disabled", []
    if rendered == total:
        return "rendered", []
    return (
        "degraded",
        [
            "Thumbnail rendering was incomplete; cards without a thumbnail use placeholders. "
            f"Rendered {rendered} of {total} scenarios."
        ],
    )


def _render_gallery_thumbnails(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    thumbnail_dir: Path,
    base_seed: int,
    requested: bool,
) -> tuple[dict[str, str], Path, int]:
    """Render gallery thumbnails through the existing thumbnail renderer.

    Returns:
        Tuple of scenario-id-to-relative-path mappings, absolute thumbnail
        directory, and the number of thumbnails rendered.
    """
    thumbnail_dir_abs = thumbnail_dir.resolve()
    if not requested:
        return {}, thumbnail_dir_abs, 0

    render_payloads: list[dict[str, Any]] = []
    for scenario in scenarios:
        payload = dict(scenario)
        payload["id"] = sanitize_scenario_label(resolve_scenario_label(scenario))
        render_payloads.append(payload)
    try:
        metas = save_scenario_thumbnails(
            render_payloads,
            out_dir=thumbnail_dir,
            base_seed=base_seed,
            out_pdf=False,
        )
    except (OSError, ValueError, RuntimeError):  # pragma: no cover - rendering boundary
        logger.exception("Thumbnail rendering failed; gallery cards will omit thumbnails")
        metas = []
    relative_paths = {meta.scenario_id: "thumbnails/" + Path(meta.png).name for meta in metas}
    return relative_paths, thumbnail_dir_abs, len(metas)


def _card_html(
    card: GalleryCard,
    *,
    thumbnail_dir_abs: Path,
    embed_thumbnails: bool,
) -> str:
    """Render a single gallery card as an HTML article.

    Args:
        card: The gallery card.
        thumbnail_dir_abs: Absolute path to the thumbnail directory (used when
            reading a thumbnail to embed as a data URI).
        embed_thumbnails: When True, embed the thumbnail as a base64 data URI so
            the page is fully self-contained and portable.

    Returns:
        HTML string for the card.
    """
    # Thumbnail image markup.
    img_tag = '<div class="thumb thumb-missing" title="No thumbnail rendered"></div>'
    if card.thumbnail_relpath:
        thumb_abs = (thumbnail_dir_abs / Path(card.thumbnail_relpath).name).resolve()
        if embed_thumbnails:
            data_uri = _embed_thumbnail_base64(thumb_abs)
            if data_uri:
                img_tag = f'<img class="thumb" alt="{_escape(card.label)}" src="{data_uri}" />'
        else:
            img_tag = (
                f'<img class="thumb" alt="{_escape(card.label)}" '
                f'src="{_escape(card.thumbnail_relpath)}" />'
            )

    # Optional sample-rollout link.
    rollout_html = ""
    if card.sample_rollout_relpath:
        rollout_html = (
            f'<a class="rollout-link" href="{_escape(card.sample_rollout_relpath)}" '
            'target="_blank" rel="noopener">▶ view sample rollout</a>'
        )
    else:
        rollout_html = '<span class="rollout-link muted">no sample rollout</span>'

    planners = ", ".join(card.supported_planners)
    runtime_str = f"~{card.expected_runtime_seconds:g}s (est.)"
    diff = f"{card.difficulty_band} ({card.difficulty_score:.2f})"

    params_json = _escape(json.dumps(card.params, sort_keys=True, default=str))

    return f"""
    <article class="card">
      {img_tag}
      <div class="card-body">
        <h3 class="card-title">{_escape(card.label)}</h3>
        <dl class="card-meta">
          <dt>Scenario ID</dt><dd><code>{_escape(card.scenario_id)}</code></dd>
          <dt>Map</dt><dd>{_escape(card.map_name)}</dd>
          <dt>Pedestrians</dt><dd>{_escape(card.pedestrian_count)}</dd>
          <dt>Difficulty</dt><dd>{_escape(diff)}</dd>
          <dt>Supported planners</dt><dd>{_escape(planners)}</dd>
          <dt>Expected runtime</dt>
          <dd><em class="estimate">{_escape(runtime_str)}</em></dd>
        </dl>
        <div class="card-actions">
          <code class="run-cmd">{_escape(card.run_command)}</code>
          {rollout_html}
        </div>
        <details class="card-params"><summary>raw params</summary>
          <pre>{params_json}</pre>
        </details>
      </div>
    </article>""".strip()


_CSS = """
  :root { color-scheme: light; }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 0; background: #f7f7f9; color: #222; }
  header { background: #1f2933; color: #fff; padding: 1.25rem 1.5rem; }
  header h1 { margin: 0 0 .25rem; font-size: 1.4rem; }
  header p { margin: 0; opacity: .85; font-size: .9rem; }
  .toolbar { padding: .75rem 1.5rem; background: #fff; border-bottom: 1px solid #e3e3e8;
             font-size: .85rem; }
  .toolbar code { background: #eef; padding: .1rem .35rem; border-radius: 4px; }
  main { padding: 1.5rem; }
  .grid { display: grid; gap: 1rem;
          grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); }
  .card { background: #fff; border: 1px solid #e3e3e8; border-radius: 10px;
          overflow: hidden; display: flex; flex-direction: column; }
  .thumb { width: 100%; height: 150px; object-fit: cover; background: #eef; display:block; }
  .thumb-missing { display:flex; align-items:center; justify-content:center;
                   color:#9aa; font-size:.8rem; }
  .card-body { padding: .85rem 1rem 1rem; display:flex; flex-direction:column; gap:.5rem; }
  .card-title { margin: 0; font-size: 1.05rem; word-break: break-word; }
  .card-meta { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: .15rem .6rem;
               font-size: .82rem; }
  .card-meta dt { color: #6b7280; font-weight: 600; }
  .card-meta dd { margin: 0; }
  .card-actions { display: flex; flex-direction: column; gap: .4rem; }
  .run-cmd { font-size: .72rem; background: #0d1117; color: #c9d1d9;
             padding: .4rem .5rem; border-radius: 6px; overflow-x: auto;
             white-space: pre-wrap; word-break: break-all; }
  .rollout-link { font-size: .8rem; }
  .rollout-link.muted { color: #9aa; }
  .card-params { font-size: .72rem; }
  .card-params pre { background: #f3f4f6; padding: .4rem; border-radius: 6px;
                     overflow-x: auto; margin: .35rem 0 0; }
  .estimate { color: #6b7280; }
  footer { padding: 1rem 1.5rem; font-size: .78rem; color: #6b7280;
           border-top: 1px solid #e3e3e8; background:#fff; }
"""


def _render_html(
    cards: Sequence[GalleryCard],
    *,
    title: str,
    matrix_path: str,
    thumbnail_dir_abs: Path,
    embed_thumbnails: bool,
    generated_at: str,
) -> str:
    """Render the full self-contained HTML page.

    Args:
        cards: Gallery cards in render order.
        title: Page title.
        matrix_path: Source scenario matrix path (root-relative).
        thumbnail_dir_abs: Absolute thumbnail directory (for embedding).
        embed_thumbnails: Embed thumbnails as base64 data URIs when True.
        generated_at: ISO-ish generation timestamp for the footer.

    Returns:
        Full HTML document string.
    """
    cards_html = "\n".join(
        _card_html(c, thumbnail_dir_abs=thumbnail_dir_abs, embed_thumbnails=embed_thumbnails)
        for c in cards
    )
    n = len(cards)
    disclaimer = (
        "Discoverability and inspection artifact only — not benchmark evidence. "
        "Expected runtime is a deterministic estimate; supported planners is a "
        "documented constant set, not a per-scenario measured capability."
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{_escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>{_escape(title)}</h1>
  <p>{n} scenario(s) &middot; source matrix: <code>{_escape(matrix_path)}</code></p>
</header>
<div class="toolbar">
  <strong>Rebuild:</strong>
  <code>uv run robot-sf gallery build --matrix {_escape(matrix_path)}</code>
  &nbsp;&middot;&nbsp; <span>{disclaimer}</span>
</div>
<main>
  <section class="grid">
{cards_html}
  </section>
</main>
<footer>
  Generated {generated_at} by <code>robot-sf gallery build</code> &middot;
  schema {GALLERY_HTML_SCHEMA_VERSION} &middot; {n} card(s).
</footer>
</body>
</html>
"""


def build_gallery(  # noqa: PLR0913
    scenarios: Sequence[Mapping[str, Any]],
    *,
    matrix_path: str,
    out_dir: Path | str = _DEFAULT_OUT_DIR,
    base_seed: int = 0,
    horizon_steps: int = _DEFAULT_HORIZON_STEPS,
    render_thumbnails: bool = True,
    embed_thumbnails: bool = True,
    sample_rollout_root: Path | str | None = None,
    title: str | None = None,
) -> GalleryBuildResult:
    """Build a self-contained static scenario/planner gallery.

    Renders per-scenario thumbnails (reusing the existing thumbnail renderer —
    no new rendering code), aggregates scenario metadata into cards, and writes a
    self-contained ``index.html`` plus a JSON manifest sidecar.

    Args:
        scenarios: Scenario dicts (typically from ``load_scenario_matrix``).
        matrix_path: Repository-root-relative path of the source matrix, echoed
            into run commands and the manifest.
        out_dir: Output directory for the gallery (HTML + manifest + thumbnails).
        base_seed: Base seed for deterministic thumbnail rendering.
        horizon_steps: Horizon used for the runtime estimate.
        render_thumbnails: When False, skip thumbnail rendering entirely (cards
            show a "no thumbnail" placeholder; no thumbnail directory is
            created). Useful for headless environments without matplotlib.
        embed_thumbnails: When True (and ``render_thumbnails`` is True), embed
            thumbnails as base64 data URIs so the page is fully portable. When
            False, thumbnails are referenced by relative path.
        sample_rollout_root: Optional directory searched for per-scenario sample
            rollouts (``<id>.mp4/.webm/.jsonl/.html``).
        title: Optional page title. Defaults to a name derived from the matrix.

    Returns:
        GalleryBuildResult describing the written artifacts.

    Raises:
        ValueError: If ``scenarios`` is empty.
    """
    scenarios_list = [dict(sc) for sc in scenarios]
    if not scenarios_list:
        raise ValueError(
            "build_gallery requires at least one scenario; the source matrix is empty."
        )

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    thumbnail_dir = out_dir_path / "thumbnails"

    # Deduplicate exact repeated labels, but fail closed when distinct labels
    # collide after sanitization; otherwise one scenario silently disappears.
    seen: dict[str, str] = {}
    unique_scenarios: list[dict[str, Any]] = []
    for sc in scenarios_list:
        label = resolve_scenario_label(sc)
        sid = sanitize_scenario_label(resolve_scenario_label(sc))
        previous_label = seen.get(sid)
        if previous_label is not None:
            if previous_label == label:
                continue
            raise ValueError(
                f"Scenario labels {previous_label!r} and {label!r} both sanitize to {sid!r}; "
                "use unique labels."
            )
        seen[sid] = label
        unique_scenarios.append(sc)

    # Render thumbnails via the existing renderer (no new rendering).
    thumbnail_relpath_by_id, thumbnail_dir_abs, thumbnails_rendered = _render_gallery_thumbnails(
        unique_scenarios,
        thumbnail_dir=thumbnail_dir,
        base_seed=base_seed,
        requested=render_thumbnails,
    )

    thumbnail_render_status, thumbnail_warnings = _thumbnail_render_state(
        requested=render_thumbnails,
        rendered=thumbnails_rendered,
        total=len(unique_scenarios),
    )

    supported = resolve_supported_planners()
    rollout_root = Path(sample_rollout_root) if sample_rollout_root is not None else None

    cards: list[GalleryCard] = []
    for sc in unique_scenarios:
        label = resolve_scenario_label(sc)
        sid = sanitize_scenario_label(label)
        ped_count = int(resolve_agent_count(sc))
        difficulty = estimate_initial_difficulty(sc)
        map_name = _resolve_map_name(sc)
        runtime = estimate_expected_runtime_seconds(ped_count, horizon_steps=horizon_steps)
        run_cmd = _build_run_command(matrix_path, sid, horizon_steps)
        thumb_rel = thumbnail_relpath_by_id.get(sid)

        sample_rel: str | None = None
        rollout_abs = _resolve_sample_rollout(sc, sample_rollout_root=rollout_root)
        if rollout_abs is not None:
            sample_rel = _stage_sample_rollout(
                rollout_abs,
                out_dir_path=out_dir_path,
                scenario_id=sid,
            )

        cards.append(
            GalleryCard(
                scenario_id=sid,
                label=label,
                map_name=map_name,
                pedestrian_count=ped_count,
                supported_planners=supported,
                expected_runtime_seconds=runtime,
                difficulty_band=str(difficulty.get("band", "unknown")),
                difficulty_score=float(difficulty.get("score", 0.0)),
                run_command=run_cmd,
                thumbnail_relpath=thumb_rel,
                sample_rollout_relpath=sample_rel,
                params=dict(sc),
            )
        )

    page_title = title or f"Scenario Gallery — {Path(matrix_path).name}"
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())

    html_doc = _render_html(
        cards,
        title=page_title,
        matrix_path=matrix_path,
        thumbnail_dir_abs=thumbnail_dir_abs,
        embed_thumbnails=render_thumbnails and embed_thumbnails,
        generated_at=generated_at,
    )
    html_path = out_dir_path / _DEFAULT_HTML_NAME
    html_path.write_text(html_doc, encoding="utf-8")

    manifest = {
        "schema_version": GALLERY_HTML_SCHEMA_VERSION,
        "generated_at": generated_at,
        "title": page_title,
        "matrix_path": matrix_path,
        "scenario_count": len(cards),
        "horizon_steps": horizon_steps,
        "render_thumbnails_requested": render_thumbnails,
        "render_thumbnails": thumbnail_render_status == "rendered",
        "thumbnails_rendered": thumbnails_rendered,
        "thumbnail_render_status": thumbnail_render_status,
        "embed_thumbnails": render_thumbnails and embed_thumbnails and thumbnails_rendered > 0,
        "warnings": thumbnail_warnings,
        "supported_planners": list(supported),
        "cards": [asdict(c) for c in cards],
    }
    manifest_path = out_dir_path / _DEFAULT_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    logger.info(
        "Gallery built: {} card(s) -> {} (HTML), {} (manifest)",
        len(cards),
        html_path,
        manifest_path,
    )

    return GalleryBuildResult(
        html_path=html_path.resolve(),
        manifest_path=manifest_path.resolve(),
        cards=cards,
        thumbnail_dir=thumbnail_dir_abs,
        matrix_path=matrix_path,
        schema_version=GALLERY_HTML_SCHEMA_VERSION,
    )


__all__ = [
    "GALLERY_HTML_SCHEMA_VERSION",
    "SUPPORTED_PLANNERS",
    "GalleryBuildResult",
    "GalleryCard",
    "build_gallery",
    "estimate_expected_runtime_seconds",
    "resolve_supported_planners",
]
