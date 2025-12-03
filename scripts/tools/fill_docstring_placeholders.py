"""Replace placeholder docstring entries with descriptive text.

The repository previously contained many docstrings with the literal text
"Auto-generated placeholder description.". This helper scans source files and
replaces those placeholders with heuristically generated descriptions so the
functions remain informative and pass Ruff's docstring rules.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PLACEHOLDER = "Auto-generated placeholder description."
PATTERN = re.compile(
    r"^(?P<indent>\s*)(?P<token>[^:\n]+)(?P<colon>:\s*)" + re.escape(PLACEHOLDER) + r"\s*$"
)

# Common word replacements so humanized phrases read naturally.
WORD_MAP = {
    "acc": "acceleration",
    "algo": "algorithm",
    "cfg": "config",
    "cfgs": "configs",
    "dens": "density",
    "dir": "directory",
    "dirs": "directories",
    "dt": "time step",
    "env": "environment",
    "envs": "environments",
    "goal": "goal",
    "hist": "history",
    "info": "info",
    "map": "map",
    "param": "parameter",
    "params": "parameters",
    "ped": "pedestrian",
    "peds": "pedestrians",
    "pos": "position",
    "traj": "trajectory",
    "vel": "velocity",
}

TOKEN_MAP_EXACT = {
    "AggregateMetric": "Aggregate metric descriptor used for reporting",
    "EpisodeData": "Episode data container with robot and pedestrian trajectories",
    "Generator[_MinimalState, None, None]": "Generator yielding minimal state tuples",
    "GlobalRoute": "Global route definition used by navigation",
    "HardwareProfile | None": "Optional hardware profile describing the host machine",
    "Iterable[DeprecationEntry]": "Iterable of cataloged deprecation entries",
    "Iterable[dict[str, Any]]": "Iterable containing mapping payloads",
    "Iterable[tuple[str, float | None]]": "Iterable of (label, value) pairs",
    "Iterator[dict]": "Iterator over serialized records",
    "MapDefinition | None": "Optional map definition for rendering",
    "Mapping[str, Any]": "Mapping with string keys and arbitrary values",
    "Path": "Path-like object pointing to a file or directory",
    "PipelineRunStatus | None": "Optional pipeline run status value",
    "ReplayEpisode": "Recorded replay episode payload",
    "RunHistoryEntry": "Entry describing a past run in the registry",
    "SFPlannerConfig": "Social Force planner configuration dataclass",
    "SimulatorFactory": "Simulator factory callable",
    "StepExecutionEntry": "Entry describing a tracker step",
    "StepExecutionEntry | None": "Optional tracker step entry",
    "TelemetrySampler": "Telemetry sampler helper",
    "TelemetrySnapshot": "Snapshot emitted by the telemetry sampler",
    "TrajectoryDatasetValidationResult": "Validation result for a trajectory dataset",
    "TrajectoryQuality": "Trajectory quality classification tuple",
    "Vec2D": "2D vector (x, y)",
    "VisualizableSimState": "State structure compatible with the visualization UI",
    "]": "List of simple policy metadata entries",
    "np.ndarray": "NumPy array holding the relevant values",
    "np.ndarray | None": "Optional NumPy array; ``None`` indicates no data",
    "spaces.Box": "Gymnasium Box space describing limits",
    "th.Tensor": "Torch tensor processed by the network",
    "tuple[np.ndarray, np.ndarray, np.ndarray]": "Tuple of NumPy arrays containing position, velocity, and acceleration data",
    "tuple[np.ndarray, np.ndarray]": "Pair of NumPy arrays (typically start and goal)",
    "tuple[PedState, list[PedGrouping], ZoneAssignments]": "Pedestrian state, grouping, and zone assignment tuple",
    "argparse.ArgumentParser": "ArgumentParser configured for this CLI",
    "datetime | None": "Optional datetime; ``None`` when the timestamp is unknown",
    "dict[str, Any] | None": "Optional dictionary of arbitrary metadata",
    "float | None": "Optional floating-point value",
    "set[int]": "Set of integer identifiers",
    "set[str]": "Set of string identifiers",
    "str | None": "Optional string value",
}

TOKEN_MAP = {
    "a": "First plotted array",
    "act": "Action sampled from the policy",
    "action": "Action applied to the environment",
    "actions": "Batch of actions over the step",
    "advice": "Human-readable advice text",
    "algo": "Algorithm identifier",
    "angle": "Relative angle measurement",
    "arch": "Architecture identifier",
    "archetype": "Scenario archetype name",
    "args": "Parsed CLI arguments",
    "argv": "Command-line argument list",
    "arrays": "Collection of NumPy arrays",
    "ascending": "Whether sorting should be ascending",
    "ax": "Matplotlib axes object",
    "b": "Second plotted array",
    "baseline": "Baseline statistics bundle",
    "bins": "Histogram bin edges",
    "bool": "Boolean flag",
    "categories": "Example categories array",
    "cfg": "Configuration dictionary",
    "conf": "Configuration namespace",
    "color": "Color specification for plotting",
    "config": "Configuration object controlling the component",
    "context": "Context dictionary passed to the resolver",
    "count": "Number of requested items",
    "ctx": "Validation context dictionary",
    "d": "Dictionary of metric values",
    "data": "Data payload consumed by the helper",
    "debug": "Debug mode flag",
    "default": "Default fallback value",
    "dens": "Density configuration",
    "density": "Density parameter",
    "details": "Additional textual details",
    "diff": "Difference between baseline and comparison",
    "dir_path": "Directory path containing relevant assets",
    "dist": "Distance scalar",
    "distance": "Distance scalar",
    "dotted": "Matplotlib dotted-line style",
    "dropout": "Dropout probability",
    "dt": "Simulation time step",
    "effect": "Effect size payload",
    "effects": "List of effect descriptors",
    "env": "Environment instance",
    "err": "Exception captured during processing",
    "ep": "Episode record",
    "entries": "Manifest entries to write",
    "entry": "Individual manifest or tracker entry",
    "episode": "Episode instance",
    "error": "Exception or error payload",
    "episodes": "Episode records consumed by the CLI",
    "exc": "Captured exception type",
    "extra": "Additional context payload",
    "fallback": "Fallback option when primary data is missing",
    "factory": "Factory callable used to construct components",
    "fig": "Matplotlib Figure",
    "figsize": "Desired figure size",
    "first": "First element in the pair",
    "float": "Floating-point value",
    "flow": "Traffic flow descriptor",
    "flows": "Flow configuration data",
    "force": "Applied force value",
    "fraction": "Fractional share of the resource",
    "frame": "Frame index",
    "fps": "Frames per second",
    "func": "Callable invoked by the sampler",
    "goal": "Goal configuration",
    "goals": "Goal definitions",
    "groups": "Collection of grouped elements",
    "heading": "Heading angle",
    "high": "Upper bound value",
    "horizon": "Episode horizon (max steps)",
    "i": "Loop index",
    "idx": "Index into the dataset",
    "index": "Index value",
    "item": "Item extracted from the collection",
    "job": "Job configuration entry",
    "jobs": "Collection of jobs to execute",
    "key": "Dictionary key",
    "kind": "Kind selector",
    "kwargs": "Additional keyword arguments forwarded downstream",
    "length": "Length value",
    "lines": "Line objects plotted on the axes",
    "loc": "Normal distribution mean",
    "low": "Lower bound",
    "manifest": "Manifest payload",
    "message": "Human-readable message",
    "map": "Map definition",
    "means": "Mean metric values",
    "meta": "Metadata dictionary",
    "metas": "List of metadata entries",
    "metric": "Metric identifier",
    "metrics": "Dictionary of computed metrics",
    "mins": "Per-metric minimums",
    "mode": "Operating mode string",
    "mod": "Module object",
    "n": "Number of requested samples",
    "name": "Human-friendly name",
    "namespace": "Namespace container",
    "note": "Optional note for the record",
    "notes": "Collection of textual notes",
    "obs": "Observation dictionary or tensor",
    "obstacles": "Obstacle data used for collision checks",
    "obj": "Generic object payload",
    "ok": "Whether the operation succeeded",
    "orient": "Orientation value",
    "other": "Comparison target value",
    "out_path": "Output path for generated artifacts",
    "override": "Override identifier",
    "overrides": "Override dictionary",
    "p": "Probability or configuration dictionary",
    "p1": "First pedestrian state",
    "p2": "Second pedestrian state",
    "params": "Parameter dictionary",
    "parser": "Argument parser instance",
    "path": "Filesystem path to the resource",
    "paths": "Multiple filesystem paths",
    "pattern": "Glob or regex pattern",
    "payload": "Serialized payload passed between components",
    "ped_positions": "Pedestrian positions",
    "peds": "Collection of pedestrians",
    "pos": "Position vector",
    "pose": "Pose tuple (position + heading)",
    "prefix": "String prefix used for labeling",
    "pretrained": "Whether the checkpoint is pretrained",
    "progress_cb": "Progress callback",
    "quiet": "Suppress verbose logging flag",
    "r": "Random number generator",
    "reason": "Reason string recorded for diagnostics",
    "rec": "Single record dictionary",
    "record": "Record being processed",
    "records": "List of serialized records",
    "recommendations": "Recommendation objects emitted by telemetry",
    "recs": "List of recommendation entries",
    "rect": "Rectangle definition",
    "registry": "Run registry instance",
    "renderer": "Renderer helper",
    "report": "Report payload",
    "result": "Computation result",
    "results": "Collection of computation results",
    "resume": "Resume flag indicating incremental processing",
    "rng": "Random number generator",
    "root": "Repository or workspace root path",
    "route": "Route definition",
    "rows": "Row definitions for table output",
    "samples": "Sampled data points",
    "rules": "Recommendation ruleset",
    "s": "Scenario tuple identifier",
    "sc": "Sub-command configuration",
    "scale": "Scale parameter",
    "scenario": "Scenario definition dictionary",
    "scenarios": "Scenario list",
    "schema": "Schema definition",
    "seed": "Random seed for deterministic behavior",
    "sensor": "Sensor reference",
    "sensors": "Sensor collection wiring",
    "series": "Time series data",
    "signals": "Signal set watched by the tracker",
    "signum": "Signal number as delivered to the handler",
    "sim": "Simulator instance",
    "simulator": "Simulator backend wrapper",
    "size": "Sample size",
    "snapshot": "Telemetry snapshot",
    "sp": "Subparser entry",
    "specs": "Specification dictionary",
    "speed": "Speed scalar",
    "speeds": "Speed measurements",
    "state": "State payload",
    "status": "Status string",
    "step": "Step index",
    "steps": "Number of steps executed",
    "strict": "Whether strict validation is enabled",
    "str": "String value",
    "subparsers": "ArgumentParser subparsers object",
    "summary": "Aggregate summary data",
    "t": "Time index",
    "tau": "Time constant",
    "target": "Target path or identifier",
    "tb": "Traceback object",
    "text": "Text body for the report",
    "threshold": "Threshold value",
    "timestep": "Simulation timestep metadata",
    "title": "Plot title",
    "total": "Aggregate total",
    "ts_start": "Timestamp when processing started",
    "vals": "Numeric values extracted from the dataset",
    "value": "Scalar metric value",
    "values": "Collection of numeric values",
    "v": "Velocity magnitude",
    "v1": "First velocity component",
    "v2": "Second velocity component",
    "vel": "Velocity vector",
    "waypoint": "Waypoint metadata",
    "weights": "Weight dictionary",
    "workers": "Number of worker processes",
    "writer": "Telemetry writer",
    "x": "X-axis value",
    "y": "Y-axis value",
    "zones": "Configured zones for the scenario",
}

BASIC_TYPES = {
    "Any": "Arbitrary value passed through unchanged",
    "bool": "Boolean flag",
    "dict": "Dictionary value",
    "float": "Floating-point value",
    "int": "Integer value",
    "list": "List value",
    "np.ndarray": "NumPy array holding the computed values",
    "Path": "Filesystem path object",
    "str": "String value",
}

CAMEL_SUFFIXES = (
    ("Config", "configuration"),
    ("Entry", "entry"),
    ("Result", "result"),
    ("Dataset", "dataset"),
    ("Snapshot", "snapshot"),
    ("Manifest", "manifest"),
)


def humanize_snake(token: str) -> str:
    """Convert a ``snake_case`` token into a readable string.

    Args:
        token: Source token using underscores to separate words.

    Returns:
        str: Human-friendly phrase with underscores removed and common
            abbreviations expanded.
    """

    parts = [WORD_MAP.get(part, part) for part in token.split("_") if part]
    return " ".join(parts)


def split_camel(name: str) -> list[str]:
    """Split ``CamelCase`` identifiers into their individual words.

    Args:
        name: CamelCase token.

    Returns:
        list[str]: Ordered list of component words preserving digits.
    """

    return re.findall(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z0-9]+", name)


def humanize_camel(token: str) -> str:
    """Return a lower-case phrase for CamelCase identifiers.

    Args:
        token: CamelCase token to humanize.

    Returns:
        str: Lower-cased words joined by spaces.
    """

    words = split_camel(token)
    return " ".join(word.lower() if word.isalpha() else word for word in words)


def describe_camel(token: str) -> str | None:
    """Describe CamelCase tokens by expanding known suffixes.

    Args:
        token: CamelCase token potentially ending with ``Config`` or similar
            suffixes.

    Returns:
        str | None: Description composed from the prefix and suffix, or
        ``None`` when no heuristic applies.
    """

    for suffix, replacement in CAMEL_SUFFIXES:
        if token.endswith(suffix):
            prefix = token[: -len(suffix)]
            human = humanize_camel(prefix).strip()
            human_prefix = human if human else suffix.lower()
            return f"{human_prefix} {replacement}".strip()
    human = humanize_camel(token)
    return human if human else None


def describe_snake(token: str) -> str | None:
    """Describe snake_case tokens with contextual templates.

    Args:
        token: snake_case token.

    Returns:
        str | None: Human-readable description or ``None`` if heuristics do
        not match.
    """

    lower = token.lower()
    if lower.startswith("num_"):
        base = humanize_snake(token[4:])
        return f"number of {base}".strip()
    for suffix, template in (
        ("_traj", "trajectory of {base}"),
        ("_positions", "positions for {base}"),
        ("_position", "position for {base}"),
        ("_pos", "position for {base}"),
        ("_vel", "velocity for {base}"),
        ("_acc", "acceleration for {base}"),
        ("_goal", "goal for {base}"),
        ("_start", "start value for {base}"),
        ("_path", "filesystem path for the {base}"),
        ("_dir", "directory for {base}"),
        ("_ids", "identifiers for {base}"),
        ("_id", "identifier for {base}"),
        ("_name", "name of {base}"),
        ("_names", "names of {base}"),
        ("_seed", "random seed for {base}"),
        ("_weights", "weights for {base}"),
        ("_baseline", "baseline data for {base}"),
        ("_count", "count of {base}"),
    ):
        if lower.endswith(suffix):
            base = token[: -len(suffix)] or suffix.removeprefix("_")
            human = humanize_snake(base)
            return template.format(base=human or suffix.strip("_")).strip()
    human = humanize_snake(token)
    return human or None


def describe_generic(token: str) -> str | None:
    """Describe structured typing annotations like ``list[str]``.

    Args:
        token: Raw type annotation string.

    Returns:
        str | None: Description of the collection, or ``None`` when the
        structure is unknown.
    """

    if token.startswith("list[") and token.endswith("]"):
        inner = token[5:-1]
        return f"list of {inner}"
    if token.startswith("dict[") and token.endswith("]"):
        inner = token[5:-1]
        return f"mapping of {inner}"
    if token.startswith("tuple[") and token.endswith("]"):
        inner = token[6:-1]
        return f"tuple of {inner}"
    if token.startswith("Iterable["):
        inner = token[len("Iterable[") : -1]
        return f"iterable of {inner}"
    if token.startswith("Iterator["):
        inner = token[len("Iterator[") : -1]
        return f"iterator of {inner}"
    return None


def describe_token(token: str) -> str | None:
    """Return a human-readable description for a docstring placeholder.

    Args:
        token: Raw token captured from the placeholder docstring line.

    Returns:
        str | None: Description string or ``None`` if the token is unknown.
    """

    normalized = token.strip()
    if not normalized:
        return None

    resolvers = (
        _describe_exact_token,
        _describe_lower_token,
        _describe_basic_type_token,
        _describe_generic_token,
        _describe_camel_token,
        _describe_snake_token,
    )
    for resolver in resolvers:
        desc = resolver(normalized)
        if desc:
            return desc
    return None


def _describe_exact_token(token: str) -> str | None:
    """Look up tokens that require an exact string match."""

    return TOKEN_MAP_EXACT.get(token)


def _describe_lower_token(token: str) -> str | None:
    """Look up lower-case tokens in :data:`TOKEN_MAP`."""

    return TOKEN_MAP.get(token.lower())


def _describe_basic_type_token(token: str) -> str | None:
    """Return descriptions for primitive Python or project types."""

    return BASIC_TYPES.get(token)


def _describe_generic_token(token: str) -> str | None:
    """Describe templated collection tokens (e.g., ``list[str]``)."""

    if "[" in token and token.endswith("]"):
        return describe_generic(token)
    return None


def _describe_camel_token(token: str) -> str | None:
    """Describe CamelCase tokens when they resemble identifiers."""

    if token[0].isupper() and token.replace("_", "").isidentifier():
        return describe_camel(token)
    return None


def _describe_snake_token(token: str) -> str | None:
    """Describe ``snake_case`` tokens when an underscore is present."""

    if "_" in token:
        return describe_snake(token)
    return None


def replace_placeholders(path: Path) -> tuple[str, set[str], bool]:
    """Replace placeholder docstrings inside ``path``.

    Args:
        path: File to scan.

    Returns:
        tuple[str, set[str], bool]: Updated text, unresolved tokens, and
        whether modifications were made.
    """

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    changed = False
    missing: set[str] = set()
    for idx, line in enumerate(lines):
        match = PATTERN.match(line)
        if not match:
            continue
        token = match.group("token").strip()
        desc = describe_token(token)
        if not desc:
            missing.add(token)
            continue
        desc = desc.rstrip(".") + "."
        lines[idx] = f"{match.group('indent')}{token}{match.group('colon')}{desc}"
        changed = True
    if not changed:
        return text, missing, False
    newline = "\n" if text.endswith("\n") else ""
    return "\n".join(lines) + newline, missing, True


def main() -> None:
    """Entrypoint for the placeholder replacement CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="robot_sf", help="Base package to scan")
    parser.add_argument("--apply", action="store_true", help="Write changes back to disk")
    args = parser.parse_args()

    root = Path(args.root)
    unresolved: dict[str, set[str]] = {}
    updated_files = 0

    for path in root.rglob("*.py"):
        new_text, missing, changed = replace_placeholders(path)
        if missing:
            unresolved.setdefault(str(path), set()).update(missing)
        if not changed:
            continue
        if args.apply:
            path.write_text(new_text, encoding="utf-8")
        updated_files += 1

    if unresolved:
        print("Unresolved tokens detected:")
        for file_path, tokens in sorted(unresolved.items()):
            for token in sorted(tokens):
                print(f"{file_path}: {token}")
        raise SystemExit(1)

    print(f"Updated {updated_files} files")


if __name__ == "__main__":
    main()
