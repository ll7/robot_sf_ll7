"""Sync a derived priority score into GitHub Project #5 items.

The score is intentionally simple and benchmark-oriented:

    score = improvement * success_probability * time_criticality * unlock_factor
            / effort_hours**alpha

This helper is the deterministic `gh` fallback for Project #5 score sync. It is
intentionally kept scriptable for local/manual batch routing even as
interactive issue/PR/project work moves toward GitHub MCP / app tools.

The helper reads issue-backed project items via `gh project item-list`, applies
defaults and clamping for missing or invalid inputs, and writes the derived
numeric score back to a `Priority Score` project field.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

DEFAULT_ALPHA = 0.8
DEFAULT_IMPROVEMENT = 1.0
DEFAULT_SUCCESS_PROBABILITY = 0.7
DEFAULT_EFFORT_HOURS = 1.0
DEFAULT_TIME_CRITICALITY = 1.0
DEFAULT_UNLOCK_FACTOR = 1.0

MIN_EFFORT_HOURS = 0.1
MIN_TIME_CRITICALITY = 0.5
MAX_TIME_CRITICALITY = 2.0
MIN_UNLOCK_FACTOR = 1.0
MAX_UNLOCK_FACTOR = 3.0
MIN_IMPROVEMENT = 0.0
MIN_SUCCESS_PROBABILITY = 0.0
MAX_SUCCESS_PROBABILITY = 1.0

EFFORT_FIELD = "Expected Duration in Hours"
PRIORITY_SCORE_FIELD = "Priority Score"
REQUIRED_NUMBER_FIELDS: tuple[str, ...] = (
    "Improvement",
    "Success Probability",
    "Time Criticality",
    "Unlock Factor",
    PRIORITY_SCORE_FIELD,
)


@dataclass(frozen=True, slots=True)
class ScoreInputs:
    """Normalized score inputs for one project item."""

    improvement: float
    success_probability: float
    effort_hours: float
    time_criticality: float
    unlock_factor: float


@dataclass(frozen=True, slots=True)
class SyncPreview:
    """Summary of one computed score update."""

    issue_number: int
    title: str
    status: str
    old_score: float | None
    new_score: float
    inputs: ScoreInputs


@dataclass(frozen=True, slots=True)
class SyncOptions:
    """Configuration for one score synchronization pass."""

    owner: str
    project_number: int
    ensure_fields: bool
    limit: int
    alpha: float
    round_digits: int
    issue_number: int | None
    dry_run: bool
    skip_statuses: set[str]


def clamp(value: float, *, lower: float, upper: float | None = None) -> float:
    """Clamp a numeric value into the allowed range."""

    bounded = max(lower, value)
    if upper is not None:
        bounded = min(upper, bounded)
    return bounded


def _coerce_float(raw: object) -> float | None:
    """Parse numbers emitted by the GitHub CLI item list output."""

    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return None
        return float(stripped)
    raise TypeError(f"unsupported numeric field value: {raw!r}")


def field_keys(name: str) -> tuple[str, ...]:
    """Return the known item-list key variants for a project field name."""

    lower_first = name[:1].lower() + name[1:]
    fully_lower = name.lower()
    if lower_first == fully_lower:
        return (lower_first,)
    return (lower_first, fully_lower)


def field_value(item: dict[str, Any], name: str) -> object:
    """Return a project field value from the gh item-list payload."""

    for key in field_keys(name):
        if key in item:
            return item[key]
    return None


def normalize_inputs(item: dict[str, Any]) -> ScoreInputs:
    """Extract and clamp score inputs from a project item payload."""

    improvement = _coerce_float(field_value(item, "Improvement"))
    success_probability = _coerce_float(field_value(item, "Success Probability"))
    effort_hours = _coerce_float(field_value(item, EFFORT_FIELD))
    time_criticality = _coerce_float(field_value(item, "Time Criticality"))
    unlock_factor = _coerce_float(field_value(item, "Unlock Factor"))

    return ScoreInputs(
        improvement=clamp(
            improvement if improvement is not None else DEFAULT_IMPROVEMENT,
            lower=MIN_IMPROVEMENT,
        ),
        success_probability=clamp(
            success_probability if success_probability is not None else DEFAULT_SUCCESS_PROBABILITY,
            lower=MIN_SUCCESS_PROBABILITY,
            upper=MAX_SUCCESS_PROBABILITY,
        ),
        effort_hours=clamp(
            effort_hours if effort_hours is not None else DEFAULT_EFFORT_HOURS,
            lower=MIN_EFFORT_HOURS,
        ),
        time_criticality=clamp(
            time_criticality if time_criticality is not None else DEFAULT_TIME_CRITICALITY,
            lower=MIN_TIME_CRITICALITY,
            upper=MAX_TIME_CRITICALITY,
        ),
        unlock_factor=clamp(
            unlock_factor if unlock_factor is not None else DEFAULT_UNLOCK_FACTOR,
            lower=MIN_UNLOCK_FACTOR,
            upper=MAX_UNLOCK_FACTOR,
        ),
    )


def compute_priority_score(inputs: ScoreInputs, *, alpha: float = DEFAULT_ALPHA) -> float:
    """Compute the derived priority score for one issue."""

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    numerator = (
        inputs.improvement
        * inputs.success_probability
        * inputs.time_criticality
        * inputs.unlock_factor
    )
    return numerator / math.pow(inputs.effort_hours, alpha)


class GhProjectClient:
    """Small wrapper around the gh CLI for project field automation."""

    def _run_completed(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run a gh command and raise a high-signal error on failure."""

        try:
            return subprocess.run(
                ["gh", *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip()
            stdout = exc.stdout.strip()
            details = stderr or stdout or "no stderr/stdout captured"
            raise RuntimeError(
                "gh command failed: "
                + " ".join(["gh", *args])
                + f"\n{details}\n"
                + "For interactive issue/PR/project work, prefer the GitHub MCP/app tools. "
                + "For this scripted fallback, verify `gh auth status` and ensure the token "
                + "has `project` scope."
            ) from exc

    def run_json(self, *args: str) -> dict[str, Any]:
        """Run a gh command and parse the JSON output."""

        completed = self._run_completed(*args)
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "gh command returned invalid JSON: "
                + " ".join(["gh", *args])
                + f"\n{completed.stdout.strip() or '<empty stdout>'}"
            ) from exc

    def run(self, *args: str) -> None:
        """Run a gh command for side effects."""

        self._run_completed(*args)

    def _should_retry_with_at_me(self, *, owner: str, error: RuntimeError) -> bool:
        """Limit `@me` fallback to the known ll7 user-owner gh quirk."""

        return owner == "ll7" and "unknown owner type" in str(error).lower()

    def _run_project_command(
        self,
        subcommand: str,
        *,
        owner: str,
        project_number: int,
        extra_args: Sequence[str] = (),
        as_json: bool = False,
    ) -> Any:
        """Run a `gh project` command with the known user-owner fallback."""

        args = (
            "project",
            subcommand,
            str(project_number),
            "--owner",
            owner,
            *extra_args,
        )
        if as_json:
            args = (*args, "--format", "json")
        try:
            if as_json:
                return self.run_json(*args)
            self.run(*args)
            return None
        except RuntimeError as exc:
            if not self._should_retry_with_at_me(owner=owner, error=exc):
                raise
            retry_args = (
                "project",
                subcommand,
                str(project_number),
                "--owner",
                "@me",
                *extra_args,
            )
            if as_json:
                return self.run_json(*retry_args, "--format", "json")
            self.run(*retry_args)
            return None

    def project_id(self, *, owner: str, project_number: int) -> str:
        """Return the GraphQL project ID."""

        payload = self._run_project_command(
            "view",
            owner=owner,
            project_number=project_number,
            as_json=True,
        )
        return str(payload["id"])

    def field_list(self, *, owner: str, project_number: int) -> list[dict[str, Any]]:
        """Return the current project fields."""

        payload = self._run_project_command(
            "field-list",
            owner=owner,
            project_number=project_number,
            as_json=True,
        )
        return list(payload["fields"])

    def ensure_number_field(self, *, owner: str, project_number: int, name: str) -> None:
        """Create a number field when it is missing."""

        self._run_project_command(
            "field-create",
            owner=owner,
            project_number=project_number,
            extra_args=(
                "--name",
                name,
                "--data-type",
                "NUMBER",
            ),
        )

    def item_list(self, *, owner: str, project_number: int, limit: int) -> list[dict[str, Any]]:
        """Return project items with their visible field values."""

        payload = self._run_project_command(
            "item-list",
            owner=owner,
            project_number=project_number,
            extra_args=(
                "--limit",
                str(limit),
            ),
            as_json=True,
        )
        return list(payload["items"])

    def update_number_field(
        self,
        *,
        item_id: str,
        field_id: str,
        project_id: str,
        number: float,
    ) -> None:
        """Write a numeric field value back to the project item."""

        self.run(
            "project",
            "item-edit",
            "--id",
            item_id,
            "--project-id",
            project_id,
            "--field-id",
            field_id,
            "--number",
            f"{number:.6f}",
        )


def field_map(fields: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index fields by their visible project name."""

    return {str(field["name"]): field for field in fields}


def ensure_required_fields(
    client: GhProjectClient,
    *,
    owner: str,
    project_number: int,
) -> dict[str, dict[str, Any]]:
    """Create missing numeric fields required by the score model."""

    fields = field_map(client.field_list(owner=owner, project_number=project_number))
    created_missing_field = False
    for name in REQUIRED_NUMBER_FIELDS:
        if name not in fields:
            client.ensure_number_field(owner=owner, project_number=project_number, name=name)
            created_missing_field = True
    if created_missing_field:
        return field_map(client.field_list(owner=owner, project_number=project_number))
    return fields


def build_previews(
    items: Iterable[dict[str, Any]],
    *,
    alpha: float,
    round_digits: int,
    issue_number: int | None,
    skip_statuses: set[str],
) -> list[SyncPreview]:
    """Compute score updates for the eligible project items."""

    previews: list[SyncPreview] = []
    for item in items:
        status = str(item.get("status", ""))
        if status in skip_statuses:
            continue

        content = item.get("content") or {}
        if content.get("type") != "Issue":
            continue

        raw_number = content.get("number")
        if not isinstance(raw_number, int) or raw_number < 0:
            continue
        number = raw_number
        if issue_number is not None and number != issue_number:
            continue

        inputs = normalize_inputs(item)
        score = round(compute_priority_score(inputs, alpha=alpha), round_digits)
        previews.append(
            SyncPreview(
                issue_number=number,
                title=str(content["title"]),
                status=status,
                old_score=_coerce_float(field_value(item, PRIORITY_SCORE_FIELD)),
                new_score=score,
                inputs=inputs,
            )
        )
    return previews


def write_summary(path: Path, previews: Sequence[SyncPreview]) -> None:
    """Persist a machine-readable sync summary."""

    payload = {
        "items": [
            {
                "issue_number": preview.issue_number,
                "title": preview.title,
                "status": preview.status,
                "old_score": preview.old_score,
                "new_score": preview.new_score,
                "inputs": asdict(preview.inputs),
            }
            for preview in previews
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sync_scores(
    client: GhProjectClient,
    options: SyncOptions,
) -> list[SyncPreview]:
    """Compute and optionally write derived scores for project items."""

    fields = (
        ensure_required_fields(
            client,
            owner=options.owner,
            project_number=options.project_number,
        )
        if options.ensure_fields
        else field_map(
            client.field_list(owner=options.owner, project_number=options.project_number)
        )
    )
    missing = [name for name in (*REQUIRED_NUMBER_FIELDS, EFFORT_FIELD) if name not in fields]
    if missing:
        raise ValueError(
            "project is missing required fields: "
            + ", ".join(sorted(missing))
            + ". Re-run with --ensure-fields or create them manually."
        )

    project_id = client.project_id(owner=options.owner, project_number=options.project_number)
    items = client.item_list(
        owner=options.owner,
        project_number=options.project_number,
        limit=options.limit,
    )
    previews = build_previews(
        items,
        alpha=options.alpha,
        round_digits=options.round_digits,
        issue_number=options.issue_number,
        skip_statuses=options.skip_statuses,
    )
    items_by_issue: dict[int, dict[str, Any]] = {}
    for item in items:
        content = item.get("content")
        if not isinstance(content, dict) or content.get("type") != "Issue":
            continue
        issue_number = content.get("number")
        if not isinstance(issue_number, int) or issue_number < 0:
            continue
        items_by_issue[issue_number] = item

    score_field_id = str(fields[PRIORITY_SCORE_FIELD]["id"])
    for preview in previews:
        item = items_by_issue[preview.issue_number]
        if preview.old_score is not None and math.isclose(
            preview.old_score,
            preview.new_score,
            rel_tol=1e-9,
            abs_tol=10 ** (-options.round_digits),
        ):
            continue
        if options.dry_run:
            continue
        client.update_number_field(
            item_id=str(item["id"]),
            field_id=score_field_id,
            project_id=project_id,
            number=preview.new_score,
        )

    return previews


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser("sync", help="Compute and sync the derived Priority Score field.")
    sync.add_argument("--owner", default="ll7", help="GitHub owner of the target project.")
    sync.add_argument("--project-number", type=int, default=5, help="Projects v2 number.")
    sync.add_argument("--limit", type=int, default=400, help="Maximum project items to inspect.")
    sync.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA, help="Effort dampening exponent."
    )
    sync.add_argument(
        "--round-digits",
        type=int,
        default=6,
        help="Decimal digits kept in the written Priority Score.",
    )
    sync.add_argument(
        "--ensure-fields",
        action="store_true",
        help="Create the missing numeric score fields before syncing.",
    )
    sync.add_argument(
        "--issue-number",
        type=int,
        help="Restrict sync to one issue-backed project item.",
    )
    sync.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the score updates without writing them back.",
    )
    sync.add_argument(
        "--summary-file",
        type=Path,
        help="Optional JSON file describing the computed sync results.",
    )
    sync.add_argument(
        "--skip-status",
        action="append",
        default=["Done"],
        help="Project status values to skip. Repeatable. Default: Done.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = _build_parser().parse_args(argv)
    if args.command != "sync":
        raise ValueError(f"unsupported command: {args.command}")

    previews = sync_scores(
        GhProjectClient(),
        SyncOptions(
            owner=args.owner,
            project_number=args.project_number,
            ensure_fields=args.ensure_fields,
            limit=args.limit,
            alpha=args.alpha,
            round_digits=args.round_digits,
            issue_number=args.issue_number,
            dry_run=args.dry_run,
            skip_statuses=set(args.skip_status),
        ),
    )

    if args.summary_file is not None:
        write_summary(args.summary_file, previews)

    print(
        json.dumps(
            {
                "project_number": args.project_number,
                "owner": args.owner,
                "items": [
                    {
                        "issue_number": preview.issue_number,
                        "title": preview.title,
                        "status": preview.status,
                        "old_score": preview.old_score,
                        "new_score": preview.new_score,
                        "inputs": asdict(preview.inputs),
                    }
                    for preview in previews
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
