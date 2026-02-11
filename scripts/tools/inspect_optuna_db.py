"""Inspect Optuna sqlite databases from the CLI.

Examples:
  uv run python scripts/tools/inspect_optuna_db.py --db output/optuna/weekend_optuna_expert_ppo.db
  uv run python scripts/tools/inspect_optuna_db.py --db output/optuna/weekend_optuna_expert_ppo.db --study-name weekend_optuna_auc
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import optuna
from optuna.trial import TrialState


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Optuna sqlite study databases.")
    parser.add_argument("--db", required=True, help="Path to the Optuna sqlite database file.")
    parser.add_argument(
        "--study-name", default=None, help="Study name to inspect (default: first study)."
    )
    parser.add_argument("--list-only", action="store_true", help="Only list studies and exit.")
    parser.add_argument("--top-n", type=int, default=5, help="Show top-N completed trials.")
    parser.add_argument(
        "--show-params",
        action="store_true",
        help="Include params in the printed top-N trials.",
    )
    parser.add_argument(
        "--export-csv",
        default=None,
        help="Optional CSV output path for trial data.",
    )
    return parser.parse_args()


def _storage_url(db_path: Path) -> str:
    return f"sqlite:///{db_path}"


def _format_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def _print_study_list(summaries: list[optuna.study.StudySummary]) -> None:
    if not summaries:
        print("No studies found in the database.")
        return
    print("Studies:")
    for summary in summaries:
        best_value_raw = getattr(summary, "best_value", None)
        if best_value_raw is None:
            best_trial = getattr(summary, "best_trial", None)
            best_value_raw = getattr(best_trial, "value", None) if best_trial else None
        best_value = _format_value(best_value_raw)
        print(f"  - {summary.study_name} | trials={summary.n_trials} | best={best_value}")


def _select_study(
    summaries: list[optuna.study.StudySummary],
    requested_name: str | None,
) -> str | None:
    if not summaries:
        return None
    if requested_name:
        for summary in summaries:
            if summary.study_name == requested_name:
                return requested_name
        available = ", ".join(sorted(s.study_name for s in summaries))
        raise ValueError(f"Study '{requested_name}' not found. Available: {available}")
    return summaries[0].study_name


def _completed_trials(study: optuna.study.Study) -> list[optuna.trial.FrozenTrial]:
    return [trial for trial in study.trials if trial.state == TrialState.COMPLETE]


def _sort_trials(
    trials: list[optuna.trial.FrozenTrial],
    direction: optuna.study.StudyDirection,
) -> list[optuna.trial.FrozenTrial]:
    if direction == optuna.study.StudyDirection.MINIMIZE:
        return sorted(trials, key=lambda t: float("inf") if t.value is None else t.value)
    return sorted(trials, key=lambda t: float("-inf") if t.value is None else t.value, reverse=True)


def _print_study_summary(
    study: optuna.study.Study,
    *,
    top_n: int,
    show_params: bool,
) -> None:
    print(f"\nStudy: {study.study_name}")
    direction_label = getattr(study.direction, "name", str(study.direction))
    print(f"Direction: {direction_label}")
    print(f"Trials: {len(study.trials)}")

    completed = _completed_trials(study)
    print(f"Completed trials: {len(completed)}")

    if not completed:
        print("No completed trials to summarize.")
        return

    best = study.best_trial
    print("Best trial:")
    print(f"  number={best.number}")
    print(f"  value={_format_value(best.value)}")
    if show_params:
        print(f"  params={json.dumps(best.params, sort_keys=True)}")

    sorted_trials = _sort_trials(completed, study.direction)
    print(f"\nTop {min(top_n, len(sorted_trials))} trials:")
    for trial in sorted_trials[:top_n]:
        line = f"  #{trial.number:03d} value={_format_value(trial.value)}"
        if show_params:
            line += f" params={json.dumps(trial.params, sort_keys=True)}"
        print(line)


def _export_csv(
    study: optuna.study.Study,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["number", "state", "value", "params", "user_attrs"])
        for trial in study.trials:
            writer.writerow(
                [
                    trial.number,
                    trial.state.name,
                    trial.value,
                    json.dumps(trial.params, sort_keys=True),
                    json.dumps(trial.user_attrs, sort_keys=True),
                ]
            )


def main() -> None:
    """Run the Optuna DB inspection CLI."""
    args = _parse_args()
    db_path = Path(args.db).expanduser().resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Optuna DB not found: {db_path}")

    storage = _storage_url(db_path)
    summaries = optuna.study.get_all_study_summaries(storage=storage)
    _print_study_list(summaries)

    if args.list_only:
        return

    study_name = _select_study(summaries, args.study_name)
    if study_name is None:
        return

    study = optuna.load_study(study_name=study_name, storage=storage)
    _print_study_summary(study, top_n=args.top_n, show_params=args.show_params)

    if args.export_csv:
        _export_csv(study, Path(args.export_csv))
        print(f"\nWrote CSV to {args.export_csv}")


if __name__ == "__main__":
    main()
