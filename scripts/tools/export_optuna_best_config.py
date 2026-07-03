"""Export the selected best expert-PPO config from an Optuna study."""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import yaml
from optuna.trial import TrialState

from robot_sf.training.optuna_provenance import hash_file, write_best_config_artifacts


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for best-config export."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path, help="Optuna sqlite database path.")
    parser.add_argument("--study-name", required=True, help="Optuna study name.")
    parser.add_argument("--base-config", required=True, type=Path, help="Base expert PPO YAML.")
    parser.add_argument("--out", required=True, type=Path, help="Output best-config YAML path.")
    parser.add_argument(
        "--best-trial-json", type=Path, default=None, help="Output best trial JSON path."
    )
    parser.add_argument(
        "--report", required=True, type=Path, help="Output selection report Markdown path."
    )
    return parser


def select_best_trial(
    study: optuna.study.Study,
) -> tuple[optuna.trial.FrozenTrial, dict[str, object]]:
    """Select the best complete trial, preferring safety-feasible trials when recorded."""

    complete_trials = [
        trial
        for trial in study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        if trial.value is not None
    ]
    if not complete_trials:
        raise ValueError("No complete Optuna trials with objective values are available.")

    feasible_trials = [
        trial
        for trial in complete_trials
        if trial.user_attrs.get("safety_constraints_feasible") is True
    ]
    candidate_trials = feasible_trials or complete_trials
    reverse = study.direction == optuna.study.StudyDirection.MAXIMIZE
    selected = sorted(candidate_trials, key=lambda trial: float(trial.value), reverse=reverse)[0]
    return selected, {
        "complete_trials_considered": len(complete_trials),
        "safety_filter_applied": bool(feasible_trials),
    }


def render_effective_config(
    base_config_path: Path, trial: optuna.trial.FrozenTrial
) -> dict[str, object]:
    """Apply recorded Optuna params to a base expert PPO YAML mapping."""

    with base_config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Base config must be a YAML mapping.")
    config = dict(raw)
    ppo_hyperparams = dict(config.get("ppo_hyperparams") or {})
    for key, value in trial.params.items():
        if key == "policy_net_arch":
            config["policy_net_arch"] = _coerce_policy_net_arch(value)
        else:
            ppo_hyperparams[str(key)] = value
    if ppo_hyperparams:
        config["ppo_hyperparams"] = ppo_hyperparams
    return config


def _coerce_policy_net_arch(value: object) -> list[int]:
    if isinstance(value, str):
        return [int(dim.strip()) for dim in value.split(",") if dim.strip()]
    if isinstance(value, (list, tuple)):
        return [int(dim) for dim in value]
    raise ValueError("policy_net_arch trial param must be a comma string or list.")


def main(argv: list[str] | None = None) -> int:
    """Run best-config export."""

    args = build_arg_parser().parse_args(argv)
    db_path = args.db.expanduser().resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Optuna DB not found: {db_path}")
    base_config_path = args.base_config.expanduser().resolve()
    storage = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=args.study_name, storage=storage)
    trial, selection_counts = select_best_trial(study)
    effective_config = render_effective_config(base_config_path, trial)
    best_trial_path = args.best_trial_json or args.out.with_name("best_trial.json")
    selection = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "base_config_path": str(base_config_path),
        "base_config_sha256": hash_file(base_config_path),
        **selection_counts,
    }
    write_best_config_artifacts(
        best_trial_path=best_trial_path,
        best_config_path=args.out,
        report_path=args.report,
        trial=trial,
        effective_config=effective_config,
        selection=selection,
    )
    print(f"Wrote best config: {args.out}")
    print(f"Wrote best trial: {best_trial_path}")
    print(f"Wrote selection report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
