# Examiner development configs, 2026-10-12

Diagnostic-only; new opt-in configs. Nothing changes benchmark defaults or
frozen release inputs. Use dev seeds 1061–1090 on host CPUs.

`study.json` selects the 14-arm 0.0.8 roster and ten representative Figure 7.3
family cells. The radius/speed slice uses ORCA, social force, static hybrid v4,
goal and release PPO. `legacy.yaml` preserves spawn-coupled preferred speed;
the nine `r*_v*.yaml` matrices select radius 0.3/0.5/1.0 m and deterministic
preferred speed 0.65/1.0/1.3 m/s. Authored release episode budgets are retained.

`*_legacy.yaml` copies the release algorithm bindings. `*_swap.yaml` changes
non-2.0 command ceilings to 2.0 and 2.0 ceilings to 1.2 m/s. Run the swap with
the plant ceiling changed to the same value. DWA retains candidate count and
relative spacing; conditional lower caps and SA-CADRL's preferred speed remain.
The requested swap still has two cap groups, so it is not a fully equal-cap test.
Only inherited manifest names were cleaned up after acquisition; those labels
are the sole difference from the archived acquisition inputs. No numerical
treatment or frozen file was changed by that cleanup.

The complete dataset, source snapshots and SHA256 custody manifest are delivered
in the separate diagnostic tooling/evidence PR. Configuration changes require
the hosted behavior receipt; this development sweep is not that receipt. The
orchestrator owns receipt production. No release, model promotion, thesis
admission, automatic merge or self-approval is authorized here.
