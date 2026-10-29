# Explicit curriculum stage weight inheritance

`reward_curriculum.stage_weight_merge` controls how a stage's nested `reward_kwargs.weights`
interacts with top-level reward weights. The default is `merge`: inherit the top-level map,
then apply stage overrides. Profile defaults fill only weights missing from both maps.
Input kwargs are deep-copied per stage, so later caller mutations and changes to another
stage's nested maps cannot change its rewards. Invalid modes and every present non-mapping
`weights` value fail at construction in both modes, including null, false, zero, strings,
and lists. Empty weight mappings remain valid: merge inherits, replace uses profile defaults.

```yaml
reward_curriculum:
  stage_weight_merge: merge
  stages:
    - until_episodes: 100
      reward_kwargs:
        weights:
          progress: 1.1
    - reward_kwargs: {}
```

Use explicit `stage_weight_merge: replace` to reproduce historical stage-weight resolution:
a stage's weight map replaces the top-level map and omitted terms use profile defaults.
This reproduces weight resolution only; the broken per-env curriculum counter is separate.
Every built-in stage logs its fully resolved map at construction.

For the issue-791 large-capacity leader base, default merge gives both stages:

| Term | Weight |
| --- | ---: |
| progress | 1.1 |
| living | -0.015 |
| collision | -15.0 |
| near_miss | -1.5 |
| ttc_risk | -1.2 |
| comfort | -0.5 |
| smoothness | -0.18 |
| timeout | -6.5 |
| stagnation | -0.8 |
| terminal_bonus | 20.0 |

The sparse historical first stage consequently has the same weights as the final stage in
merge mode. A meaningful new curriculum must explicitly declare distinct overrides; this
repair does not rewrite historical configs. Replace mode retains the old first-stage defaults.

This changes future training weight resolution only, not benchmark evaluation of existing
checkpoints. No observation, action, termination, canonical benchmark metric, checkpoint,
frozen artifact or training seed-policy code is changed. No training was run.
