# Issue #3950 Pedestrian-Model Sensitivity Smoke

This is a CPU-only diagnostic smoke report. It does not train a policy and does not make paper-facing claims.

Claim boundary: CPU diagnostic sensitivity harness. No new training. Development-model axis is declared policy provenance unless backed by a training artifact.

The development-model axis is declared policy provenance unless a training artifact explicitly supports it. The evaluation-model axis is the active `simulation_config.pedestrian_model` selector.

| Development model | Evaluation model | Episodes | Success incidence | Collision incidence | Status |
| --- | --- | ---: | ---: | ---: | --- |
| social_force_default | social_force_default | 3 | 0.000000 | 0.000000 | ok |
| social_force_default | hsfm_total_force_v1 | 3 | 0.000000 | 0.000000 | ok |
| hsfm_total_force_v1 | social_force_default | 3 | 0.000000 | 0.000000 | ok |
| hsfm_total_force_v1 | hsfm_total_force_v1 | 3 | 0.000000 | 0.000000 | ok |
