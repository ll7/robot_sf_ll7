# Issue #4014 PPO sequence encoder smoke comparison

Claim boundary: diagnostic-only matched smoke comparison; not benchmark-strength, paper-grade, or dissertation claim evidence.

This artifact compares proximal policy optimization (PPO), true RecurrentPPO long short-term memory (LSTM), and PPO-Mamba smoke summaries only when every row has populated throughput and parameter-count metadata.

It is diagnostic-only evidence and does not promote benchmark, paper, or dissertation claims.

| Model | Wall-clock seconds | Steps/sec | Trainable policy parameters |
| --- | ---: | ---: | ---: |
| ppo | 10.7252 | 509.54 | 115013 |
| recurrent_ppo_lstm | 10.4531 | 195.923 | 1009093 |
| ppo_mamba | 15.9484 | 308.897 | 53781 |

Closure note: All three smoke rows supplied real parameter and throughput summaries. This supports #4014 closure at diagnostic smoke tier only.
