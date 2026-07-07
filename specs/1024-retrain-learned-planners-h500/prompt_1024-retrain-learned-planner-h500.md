Below is a structured plan to design a training suite for the updated scenario described in the GitHub issue. The recommendations draw on recent documentation, tutorials and research in reinforcement learning and hyper‑parameter optimization.

1. Establish a reliable baseline

1. Update the environment and reward – the horizon (episode length) is a property of the environment, but the effective horizon of an agent depends on the discount factor γ ￼.  A longer horizon typically requires a higher γ (closer to 1) so that the agent values long‑term rewards ￼, but very large γ can slow or destabilize learning ￼.  Start by porting the previous best setup (algorithm, network architecture, reward structure) to the new scenario and adjust γ (e.g., test 0.95–0.99, as moderate values often balance short‑ and long‑term planning ￼).
2. Normalize inputs and outputs – when using PPO/A2C the observation vector should be normalized (e.g., using VecNormalize) ￼.  Normalization helps the optimizer see a consistent scale across different observations and can improve stability.
3. Set up evaluation – use a separate test environment and periodically evaluate over multiple episodes (5–20) ￼.  This avoids biased estimates due to exploration noise and makes it possible to compare runs fairly.  Track mean and standard deviation of returns across several seeds.

2. Select candidate algorithms

Stable‑Baselines3 notes that PPO, SAC, TD3 and DroQ usually require little tuning, but default settings rarely work for new environments ￼.  The RL Zoo and its “early‑2026 update” recommend:

* PPO – emphasises stability via trust‑region updates ￼.  A recent recipe suggests using a large mini‑batch size (6 400–25 600 transitions), a larger network, a KL‑adaptive learning‑rate schedule, and unbounded actions ￼.
* SAC / TD3 / TQC / DroQ – off‑policy algorithms that trade stability for improved sample efficiency.  They are considered state‑of‑the‑art for continuous control ￼.  SAC is sometimes preferred for robots because it fine‑tunes well and can be made fast using parallel simulators ￼.
* Consider new variants – recent work on Relative‑Entropy Pathwise Policy Optimization (REPPO) and DroQ aims to improve the robustness and efficiency of on‑policy algorithms; including them in the sweep can reveal whether they offer advantages in the new scenario.

3. Define the hyper‑parameter search space

Good RL performance depends on appropriate hyper‑parameters ￼.  It is not efficient to exhaustively try all combinations ￼.  A well‑designed sweep should:

1. Choose ranges informed by prior work – use tuned values from RL Zoo and papers as starting points; widen ranges where the new horizon or observation space may affect performance.  Example parameters to vary for PPO include:
    * learning rate schedule (constant, linear decay, KL‑adaptive);
    * number of environment steps between policy updates;
    * number of epochs per update;
    * clip range;
    * entropy coefficient;
    * network architecture (e.g., MLP vs. convolutional or attention‑based extractors);
    * discount factor γ and GAE λ (if long horizon, test higher γ and λ).
        For SAC/TD3, consider learning rates for actor/critic, target‑update rates (τ), and network sizes.  Avoid guessing values – consult the RL Zoo or algorithm papers for sensible defaults ￼.
2. Use automatic hyper‑parameter optimization – grid search wastes resources and scales poorly ￼.  Instead, adopt random search or Bayesian optimization.  Random search samples uniformly and avoids missing important regions of the parameter space ￼; Bayesian optimization uses a surrogate model and acquisition function to guide sampling toward promising regions ￼.  These methods are available through libraries such as Optuna and can integrate with Stable‑Baselines3.
3. Employ pruning/schedulers – allocate a fixed budget and prune unpromising trials early ￼ ￼.  For example, run each trial for 25 % of the total training timesteps; drop the worst‑performing trials and devote the remaining budget to the better ones.  This accelerates convergence to a good configuration and reduces wasted computation.
4. Parallelize trials – use vectorized environments to run multiple environments concurrently; this significantly speeds up rollout and training ￼.  Libraries such as SB3 + JAX (SBX) or JAX/Flax allow fast simulation and gradient computations; switching to these frameworks can yield substantial speed‑ups ￼.
5. Track experiments systematically – use tools like MLflow or Weights & Biases Sweep to manage trials, record hyper‑parameters, and store training curves.  MLflow provides a unified interface for tracking multiple runs, nested sweeps and metrics ￼.

4. Consider reward redesign

Reward shaping often requires several iterations ￼.  A longer horizon may dilute reward signals; dense or potential‑based shaping can help by providing intermediate feedback without altering the optimal policy.  For example, combine terminal success rewards with step penalties that encourage progress along the path, or use shaping terms that reduce energy consumption or maintain stability.  Keep the shaping function bounded to avoid large gradients and test its effect on learning.

5. Evaluate and iterate

1. Multiple seeds – because RL results vary with random seeds ￼, evaluate each promising configuration across several seeds and report mean ± standard deviation.  Use rliable or similar tools to compare algorithms and hyper‑parameter choices fairly ￼.
2. Continuous monitoring – monitor training curves to detect instability; early plateaus may indicate that a configuration is not promising and can be pruned.  Stable‑Baselines3 provides EvalCallback to automate periodic evaluations ￼.
3. Iterate – refine the search space and reward design based on the outcomes of initial sweeps.  Focus on configurations that yield a good balance between sample efficiency and final performance.

6. Structuring the GitHub issue/prompt

When creating the GitHub issue for this training sweep, include:

* A clear description of the environment changes (observation space, action space, episode horizon) and any modifications to the reward structure.
* The list of algorithms to test (e.g., PPO, SAC, TD3, TQC/DroQ, REPPO).
* The hyper‑parameter ranges and search method (random/Bayesian), along with total budget (number of timesteps) and pruning schedule.
* The feature extractors or network architectures to consider (e.g., MLP, CNN, transformer).
* The evaluation protocol: number of seeds, number of episodes per evaluation, and metrics to track.
* References to RL Zoo or specific configurations used as baselines for reproducibility.

This structured approach leverages best practices from recent reinforcement‑learning literature and should help identify promising configurations efficiently without exhaustive trial‑and‑error.

⸻

References

<!-- Each reference explains a key aspect of algorithm or hyper‑parameter selection. -->

1. Stable‑Baselines3 Team. “Reinforcement Learning Tips and Tricks.” Stable‑Baselines3 User Guide, version 2.9.0a2. Highlights the importance of tuning hyper‑parameters, suggests using the RL Zoo for tuned values and automatic hyper‑parameter optimization ￼; stresses input normalization ￼; notes sample inefficiency and recommends increasing training budget ￼; and explains that PPO uses trust regions for stability ￼. Accessed 6 May 2026.
2. Antonin Raffin. “Automatic Hyperparameter Tuning – A Visual Guide (Part 1).” Personal Blog, 15 May 2023. Discusses the trade‑off between the number of configurations and total budget; advocates pruning poor trials and focusing resources on promising ones ￼; explains why grid search is inefficient and random or Bayesian search is preferable ￼ ￼; and introduces the role of samplers and pruners in hyper‑parameter optimization ￼. Accessed 6 May 2026.
3. Antonin Raffin. “Recent Advances in RL for Continuous Control – SOTA Early 2026.” Mannheim RL Workshop Slides, 6 Feb 2026. Provides a concise “PPO recipe” recommending large mini‑batch sizes (6 400–25 600 transitions), larger networks, KL‑adaptive learning‑rate schedules and unbounded action spaces ￼. Accessed 6 May 2026.
4. Felipe Vieira Frujeri. “Effective Horizon in Reinforcement Learning.” Blog, Sept 2025. Clarifies that the horizon is defined by the environment, whereas the discount factor γ determines the agent’s effective horizon ￼; describes how low, moderate and near‑undiscounted discount factors influence learning dynamics ￼; and notes that γ ≈ 0.95–0.99 often yields a good balance between short‑term and long‑term planning ￼. Accessed 6 May 2026.
5. Yasin Yousif. “Speeding up Training of Model‑Free Reinforcement Learning: A Comparative Evaluation for Fast and Accurate Learning.” Robot Learning by Example, 17 Aug 2025. Reports that JAX‑based environment batching yields significant speed‑ups on GPUs and that advanced hyper‑parameter search methods implemented in Optuna can improve RL training ￼; describes vectorized environments in Gymnasium, which allow running multiple instances concurrently and speed up trajectory rollouts ￼. Accessed 6 May 2026.
