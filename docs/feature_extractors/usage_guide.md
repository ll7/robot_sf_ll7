# Usage Guide: Feature Extractors

This guide provides practical examples and best practices for using the enhanced feature extraction system.

## Basic Usage Patterns

### 1. Quick Start with Presets

```python
from robot_sf.feature_extractors.config import FeatureExtractorPresets
from stable_baselines3 import PPO
from robot_sf.gym_env.environment_factory import make_robot_env

# Create environment
env = make_robot_env()

# Choose a preset configuration
config = FeatureExtractorPresets.mlp_small()  # Fast, parameter-efficient
# config = FeatureExtractorPresets.attention_small()  # Interpretable
# config = FeatureExtractorPresets.lightweight_cnn()  # Balanced

# Get policy kwargs
policy_kwargs = config.get_policy_kwargs()

# Create and train model
model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=100_000)
```

### 2. Custom Configuration

```python
from robot_sf.feature_extractors.config import create_feature_extractor_config

# Custom MLP with specific architecture
mlp_config = create_feature_extractor_config(
    "mlp",
    ray_hidden_dims=[512, 256, 128],    # Larger ray processing
    drive_hidden_dims=[128, 64],        # Larger drive state processing
    dropout_rate=0.2                    # Higher regularization
)

# Custom attention with more heads
attention_config = create_feature_extractor_config(
    "attention",
    embed_dim=128,      # Larger embedding
    num_heads=16,       # More attention heads
    num_layers=3,       # Deeper network
    dropout_rate=0.1
)

# Use in training
policy_kwargs = mlp_config.get_policy_kwargs()
model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs)
```

### 3. Switching Between Extractors

```python
# Define different configurations
extractors = {
    'baseline': FeatureExtractorPresets.dynamics_original(),
    'fast': FeatureExtractorPresets.mlp_small(), 
    'powerful': FeatureExtractorPresets.mlp_large(),
    'interpretable': FeatureExtractorPresets.attention_small()
}

# Train with different extractors
results = {}
for name, config in extractors.items():
    print(f"Training with {name} extractor...")
    
    policy_kwargs = config.get_policy_kwargs()
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=50_000)
    
    # Evaluate performance
    mean_reward = evaluate_policy(model, env, n_eval_episodes=10)[0]
    results[name] = mean_reward
    
print("Results:", results)
```

## Training Scenarios

### Scenario 1: Quick Prototyping

**Goal**: Fast iteration during development

```python
# Use lightweight, fast extractors
config = FeatureExtractorPresets.mlp_small()
# or
config = FeatureExtractorPresets.lightweight_cnn()

# Short training with few environments
env = make_vec_env(lambda: make_robot_env(), n_envs=4)
model = PPO("MultiInputPolicy", env, 
           policy_kwargs=config.get_policy_kwargs(),
           n_steps=128)  # Smaller batch size
model.learn(total_timesteps=50_000)
```

### Scenario 2: Performance Optimization

**Goal**: Best possible performance

```python
# Use larger, more powerful extractors
config = FeatureExtractorPresets.mlp_large()
# or
config = FeatureExtractorPresets.attention_large()

# More environments and longer training
env = make_vec_env(lambda: make_robot_env(), n_envs=16)
model = PPO("MultiInputPolicy", env,
           policy_kwargs=config.get_policy_kwargs(),
           n_steps=2048,  # Larger batch size
           learning_rate=3e-4)
model.learn(total_timesteps=2_000_000)
```

### Scenario 3: Parameter Budget Constraints

**Goal**: Minimize model size while maintaining performance

```python
# Start with smallest extractor
config = FeatureExtractorPresets.lightweight_cnn()

# Check parameter count
env = make_robot_env()
model = PPO("MultiInputPolicy", env, policy_kwargs=config.get_policy_kwargs())
param_count = sum(p.numel() for p in model.policy.parameters())
print(f"Total parameters: {param_count:,}")

# If budget allows, try slightly larger
if param_count < 100_000:  # Your budget
    config = FeatureExtractorPresets.mlp_small()
    # Re-create model with new config
```

## Systematic Comparison Workflows

### 1. Complete Comparison Study

```bash
uv run python scripts/multi_extractor_training.py \
  --config configs/scenarios/multi_extractor_default.yaml \
  --run-id study-default \
  --output-root results/feature_extractor_comparison
```

This runs all extractors with the default hyperparameters and saves timestamped results for analysis.

### 2. Custom Comparison

```python
from pathlib import Path

import yaml  # type: ignore

config = {
    "run": {
        "run_id": "custom-mlp-attn",
        "worker_mode": "single-thread",
        "num_envs": 1,
        "total_timesteps": 200_000,
        "eval_freq": 10_000,
        "save_freq": 50_000,
        "n_eval_episodes": 5,
        "device": "cpu",
        "seed": 13,
    },
    "extractors": [
        {"name": "baseline", "preset": "dynamics_original", "expected_resources": "cpu"},
        {"name": "mlp_custom", "preset": "mlp_small", "parameters": {"dropout_rate": 0.05}},
        {"name": "attention_custom", "preset": "attention_small", "parameters": {"num_heads": 4}},
    ],
}

config_path = Path("configs/scenarios/custom_multi_extractor.yaml")
config_path.parent.mkdir(parents=True, exist_ok=True)
config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

output_root = Path("results/custom_multi_extractor")
output_root.mkdir(parents=True, exist_ok=True)

from scripts import multi_extractor_training

multi_extractor_training.main(
    [
        "--config",
        str(config_path),
        "--run-id",
        "custom-mix",
        "--output-root",
        str(output_root),
    ]
)
```

### 3. Ablation Studies

```python
# Study effect of MLP depth
from pathlib import Path

import yaml  # type: ignore

study_root = Path("results/ablation_studies")
study_root.mkdir(parents=True, exist_ok=True)

def run_ablation(run_label: str, extractor_names: list[str]) -> None:
    config = {
        "run": {
            "run_id": run_label,
            "worker_mode": "vectorized",
            "num_envs": 4,
            "total_timesteps": 150_000,
            "eval_freq": 5_000,
            "save_freq": 25_000,
            "n_eval_episodes": 5,
            "device": "cuda",
        },
        "extractors": [
            {"name": name, "preset": name, "expected_resources": "gpu"}
            for name in extractor_names
        ],
    }

    cfg_path = study_root / f"{run_label}.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    from scripts import multi_extractor_training

    multi_extractor_training.main(
        [
            "--config",
            str(cfg_path),
            "--run-id",
            run_label,
            "--output-root",
            str(study_root / run_label),
        ]
    )

run_ablation("mlp_depth", ["mlp_small", "mlp_large"])
run_ablation("attention_heads", ["attention_small", "attention_large"])
```

## Advanced Usage

### 1. Hyperparameter Tuning with Optuna

```python
import optuna
from optuna.integration import ChainedTrial

def objective(trial):
    # Sample hyperparameters
    ray_dims = [trial.suggest_categorical(f'ray_dim_{i}', [32, 64, 128, 256]) 
                for i in range(trial.suggest_int('n_ray_layers', 1, 3))]
    dropout = trial.suggest_float('dropout_rate', 0.0, 0.3)
    
    # Create config
    config = create_feature_extractor_config(
        "mlp",
        ray_hidden_dims=ray_dims,
        dropout_rate=dropout
    )
    
    # Train model
    env = make_vec_env(lambda: make_robot_env(), n_envs=4)
    model = PPO("MultiInputPolicy", env, policy_kwargs=config.get_policy_kwargs())
    model.learn(total_timesteps=100_000)
    
    # Evaluate
    mean_reward = evaluate_policy(model, env, n_eval_episodes=5)[0]
    env.close()
    
    return mean_reward

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print("Best params:", study.best_params)
```

### 2. Ensemble Methods

```python
# Train multiple models with different extractors
models = {}
extractors = ["mlp_small", "attention_small", "lightweight_cnn"]

for name in extractors:
    config = getattr(FeatureExtractorPresets, name)()
    model = PPO("MultiInputPolicy", env, policy_kwargs=config.get_policy_kwargs())
    model.learn(total_timesteps=200_000)
    models[name] = model

# Ensemble prediction
def ensemble_predict(obs):
    predictions = []
    for model in models.values():
        action, _ = model.predict(obs, deterministic=True)
        predictions.append(action)
    
    # Simple averaging (or use more sophisticated combination)
    return np.mean(predictions, axis=0)

# Use ensemble for evaluation
obs = env.reset()
action = ensemble_predict(obs)
```

### 3. Transfer Learning

```python
# Train base model with one extractor
base_config = FeatureExtractorPresets.mlp_large()
base_model = PPO("MultiInputPolicy", env, policy_kwargs=base_config.get_policy_kwargs())
base_model.learn(total_timesteps=1_000_000)

# Fine-tune with different extractor
finetune_config = FeatureExtractorPresets.attention_small() 
finetune_model = PPO("MultiInputPolicy", env, policy_kwargs=finetune_config.get_policy_kwargs())

# Initialize policy network from base model (feature extractor will be different)
# This is more complex and requires careful state dict manipulation
# See PyTorch transfer learning tutorials for detailed implementation
```

## Cluster Training Workflows

### 1. SLURM Job Submission

```bash
# Submit complete comparison
sbatch SLURM/feature_extractor_comparison/run_comparison.slurm

# Submit parallel jobs for faster execution
./SLURM/feature_extractor_comparison/submit_parallel.sh

# Check job status
squeue -u $USER

# Monitor logs
tail -f slurm_logs/feature_comparison_*.out
```

### 2. Custom SLURM Jobs

```bash
# Create custom extractor script
cat > my_extractor_job.slurm << EOF
#!/bin/bash
#SBATCH --job-name=my_extractor
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

source .venv/bin/activate
export DISPLAY=""

python -c "
from scripts.multi_extractor_training import MultiExtractorTraining
from robot_sf.feature_extractors.config import create_feature_extractor_config

config = create_feature_extractor_config('mlp', ray_hidden_dims=[512, 256])
trainer = MultiExtractorTraining(output_dir='./my_results', total_timesteps=2_000_000)
trainer.run_comparison({'my_extractor': config})
"
EOF

sbatch my_extractor_job.slurm
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Use smaller batch sizes or fewer environments
   config = FeatureExtractorPresets.mlp_small()  # Fewer parameters
   env = make_vec_env(lambda: make_robot_env(), n_envs=4)  # Fewer envs
   ```

2. **Slow Training**
   ```python
   # Switch to faster extractors
   config = FeatureExtractorPresets.lightweight_cnn()
   # Or reduce model complexity
   config = create_feature_extractor_config("mlp", ray_hidden_dims=[64, 32])
   ```

3. **Poor Performance**
   ```python
   # Try larger extractors
   config = FeatureExtractorPresets.mlp_large()
   # Or attention-based
   config = FeatureExtractorPresets.attention_small()
   ```

4. **Configuration Errors**
   ```python
   # Validate configuration
   try:
       config = create_feature_extractor_config("mlp", invalid_param=True)
   except Exception as e:
       print(f"Config error: {e}")
   
   # Check available presets
   available = [attr for attr in dir(FeatureExtractorPresets) if not attr.startswith('_')]
   print("Available presets:", available)
   ```

### Debugging

```python
# Test extractor in isolation
from gymnasium import spaces
import torch

# Create test observation space
obs_space = env.observation_space
sample_obs = env.observation_space.sample()

# Test extractor
config = FeatureExtractorPresets.mlp_small()
extractor_class = config.get_extractor_class()
extractor = extractor_class(obs_space, **config.params)

# Test forward pass
features = extractor(sample_obs)
print(f"Input shapes: {[(k, v.shape) for k, v in sample_obs.items()]}")
print(f"Output shape: {features.shape}")
print(f"Parameters: {sum(p.numel() for p in extractor.parameters())}")
```

## Best Practices

1. **Start Simple**: Begin with lightweight extractors for prototyping
2. **Measure Everything**: Always measure parameter count and training time
3. **Use Presets First**: Try preset configurations before customizing
4. **Compare Systematically**: Use provided comparison scripts
5. **Monitor Training**: Watch for overfitting with complex extractors
6. **Document Choices**: Record which extractors work best for your use case
7. **Version Control**: Save configurations with your results
