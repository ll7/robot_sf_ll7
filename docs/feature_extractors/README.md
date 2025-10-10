# Feature Extractor Evaluation and Improvements

This directory contains documentation for the enhanced feature extraction system that provides alternative feature extraction approaches while maintaining full backward compatibility with the original `DynamicsExtractor`.

## Quick Start

```python
# Basic usage with preset configurations
from robot_sf.feature_extractors.config import FeatureExtractorPresets
from stable_baselines3 import PPO
from robot_sf.gym_env.environment_factory import make_robot_env

# Create environment
env = make_robot_env(debug=False)

# Use MLP feature extractor
config = FeatureExtractorPresets.mlp_small()
policy_kwargs = config.get_policy_kwargs()

# Train with PPO
model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=100_000)
```

## Available Feature Extractors

### 1. Original DynamicsExtractor (Legacy)
- **File**: `robot_sf/feature_extractor.py`
- **Description**: Original convolutional approach with multiple layers
- **Parameters**: ~5K (conv mode) or 0 (flatten mode)
- **Use Case**: Baseline comparison, established performance

### 2. MLP Feature Extractor 
- **File**: `robot_sf/feature_extractors/mlp_extractor.py`
- **Description**: Simple Multi-Layer Perceptron approach
- **Parameters**: 50K-250K (configurable)
- **Advantages**: Fast training, fewer parameters, easy to understand
- **Use Case**: Quick prototyping, parameter-efficient training

### 3. Attention Feature Extractor
- **File**: `robot_sf/feature_extractors/attention_extractor.py`
- **Description**: Self-attention mechanism for LiDAR ray processing
- **Parameters**: 35K-200K (configurable)
- **Advantages**: Focuses on relevant rays, interpretable, handles sequences well
- **Use Case**: When environmental context matters most

### 4. Lightweight CNN Extractor
- **File**: `robot_sf/feature_extractors/lightweight_cnn_extractor.py`
- **Description**: Simplified convolutional approach
- **Parameters**: 4K-50K (configurable)
- **Advantages**: Preserves spatial relationships, fewer parameters than original
- **Use Case**: Balance between CNNs and efficiency

## Configuration System

The configuration system provides a unified way to select and configure feature extractors:

```python
from robot_sf.feature_extractors.config import (
    FeatureExtractorPresets,
    create_feature_extractor_config,
    FeatureExtractorType
)

# Use presets
config = FeatureExtractorPresets.mlp_large()

# Custom configuration
config = create_feature_extractor_config(
    "attention",
    embed_dim=128,
    num_heads=8,
    num_layers=2
)

# With enum
config = create_feature_extractor_config(
    FeatureExtractorType.LIGHTWEIGHT_CNN,
    num_filters=[64, 32, 16],
    kernel_sizes=[7, 5, 3]
)
```

## Training and Comparison

### Multi-Extractor Training
Run systematic comparison across all extractors:

```bash
# Run the default macOS-friendly comparison (single-thread)
uv run python scripts/multi_extractor_training.py \
  --config configs/scenarios/multi_extractor_default.yaml \
  --run-id doc-demo \
  --output-root results/feature_extractor_comparison

# Run the GPU/vectorized comparison (skips gracefully when CUDA is absent)
uv run python scripts/multi_extractor_training.py \
  --config configs/scenarios/multi_extractor_gpu.yaml \
  --run-id doc-gpu \
  --output-root results/feature_extractor_comparison

# Analyze any `complete_results.json` or `summary.json`
uv run python scripts/analyze_feature_extractors.py \
  results/feature_extractor_comparison/complete_results.json
```

### SLURM Cluster Training
For large-scale experiments on HPC clusters:

```bash
# Submit complete comparison job
sbatch SLURM/feature_extractor_comparison/run_comparison.slurm

# Or submit individual parallel jobs
./SLURM/feature_extractor_comparison/submit_parallel.sh
```

## Files and Structure

```
robot_sf/
├── feature_extractor.py                 # Original DynamicsExtractor (unchanged)
├── feature_extractors/                  # New feature extractors
│   ├── __init__.py                      # Module exports
│   ├── config.py                        # Configuration system
│   ├── mlp_extractor.py                 # MLP-based extractor
│   ├── attention_extractor.py           # Attention-based extractor
│   └── lightweight_cnn_extractor.py     # Lightweight CNN extractor
│
scripts/
├── multi_extractor_training.py          # Training comparison script
└── analyze_feature_extractors.py        # Statistical analysis script

SLURM/feature_extractor_comparison/
├── run_comparison.slurm                 # Complete comparison job
├── single_extractor_template.slurm      # Single extractor template
├── submit_parallel.sh                   # Parallel job submission
└── analyze_results.slurm                # Analysis job

examples/
└── demo_feature_extractors.py           # Demonstration script

tests/
└── test_feature_extractors.py           # Comprehensive test suite
```

## Performance Characteristics

| Extractor | Parameters | Speed | Use Case |
|-----------|------------|-------|----------|
| Dynamics (Conv) | ~5K | Medium | Baseline, established |
| Dynamics (Flatten) | 0 | Fast | Simple baseline |
| MLP Small | ~55K | Fast | Quick experiments |  
| MLP Large | ~254K | Medium | Better performance |
| Attention Small | ~35K | Medium | Interpretable results |
| Attention Large | ~200K | Slow | Best context modeling |
| Lightweight CNN | ~4K | Fast | Spatial relationships |

## Testing

Run comprehensive tests:

```bash
# Test all feature extractors
pytest tests/test_feature_extractors.py -v

# Test integration with StableBaselines3
pytest tests/test_feature_extractors.py::TestIntegrationWithStableBaselines3 -v
```

## Backward Compatibility

**Full backward compatibility is maintained:**
- Original `DynamicsExtractor` is unchanged
- Existing training scripts continue to work
- All current model files remain compatible
- No breaking changes to existing APIs

## Implementation Notes

### Design Principles
1. **Legacy Compatibility**: Never change existing code
2. **Consistent Interface**: All extractors follow the same API
3. **Configuration-Driven**: Easy to switch between extractors
4. **Testing**: Comprehensive test coverage
5. **Documentation**: Clear usage examples

### Key Features
- All extractors implement `BaseFeaturesExtractor` interface
- Unified observation space handling (rays + drive state)
- Configurable architectures for different use cases
- Statistical comparison framework
- Production-ready SLURM scripts

### Performance Considerations
- MLP extractors are fastest for training
- Attention extractors provide best interpretability
- CNN extractors balance performance and efficiency
- Original extractor remains available for comparison

## Future Enhancements

Potential improvements for future work:
- Graph-based feature extractors for structured environments
- Transformer-based architectures for sequence modeling
- Multi-modal fusion for additional sensor types
- AutoML for automatic architecture search
- Quantized models for deployment efficiency
