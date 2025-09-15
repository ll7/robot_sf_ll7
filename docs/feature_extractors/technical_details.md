# Technical Details: Feature Extractor Architecture

This document provides detailed technical information about the feature extractor implementations and their architectural decisions.

## Architecture Overview

All feature extractors follow the same interface pattern:

```python
class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, **kwargs):
        # Calculate features_dim based on architecture
        super().__init__(observation_space, features_dim=total_features)
        
        # Initialize ray processor
        self.ray_extractor = ...
        
        # Initialize drive state processor
        self.drive_state_extractor = ...
    
    def forward(self, obs: dict) -> torch.Tensor:
        # Process rays and drive state separately
        ray_features = self.ray_extractor(obs[OBS_RAYS])
        drive_features = self.drive_state_extractor(obs[OBS_DRIVE_STATE])
        
        # Concatenate features
        return torch.cat([ray_features, drive_features], dim=1)
```

## Observation Space Structure

All extractors work with the same observation format:

```python
observation = {
    'rays': torch.Tensor,      # Shape: (batch, timesteps, num_rays)
    'drive_state': torch.Tensor # Shape: (batch, timesteps, state_dim)
}

# Typical dimensions:
# rays: (batch, 5, 64) - 5 timesteps, 64 LiDAR rays  
# drive_state: (batch, 5, 5) - 5 timesteps, 5 state variables
```

## Detailed Architecture Analysis

### 1. MLP Feature Extractor

**Core Concept**: Flatten all inputs and process through fully-connected layers.

```python
# Ray processing: (batch, 5, 64) -> (batch, 320) -> MLP -> (batch, ray_output_dim)
ray_input_dim = 5 * 64 = 320
ray_layers = [Linear(320, 128), ReLU(), Dropout(0.1),
              Linear(128, 64), ReLU(), Dropout(0.1)]

# Drive state processing: (batch, 5, 5) -> (batch, 25) -> MLP -> (batch, drive_output_dim)  
drive_input_dim = 5 * 5 = 25
drive_layers = [Linear(25, 32), ReLU(), Dropout(0.1),
                Linear(32, 16), ReLU(), Dropout(0.1)]

# Final features: (batch, 64 + 16) = (batch, 80)
```

**Parameter Count**: Approximately 54,760 parameters for default configuration
- Ray MLP: 320→128→64 ≈ 49,280 parameters
- Drive MLP: 25→32→16 ≈ 1,312 parameters  
- Policy network: Additional ~14,000 parameters

**Advantages**:
- Simple and interpretable
- Fast training and inference
- No assumptions about spatial structure
- Easy to tune hyperparameters

**Disadvantages**:
- Loses spatial relationships in LiDAR data
- May overfit with high-dimensional inputs
- Doesn't capture temporal dependencies explicitly

### 2. Attention Feature Extractor

**Core Concept**: Treat LiDAR rays as a sequence and apply self-attention.

```python
# Ray embedding: (batch, num_rays, timesteps) -> (batch, num_rays, embed_dim)
ray_embedding = Linear(timesteps=5, embed_dim=64)

# Multi-head self-attention
for layer in attention_layers:
    # Q, K, V projections
    Q = query_proj(rays)  # (batch, num_rays, embed_dim)
    K = key_proj(rays)    # (batch, num_rays, embed_dim)  
    V = value_proj(rays)  # (batch, num_rays, embed_dim)
    
    # Attention weights: softmax(QK^T / sqrt(d_k))
    attention_weights = softmax(Q @ K.T / sqrt(embed_dim))
    
    # Attended features: attention_weights @ V
    attended = attention_weights @ V
    
    # Residual connection and layer norm
    rays = layer_norm(rays + attended)

# Global pooling: (batch, num_rays, embed_dim) -> (batch, embed_dim)
ray_features = global_average_pool(attended_rays)
```

**Parameter Count**: Approximately 35,408 parameters for default configuration
- Embedding layers: ~5,000 parameters
- Attention layers: ~25,000 parameters
- Drive state MLP: ~1,500 parameters
- Policy network: ~14,000 parameters

**Advantages**:
- Learns which rays are most important
- Handles variable-length sequences naturally
- More interpretable than pure CNNs
- Can model long-range dependencies

**Disadvantages**:
- More complex than MLPs
- Quadratic complexity in sequence length
- May be overkill for simple navigation tasks

### 3. Lightweight CNN Extractor  

**Core Concept**: Use simplified 1D convolutions to preserve spatial relationships.

```python
# Ray processing: 1D convolutions over ray dimension
# Input: (batch, timesteps, num_rays) = (batch, 5, 64)

conv_layers = [
    Conv1d(in_channels=5, out_channels=32, kernel_size=5, padding=2),
    BatchNorm1d(32), ReLU(), MaxPool1d(2), Dropout(0.1),
    
    Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1), 
    BatchNorm1d(16), ReLU(), MaxPool1d(2), Dropout(0.1),
    
    AdaptiveAvgPool1d(output_size=16),  # Ensure consistent output size
    Flatten()  # (batch, 16 * 16) = (batch, 256)
]
```

**Parameter Count**: Approximately 3,840 parameters for default configuration
- Conv layers: ~3,000 parameters
- Drive state MLP: ~500 parameters
- BatchNorm: ~300 parameters

**Advantages**:
- Preserves spatial relationships between adjacent rays
- Much fewer parameters than original DynamicsExtractor
- Fast training and inference
- Batch normalization improves stability

**Disadvantages**:
- Still assumes spatial structure in ray ordering
- Less flexible than attention mechanisms
- Fixed receptive field size

## Performance Comparison

### Computational Complexity

| Extractor | Forward Pass | Memory Usage | Training Speed |
|-----------|-------------|-------------|----------------|
| MLP | O(n) | Medium | Fast |
| Attention | O(n²) | High | Medium |
| Lightweight CNN | O(n) | Low | Fast |
| Original CNN | O(n) | Medium | Medium |

### Parameter Efficiency

```python
# Typical parameter counts for default configurations:
{
    'dynamics_conv': 5296,
    'dynamics_flatten': 0, 
    'mlp_small': 54760,
    'mlp_large': 253936,
    'attention_small': 35408,
    'attention_large': ~200000,
    'lightweight_cnn': 3840
}
```

### Feature Dimensionality

```python
# Output feature dimensions:
{
    'dynamics_conv': 287,      # Variable based on convolution output
    'dynamics_flatten': 831,   # Flattened input size + drive state
    'mlp_small': 40,          # ray_dim + drive_dim
    'mlp_large': 80,          # ray_dim + drive_dim
    'attention_small': 80,     # embed_dim + drive_dim  
    'lightweight_cnn': 272    # conv_output + drive_dim
}
```

## Design Decisions and Trade-offs

### 1. Separate Processing Pipelines
**Decision**: Process rays and drive state separately, then concatenate.

**Rationale**: 
- Drive state is already compact and structured
- LiDAR rays need more sophisticated processing
- Allows different architectures for different data types
- Maintains compatibility with original observation format

### 2. Configurable Architectures
**Decision**: Make most architectural choices configurable via parameters.

**Rationale**:
- Different use cases may need different trade-offs
- Enables hyperparameter tuning and ablation studies
- Provides flexibility without code changes
- Supports preset configurations for common use cases

### 3. Standard PyTorch Layers
**Decision**: Use standard PyTorch layers instead of custom implementations.

**Rationale**:
- Better optimization and GPU utilization
- Easier debugging and profiling
- Compatibility with PyTorch ecosystem
- Automatic differentiation support

### 4. Feature Dimension Calculation
**Decision**: Calculate feature dimensions automatically in `__init__`.

**Rationale**:
- Prevents dimension mismatch errors
- Works with different observation spaces
- Simplifies configuration management
- Required by StableBaselines3 interface

## Integration with StableBaselines3

### Policy Network Integration

```python
# The feature extractor is automatically integrated:
policy = ActorCriticPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    features_extractor_class=FeatureExtractor,
    features_extractor_kwargs=extractor_config.params
)

# Training loop:
features = policy.features_extractor(observations)  # Extract features
actions = policy.actor(features)                   # Generate actions  
values = policy.critic(features)                   # Estimate values
```

### Gradient Flow

All extractors maintain proper gradient flow:
- No gradient blocking operations
- Differentiable activation functions
- Proper weight initialization
- Compatible with automatic differentiation

### Memory Management

- Efficient tensor operations
- Minimal temporary tensor creation
- GPU-compatible implementations
- Batch processing support

## Testing Strategy

### Unit Tests
- Individual component functionality
- Forward pass correctness
- Parameter counting
- Gradient flow verification

### Integration Tests  
- StableBaselines3 compatibility
- Training loop integration
- Model saving/loading
- Multi-environment support

### Performance Tests
- Training speed benchmarks
- Memory usage profiling
- Inference speed measurements
- Parameter efficiency analysis

## Future Extensions

### Potential Improvements
1. **Graph Neural Networks**: For structured environments
2. **Temporal Transformers**: For better sequence modeling  
3. **Multi-scale CNNs**: For different spatial scales
4. **Ensemble Methods**: Combining multiple extractors
5. **Neural Architecture Search**: Automated design optimization

### Extensibility Points
1. **New Extractor Types**: Add to registry in `config.py`
2. **Custom Observation Spaces**: Extend base classes
3. **Multi-modal Fusion**: Add image/audio processing
4. **Domain-specific Architectures**: Task-specific designs