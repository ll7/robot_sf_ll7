# Metrics Specification

This document provides the formal specification for all metrics computed in the Social Navigation Benchmark.

## Overview

The benchmark computes a comprehensive set of metrics covering safety, efficiency, comfort, and overall navigation quality. All metrics are computed from episode trajectory data and aggregated across multiple runs.

## Distance-Based Thresholds

The following constants define the distance thresholds used throughout the metrics:

- **Collision distance**: 0.25m (strict boundary for collision events)
- **Near-miss distance**: 0.50m (upper bound for near-miss region)
- **Force comfort threshold**: 2.0 (unitless, force magnitude above which interaction is considered uncomfortable)

## Core Metrics

### Safety Metrics

#### Collisions
- **Definition**: Number of timesteps where minimum pedestrian distance < 0.25m
- **Type**: Integer count
- **Range**: [0, episode_length]
- **Purpose**: Direct safety measure

#### Near Misses  
- **Definition**: Number of timesteps where distance ∈ [0.25m, 0.50m)
- **Type**: Integer count
- **Range**: [0, episode_length]
- **Purpose**: Proximity safety measure

#### Minimum Interpersonal Distance
- **Definition**: Global minimum distance to any pedestrian across the episode
- **Type**: Float (meters)
- **Range**: [0, ∞)
- **Purpose**: Worst-case proximity analysis

#### Mean Interpersonal Distance
- **Definition**: Average minimum distance per timestep
- **Type**: Float (meters) 
- **Range**: [0, ∞)
- **Purpose**: Overall proximity behavior

### Task Performance Metrics

#### Success
- **Definition**: 1 if goal reached before horizon with zero collisions, 0 otherwise
- **Type**: Binary (0 or 1)
- **Purpose**: Task completion with safety constraint

#### Time to Goal (Normalized)
- **Definition**: Actual time to goal / horizon if successful, 1.0 if unsuccessful
- **Type**: Float
- **Range**: [0, 1]
- **Purpose**: Efficiency measure

#### Path Efficiency
- **Definition**: Shortest path length / actual path length
- **Type**: Float
- **Range**: (0, 1]
- **Purpose**: Path optimality measure

### Comfort Metrics

#### Force Mean
- **Definition**: Mean norm of interaction forces on robot
- **Type**: Float (force units)
- **Range**: [0, ∞)
- **Purpose**: Average interaction intensity

#### Force 95th Percentile
- **Definition**: 95th percentile of force norm distribution
- **Type**: Float (force units)
- **Range**: [0, ∞)
- **Purpose**: Extreme interaction events

#### Force Exceedance Events
- **Definition**: Number of timesteps with force > comfort threshold (2.0)
- **Type**: Integer count
- **Range**: [0, episode_length]
- **Purpose**: Discomfort event frequency

#### Comfort Exposure
- **Definition**: Proportion of timesteps with force > comfort threshold
- **Type**: Float
- **Range**: [0, 1]
- **Purpose**: Relative discomfort exposure

### Motion Quality Metrics

#### Smoothness (Jerk)
- **Definition**: Mean jerk magnitude (third derivative of position)
- **Type**: Float (m/s³)
- **Range**: [0, ∞)
- **Purpose**: Motion smoothness assessment

#### Path Curvature
- **Definition**: Mean path curvature
- **Type**: Float (1/m)
- **Range**: [0, ∞)
- **Purpose**: Path geometry analysis

#### Energy Consumption
- **Definition**: Sum of acceleration magnitudes over episode
- **Type**: Float (m/s²)
- **Range**: [0, ∞)
- **Purpose**: Energy efficiency proxy

### Composite Metrics

#### SNQI (Social Navigation Quality Index)
- **Definition**: Weighted combination of normalized component metrics
- **Type**: Float
- **Range**: Depends on normalization and weights
- **Purpose**: Overall navigation quality assessment
- **Formula**: SNQI = Σ(w_i × normalized_metric_i) where w_i are learned weights

## Normalization Strategy

For SNQI computation, metrics are normalized using baseline statistics:
- **Central tendency**: Median of baseline distribution
- **Scale**: 95th percentile - median range
- **Formula**: normalized = (raw_value - baseline_median) / (baseline_p95 - baseline_median)

## Implementation Notes

- Missing pedestrians (K=0): Distance-based metrics return appropriate defaults (NaN for min/mean distance, 0 for counts)
- Zero-length trajectories: Motion metrics return 0 or NaN as appropriate
- Force availability: Force-based metrics require `record_forces=True` during episode generation

## References

- Collision and near-miss thresholds aligned with existing test suite for backward compatibility
- Force threshold calibrated empirically for current fast-pysf physics model
- SNQI methodology follows composite index best practices with learned weights

*Last updated: September 2025*