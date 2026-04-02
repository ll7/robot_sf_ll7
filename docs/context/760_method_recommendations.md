# Recommendations for Benchmark Testing - Issue 760

## Executive Summary

Based on the detailed analysis of HEIGHT's performance issues and the comprehensive survey of available navigation methods, this document recommends specific approaches to test with the Robot SF benchmark. The recommendations prioritize methods that address HEIGHT's key failure modes while leveraging approaches that have already shown success.

## Top Recommendations

### 1. ORCA Variants (HIGH PRIORITY)

**Recommended Methods:**
- **Non-holonomic ORCA** (Optimal Reciprocal Collision Avoidance for Multiple Non-Holonomic Robots, DARS 2013)
- **Relaxing ORCA Limitations** (RA-L 2024) - addresses ORCA's limitations in crowded scenarios  
- **ORCA-DD** (IROS 2010) - specifically for differential-drive constraints

**Rationale:**
- ORCA already performs well (70.21%) in the benchmark
- Rule-based approaches require no training and are robust to domain shifts
- Multiple variants available for different robot kinematics
- Directly addresses HEIGHT's action projection and domain mismatch issues

**Expected Benefits:**
- Immediate performance improvement over HEIGHT
- No training required, easy to implement
- Compatible with existing benchmark infrastructure

**Implementation Complexity:** Low
**Time Estimate:** 1-2 weeks

### 2. DRL-VO (HIGH PRIORITY)

**Recommended Method:**
- **DRL-VO: Learning to Navigate Through Crowded Dynamic Scenes Using Velocity Obstacles** (T-RO 2023)

**Rationale:**
- Hybrid approach combining deep reinforcement learning with velocity obstacles
- Specifically designed for crowded dynamic scenes
- Addresses limitations of both pure RL and pure rule-based approaches
- Could outperform both ORCA and HEIGHT by combining their strengths

**Expected Benefits:**
- Better social navigation than pure ORCA
- More robust than pure RL approaches
- Potential for state-of-the-art performance

**Implementation Complexity:** Medium
**Time Estimate:** 2-3 weeks

### 3. MPC-Based Approaches (MEDIUM PRIORITY)

**Recommended Methods:**
- **DR-MPC: Deep Residual Model Predictive Control for Real-world Social Navigation** (RA-L 2025)
- **SICNav: Safe and Interactive Crowd Navigation using Model Predictive Control and Bilevel Optimization** (T-RO 2024)

**Rationale:**
- Model Predictive Control provides explicit safety guarantees
- Can handle complex constraints and dynamic environments
- DR-MPC specifically addresses social navigation challenges
- MPC approaches are less sensitive to observation format differences

**Expected Benefits:**
- Explicit safety guarantees in crowded scenarios
- Better handling of complex constraints
- Potential for smoother, more natural navigation

**Implementation Complexity:** High
**Time Estimate:** 3-4 weeks

### 4. Attention-Based RL Methods (MEDIUM PRIORITY)

**Recommended Methods:**
- **Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph** (ICRA 2023)
- **ST2: Spatial-Temporal State Transformer for Crowd-Aware Autonomous Navigation** (RA-L 2023)

**Rationale:**
- Conceptually similar to HEIGHT but with more recent architectures
- Addresses the same heterogeneous interaction modeling problem
- Potential improvements over HEIGHT's graph transformer approach
- Can leverage existing HEIGHT integration experience

**Expected Benefits:**
- Better performance than HEIGHT on same scenario types
- More modern architectures with potential improvements
- Direct comparison point to understand HEIGHT's limitations

**Implementation Complexity:** Medium-High
**Time Estimate:** 3-5 weeks

### 5. Foundation Model Approaches (EXPLORATORY)

**Recommended Methods:**
- **VLM-Social-Nav: Socially Aware Robot Navigation Through Scoring Using Vision-Language Models** (RA-L 2025)
- **GSON: A Group-Based Social Navigation Framework With Large Multimodal Model** (RA-L 2025)

**Rationale:**
- Cutting-edge approaches using vision-language models
- Potential for better generalization across diverse scenarios
- Could handle the scenario complexity that challenged HEIGHT
- Emerging area with significant potential

**Expected Benefits:**
- Better generalization across scenario types
- Potential for handling complex social interactions
- Future-proof approach with significant upside

**Implementation Complexity:** High
**Time Estimate:** 4-6 weeks

## Implementation Strategy

### Phase 1: Quick Wins (1-2 weeks)
**Objective:** Achieve immediate performance improvements

1. **Implement ORCA variants**
   - Start with non-holonomic ORCA
   - Test ORCA-DD for differential-drive robots
   - Evaluate "Relaxing ORCA Limitations" variant

2. **Baseline testing**
   - Compare all ORCA variants against current ORCA implementation
   - Identify best performer for each scenario type

**Success Criteria:**
- Achieve >75% success rate across benchmark scenarios
- Identify best ORCA variant for each scenario category

### Phase 2: Performance Optimization (2-4 weeks)
**Objective:** Push performance beyond current state-of-the-art

1. **Implement DRL-VO**
   - Build on existing ORCA infrastructure
   - Train/test on benchmark scenarios

2. **Test MPC approaches**
   - Start with DR-MPC implementation
   - Compare against DRL-VO and ORCA variants

3. **Attention-based RL**
   - Implement "Intention Aware" method
   - Compare directly with HEIGHT performance

**Success Criteria:**
- Achieve >80% success rate on benchmark
- Identify 2-3 top performing methods
- Understand performance trade-offs

### Phase 3: Future Exploration (Longer term)
**Objective:** Explore cutting-edge approaches with high potential

1. **Foundation model approaches**
   - Implement VLM-Social-Nav
   - Test GSON framework

2. **Hybrid approaches**
   - Combine best performers from Phase 1 and 2
   - Explore custom combinations

**Success Criteria:**
- Evaluate potential of foundation models
- Identify promising directions for future work
- Establish baseline for next-generation approaches

## Addressing HEIGHT's Specific Failure Modes

### Domain Mismatch
**Solution:** Focus on methods with better generalization:
- **ORCA variants**: Rule-based, scenario-agnostic
- **MPC approaches**: Explicit constraint handling
- **Foundation models**: Designed for generalization

### Action Projection Issues  
**Solution:** Focus on methods with direct control:
- **ORCA variants**: Direct velocity commands
- **MPC approaches**: Optimized control outputs
- **DRL-VO**: Hybrid approach with velocity obstacles

### Observation Differences
**Solution:** Focus on methods less sensitive to observation format:
- **Rule-based methods**: Work with any observation format
- **MPC approaches**: Can adapt to different observation models
- **Foundation models**: Designed to handle diverse inputs

## Expected Outcomes

### Short-term (1-2 months)
- Replace HEIGHT with better-performing ORCA variant
- Achieve 75-85% success rate across benchmark
- Establish strong baseline for comparison

### Medium-term (3-6 months)
- Identify 2-3 top-performing methods
- Achieve 85-95% success rate on benchmark
- Understand method strengths/weaknesses by scenario type

### Long-term (6+ months)
- Develop hybrid or custom approach
- Potentially achieve >95% success rate
- Establish state-of-the-art for social navigation benchmark

## Resource Allocation Recommendation

Based on the phased approach:
- **Phase 1 (2 weeks)**: 1-2 engineers
- **Phase 2 (4 weeks)**: 2-3 engineers  
- **Phase 3 (ongoing)**: 1 engineer for exploration

## Risk Assessment

### Low Risk
- ORCA variants (proven approach, already works well)
- DRL-VO (builds on existing ORCA infrastructure)

### Medium Risk
- MPC approaches (more complex but promising)
- Attention-based RL (similar to HEIGHT but improved)

### High Risk/High Reward
- Foundation model approaches (cutting-edge, potentially transformative)

## Recommendation Summary

| Priority | Method Category | Specific Methods | Time Estimate | Expected Impact |
|----------|-----------------|-------------------|---------------|-----------------|
| HIGH | ORCA Variants | Non-holonomic ORCA, ORCA-DD, Relaxed ORCA | 1-2 weeks | Immediate improvement, 75-85% success |
| HIGH | DRL-VO | DRL-VO (T-RO 2023) | 2-3 weeks | Potential SOTA, 80-90% success |
| MEDIUM | MPC Approaches | DR-MPC, SICNav | 3-4 weeks | Safety guarantees, 85-95% success |
| MEDIUM | Attention RL | Intention Aware, ST2 | 3-5 weeks | HEIGHT replacement, 80-90% success |
| LOW | Foundation Models | VLM-Social-Nav, GSON | 4-6 weeks | Future potential, exploratory |

## Next Steps

1. **Immediate Action**: Implement ORCA variants and establish new baseline
2. **Parallel Track**: Begin DRL-VO implementation
3. **Analysis**: Compare results with HEIGHT failure analysis
4. **Iteration**: Refine approach based on initial findings

This phased approach balances quick wins with longer-term exploration, addressing HEIGHT's specific failure modes while building on methods that have already shown success in the benchmark.