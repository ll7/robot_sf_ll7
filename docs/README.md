# Robot SF Documentation

Welcome to the Robot SF documentation! This directory contains comprehensive guides and references for using and developing with the Robot SF simulation framework.

## 📚 Documentation Index

### 🏗️ Architecture & Development
- **[Environment Refactoring](./refactoring/)** - **NEW**: Complete guide to the refactored environment architecture
  - [Deployment Status](./refactoring/DEPLOYMENT_READY.md) - Current implementation status
  - [Refactoring Plan](./refactoring/refactoring_plan.md) - Technical architecture details
  - [Migration Guide](./refactoring/migration_guide.md) - Step-by-step migration instructions
  - [Implementation Summary](./refactoring/refactoring_summary.md) - What was accomplished
  - [Migration Report](./refactoring/migration_report.md) - Automated codebase analysis

### 🎮 Simulation & Environment
- [**Simulation View**](./SIM_VIEW.md) - Visualization and rendering system
- [**Map Editor Usage**](./MAP_EDITOR_USAGE.md) - Creating and editing simulation maps
- [**SVG Map Editor**](./SVG_MAP_EDITOR.md) - SVG-based map creation tools

### 📊 Analysis & Tools  
- [**Data Analysis**](./DATA_ANALYSIS.md) - Analysis tools and utilities for simulation data
- [**Pyreverse**](./pyreverse.md) - Code structure visualization

### ⚙️ Setup & Configuration
- [**GPU Setup**](./GPU_SETUP.md) - GPU configuration for accelerated training
- [**UV Migration**](./UV_MIGRATION.md) - Migration to UV package manager

### 📈 Pedestrian Metrics  
- [**Pedestrian Metrics Overview**](./ped_metrics/PED_METRICS.md) - Summary of implemented metrics and their purpose
- [**Metric Analysis**](./ped_metrics/PED_METRICS_ANALYSIS.md) - Overview of metrics used in research and validation
- [**NPC Pedestrian Design**](./ped_metrics/NPC_PEDESTRIAN.md) - Details on the design and behavior of NPC pedestrians

### 📁 Media Resources
- [`img/`](./img/) - Documentation images and diagrams
- [`video/`](./video/) - Demo videos and animations

## 🚀 Quick Start Guides

### New Environment Architecture (Recommended)
```python
# Modern factory pattern for clean environment creation
from robot_sf.gym_env.environment_factory import (
    make_robot_env,
    make_image_robot_env,
    make_pedestrian_env
)

# Create environments with consistent interface
robot_env = make_robot_env(debug=True)
image_env = make_image_robot_env(debug=True)
ped_env = make_pedestrian_env(robot_model=model, debug=True)
```

### Legacy Pattern (Still Supported)
```python
# Traditional approach - still works for backward compatibility
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.gym_env.env_config import EnvSettings

env = RobotEnv(env_config=EnvSettings(), debug=True)
```

## 🎯 Key Features

### Environment System
- **Unified Architecture**: Consistent base classes for all environments
- **Factory Pattern**: Clean, intuitive environment creation
- **Configuration Hierarchy**: Structured, extensible configuration system
- **Backward Compatibility**: Existing code continues to work

### Simulation Capabilities
- **Multi-Agent Support**: Robot and pedestrian simulation
- **Advanced Sensors**: LiDAR, image observations, target sensors
- **Map Integration**: Support for SVG maps and OpenStreetMap data
- **Visualization**: Real-time rendering and video recording

### Training & Analysis
- **Gymnasium Integration**: Compatible with modern RL frameworks
- **StableBaselines3 Support**: Ready for SOTA RL algorithms
- **Data Analysis Tools**: Comprehensive analysis utilities
- **Performance Monitoring**: Built-in metrics and logging

## 📖 Documentation Highlights

### 🆕 Latest Updates
- **Environment Refactoring Complete**: New unified architecture deployed
- **Migration Tools Available**: Automated migration script for updating code
- **Factory Pattern**: Clean, consistent environment creation interface
- **Comprehensive Testing**: All patterns validated and working

### 📋 Migration Status
- **33 files** identified for migration to new pattern
- **Migration script** available for automated updates
- **Backward compatibility** maintained throughout transition
- **Full documentation** provided for smooth migration

## 🔗 External Links

- [**Project Repository**](https://github.com/ll7/robot_sf_ll7) - Main GitHub repository
- [**Gymnasium Documentation**](https://gymnasium.farama.org/) - RL environment framework
- [**StableBaselines3**](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [**PySocialForce**](https://github.com/svenkreiss/PySocialForce) - Pedestrian simulation

## 🤝 Contributing

When contributing to the project:

1. **Use the new factory pattern** for environment creation
2. **Follow the unified configuration system** for settings
3. **Check the migration guide** when updating existing code
4. **Run tests** to ensure compatibility with both old and new patterns

## 📞 Support

- **Environment Issues**: Check the [refactoring documentation](./refactoring/)
- **Migration Help**: Use the [migration guide](./refactoring/migration_guide.md)
- **General Questions**: See individual documentation files
- **Bug Reports**: Use the GitHub issue tracker

---

*Last updated: June 2025 - Environment refactoring complete and deployed*
