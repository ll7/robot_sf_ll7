# Robot SF Utilities

This directory contains utility scripts and tools for the Robot SF project.

## Available Utilities

### 🔄 `migrate_environments.py`
**Environment Migration Tool** - Automated script for migrating from old environment patterns to the new factory-based system.

#### Features:
- **Analyze** existing Python files for migration opportunities
- **Generate reports** showing which files need updates
- **Suggest** specific changes for individual files
- **Automatically migrate** files with backup creation
- **Dry-run mode** to preview changes without applying them

#### Usage Examples:

```bash
# Generate comprehensive migration report
python utilities/migrate_environments.py --report

# Get migration suggestions for a specific file
python utilities/migrate_environments.py --suggest examples/demo_pedestrian.py

# Migrate a specific file (creates backup)
python utilities/migrate_environments.py --migrate examples/demo_pedestrian.py

# Preview changes without applying them
python utilities/migrate_environments.py --migrate examples/demo_pedestrian.py --dry-run

# Analyze specific directories
python utilities/migrate_environments.py --report --directories examples tests scripts
```

#### Migration Patterns:
The script automatically converts:

- **Old imports** → **New factory imports**
  ```python
  # Before
  from robot_sf.gym_env.robot_env import RobotEnv
  from robot_sf.gym_env.env_config import EnvSettings
  
  # After  
  from robot_sf.gym_env.environment_factory import make_robot_env
  from robot_sf.gym_env.unified_config import RobotSimulationConfig
  ```

- **Old environment creation** → **New factory calls**
  ```python
  # Before
  env = RobotEnv(env_config=config, debug=True)
  
  # After
  env = make_robot_env(config=config, debug=True)
  ```

- **Old config classes** → **New unified configs**
  ```python
  # Before
  config = EnvSettings()
  
  # After
  config = RobotSimulationConfig()
  ```

#### Safety Features:
- ✅ **Backup creation** - Original files saved with `.backup` extension
- ✅ **Dry-run mode** - Preview changes before applying
- ✅ **Error handling** - Graceful handling of read/write errors
- ✅ **Pattern matching** - Conservative regex patterns to avoid false positives

#### Related Documentation:
- [Migration Guide](../docs/refactoring/migration_guide.md) - Step-by-step migration instructions
- [Refactoring Summary](../docs/refactoring/refactoring_summary.md) - Overview of the new architecture
- [Deployment Guide](../docs/refactoring/DEPLOYMENT_READY.md) - Production deployment information

---

## Adding New Utilities

When adding new utility scripts to this directory:

1. **Follow naming conventions**: Use descriptive, snake_case names
2. **Add documentation**: Include a docstring and usage examples
3. **Update this README**: Add your utility to the list above
4. **Consider error handling**: Make scripts robust and user-friendly
5. **Add to .gitignore if needed**: For generated files or temporary outputs

## Development Notes

- All utilities should be runnable from the project root directory
- Use relative imports when possible
- Include type hints for better code quality
- Add command-line argument parsing for better usability
