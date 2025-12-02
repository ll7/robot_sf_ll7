#!/usr/bin/env python3
"""
Migration script for Robot SF environment refactoring.

This script helps automate the migration from old environment patterns
to the new factory-based system. It can analyze existing code and
suggest or make changes automatically.
"""

import argparse
import re
from pathlib import Path


class EnvironmentMigrator:
    """Automates migration to new environment patterns."""

    def __init__(self, project_root: str):
        """Init.

        Args:
            project_root: Auto-generated placeholder description.

        Returns:
            Any: Auto-generated placeholder description.
        """
        self.project_root = Path(project_root)
        self.changes_made = []

    def find_python_files(self, directories: list[str] | None = None) -> list[Path]:
        """Find all Python files in specified directories."""
        if directories is None:
            directories = ["examples", "tests", "scripts"]

        python_files = []
        for directory in directories:
            dir_path = self.project_root / directory
            if dir_path.exists():
                python_files.extend(dir_path.glob("**/*.py"))
        return python_files

    def analyze_file(self, file_path: Path) -> dict:
        """Analyze a file for migration opportunities."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except (FileNotFoundError, OSError, UnicodeDecodeError) as e:
            return {"error": str(e)}

        analysis = {
            "file": str(file_path),
            "old_imports": [],
            "old_env_creation": [],
            "old_config_usage": [],
            "recommendations": [],
        }

        # Analyze different patterns
        self._check_old_imports(content, analysis)
        self._check_old_env_creation(content, analysis)
        self._check_old_config_usage(content, analysis)
        self._generate_recommendations(content, analysis)

        return analysis

    def _check_old_imports(self, content: str, analysis: dict) -> None:
        """Check for old import patterns."""
        old_import_patterns = [
            r"from robot_sf\.gym_env\.robot_env import RobotEnv",
            r"from robot_sf\.gym_env\.pedestrian_env import PedestrianEnv",
            r"from robot_sf\.gym_env\.env_config import (EnvSettings|RobotEnvSettings|PedEnvSettings)",
        ]

        for pattern in old_import_patterns:
            matches = re.findall(pattern, content)
            if matches:
                analysis["old_imports"].extend(matches)

    def _check_old_env_creation(self, content: str, analysis: dict) -> None:
        """Check for old environment creation patterns."""
        old_creation_patterns = [
            r"RobotEnv\s*\(",
            r"PedestrianEnv\s*\(",
            r"RobotEnvWithImage\s*\(",
        ]

        for pattern in old_creation_patterns:
            matches = re.findall(pattern, content)
            if matches:
                analysis["old_env_creation"].extend(matches)

    def _check_old_config_usage(self, content: str, analysis: dict) -> None:
        """Check for old config usage patterns."""
        old_config_patterns = [
            r"EnvSettings\s*\(",
            r"RobotEnvSettings\s*\(",
            r"PedEnvSettings\s*\(",
        ]

        for pattern in old_config_patterns:
            matches = re.findall(pattern, content)
            if matches:
                analysis["old_config_usage"].extend(matches)

    def _generate_recommendations(self, content: str, analysis: dict) -> None:
        """Generate migration recommendations."""
        if analysis["old_imports"] or analysis["old_env_creation"] or analysis["old_config_usage"]:
            analysis["recommendations"].append("Consider migrating to factory pattern")

        if "RobotEnv(" in content:
            analysis["recommendations"].append("Replace RobotEnv() with make_robot_env()")

        if "PedestrianEnv(" in content:
            analysis["recommendations"].append("Replace PedestrianEnv() with make_pedestrian_env()")

        if "EnvSettings(" in content:
            analysis["recommendations"].append("Replace EnvSettings with RobotSimulationConfig")

    def suggest_migration(self, file_path: Path) -> str:
        """Generate migration suggestions for a file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            return "Could not read file"

        suggestions = []

        # Suggest import replacements
        if "from robot_sf.gym_env.robot_env import RobotEnv" in content:
            suggestions.append(
                "Replace:\n"
                "  from robot_sf.gym_env.robot_env import RobotEnv\n"
                "With:\n"
                "  from robot_sf.gym_env.environment_factory import make_robot_env",
            )

        if "from robot_sf.gym_env.env_config import" in content:
            suggestions.append(
                "Replace:\n"
                "  from robot_sf.gym_env.env_config import EnvSettings\n"
                "With:\n"
                "  from robot_sf.gym_env.unified_config import RobotSimulationConfig",
            )

        # Suggest environment creation replacements
        if "RobotEnv(" in content:
            suggestions.append(
                "Replace:\n"
                "  env = RobotEnv(env_config=config, debug=True)\n"
                "With:\n"
                "  env = make_robot_env(config=config, debug=True)",
            )

        if "PedestrianEnv(" in content:
            suggestions.append(
                "Replace:\n"
                "  env = PedestrianEnv(env_config=config, robot_model=model)\n"
                "With:\n"
                "  env = make_pedestrian_env(config=config, robot_model=model)",
            )

        return "\n\n".join(suggestions) if suggestions else "No migration needed"

    def create_migrated_version(self, file_path: Path, dry_run: bool = True) -> str:
        """Create a migrated version of the file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except (FileNotFoundError, OSError, UnicodeDecodeError) as e:
            return f"Error reading file: {e}"

        original_content = content

        # Replace imports
        replacements = [
            # Import replacements
            (
                r"from robot_sf\.gym_env\.robot_env import RobotEnv",
                "from robot_sf.gym_env.environment_factory import make_robot_env",
            ),
            (
                r"from robot_sf\.gym_env\.pedestrian_env import PedestrianEnv",
                "from robot_sf.gym_env.environment_factory import make_pedestrian_env",
            ),
            (
                r"from robot_sf\.gym_env\.env_config import EnvSettings",
                "from robot_sf.gym_env.unified_config import RobotSimulationConfig",
            ),
            (
                r"from robot_sf\.gym_env\.env_config import RobotEnvSettings",
                "from robot_sf.gym_env.unified_config import ImageRobotConfig",
            ),
            (
                r"from robot_sf\.gym_env\.env_config import PedEnvSettings",
                "from robot_sf.gym_env.unified_config import PedestrianSimulationConfig",
            ),
            # Environment creation replacements
            (r"RobotEnv\s*\(", "make_robot_env("),
            (r"PedestrianEnv\s*\(", "make_pedestrian_env("),
            # Config class replacements
            (r"EnvSettings\s*\(", "RobotSimulationConfig("),
            (r"RobotEnvSettings\s*\(", "ImageRobotConfig("),
            (r"PedEnvSettings\s*\(", "PedestrianSimulationConfig("),
        ]

        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)

        if content != original_content:
            if not dry_run:
                try:
                    # Create backup
                    backup_path = file_path.with_suffix(file_path.suffix + ".backup")
                    with open(backup_path, "w", encoding="utf-8") as f:
                        f.write(original_content)

                    # Write migrated version
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)

                    self.changes_made.append(str(file_path))
                    return f"Migrated {file_path} (backup created)"
                except (OSError, UnicodeEncodeError) as e:
                    return f"Error writing file {file_path}: {e}"
            else:
                return f"Would migrate {file_path}"
        else:
            return f"No changes needed for {file_path}"

    def generate_migration_report(self, directories: list[str] | None = None) -> str:
        """Generate a comprehensive migration report."""
        files = self.find_python_files(directories)

        from datetime import datetime

        current_date = datetime.now().strftime("%B %d, %Y")

        report = ["# Migration Report for Robot SF Environment Refactoring\n\n"]
        report.append(
            "> 📚 **Documentation Navigation**: [← Back to Refactoring Index](README.md) | [🚀 Deployment Status](DEPLOYMENT_READY.md) | [📋 Plan](refactoring_plan.md) | [🔄 Migration Guide](migration_guide.md) | [📊 Summary](refactoring_summary.md)\n",
        )
        report.append("> \n")
        report.append(
            f"> 🔧 **Generated by**: [`utilities/migrate_environments.py`](../../utilities/migrate_environments.py) | **Last updated**: {current_date}\n\n",
        )
        report.append(f"Analyzed {len(files)} Python files\n")

        needs_migration = []
        no_migration_needed = []

        for file_path in files:
            analysis = self.analyze_file(file_path)

            if analysis.get("error"):
                report.append(f"ERROR analyzing {file_path}: {analysis['error']}\n")
                continue

            if (
                analysis["old_imports"]
                or analysis["old_env_creation"]
                or analysis["old_config_usage"]
            ):
                needs_migration.append(file_path)
            else:
                no_migration_needed.append(file_path)

        report.append(f"## Files needing migration: {len(needs_migration)}\n")
        for file_path in needs_migration:
            analysis = self.analyze_file(file_path)
            report.append(f"### {file_path}\n")
            if analysis["old_imports"]:
                report.append(f"- Old imports: {analysis['old_imports']}\n")
            if analysis["old_env_creation"]:
                report.append(f"- Old environment creation: {analysis['old_env_creation']}\n")
            if analysis["old_config_usage"]:
                report.append(f"- Old config usage: {analysis['old_config_usage']}\n")
            report.append(f"- Recommendations: {', '.join(analysis['recommendations'])}\n")
            report.append("\n")

        report.append(f"## Files already up to date: {len(no_migration_needed)}\n")
        for file_path in no_migration_needed:
            report.append(f"- {file_path}\n")

        return "".join(report)


def main():
    """Main.

    Returns:
        Any: Auto-generated placeholder description.
    """
    parser = argparse.ArgumentParser(
        description="Migrate Robot SF environments to new factory pattern",
    )
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument(
        "--directories",
        nargs="+",
        default=["examples", "tests", "scripts"],
        help="Directories to analyze",
    )
    parser.add_argument("--report", action="store_true", help="Generate migration report")
    parser.add_argument("--suggest", help="Show migration suggestions for specific file")
    parser.add_argument("--migrate", help="Migrate specific file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes",
    )

    args = parser.parse_args()

    migrator = EnvironmentMigrator(args.project_root)

    if args.report:
        report = migrator.generate_migration_report(args.directories)

        # Save report to file
        report_path = Path(args.project_root) / "migration_report.md"
        try:
            with open(report_path, "w") as f:
                f.write(report)
        except (OSError, UnicodeEncodeError):
            pass

    elif args.suggest:
        file_path = Path(args.suggest)
        migrator.suggest_migration(file_path)

    elif args.migrate:
        file_path = Path(args.migrate)
        migrator.create_migrated_version(file_path, dry_run=args.dry_run)

    else:
        pass


if __name__ == "__main__":
    main()
