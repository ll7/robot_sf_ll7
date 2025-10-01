"""Docs helper catalog for managing documentation index updates.

This module provides helper functions for programmatically updating
documentation indices and maintaining helper documentation.
"""

from pathlib import Path

from loguru import logger


def register_helper(name: str, summary: str, doc_path: str) -> None:
    """Update central docs index programmatically with helper information.

    This function ensures helper entries are deduplicated and properly
    linked in the main documentation index.

    Args:
        name: Name of the helper function or capability
        summary: Brief description of what the helper does
        doc_path: Relative path to detailed documentation

    Raises:
        FileNotFoundError: If the docs index file doesn't exist
        OSError: If the docs index cannot be updated
    """
    docs_readme_path = Path("docs/README.md")

    if not docs_readme_path.exists():
        logger.error(f"Documentation index not found: {docs_readme_path}")
        raise FileNotFoundError(f"Documentation index not found: {docs_readme_path}")

    try:
        # Read existing content
        with docs_readme_path.open("r", encoding="utf-8") as f:
            content = f.read()

        # Create helper entry
        helper_entry = f"- **{name}**: {summary} ([details]({doc_path}))"

        # Check if helper is already documented
        if name in content:
            logger.debug(f"Helper '{name}' already exists in documentation index")
            return

        # Find appropriate section or create one
        helper_section_marker = "## Helper Catalog"
        if helper_section_marker not in content:
            # Add helper catalog section at the end
            content += f"\n\n{helper_section_marker}\n\n{helper_entry}\n"
            logger.info(f"Created new helper catalog section and added '{name}'")
        else:
            # Insert in existing section
            lines = content.split("\n")
            section_found = False
            for i, line in enumerate(lines):
                if line.strip() == helper_section_marker:
                    section_found = True
                    # Insert after the section header
                    lines.insert(i + 2, helper_entry)
                    break

            if section_found:
                content = "\n".join(lines)
                logger.info(f"Added helper '{name}' to existing catalog section")
            else:
                # Fallback: append to end
                content += f"\n{helper_entry}\n"
                logger.warning(f"Could not find helper section, appended '{name}' to end")

        # Write updated content
        with docs_readme_path.open("w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Successfully registered helper '{name}' in documentation index")

    except OSError as e:
        logger.error(f"Failed to update documentation index: {e}")
        raise OSError(f"Could not update docs index: {e}") from e
