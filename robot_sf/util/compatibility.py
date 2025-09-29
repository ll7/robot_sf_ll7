"""Python compatibility utilities."""


def validate_compatibility(version: str) -> bool:
    """Validate if the given Python version is compatible.

    Args:
        version: Python version string (e.g., "3.11", "3.12")

    Returns:
        True if the version is supported (3.11+), False otherwise
    """
    try:
        major, minor = map(int, version.split(".")[:2])
        return (major == 3 and minor >= 11) or major > 3
    except (ValueError, IndexError):
        return False
