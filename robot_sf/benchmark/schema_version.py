"""
SchemaVersion entity for semantic versioning of schemas.

This module provides the SchemaVersion entity that handles semantic versioning
for schema evolution, compatibility checking, and version parsing.
"""

import re


class SchemaVersion:
    """
    Entity representing a semantic version for schema evolution.

    This class implements semantic versioning (semver.org) for schemas,
    providing parsing, comparison, and compatibility checking functionality.
    """

    # Regex pattern for semantic versioning
    SEMVER_PATTERN = re.compile(
        r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
        r"(?:-(?P<prerelease>[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*))?"
        r"(?:\+(?P<build>[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*))?$",
    )

    def __init__(
        self,
        major: int,
        minor: int,
        patch: int,
        prerelease: str | None = None,
        build: str | None = None,
    ):
        """
        Initialize SchemaVersion.

        Args:
            major: Major version number (>= 0)
            minor: Minor version number (>= 0)
            patch: Patch version number (>= 0)
            prerelease: Optional prerelease identifier
            build: Optional build metadata

        Raises:
            ValueError: If version components are invalid
        """
        if major < 0 or minor < 0 or patch < 0:
            raise ValueError("Version numbers must be non-negative integers")

        self.major = major
        self.minor = minor
        self.patch = patch
        self.prerelease = prerelease
        self.build = build

    @classmethod
    def parse(cls, version_string: str) -> "SchemaVersion":
        """
        Parse a version string into a SchemaVersion instance.

        Args:
            version_string: Version string (e.g., "1.2.3", "2.0.0-alpha.1")

        Returns:
            Parsed SchemaVersion instance

        Raises:
            ValueError: If version string is invalid
        """
        if not version_string:
            raise ValueError("Version string cannot be empty")

        match = cls.SEMVER_PATTERN.match(version_string.strip())
        if not match:
            raise ValueError(f"Invalid semantic version format: {version_string}")

        groups = match.groupdict()
        return cls(
            major=int(groups["major"]),
            minor=int(groups["minor"]),
            patch=int(groups["patch"]),
            prerelease=groups.get("prerelease"),
            build=groups.get("build"),
        )

    @classmethod
    def from_string(cls, version_string: str) -> "SchemaVersion":
        """Alias for parse() for backward compatibility.

        Returns:
            SchemaVersion instance parsed from the version string.
        """
        return cls.parse(version_string)

    def __str__(self) -> str:
        """String representation of the version.

        Returns:
            Semantic version string (e.g., '1.2.3' or '1.2.3-beta+build.123').
        """
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __repr__(self) -> str:
        """Detailed string representation.

        Returns:
            Constructor-style representation showing all version components.
        """
        return f"SchemaVersion(major={self.major}, minor={self.minor}, patch={self.patch}, prerelease={self.prerelease!r}, build={self.build!r})"

    def __eq__(self, other: object) -> bool:
        """Check equality with another SchemaVersion.

        Returns:
            True if all version components match.
        """
        if not isinstance(other, SchemaVersion):
            return False
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.prerelease == other.prerelease
            and self.build == other.build
        )

    def __lt__(self, other: "SchemaVersion") -> bool:
        """Compare versions for ordering.

        Returns:
            True if this version is semantically less than the other version.
        """
        if not isinstance(other, SchemaVersion):
            return NotImplemented

        # Compare major, minor, patch
        self_tuple = (self.major, self.minor, self.patch)
        other_tuple = (other.major, other.minor, other.patch)

        if self_tuple != other_tuple:
            return self_tuple < other_tuple

        # If versions are equal, compare prerelease
        # Prerelease versions have lower precedence than normal versions
        if self.prerelease and not other.prerelease:
            return True
        elif not self.prerelease and other.prerelease:
            return False
        elif self.prerelease and other.prerelease:
            return self.prerelease < other.prerelease

        # Build metadata doesn't affect precedence
        return False

    def __le__(self, other: "SchemaVersion") -> bool:
        """TODO docstring. Document this function.

        Args:
            other: TODO docstring.

        Returns:
            TODO docstring.
        """
        return self < other or self == other

    def __gt__(self, other: "SchemaVersion") -> bool:
        """TODO docstring. Document this function.

        Args:
            other: TODO docstring.

        Returns:
            TODO docstring.
        """
        return not (self <= other)

    def __ge__(self, other: "SchemaVersion") -> bool:
        """TODO docstring. Document this function.

        Args:
            other: TODO docstring.

        Returns:
            TODO docstring.
        """
        return not (self < other)

    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries.

        Returns:
            Hash value computed from all version components.
        """
        return hash((self.major, self.minor, self.patch, self.prerelease, self.build))

    def is_compatible_with(self, other: "SchemaVersion") -> bool:
        """
        Check if this version is backward compatible with another version.

        In semantic versioning:
        - Major version changes indicate breaking changes
        - Minor and patch changes are backward compatible

        Args:
            other: Version to check compatibility against

        Returns:
            True if this version is backward compatible with other
        """
        # Same major version and this version >= other version
        return self.major == other.major and self >= other

    def next_major(self) -> "SchemaVersion":
        """Get the next major version (resets minor and patch).

        Returns:
            New SchemaVersion with incremented major and zero minor/patch.
        """
        return SchemaVersion(self.major + 1, 0, 0)

    def next_minor(self) -> "SchemaVersion":
        """Get the next minor version (resets patch).

        Returns:
            New SchemaVersion with incremented minor and zero patch.
        """
        return SchemaVersion(self.major, self.minor + 1, 0)

    def next_patch(self) -> "SchemaVersion":
        """Get the next patch version.

        Returns:
            New SchemaVersion with incremented patch.
        """
        return SchemaVersion(self.major, self.minor, self.patch + 1)

    @property
    def is_prerelease(self) -> bool:
        """Check if this is a prerelease version.

        Returns:
            True if prerelease component is present.
        """
        return self.prerelease is not None

    @property
    def is_stable(self) -> bool:
        """Check if this is a stable (non-prerelease) version."""
        return not self.is_prerelease

    @property
    def tuple(self) -> tuple[int, int, int]:
        """Get version as a tuple (major, minor, patch)."""
        return (self.major, self.minor, self.patch)

    def bump(self, bump_type: str) -> "SchemaVersion":
        """
        Bump the version according to the specified type.

        Args:
            bump_type: Type of bump ('major', 'minor', or 'patch')

        Returns:
            New SchemaVersion with bumped version

        Raises:
            ValueError: If bump_type is invalid
        """
        if bump_type == "major":
            return self.next_major()
        elif bump_type == "minor":
            return self.next_minor()
        elif bump_type == "patch":
            return self.next_patch()
        else:
            raise ValueError(
                f"Invalid bump type: {bump_type}. Must be 'major', 'minor', or 'patch'",
            )
