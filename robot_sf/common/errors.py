"""Error handling policy helpers.

Provides consistent error raising and warning patterns across the codebase.
"""

from loguru import logger


def raise_fatal_with_remedy(msg: str, remedy: str) -> None:
    """Raise a RuntimeError with an actionable remediation message.

    Parameters
    ----------
    msg : str
        Primary error description
    remedy : str
        Concrete steps to fix the issue
    """
    full_msg = f"{msg}\n\nRemediation: {remedy}"
    raise RuntimeError(full_msg)


def warn_soft_degrade(component: str, issue: str, fallback: str) -> None:
    """Log a warning for optional component failures with soft degradation.

    Parameters
    ----------
    component : str
        Name of the optional component
    issue : str
        Description of what failed
    fallback : str
        What behavior will occur instead
    """
    logger.warning(
        "Optional component '{}' issue: {}. Fallback: {}",
        component,
        issue,
        fallback,
    )
