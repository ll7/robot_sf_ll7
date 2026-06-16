"""Shared forecast variant names for prediction and replay smoke paths."""

FORECAST_VARIANT_CHOICES: tuple[str, ...] = (
    "none",
    "cv",
    "semantic",
    "interaction_aware",
    "risk_filtered",
)

__all__ = ["FORECAST_VARIANT_CHOICES"]
