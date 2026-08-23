"""REINFORCE rollout entrypoints for RegionFocus grounding.

Same side-effect registration as :mod:`examples.grounding.rollout_grpo`.
"""

from __future__ import annotations

import importlib
from typing import Any

from examples.grounding import adapter  # noqa: F401
from examples.grounding.utils import rf_metrics


def generate(*args: Any, **kwargs: Any) -> Any:
    """Generate RegionFocus samples through the shared REINFORCE entry point."""
    reinforce = importlib.import_module("lite.train.rollout.reinforce")

    return reinforce.generate(*args, **kwargs)


def convert_samples_to_train_data(*args: Any, **kwargs: Any) -> Any:
    """Convert RegionFocus samples through the shared REINFORCE converter."""
    reinforce = importlib.import_module("lite.train.rollout.reinforce")

    return reinforce.convert_samples_to_train_data(*args, **kwargs)


def generate_rollout(args: Any, rollout_id: int, data_source: Any, evaluation: bool = False) -> Any:
    """Wrap the base rollout so RegionFocus judge metrics match GRPO logging."""
    reinforce = importlib.import_module("lite.train.rollout.reinforce")

    result = reinforce.generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
    rf_metrics.drain_and_log_after_rollout(args, evaluation=evaluation)
    return result


__all__ = ["generate", "generate_rollout", "convert_samples_to_train_data"]
