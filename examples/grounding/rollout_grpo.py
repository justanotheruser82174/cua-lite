"""GRPO rollout entrypoints for RegionFocus grounding.

Side-effect-imports :mod:`examples.grounding.adapter` so the custom
RegionFocus grounding adapter+agent register before slime resolves the
``agent_id`` from ``--custom-config-path``.

Usage:
  --custom-generate-function-path examples.grounding.rollout_grpo.generate
  --custom-convert-samples-to-train-data-path \
      examples.grounding.rollout_grpo.convert_samples_to_train_data
  --rollout-function-path examples.grounding.rollout_grpo.generate_rollout
"""

from __future__ import annotations

import importlib
from typing import Any

from examples.grounding import adapter  # noqa: F401
from examples.grounding.utils import rf_metrics


def generate(*args: Any, **kwargs: Any) -> Any:
    """Generate RegionFocus samples through the shared GRPO entry point."""
    grpo = importlib.import_module("lite.train.rollout.grpo")

    return grpo.generate(*args, **kwargs)


def convert_samples_to_train_data(*args: Any, **kwargs: Any) -> Any:
    """Convert RegionFocus samples through the shared GRPO converter."""
    grpo = importlib.import_module("lite.train.rollout.grpo")

    return grpo.convert_samples_to_train_data(*args, **kwargs)


def generate_rollout(args: Any, rollout_id: int, data_source: Any, evaluation: bool = False) -> Any:
    """Wrap the base rollout to drain and log judge metrics after eval batches."""
    grpo = importlib.import_module("lite.train.rollout.grpo")

    result = grpo.generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
    rf_metrics.drain_and_log_after_rollout(args, evaluation=evaluation)
    return result


__all__ = ["generate", "generate_rollout", "convert_samples_to_train_data"]
