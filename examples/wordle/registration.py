"""Register the Wordle example env without importing Slime.

Training launchers and task-export helpers import this module for its
registration side effect before they ask the gym registry about ``wordle``.
Keeping it Slime-free lets the same registration path run on the host and
inside the training container.
"""

from __future__ import annotations

from examples.wordle.env import register_wordle

register_wordle()

__all__ = ["register_wordle"]
