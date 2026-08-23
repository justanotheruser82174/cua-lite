"""Register the Geo3K example env without importing Slime.

Training launchers and task-export helpers import this module for its
registration side effect before they ask the gym registry about ``geo3k``.
Keeping it Slime-free lets the same registration path run on the host and
inside the training container.
"""

from __future__ import annotations

from examples.geo3k.env import register_geo3k

register_geo3k()

__all__ = ["register_geo3k"]
