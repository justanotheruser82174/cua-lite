"""Reward equivalence checks for action-batch ``computer`` calls.

The action-batch ``computer{actions:[...]}`` form is reward/eval-neutral:
an eval fn that reads a flat action list scores a per-action ``[click, type]``
sequence identically to the same actions recovered by unpacking a batched
``computer{actions:[click, type]}``.

Status of the real code (verified at authoring time):
  * ``lite/gym/envs/cua/utils.py`` exposes ``unpack``; ``lite.core.tools.action_space``
    exposes ``unpack_action_batch_call``.
    The score-parity proof checks the shipped helper directly.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/gym/envs/cua/test_cua_reward_equivalence.py -p no:cacheprovider -q
"""

from __future__ import annotations

from lite.core.tools import make_tool_call
from lite.core.tools.action_space import unpack_action_batch_call
from lite.gym.envs.cua.utils import unpack

# ---------------------------------------------------------------------------
# Fixtures: the same two actions, in per-action vs action-batch form.
# ---------------------------------------------------------------------------

def _click(x: int, y: int) -> dict:
    return {"name": "click", "arguments": {"coordinate": [x, y]}}


def _type(text: str) -> dict:
    return {"name": "type", "arguments": {"text": text}}


# Per-action sequence: two separate env-lowered calls.
PER_ACTION = [_click(100, 200), _type("hello")]

# Batched sequence (canonical Lite shape): one ``computer`` call whose
# ``actions`` array holds the two env actions.
BATCHED = make_tool_call(
    "computer",
    {"actions": [
        {"action": "click", "coordinate": [100, 200]},
        {"action": "type", "text": "hello"},
    ]},
)


# ---------------------------------------------------------------------------
# A representative "reads the action list" eval fn. Deterministic, pure — the
# kind of scorer screenspot_pro/osworld_g ``_evaluate(actions)`` embody
# (read names + args off the flat list, derive a scalar).
# ---------------------------------------------------------------------------

def _score(actions: list[dict]) -> tuple:
    """Order-sensitive digest of a flat action list via the real ``unpack``."""
    return tuple(unpack(a) for a in actions)


# =============================================================================
# CUA score parity
# =============================================================================

def test_per_action_score_is_stable() -> None:
    """The per-action path scores
    deterministically off the flat action list."""
    assert _score(PER_ACTION) == (
        ("click", {"coordinate": [100, 200]}),
        ("type", {"text": "hello"}),
    )


def test_batched_actions_unpack_to_identical_score() -> None:
    """The shipped unpacker yields the same flat list and the same score."""
    recovered = unpack_action_batch_call(BATCHED)
    assert recovered == PER_ACTION
    assert _score(recovered) == _score(PER_ACTION)
