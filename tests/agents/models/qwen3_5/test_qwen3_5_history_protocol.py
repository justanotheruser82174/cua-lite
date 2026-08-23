"""Static validation gate for the Qwen3.5 history protocol default.

These tests deliberately avoid GPU servers, Docker, and real env-server calls.
They live with the model family because the history window is owned by
``Qwen3_5HistoryProtocol``.

Run:
    uv run pytest tests/agents/models/qwen3_5/test_qwen3_5_history_protocol.py -q
"""

from __future__ import annotations

from lite.agents.models.qwen3_5.protocol import Qwen3_5HistoryProtocol


def test_qwen3_5_default_history_contract() -> None:
    protocol = Qwen3_5HistoryProtocol()

    assert protocol.history_n == 50
    assert protocol.image_max == 4
    assert protocol.fold_size == 4
    assert protocol._compute_folded_count(4) == 0
    assert protocol._compute_folded_count(5) == 4
    assert protocol._compute_folded_count(9) == 8
