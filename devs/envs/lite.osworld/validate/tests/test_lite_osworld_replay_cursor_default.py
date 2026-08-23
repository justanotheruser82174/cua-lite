from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_replay_module():
    replay_path = Path(__file__).resolve().parents[1] / "rollout" / "replay_trajectory.py"
    spec = importlib.util.spec_from_file_location(
        "_cursor_replay_trajectory",
        replay_path,
    )
    assert spec is not None and spec.loader is not None
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    return replay


def test_lite_osworld_replay_drops_public_cursor_keys_for_bare_sandbox():
    replay = _load_replay_module()

    assert "cursor_overlay" not in replay.BARE_REPLAY_ENV_KEYS
    assert "cursor" not in replay.BARE_REPLAY_ENV_KEYS

    normalized = replay._normalize_replay_env_kwargs_for_bare_sandbox(
        {
            "computer": {
                "display_resolution": [1024, 768],
                "image": "cua-lite/lite.osworld:test",
            },
            "cursor_overlay": True,
            "cursor": False,
            "max_steps": 3,
        }
    )

    assert normalized == {
        "display_resolution": (1024, 768),
        "image": "cua-lite/lite.osworld:test",
        "max_steps": 3,
    }
