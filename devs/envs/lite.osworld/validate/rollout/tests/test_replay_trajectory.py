"""Tests for lite.osworld rollout replay validation helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


class TestReplayTrajectoryConfig:
    """Static tests for the rollout replay helper script."""

    @staticmethod
    def _load_replay_module():
        repo = Path(__file__).resolve().parents[6]
        path = repo / "devs/envs/lite.osworld/validate/rollout/replay_trajectory.py"
        spec = importlib.util.spec_from_file_location(
            "_lite_osworld_replay_trajectory_under_test",
            path,
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_empty_config_path_has_no_env_kwargs(self):
        mod = self._load_replay_module()

        assert mod._load_env_kwargs_from_config(None) == {}
        assert mod._merge_replay_env_kwargs(None, "multi_apps", None) == {}
        assert mod._effective_replay_image({}) == mod.DEFAULT_REPLAY_IMAGE

    def test_replay_config_path_defaults_to_private_env_config(self):
        mod = self._load_replay_module()

        assert (
            mod._resolve_replay_config_path(
                None,
                {"LITE_OSWORLD_CONFIG": ".tmp/osworld.private.yaml"},
            )
            == ".tmp/osworld.private.yaml"
        )

    def test_replay_config_path_cli_override_and_empty_string_opt_out(self):
        mod = self._load_replay_module()
        env = {"LITE_OSWORLD_CONFIG": ".tmp/osworld.private.yaml"}

        assert mod._resolve_replay_config_path("custom.yaml", env) == "custom.yaml"
        assert mod._resolve_replay_config_path("", env) is None
        assert mod._resolve_replay_config_path(None, {}) == mod.DEFAULT_REPLAY_CONFIG_PATH

    def test_replay_main_passes_env_resolved_config_path_to_run(
        self,
        monkeypatch,
        tmp_path,
    ):
        mod = self._load_replay_module()
        seen = {}

        async def fake_run(task_id, sample_dir, spec, max_turns, **kwargs):
            seen.update(
                task_id=task_id,
                sample_dir=sample_dir,
                spec=spec,
                max_turns=max_turns,
                **kwargs,
            )

        monkeypatch.setattr(
            mod,
            "_load_task",
            lambda task_id: {
                "instruction": "diagnose replay config",
                "metadata": {"others": {"domain": "multi_apps"}},
            },
        )
        monkeypatch.setattr(mod, "_find_rollout_dir", lambda task_id: tmp_path)
        monkeypatch.setattr(mod, "_run", fake_run)
        monkeypatch.setenv("LITE_OSWORLD_CONFIG", ".tmp/osworld.private.yaml")
        monkeypatch.setattr(sys, "argv", ["replay_trajectory.py", "task-id"])

        mod.main()

        assert seen["task_id"] == "task-id"
        assert seen["sample_dir"] == tmp_path
        assert seen["config_path"] == ".tmp/osworld.private.yaml"

    def test_replay_forces_local_direct_mode(self, monkeypatch, capsys):
        mod = self._load_replay_module()
        monkeypatch.setenv("CUA_LITE_ENV_SERVER_URL", "http://remote.invalid")
        monkeypatch.setenv("CUA_LITE_ENV_SERVER_TOKEN", "secret")

        assert mod._force_direct_local_replay() is True

        assert "CUA_LITE_ENV_SERVER_URL" not in mod.os.environ
        assert "CUA_LITE_ENV_SERVER_TOKEN" not in mod.os.environ
        assert "ignoring CUA_LITE_ENV_SERVER_URL/TOKEN" in capsys.readouterr().out

    def test_replay_local_direct_mode_is_quiet_without_remote_env(
        self,
        monkeypatch,
        capsys,
    ):
        mod = self._load_replay_module()
        monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
        monkeypatch.delenv("CUA_LITE_ENV_SERVER_TOKEN", raising=False)

        assert mod._force_direct_local_replay() is False

        assert capsys.readouterr().out == ""

    def test_replay_env_kwargs_merge_applies_private_image_and_domain_override(
        self,
        tmp_path,
    ):
        mod = self._load_replay_module()
        config = tmp_path / "rollout.yaml"
        config.write_text(
            "\n".join(
                [
                    "env_kwargs:",
                    "  max_steps: 15",
                    "  resolution: [1024, 768]",
                    "  loop_detect: 0",
                    "  domain_overrides:",
                    "    multi_apps:",
                    "      max_steps: 30",
                ]
            )
        )

        env_kwargs = mod._merge_replay_env_kwargs(
            str(config),
            "multi_apps",
            {
                "image": "cua-lite/lite.osworld:mine",
                "resolution": [1920, 1080],
                "noise": True,
                "extra_tools": ["report_infeasible"],
            },
        )
        replay_image = mod._effective_replay_image(env_kwargs)

        assert env_kwargs == {
            "max_steps": 30,
            "display_resolution": (1920, 1080),
            "loop_detect": 0,
            "image": "cua-lite/lite.osworld:mine",
        }
        assert replay_image == "cua-lite/lite.osworld:mine"
        assert mod._replay_computer_config(replay_image)["image"] == replay_image

    def test_replay_cli_domain_override_wins_and_computer_image_is_supported(
        self,
        tmp_path,
    ):
        mod = self._load_replay_module()
        config = tmp_path / "rollout.yaml"
        config.write_text(
            "\n".join(
                [
                    "env_kwargs:",
                    "  max_steps: 15",
                    "  computer:",
                    "    image: cua-lite/lite.osworld:latest",
                    "    display_resolution: [1920, 1080]",
                    "    memory: 4GB",
                    "  domain_overrides:",
                    "    multi_apps:",
                    "      max_steps: 30",
                ]
            )
        )

        env_kwargs = mod._merge_replay_env_kwargs(
            str(config),
            "multi_apps",
            {
                "computer": {"image": "cua-lite/lite.osworld:mine"},
                "domain_overrides": {"multi_apps": {"max_steps": 99}},
            },
        )

        assert env_kwargs["max_steps"] == 99
        assert env_kwargs["display_resolution"] == (1920, 1080)
        assert env_kwargs["image"] == "cua-lite/lite.osworld:mine"
        assert "computer" not in env_kwargs
        assert "computer_config" not in env_kwargs
        assert "domain_overrides" not in env_kwargs
        assert mod._effective_replay_image(env_kwargs) == "cua-lite/lite.osworld:mine"

    def test_load_actions_reads_nested_canonical_calls(self, tmp_path):
        from lite.core.tools import make_tool_call

        mod = self._load_replay_module()
        skipped = tmp_path / "turn_0000"
        skipped.mkdir()
        (skipped / "03_actions.json").write_text(
            json.dumps({"lite_message": {"tool_calls": []}}),
            encoding="utf-8",
        )
        turn = tmp_path / "turn_0001"
        turn.mkdir()
        call = make_tool_call(
            "click",
            {"coordinate": [100, 200]},
            call_id="call_click",
        )
        (turn / "03_actions.json").write_text(
            json.dumps({"lite_message": {"tool_calls": [call]}}),
            encoding="utf-8",
        )

        assert mod._load_actions(tmp_path, None) == [[call]]

    def test_load_actions_rejects_legacy_flat_calls(self, tmp_path):
        mod = self._load_replay_module()
        turn = tmp_path / "turn_0000"
        turn.mkdir()
        (turn / "03_actions.json").write_text(
            json.dumps(
                {
                    "lite_message": {
                        "tool_calls": [{"name": "click", "arguments": {}}],
                    },
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="bare model-function projection"):
            mod._load_actions(tmp_path, None)

    def test_tool_call_names_for_display_expands_action_batch_children_without_mutating(self):
        from lite.core.tools import make_tool_call

        mod = self._load_replay_module()
        call = make_tool_call(
            "computer",
            {
                "actions": [
                    {"action": "click", "coordinate": [1, 2]},
                    {"action": "type", "text": "hi"},
                ]
            },
            call_id="call_0000",
        )
        calls = [call]

        assert mod._tool_call_names_for_display(calls) == ["click", "type"]
        assert calls == [call]
