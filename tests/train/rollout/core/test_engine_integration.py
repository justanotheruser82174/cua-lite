"""Integration tests for lite.train.rollout.core.engine.generate().

Tests the full generate() flow with mocked SGLang server and real CUA-lite
agent/adapter/protocol stack. Verifies:
  - Prompt/response consistency with inference (Step 0 verification)
  - Token ID / log-prob alignment
  - on_step callback → Sample building
  - Abort handling
  - Env fault isolation

Usage:
    pytest tests/train/rollout/core/test_engine_integration.py -v -s
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("slime.utils.types", reason="slime not installed")
from slime.utils.types import Sample

import lite.train.rollout.core.engine as engine
from lite.agents.core.adapter import AGENT_KWARGS_TOOL_SURFACE_KEYS
from lite.core.tools.calls import make_tool_call, tool_call_id
from lite.core.tools.results import LiteToolResult
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult
from lite.train.rollout.core.engine import (
    _active_envs,
    generate,
)

# ---------------------------------------------------------------------------
# Fixtures / Mocks
# ---------------------------------------------------------------------------

class FakeGenerateState:
    """Minimal mock of slime's GenerateState singleton."""
    _instance = None

    def __new__(cls, args=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, args=None):
        if self._initialized:
            return
        self._initialized = True
        self.args = args
        self.tokenizer = MagicMock()
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.processor = None  # no processor → use tokenizer
        self.aborted = False
        self.remaining_batch_size = 0
        self.pendings = set()
        self.semaphore = asyncio.Semaphore(64)

    def reset(self):
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    @classmethod
    def teardown(cls):
        cls._instance = None

class FakeEnvMetaData:
    dims = ("desktop", "use")
    platform = "desktop"
    task_type = "use"
    extra_tools = []
    valid_actions = None

class FakeEnv:
    """Mock CUA-Lite environment."""
    def __init__(self, n_steps=3, terminal_reward=1.0, fail_on_step=None):
        self._n_steps = n_steps
        self._terminal_reward = terminal_reward
        self._fail_on_step = fail_on_step
        self._step_count = 0
        self._closed = False
        self.metadata = FakeEnvMetaData()

    async def reset(self):
        self._step_count = 0
        return LiteEnvObservation(image=b"iVBORw0KGgo=", text="Click the search button")

    async def step(self, actions):
        self._step_count += 1
        if self._fail_on_step is not None and self._step_count == self._fail_on_step:
            raise RuntimeError("Simulated env failure")
        is_terminal = self._step_count >= self._n_steps
        return LiteEnvStepResult(
            reward=self._terminal_reward if is_terminal else 0.0,
            terminated=is_terminal,
            results=[] if is_terminal else [
                LiteToolResult(tool_call_id=tool_call_id(action), images=[b"iVBORw0KGgo="])
                for action in actions
            ],
        )

    async def close(self):
        self._closed = True


class CloseRaisesEnv(FakeEnv):
    async def close(self):
        self._closed = True
        raise RuntimeError("engine close exploded")


class FakeAgent:
    """Mock agent that captures generate_fn calls."""
    def __init__(self, generate_fn, n_steps=3, terminal_reward=1.0):
        self._generate_fn = generate_fn
        self._n_steps = n_steps
        self._terminal_reward = terminal_reward

    async def predict(self, sample):
        pass

    async def sample(self, env, max_steps=100, hooks=None):
        from lite.core import LiteCUAMetadata, LiteSample
        from lite.core.samples import STATUS_COMPLETED, STATUS_TRUNCATED, LiteRLSample, LiteRLStep

        await env.reset()
        steps: list[LiteRLStep] = []
        episode_return = 0.0
        terminated = False
        truncated = False

        for step in range(min(max_steps, self._n_steps)):
            prompt = f"<prompt_step_{step}>"
            result = await self._generate_fn(prompt=prompt, images=[])
            response = result["response"]

            step_result = await env.step(
                [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": [100, 200]}]},
                        call_id="call_0000",
                    )
                ]
            )
            reward = step_result.reward
            terminated = step_result.terminated
            truncated = step_result.truncated
            if reward is not None:
                episode_return += reward

            rl_step = LiteRLStep(
                prompt=prompt,
                image_indices=(),
                response=response,
                response_tokens=list(result.get("response_tokens") or []),
                response_log_probs=list(result.get("response_log_probs") or []),
                reward=float(reward) if reward is not None else 0.0,
                status=STATUS_TRUNCATED if truncated else STATUS_COMPLETED,
            )
            steps.append(rl_step)

            if terminated or truncated:
                break

        await env.close()
        return LiteRLSample(
            processed_images=[],
            steps=steps,
            lite_sample=LiteSample(metadata=LiteCUAMetadata(dims=("desktop", "use"))),
            episode_return=episode_return,
            terminated=terminated,
            truncated=truncated,
        )

# Step counter for tracking SGLang calls
_sglang_calls = []

async def fake_post(url, payload):
    """Mock SGLang HTTP post."""
    _sglang_calls.append({"url": url, "payload": payload})

    # Simulate SGLang response
    return {
        "text": "Action: click(100, 200)",
        "meta_info": {
            "output_token_logprobs": [
                (-0.5, 101), (-0.3, 102), (-0.1, 103),
            ],
            "finish_reason": {"type": "stop"},
        },
    }

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def cleanup():
    """Clean up module state between tests."""
    _active_envs.clear()
    engine._current_rollout_id = 0
    _sglang_calls.clear()
    FakeGenerateState.teardown()
    yield
    _active_envs.clear()
    engine._current_rollout_id = None
    _sglang_calls.clear()
    FakeGenerateState.teardown()

class MockArgs:
    """Minimal args for generate()."""
    partial_rollout = False
    max_steps = 5
    agent_id = "qwen3_vl"
    resize = None
    post_action_delay = None
    sglang_router_ip = "127.0.0.1"
    sglang_router_port = 30000
    hf_checkpoint = "test"
    sglang_server_concurrency = 64
    rollout_num_gpus = 2
    rollout_num_gpus_per_engine = 2
    rollout_temperature = 1.0
    rollout_top_p = 1.0
    rollout_top_k = -1
    rollout_max_response_len = 4096
    rollout_stop = None
    rollout_stop_token_ids = None
    rollout_skip_special_tokens = True
    # v0.3.0: generate() derives a per-group env seed from (_current_rollout_id,
    # group_index) for non-eval rollouts and asserts _current_rollout_id is set
    # (it's populated by _generate_rollout_async in production). These unit tests
    # call generate() directly, outside that loop, so we disable the shared-seed
    # branch — a legitimate supported config flag (group_shared_seed: false). The
    # env is mocked here anyway, so the seed value is never exercised.
    group_shared_seed = False

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateFlow:
    """Test the generate() function end-to-end with mocks."""

    @pytest.mark.asyncio
    async def test_basic_trajectory(self):
        """Verify basic multi-turn trajectory produces correct samples."""
        args = MockArgs()
        sample = Sample(
            group_index=0, index=0, prompt="test",
            metadata={"env_key": "test@task1"},
        )

        fake_env = FakeEnv(n_steps=3, terminal_reward=1.0)

        with (
            patch("lite.train.rollout.core.engine.GenerateState", FakeGenerateState),
            patch("lite.train.rollout.core.engine.post", fake_post),
            patch("lite.train.rollout.core.engine.gym") as mock_gym,
            patch("lite.train.rollout.core.engine.AgentRegistry") as mock_registry,
        ):
            mock_gym.make.return_value = fake_env

            # Make AgentRegistry.get return a FakeAgent that uses our generate_fn
            def make(key, processor=None, generate_fn=None, **kwargs):
                return FakeAgent(generate_fn, n_steps=3, terminal_reward=1.0)
            mock_registry.get.side_effect = make

            sampling_params = {"temperature": 1.0, "max_new_tokens": 4096}
            result = await generate(args, sample, sampling_params)

        # Should have 3 turn samples
        assert len(result) == 3, f"Expected 3 turns, got {len(result)}"

        # Check each turn sample
        for i, ts in enumerate(result):
            assert ts.group_index == 0
            assert ts.index == 0
            assert ts.metadata["turn_range"] == (i, i)
            assert ts.response_length == 3  # 3 tokens from fake SGLang
            assert ts.rollout_log_probs == [-0.5, -0.3, -0.1]
            assert ts.status == Sample.Status.COMPLETED
            assert ts.tokens  # should have context + response tokens

        # Terminal reward on last sample
        assert result[-1].reward == 1.0
        # Intermediate rewards should be 0.0
        assert result[0].reward == 0.0
        assert result[1].reward == 0.0

        # SGLang should have been called 3 times
        assert len(_sglang_calls) == 3

        # Env should be unregistered
        assert fake_env not in _active_envs

    @pytest.mark.asyncio
    async def test_generate_finally_close_failure_is_logged_and_swallowed(self, caplog):
        """The outer training loop may need to close envs the agent did not."""
        from lite.core import LiteCUAMetadata, LiteSample
        from lite.core.samples import LiteRLSample

        class NoCloseAgent:
            async def sample(self, env, max_steps=100, hooks=None):
                return LiteRLSample(
                    lite_sample=LiteSample(
                        metadata=LiteCUAMetadata(dims=("desktop", "use"))
                    ),
                    processed_images=[],
                    steps=[],
                    terminated=True,
                )

        args = MockArgs()
        sample = Sample(
            group_index=0, index=0, prompt="test",
            metadata={"env_key": "test@task1"},
        )
        fake_env = CloseRaisesEnv(n_steps=1)
        caplog.set_level(logging.WARNING, logger="lite.train.rollout.core.engine")

        with (
            patch("lite.train.rollout.core.engine.GenerateState", FakeGenerateState),
            patch("lite.train.rollout.core.engine.gym") as mock_gym,
            patch("lite.train.rollout.core.engine.AgentRegistry") as mock_registry,
            patch(
                "lite.train.rollout.core.engine.build_segment_samples",
                return_value=[sample],
            ),
        ):
            mock_gym.make.return_value = fake_env
            mock_registry.get.return_value = NoCloseAgent()

            result = await generate(
                args, sample, {"temperature": 1.0, "max_new_tokens": 4096}
            )

        assert result == [sample]
        assert fake_env._closed is True
        assert fake_env not in _active_envs
        assert (
            "env.close() failed in generate() finally: engine close exploded"
            in caplog.text
        )

    @pytest.mark.asyncio
    async def test_generate_agent_error_is_not_masked_by_close_failure(self, caplog):
        class RaisingAgent:
            async def sample(self, env, max_steps=100, hooks=None):
                raise RuntimeError("agent sample exploded")

        args = MockArgs()
        sample = Sample(
            group_index=0, index=0, prompt="test",
            metadata={"env_key": "test@task1"},
        )
        fake_env = CloseRaisesEnv(n_steps=1)
        caplog.set_level(logging.WARNING, logger="lite.train.rollout.core.engine")

        with (
            patch("lite.train.rollout.core.engine.GenerateState", FakeGenerateState),
            patch("lite.train.rollout.core.engine.gym") as mock_gym,
            patch("lite.train.rollout.core.engine.AgentRegistry") as mock_registry,
        ):
            mock_gym.make.return_value = fake_env
            mock_registry.get.return_value = RaisingAgent()

            result = await generate(
                args, sample, {"temperature": 1.0, "max_new_tokens": 4096}
            )

        assert len(result) == 1
        assert result[0].status == Sample.Status.FAILED
        assert result[0].metadata["lite_failure_reason"] == "task_crash"
        assert fake_env._closed is True
        assert fake_env not in _active_envs
        assert "Error in generate() for test@task1 [reason=task_crash]" in caplog.text
        assert (
            "env.close() failed in generate() finally: engine close exploded"
            in caplog.text
        )

    @pytest.mark.asyncio
    async def test_env_creation_failure(self):
        """Env creation failure should return empty sample, not crash."""
        args = MockArgs()
        sample = Sample(
            group_index=0, index=0, prompt="test",
            metadata={"env_key": "broken:task"},
        )

        with (
            patch("lite.train.rollout.core.engine.GenerateState", FakeGenerateState),
            patch("lite.train.rollout.core.engine.gym") as mock_gym,
        ):
            mock_gym.make.side_effect = RuntimeError("Docker failed")

            sampling_params = {"temperature": 1.0, "max_new_tokens": 4096}
            result = await generate(args, sample, sampling_params)

        assert len(result) == 1
        assert result[0].status == Sample.Status.FAILED
        assert result[0].reward == 0.0
        assert result[0].tokens == []

    @pytest.mark.asyncio
    async def test_abort_during_generation(self):
        """AbortError should be caught and return partial samples."""
        args = MockArgs()
        sample = Sample(
            group_index=0, index=0, prompt="test",
            metadata={"env_key": "test@task1"},
        )

        call_count = 0

        async def aborting_post(url, payload):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                # Simulate abort by setting state
                FakeGenerateState().aborted = True
            return await fake_post(url, payload)

        fake_env = FakeEnv(n_steps=5, terminal_reward=1.0)

        with (
            patch("lite.train.rollout.core.engine.GenerateState", FakeGenerateState),
            patch("lite.train.rollout.core.engine.post", aborting_post),
            patch("lite.train.rollout.core.engine.gym") as mock_gym,
            patch("lite.train.rollout.core.engine.AgentRegistry") as mock_registry,
        ):
            mock_gym.make.return_value = fake_env

            def make(key, processor=None, generate_fn=None, **kwargs):
                return FakeAgent(generate_fn, n_steps=5, terminal_reward=1.0)
            mock_registry.get.side_effect = make

            sampling_params = {"temperature": 1.0, "max_new_tokens": 4096}
            result = await generate(args, sample, sampling_params)

        # Should have at least one sample (empty on abort)
        assert len(result) >= 1
        # Abort returns empty/failed sample since agent.sample() didn't complete
        assert result[-1].status in (Sample.Status.ABORTED, Sample.Status.FAILED)

    @pytest.mark.asyncio
    async def test_sglang_payload_format(self):
        """Verify SGLang payload uses 'text' key (not 'input_ids')."""
        args = MockArgs()
        sample = Sample(
            group_index=0, index=0, prompt="test",
            metadata={"env_key": "test@task1"},
        )

        fake_env = FakeEnv(n_steps=1, terminal_reward=1.0)

        with (
            patch("lite.train.rollout.core.engine.GenerateState", FakeGenerateState),
            patch("lite.train.rollout.core.engine.post", fake_post),
            patch("lite.train.rollout.core.engine.gym") as mock_gym,
            patch("lite.train.rollout.core.engine.AgentRegistry") as mock_registry,
        ):
            mock_gym.make.return_value = fake_env

            def make(key, processor=None, generate_fn=None, **kwargs):
                return FakeAgent(generate_fn, n_steps=1, terminal_reward=1.0)
            mock_registry.get.side_effect = make

            sampling_params = {"temperature": 1.0, "max_new_tokens": 4096}
            await generate(args, sample, sampling_params)

        # Verify payload format
        assert len(_sglang_calls) == 1
        payload = _sglang_calls[0]["payload"]
        assert "text" in payload, "Should use 'text' key"
        assert "input_ids" not in payload, "Should NOT use 'input_ids' key"
        assert payload["return_logprob"] is True
        assert "sampling_params" in payload

    @pytest.mark.asyncio
    async def test_agent_key_derivation(self):
        """Verify agent_key is derived from agent_id + env metadata."""
        args = MockArgs()
        args.agent_id = "ui_tars"
        sample = Sample(
            group_index=0, index=0, prompt="test",
            metadata={"env_key": "test@task1"},
        )

        fake_env = FakeEnv(n_steps=1, terminal_reward=1.0)

        captured_key = None

        with (
            patch("lite.train.rollout.core.engine.GenerateState", FakeGenerateState),
            patch("lite.train.rollout.core.engine.post", fake_post),
            patch("lite.train.rollout.core.engine.gym") as mock_gym,
            patch("lite.train.rollout.core.engine.AgentRegistry") as mock_registry,
        ):
            mock_gym.make.return_value = fake_env

            def capture_agent(key, processor=None, generate_fn=None, **kwargs):
                nonlocal captured_key
                captured_key = key
                return FakeAgent(generate_fn, n_steps=1, terminal_reward=1.0)
            mock_registry.get.side_effect = capture_agent

            sampling_params = {"temperature": 1.0, "max_new_tokens": 4096}
            await generate(args, sample, sampling_params)

        assert captured_key == "ui_tars@desktop@use"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("field", sorted(AGENT_KWARGS_TOOL_SURFACE_KEYS))
    async def test_agent_kwargs_reject_tool_surface_fields(self, field):
        args = MockArgs()
        args.agent_kwargs = {field: "bad"}
        sample = Sample(
            group_index=0, index=0, prompt="test",
            metadata={"env_key": "test@task1"},
        )

        fake_env = FakeEnv(n_steps=1, terminal_reward=1.0)

        with (
            patch("lite.train.rollout.core.engine.GenerateState", FakeGenerateState),
            patch("lite.train.rollout.core.engine.gym") as mock_gym,
            patch("lite.train.rollout.core.engine.AgentRegistry") as mock_registry,
        ):
            mock_gym.make.return_value = fake_env

            sampling_params = {"temperature": 1.0, "max_new_tokens": 4096}
            result = await generate(args, sample, sampling_params)

        assert result[0].status == Sample.Status.FAILED
        mock_gym.make.assert_not_called()
        mock_registry.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_sampling_kwargs_is_dropped_not_passed_to_the_agent(self):
        """`sampling_kwargs` is a generate_fn concern and must never reach the adapter.

        Rollout and training SHARE one yaml, so a key that is legitimate for
        `rollout.py` must not kill the training path. 125 shipped configs under
        `scripts/configs/` carry `sampling_kwargs`; when the adapter base grew a
        strict unknown-kwarg check, every one of them crashed every RL task with
        `unknown adapter kwargs ['sampling_kwargs']` -- and the failure was easy
        to misread, because eval still reported a number (`0.0`) rather than an
        error.

        Training DROPS it rather than honoring it: the trainer owns sampling via
        `--rollout-temperature` / `--eval-temperature`, which is what those
        configs' own comment says.
        """
        args = MockArgs()
        args.agent_kwargs = {"sampling_kwargs": {"temperature": 0.0}, "resolution": [720, 1600]}
        sample = Sample(
            group_index=0, index=0, prompt="test",
            metadata={"env_key": "test@task1"},
        )

        fake_env = FakeEnv(n_steps=1, terminal_reward=1.0)
        captured_kwargs = {}

        with (
            patch("lite.train.rollout.core.engine.GenerateState", FakeGenerateState),
            patch("lite.train.rollout.core.engine.gym") as mock_gym,
            patch("lite.train.rollout.core.engine.AgentRegistry") as mock_registry,
        ):
            mock_gym.make.return_value = fake_env

            def capture_agent(key, processor=None, generate_fn=None, **kwargs):
                captured_kwargs.update(kwargs)
                return FakeAgent(generate_fn, n_steps=1, terminal_reward=1.0)
            mock_registry.get.side_effect = capture_agent

            sampling_params = {"temperature": 1.0, "max_new_tokens": 4096}
            await generate(args, sample, sampling_params)

        # Dropped on the way in...
        assert "sampling_kwargs" not in captured_kwargs
        # ...while genuine adapter kwargs still arrive, so this is a targeted
        # drop and not a blanket "swallow everything unknown".
        assert captured_kwargs.get("resolution") == [720, 1600]
        # ...and the run got PAST the kwarg gate: the tool-surface test above
        # asserts both of these were never called. This is the contrast that
        # matters -- `sampling_kwargs` must not abort the task the way a
        # tool-surface key does. (The final Sample status is not asserted: this
        # class's FakeAgent drives the real `generate_fn`, whose HTTP call is
        # only stubbed by the tests that need it.)
        mock_gym.make.assert_called()
        mock_registry.get.assert_called()

    @pytest.mark.asyncio
    async def test_agent_key_override_from_metadata(self):
        """agent_key from sample.metadata should override derived key."""
        args = MockArgs()
        sample = Sample(
            group_index=0, index=0, prompt="test",
            metadata={"env_key": "test@task1", "agent_key": "custom@agent@key"},
        )

        fake_env = FakeEnv(n_steps=1, terminal_reward=1.0)
        captured_key = None

        with (
            patch("lite.train.rollout.core.engine.GenerateState", FakeGenerateState),
            patch("lite.train.rollout.core.engine.post", fake_post),
            patch("lite.train.rollout.core.engine.gym") as mock_gym,
            patch("lite.train.rollout.core.engine.AgentRegistry") as mock_registry,
        ):
            mock_gym.make.return_value = fake_env

            def capture_agent(key, processor=None, generate_fn=None, **kwargs):
                nonlocal captured_key
                captured_key = key
                return FakeAgent(generate_fn, n_steps=1, terminal_reward=1.0)
            mock_registry.get.side_effect = capture_agent

            sampling_params = {"temperature": 1.0, "max_new_tokens": 4096}
            await generate(args, sample, sampling_params)

        assert captured_key == "custom@agent@key"

class TestGenerateStress:
    """Stress tests for concurrent generate() calls."""

    @pytest.mark.asyncio
    async def test_concurrent_generates(self):
        """Run multiple generate() calls concurrently."""
        args = MockArgs()

        async def run_one(idx):
            sample = Sample(
                group_index=idx // 4, index=idx, prompt="test",
                metadata={"env_key": f"test@task_{idx}"},
            )
            fake_env = FakeEnv(n_steps=2, terminal_reward=float(idx + 1))

            with (
                patch("lite.train.rollout.core.engine.GenerateState", FakeGenerateState),
                patch("lite.train.rollout.core.engine.post", fake_post),
                patch("lite.train.rollout.core.engine.gym") as mock_gym,
                patch("lite.train.rollout.core.engine.AgentRegistry") as mock_registry,
            ):
                mock_gym.make.return_value = fake_env

                def make(key, processor=None, generate_fn=None, **kwargs):
                    return FakeAgent(generate_fn, n_steps=2, terminal_reward=float(idx + 1))
                mock_registry.get.side_effect = make

                sampling_params = {"temperature": 1.0, "max_new_tokens": 4096}
                return await generate(args, sample, sampling_params)

        # Run 8 concurrent generates
        tasks = [run_one(i) for i in range(8)]
        results = await asyncio.gather(*tasks)

        for i, result in enumerate(results):
            assert len(result) == 2, f"Task {i}: expected 2 turns, got {len(result)}"
            assert result[-1].reward == float(i + 1)
            assert result[-1].index == i

    @pytest.mark.asyncio
    async def test_mixed_success_failure(self):
        """Mix of successful and failing envs should not crash."""
        args = MockArgs()

        async def run_one(idx, should_fail):
            sample = Sample(
                group_index=0, index=idx, prompt="test",
                metadata={"env_key": f"test@task_{idx}"},
            )

            if should_fail:
                # Env creation will fail
                with (
                    patch("lite.train.rollout.core.engine.GenerateState", FakeGenerateState),
                    patch("lite.train.rollout.core.engine.gym") as mock_gym,
                ):
                    mock_gym.make.side_effect = RuntimeError("Docker OOM")
                    sampling_params = {"temperature": 1.0, "max_new_tokens": 4096}
                    return await generate(args, sample, sampling_params)
            else:
                fake_env = FakeEnv(n_steps=2, terminal_reward=1.0)
                with (
                    patch("lite.train.rollout.core.engine.GenerateState", FakeGenerateState),
                    patch("lite.train.rollout.core.engine.post", fake_post),
                    patch("lite.train.rollout.core.engine.gym") as mock_gym,
                    patch("lite.train.rollout.core.engine.AgentRegistry") as mock_registry,
                ):
                    mock_gym.make.return_value = fake_env

                    def make(key, processor=None, generate_fn=None, **kwargs):
                        return FakeAgent(generate_fn, n_steps=2, terminal_reward=1.0)
                    mock_registry.get.side_effect = make

                    sampling_params = {"temperature": 1.0, "max_new_tokens": 4096}
                    return await generate(args, sample, sampling_params)

        # 4 successes, 4 failures
        tasks = [run_one(i, should_fail=(i % 2 == 0)) for i in range(8)]
        results = await asyncio.gather(*tasks)

        for i, result in enumerate(results):
            assert len(result) >= 1
            if i % 2 == 0:
                # Failed env
                assert result[0].status == Sample.Status.FAILED
            else:
                # Successful
                assert len(result) == 2
                assert result[-1].status == Sample.Status.COMPLETED
