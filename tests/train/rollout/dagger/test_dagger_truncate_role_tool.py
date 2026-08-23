"""U1 — DAgger ``_truncate_to_observation`` must be role-agnostic (user | tool).

Guardrail for role-agnostic observation truncation. The DAgger teacher
truncates a trajectory to turn ``k``'s observation before re-querying the teacher
(``lite.train.rollout.dagger.teacher._truncate_to_observation``). The historical
implementation counted a "turn" only on ``role=="user"`` messages, so this file
guards the role-agnostic observation contract.

Current trajectories can carry each turn's observation on a per-call
``role:"tool"`` message (the turn-0 task query stays ``role:"user"``). The
truncation counter must treat both observation roles correctly so the teacher never
sees turn ``k``'s assistant or later student actions when relabeling turn ``k``.

  * ``test_truncate_to_observation_role_user`` freezes the current correct
    behavior on ``role:"user"`` observation messages.
  * ``test_truncate_to_observation_role_tool`` pins the same contract on the
    ``role:"tool"`` observation shape.
  * ``test_teacher_relabel_prompt_image_data_keeps_role_tool_order`` verifies
    the teacher-render path carries role:tool images into ordered ``image_data``.

Hermetic: ``dagger.teacher`` imports ``slime.utils`` at module load (unavailable
outside the Slime container). Instead of ``importorskip`` (which would SKIP the
whole module and give zero coverage off-container), we install tiny
``slime.utils.*`` stubs before import and monkeypatch the imported symbols in the
teacher-render test.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/train/rollout/dagger/test_dagger_truncate_role_tool.py -p no:cacheprovider -q
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest


# -----------------------------------------------------------------------------
# Stub slime.utils.{http_utils,processing_utils} so ``dagger.teacher`` imports
# off-container.
# -----------------------------------------------------------------------------
def _install_slime_stubs() -> list[str]:
    if "slime.utils" in sys.modules:
        return []
    inserted: list[str] = []
    for name in ("slime", "slime.utils"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__dict__["_cua_lite_test_stub"] = True
            sys.modules[name] = mod
            inserted.append(name)
    http = types.ModuleType("slime.utils.http_utils")
    http.post = lambda *a, **k: None  # never called here
    proc = types.ModuleType("slime.utils.processing_utils")
    proc.encode_image_for_rollout_engine = lambda *a, **k: None  # never called here
    sys.modules["slime.utils.http_utils"] = http
    sys.modules["slime.utils.processing_utils"] = proc
    inserted.extend(["slime.utils.http_utils", "slime.utils.processing_utils"])
    return inserted


_SLIME_STUBS = _install_slime_stubs()

import lite.train.rollout.dagger.teacher as dt  # noqa: E402
from lite.core import LiteCUAMetadata, LiteSample  # noqa: E402
from lite.core.samples import LiteRLSample, LiteRLStep  # noqa: E402
from lite.core.tools.calls import make_tool_call  # noqa: E402

for _name in reversed(_SLIME_STUBS):
    sys.modules.pop(_name, None)

_truncate_to_observation = dt._truncate_to_observation


def _computer_call(call_id: str, *actions: dict) -> dict:
    return make_tool_call("computer", {"actions": list(actions)}, call_id=call_id)


# -----------------------------------------------------------------------------
# Fixture: a 3-turn trajectory whose OBSERVATION messages carry a configurable
# role. In the user-observation fixture every observation is ``role:"user"``; in the role:tool
# shape only the turn-0 task query stays ``role:"user"`` and the later per-turn
# observations become ``role:"tool"``.
# -----------------------------------------------------------------------------
def _traj(obs_role: str, n_turns: int = 3) -> LiteSample:
    messages: list[dict] = [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]}
    ]
    for k in range(n_turns):
        # turn-0 is the task query (always user); later observations use obs_role.
        role = "user" if k == 0 else obs_role
        content: list[dict] = [{"type": "image", "index": k}]
        if k == 0:
            content.append({"type": "text", "text": "Open GIMP and apply a filter."})
        obs_msg = {"role": role, "content": content}
        if role == "tool":
            obs_msg["tool_call_id"] = f"call_{k - 1}"
        messages.append(obs_msg)
        messages.append({
            "role": "assistant",
            "content": [{"type": "action_description", "text": f"Action {k}"}],
            "tool_calls": [
                _computer_call(
                    f"call_{k}",
                    {"action": "click", "coordinate": [10 + k, 20 + k]},
                )
            ],
        })
    return LiteSample(metadata=LiteCUAMetadata(), images=[], messages=messages)


def _assistant_actions(sample: LiteSample) -> list[str]:
    """The ``action_description`` text of every assistant message kept in the
    slice — the "assistant coverage" the slice exposes to the teacher."""
    return [
        m["content"][0]["text"]
        for m in sample.messages
        if m["role"] == "assistant"
    ]


_EOS = 99999


class _FakeTokenizer:
    eos_token_id = _EOS

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **_kwargs
    ):
        rendered = []
        for msg in messages:
            parts = []
            for part in msg.get("content") or []:
                if part.get("type") == "text":
                    parts.append(part["text"])
                elif part.get("type") == "image":
                    parts.append(f"<image:{part['index']}>")
            rendered.append(f"[{msg['role']}]{''.join(parts)}")
        if add_generation_prompt:
            rendered.append("[assistant]")
        return "".join(rendered)


class _FakeStudentAdapter:
    metadata = LiteCUAMetadata(dims=("mobile", "use"))


class _FakeTeacherAdapter:
    metadata = LiteCUAMetadata(dims=("mobile", "use"))
    enable_thinking = False

    def __init__(self):
        self.render_roles: list[list[str]] = []

    def process_image(self, image):
        return f"processed:{image}"

    def render_step(self, sample: LiteSample, k: int, processed):
        assert k == 2
        assert processed == ["processed:raw0", "processed:raw1", "processed:raw2"]
        self.render_roles.append([m["role"] for m in sample.messages])
        return sample.messages

    def parse_raw_assistant_response(self, response: str):
        return {"role": "assistant", "_raw": response}

    def convert_message_from_agent(self, agent_message):
        return {
            "role": "assistant",
            "tool_calls": [
                _computer_call("call_0", {"action": "click", "coordinate": [1, 2]})
            ],
        }


def _args() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        rollout_max_response_len=128,
        dagger={
            "teacher_agent_id": "fake_teacher",
            "teacher_agent_kwargs": None,
            "teacher_url": "http://fake/generate",
            "teacher_temperature": 0.0,
            "teacher_max_new_tokens": 16,
            "strip_teacher_think": True,
        },
    )


def _agent() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        adapter=_FakeStudentAdapter(),
        processor=_FakeProcessor(),
    )


# -----------------------------------------------------------------------------
# Role:"user" observations
# -----------------------------------------------------------------------------
def test_truncate_to_observation_role_user() -> None:
    """Freeze behavior for the role:user fixture: ``k=2`` keeps up to & incl.
    the 2nd user message and drops that turn's assistant onward.

    Trace of the role:user fixture (turn is 1-indexed on observation messages):
      u0→turn1(keep) a0(keep) u1→turn2(keep) a1: turn==k → BREAK.
    So the slice is ``[system, u0, a0, u1]`` — assistant coverage = just Action 0
    (the teacher never sees the student's turn-1/2 actions). This is correct."""
    sliced = _truncate_to_observation(_traj("user"), k=2)

    assert [m["role"] for m in sliced.messages] == ["system", "user", "assistant", "user"]
    # Coverage stops before turn-2's assistant — no label leakage.
    assert _assistant_actions(sliced) == ["Action 0"]


def test_truncate_to_observation_role_tool() -> None:
    """The SAME trajectory with observations as ``role:"tool"`` (turn-0 query
    stays ``role:"user"``) slices to the SAME assistant coverage as the
    ``role:"user"`` slice."""
    sliced = _truncate_to_observation(_traj("tool"), k=2)
    role_user_coverage = _assistant_actions(_truncate_to_observation(_traj("user"), k=2))
    tool_coverage = _assistant_actions(sliced)

    assert [m["role"] for m in sliced.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert tool_coverage == role_user_coverage


def test_truncate_groups_multiple_role_tool_results_as_one_observation() -> None:
    sample = LiteSample(
        metadata=LiteCUAMetadata(),
        images=[],
        messages=[
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": [{"type": "text", "text": "task"}]},
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": "Action 0"}],
                "tool_calls": [
                    _computer_call("call_0", {"action": "click", "coordinate": [10, 20]}),
                    make_tool_call(
                        "goto",
                        {"url": "https://example.com"},
                        call_id="call_1",
                    ),
                ],
            },
            {"role": "tool", "tool_call_id": "call_0", "content": [{"type": "image", "index": 1}]},
            {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "image", "index": 2}]},
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": "Action 1"}],
                "tool_calls": [
                    _computer_call("call_2", {"action": "click", "coordinate": [30, 40]}),
                ],
            },
        ],
    )

    sliced = _truncate_to_observation(sample, k=2)

    assert [m["role"] for m in sliced.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert _assistant_actions(sliced) == ["Action 0"]


def test_teacher_agent_kwargs_reject_tool_surface_metadata() -> None:
    with pytest.raises(ValueError, match="teacher_agent_kwargs contains tool-surface"):
        dt._build_teacher_adapter(
            _FakeStudentAdapter(),
            "fake_teacher",
            {"metadata": LiteCUAMetadata(dims=("browser", "use"))},
            "qwen3_vl@mobile@use",
        )


def test_teacher_relabel_prompt_image_data_keeps_role_tool_order(monkeypatch) -> None:
    """Teacher relabeling should render the truncated role:tool observation and
    send image_data in the exact message order, including repeated/out-of-order
    role:tool image placeholders."""
    teacher = _FakeTeacherAdapter()
    payloads = []

    def fake_build_teacher_adapter(*_args, **_kwargs):
        return teacher

    async def fake_post(url, payload):
        payloads.append(payload)
        return {"text": "<tool_call>{}</tool_call>"}

    monkeypatch.setattr(dt, "_build_teacher_adapter", fake_build_teacher_adapter)
    monkeypatch.setattr(
        dt,
        "encode_image_for_rollout_engine",
        lambda image: f"encoded:{image}",
    )
    monkeypatch.setattr(dt, "post", fake_post)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {"role": "user", "content": [
            {"type": "image", "index": 0},
            {"type": "text", "text": "task"},
        ]},
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "Action 0"}],
            "tool_calls": [
                _computer_call("call_0", {"action": "click", "coordinate": [10, 20]})
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": [
            {"type": "image", "index": 2},
            {"type": "text", "text": "after action zero"},
            {"type": "image", "index": 1},
        ]},
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "Action 1"}],
            "tool_calls": [
                _computer_call("call_1", {"action": "click", "coordinate": [30, 40]})
            ],
        },
    ]
    lite_sample = LiteSample(
        metadata=LiteCUAMetadata(dims=("mobile", "use")),
        images=["raw0", "raw1", "raw2"],
        messages=messages,
    )
    rl_sample = LiteRLSample(
        lite_sample=lite_sample,
        processed_images=[],
        steps=[
            LiteRLStep(prompt="P0", image_indices=(), response="", response_tokens=[]),
            LiteRLStep(prompt="P1", image_indices=(), response="student", response_tokens=[1]),
        ],
    )

    asyncio.run(
        dt.relabel_with_teacher(
            _args(),
            rl_sample,
            types.SimpleNamespace(metadata={}),
            _agent(),
            "qwen3_vl@mobile@use",
        )
    )

    assert teacher.render_roles == [["system", "user", "assistant", "tool"]]
    assert len(payloads) == 1
    assert "[tool]<image:2>after action zero<image:1>" in payloads[0]["text"]
    assert payloads[0]["image_data"] == [
        "encoded:processed:raw0",
        "encoded:processed:raw2",
        "encoded:processed:raw1",
    ]
