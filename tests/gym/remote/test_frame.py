"""Unit tests for the binary reset/step wire frames.

Covers the split remote-frame contract:
  * reset frames round-trip ``LiteEnvObservation`` directly;
  * step frames round-trip ``LiteEnvStepResult`` without a top-level
    observation, with ordered per-call ``LiteToolResult`` image/text/metadata/error;
  * header type guards name the offending ``metadata``/``info`` key;
  * a ``NaN``/``Inf`` reward fails fast at encode (``allow_nan=False``);
  * typed decode errors: every truncation site -> :class:`TruncatedFrame`;
    bad MAGIC -> :class:`BadMagic`; valid-MAGIC wrong version/kind ->
    :class:`BadVersion`.

Run:  uv run pytest tests/gym/remote/test_frame.py -x -q
"""
from __future__ import annotations

import base64
import io
import json
import math

import pytest
from PIL import Image

from lite.core.tools.results import LiteToolResult, make_tool_result
from lite.gym.remote.frame import (
    FRAME_MAGIC,
    FRAME_VERSION,
    BadMagic,
    BadVersion,
    TruncatedFrame,
    decode_reset_observation,
    decode_step_result,
    encode_reset_observation,
    encode_step_result,
)
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult

_MAGIC = FRAME_MAGIC.encode("ascii")  # kept in sync with frame._MAGIC


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def _png(w: int, h: int) -> bytes:
    """A real PNG so the body is a plausible multi-KB blob."""
    img = Image.new("RGB", (w, h))
    # A little structure so the PNG is not a trivial all-zero run.
    for x in range(0, w, 7):
        img.putpixel((x, x % h), (x % 256, (2 * x) % 256, (3 * x) % 256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_FULL_PNG = _png(320, 240)


def _obs(
    *,
    image: bytes | None = _FULL_PNG,
    text: str | None = "hello",
    metadata: dict | None = None,
) -> LiteEnvObservation:
    return LiteEnvObservation(image=image, text=text, metadata=metadata)


def _step(
    *,
    reward: float | None = 1.0,
    terminated: bool = False,
    truncated: bool = False,
    info: dict | None = None,
    results: list[LiteToolResult] | None = None,
) -> LiteEnvStepResult:
    return LiteEnvStepResult(
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info={} if info is None else info,
        results=[] if results is None else results,
    )


# ---------------------------------------------------------------------------
# Round-trip / bit identity
# ---------------------------------------------------------------------------


_RESET_ROUNDTRIP_CASES = {
    "full_res_png": _obs(image=_FULL_PNG),
    "empty_bytes_shot": _obs(image=b""),
    "none_shot": _obs(image=None),
    "text_none": _obs(text=None),
    # A lone (unpaired) surrogate is exactly the DOM-text hazard the Scope
    # decision keeps in the JSON header: JSON emits a ``\udxxx`` escape, so no
    # ``surrogatepass`` is needed.
    "lone_surrogate_text": _obs(text="a\ud800b" + "x" * 4000),
    "nested_metadata": _obs(
        metadata={"url": "http://x", "nested": {"a": [1, 2, {"b": None}]}, "n": 3}
    ),
    "metadata_none": _obs(metadata=None),
}


_STEP_ROUNDTRIP_CASES = {
    "empty_results": _step(),
    "reward_zero": _step(reward=0.0),
    "reward_none": _step(reward=None),
    "flags_terminated": _step(terminated=True),
    "flags_truncated": _step(truncated=True),
    "flags_both": _step(terminated=True, truncated=True),
    "rich_info": _step(info={"executed_actions": [{"call": "click", "args": {"x": 9}}]}),
    "mixed_results": _step(
        results=[
            make_tool_result(
                "gui_0",
                images=[b"\x89PNG-result"],
                text="## AXTree:\nbody",
                metadata={"n": 1},
                error="invalid action arguments",
            ),
            LiteToolResult(
                tool_call_id="bash_0", text="stdout\n", metadata={"exit_status": 0}
            ),
            LiteToolResult(tool_call_id=None, text="unpaired"),
        ],
    ),
}


@pytest.mark.parametrize(
    "case", list(_RESET_ROUNDTRIP_CASES), ids=list(_RESET_ROUNDTRIP_CASES)
)
def test_reset_roundtrip_bit_identical(case: str):
    x = _RESET_ROUNDTRIP_CASES[case]
    got = decode_reset_observation(encode_reset_observation(x))
    assert got == x
    assert got.image == x.image
    if x.image is not None:
        assert isinstance(got.image, bytes)


@pytest.mark.parametrize(
    "case", list(_STEP_ROUNDTRIP_CASES), ids=list(_STEP_ROUNDTRIP_CASES)
)
def test_step_roundtrip_bit_identical(case: str):
    x = _STEP_ROUNDTRIP_CASES[case]
    assert decode_step_result(encode_step_result(x)) == x


def test_step_roundtrip_preserves_ordered_images_across_results():
    first = b"\x89PNG-first"
    second = b"\x89PNG-second"
    third = b"\x89PNG-third"
    x = _step(
        results=[
            LiteToolResult(
                tool_call_id="gui_0",
                images=[first, second],
                text="two captures",
            ),
            LiteToolResult(tool_call_id="bash_0", text="no capture"),
            LiteToolResult(tool_call_id="gui_1", images=[third]),
        ]
    )

    frame = encode_step_result(x)
    hlen = int.from_bytes(frame[4:8], "big")
    header = json.loads(frame[8 : 8 + hlen])
    tail = frame[8 + hlen :]

    assert header["results"][0]["shot_n"] == [len(first), len(second)]
    assert header["results"][1]["shot_n"] == []
    assert header["results"][2]["shot_n"] == [len(third)]
    assert len(tail) == len(first) + len(second) + len(third)
    assert tail == first + second + third
    assert decode_step_result(frame) == x


def test_step_roundtrip_preserves_raw_error_metadata():
    metadata = {"kind": "gui"}
    x = _step(
        results=[
            LiteToolResult(
                tool_call_id="gui_0",
                text="body",
                metadata=metadata,
                error="invalid action arguments",
            )
        ]
    )

    assert decode_step_result(encode_step_result(x)) == x
    assert metadata == {"kind": "gui"}


def test_reset_none_and_empty_bytes_are_distinct():
    """``None`` (no screenshot) and ``b""`` (empty screenshot) must survive as
    two distinct values via the ``shot_n`` length-list sentinel."""
    none_out = decode_reset_observation(encode_reset_observation(_obs(image=None)))
    empty_out = decode_reset_observation(encode_reset_observation(_obs(image=b"")))
    assert none_out.image is None
    assert empty_out.image == b""
    assert empty_out.image is not None


def test_step_info_empty_dict_preserved_not_coerced():
    """``info`` is typed ``dict`` (never ``None``); an empty dict must
    round-trip as ``{}``."""
    out = decode_step_result(encode_step_result(_step(info={})))
    assert out.info == {}


def test_reset_frame_layout_magic_and_header():
    """Reset frames carry only the reset observation fields."""
    frame = encode_reset_observation(_obs(image=_FULL_PNG, text="task"))
    assert frame[:4] == _MAGIC
    hlen = int.from_bytes(frame[4:8], "big")
    header = json.loads(frame[8 : 8 + hlen])
    assert header == {
        "v": FRAME_VERSION,
        "kind": "reset",
        "metadata": None,
        "text": "task",
        "shot_n": [len(_FULL_PNG)],
    }
    assert 8 + hlen + len(_FULL_PNG) == len(frame)
    assert b"screenshot" not in frame[: 8 + hlen]


def test_step_frame_layout_magic_and_header():
    """Step frames carry reward/flags/info/results, not an observation."""
    result_image = b"\x89PNG-result"
    x = _step(
        reward=1.0,
        terminated=True,
        info={"n": 1},
        results=[
            make_tool_result(
                "gui_0",
                images=[result_image],
                text="## AXTree:\nbody",
                metadata={"kind": "gui"},
                error="invalid action arguments",
            ),
            LiteToolResult(
                tool_call_id="bash_0", text="bash out", metadata={"exit_status": 0}
            ),
        ],
    )

    frame = encode_step_result(x)
    assert frame[:4] == _MAGIC
    hlen = int.from_bytes(frame[4:8], "big")
    header = json.loads(frame[8 : 8 + hlen])

    assert header == {
        "v": FRAME_VERSION,
        "kind": "step",
        "reward": 1.0,
        "terminated": True,
        "truncated": False,
        "info": {"n": 1},
        "results": [
            {
                "tool_call_id": "gui_0",
                "text": "## AXTree:\nbody",
                "metadata": {"kind": "gui", "is_error": True},
                "error": "invalid action arguments",
                "shot_n": [len(result_image)],
            },
            {
                "tool_call_id": "bash_0",
                "text": "bash out",
                "metadata": {"exit_status": 0},
                "error": None,
                "shot_n": [],
            },
        ],
    }
    assert "observation" not in header
    assert "category" not in header["results"][0]
    assert 8 + hlen + len(result_image) == len(frame)
    assert decode_step_result(frame) == x


# ---------------------------------------------------------------------------
# Header type guard names the offending key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload,badkey",
    [
        ({"ok": 1, "blob": b"\x00\x01"}, "blob"),
        ({"outer": {"inner": {1, 2}}}, "outer"),
    ],
)
def test_reset_metadata_guard_raises_typeerror_naming_key(
    payload: dict, badkey: str
):
    with pytest.raises(TypeError) as exc:
        encode_reset_observation(_obs(metadata=payload))
    msg = str(exc.value)
    assert "metadata" in msg
    assert repr(badkey) in msg


@pytest.mark.parametrize(
    "field,payload,badkey",
    [
        ("info", {"n": 1, "s": {1, 2, 3}}, "s"),
        ("info", {"outer": {"inner": {1, 2}}}, "outer"),
        ("result_metadata", {"blob": b"\x00\x01"}, "blob"),
    ],
)
def test_step_guard_raises_typeerror_naming_key(
    field: str, payload: dict, badkey: str
):
    if field == "info":
        x = _step(info=payload)
    else:
        x = _step(results=[LiteToolResult(tool_call_id="gui_0", metadata=payload)])
    with pytest.raises(TypeError) as exc:
        encode_step_result(x)
    msg = str(exc.value)
    assert ("info" in msg) if field == "info" else ("results[0].metadata" in msg)
    assert repr(badkey) in msg


def test_serializable_metadata_info_and_result_metadata_pass():
    reset = _obs(metadata={"url": "http://x"})
    step = _step(
        info={"score": 1.0},
        results=[LiteToolResult(tool_call_id="gui_0", metadata={"url": "http://x"})],
    )
    assert decode_reset_observation(encode_reset_observation(reset)).metadata == {
        "url": "http://x"
    }
    assert decode_step_result(encode_step_result(step)) == step


def test_reset_metadata_carries_browsergym_goal_images_verbatim():
    """The frame owns no env key: BrowserGym's turn-0 ``goal_images_b64``
    round-trips as ordinary reset metadata, alongside the raw-PNG tail."""
    goals = [base64.b64encode(b"\x89PNG-goal-%d" % i).decode() for i in (1, 2)]
    reset = _obs(metadata={"goal_images_b64": goals, "url": "http://x"})

    frame = encode_reset_observation(reset)
    hlen = int.from_bytes(frame[4:8], "big")
    header = json.loads(frame[8 : 8 + hlen])

    assert header["metadata"] == {"goal_images_b64": goals, "url": "http://x"}
    assert frame[8 + hlen :] == _FULL_PNG  # only the screenshot rides the tail

    assert decode_reset_observation(frame).metadata == reset.metadata

    # Production-sized goal image: reset metadata has no size cap.
    big = [base64.b64encode(_FULL_PNG).decode()]
    big_reset = _obs(metadata={"goal_images_b64": big})
    assert decode_reset_observation(
        encode_reset_observation(big_reset)
    ).metadata == {"goal_images_b64": big}


# ---------------------------------------------------------------------------
# NaN/Inf reward fails fast at encode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nan_inf_reward_raises_at_encode(bad: float):
    with pytest.raises(ValueError):
        encode_step_result(_step(reward=bad))


def test_decode_rfc_valid_overflow_float_is_truncated():
    """``1e400`` is valid JSON but Python parses it to ``inf`` by default.

    Encode cannot produce it because ``allow_nan=False`` sees only the already
    materialized Python value, so this forges the remote wire header directly.
    """
    header = (
        b'{"v":'
        + str(FRAME_VERSION).encode("ascii")
        + b',"kind":"step","reward":1e400,'
        b'"terminated":false,"truncated":false,"info":{},"results":[]}'
    )
    frame = _MAGIC + len(header).to_bytes(4, "big") + header

    with pytest.raises(TruncatedFrame, match="non-finite json number"):
        decode_step_result(frame)


def test_nan_reward_would_void_bit_identity():
    # Sanity: NaN != NaN, which is exactly why encode refuses it.
    assert math.isnan(float("nan"))
    assert float("nan") != float("nan")


# ---------------------------------------------------------------------------
# Typed decode errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decode", [decode_reset_observation, decode_step_result])
def test_decode_empty_buffer_is_truncated(decode):
    with pytest.raises(TruncatedFrame):
        decode(b"")


@pytest.mark.parametrize("decode", [decode_reset_observation, decode_step_result])
@pytest.mark.parametrize("n", [1, 3, 7])
def test_decode_sub_head_buffer_is_truncated_not_badmagic(decode, n: int):
    """A 1-7 byte buffer is shorter than the 8-byte frame head."""
    with pytest.raises(TruncatedFrame):
        decode(_MAGIC[:n] if n <= 4 else _MAGIC + b"\x00" * (n - 4))


@pytest.mark.parametrize(
    "encode,decode,value",
    [
        (encode_reset_observation, decode_reset_observation, _obs(image=_FULL_PNG)),
        (
            encode_step_result,
            decode_step_result,
            _step(results=[LiteToolResult(tool_call_id="gui_0", images=[_FULL_PNG])]),
        ),
    ],
    ids=["reset", "step"],
)
def test_decode_cut_inside_header_is_truncated_not_jsondecode(encode, decode, value):
    """A slice that ends inside the header must raise TruncatedFrame."""
    frame = encode(value)
    hlen = int.from_bytes(frame[4:8], "big")
    cut = frame[: 8 + hlen - 5]  # header claims hlen but body is short
    with pytest.raises(TruncatedFrame):
        decode(cut)


@pytest.mark.parametrize(
    "encode,decode,value",
    [
        (encode_reset_observation, decode_reset_observation, _obs(image=_FULL_PNG)),
        (
            encode_step_result,
            decode_step_result,
            _step(results=[LiteToolResult(tool_call_id="gui_0", images=[_FULL_PNG])]),
        ),
    ],
    ids=["reset", "step"],
)
def test_decode_truncated_tail_is_truncated(encode, decode, value):
    frame = encode(value)
    with pytest.raises(TruncatedFrame):
        decode(frame[:-10])


@pytest.mark.parametrize(
    "encode,decode,value",
    [
        (encode_reset_observation, decode_reset_observation, _obs(image=_FULL_PNG)),
        (
            encode_step_result,
            decode_step_result,
            _step(results=[LiteToolResult(tool_call_id="gui_0", images=[_FULL_PNG])]),
        ),
    ],
    ids=["reset", "step"],
)
def test_decode_trailing_bytes_is_truncated(encode, decode, value):
    frame = encode(value)
    with pytest.raises(TruncatedFrame):
        decode(frame + b"junk")


@pytest.mark.parametrize("decode", [decode_reset_observation, decode_step_result])
def test_decode_bad_magic_is_badmagic(decode):
    frame = bytearray(encode_reset_observation(_obs(image=_FULL_PNG)))
    assert frame[:4] != b"LEF1"
    forged = b"LEF1" + bytes(frame[4:])  # foreign/older MAGIC
    with pytest.raises(BadMagic):
        decode(forged)


def test_decode_wrong_version_is_badversion():
    """Correct MAGIC but mismatched header ``v`` is terminal."""
    header = {
        "v": FRAME_VERSION + 99,
        "kind": "reset",
        "metadata": None,
        "text": "hi",
        "shot_n": [],
    }
    hb = json.dumps(header).encode("utf-8")
    forged = _MAGIC + len(hb).to_bytes(4, "big") + hb
    with pytest.raises(BadVersion):
        decode_reset_observation(forged)


@pytest.mark.parametrize(
    "header,decode",
    [
        (
            {
                "v": FRAME_VERSION,
                "kind": "reset",
                "metadata": None,
                "text": "hi",
                "shot_n": None,
            },
            decode_reset_observation,
        ),
        (
            {
                "v": FRAME_VERSION,
                "kind": "step",
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
                "info": {},
                "results": [
                    {
                        "tool_call_id": "gui_0",
                        "text": None,
                        "metadata": None,
                        "error": None,
                        "shot_n": 0,
                    }
                ],
            },
            decode_step_result,
        ),
    ],
    ids=["reset", "step"],
)
def test_decode_scalar_shot_n_is_badversion(header: dict, decode):
    hb = json.dumps(header).encode("utf-8")
    forged = _MAGIC + len(hb).to_bytes(4, "big") + hb
    with pytest.raises(BadVersion, match="shot_n must be a list"):
        decode(forged)


@pytest.mark.parametrize(
    "shot_lengths",
    [
        ["1"],
        [-1],
        [True],
    ],
    ids=["string", "negative", "bool"],
)
def test_decode_invalid_shot_n_item_is_badversion(shot_lengths: list[object]):
    header = {
        "v": FRAME_VERSION,
        "kind": "reset",
        "metadata": None,
        "text": "hi",
        "shot_n": shot_lengths,
    }
    hb = json.dumps(header).encode("utf-8")
    forged = _MAGIC + len(hb).to_bytes(4, "big") + hb
    with pytest.raises(BadVersion, match="shot_n\\[0\\]"):
        decode_reset_observation(forged)


def test_decode_step_result_requires_tool_call_id_header():
    header = {
        "v": FRAME_VERSION,
        "kind": "step",
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "info": {},
        "results": [
            {
                "call_id": "gui_0",
                "text": None,
                "metadata": None,
                "error": None,
                "shot_n": [],
            }
        ],
    }
    hb = json.dumps(header).encode("utf-8")
    forged = _MAGIC + len(hb).to_bytes(4, "big") + hb
    with pytest.raises(BadVersion, match="tool_call_id"):
        decode_step_result(forged)


def test_decode_step_result_rejects_non_string_tool_call_id():
    header = {
        "v": FRAME_VERSION,
        "kind": "step",
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "info": {},
        "results": [
            {
                "tool_call_id": 1,
                "text": None,
                "metadata": None,
                "error": None,
                "shot_n": [],
            }
        ],
    }
    hb = json.dumps(header).encode("utf-8")
    forged = _MAGIC + len(hb).to_bytes(4, "big") + hb
    with pytest.raises(BadVersion, match="tool_call_id must be"):
        decode_step_result(forged)


def test_decode_wrong_kind_is_badversion():
    reset_frame = encode_reset_observation(_obs())
    step_frame = encode_step_result(_step())
    with pytest.raises(BadVersion):
        decode_step_result(reset_frame)
    with pytest.raises(BadVersion):
        decode_reset_observation(step_frame)
