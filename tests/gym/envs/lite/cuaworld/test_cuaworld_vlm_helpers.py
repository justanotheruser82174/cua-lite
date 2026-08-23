"""CUAWorld tests split from _cuaworld_support.py: vlm helpers."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from lite.gym.envs.lite.cuaworld.src import software
from lite.gym.envs.lite.cuaworld.src.adapter import run_cuaworld_verify
from tests.gym.envs.lite._cuaworld_support import (
    _VLM_SCHEMA_JSON,
    _cuaworld_root,
    _FakeInterface,
    _jpeg_bytes,
    _materials_root,
    _png_bytes,
    _stub_completion,
    _verifier_task,
)


def test_env_info_supplies_every_vlm_alias_the_pinned_verifiers_read():
    """`env_info` must answer to every name a verifier uses for the judge.

    Reads are all `.get()`, so a missing alias is `None`, the branch is skipped, and
    the points vanish with no error. The exact alias census moves with the pinned
    materials; the adapter must cover every VLM-flavored key present in that tree.
    """
    import ast as ast_module
    import warnings

    root = _materials_root()
    sources = sorted(root.glob("*/*/tasks/*/**/*.py"))
    if not sources:
        pytest.skip("cuaworld materials not fetched")

    read: dict[str, set[tuple[str, str]]] = {}
    for source in sources:
        try:
            with warnings.catch_warnings():
                # A few upstream verifiers carry invalid escapes; compiling them
                # here is a census, not an execution.
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast_module.parse(
                    source.read_text(encoding="utf-8", errors="surrogateescape")
                )
        except SyntaxError:
            continue  # 6 upstream files do not parse; unchanged by anything here
        software, _, _, task = source.relative_to(root).parts[:4]
        for node in ast_module.walk(tree):
            key = None
            if (
                isinstance(node, ast_module.Subscript)
                and isinstance(node.value, ast_module.Name)
                and node.value.id == "env_info"
                and isinstance(node.slice, ast_module.Constant)
                and isinstance(node.slice.value, str)
            ):
                key = node.slice.value
            elif (
                isinstance(node, ast_module.Call)
                and isinstance(node.func, ast_module.Attribute)
                and node.func.attr in ("get", "pop")
                and isinstance(node.func.value, ast_module.Name)
                and node.func.value.id == "env_info"
                and node.args
                and isinstance(node.args[0], ast_module.Constant)
                and isinstance(node.args[0].value, str)
            ):
                key = node.args[0].value
            if key is not None:
                read.setdefault(key, set()).add((software, task))

    supported_vlm_aliases = {
        "query_vlm",
        "vlm",
        "vlm_query",
        "call_vlm",
    }
    read_vlm_aliases = {key for key in read if "vlm" in key}
    assert read_vlm_aliases
    assert read_vlm_aliases <= supported_vlm_aliases
    if "vlm" in read:
        assert read["vlm"] == {
            ("vlc_media_player", "corporate_training_localization_dubbing")
        }
    if "call_vlm" in read:
        assert read["call_vlm"] == {
            ("slicer3d", "measure_vertebral_compression_ratio")
        }
    if "vlm_query" in read:
        assert read["vlm_query"] == {
            ("slicer3d", "calculate_evans_index"),
            ("slicer3d", "split_segment_scissors"),
        }

    # The one that is structurally unpassable without the alias: +20 and +15 are the
    # only reachable points, +25/+20/+20 sit behind `if … and vlm`, gate is 70.
    verifier = (
        root
        / "vlc_media_player/vlc_media_player_env/tasks"
        / "corporate_training_localization_dubbing/verifier.py"
    )
    if verifier.is_file():
        text = verifier.read_text()
        assert "vlm = env_info.get('vlm')" in text
        assert "passed = score >= 70 and key_criteria_met" in text
        assert "if frame_copied and vlm:" in text and "if vlm and traj:" in text
        excludes = json.loads(
            (_cuaworld_root() / "data/validation_excludes.json").read_text()
        )
        assert not (excludes.get("vlc_media_player") or {}).get(
            "corporate_training_localization_dubbing"
        )


@pytest.mark.asyncio
async def test_verifier_vlm_aliases_are_the_same_callable(tmp_path):
    """End to end through the real verifier bridge: all four spellings arrive, and
    they are the one `query_vlm` whose envelope these verifiers read."""
    task = _verifier_task(
        tmp_path / "vlm-aliases",
        "from lite.gym.envs.lite.cuaworld.src.vlm import query_vlm; "
        "names = ('query_vlm', 'vlm', 'vlm_query', 'call_vlm'); "
        "return {'score': 100 if all("
        "env_info.get(n) is query_vlm for n in names) else 0}",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")

    reward, info = await run_cuaworld_verify(computer, task, "verifier.py::verify")

    assert reward == 1.0, info


def test_sample_trajectory_frames_honors_endpoint_flags(tmp_path):
    from lite.gym.envs.lite.cuaworld.src.vlm import sample_trajectory_frames

    frames = []
    for index in range(5):
        path = tmp_path / f"frame_{index}.png"
        path.write_bytes(_png_bytes())
        frames.append(str(path))

    assert sample_trajectory_frames(
        {"frames": frames},
        num_samples=2,
        include_first=False,
        include_last=False,
    ) == frames[1:3]
    assert sample_trajectory_frames(
        {"frames": frames},
        num_samples=2,
        include_first=True,
        include_last=False,
    ) == [frames[0], frames[2]]


def test_vlm_helpers_match_upstream_config_and_final_screenshot_contract(
    tmp_path, monkeypatch
):
    from lite.gym.envs.lite.cuaworld.src.vlm import (
        get_final_screenshot,
        get_vlm_config,
    )

    monkeypatch.setenv("VLM_MODEL", "openai/local-model")
    monkeypatch.setenv("VLM_BASE_URL", "http://localhost:9000/v1")
    monkeypatch.setenv("VLM_MAX_RETRIES", "4")
    config = get_vlm_config()
    assert config["model"] == "openai/local-model"
    assert config["base_url"] == "http://localhost:9000/v1"
    assert config["max_retries"] == 4

    final = tmp_path / "final.png"
    post = tmp_path / "post.png"
    final.write_bytes(_png_bytes())
    post.write_bytes(_png_bytes())
    assert get_final_screenshot({
        "final_screenshot": str(final),
        "post_verification_screenshot": str(post),
    }) == str(post)


def test_vlm_config_is_a_model_id_and_nothing_else(monkeypatch):
    """The provider comes from the model id; litellm resolves the rest.

    There is deliberately no backend switch and no per-provider key table: naming
    `anthropic/claude-…` is how you reach Anthropic, and its credentials come from
    the environment litellm already reads — exactly like the agent side, where
    `--model-id gpt-5.5` plus OPENAI_BASE_URL/OPENAI_API_KEY is the whole config.
    """
    from lite.gym.envs.lite.cuaworld.src.vlm import get_vlm_config

    for key in ("VLM_MODEL", "VLM_BASE_URL", "VLM_API_KEY", "LITE_CUAWORLD_VLM_MODEL"):
        monkeypatch.delenv(key, raising=False)

    default = get_vlm_config()
    # The default comes FROM configs/default.yaml, so assert the wiring, not a literal.
    # Pinning the literal here would mean editing this test every time the judge model
    # changes — which is exactly the coupling that moving it into the config removed.
    declared = software.CFG.env_kwargs["judge"]["model"]
    assert isinstance(declared, str) and declared, "no judge model declared in the yaml"
    assert default["model"] == declared
    # Unset => omitted from the request, so litellm's own resolution wins. An
    # empty string here would override it with a broken value.
    assert default["base_url"] is None
    assert default["api_key"] is None
    # timeout is the one thing that is NOT left to litellm: unset must still mean a
    # bound, or a stalled endpoint hangs the verifier thread and the enclosing /step
    # forever. This line previously pinned `is None`, which is what let the floor be
    # deleted without any test noticing.
    assert default["timeout"] == 180.0

    assert get_vlm_config({"VLM_MODEL": "anthropic/claude-test"})["model"] == (
        "anthropic/claude-test"
    )
    # The DOCUMENTED self-hosted recipe — a model id and a base_url, no key (see
    # vlm.py's module docstring and devs/data/lite.cuaworld/AGENTS.md). It has to work
    # exactly as written: litellm's OpenAI path rejects api_key=None client-side, so
    # get_vlm_config supplies the placeholder these servers ignore. Passing
    # VLM_API_KEY here instead would test a recipe nobody is told to use.
    served = get_vlm_config({
        "VLM_MODEL": "openai/Qwen/Qwen3-VL-8B-Instruct",
        "VLM_BASE_URL": "http://localhost:8080",
    })
    assert served["model"] == "openai/Qwen/Qwen3-VL-8B-Instruct"
    assert served["base_url"] == "http://localhost:8080"
    assert served["api_key"] == "EMPTY"


def test_vlm_config_validation_and_redacted_smoke_output(monkeypatch):
    from lite.gym.envs.lite.cuaworld.src.vlm import get_vlm_config, redacted_vlm_config

    monkeypatch.setenv("VLM_MODEL", "   ")
    with pytest.raises(ValueError, match="VLM model is empty"):
        get_vlm_config()

    monkeypatch.setenv("VLM_MODEL", "openai/local")
    monkeypatch.setenv("VLM_MAX_RETRIES", "0")
    with pytest.raises(ValueError, match="VLM_MAX_RETRIES"):
        get_vlm_config()

    monkeypatch.setenv("VLM_MAX_RETRIES", "1")
    monkeypatch.setenv("VLM_TIMEOUT", "nan")
    with pytest.raises(ValueError, match="VLM_TIMEOUT"):
        get_vlm_config()

    monkeypatch.setenv("VLM_TIMEOUT", "60")
    monkeypatch.setenv("VLM_API_KEY", "secret-token")
    assert redacted_vlm_config()["api_key"] == "<redacted>"


def test_vlm_never_fabricates_a_credential(monkeypatch):
    """An unset key stays ``None`` so the request omits it entirely.

    (This replaces a test that pinned a per-provider key table —
    ANTHROPIC_API_KEY / GEMINI_API_KEY / VLM_API_KEY with an ``"EMPTY"`` default
    for the local backend. That table is gone: litellm reads whichever provider's
    credential the model id implies. What still matters is that we never invent a
    value — forwarding `""` would override litellm's own resolution with a broken
    key, which is worse than sending nothing.)
    """
    from lite.gym.envs.lite.cuaworld.src.vlm import get_vlm_config

    monkeypatch.delenv("VLM_API_KEY", raising=False)
    monkeypatch.delenv("VLM_BASE_URL", raising=False)
    assert get_vlm_config()["api_key"] is None
    assert get_vlm_config({"VLM_MODEL": "anthropic/claude-test"})["api_key"] is None

    monkeypatch.setenv("VLM_API_KEY", "explicit")
    assert get_vlm_config()["api_key"] == "explicit"
    # …and an explicit key is never clobbered by the self-hosted placeholder.
    assert get_vlm_config({"VLM_BASE_URL": "http://localhost:8080"})["api_key"] == (
        "explicit"
    )

    # THE BOUNDARY. The rule above is "don't override litellm's resolution with a
    # broken value" — which only applies where litellm HAS a resolution. Once a
    # base_url is given there is no provider to resolve against, and litellm's OpenAI
    # path raises on api_key=None before sending anything, so the placeholder is the
    # only thing that makes a self-hosted endpoint reachable.
    monkeypatch.delenv("VLM_API_KEY", raising=False)
    assert get_vlm_config({"VLM_BASE_URL": "http://localhost:8080"})["api_key"] == (
        "EMPTY"
    )


def test_gym_anything_vlm_accepts_n_used_by_pinned_verifiers():
    from lite.gym.envs.lite.cuaworld.src.vlm import sample_trajectory_frames

    assert sample_trajectory_frames(
        {"frames": ["a", "b", "c"]},
        n=2,
    ) == ["a", "c"]


def test_top_level_vlm_utils_preserves_upstream_n_wrapper():
    from lite.gym.envs.lite.cuaworld.src import vlm_utils

    assert vlm_utils.sample_trajectory_frames(
        {"frames": ["a", "b", "c"]},
        n=2,
    ) == ["a", "c"]
    # Pinned upstream adds both endpoints before applying the remaining budget.
    assert vlm_utils.sample_trajectory_frames(
        {"frames": ["a", "b", "c"]},
        n=1,
    ) == ["a", "c"]


def test_vlm_frame_helpers_preserve_upstream_path_semantics():
    from lite.gym.envs.lite.cuaworld.src.vlm import (
        get_final_screenshot,
        sample_trajectory_frames,
    )

    missing = "/path/that/does/not/exist.png"
    assert sample_trajectory_frames({"frames": [missing]}) == [missing]
    assert sample_trajectory_frames({"final_screenshot": missing}) == [missing]
    assert get_final_screenshot({"frames": [missing]}) is None


def test_query_vlm_provider_failure_raises(tmp_path, monkeypatch):
    from lite.gym.envs.lite.cuaworld.src.vlm import VLMProviderError, query_vlm

    image = tmp_path / "frame.png"
    image.write_bytes(_png_bytes())
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-secret")
    fake = SimpleNamespace(
        completion=lambda **_kwargs: (_ for _ in ()).throw(
            TimeoutError("api_key=unit-test-secret judge timeout")
        )
    )
    monkeypatch.setitem(sys.modules, "litellm", fake)
    with pytest.raises(VLMProviderError) as raised:
        query_vlm(
            "judge",
            images=[str(image)],
            config={"max_retries": 1},
        )
    assert "judge timeout" in str(raised.value)
    assert "unit-test-secret" not in str(raised.value)


def test_query_vlm_retries_with_capped_jittered_backoff(tmp_path, monkeypatch):
    """The judge owns its backoff locally (``lite/gym`` never imports ``lite/agents``).

    Pins the formula itself: capped ``2**attempt`` seconds scaled by U(0.5, 1.5).
    """
    from lite.gym.envs.lite.cuaworld.src import vlm

    monkeypatch.setattr(vlm.random, "random", lambda: 0.5)
    assert vlm._retry_delay(0) == 1.0  # 2**0 * (0.5 + 0.5)
    assert vlm._retry_delay(20) == 60.0  # capped at 60s, then jittered

    image = tmp_path / "frame.png"
    image.write_bytes(_png_bytes())
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-secret")
    sleeps: list[float] = []
    calls = {"n": 0}

    def completion(**_kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise TimeoutError("transient judge timeout")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    monkeypatch.setattr(vlm.time, "sleep", sleeps.append)

    result = vlm.query_vlm(
        "judge",
        images=[str(image)],
        config={"max_retries": 3},
    )

    assert result["response"] == "ok"
    assert calls["n"] == 3
    assert sleeps == [1.0, 2.0]


def test_query_vlm_rejects_missing_image_instead_of_text_only(tmp_path, monkeypatch):
    from lite.gym.envs.lite.cuaworld.src.vlm import VLMProviderError, query_vlm

    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
        )

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    with pytest.raises(VLMProviderError, match="no valid VLM images"):
        query_vlm("judge", images=[str(tmp_path / "missing.png")])
    assert captured == {}


def test_query_vlm_matches_upstream_local_model_and_content_order(
    tmp_path, monkeypatch
):
    from lite.gym.envs.lite.cuaworld.src.vlm import query_vlm

    image = tmp_path / "frame.png"
    image.write_bytes(_png_bytes())
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
        )

    for key in (
        "VLM_BACKEND",
        "VLM_MODEL",
        "VLM_BASE_URL",
        "LITE_CUAWORLD_VLM_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    # Name the model you want served locally — that IS the whole configuration
    # now (no backend switch, no implicit `openai/` prefixing). This test pins the
    # upstream local model id and the image-before-text content order.
    monkeypatch.setenv("VLM_MODEL", "openai/Qwen/Qwen3-VL-8B-Instruct")
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))

    result = query_vlm("judge", images=[str(image)])

    assert result["success"] is True
    assert captured["model"] == "openai/Qwen/Qwen3-VL-8B-Instruct"
    content = captured["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[-1] == {"type": "text", "text": "judge"}


def test_query_vlm_rejects_empty_image_instead_of_text_only(tmp_path, monkeypatch):
    from lite.gym.envs.lite.cuaworld.src.vlm import VLMProviderError, query_vlm

    image = tmp_path / "frame.png"
    image.write_bytes(b"")
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="yes"))]
        )

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    with pytest.raises(VLMProviderError, match="no valid VLM images"):
        query_vlm("judge", images=[str(image)])
    assert captured == {}


@pytest.mark.parametrize(
    ("suffix", "payload", "mime"),
    [
        (".png", _png_bytes(), "image/png"),
        (".jpg", _jpeg_bytes(), "image/jpeg"),
    ],
)
def test_query_vlm_encodes_decodable_images_by_content(
    tmp_path, monkeypatch, suffix, payload, mime
):
    from lite.gym.envs.lite.cuaworld.src.vlm import query_vlm

    image = tmp_path / f"frame{suffix}"
    image.write_bytes(payload)
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="yes"))]
        )

    fake = SimpleNamespace(
        completion=completion
    )
    monkeypatch.setitem(sys.modules, "litellm", fake)
    result = query_vlm("judge", images=[str(image)])
    assert result["success"] is True
    assert result["parsed"] == {"answer": True}
    image_part = captured["messages"][0]["content"][0]
    assert image_part["image_url"]["url"].startswith(f"data:{mime};base64,")


@pytest.mark.parametrize(
    "text",
    [
        _VLM_SCHEMA_JSON,
        f"```json\n{_VLM_SCHEMA_JSON}\n```",
        f"```\n{_VLM_SCHEMA_JSON}\n```",
        f"Here is my analysis.\n{_VLM_SCHEMA_JSON}\nLet me know if you need more.",
    ],
)
@pytest.mark.parametrize(
    ("kwarg", "value"),
    [
        # The complete set of JSON-requesting kwargs in the pinned corpus, by AST
        # census over 920 call sites: return_json 3, json_response 1, output_schema 2.
        # `response_model` is passed by ZERO verifiers, and `format_response`'s single
        # site (gvsig/convert_vector_to_raster) asks a yes/no question and reads
        # `.get("answer_bool")` — it wants a bool, not a parsed schema, so treating it
        # as a JSON kwarg returned `{}` and cost it the points either way.
        ("return_json", True),
        ("json_response", True),
        ("output_schema", {"currency_hidden": "bool"}),
    ],
)
def test_query_vlm_json_kwargs_return_the_parsed_object(monkeypatch, text, kwarg, value):
    """The pinned verifiers subscript the RESULT (`vlm_response.get("currency_hidden")`),
    so a JSON-requesting call must hand back the parsed object, not the envelope."""
    from lite.gym.envs.lite.cuaworld.src.vlm import query_vlm

    _stub_completion(monkeypatch, text)
    result = query_vlm("judge", images=[], **{kwarg: value})
    assert result == {
        "currency_hidden": True,
        "percent_visible": True,
        "symbols_visible": False,
    }


def test_query_vlm_without_json_kwarg_keeps_the_envelope(monkeypatch):
    from lite.gym.envs.lite.cuaworld.src.vlm import query_vlm

    _stub_completion(monkeypatch, _VLM_SCHEMA_JSON)
    result = query_vlm("judge", images=[])
    assert isinstance(result, dict)
    assert result["success"] is True
    assert result["parsed"]["currency_hidden"] is True
    assert result.strip() == _VLM_SCHEMA_JSON
    assert result.upper() == _VLM_SCHEMA_JSON.upper()
    assert result.get("currency_hidden") is True
    assert result["percent_visible"] is True
    assert "currency_hidden" in result
    assert "success" in result
    # `options` is a multiple-choice request, not a JSON one — its two call sites want
    # incompatible types back, so it stays absorbed and the envelope is unchanged.
    assert query_vlm("judge", images=[], options=["YES", "NO"])["success"] is True


def test_query_vlm_json_kwarg_wraps_a_bare_top_level_array(monkeypatch):
    from lite.gym.envs.lite.cuaworld.src.vlm import query_vlm

    _stub_completion(monkeypatch, '[{"symbol": "AAPL"}]')
    assert query_vlm("judge", images=[], return_json=True) == {
        "items": [{"symbol": "AAPL"}]
    }


def test_query_vlm_json_kwarg_unparseable_is_falsy_not_an_all_false_answer(monkeypatch):
    """Garbage must degrade to a FALSY dict: `.get` still answers None (no crash, no
    points) and a caller testing `if vlm_response:` reports "no response" instead of
    mistaking a dead judge for a confident all-false verdict."""
    from lite.gym.envs.lite.cuaworld.src.vlm import query_vlm

    _stub_completion(monkeypatch, "I'm sorry, I cannot analyze images right now.")
    result = query_vlm("judge", images=[], return_json=True)
    assert result == {}
    assert not result
    assert result.get("currency_hidden") is None
    # The envelope path keeps upstream's yes/no keyword fallback, which is exactly the
    # truthy `{"answer": ...}` the JSON path must not return.
    assert query_vlm("judge", images=[])["parsed"] == {"answer": False}


def test_query_vlm_json_kwarg_provider_failure_raises(tmp_path, monkeypatch):
    """A dead judge must NOT look like a confident all-False verdict.

    Unparseable model TEXT degrades to `{}` (the test above) — that is a real answer
    we could not read. A PROVIDER failure is different: every JSON call site wraps
    `query_vlm` in `except Exception` and grants fixed consolation points there, so
    swallowing the failure into `{}` silently deletes those points and scores the
    episode 0 for an outage the agent had nothing to do with."""
    from lite.gym.envs.lite.cuaworld.src.vlm import VLMProviderError, query_vlm

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(
            completion=lambda **_kwargs: (_ for _ in ()).throw(TimeoutError("judge timeout"))
        ),
    )
    with pytest.raises(VLMProviderError):
        query_vlm("judge", images=[], return_json=True, config={"max_retries": 1})

    with pytest.raises(VLMProviderError):
        query_vlm("judge", images=[], config={"max_retries": 1})


def test_query_vlm_json_kwarg_keeps_positional_swap_and_str_images(tmp_path, monkeypatch):
    from lite.gym.envs.lite.cuaworld.src.vlm import query_vlm

    image = tmp_path / "frame.png"
    image.write_bytes(_png_bytes())
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=_VLM_SCHEMA_JSON))]
        )

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    # `query_vlm(frames, prompt, output_schema=…)` — gvsig/extract_country_boundaries.
    result = query_vlm([str(image)], "judge", output_schema={"a": "bool"})
    content = captured["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[-1] == {"type": "text", "text": "judge"}
    assert result["currency_hidden"] is True
    # A bare path where a list is expected must not splat into one "path" per char.
    query_vlm("judge", images=str(image), return_json=True)
    content = captured["messages"][0]["content"]
    assert sum(1 for part in content if part["type"] == "image_url") == 1


def test_query_vlm_json_kwarg_accepts_trajectory_dict_images(tmp_path, monkeypatch):
    """`gcompris/traffic_puzzle` passes `images=traj`, not sampled frames.

    The shim must adapt that upstream mistake into real image paths. Otherwise the
    judge receives a text-only request and returns a confident false negative.
    """
    from lite.gym.envs.lite.cuaworld.src.vlm import query_vlm

    frames = []
    for index in range(6):
        image = tmp_path / f"frame_{index}.png"
        image.write_bytes(_png_bytes())
        frames.append(str(image))
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=_VLM_SCHEMA_JSON))]
        )

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))

    result = query_vlm("judge", images={"frames": frames}, json_response=True)

    assert result["currency_hidden"] is True
    content = captured["messages"][0]["content"]
    assert sum(1 for part in content if part["type"] == "image_url") == 5
    assert content[-1] == {"type": "text", "text": "judge"}


def test_query_vlm_json_kwarg_makes_the_jstock_scoring_block_reachable(monkeypatch):
    """Mirrors jstock/anonymize_portfolio_view's scoring block verbatim.

    Some verifiers do not pass a JSON-return kwarg but still read schema keys with
    ``.get``. The ordinary envelope must therefore fall back to ``parsed`` too.
    """
    from lite.gym.envs.lite.cuaworld.src.vlm import query_vlm

    def score_like_the_verifier(vlm_response):
        vlm_data = vlm_response if isinstance(vlm_response, dict) else {}
        score = 20  # the two programmatic file checks
        score += 20 if vlm_data.get("symbols_visible") else 0
        score += 20 if vlm_data.get("percent_visible") else 0
        score += 40 if vlm_data.get("currency_hidden") else 0
        return score

    all_true = '{"currency_hidden": true, "percent_visible": true, "symbols_visible": true}'
    _stub_completion(monkeypatch, all_true)
    assert score_like_the_verifier(query_vlm("judge", images=[])) == 100
    assert score_like_the_verifier(
        query_vlm("judge", images=[], return_json=True)
    ) == 100
