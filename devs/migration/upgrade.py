"""One-time LiteSample forward migration helpers.

Run:
    uv run python -m devs.migration.run <input> -o <output> --verify

The migration transform is local to ``devs/migration``: runtime ``lite/``
readers and exporters should reject malformed new data instead of repairing old
provider envelopes. Its target is the current canonical LiteSample contract for
the allow-listed HF-published migration rows, not byte identity with a fresh
raw-source preprocessing run whose source policy may have changed.

**Policy is NOT re-declared here.** ``lite/data`` is the reference for current
row construction, so everything that is *policy* is imported from it rather than
mirrored:

* ``lite.data.utils.messages.structural_final_message`` -- the content-only
  ``Done.`` final turn (exact keys, exact order).
* ``lite.data.utils.messages.pop_terminal_terminate`` /
  ``terminate_outcome_others`` -- the ``use`` terminal-turn rule.
* ``lite.core.tools.calls.make_tool_call`` / ``stamp_messages_tool_call_ids`` --
  nested call construction and message-list id minting.
* ``lite.core.tools.extra_tools.LiteFinishToolSet.get_tool_schema`` /
  ``make_open_app_tool`` / ``LiteBrowserNavToolSet.get_tool_schemas`` -- canonical tool
  schema bodies.
* ``lite.core.tools.action_space.LiteDesktopActionSet`` / ``LiteMobileActionSet``
  (``get_action_names()``) -- GUI vocabulary.

What stays migration-local is exactly the OLD-INPUT reader surface, which has no
``lite/data`` equivalent because current preprocessing never sees it:
provider ``{"type":"function","function":{...}}`` envelopes, JSON-string
``arguments``, legacy ``metadata.extra_tools``, GUI-run batching of bare
top-level actions, noop ``screenshot`` / ``wait`` stripping, and conversion
of post-assistant user observations into ``role:"tool"`` result messages owned
by the producing call's ``id``.

Legacy parquet/HF materialization repair is local to this module:
``coerce_legacy_materialized_messages`` strips Arrow/HF padding before the old
provider repair starts. Current ``lite.data.staging.coerce_messages`` deliberately
does not perform that cleanup.

Images are COMPACTED exactly once, at the row write-out point (``_finalize_row``).
Migration drops TURNS -- noop ``screenshot`` / ``wait`` stripping does -- and
dropping a turn drops no picture, so without compaction the row would publish
images no message references and leave the surviving indices non-contiguous.
``devs.data.utils.compact_row_images`` (the one place allowed to renumber an
index) removes the orphans and renumbers the survivors ``0..N-1``, rewriting the
``images`` list and every ``{"type":"image","index": N}`` part in one step; an
index pointing outside the ORIGINAL list is still rejected rather than repaired.
Noop stripping may carry authored task content forward, but uses the shared
dev-side carry helper so stale observation screenshots are not promoted into
reference images.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from lite.core.errors import LiteContractError
from lite.core.metadata import LiteCUAMetadata, metadata_from_dict
from lite.core.tools.action_space import (
    LITE_ACTION_BATCH_TOOL_NAMES,
    LITE_DESKTOP_KEY_ACTION_NAMES,
    LiteDesktopActionSet,
    LiteMobileActionSet,
    lite_action_batch_tool_name_for_platform,
    lite_action_names_by_action_batch_tool,
    lite_builtin_tool_names_for_metadata,
)
from lite.core.tools.action_space.keys import normalize_keys

# Re-implementing ``stamp_messages_tool_call_ids`` is exactly the drift this
# module exists to avoid: migration must mint the same ``call_%04d`` ids, in the
# same message order, with the same collision-avoiding ``used`` set as the
# shared preproc finalization path.
from lite.core.tools.calls import (
    make_tool_call,
    stamp_messages_tool_call_ids,
    tool_call_arguments,
    tool_call_id,
    tool_call_name,
    validate_lite_tool_call,
)
from lite.core.tools.extra_tools import (
    BASH_TOOL_NAME,
    LiteBrowserNavToolSet,
    LiteFinishToolSet,
    LiteShellToolSet,
    make_open_app_tool,
)
from lite.core.tools.schemas import make_tool_schema, tool_schema_name
from lite.data.staging import coerce_messages as coerce_canonical_messages
from lite.data.staging import to_plain
from lite.data.utils.messages import (
    pop_terminal_terminate,
    structural_final_message,
    terminate_outcome_others,
)

# ``devs`` is not an installed package (``pyproject.toml`` ships ``lite*`` only),
# and this module is also loaded by path via ``importlib`` (see
# ``_load_sibling_module`` / ``run.py``), where nothing puts the repo root on
# ``sys.path``. Depth is per-file and differs from the filters': this file is
# ``<repo>/devs/migration/upgrade.py``, so the root is ``parents[2]``, not [3].
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from devs.data.utils import (  # noqa: E402  (needs _REPO_ROOT on sys.path)
    carry_content_without_observation_images,
    compact_row_images,
)


def _load_sibling_module(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"cua_lite_migration_{name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib guard
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:  # Package import, e.g. ``python -m devs.migration.run``.
    from .verify import verify_lite_sample as _verify_lite_sample
except ImportError:  # Direct script/importlib execution.
    _verify_lite_sample = _load_sibling_module("verify").verify_lite_sample


FINISH_TOOL_NAMES = LiteFinishToolSet.get_tool_names()
BROWSER_NAV_TOOLS = LiteBrowserNavToolSet.get_tool_names()
APP_LAUNCH_TOOL_NAME = tool_schema_name(make_open_app_tool())
APP_LAUNCH_TOOLS = frozenset({APP_LAUNCH_TOOL_NAME})
NAV_TOOLS = BROWSER_NAV_TOOLS | APP_LAUNCH_TOOLS
MIGRATION_LOCAL_TEXT_RESULT_TOOLS = frozenset({BASH_TOOL_NAME, "ask_user"})
STANDALONE_EXTRA_TOOLS = FINISH_TOOL_NAMES | NAV_TOOLS | MIGRATION_LOCAL_TEXT_RESULT_TOOLS
NOOP_GUI_ACTIONS = frozenset({"screenshot", "wait"})
LEGACY_TOP_LEVEL_DURABLE_METADATA_KEYS = (
    "env_id",
    "task_id",
    "episode_return",
    "terminated",
    "truncated",
)

#: Old on-disk rows already stored *Lite* action names: provider-native
#: spellings (Claude ``double_click``/``hotkey``, GPT ``move``/``keypress``,
#: mobile ``double_tap``) are normalized by
#: ``BaseActionSpace.convert_tool_calls_from_agent`` at the agent boundary and
#: never reach LiteSample storage. So there are no legacy input aliases to
#: accept here, and these are exactly the upstream catalogs.
#:
#: The only Lite actions ever *renamed* were mobile ``press_home`` /
#: ``press_back`` -> ``system_button`` (f91920072, 2026-03-18), which predates
#: the canonical staging pipeline (e4ee2fe32, 2026-06-01) -- so no staged row
#: can carry them and they are deliberately NOT accepted. If such a row ever
#: turns up, it fails loudly as a schema-less standalone tool rather than
#: silently batching; add it here as a documented alias only with real data.
DESKTOP_GUI_ACTIONS = LiteDesktopActionSet.get_action_names()
MOBILE_GUI_ACTIONS = LiteMobileActionSet.get_action_names()
ACTION_NAMES_BY_ACTION_BATCH_TOOL = lite_action_names_by_action_batch_tool()
BROWSER_PLATFORM = LiteCUAMetadata.Platform.BROWSER.value
LEGACY_WEB_PLATFORM = "web"
DESKTOP_PLATFORM = LiteCUAMetadata.Platform.DESKTOP.value
MOBILE_PLATFORM = LiteCUAMetadata.Platform.MOBILE.value


# ---------------------------------------------------------------------------
# Terminal-turn policy
# ---------------------------------------------------------------------------
#
# The terminal rule is NOT re-derived here. ``lite/data/preproc`` owns it, in
# two imported helpers -- ``messages.pop_terminal_terminate`` and
# ``messages.terminate_outcome_others`` -- and every ``use.py`` now ends an
# episode in exactly ONE way:
#
#   * no ``terminate`` call is ever persisted, and no ``terminate`` schema
#     survives in ``metadata.extra_tool_schemas``;
#   * the row ends on ``structural_final_message()`` -- the content-only
#     ``Done.``;
#   * whatever the dropped call asserted beyond "the episode ended" (a
#     non-success ``status`` and any authored ``reason``) moves to
#     ``metadata.others``.
#
# Old published rows always ended on a live terminate, so migration re-applies
# exactly that. Note the shape this replaced: the split used to keep a
# ``status="failure"`` call alive, and migration's local predicate was narrower
# still (``set(args) <= {"status"}``), which additionally made it miss every
# parquet row whose ``arguments`` were null-padded -- see migration's local
# legacy materialization normalizer below.
#
#: OLD-INPUT ``status`` spellings, normalized before the imported policy sees
#: them. ``lite/data`` writes only ``success``/``failure`` today, but published
#: rows predate that: ``completed`` / a missing ``status`` are legacy spellings
#: of "the episode ended", and ``fail`` is what scalecua's regex path stored
#: before it learned to normalize (``scalecua/use.py`` terminate pattern).
_LEGACY_STATUS_ALIASES = {
    "completed": "success",
    "fail": "failure",
}

#: Sources whose new preproc output is NOT recoverable from a published row.
#:
#: ``run.py`` refuses every non-Lite input path (``_require_allowed_lite_dataset
#: _path``) before this is consulted, so an entry here only guards a direct
#: library call to :func:`upgrade_lite_sample`.
_UNRECOVERABLE_SOURCES = {
    # Re-derived over the ENTIRE old published ``use`` corpus (13,170 rows:
    # web/use 5,501 + 193, mobile/use 7,246 + 230) and against a current
    # ``lite/data/preproc/guiact/use.py --head 300`` run of the same ids. Three
    # independent divergences, and NONE of them is a terminal/nonterminal
    # ambiguity -- an earlier revision of this entry gave that as the reason and
    # it is false: every one of the 13,170 rows ends on exactly one
    # ``terminate``, ``response`` occurs 11 times and all 11 are mid-episode,
    # and mobile/use has none. Trailing ``terminate.reason`` <=> terminal answer
    # is exact and total.
    #
    # * THE ANSWER IS LOST. preproc publishes the terminal ``terminate.reason``
    #   as the web cohort's final-turn text (5,617 rows carry non-blank text),
    #   but ``_apply_terminal_policy`` ends every row on
    #   ``structural_final_message()`` and ``terminate_outcome_others`` keeps
    #   nothing from a ``status="success"`` terminate -- which all 13,170 are.
    #   The 10 ``"task impossible"`` smartphone episodes likewise lose the
    #   ``terminate_status`` their subset rule assigns.
    # * 2,616 web rows fold the terminate onto a turn that still holds
    #   executable actions, which current preproc drops entirely. Popping the
    #   terminate leaves that call unpaired: the row migrates and verifies, but
    #   it is still not equivalent to current preproc and should be handled by
    #   publish policy.
    # * all 39,543 old mobile ``tap`` calls predate the ``clicks`` argument that
    #   ``LiteMobileActionSet.tap`` now emits, so 7,476/7,476 mobile rows differ.
    "yiye2023/GUIAct": (
        "current preproc publishes the terminal answer as the final turn's text "
        "and drops the terminal step's actions, while migration ends every row "
        "on the structural marker and leaves those actions unpaired"
    ),
}


def coerce_legacy_materialized_messages(messages: Any) -> list[Any]:
    """Normalize old parquet/HF message materialization for migration input.

    Published legacy parquet rows may arrive as nested Arrow/HF structs where
    fields absent from one sibling are padded as ``None`` on another. Migration
    owns that raw-boundary cleanup; current staging preserves such values.
    """
    messages = to_plain(messages)
    if isinstance(messages, str):
        return coerce_canonical_messages(messages)
    if not isinstance(messages, list):
        raise ValueError("messages JSON must decode to a list")
    normalized = coerce_canonical_messages(_strip_legacy_padded_message_nones(messages))
    return _strip_legacy_empty_tool_call_padding(normalized)


def _strip_legacy_padded_message_nones(value: Any) -> Any:
    """Strip Arrow/HF null padding while preserving invalid image-ref evidence."""
    if isinstance(value, dict):
        is_image_part = value.get("type") == "image"
        return {
            key: _strip_legacy_padded_message_nones(item)
            for key, item in value.items()
            if item is not None or (is_image_part and key == "index")
        }
    if isinstance(value, list):
        return [_strip_legacy_padded_message_nones(item) for item in value]
    return value


def _strip_legacy_empty_tool_call_padding(messages: list[Any]) -> list[Any]:
    out: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            out.append(message)
            continue
        msg = dict(message)
        if msg.get("tool_calls") == []:
            msg.pop("tool_calls")
        out.append(msg)
    return out


def upgrade_parquet_row(
    row: dict[str, Any],
    *,
    allow_lossy: bool = False,
) -> dict[str, Any]:
    """Upgrade a parquet-style row, preserving JSON-string fields as strings."""
    upgraded = copy.deepcopy(row)
    messages_was_json = isinstance(upgraded.get("messages"), str)
    metadata_was_json = isinstance(upgraded.get("metadata"), str)

    if messages_was_json:
        upgraded["messages"] = json.loads(upgraded["messages"])
    if metadata_was_json:
        upgraded["metadata"] = json.loads(upgraded["metadata"])

    upgraded = upgrade_lite_sample(
        upgraded,
        allow_lossy=allow_lossy,
    )

    if messages_was_json:
        upgraded["messages"] = json.dumps(upgraded["messages"], separators=(",", ":"))
    if metadata_was_json:
        upgraded["metadata"] = json.dumps(upgraded["metadata"], separators=(",", ":"))
    return upgraded


def upgrade_lite_sample(
    sample: dict[str, Any],
    *,
    allow_lossy: bool = False,
) -> dict[str, Any]:
    """Upgrade one old LiteSample-shaped dict into the canonical nested contract.

    Non-``use`` task families keep their task semantics untouched: old provider
    calls become nested Lite calls, but labels are not converted into tool
    results. Use rows are migrated whole-sample: old provider calls become
    nested calls, old top-level GUI action runs become ``computer``/``mobile``
    action-batch calls,
    post-assistant user observations become ``role:"tool"`` result messages
    owned by the screen-producing call, and the terminal turn follows
    ``lite.data.utils.messages.pop_terminal_terminate`` /
    ``terminate_outcome_others``.

    ``allow_lossy`` downgrades the ``_UNRECOVERABLE_SOURCES`` refusal to a
    best-effort migration; the result does not claim to recover the current
    raw-source preproc output.
    """
    row = copy.deepcopy(sample)
    metadata = _upgrade_task_type(_as_dict(to_plain(row.get("metadata")), field="metadata"))
    # Parquet stores legacy ``messages`` as one unified Arrow struct, so every
    # published row pads tool-call ``arguments`` and content parts with the union
    # of all sibling keys set to null (verified on cua-lite/Lite.OSWorld: a
    # terminate call reads back as {"coordinate":null,"keys":null,"text":null,
    # "status":"success"}). This cleanup is migration-owned: current staging
    # preserves producer evidence instead of deciding that nulls are padding.
    # ``metadata`` is deliberately NOT cleaned: preproc emits real ``None``
    # values there (``valid_actions``, ``others.os``, ``others.domain``).
    messages = _as_list(
        coerce_legacy_materialized_messages(row.get("messages")),
        field="messages",
    )

    if _is_nested_lite_input(messages):
        raise ValueError(
            "devs/migration accepts legacy-source rows only; refusing nested "
            "Lite input that has already been migrated or staged"
        )

    if metadata.get("task_type") != "use":
        platform = _canonical_platform(metadata.get("platform", DESKTOP_PLATFORM))
        return _finalize_row(
            row,
            messages=_nest_messages(messages, platform=platform),
            metadata=_upgrade_metadata(metadata, messages, suppress_schema_names=set()),
        )

    _reject_unrecoverable_source(metadata, allow_lossy=allow_lossy)
    platform = _canonical_platform(metadata.get("platform", DESKTOP_PLATFORM))
    upgraded_messages, suppress_schema_names, terminate_others = _upgrade_messages(
        messages,
        platform=platform,
    )
    return _finalize_row(
        row,
        messages=upgraded_messages,
        metadata=_upgrade_metadata(
            metadata,
            upgraded_messages,
            suppress_schema_names=suppress_schema_names,
            terminate_others=terminate_others,
        ),
    )


def _finalize_row(
    row: dict[str, Any],
    *,
    messages: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Attach the upgraded messages/metadata, validate, then compact the images.

    THE row-level write-out point: every ``upgrade_lite_sample`` exit goes
    through it, so compaction runs exactly ONCE with the whole row in hand --
    the way :func:`devs.data.utils.compact_row_images` requires. Calling it
    from inside the message transforms instead (``_strip_noop_gui_actions``,
    ``_flip_observations``, ...) is the failure mode its docstring warns about:
    each pass would renumber against a different image list and a surviving
    index would silently address the wrong picture.

    Validate BEFORE compacting, against the row's ORIGINAL image list. A
    reference that was already broken -- out of range, ``0.5``, ``None`` -- must
    be REJECTED with the verifier's own message, never quietly renumbered into
    range by compaction. After that check every surviving index is an in-range
    ``int``, which is exactly the input compaction is total on; it then asserts
    its own postcondition, and ``run.py --verify`` re-verifies the final row.

    The canonical output row always carries ``images``. Legacy text-only rows
    that had no image column normalize to an empty list.
    """
    row["messages"] = messages
    row["metadata"] = metadata
    images = _coerce_images(row.get("images", []))
    _validate_canonical_output(row["messages"], row["metadata"], images)
    row["images"], row["messages"] = _compact_images(images, row["messages"])
    return row


def _compact_images(
    images: Any,
    messages: list[dict[str, Any]],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Drop images no surviving message references, via the shared compactor.

    ``compact_row_images`` is the one function allowed to renumber an image
    index, and it speaks published path strings. A migration row carries either
    those or live PIL images (``upgrade_lite_sample`` takes a LiteSample-shaped
    dict, whose ``images`` are ``PIL.Image``), so it is handed one positional
    stand-in per image and the surviving ORDER is applied back onto the real
    list. The renumbering decision stays in the shared function; migration only
    carries a payload it cannot type.
    """
    values = _coerce_images(images)
    kept, compacted = compact_row_images([str(i) for i in range(len(values))], messages)
    return [values[int(position)] for position in kept], compacted


def _reject_unrecoverable_source(metadata: dict[str, Any], *, allow_lossy: bool) -> None:
    source = (metadata.get("others") or {}).get("source")
    if allow_lossy or source not in _UNRECOVERABLE_SOURCES:
        return
    raise ValueError(
        f"cannot migrate rows from source {source!r} without source-policy loss: "
        f"{_UNRECOVERABLE_SOURCES[source]}. Re-run lite/data/preproc from the "
        "raw source, or pass allow_lossy=True to accept a lossy migration."
    )


def _upgrade_task_type(metadata: dict[str, Any]) -> dict[str, Any]:
    """Reject old non-Lite ``navigation`` rows instead of preserving a second spelling."""
    if metadata.get("task_type") != "navigation":
        return metadata
    raise ValueError(
        "task_type 'navigation' is non-Lite legacy compatibility and is "
        "not accepted by the Lite migration path"
    )


def _canonical_platform(platform: Any) -> str:
    value = str(platform or DESKTOP_PLATFORM)
    if value == LEGACY_WEB_PLATFORM:
        return BROWSER_PLATFORM
    return value


def _action_batch_tool_name_for_platform(platform: str) -> str | None:
    return lite_action_batch_tool_name_for_platform(_canonical_platform(platform))


def _upgrade_metadata(
    metadata: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    suppress_schema_names: set[str],
    terminate_others: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(metadata)

    # The dropped terminate's payload is appended to ``others`` at the canonical
    # migration position. ``setdefault`` never overwrites -- if a legacy source
    # already published its own value for one of these keys, that value is the
    # authority.
    for key, value in (terminate_others or {}).items():
        out.setdefault("others", {}).setdefault(key, value)

    # Published Lite.ScaleCUA migration rows still spell the field ``extra_tools``.
    # Current tagged metadata reads ``extra_tool_schemas``, so leaving it would
    # silently drop the schemas AND leave an unknown key in the row.
    if "extra_tools" in out:
        legacy = out.pop("extra_tools")
        out.setdefault("extra_tool_schemas", legacy)

    others = out.setdefault("others", {})
    if not isinstance(others, dict):
        others = {}
        out["others"] = others
    for key in LEGACY_TOP_LEVEL_DURABLE_METADATA_KEYS:
        if key in out:
            others.setdefault(key, out.pop(key))

    # Every ``lite/data/preproc`` script hardcodes ``valid_actions: None``; the
    # canonical action surface is the tool schemas, not a name filter.
    out["valid_actions"] = None

    by_name: dict[str, dict[str, Any]] = {}
    for schema in out.get("extra_tool_schemas") or []:
        canonical_schema = _nest_schema(schema)
        name = tool_schema_name(canonical_schema)
        if name in suppress_schema_names:
            continue
        if name in by_name:
            raise ValueError(f"duplicate extra_tool_schemas name {name!r}")
        by_name[name] = canonical_schema

    requested = (
        _schema_names_from_old_valid_actions(metadata) | _standalone_tool_names(messages)
    ) - suppress_schema_names
    for name in requested:
        if name in STANDALONE_EXTRA_TOOLS and name not in by_name:
            by_name[name] = _default_schema(name)

    out["extra_tool_schemas"] = [by_name[name] for name in sorted(by_name, key=_schema_sort_key)]

    platform = _canonical_platform(out.pop("platform", DESKTOP_PLATFORM))
    task_type = out.pop("task_type", "use")
    extra_tool_schemas = out.pop("extra_tool_schemas", [])
    valid_actions = out.pop("valid_actions", None)
    others = out.pop("others", {})
    if not isinstance(others, dict):
        others = {}
    for key, value in out.items():
        others.setdefault(key, value)

    return LiteCUAMetadata(
        dims=(str(platform), str(task_type)),
        extra_tool_schemas=extra_tool_schemas,
        valid_actions=valid_actions,
        others=others,
    ).to_dict()


def _upgrade_messages(
    messages: list[dict[str, Any]],
    *,
    platform: str,
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    nested = [
        _upgrade_assistant_message(msg, platform=platform)
        if msg.get("role") == "assistant"
        else copy.deepcopy(msg)
        for msg in messages
    ]
    out, suppress_schema_names, terminate_others = _apply_terminal_policy(
        nested,
        platform=platform,
    )
    out = _strip_noop_gui_actions(out, platform=platform)
    # Minted AFTER the terminal policy so a dropped terminate does not burn an
    # id -- exactly the ``finalize_use_messages`` ordering in preproc.
    stamp_messages_tool_call_ids(out, preserve=False)
    return _flip_observations(out, platform=platform), suppress_schema_names, terminate_others


def _strip_noop_gui_actions(
    messages: list[dict[str, Any]],
    *,
    platform: str,
) -> list[dict[str, Any]]:
    """Drop old noop GUI turns; ``_finalize_row`` compacts the images after.

    This pass renumbers NOTHING: it drops turns, and the image list it leaves
    behind is still the input row's. The turns dropped here are exactly what
    orphans an image, so the row-level write-out runs the shared compactor once
    over the final message list -- doing it here instead would renumber against
    a list the later passes have not seen yet.

    From-scratch preproc does not persist standalone ``screenshot`` / ``wait``
    actions. Dropping a noop-only turn also drops the user turn it answered --
    otherwise two consecutive observations would survive and the second could
    not be owned by any call -- so that turn's authored content is carried
    forward onto the next user turn with the shared dev-side carry helper: text,
    metadata, and earlier indexed reference image parts survive unchanged, while
    the stale observation screenshot is not promoted (the next observation
    supersedes it), which is what leaves its image unreferenced.

    This holds for BOTH user turns: the goal prompt (whose instruction/goal
    images must reach the new turn-0) and a MID-EPISODE observation (whose text
    can be the only record that the producing action failed). Mid-episode the
    carry also re-homes that text correctly, because removing the noop turn
    makes the next observation the result of the same call the popped one
    described.
    """
    out: list[dict[str, Any]] = []
    carried: list[dict[str, Any]] = []
    action_batch_tool_name = _action_batch_tool_name_for_platform(platform)
    for msg in messages:
        if msg.get("role") != "assistant":
            if msg.get("role") == "user" and carried:
                msg = copy.deepcopy(msg)
                msg["content"] = carried + list(msg.get("content") or [])
                carried = []
            out.append(msg)
            continue

        kept: list[dict[str, Any]] = []
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            out.append(copy.deepcopy(msg))
            continue
        for call in tool_calls:
            name = tool_call_name(call)
            args = tool_call_arguments(call)
            actions = args.get("actions")
            if name == action_batch_tool_name and isinstance(actions, list):
                kept_actions = [
                    action for action in actions if action.get("action") not in NOOP_GUI_ACTIONS
                ]
                if kept_actions:
                    kept.append(
                        make_tool_call(
                            tool_call_name(call),
                            {**args, "actions": kept_actions},
                            call_id=tool_call_id(call),
                        )
                    )
                continue
            if _is_gui_action(str(name), platform=platform) and name in NOOP_GUI_ACTIONS:
                continue
            kept.append(call)

        if not kept and tool_calls:
            if out and out[-1].get("role") == "user":
                previous = out.pop()
                carried = (
                    carry_content_without_observation_images(previous.get("content") or [])
                    + carried
                )
            continue

        copied = copy.deepcopy(msg)
        copied["tool_calls"] = kept
        out.append(copied)
    return out


def _flip_observations(
    messages: list[dict[str, Any]],
    *,
    platform: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pending_result_call_id: str | None = None
    for msg in messages:
        if msg.get("role") == "assistant":
            out.append(msg)
            pending_result_call_id = _screen_result_call_id(msg, platform=platform)
            continue
        if msg.get("role") == "user" and pending_result_call_id is not None:
            # ``finalize_use_messages`` builds {role, tool_call_id, content} in that
            # order. The result belongs only to the screen-producing call that
            # minted ``pending_result_call_id``; text-only siblings do not own
            # the screenshot. Any further key the old row carried follows the
            # canonical block rather than being dropped.
            rest = {
                k: v
                for k, v in msg.items()
                if k not in {"role", "call_id", "tool_call_id", "content"}
            }
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": pending_result_call_id,
                    "content": msg["content"],
                    **rest,
                }
            )
            pending_result_call_id = None
            continue
        out.append(msg)
        pending_result_call_id = None

    # A screen-producing call in the FINAL assistant turn has no observation to
    # flip: the episode ended there, so no screenshot was ever captured. This is
    # the ordinary shape of cagui / guiodyssey / ui_genie_agent, whose semantic
    # terminate is folded onto the last action turn -- nothing to flip.
    return out


def _nest_messages(
    messages: list[dict[str, Any]],
    *,
    platform: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        copied = copy.deepcopy(msg)
        if copied.get("role") != "assistant":
            out.append(copied)
            continue
        canonical_calls = [
            _nest_call(call, platform=platform) for call in (copied.get("tool_calls") or [])
        ]
        if canonical_calls:
            copied["tool_calls"] = canonical_calls
        out.append(copied)
    stamp_messages_tool_call_ids(out, preserve=False)
    return out


def _normalize_old_terminate(call: dict[str, Any]) -> dict[str, Any]:
    """An old row's ``terminate`` call, with its status spelled canonically.

    The policy itself is imported, never re-implemented; the only thing added
    here is the OLD-INPUT normalization it cannot be asked to do -- see
    ``_LEGACY_STATUS_ALIASES``. A missing ``status`` after
    ``coerce_legacy_materialized_messages`` strips null-padded parquet struct
    fields means "the episode ended".
    """
    arguments = tool_call_arguments(call)
    status = arguments.get("status", "success")
    return make_tool_call(
        tool_call_name(call),
        {
            **arguments,
            "status": _LEGACY_STATUS_ALIASES.get(status, status),
        },
        call_id=tool_call_id(call),
    )


def _apply_terminal_policy(
    messages: list[dict[str, Any]],
    *,
    platform: str,
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    """Re-apply the shared ``lite/data`` terminal rule to an old row.

    Every terminate is dropped and the row ends on a fresh
    ``structural_final_message()``; the third return value is the
    ``metadata.others`` fragment that carries whatever the call asserted.

    The old final message's auxiliary fields and reasoning content go with the
    dropped terminate -- preproc builds a fresh 2-key ``{role, content}`` dict,
    so migration must not carry them over.
    """
    if not messages:
        return messages, set(), {}

    terminal_observation: dict[str, Any] | None = None
    terminal_messages = messages
    if (
        len(messages) >= 2
        and _is_screenshot_only_user(messages[-1])
        and messages[-2].get("role") == "assistant"
    ):
        terminal_observation = messages[-1]
        terminal_messages = messages[:-1]

    if terminal_messages[-1].get("role") != "assistant":
        return messages, set(), {}

    calls = terminal_messages[-1].get("tool_calls") or []
    if not calls:
        # A published row can already have a no-tool-call terminal assistant
        # turn. It is terminal, but legacy content channels such as
        # action_description are not trainable final text; normalize to the
        # same shared structural final used by preproc/filter.
        return _normalize_content_only_final(messages), set(), {}
    if tool_call_name(calls[-1]) != "terminate":
        # No terminator at all. This cannot come from a published preproc row --
        # every old ``use.py`` guaranteed a trailing terminate, and
        # ``devs/data/lite.*`` rollouts are staged only with first-class episode
        # outcome metadata -- so a ``Done.`` is NOT invented for it.
        return messages, set(), {}

    dropped = pop_terminal_terminate(terminal_messages)
    if (
        terminal_observation is not None
        and terminal_messages
        and terminal_messages[-1].get("role") == "assistant"
        and _has_screen_result_call(terminal_messages[-1], platform=platform)
    ):
        terminal_messages.append(terminal_observation)
    terminal_messages.append(structural_final_message())
    return (
        terminal_messages,
        {"terminate"},
        terminate_outcome_others(_normalize_old_terminate(dropped)),
    )


def _normalize_content_only_final(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not messages or messages[-1].get("role") != "assistant":
        return messages
    if messages[-1].get("tool_calls"):
        return messages
    canonical = structural_final_message()
    if messages[-1] == canonical:
        return messages
    return [*messages[:-1], canonical]


def _upgrade_assistant_message(
    msg: dict[str, Any],
    *,
    platform: str,
) -> dict[str, Any]:
    out = copy.deepcopy(msg)
    calls = [(call, _call_parts(call)) for call in (msg.get("tool_calls") or [])]
    if not calls:
        return out

    action_batch_tool_name = _action_batch_tool_name_for_platform(platform)
    if action_batch_tool_name is None:
        raise ValueError(f"no action-batch tool for platform {platform!r}")
    batched: list[dict[str, Any]] = []
    action_run: list[dict[str, Any]] = []

    def flush_action_run() -> None:
        if not action_run:
            return
        batched.append(
            make_tool_call(
                action_batch_tool_name,
                {"actions": list(action_run)},
            )
        )
        action_run.clear()

    for original, parts in calls:
        name = str(parts["name"])
        args = _coerce_arguments(parts.get("arguments"))
        if _is_gui_action(name, platform=platform):
            if "id" not in parts and "function" not in original:
                raise ValueError("missing id on canonical assistant tool call")
            action_run.append({"action": name, **_legacy_gui_arguments(name, args)})
            continue

        flush_action_run()
        batched.append(_nest_call(original, platform=platform))

    flush_action_run()
    upgraded = {"role": out.get("role", "assistant")}
    if content := out.get("content"):
        upgraded["content"] = content
    upgraded["tool_calls"] = batched
    upgraded.update(
        {key: value for key, value in out.items() if key not in {"role", "content", "tool_calls"}}
    )
    return upgraded


def _legacy_gui_arguments(name: str, args: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(args)
    if name in LITE_DESKTOP_KEY_ACTION_NAMES and out.get("keys") is not None:
        out["keys"] = normalize_keys(out["keys"])
    return out


def _legacy_tool_arguments(
    name: str,
    args: dict[str, Any],
    *,
    platform: str,
) -> dict[str, Any]:
    if name in LITE_ACTION_BATCH_TOOL_NAMES:
        out = copy.deepcopy(args)
        actions = out.get("actions")
        if isinstance(actions, list):
            out["actions"] = [
                _legacy_action_batch_child(action)
                for action in actions
            ]
        return out
    if _is_gui_action(name, platform=platform):
        return _legacy_gui_arguments(name, args)
    return copy.deepcopy(args)


def _legacy_action_batch_child(action: Any) -> Any:
    if not isinstance(action, dict):
        return copy.deepcopy(action)
    name = action.get("action")
    if not isinstance(name, str):
        return copy.deepcopy(action)
    args = {key: value for key, value in action.items() if key != "action"}
    return {"action": name, **_legacy_gui_arguments(name, args)}


def _nest_call(call: dict[str, Any], *, platform: str) -> dict[str, Any]:
    """Nested Lite call; a missing ``id`` is minted later."""
    parts = _call_parts(call)
    if "id" not in parts and "function" not in call:
        raise ValueError("missing id on canonical assistant tool call")
    return make_tool_call(
        parts["name"],
        _legacy_tool_arguments(
            str(parts["name"]),
            _coerce_arguments(parts.get("arguments")),
            platform=platform,
        ),
        call_id=parts.get("id"),
    )


def _call_parts(call: dict[str, Any]) -> dict[str, Any]:
    """Read one migration input call shape into ``name``/``arguments``/``id`` parts."""
    if "function" in call:
        fn = call.get("function")
        if not isinstance(fn, dict):
            raise ValueError("legacy tool_call.function must be an object")
        out = {
            "name": str(fn["name"]),
            "arguments": _coerce_arguments(fn.get("arguments")),
        }
        if "id" in call:
            out["id"] = call["id"]
        elif "call_id" in call:
            out["id"] = call["call_id"]
        return out
    if "name" in call:
        extra_keys = sorted(set(call) - {"call_id", "name", "arguments"})
        if extra_keys:
            raise ValueError(f"noncanonical keys on legacy bare tool call {extra_keys}")
        out = {
            "name": str(call["name"]),
            "arguments": _coerce_arguments(call.get("arguments")),
        }
        if "call_id" in call:
            out["id"] = call["call_id"]
        return out
    raise ValueError("tool call missing name/function")


def _nest_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if "function" in schema:
        extra_keys = sorted(set(schema) - {"type", "function", "strict"})
        if extra_keys:
            raise ValueError(f"extra_tool_schemas[*] has noncanonical outer keys {extra_keys}")
        if "type" in schema and schema["type"] != "function":
            raise ValueError("extra_tool_schemas[*].type must be 'function'")
        fn = schema.get("function")
        if not isinstance(fn, dict):
            raise ValueError("legacy extra_tool_schemas[*].function must be an object")
        return make_tool_schema(
            str(fn["name"]),
            description=str(fn.get("description") or ""),
            parameters=_legacy_schema_parameters(fn.get("parameters")),
            strict=_legacy_schema_strict(fn, schema),
        )
    if "name" not in schema:
        raise ValueError("extra_tool_schemas[*] missing name/function")
    extra_keys = sorted(set(schema) - {"type", "name", "description", "parameters", "strict"})
    if extra_keys:
        raise ValueError(f"extra_tool_schemas[*] has noncanonical keys {extra_keys}")
    if "type" in schema and schema["type"] != "function":
        raise ValueError("extra_tool_schemas[*].type must be 'function'")
    return make_tool_schema(
        str(schema["name"]),
        description=str(schema.get("description") or ""),
        parameters=_legacy_schema_parameters(schema.get("parameters")),
        strict=_legacy_schema_strict(schema),
    )


def _legacy_schema_parameters(parameters: Any) -> dict[str, Any]:
    params = copy.deepcopy(parameters or {"type": "object", "properties": {}, "required": []})
    if isinstance(params, dict) and params.get("properties") is None:
        params["properties"] = {}
    return params


def _legacy_schema_strict(*containers: Any) -> bool | None:
    """Return the old schema strict flag, preferring canonical function scope."""
    for container in containers:
        if isinstance(container, dict) and "strict" in container:
            return container["strict"]
    return None


def _coerce_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        if not arguments.strip():
            return {}
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError(f"tool_call arguments must be an object, got {type(arguments).__name__}")
    return copy.deepcopy(arguments)


def _is_gui_action(name: str, *, platform: str) -> bool:
    if name in STANDALONE_EXTRA_TOOLS:
        return False
    if platform == "mobile":
        return name in MOBILE_GUI_ACTIONS
    return name in DESKTOP_GUI_ACTIONS


def _screen_result_call_id(msg: dict[str, Any], *, platform: str) -> str | None:
    """Call whose result is the next observation.

    ``messages._screen_result_call`` takes the *first* result-boundary call
    in the turn, so this does too -- a turn that mixes an action-batch call with
    a nav tool must attach the screenshot to the same owning call on both paths.
    Finish and text-result-only calls never own a screenshot result.
    """
    for call in msg.get("tool_calls") or []:
        name = tool_call_name(call)
        if name in FINISH_TOOL_NAMES:
            continue
        if (
            name in LITE_ACTION_BATCH_TOOL_NAMES
            or _is_gui_action(str(name), platform=platform)
            or name in NAV_TOOLS
        ):
            return tool_call_id(call)
    return None


def _has_screen_result_call(msg: dict[str, Any], *, platform: str) -> bool:
    for call in msg.get("tool_calls") or []:
        name = tool_call_name(call)
        if name in FINISH_TOOL_NAMES:
            continue
        if (
            name in LITE_ACTION_BATCH_TOOL_NAMES
            or _is_gui_action(str(name), platform=platform)
            or name in NAV_TOOLS
        ):
            return True
    return False


def _is_screenshot_only_user(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "user"
        and isinstance(content, list)
        and len(content) == 1
        and isinstance(content[0], dict)
        and content[0].get("type") == "image"
    )


def _schema_names_from_old_valid_actions(metadata: dict[str, Any]) -> set[str]:
    valid_actions = to_plain(metadata.get("valid_actions"))
    if valid_actions is None:
        valid_actions = []
    if not isinstance(valid_actions, list):
        return set()
    return {str(name) for name in valid_actions if name in STANDALONE_EXTRA_TOOLS}


def _standalone_tool_names(messages: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            try:
                name = _call_parts(call)["name"]
            except (KeyError, TypeError, ValueError):
                continue
            if name in STANDALONE_EXTRA_TOOLS:
                names.add(name)
    return names


#: Legacy metadata key order accepted at the migration input boundary.
_CANONICAL_METADATA_KEYS = (
    "platform",
    "task_type",
    "extra_tool_schemas",
    "valid_actions",
    "others",
)

#: Mirrors ``lite/data/utils/messages._SCHEMAS_BY_NAME`` insertion order.
#: Names outside it are rollout-only extras and sort after, alphabetically.
_CANONICAL_SCHEMA_ORDER = (APP_LAUNCH_TOOL_NAME, "response", "terminate")


def _schema_sort_key(name: str) -> tuple[int, str]:
    if name in _CANONICAL_SCHEMA_ORDER:
        return (_CANONICAL_SCHEMA_ORDER.index(name), "")
    return (len(_CANONICAL_SCHEMA_ORDER), name)


def _default_schema(name: str) -> dict[str, Any]:
    """Canonical nested schema for a standalone extra tool.

    Bodies come from ``lite.core.tools.action_space`` builders, never from local literals:
    a duplicated description/parameter block is exactly what drifted before.
    """
    if name in FINISH_TOOL_NAMES:
        return LiteFinishToolSet.get_tool_schema(name)
    if name in APP_LAUNCH_TOOLS:
        return make_open_app_tool()
    if name == BASH_TOOL_NAME:
        return LiteShellToolSet.get_tool_schema(BASH_TOOL_NAME)
    if name in BROWSER_NAV_TOOLS:
        return LiteBrowserNavToolSet.get_tool_schemas(include=[name])[0]
    # ``ask_user`` is a rollout-only extra: no ``lite/data/preproc`` script emits
    # it, so there is no runtime-owned schema builder to import.
    return make_tool_schema(
        name,
        description=_ROLLOUT_ONLY_DESCRIPTIONS[name],
        parameters=copy.deepcopy(_ROLLOUT_ONLY_PARAMETERS[name]),
    )


_ROLLOUT_ONLY_DESCRIPTIONS = {
    "ask_user": "Ask the user for clarification.",
}

_ROLLOUT_ONLY_PARAMETERS = {
    "ask_user": {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Question for the user."}},
        "required": ["text"],
    },
}


def _as_dict(value: Any, *, field: str) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _as_list(value: Any, *, field: str) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _coerce_images(value: Any) -> list[Any]:
    """Parse a row's image list into plain python; order and count untouched.

    Pure parsing (JSON string / numpy array / tuple -> list). Which images
    survive and how they are numbered is decided by ``compact_row_images``
    alone, via :func:`_compact_images`.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError("images must be a list")
    return value


def _is_nested_lite_input(messages: list[dict[str, Any]]) -> bool:
    """True when the input is already in the nested Lite storage shape.

    Migration is a one-time legacy-source repair path. A row that already has a
    persisted nested Lite call id, or a ``role:"tool"`` result, belongs on the
    verify/stage path instead of being silently rewritten here.
    """
    for mi, msg in enumerate(messages):
        if msg.get("role") == "tool":
            return True
        for ti, call in enumerate(msg.get("tool_calls") or []):
            if (
                validate_lite_tool_call(
                    call,
                    f"messages[{mi}].tool_calls[{ti}]",
                    require_id=True,
                )
                is not None
            ):
                continue
            return True
    return False


def _validate_canonical_output(
    messages: list[dict[str, Any]],
    metadata: dict[str, Any],
    images: Any,
) -> None:
    # Validate references against the original image list before compaction.
    # Nothing is repaired or renumbered here; the verifier owns canonical
    # schema/tool checks for both CLI verification and post-upgrade assertions.
    _verify_lite_sample({"messages": messages, "metadata": metadata, "images": images})


def schema_free_names(metadata: dict[str, Any]) -> frozenset[str]:
    """Tool names valid without schemas for migration input/output checks.

    Runtime/current rows accept only wrapper-native ``grounding.action`` calls.
    Migration additionally accepts legacy bare GUI action labels from old
    parquet rows, keeping that compatibility local to ``devs/migration``.
    """
    try:
        lite_meta = metadata_from_dict(metadata)
    except LiteContractError:
        platform, task_type = _metadata_dims_or_legacy_fields(metadata)
        lite_meta = _lite_metadata_for_tool_policy(
            _canonical_platform(platform),
            task_type,
        )
    if not isinstance(lite_meta, LiteCUAMetadata):
        raise ValueError("migration schema-free tool policy only supports CUA metadata")
    return _migration_builtin_tool_names_for_metadata(lite_meta)


def _metadata_dims_or_legacy_fields(metadata: dict[str, Any]) -> tuple[str, str]:
    dims = metadata.get("dims")
    if isinstance(dims, (list, tuple)) and len(dims) >= 2:
        return str(dims[0]), str(dims[1])
    return (
        str(metadata.get("platform", DESKTOP_PLATFORM)),
        str(metadata.get("task_type", "use")),
    )


def _lite_metadata_for_tool_policy(platform: str, task_type: str) -> LiteCUAMetadata:
    return LiteCUAMetadata(dims=(platform, task_type))


def _migration_builtin_tool_names_for_metadata(metadata: LiteCUAMetadata) -> frozenset[str]:
    names = set(lite_builtin_tool_names_for_metadata(metadata))
    if metadata.task_type == LiteCUAMetadata.TaskType.GROUNDING_ACTION:
        if metadata.platform == LiteCUAMetadata.Platform.MOBILE:
            names.update(MOBILE_GUI_ACTIONS)
        else:
            names.update(DESKTOP_GUI_ACTIONS)
    return frozenset(names)


def _is_top_level_use_action(name: str, *, platform: str, task_type: str) -> bool:
    return (
        task_type == "use"
        and _action_batch_tool_name_for_platform(platform) is not None
        and _is_gui_action(name, platform=platform)
    )


__all__ = [
    "coerce_legacy_materialized_messages",
    "schema_free_names",
    "upgrade_lite_sample",
    "upgrade_parquet_row",
]
