"""Registry platform metadata for public model action-space keys."""

from __future__ import annotations

import importlib

import pytest

from lite.agents.core.action_space.base import (
    ActionSpaceRegistry,
    LiteBBoxActionSpace,
    LitePointActionSpace,
)
from lite.core.metadata import LiteCUAMetadata

_ACTION_SPACE_MODULES = (
    "lite.agents.models.claude.action_space",
    "lite.agents.models.evocua.action_space",
    "lite.agents.models.fara.action_space",
    "lite.agents.models.gemini.action_space",
    "lite.agents.models.gpt.action_space",
    "lite.agents.models.mai_ui.action_space",
    "lite.agents.models.qwen2_5_vl.action_space",
    "lite.agents.models.qwen3_5.action_space",
    "lite.agents.models.qwen3_8.action_space",
    "lite.agents.models.qwen3_vl.action_space",
    "lite.agents.models.step_gui.action_space",
    "lite.agents.models.ui_tars.action_space",
    "lite.agents.models.ui_tars_15_v1.action_space",
)

for module_name in _ACTION_SPACE_MODULES:
    importlib.import_module(module_name)


@pytest.mark.parametrize(
    ("key", "expected_platform"),
    [
        ("claude@desktop", "desktop"),
        ("claude@browser", "browser"),
        ("claude@desktop@point", "desktop"),
        ("claude@browser@point", "browser"),
        ("evocua@desktop", "desktop"),
        ("evocua@browser", "browser"),
        ("evocua@desktop@point", "desktop"),
        ("evocua@browser@point", "browser"),
        ("fara@desktop", "desktop"),
        ("fara@browser", "browser"),
        ("fara@desktop@point", "desktop"),
        ("fara@browser@point", "browser"),
        ("gemini@desktop", "desktop"),
        ("gemini@browser", "browser"),
        ("gemini@mobile", "mobile"),
        ("gpt@desktop", "desktop"),
        ("gpt@browser", "browser"),
        ("gpt@desktop@point", "desktop"),
        ("gpt@browser@point", "browser"),
        ("mai_ui@desktop@point", "desktop"),
        ("mai_ui@browser@point", "browser"),
        ("mai_ui@mobile@point", "mobile"),
        ("qwen2_5_vl@desktop", "desktop"),
        ("qwen2_5_vl@browser", "browser"),
        ("qwen2_5_vl@desktop@point", "desktop"),
        ("qwen2_5_vl@browser@point", "browser"),
        ("qwen3_5@desktop", "desktop"),
        ("qwen3_5@browser", "browser"),
        ("qwen3_5@desktop@point", "desktop"),
        ("qwen3_5@browser@point", "browser"),
        ("qwen3_8@desktop", "desktop"),
        ("qwen3_8@browser", "browser"),
        ("qwen3_8@desktop@point", "desktop"),
        ("qwen3_8@browser@point", "browser"),
        ("qwen3_vl@desktop", "desktop"),
        ("qwen3_vl@browser", "browser"),
        ("qwen3_vl@desktop@point", "desktop"),
        ("qwen3_vl@browser@point", "browser"),
        ("step_gui@mobile", "mobile"),
        ("ui_tars@desktop", "desktop"),
        ("ui_tars@browser", "browser"),
        ("ui_tars@desktop@point", "desktop"),
        ("ui_tars@browser@point", "browser"),
        ("ui_tars_15_v1@desktop", "desktop"),
        ("ui_tars_15_v1@browser", "browser"),
        ("ui_tars_15_v1@desktop@point", "desktop"),
        ("ui_tars_15_v1@browser@point", "browser"),
    ],
)
def test_registry_instance_platform_matches_concrete_key(
    key: str,
    expected_platform: str,
) -> None:
    assert ActionSpaceRegistry.get(key).platform == expected_platform


def test_shared_desktop_browser_registration_keeps_key_specific_platform_metadata() -> None:
    desktop = ActionSpaceRegistry.get("qwen3_vl@desktop")
    browser = ActionSpaceRegistry.get("qwen3_vl@browser")

    assert type(desktop) is type(browser)
    assert desktop.platform == "desktop"
    assert browser.platform == "browser"


# -----------------------------------------------------------------------------
# Every action space DECLARES a platform. ``platform`` feeds
# ``LiteCUAMetadata.Platform(...)`` in ``adapter/base.py::_default_metadata_for_
# adapter``; a ``None`` there raised and fell into the ValueError rescue, which
# silently relabelled the space DESKTOP. The rescue stays (it is a real
# boundary), but no registered space may DEPEND on it.
# -----------------------------------------------------------------------------

def test_every_registered_action_space_declares_a_valid_platform() -> None:
    offenders = []
    for key in sorted(ActionSpaceRegistry.list()):
        platform = ActionSpaceRegistry.get(key).platform
        try:
            LiteCUAMetadata.Platform(platform)
        except ValueError:
            offenders.append((key, platform))
    assert not offenders, (
        "action spaces whose platform is not a LiteCUAMetadata.Platform — they "
        f"reach DESKTOP only through the adapter rescue path: {offenders}"
    )


@pytest.mark.parametrize(
    "space_cls", [LitePointActionSpace, LiteBBoxActionSpace]
)
def test_core_grounding_spaces_declare_platform_as_str(space_cls) -> None:
    """The two keys with no ``@<platform>`` segment (``lite@point`` /
    ``lite@bbox``) are the ones the registry never overwrites from the key, so
    the class default IS the declared value — it must be a real platform, not
    ``None``."""
    assert space_cls().platform == "desktop"
    assert LiteCUAMetadata.Platform(space_cls().platform) is LiteCUAMetadata.Platform.DESKTOP


@pytest.mark.parametrize("key", ["lite@point", "lite@bbox"])
def test_platformless_grounding_keys_resolve_to_desktop(key: str) -> None:
    assert ActionSpaceRegistry.get(key).platform == "desktop"


@pytest.mark.parametrize(
    "key",
    [
        "gpt@desktop@grounding.point",
        "gpt@browser@grounding.point",
        "gpt@desktop@grounding.bbox",
        "gpt@browser@grounding.bbox",
        "claude@desktop@grounding.point",
        "claude@browser@grounding.point",
        "claude@desktop@grounding.bbox",
        "claude@browser@grounding.bbox",
        "gpt@web@grounding.point",
        "gpt@web@grounding.bbox",
        "claude@web@grounding.point",
        "claude@web@grounding.bbox",
        "claude@web",
        "claude@web@point",
        "evocua@web",
        "evocua@web@point",
        "fara@web",
        "fara@web@point",
        "gemini@web",
        "gpt@web",
        "gpt@web@point",
        "mai_ui@web@point",
        "qwen2_5_vl@web",
        "qwen2_5_vl@web@point",
        "qwen3_5@web",
        "qwen3_5@web@point",
        "qwen3_8@web",
        "qwen3_8@web@point",
        "qwen3_vl@web",
        "qwen3_vl@web@point",
        "step_gui@web",
        "ui_tars@web",
        "ui_tars@web@point",
        "ui_tars_15_v1@web",
        "ui_tars_15_v1@web@point",
    ],
)
def test_action_spaces_do_not_alias_task_suffixes_or_stale_web_keys(
    key: str,
) -> None:
    with pytest.raises(KeyError):
        ActionSpaceRegistry.get(key)
