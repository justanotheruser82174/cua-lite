"""ScaleCUA XCF helper tests."""

from __future__ import annotations

import struct
from types import SimpleNamespace

import pytest

from lite.gym.envs.lite.scalecua.src.osworld import judges, xcf
from lite.gym.envs.lite.scalecua.src.utils import dataset


def _xcf_prop(prop_type: int, payload: bytes = b"") -> bytes:
    return struct.pack(">II", prop_type, len(payload)) + payload


def _minimal_xcf(layers: list[tuple[str, bool]]) -> bytes:
    header = b"gimp xcf v011\x00" + struct.pack(">III", 64, 64, 0)
    image_props = _xcf_prop(xcf.PROP_END)
    records: list[bytes] = []
    for name, is_group in layers:
        name_bytes = name.encode("utf-8") + b"\x00"
        layer_props = _xcf_prop(
            xcf.PROP_GROUP_ITEM,
            struct.pack(">I", 1 if is_group else 0),
        ) + _xcf_prop(xcf.PROP_END)
        records.append(
            struct.pack(">IIII", 64, 64, 0, len(name_bytes)) + name_bytes + layer_props
        )
    offsets_start = len(header) + len(image_props)
    records_start = offsets_start + (len(records) + 1) * 8
    offsets = []
    cursor = records_start
    for record in records:
        offsets.append(cursor)
        cursor += len(record)
    offset_table = b"".join(struct.pack(">Q", offset) for offset in offsets)
    offset_table += struct.pack(">Q", 0)
    return header + image_props + offset_table + b"".join(records)


def test_scalecua_xcf_parser_extracts_layer_names_and_group_flags(tmp_path):
    path = tmp_path / "layers.xcf"
    path.write_bytes(_minimal_xcf([("背景", False), ("Shapes", True)]))

    assert xcf.parse_xcf_layers(path) == ["背景", "Shapes"]
    assert xcf.parse_xcf_layer_groups(path) == [
        {"name": "背景", "is_group": False},
        {"name": "Shapes", "is_group": True},
    ]

    env = SimpleNamespace(
        controller=SimpleNamespace(get_file=lambda vm_path: path.read_bytes())
    )
    assert xcf.parse_xcf_file_directly(env, "/home/user/Desktop/layers.xcf") == [
        "背景",
        "Shapes",
    ]


def _cache_ready() -> bool:
    return all(dataset.catalog_path(split).is_file() for split in dataset.RUNTIME_SPLITS)


@pytest.mark.skipif(not _cache_ready(), reason="lite.scalecua cache not imported")
def test_scalecua_gimp_xcf_helpers_are_injected_for_generated_getters():
    pytest.importorskip("desktop_env", reason="desktop_env not installed")
    judges._load_overlay_module.cache_clear()

    mod = judges._load_overlay_module("train", "getters")

    getter = getattr(mod, "get_gimp_layer_names__8565a91c")
    assert getter.__globals__["parse_xcf_layers"] is xcf.parse_xcf_layers
    fallback = getattr(mod, "get_xcf_layer_names__b148e375")
    assert fallback.__globals__["parse_xcf_file_directly"] is xcf.parse_xcf_file_directly

    from desktop_env.evaluators.getters import gimp as upstream_gimp_getters

    assert upstream_gimp_getters.parse_xcf_layers is xcf.parse_xcf_layers
