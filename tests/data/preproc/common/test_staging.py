from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from lite.core import LiteCUAMetadata
from lite.core.messages.image_refs import validate_image_references
from lite.core.tools import make_tool_call
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.data.preproc.aguvis import utils as aguvis_utils
from lite.data.preproc.cagui import utils as cagui_utils
from lite.data.preproc.common import SourceStaging, prestage_images
from lite.data.preproc.gui360 import utils as gui360_utils
from lite.data.preproc.guiact import utils as guiact_utils
from lite.data.preproc.guiodyssey import utils as guiodyssey_utils
from lite.data.preproc.multimodal_mind2web import utils as mind2web_utils
from lite.data.preproc.opencua import utils as opencua_utils
from lite.data.preproc.scalecua import utils as scalecua_utils
from lite.data.preproc.ui_genie_agent import utils as ui_genie_utils
from lite.data.staging import ImageStore, coerce_messages
from lite.data.utils.rows import validate_canonical_rows


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(12, 34, 56)).save(path)


def test_prestage_images_isolates_corrupt_file(tmp_path) -> None:
    good = tmp_path / "good.png"
    Image.new("RGB", (2, 2)).save(good)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    store = ImageStore(tmp_path / "images")

    assert prestage_images(store, [good, bad, good]) == {bad}
    assert store.put(good).endswith(".png")


@pytest.mark.parametrize(
    "module,dataset_name,platform,task_type,row_id",
    [
        (scalecua_utils, "ScaleCUA", "desktop", "use", "scalecua_row_1"),
        (guiodyssey_utils, "GUIOdyssey", "mobile", "understanding", "guiodyssey_ep0001_0"),
        (gui360_utils, "GUI-360", "desktop", "grounding.point", "gui360_excel_1"),
        (mind2web_utils, "Multimodal-Mind2Web", "browser", "use", "mind2web_1"),
        (cagui_utils, "CAGUI", "mobile", "understanding", "cagui_cap_1"),
        (ui_genie_utils, "UI-Genie-Agent", "mobile", "use", "ui_genie_amex_1"),
        (guiact_utils, "GUIAct", "browser", "use", "guiact_row_1"),
        (aguvis_utils, "Aguvis", "desktop", "grounding.action", "aguvis_omniact_1"),
        (opencua_utils, "OpenCUA", "desktop", "use", "agentnet_desktop_1"),
    ],
)
def test_migrated_sources_use_common_staging_helper(
    tmp_path: Path,
    module,
    dataset_name: str,
    platform: str,
    task_type: str,
    row_id: str,
) -> None:
    image = tmp_path / "raw" / f"{row_id}.png"
    _write_png(image)
    row = {
        "images": [str(image)],
        "messages": [{"role": "user", "content": [{"type": "image", "index": 0}]}],
        "metadata": LiteCUAMetadata(
            dims=(platform, task_type),
            others={"id": row_id},
        ).to_dict(),
    }

    assert isinstance(module.out_dir_for.__self__, SourceStaging)
    assert module.out_dir_for.__self__.dataset_name == dataset_name

    out_dir = module.out_dir_for(root=tmp_path / "out")
    key, staged = module.stage_entry(
        row,
        store=module.make_image_store(out_dir),
        splitter=module.make_splitter(),
        variant="proof",
    )

    assert key[0] == platform
    assert key[1] == task_type
    assert key[2] in {"train", "validation"}
    assert key[3] == "proof"
    assert staged["metadata"]["extra_tool_schemas"] == []
    assert staged["metadata"]["valid_actions"] is None
    assert staged["images"][0].startswith(f"cua-lite/{dataset_name}/images/")
    assert (out_dir / "images" / staged["images"][0].split("/images/", 1)[1]).is_file()
    validate_canonical_rows([staged], f"{dataset_name}/common-staging")


@pytest.mark.parametrize(
    "module,dataset_name,val_rows,digest",
    [
        (
            aguvis_utils,
            "Aguvis",
            84,
            "ff135e4d3a84c1831933ea6df7a410c7d89ba1a2de6f8cf4097e29b3e81eac7c",
        ),
        (
            opencua_utils,
            "OpenCUA",
            85,
            "a621060ffc1a62f5f14470a5e33b6fec48ceb46075bf5fb484c71828b917f5b0",
        ),
    ],
)
def test_migrated_sources_keep_their_published_split(
    tmp_path: Path,
    module,
    dataset_name: str,
    val_rows: int,
    digest: str,
) -> None:
    """A drift here moves rows of an already-published dataset across train/validation."""
    splitter = module.make_splitter()
    assigned = [
        splitter.assign(
            {
                # Distinct content per probe row, because a split is now also a
                # property of the CONTENT: content-identical rows are co-located.
                # 4000 distinct rows is the case where that is a no-op, which is
                # what makes the pinned digest below evidence of non-regression.
                "images": [f"{dataset_name}_{i}.png"],
                "messages": [
                    {"role": "user", "content": [{"type": "image", "index": 0}]}
                ],
                "metadata": LiteCUAMetadata(
                    dims=("desktop", "use"),
                    others={"id": f"{dataset_name}_{i}"},
                ).to_dict(),
            },
            bucket_extra=("probe",),
        )
        for i in range(4000)
    ]
    assert assigned.count("validation") == val_rows
    assert hashlib.sha256("".join(assigned).encode()).hexdigest() == digest

    # An upstream hint still wins over the hash — and is consumed, not published.
    image = tmp_path / "raw" / "hinted.png"
    _write_png(image)
    key, staged = module.stage_entry(
        {
            "images": [str(image)],
            "messages": [{"role": "user", "content": [{"type": "image", "index": 0}]}],
            "metadata": LiteCUAMetadata(
                dims=("desktop", "use"),
                others={"id": f"{dataset_name}_0", "split": "test"},
            ).to_dict(),
        },
        store=module.make_image_store(module.out_dir_for(root=tmp_path / "out")),
        splitter=module.make_splitter(),
        variant="probe",
    )
    assert key[2] == "validation"
    assert "split" not in staged["metadata"]["others"]


def test_source_staging_preserves_surface_metadata_and_consumes_the_split_hint(
    tmp_path: Path,
) -> None:
    """Surface metadata survives staging; the upstream split hint does not.

    ``others.split`` routes the row (``"dev"`` → ``validation``) and is then
    popped: it is a transient hint, and a persisted copy is what
    ``lite/data/utils/rows.py`` rejects downstream.
    """
    image = tmp_path / "raw" / "row_1.png"
    _write_png(image)
    response_schema = LiteFinishToolSet.get_tool_schema("response")
    row = {
        "images": [str(image)],
        "messages": [{"role": "user", "content": [{"type": "image", "index": 0}]}],
        "metadata": LiteCUAMetadata(
            dims=("browser", "use"),
            extra_tool_schemas=[response_schema],
            valid_actions=["click", "scroll"],
            others={
                "id": "row_1",
                "split": "dev",
                "source": "unit",
            },
        ).to_dict(),
    }
    helper = SourceStaging("UnitSource")
    out_dir = helper.out_dir_for(root=tmp_path / "out")

    key, staged = helper.stage_entry(
        row,
        store=helper.make_image_store(out_dir),
        splitter=helper.make_splitter(),
        variant="preserve",
    )

    assert key == ("browser", "use", "validation", "preserve")
    assert staged["metadata"]["extra_tool_schemas"] == [response_schema]
    assert staged["metadata"]["valid_actions"] == ["click", "scroll"]
    assert staged["metadata"]["others"] == {"id": "row_1", "source": "unit"}
    assert staged["images"][0].startswith("cua-lite/UnitSource/images/")
    assert (out_dir / "images" / staged["images"][0].split("/images/", 1)[1]).is_file()
    validate_canonical_rows([staged], "UnitSource/common-staging-preserve")


def test_stage_entry_rejects_noncanonical_rows(tmp_path: Path) -> None:
    """Fresh preproc producers share the same publication gate as staged rows."""
    image = tmp_path / "raw" / "row_bad.png"
    _write_png(image)
    row = {
        "images": [str(image)],
        "messages": [
            {"role": "user", "content": [{"type": "image", "index": 0}]},
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": "click"}],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                    )
                ],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            others={"id": "row_bad"},
        ).to_dict(),
    }
    helper = SourceStaging("UnitSource")
    out_dir = helper.out_dir_for(root=tmp_path / "out")

    with pytest.raises(ValueError, match="missing non-empty id"):
        helper.stage_entry(
            row,
            store=helper.make_image_store(out_dir),
            splitter=helper.make_splitter(),
            variant="reject",
        )


def test_json_string_messages_coerce_storage_image_index_like_struct_messages() -> None:
    """Both read shapes must land integral float image refs as ``int``.

    The JSON-string transport shape stays opaque for provider evidence
    (explicit ``null``s, empty ``tool_calls``), but an image index is a
    storage-typed scalar, so it is normalized on both read paths.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0.0},
                {"type": "text", "text": "start"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Done.", "payload": {"explicit_null": None}}],
            "tool_calls": [],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0000",
            "content": [{"type": "image", "index": 1.0}],
        },
    ]

    out = coerce_messages(json.dumps(messages))

    assert out[0]["content"][0] == {"type": "image", "index": 0}
    assert out[2]["content"][0] == {"type": "image", "index": 1}
    assert type(out[0]["content"][0]["index"]) is int
    assert type(out[2]["content"][0]["index"]) is int
    validate_image_references(out, ["img0.png", "img1.png"])

    # Opaque JSON transport keeps provider-visible evidence that the struct
    # read path treats as Arrow padding.
    assert out[1]["tool_calls"] == []
    assert out[1]["content"][0]["payload"] == {"explicit_null": None}


def test_staging_records_the_split_policy_beside_the_output(tmp_path: Path) -> None:
    """The split is a path artifact, so nothing on disk says how it was drawn.

    ``split.json`` is that record, and ``lite.data.hf.upload`` folds it into the
    card's notes — the only place a consumer of the published dataset can read it.
    """
    staging = SourceStaging(dataset_name="Probe", val_cap=11)
    out_dir = staging.out_dir_for(root=tmp_path / "out")
    store = staging.make_image_store(out_dir)
    splitter = staging.make_splitter()
    image = tmp_path / "raw" / "row_0.png"
    _write_png(image)
    row = {
        "images": [str(image)],
        "messages": [{"role": "user", "content": [{"type": "image", "index": 0}]}],
        "metadata": LiteCUAMetadata(
            dims=("browser", "understanding"),
            others={"id": "probe_0"},
        ).to_dict(),
    }

    staging.stage_entry(row, store=store, splitter=splitter, variant="probe")

    policy = json.loads((out_dir / "split.json").read_text())["split_policy"]
    assert policy == splitter.describe()
    assert "val_cap=11" in policy and "metadata.others.id" in policy
