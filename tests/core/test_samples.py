"""Core LiteSample metadata-container boundary.

Run:
    uv run pytest tests/core/test_samples.py -q
"""

from __future__ import annotations

import json

import pytest

from lite.core import samples
from lite.core.errors import LiteContractError
from lite.core.metadata import LiteCUAMetadata, LiteGenericMetadata, metadata_from_dict
from lite.core.samples import (
    STATUS_ABORTED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_TRUNCATED,
    STEP_STATUSES_BY_SEVERITY,
    LiteSample,
)
from lite.core.tools.schemas import make_tool_schema

_ROLLOUT_FACTS = {
    "env_id": "webgym",
    "task_id": "task_001",
    "episode_return": 1.0,
    "terminated": True,
    "truncated": False,
}


def _metadata_dict() -> dict:
    return LiteCUAMetadata(others=dict(_ROLLOUT_FACTS)).to_dict()


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(_metadata_dict(), id="dict"),
        pytest.param(metadata_from_dict(_metadata_dict()), id="LiteCUAMetadata"),
    ],
)
def test_lite_sample_from_dict_accepts_canonical_row_metadata(metadata) -> None:
    sample = LiteSample.from_dict({"metadata": metadata, "images": [], "messages": []})

    assert isinstance(sample.metadata, LiteCUAMetadata)
    assert sample.metadata.others == _ROLLOUT_FACTS


def test_lite_sample_from_dict_accepts_tagged_generic_metadata_without_valid_actions() -> None:
    schema = make_tool_schema("answer", parameters={})
    metadata = {
        "metadata_kind": "generic",
        "dims": [],
        "extra_tool_schemas": [schema],
        "others": {"dataset": "geo3k"},
    }

    sample = LiteSample.from_dict({"metadata": metadata, "images": [], "messages": []})

    assert isinstance(sample.metadata, LiteGenericMetadata)
    assert sample.metadata.dims == ()
    assert sample.metadata.extra_tool_schemas == [schema]
    assert sample.metadata.others == {"dataset": "geo3k"}
    assert not hasattr(sample.metadata, "valid_actions")
    assert sample.to_dict()["metadata"] == metadata
    assert "valid_actions" not in sample.to_dict()["metadata"]


def test_lite_sample_to_dict_publishes_canonical_metadata_dict() -> None:
    metadata = LiteCUAMetadata.from_dict(_metadata_dict())

    row = LiteSample(metadata=metadata, images=[], messages=[]).to_dict()

    assert row["metadata"] == _metadata_dict()
    for key, value in _ROLLOUT_FACTS.items():
        assert key not in row["metadata"]
        assert row["metadata"]["others"][key] == value


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(None, id="none"),
        pytest.param([], id="list"),
        pytest.param(object(), id="object"),
        pytest.param(json.dumps(_metadata_dict()), id="json"),
        pytest.param("{not json", id="malformed-json"),
    ],
)
def test_lite_sample_from_dict_rejects_unsupported_metadata_containers(metadata) -> None:
    """Row metadata is a dict. A serialized JSON string is decoded by the data
    ingress owner (``lite.data.staging.coerce_meta``), so this boundary rejects
    strings through the named core error instead of parsing them itself."""
    with pytest.raises(LiteContractError, match="metadata must be a dict"):
        LiteSample.from_dict({"metadata": metadata, "images": [], "messages": []})


@pytest.mark.parametrize(
    ("index", "match"),
    [
        pytest.param("0", "non-negative integer", id="string"),
        pytest.param(True, "non-negative integer", id="bool"),
        pytest.param(-1, "non-negative", id="negative"),
        pytest.param(1, "out of range", id="out-of-range"),
    ],
)
def test_lite_sample_from_dict_validates_image_reference_binding(
    index,
    match: str,
) -> None:
    row = {
        "metadata": _metadata_dict(),
        "images": ["img0.png"],
        "messages": [
            {"role": "user", "content": [{"type": "image", "index": index}]},
        ],
    }

    with pytest.raises(LiteContractError, match=match):
        LiteSample.from_dict(row)


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(_metadata_dict(), id="dict"),
        pytest.param(json.dumps(_metadata_dict()), id="json"),
        pytest.param(None, id="none"),
    ],
)
def test_lite_sample_to_dict_does_not_read_raw_metadata(metadata) -> None:
    """``to_dict()`` serializes a ``LiteCUAMetadata`` and nothing else. It is not a
    second normalization point, so a sample built with raw metadata fails here
    rather than silently gaining a second accepted metadata shape."""
    with pytest.raises(AttributeError):
        LiteSample(metadata=metadata, images=[], messages=[]).to_dict()


def test_step_status_vocabulary_is_declared_once_and_ordered_by_severity() -> None:
    """``STEP_STATUSES_BY_SEVERITY`` is the single status declaration.

    Its ORDER is load-bearing: ``lite.train.rollout.core.segmenter`` turns it
    into severity ranks and takes the worst status over a packed segment, so a
    reordering here would silently change the emitted training status.
    """
    assert STEP_STATUSES_BY_SEVERITY == (
        STATUS_COMPLETED,
        STATUS_TRUNCATED,
        STATUS_ABORTED,
        STATUS_FAILED,
    )
    assert "STEP_STATUSES_BY_SEVERITY" in samples.__all__
