"""Tests for ``lite.data.hf.download``.

Coverage includes --allow-patterns handling, snapshot walk bounds, and the
canonical-row output contract. Offline: the guard's ``HfApi`` is monkeypatched
and every download runs against a local ``snapshot_dir``, so nothing hits the
network.

Run: uv run pytest tests/data/hf/test_download.py
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from lite.core import LiteCUAMetadata
from lite.core.errors import LiteContractError
from lite.core.tools import make_tool_call
from lite.data.hf import download as dl
from lite.data.staging import ImageStore
from lite.data.utils.rows import validate_canonical_rows


def _download_row() -> dict:
    return {
        "images": [],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "finish"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }


def _patch_snapshot_rows(monkeypatch, snapshot, rows):
    parquet = snapshot / "desktop" / "use" / "train.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.write_text("not read")
    monkeypatch.setattr(dl, "_read_rows", lambda path: rows)


class TestExpandAlternations:
    def test_none_passthrough(self):
        assert dl._expand_alternations(None) is None

    def test_plain_glob_unchanged(self):
        # No ``(...)`` alternation: preserve existing plain-glob behavior.
        assert dl._expand_alternations("*/grounding.action/**") == ["*/grounding.action/**"]
        assert dl._expand_alternations("mobile/grounding.point/**") == ["mobile/grounding.point/**"]

    def test_single_alternation_cartesian(self):
        assert dl._expand_alternations("(desktop|browser|mobile)/grounding.action/**") == [
            "desktop/grounding.action/**",
            "browser/grounding.action/**",
            "mobile/grounding.action/**",
        ]

    def test_list_input_mixed(self):
        assert dl._expand_alternations(["a/**", "(x|y)/b/**"]) == ["a/**", "x/b/**", "y/b/**"]

    def test_multiple_groups_product(self):
        assert dl._expand_alternations("(a|b)/(x|y)") == ["a/x", "a/y", "b/x", "b/y"]

    def test_plain_parentheses_not_stripped(self):
        # parens without a ``|`` are literal glob chars, not an alternation —
        # must pass through verbatim (regressed by the naive ``\(([^()]*)\)``).
        assert dl._expand_alternations("some_file_(1).parquet") == ["some_file_(1).parquet"]
        # mixed: real alternation expands, literal parens survive
        assert dl._expand_alternations("(a|b)/file_(1).parquet") == [
            "a/file_(1).parquet",
            "b/file_(1).parquet",
        ]


class TestAssertPatternsMatch:
    @staticmethod
    def _patch_files(monkeypatch, files):
        class _FakeApi:
            def list_repo_files(self, *a, **k):
                return files
        monkeypatch.setattr(dl, "HfApi", lambda *a, **k: _FakeApi())

    def test_match_ok(self, monkeypatch):
        self._patch_files(monkeypatch, ["desktop/grounding.action/train/s.parquet", "README.md"])
        dl._assert_patterns_match("cua-lite/X", ["desktop/grounding.action/**"], revision=None)

    def test_trailing_slash_directory_pattern_does_not_false_raise(self, monkeypatch):
        # snapshot_download applies hf_hub's _add_wildcard_to_directories, so a
        # trailing-slash `dir/` pattern becomes `dir/*` and pulls files. The guard
        # uses filter_repo_objects (hf_hub's own matcher) to mirror that exactly —
        # a hand-rolled fnmatch would compute 0 matches here and FALSE-raise on a
        # directory form the pull actually honors.
        self._patch_files(monkeypatch, ["mobile/grounding.point/train/s.parquet", "README.md"])
        dl._assert_patterns_match("cua-lite/X", ["mobile/grounding.point/"], revision=None)

    def test_literal_alternation_matches_zero_raises(self, monkeypatch):
        # the exact #64 footgun: a raw (unexpanded) regex-alternation is a
        # literal glob and matches nothing -> must raise, not silently pull 0.
        self._patch_files(monkeypatch, ["desktop/grounding.action/train/s.parquet"])
        with pytest.raises(ValueError, match="matched 0 of"):
            dl._assert_patterns_match(
                "cua-lite/X", ["(desktop|browser|mobile)/grounding.action/**"], revision=None
            )

    def test_expanded_alternation_matches(self, monkeypatch):
        # once expanded via _expand_alternations, the same intent matches.
        self._patch_files(monkeypatch, ["browser/grounding.action/train/s.parquet"])
        pats = dl._expand_alternations("(desktop|browser|mobile)/grounding.action/**")
        dl._assert_patterns_match("cua-lite/X", pats, revision=None)


def test_rewrite_rows_preserves_role_tool_and_batched_calls(tmp_path):
    messages = [
        {"role": "assistant", "content": [], "tool_calls": [make_tool_call(
            "computer",
            {"actions": [
                {"action": "click", "coordinate": [1, 2]},
                {"action": "type", "text": "hello"},
            ]},
            call_id="call_0000",
        )]},
        {"role": "tool", "tool_call_id": "call_0000", "content": [{"type": "text", "text": "ok"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    ]
    rows = [{
        "images": [{"bytes": b"\x89PNG\r\n\x1a\n", "path": None}],
        "messages": messages,
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]
    store = ImageStore(tmp_path / "images", rel_prefix="cua-lite/X/images")

    rewritten, n_new = dl._rewrite_rows(rows, store)

    assert n_new == 1
    assert rewritten[0]["messages"] == messages
    assert rewritten[0]["images"][0].startswith("cua-lite/X/images/")


def test_rewrite_rows_does_not_repair_hf_padded_message_shape(tmp_path):
    rows = [{
        "images": [{"bytes": b"\x89PNG\r\n\x1a\n", "path": None}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0.0, "text": None},
                    {"type": "text", "text": "finish", "index": None},
                ],
                "tool_calls": None,
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Done.", "index": None}],
                "tool_calls": [],
            },
        ],
        "metadata": json.dumps(LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"env_id": "lite.osworld"},
        ).to_dict()),
    }]
    store = ImageStore(tmp_path / "images", rel_prefix="cua-lite/X/images")

    rewritten, n_new = dl._rewrite_rows(rows, store)

    assert n_new == 1
    assert rewritten[0]["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0, "text": None},
                {"type": "text", "text": "finish", "index": None},
            ],
            "tool_calls": None,
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Done.", "index": None}],
            "tool_calls": [],
        },
    ]
    assert rewritten[0]["metadata"]["others"] == {"env_id": "lite.osworld"}


def test_download_refuses_stale_output_without_overwrite(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    stale = out / "old.parquet"
    stale.write_text("old")

    with pytest.raises(FileExistsError, match="--overwrite"):
        dl.download_dataset("X", out_dir=out, snapshot_dir=snapshot)

    dl.download_dataset("X", out_dir=out, snapshot_dir=snapshot, overwrite=True)
    assert not stale.exists()


# ---------------------------------------------------------------------------
# Output contract: layout-canonical, NOT content-canonical.
#
# ``download`` is the only entry point whose input is by definition possibly
# unmigrated -- it reads historical data already published on HF. A content gate
# here makes migration's own step 1 (pull the old rows) unrunnable on exactly the
# rows migration exists to repair. Content is gated by canonical producers such
# as ``hf/stage.py`` before upload transports rows -- so these rows must SURVIVE
# download and still be REJECTED by that shared producer-side gate.
# ---------------------------------------------------------------------------


def _pre_migration_row() -> dict:
    """A row in the pre-migration unstamped Lite-call shape.

    ``tool_calls[0]`` uses the canonical outer ``type/function`` envelope, but
    carries no persisted ``id`` and stores ``function.arguments`` as a JSON
    string instead of a dict. This is the exact shape ``devs/migration`` exists
    to rewrite, and the exact shape whose rejection at download time made
    ``Lite.OSWorld`` unfetchable.
    """
    return {
        "images": [],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "clicking"}],
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": "computer",
                        "arguments": '{"actions": [{"action": "click", "coordinate": [1, 2]}]}',
                    },
                }],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"task_id": "t1"},
        ).to_dict(),
    }


@pytest.mark.parametrize(
    ("mutate", "error_type", "message"),
    [
        pytest.param(
            lambda row: row["messages"][-1].update({
                "raw_response": {
                    "adapter_key": "qwen3_vl@desktop@use",
                    "text": "Action: Done.",
                },
            }),
            ValueError,
            "raw_response.*must not be published",
            id="raw-response-sidecar",
        ),
        pytest.param(
            lambda row: row["metadata"].update({"split": "train"}),
            ValueError,
            "metadata.split must not be present",
            id="metadata-split",
        ),
        pytest.param(
            lambda row: row["messages"][0].update({
                "content": [{"type": "image", "index": 0}],
            }),
            LiteContractError,
            r"content\[0\]\.index.*out of range",
            id="invalid-image-index",
        ),
    ],
)
def test_download_does_not_gate_row_content(
    tmp_path,
    monkeypatch,
    mutate,
    error_type,
    message,
):
    # download reshapes layout only; each of these rows reaches disk unrejected...
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    row = _download_row()
    mutate(row)
    _patch_snapshot_rows(monkeypatch, snapshot, [row])

    out = dl.download_dataset("X", out_dir=tmp_path / "out", snapshot_dir=snapshot)
    written = list(out.rglob("*.parquet"))
    assert written, "download produced no partition"
    persisted = pd.read_parquet(written[0]).iloc[0].to_dict()
    persisted["images"] = list(persisted["images"])
    persisted["messages"] = json.loads(persisted["messages"])
    persisted["metadata"] = json.loads(persisted["metadata"])

    # ...and the shared content gate used by stage/preproc still rejects it, so
    # keeping download transport-only opened no publication hole.
    with pytest.raises(error_type, match=message):
        validate_canonical_rows([persisted], "contract-check")


def test_download_accepts_pre_migration_unstamped_tool_call(tmp_path, monkeypatch):
    """NEGATIVE CONTROL: the row that blocked ``Lite.OSWorld`` must now download.

    Download is a transport/layout step, so it must not reject historical rows
    that still need migration before they are publishable.
    """
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    row = _pre_migration_row()
    assert sorted(row["messages"][1]["tool_calls"][0]) == ["function", "type"]
    _patch_snapshot_rows(monkeypatch, snapshot, [row])

    out = dl.download_dataset("X", out_dir=tmp_path / "out", snapshot_dir=snapshot)

    written = list(out.rglob("*.parquet"))
    assert written, "download produced no partition"
    landed = json.loads(pd.read_parquet(written[0]).iloc[0]["messages"])
    # The tool call is carried through UNREPAIRED -- download reshapes storage,
    # devs/migration repairs content before producer-side gates publish it.
    assert sorted(landed[1]["tool_calls"][0]) == ["function", "type"]


# ---------------------------------------------------------------------------
# R26: --allow-patterns must bound the WALK, not just the fetch.
# ---------------------------------------------------------------------------


def test_allow_patterns_bounds_the_walk_not_just_the_fetch(tmp_path, monkeypatch):
    """A warm HF cache holds shards from earlier, differently-patterned pulls.

    ``snapshot_download`` returns the whole shared cache dir, so an unfiltered
    walk validates and imports cohorts the caller did not request.
    """
    snapshot = tmp_path / "snapshot"
    wanted = snapshot / "desktop" / "use" / "train" / "desktop.use.synth.parquet"
    stale = snapshot / "desktop" / "use" / "train" / "desktop.use.perturb.parquet"
    for p in (wanted, stale):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("not read")

    seen: list[str] = []

    def _fake_read(path):
        seen.append(path.name)
        return [_download_row()]

    monkeypatch.setattr(dl, "_read_rows", _fake_read)

    dl.download_dataset(
        "X",
        out_dir=tmp_path / "out",
        snapshot_dir=snapshot,
        allow_patterns=["desktop/use/train/desktop.use.synth.parquet"],
    )

    assert seen == ["desktop.use.synth.parquet"], (
        f"walk was not bounded by --allow-patterns; read {seen}"
    )


def test_allow_patterns_zero_match_fails_for_local_snapshot(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    parquet = snapshot / "desktop" / "use" / "train.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.write_text("not read")
    monkeypatch.setattr(dl, "_read_rows", lambda path: [_download_row()])

    with pytest.raises(ValueError, match="matched 0"):
        dl.download_dataset(
            "X",
            out_dir=tmp_path / "out",
            snapshot_dir=snapshot,
            allow_patterns=["browser/use/train.parquet"],
        )
