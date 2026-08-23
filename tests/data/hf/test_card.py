"""Tests for the HF dataset-card config rendering (lite/data/hf/card.py).

Focus: the `configs:` YAML must declare ONLY splits that actually have files.
A config whose declared split resolves to zero files makes the HF dataset-viewer
fail with "the splits use different data file formats" — which is what bit a
small rollout-staged dataset that hash-splits to an empty `validation`.

Run:
    uv run pytest tests/data/hf/test_card.py -v
"""
from __future__ import annotations

from lite.data.hf.card import configs_yaml, render_card
from lite.data.staging import DatasetStats


def _stats(by_partition: dict) -> DatasetStats:
    s = DatasetStats()
    s.by_partition = dict(by_partition)
    return s


def _repo_meta() -> dict:
    return {
        "description": "Synthetic test dataset.",
        "original_urls": ["https://example.com/source"],
        "license": "Test license.",
        "citation": "Test citation.",
    }


def test_both_splits_present_emits_both():
    # existing datasets (e.g. Multimodal-Mind2Web) have train + validation:
    # rendering must be UNCHANGED (no regression).
    y = configs_yaml(_stats({
        ("browser", "use", "train", "v"): 20,
        ("browser", "use", "validation", "v"): 1,
    }))
    assert "split: train" in y and "split: validation" in y


def test_train_only_omits_validation():
    # small / rollout-staged dataset with no validation rows: the `validation`
    # split must NOT be declared (else the viewer rejects the config).
    y = configs_yaml(_stats({("browser", "use", "train", "v"): 33}))
    assert "split: train" in y
    assert "validation" not in y


def test_per_cohort_splits_are_independent():
    # mobile cohort has train only; browser cohort has both. Each config declares
    # only its own present splits; the global `default` still spans both.
    y = configs_yaml(_stats({
        ("browser", "use", "train", "v"): 5,
        ("browser", "use", "validation", "v"): 1,
        ("mobile", "grounding.action", "train", "v"): 7,
    }))
    # the mobile cohort config block must not carry a validation split
    mobile_block = y.split("config_name: mobile.grounding.action")[1].split("config_name:")[0]
    assert "split: train" in mobile_block and "validation" not in mobile_block
    # the browser cohort config block keeps both
    browser_block = y.split("config_name: browser.use")[1]
    assert "split: train" in browser_block and "split: validation" in browser_block


# --- --config-names override (config_name_override=True) -------------------

def test_config_name_override_emits_verbatim_labels():
    # `stage --config-names synth perturb` → configs named VERBATIM after the
    # variant labels, NOT the derived desktop.use cohort.
    y = configs_yaml(_stats({
        ("desktop", "use", "train", "synth"): 1485,
        ("desktop", "use", "train", "perturb"): 700,
    }), config_name_override=True)
    names = [ln.split("config_name: ")[1] for ln in y.splitlines() if "config_name:" in ln]
    assert names == ["default", "perturb", "synth"]  # sorted; no desktop.use
    assert "desktop.use" not in y
    # each label globs only its own <split>/<label>.parquet across cohorts
    synth_block = y.split("config_name: synth")[1]
    assert "*/*/train/synth.parquet" in synth_block


def test_config_name_override_single_label_kept_verbatim():
    # single `--config-name synth` → config `synth` (verbatim), NOT collapsed away.
    y = configs_yaml(_stats({("desktop", "use", "train", "synth"): 10}),
                     config_name_override=True)
    names = [ln.split("config_name: ")[1] for ln in y.splitlines() if "config_name:" in ln]
    assert names == ["default", "synth"]
    assert "desktop.use" not in y


def test_config_name_override_card_example_loads_existing_config():
    # The rendered card's load_dataset example must reference a config that
    # actually exists in override mode (not the derived desktop.use).
    from lite.data.hf.card import _config_examples
    s = _stats({
        ("desktop", "use", "train", "synth"): 10,
        ("desktop", "use", "train", "perturb"): 5,
    })
    ex = _config_examples("Lite.OSWorld", s, config_name_override=True)
    assert '"perturb"' in ex or '"synth"' in ex   # a real override config
    assert "desktop.use" not in ex          # not the (absent) cohort config
    # no-flag example still uses the cohort config
    assert "desktop.use" in _config_examples("Lite.OSWorld", s)


def test_no_override_is_unchanged_cohort_naming():
    # WITHOUT the flag, even multi-variant data merges into the derived cohort
    # config (today's existing behavior, preproc-safe).
    part = {
        ("desktop", "use", "train", "synth"): 1485,
        ("desktop", "use", "train", "perturb"): 700,
    }
    y = configs_yaml(_stats(part))                      # default: no override
    names = [ln.split("config_name: ")[1] for ln in y.splitlines() if "config_name:" in ln]
    assert names == ["default", "desktop.use"]
    assert "synth" not in y and "perturb" not in y


def test_upload_rendered_card_publishes_json_columns_and_nested_tool_examples() -> None:
    card = render_card(
        name="TestSet",
        repo=_repo_meta(),
        stats=_stats({("desktop", "use", "train", "synth"): 1}),
    )

    assert "| `messages` | string (JSON array) |" in card
    assert "| `metadata` | string (JSON object) |" in card
    assert "| `_folded` | string (JSON array, optional) |" in card
    assert "| `messages` | list[struct] |" not in card
    assert "| `metadata` | struct |" not in card

    assert '"extra_tool_schemas": [' in card
    assert '"type": "function"' in card
    assert '"function": {' in card
    assert '"parameters": {' in card
    assert '"tool_calls": [' in card
    assert '"id": "call_0000"' in card
    assert '"arguments": {' in card
    assert '"tool_call_id": "call_0000"' in card
