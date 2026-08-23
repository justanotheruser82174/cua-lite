"""Eval campaign path-keying guards.

Eval runbooks reuse an existing campaign directory only when the
pipeline-relevant file set has not changed since that campaign's commit. Keep
direct/server protocol files and live config siblings in that set so
action-surface and remote-wire changes cannot reuse stale result directories.

Run:
    uv run pytest devs/exps/eval/tests/test_pipeline_keying.py -q
"""

from __future__ import annotations

import re
import shlex
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCREENSPOT_RUN_SH = ROOT / "devs/exps/eval/screenspot_pro/run.sh"
LITE_OSWORLD_RUN_SH = ROOT / "devs/exps/eval/lite.osworld/run.sh"
CAMPAIGN_DIR_HELPER = ROOT / "devs/exps/eval/utils/campaign_dir.sh"


def _pipeline_paths(run_sh: Path) -> set[str]:
    text = run_sh.read_text(encoding="utf-8")
    match = re.search(r"PIPELINE_PATHS=\(\n(?P<body>.*?)\n\t?\)", text, re.S)
    assert match is not None, f"{run_sh} must declare PIPELINE_PATHS"
    return set(shlex.split(match.group("body"), comments=True))


def _assert_sources_and_invokes_campaign_dir_helper(run_sh: Path) -> None:
    lines = [
        line.strip()
        for line in run_sh.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    source = 'source "$ROOT/devs/exps/eval/utils/campaign_dir.sh"'
    invoke = "resolve_eval_commit_dir"

    assert source in lines, f"{run_sh} must source campaign_dir.sh"
    source_pos = lines.index(source)
    assert invoke in lines[source_pos + 1 :], f"{run_sh} must call resolve_eval_commit_dir"


def test_screenspot_pipeline_paths_cover_direct_server_protocol_files() -> None:
    paths = _pipeline_paths(SCREENSPOT_RUN_SH)

    required = {
        "devs/exps/eval/screenspot_pro/run.sh",
        "devs/exps/eval/utils/runtime_mode.sh",
        "lite/core",
        "lite/agents",
        "lite/gym/envs/screenspot_pro",
        "lite/gym/factory.py",
        "lite/gym/remote",
        "lite/gym/services.py",
        "lite/gym/registry.py",
        "lite/gym/types.py",
        "lite/gym/utils",
        "scripts/serve_env.py",
        "scripts/rollout.py",
        "scripts/configs/*/default/screenspot_pro.yaml",
    }

    assert required <= paths


def test_screenspot_runbook_uses_pipeline_paths_for_dirty_and_reuse_checks() -> None:
    text = SCREENSPOT_RUN_SH.read_text(encoding="utf-8")
    helper_text = CAMPAIGN_DIR_HELPER.read_text(encoding="utf-8")

    assert 'git status --porcelain -- "${PIPELINE_PATHS[@]}"' in text
    _assert_sources_and_invokes_campaign_dir_helper(SCREENSPOT_RUN_SH)
    assert 'git log --format=%h "${sha}..HEAD" -- "${PIPELINE_PATHS[@]}"' in helper_text


def test_lite_osworld_pipeline_paths_cover_live_bash_config_siblings() -> None:
    paths = _pipeline_paths(LITE_OSWORLD_RUN_SH)

    config_glob = "scripts/configs/*/default/lite.osworld*.yaml"
    assert config_glob in paths

    bash_configs = sorted(ROOT.glob("scripts/configs/*/default/lite.osworld.bash.yaml"))
    assert bash_configs, "no live lite.osworld bash config siblings found"
    for path in bash_configs:
        assert fnmatch(str(path.relative_to(ROOT)), config_glob)


def test_lite_osworld_runbook_invokes_campaign_dir_helper() -> None:
    _assert_sources_and_invokes_campaign_dir_helper(LITE_OSWORLD_RUN_SH)
