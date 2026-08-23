from __future__ import annotations

import pytest

from lite.gym.envs.lite.osworld import exclude_reasons


def test_task_level_exclude_reason_registry_accepts_canonical_values():
    for reason in (
        "infeasible",
        "upstream_live_site_drift",
        "upstream_generated_eval_bug",
        "instruction_eval_mismatch",
        "instruction_setup_mismatch",
        "google_auth",
        "proxy_required",
        "trivial_pass:color_precheck",
        "unverifiable:pdf_generation",
        "missing_dependency:imagemagick",
        "unsupported_setup:close_all_libreoffice",
        "unsupported_schema:action_list",
        "flake:chrome_cdp_active_tab_nav",
    ):
        assert exclude_reasons.validate(reason) == reason


@pytest.mark.parametrize(
    "reason",
    [
        "",
        "block: live-site redirect",
        "block:foo",
        "trivial_pass: color check skipped",
        "missing_dependency",
        "proxy_required:any",
        "infeasible:any",
        "missing_dependency:any",
        "unsupported_schema:any",
        "unsupported_schema:legacy_config",
        "flake:any",
        "Infeasible",
        "made_up_category",
        "reward_vision_disagree",
        "footgun:loop",
    ],
)
def test_task_level_exclude_reason_rejects_prose_and_wrong_namespace(reason):
    with pytest.raises(ValueError):
        exclude_reasons.validate(reason)
