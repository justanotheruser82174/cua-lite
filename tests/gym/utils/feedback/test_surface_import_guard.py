"""Static guards for the gym env feedback surface owner."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

from active_sources import active_source_files, scan_active_sources

ROOT = Path(__file__).resolve().parents[4]

_GYM_FEEDBACK_ERRORS = "lite.gym.utils.feedback.errors"
_GYM_FEEDBACK_ERRORS_PATH = "lite/gym/utils/feedback/errors.py"
_GYM_FEEDBACK_INGRESS = "lite.gym.utils.feedback.ingress"
_GYM_FEEDBACK_INGRESS_PATH = "lite/gym/utils/feedback/ingress.py"
_GYM_FEEDBACK_RESULTS = "lite.gym.utils.feedback.results"
_GYM_FEEDBACK_RESULTS_PATH = "lite/gym/utils/feedback/results.py"
_GYM_FEEDBACK_SURFACE = "lite.gym.utils.feedback.surface"
_GYM_FEEDBACK_SURFACE_PATH = "lite/gym/utils/feedback/surface.py"
_ERROR_HELPERS = {
    "ACTION_ERROR_KINDS",
    "ActionErrorKind",
    "MODEL_ACTION_ERROR_TYPES",
    "BackendExecutionErrorDetail",
    "ContainerActionErrorRecord",
    "ModelVisibleErrorDetail",
    "ToolErrorCarrier",
    "ToolErrorFeedback",
    "append_feedback",
    "current_feedback",
    "error_only_feedback",
    "invalid_action_arguments_message",
    "invalid_action_not_available_message",
    "model_visible_error_detail",
    "parse_container_action_error_record",
    "record_model_action_error",
    "record_tool_execution_error",
    "tool_execution_error_message",
    "unavailable_action_message",
    "unknown_tool_message",
    "unsupported_action_message",
}
_RESULT_HELPERS = {
    "RUNTIME_RESULT_CALL_ID_KEY",
    "assert_safe_tool_call_envelopes",
    "build_tool_results_from_decisions",
    "invalid_env_tool_call_envelope_message",
    "invalid_tool_call_envelope_message",
    "ordered_tool_call_ids",
}
_INGRESS_HELPERS = {
    "ToolCallAvailability",
    "action_satisfies_extra_tool_schema",
    "active_extra_tool_names",
    "classify_standalone_tool_call",
    "extra_tool_schema_argument_error",
    "action_batch_structure_error_message",
    "invalid_action_message",
    "invalid_top_level_action_message",
    "is_active_extra_tool_call",
    "is_lite_action_name_or_action_batch_tool_name",
    "is_inactive_tool_call",
    "is_internal_finish_tool_call",
    "is_loop_detect_terminate",
    "is_unknown_tool_call",
    "make_internal_terminate_action",
    "nested_extra_tool_action_batch_child_message",
    "prepare_env_tool_calls",
    "standalone_tool_call_feedback",
    "standalone_tool_call_feedback_with_reason",
    "unsupported_env_action_message",
}
_DEFAULT_STANDALONE_FEEDBACK_ADOPTERS = {
    "lite/gym/envs/androidlab/main.py",
    "lite/gym/envs/androidworld/main.py",
    "lite/gym/envs/captcha/main.py",
    "lite/gym/envs/cua/bench/main.py",
    "lite/gym/envs/cua/sandbox/env.py",
    "lite/gym/envs/mobilegym/main.py",
    "lite/gym/envs/mobileworld/main.py",
    "lite/gym/envs/online_mind2web/main.py",
    "lite/gym/envs/osworld/main.py",
    "lite/gym/envs/osworld_2/main.py",
    "lite/gym/envs/waa/main.py",
    "lite/gym/envs/webharbor/webvoyager/main.py",
    "lite/gym/sandbox/base.py",
}
_DEFAULT_STANDALONE_FEEDBACK_PRIMITIVES = {
    "classify_standalone_tool_call",
    "is_inactive_tool_call",
    "is_unknown_tool_call",
}
_RETIRED_GYM_UTILS_TOOLS = "lite.gym.utils.tools"
_RETIRED_GYM_UTILS_TOOLS_PATH = "lite/gym/utils/tools.py"
_RETIRED_GYM_UTILS_ACTIONS = "lite.gym.utils.actions"
_RETIRED_GYM_UTILS_ACTIONS_PATH = "lite/gym/utils/actions.py"


def _allowed_retired_text_reference(offender: str) -> bool:
    rel, _lineno, line = offender.split(":", 2)
    if rel == "tests/gym/utils/feedback/test_surface_import_guard.py":
        return True
    return (
        rel == "tests/agents/core/agent/test_trajectory_logger.py"
        and 'assert "lite.gym.utils.actions" not in source' in line
    )


def test_feedback_surface_owner_is_tracked() -> None:
    spec = importlib.util.find_spec(_GYM_FEEDBACK_SURFACE)
    assert spec is not None
    assert spec.origin is not None
    assert Path(spec.origin).resolve() == (ROOT / _GYM_FEEDBACK_SURFACE_PATH).resolve()

    tracked = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--",
            _GYM_FEEDBACK_SURFACE_PATH,
            "lite/gym/utils/feedback/__init__.py",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    assert _GYM_FEEDBACK_SURFACE_PATH in tracked
    assert "lite/gym/utils/feedback/__init__.py" in tracked


def test_retired_flat_feedback_modules_are_absent() -> None:
    assert not (ROOT / _RETIRED_GYM_UTILS_TOOLS_PATH).exists()
    assert not (ROOT / _RETIRED_GYM_UTILS_ACTIONS_PATH).exists()
    assert importlib.util.find_spec(_RETIRED_GYM_UTILS_TOOLS) is None
    assert importlib.util.find_spec(_RETIRED_GYM_UTILS_ACTIONS) is None

    import lite.gym.utils as gym_utils

    assert not hasattr(gym_utils, "tools")
    assert not hasattr(gym_utils, "actions")


def test_retired_flat_feedback_modules_stay_out_of_active_text() -> None:
    offenders = [
        offender
        for offender in scan_active_sources(
            (
                r"\blite\.gym\.utils\.(tools|actions)\b",
                r"from\s+lite\.gym\.utils\s+import\s+[^#\n]*(tools|actions)\b",
                r"(^|/)lite/gym/utils/(tools|actions)\.py\b",
            ),
            exclude=("tests/gym/utils/feedback/test_surface_import_guard.py",),
        )
        if not _allowed_retired_text_reference(offender)
    ]

    assert not offenders, (
        "retired flat gym feedback modules should not appear in active source "
        "text; import from lite.gym.utils.feedback.* instead:\n  "
        + "\n  ".join(offenders)
    )


def test_retired_flat_feedback_import_forms_stay_out_of_python_sources() -> None:
    retired_modules = {_RETIRED_GYM_UTILS_TOOLS, _RETIRED_GYM_UTILS_ACTIONS}
    retired_attrs = {"tools", "actions"}
    offenders: list[str] = []
    for rel in active_source_files():
        if rel == "tests/gym/utils/feedback/test_surface_import_guard.py":
            continue
        if not rel.endswith(".py"):
            continue
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders.extend(
                    f"{rel}:{node.lineno}: import {alias.name}"
                    for alias in node.names
                    if alias.name in retired_modules
                )
            elif isinstance(node, ast.ImportFrom):
                if node.module in retired_modules:
                    offenders.append(f"{rel}:{node.lineno}: from {node.module} import")
                if node.module == "lite.gym.utils":
                    offenders.extend(
                        f"{rel}:{node.lineno}: from lite.gym.utils import {alias.name}"
                        for alias in node.names
                        if alias.name in retired_attrs
                    )

    assert not offenders, (
        "retired flat gym feedback modules should not be imported; "
        "import from lite.gym.utils.feedback.* instead:\n  "
        + "\n  ".join(offenders)
    )


def test_feedback_errors_owner_is_tracked_and_provider_free() -> None:
    spec = importlib.util.find_spec(_GYM_FEEDBACK_ERRORS)
    assert spec is not None
    assert spec.origin is not None
    assert Path(spec.origin).resolve() == (ROOT / _GYM_FEEDBACK_ERRORS_PATH).resolve()

    code = (
        "import sys; import lite.gym.utils.feedback.errors; "
        "bad=[m for m in sys.modules if m == 'lite.agents' or m.startswith('lite.agents.')]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)

    errors = importlib.import_module(_GYM_FEEDBACK_ERRORS)
    for name in _ERROR_HELPERS:
        assert hasattr(errors, name), name

    tracked = subprocess.check_output(
        ["git", "ls-files", "--", _GYM_FEEDBACK_ERRORS_PATH],
        cwd=ROOT,
        text=True,
    ).splitlines()
    assert _GYM_FEEDBACK_ERRORS_PATH in tracked


def test_feedback_results_owner_is_tracked_provider_free_and_not_actions_backed() -> None:
    spec = importlib.util.find_spec(_GYM_FEEDBACK_RESULTS)
    assert spec is not None
    assert spec.origin is not None
    assert Path(spec.origin).resolve() == (ROOT / _GYM_FEEDBACK_RESULTS_PATH).resolve()

    code = (
        "import sys; import lite.gym.utils.feedback.results; "
        "bad=[m for m in sys.modules "
        "if m == 'lite.agents' or m.startswith('lite.agents.') "
        "or m == 'lite.gym.utils.actions']; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)

    results = importlib.import_module(_GYM_FEEDBACK_RESULTS)
    for name in _RESULT_HELPERS:
        assert hasattr(results, name), name

    tree = ast.parse(
        (ROOT / _GYM_FEEDBACK_RESULTS_PATH).read_text(encoding="utf-8"),
        filename=_GYM_FEEDBACK_RESULTS_PATH,
    )
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders.extend(
                f"{_GYM_FEEDBACK_RESULTS_PATH}:{node.lineno}: import {alias.name}"
                for alias in node.names
                if alias.name == "lite.gym.utils.actions"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "lite.gym.utils.actions":
            offenders.append(f"{_GYM_FEEDBACK_RESULTS_PATH}:{node.lineno}: from actions import")

    assert not offenders, (
        f"{_GYM_FEEDBACK_RESULTS} must not import lite.gym.utils.actions:\n  "
        + "\n  ".join(offenders)
    )

    tracked = subprocess.check_output(
        ["git", "ls-files", "--", _GYM_FEEDBACK_RESULTS_PATH],
        cwd=ROOT,
        text=True,
    ).splitlines()
    assert _GYM_FEEDBACK_RESULTS_PATH in tracked


def test_feedback_ingress_owner_is_tracked_provider_free_and_not_actions_backed() -> None:
    spec = importlib.util.find_spec(_GYM_FEEDBACK_INGRESS)
    assert spec is not None
    assert spec.origin is not None
    assert Path(spec.origin).resolve() == (ROOT / _GYM_FEEDBACK_INGRESS_PATH).resolve()

    code = (
        "import sys; import lite.gym.utils.feedback.ingress; "
        "bad=[m for m in sys.modules "
        "if m == 'lite.agents' or m.startswith('lite.agents.') "
        "or m == 'lite.gym.utils.actions']; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)

    ingress = importlib.import_module(_GYM_FEEDBACK_INGRESS)
    for name in _INGRESS_HELPERS:
        assert hasattr(ingress, name), name

    ingress_tree = ast.parse(
        (ROOT / _GYM_FEEDBACK_INGRESS_PATH).read_text(encoding="utf-8"),
        filename=_GYM_FEEDBACK_INGRESS_PATH,
    )
    offenders: list[str] = []
    for node in ast.walk(ingress_tree):
        if isinstance(node, ast.Import):
            offenders.extend(
                f"{_GYM_FEEDBACK_INGRESS_PATH}:{node.lineno}: import {alias.name}"
                for alias in node.names
                if alias.name == "lite.gym.utils.actions"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "lite.gym.utils.actions":
            offenders.append(f"{_GYM_FEEDBACK_INGRESS_PATH}:{node.lineno}: from actions import")

    assert not offenders, (
        f"{_GYM_FEEDBACK_INGRESS} must own ingress validation helpers without "
        "depending on lite.gym.utils.actions:\n  " + "\n  ".join(offenders)
    )

    tracked = subprocess.check_output(
        ["git", "ls-files", "--", _GYM_FEEDBACK_INGRESS_PATH],
        cwd=ROOT,
        text=True,
    ).splitlines()
    assert _GYM_FEEDBACK_INGRESS_PATH in tracked


def test_default_standalone_feedback_envs_use_shared_owner() -> None:
    offenders: list[str] = []

    for rel in sorted(_DEFAULT_STANDALONE_FEEDBACK_ADOPTERS):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"), filename=rel)
        ingress_imports: dict[str, int] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == _GYM_FEEDBACK_INGRESS:
                for alias in node.names:
                    ingress_imports[alias.asname or alias.name] = node.lineno

        shared_feedback_owners = {
            "standalone_tool_call_feedback",
            "standalone_tool_call_feedback_with_reason",
        }
        if not (shared_feedback_owners & ingress_imports.keys()):
            offenders.append(
                f"{rel}: missing standalone_tool_call_feedback import"
            )
        for name in sorted(_DEFAULT_STANDALONE_FEEDBACK_PRIMITIVES & ingress_imports.keys()):
            offenders.append(f"{rel}:{ingress_imports[name]}: imports {name}")

    assert not offenders, (
        "default inactive/unknown standalone tool feedback must use "
        "standalone_tool_call_feedback or standalone_tool_call_feedback_with_reason; "
        "local classifier/predicate branches are "
        "reserved for documented variants:\n  " + "\n  ".join(offenders)
    )
