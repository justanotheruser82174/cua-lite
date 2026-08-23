"""CUAWorld tests split from _cuaworld_support.py: task hooks."""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from lite.gym.envs.lite.cuaworld.src.adapter import run_cuaworld_setup
from lite.gym.errors import CuaWorldVerifierError
from tests.gym.envs.lite._cuaworld_support import (
    _cuaworld_root,
    _fake_env_tree,
    _materials_root,
    _pinned_library,
    _RecordingInterface,
    _registered_non_excluded,
    _require_full_pinned_materials,
    _task_dir_shell_scripts,
)


@pytest.mark.asyncio
async def test_setup_uses_task_spec_hook_filename(tmp_path):
    task = tmp_path / "alternate-setup"
    task.mkdir()
    setup_script = task / "setup_text_task.sh"
    setup_script.write_text("#!/bin/bash\necho alternate-setup\n")
    setup_script.chmod(0o600)
    dependencies = task / "data"
    dependencies.mkdir()
    (dependencies / "scenario.txt").write_text("scenario\n")
    helper = task / "helper.py"
    helper.write_text("#!/usr/bin/env python3\n")
    helper.chmod(0o755)
    interface = _RecordingInterface()

    await run_cuaworld_setup(
        SimpleNamespace(interface=interface),
        task,
        task_spec={
            "hooks": {
                "pre_task": (
                    "/workspace/tasks/alternate-setup/setup_text_task.sh"
                )
            }
        },
    )

    remote = "/tmp/cuaworld_task_sources/alternate-setup"
    assert interface.writes[f"{remote}/setup_text_task.sh"] == (
        b"#!/bin/bash\necho alternate-setup\n"
    )
    assert interface.writes[f"{remote}/data/scenario.txt"] == b"scenario\n"
    assert interface.writes[f"{remote}/helper.py"].startswith(b"#!")
    assert remote + "/setup_text_task.sh" in interface.writes[
        "/tmp/cuaworld_setup.sh"
    ].decode()
    hook = interface.writes["/tmp/cuaworld_setup.sh"].decode()
    assert (
        "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        in hook
    )
    assert any(
        command.startswith("chmod 755 ")
        and f"{remote}/setup_text_task.sh" in command
        and f"{remote}/helper.py" in command
        for command in interface.commands
    )


@pytest.mark.asyncio
async def test_setup_rejects_symlinked_task_dependencies(tmp_path):
    task = tmp_path / "symlinked-setup"
    task.mkdir()
    (task / "setup_task.sh").write_text("true\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (task / "linked").symlink_to(outside, target_is_directory=True)
    interface = _RecordingInterface()

    with pytest.raises(CuaWorldVerifierError) as raised:
        await run_cuaworld_setup(
            SimpleNamespace(interface=interface),
            task,
            task_spec={
                "hooks": {
                    "pre_task": "/workspace/tasks/symlinked-setup/setup_task.sh"
                }
            },
        )

    assert interface.writes == {}
    assert raised.value.phase == "setup"
    assert raised.value.kind == "spawn"
    assert "task hook dependency is not a regular file" in str(raised.value)


@pytest.mark.asyncio
async def test_setup_skips_undeclared_legacy_hook(tmp_path):
    task = tmp_path / "no-setup-hook"
    task.mkdir()
    (task / "setup_task.sh").write_text("exit 99\n")
    interface = _RecordingInterface()

    await run_cuaworld_setup(
        SimpleNamespace(interface=interface),
        task,
        task_spec={"hooks": {}},
    )

    assert interface.writes == {}
    assert interface.commands == []


@pytest.mark.asyncio
async def test_task_hook_content_rewrites_hardcoded_workspace_tasks_path(tmp_path):
    """`/workspace/tasks/<task>` must be rewritten in the hook CONTENT, not just the
    hook path.

    That directory does not exist in any lite.cuaworld image (verified: `ls -d
    /workspace/tasks` fails in diagrams_net/imagej/webots; /workspace holds only
    assets, config, data, env.json, scripts) — task materials are uploaded to
    `/tmp/cuaworld_task_sources/<task>/`. `_run_task_hook` already rewrote the hook
    PATH; the content rewrite is what makes the 11 on-disk `.sh` files (10 registered
    ∧ non-excluded) that hardcode the string actually find their fixtures.
    """
    task = tmp_path / "hardcoded-root"
    task.mkdir()
    (task / "data").mkdir()
    (task / "data" / "scenario.wbt").write_text("world\n")
    (task / "setup_task.sh").write_text(
        "#!/bin/bash\n"
        'TASK_WORLD="/workspace/tasks/hardcoded-root/data/scenario.wbt"\n'
        "cp \"$TASK_WORLD\" /home/ga/scenario.wbt\n"
        "python3 /workspace/tasks/hardcoded-root/parse_results.py\n"
    )
    interface = _RecordingInterface()

    await run_cuaworld_setup(
        SimpleNamespace(interface=interface),
        task,
        task_spec={
            "hooks": {"pre_task": "/workspace/tasks/hardcoded-root/setup_task.sh"}
        },
    )

    remote = "/tmp/cuaworld_task_sources/hardcoded-root"
    body = interface.writes[f"{remote}/setup_task.sh"].decode()
    assert "/workspace/tasks" not in body
    assert f'TASK_WORLD="{remote}/data/scenario.wbt"' in body
    assert f"python3 {remote}/parse_results.py" in body
    # Non-shell payloads are uploaded byte-for-byte; only `.sh` is normalized.
    assert interface.writes[f"{remote}/data/scenario.wbt"] == b"world\n"


@pytest.mark.asyncio
async def test_task_hook_content_guards_the_helpers_its_own_library_names(tmp_path):
    """The guard reaches TASK hook file CONTENT, and its name set comes from the
    `task_utils.sh` the task's env bakes — not from a hardcoded function name.

    Both abort-capable shapes must be covered: the bare call (kstars_sim's 59
    unguarded `wait_for_slew_complete` sites) and the plain assignment from a command
    substitution (`WID=$(wait_for_gmat_window 60)`, which a whole-line bare-call
    anchor cannot see at all). An already-guarded call, an ERROR-intent helper from
    the same library, and a `local` assignment (whose rc the builtin masks anyway)
    are all left alone.
    """
    task = _fake_env_tree(tmp_path, "kstars_sim", "slew-task")
    (task / "setup_task.sh").write_text(
        "#!/bin/bash\n"
        "set -e\n"
        "wait_for_slew_complete 20\n"
        "    wait_for_slew_complete 15\n"
        "wait_for_slew_complete 25 || true\n"
        "STATE=$(wait_for_slew_complete 30)\n"
        "local CACHED=$(wait_for_slew_complete 30)\n"
        "echo done\n"
    )
    interface = _RecordingInterface()

    await run_cuaworld_setup(
        SimpleNamespace(interface=interface),
        task,
        task_spec={"hooks": {"pre_task": "/workspace/tasks/slew-task/setup_task.sh"}},
    )

    body = interface.writes[
        "/tmp/cuaworld_task_sources/slew-task/setup_task.sh"
    ].decode()
    assert "wait_for_slew_complete 20 || true\n" in body
    assert "    wait_for_slew_complete 15 || true\n" in body
    assert "STATE=$(wait_for_slew_complete 30) || true\n" in body
    assert "local CACHED=$(wait_for_slew_complete 30)\n" in body
    assert "|| true || true" not in body


@pytest.mark.asyncio
async def test_task_hook_guard_is_inert_without_errexit_and_for_error_helpers(
    tmp_path,
):
    """Two deliberate limits, both end-to-end through the upload path.

    Without `set -e` the helper's `return 1` aborts nothing, so rewriting would only
    move the script's own exit status — which `run_cuaworld_setup` reads and logs.
    And an ERROR-intent helper is never guarded: imagej announces "ERROR: Fiji not
    found" from `launch_fiji`, and turning that crash into a silent continue is
    exactly the failure this campaign ranks as worse than the crash.
    """
    quiet = _fake_env_tree(tmp_path / "a", "kstars_sim", "no-errexit")
    (quiet / "setup_task.sh").write_text(
        "#!/bin/bash\nwait_for_slew_complete 20\necho done\n"
    )
    loud = _fake_env_tree(tmp_path / "b", "imagej", "error-helper")
    (loud / "setup_task.sh").write_text(
        "#!/bin/bash\nset -e\nlaunch_fiji\nwait_for_fiji 30\n"
    )

    for task, name in ((quiet, "no-errexit"), (loud, "error-helper")):
        interface = _RecordingInterface()
        await run_cuaworld_setup(
            SimpleNamespace(interface=interface),
            task,
            task_spec={"hooks": {"pre_task": f"/workspace/tasks/{name}/setup_task.sh"}},
        )
        body = interface.writes[
            f"/tmp/cuaworld_task_sources/{name}/setup_task.sh"
        ].decode()
        if name == "no-errexit":
            assert body == (task / "setup_task.sh").read_text()
        else:
            assert "launch_fiji\n" in body and "launch_fiji || true" not in body
            # ...while the SILENT sibling in the same library still gets guarded.
            assert "wait_for_fiji 30 || true\n" in body


@pytest.mark.parametrize(
    "line",
    [
        "wait_for_slew_complete 20 || true",  # already guarded
        "wait_for_slew_complete() {",  # the helper's own definition
        "if wait_for_slew_complete 20; then :; fi",  # rc already consumed
        "wait_for_slew_complete 20 &",  # backgrounded
        "wait_for_slew_complete 20  # settle",  # `|| true` would land in a comment
        "echo wait_for_slew_complete",  # not a call
        "my_wait_for_slew_complete 20",  # a different, longer identifier
        "local WID=$(wait_for_slew_complete 20)",  # `local` already masks the rc
        "export WID=$(wait_for_slew_complete 20)",  # so does `export`
        'WID="$(wait_for_slew_complete 20) extra"',  # not a lone substitution
    ],
)
def test_helper_guard_leaves_non_abort_shapes_alone(line):
    from lite.gym.envs.lite.cuaworld.src.adapter import _guard_hook_body

    text = f"#!/bin/bash\nset -e\n{line}\necho done\n"
    assert _guard_hook_body(text, ("wait_for_slew_complete",)) == text


def test_helper_guard_is_what_actually_stops_the_abort():
    """Behavioral proof in real bash against the REAL pinned library, on the shape a
    whole-line bare-call anchor cannot match.

    Reproduces `openvsp/openvsp_fuselage_lofting/setup_task.sh:58`
    (`WID=$(wait_for_openvsp 60)` under `set -e`): as shipped the assignment inherits
    the substitution's rc=1 and the hook dies before its first screenshot; guarded,
    the author's whole `else` branch runs and the hook exits 0.
    """
    import subprocess

    from lite.gym.envs.lite.cuaworld.src.adapter import (
        _guard_hook_body,
        _guardable_helper_names,
    )

    library = _pinned_library("openvsp")
    names = _guardable_helper_names(library.read_text())
    assert "wait_for_openvsp" in names  # `echo ""; return 1` -> SILENT intent

    script = (
        "#!/bin/bash\n"
        "set -e\n"
        f"source {library}\n"
        "WID=$(wait_for_openvsp 2)\n"
        'if [ -n "$WID" ]; then\n'
        "    echo LAUNCHED\n"
        "else\n"
        '    echo "WARNING: OpenVSP did not appear"\n'
        "fi\n"
        "echo REACHED-TAIL\n"
    )
    guarded = _guard_hook_body(script, names)
    assert guarded != script

    env = {"PATH": os.environ["PATH"], "DISPLAY": ":99"}
    before = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env
    )
    after = subprocess.run(
        ["bash", "-c", guarded], capture_output=True, text=True, env=env
    )

    assert before.returncode == 1 and "REACHED-TAIL" not in before.stdout
    assert after.returncode == 0
    assert "WARNING: OpenVSP did not appear" in after.stdout
    assert "REACHED-TAIL" in after.stdout


def test_guardable_helper_names_reads_intent_out_of_the_pinned_libraries():
    """The name set is DERIVED, so pin what the derivation finds in the real
    libraries — 56 (software, helper) pairs across the 35 that ship one — and the
    intent calls that keep it honest at both edges."""
    from lite.gym.envs.lite.cuaworld.src.adapter import _guardable_helper_names

    _require_full_pinned_materials()
    libraries = sorted(_materials_root().glob("*/*/scripts/task_utils.sh"))
    assert len(libraries) == 35

    found = {
        library.relative_to(_materials_root()).parts[0]: set(
            _guardable_helper_names(library.read_text(errors="surrogateescape"))
        )
        for library in libraries
    }
    assert sum(len(names) for names in found.values()) == 56

    # WARNING intent: the author says in words that this is not a failure.
    assert "wait_for_slew_complete" in found["kstars_sim"]
    # SILENT intent: `echo ""; return 1` is a designed "returned nothing".
    assert "wait_for_openvsp" in found["openvsp"]
    # SILENT with no print at all, and a `for` lookup rather than a poll. Every one
    # of its call sites is `FIJI_PATH=$(find_fiji_executable)` immediately followed
    # by `if [ -z "$FIJI_PATH" ]; then echo "ERROR: Fiji not found!"; exit 1; fi` —
    # the author's error handling IS the emptiness test, and the abort pre-empts it.
    assert "find_fiji_executable" in found["imagej"]
    # ERROR intent, same library: never guarded.
    assert "launch_fiji" not in found["imagej"]
    # The message wins over the exit's neighbours: `launch_openlca` prints
    # "WARNING: …" and then dumps a log before `return 1`.
    assert "launch_openlca" in found["openlca"]
    # A bare `Timeout: …` states a fact without ruling on severity -> left out.
    assert "wait_for_window" not in found["qgis"]
    # The SAME NAME carries opposite intent across libraries, which is why the set is
    # per-software and a corpus-wide union would be wrong.
    assert "wait_for_window" in found["ardour"]
    assert "wait_for_window" not in found["vlc_media_player"]
    # A helper with no failure exit at all is not in scope.
    assert "focus_window" not in found["slicer3d"]


def test_helper_guard_blast_radius_over_the_pinned_materials():
    """Pin the blast radius of the helper guard by running PRODUCTION over every
    pinned hook, and prove the rewrite is a strict improvement.

    Counts are on the ON-DISK basis (every `.sh` under the 3216 task dirs), with the
    registered (3083) and registered ∧ non-excluded subsets spelled out in the
    current handoff.
    """
    import subprocess

    from lite.gym.envs.lite.cuaworld.src.adapter import (
        _guard_helper_calls,
        _guard_hook_body,
        _hook_helpers,
    )

    root = _materials_root()
    _require_full_pinned_materials()
    scripts = _task_dir_shell_scripts()
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )
    registered_ids: dict[str, set] = {}

    def live(software: str, task: str) -> bool:
        if (excludes.get(software) or {}).get(task):
            return False
        if software not in registered_ids:
            catalog = json.loads(
                next(root.glob(f"{software}/*/registered.json")).read_text()
            )
            registered_ids[software] = {
                task_id
                for ids in catalog.values()
                if isinstance(ids, list)
                for task_id in ids
            }
        return task in registered_ids[software]

    # The single-name rule this guard replaces, so its 50 live tasks can be shown to
    # be a strict subset rather than merely re-counted.
    superseded = re.compile(
        r"(?m)^([ \t]*wait_for_slew_complete(?:[ \t]+[^\r\n|&;()<>#]*?)?)[ \t]*$"
    )

    identical = files = sites = 0
    changed: set[tuple[str, str]] = set()
    old_rule: set[tuple[str, str]] = set()
    parse_before = parse_after = 0
    with tempfile.TemporaryDirectory() as tmp:
        for index, source in enumerate(scripts):
            # `.cache/<sw>/<env>/tasks/<task>/.../<file>.sh`
            software, _, _, task = source.relative_to(root).parts[:4]
            task_dir = next(root.glob(f"{software}/*/tasks/{task}"))
            original = source.read_text(encoding="utf-8", errors="surrogateescape")
            helpers = _hook_helpers(task_dir)
            guarded = _guard_helper_calls(original, helpers)
            for site in superseded.finditer(original):
                old_rule.add((software, task))
                # every site the old rule fixed is still fixed
                assert f"{site.group(1)} || true" in guarded, source
            if guarded == original:
                identical += 1
                continue
            files += 1
            sites += guarded.count(" || true") - original.count(" || true")
            changed.add((software, task))
            assert "|| true || true" not in guarded, source
            # The helper guard must reach the corpus through `_guard_hook_body`,
            # which is the only thing `_run_task_hook` calls.
            assert _guard_hook_body(original, helpers) != _guard_hook_body(
                original, ()
            ), source
            # Never INTRODUCE a parse error. Five upstream files already fail
            # `bash -n` unrewritten (openemr create_cardiology_referral, slicer3d
            # create_mip_visualization / export_landmarks_csv /
            # fill_segmentation_holes, blender3d python_exoplanet_scatter_plot), so
            # the honest invariant is "no worse", not "always clean".
            after = Path(tmp) / f"{index}.sh"
            after.write_text(
                _guard_hook_body(original, helpers),
                encoding="utf-8",
                errors="surrogateescape",
            )
            parse_before += bool(
                subprocess.run(["bash", "-n", str(source)], capture_output=True).returncode
            )
            parse_after += bool(
                subprocess.run(["bash", "-n", str(after)], capture_output=True).returncode
            )
            after.unlink()

    # 782, not 781: the call pattern now tolerates a trailing REDIRECTION, which adds
    # exactly one site — `openlca/waste_treatment_linkage_setup:20`
    # (`ensure_uslci_database > /dev/null`). The argument run excludes `<>` so a
    # pipeline still cannot be swallowed; `>` merely stopped the whole-line anchor
    # from matching, leaving that setup to abort under `set -e`. Pinning these three
    # IS right — unlike the live count below, they are properties of the pinned
    # materials and of the guard, so they move only when one of those deliberately does.
    assert identical == 5863 and files == 512 and sites == 782
    assert parse_after <= parse_before == 0
    assert len(changed) == 507
    # NOT a pinned constant: the live count is a FUNCTION of validation_excludes.json,
    # so every correct new exclusion moves it (354 -> 351 when the forged-artifact
    # sweep landed 11 entries). Pinning it makes an unrelated, correct change look
    # like a regression — the "right figure on the wrong denominator" trap that has
    # cost this campaign more than any single bug. Pin the invariant instead:
    # every live site is registered, and the live set shrinks only via exclusions.
    live_sites = sum(1 for key in changed if live(*key))
    registered_sites = sum(1 for key in changed if key[1] in registered_ids[key[0]])
    assert registered_sites == 448
    assert live_sites <= registered_sites
    assert live_sites == registered_sites - sum(
        1 for key in changed
        if key[1] in registered_ids[key[0]] and not live(*key)
    )
    # 7x the single-name rule it replaces, and a strict superset of it.
    assert len(old_rule) == 59
    assert sum(1 for key in old_rule if live(*key)) == 50
    assert old_rule <= changed


def test_helper_guard_never_moves_a_script_or_function_exit_status():
    """`|| true` is only safe where nothing downstream reads the rc it erases.

    Measured over the pinned materials: of the 782 guarded sites, ZERO are the last
    statement of their script or of an enclosing shell function, so no hook's exit
    code (which `run_cuaworld_setup` logs and `run_cuaworld_verify` branches on) and
    no task-local function's return value changes.
    """
    from lite.gym.envs.lite.cuaworld.src.adapter import (
        _helper_guard_patterns,
        _hook_helpers,
    )

    root = _materials_root()
    _require_full_pinned_materials()
    scripts = _task_dir_shell_scripts()

    terminal = 0
    checked = 0
    for source in scripts:
        software, _, _, task = source.relative_to(root).parts[:4]
        helpers = _hook_helpers(next(root.glob(f"{software}/*/tasks/{task}")))
        if not helpers:
            continue
        text = source.read_text(encoding="utf-8", errors="surrogateescape")
        if not re.search(r"(?m)^[ \t]*set[ \t]+-(?:[a-zA-Z]*e[a-zA-Z]*|o[ \t]+errexit)\b", text):
            continue
        lines = text.splitlines()
        patterns = [re.compile(p) for p in _helper_guard_patterns(helpers)]
        for index, line in enumerate(lines):
            if not any(p.match(line) for p in patterns):
                continue
            checked += 1
            rest = [
                later.strip()
                for later in lines[index + 1:]
                if later.strip() and not later.strip().startswith("#")
            ]
            if not rest or rest[0] == "}":
                terminal += 1
    assert checked == 782
    assert terminal == 0


@pytest.mark.parametrize(
    "line",
    [
        '    "count": ${COUNT:-null},',  # already defaulted
        '    "count": $(wc -l < /tmp/f),',  # a command substitution, not a slot
        '    "count": $COUNT + 1,',  # slot is an expression, not one expansion
        '    "count": "$COUNT",',  # already a string literal
        '    "note": $COUNT # trailing',  # a comment would swallow the rewrite
        "    count = $COUNT",  # `name = $VAR` shape is deliberately out of scope
        "Exec=$VSPBIN",  # the only such line in an image-baked post_start script
    ],
)
def test_empty_slot_guard_leaves_non_slot_lines_alone(line):
    from lite.gym.envs.lite.cuaworld.src.adapter import _guard_hook_body

    text = f"#!/bin/bash\ncat > /tmp/r.json <<EOF\n{{\n{line}\n}}\nEOF\n"
    assert _guard_hook_body(text, ()) == text


def test_empty_slot_guard_is_what_actually_saves_the_result_json():
    """Behavioral proof in real bash + real python3, mirroring the reproduction on
    `slicer3d/convert_volume_nifti/export_result.sh`: `NIFTI_DIMENSIONS=""` survives
    the `[ -f … ]` guard, the heredoc emits `"nifti_dimensions": ,` and the verifier's
    `json.load` dies — "Could not retrieve result JSON" on a real run."""
    import subprocess

    from lite.gym.envs.lite.cuaworld.src.adapter import _guard_hook_body

    script = (
        "#!/bin/bash\n"
        'NIFTI_DIMENSIONS=""\n'
        "VALID=false\n"
        'cat > "$OUT" <<EOF\n'
        "{\n"
        '    "valid_nifti": $VALID,\n'
        '    "nifti_dimensions": $NIFTI_DIMENSIONS\n'
        "}\n"
        "EOF\n"
        'python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$OUT"\n'
    )
    guarded = _guard_hook_body(script, ())
    assert guarded != script

    def run(body: str, out: str):
        return subprocess.run(
            ["bash", "-c", body], capture_output=True, text=True, env={"OUT": out}
        )

    with tempfile.TemporaryDirectory() as tmp:
        before = run(script, f"{tmp}/before.json")
        after = run(guarded, f"{tmp}/after.json")
        assert before.returncode != 0 and "JSONDecodeError" in before.stderr
        assert after.returncode == 0, after.stderr
        assert json.loads(Path(f"{tmp}/after.json").read_text()) == {
            "valid_nifti": False,
            "nifti_dimensions": None,
        }

    # A non-empty value is passed through byte-for-byte: `${V:-null}` == `$V`.
    populated = script.replace('NIFTI_DIMENSIONS=""', "NIFTI_DIMENSIONS='[2,3]'")
    with tempfile.TemporaryDirectory() as tmp:
        assert run(populated, f"{tmp}/raw.json").returncode == 0
        assert run(_guard_hook_body(populated, ()), f"{tmp}/fix.json").returncode == 0
        assert (
            Path(f"{tmp}/raw.json").read_text() == Path(f"{tmp}/fix.json").read_text()
        )


def test_material_hook_fixups_reach_task_hook_upload_path():
    from lite.gym.envs.lite.cuaworld.src.adapter import (
        _ASTRO_ALIGN_SLOW_EXTRACT_BLOCK,
        _guard_hook_body,
    )

    openemr = _guard_hook_body(
        "#!/bin/bash\nset -e\ncd /home/ga/openemr\nsystemctl start apache2\n",
        (),
    )
    assert "cd /var/www/html/openemr" in openemr
    assert "cd /home/ga/openemr" in openemr

    recist = _guard_hook_body(
        'if [ -f "$TARGET_DIR/ct_volume.nii.gz" ]; then\n'
        '    CT_FILE="$TARGET_DIR/ct_volume.nii.gz"\n'
        'elif [ -d "$TARGET_DIR/PATIENT_DICOM" ] && '
        '[ "$(ls -1 "$TARGET_DIR/PATIENT_DICOM" 2>/dev/null | wc -l)" -gt 0 ]; then\n'
        '    CT_FILE="$TARGET_DIR/PATIENT_DICOM"\n'
        "fi\n",
        (),
    )
    assert 'patient_${PATIENT_NUM}_ct.nii.gz' in recist

    tumor_vessel = _guard_hook_body(
        'PATIENT_DIR="$IRCADB_DIR/patient_${PATIENT_NUM}"\n'
        'CT_FILE="$IRCADB_DIR/ircadb_patient${PATIENT_NUM}.nii.gz"\n'
        'SEG_FILE="$IRCADB_DIR/ircadb_patient${PATIENT_NUM}_seg.nrrd"\n'
        'if [ -d "$PATIENT_DIR/PATIENT_DICOM" ]; then\n'
        '    DATA_SOURCE="DICOM"\n'
        'fi\n',
        (),
    )
    assert 'SYNTHETIC_CT_FILE="$PATIENT_DIR/patient_${PATIENT_NUM}_ct.nii.gz"' in tumor_vessel
    assert 'ln -sf "$SYNTHETIC_CT_FILE" "$CT_FILE"' in tumor_vessel

    slicer_wait = _guard_hook_body("wait_for_slicer 90\n", ())
    assert "CUA-Lite Slicer startup modal normalization" in slicer_wait
    assert 'xdotool key "$key"' in slicer_wait

    slicer_direct = _guard_hook_body(
        'DISPLAY=:1 wmctrl -a "Slicer" 2>/dev/null || true\nsleep 2\n',
        (),
    )
    assert "CUA-Lite Slicer startup modal normalization" in slicer_direct

    astro_align = _guard_hook_body(
        "python3 << 'PYEOF'\n" + _ASTRO_ALIGN_SLOW_EXTRACT_BLOCK + "PYEOF\n",
        (),
    )
    assert "tar.getmembers()" not in astro_align
    assert "extractall" not in astro_align
    assert "shutil.rmtree" not in astro_align
    assert "tarfile.open(WASP12_CACHE, 'r|gz')" in astro_align
    assert "Expected 20 FITS frames" in astro_align

    slicer_multi_launch = _guard_hook_body(
        "wait_for_slicer 60\n"
        "echo first\n"
        "wait_for_slicer 120\n"
        'wmctrl -a "Slicer" 2>/dev/null || true\n',
        (),
    )
    assert slicer_multi_launch.count("CUA-Lite Slicer startup modal normalization") == 3
    assert _guard_hook_body(slicer_multi_launch, ()) == slicer_multi_launch


@pytest.mark.asyncio
async def test_setup_upload_does_not_apply_material_fixups_to_export_hook(tmp_path):
    task = tmp_path / "mixed-hook"
    task.mkdir()
    (task / "setup_task.sh").write_text("wait_for_slicer 60\n")
    (task / "export_result.sh").write_text("wait_for_slicer 60\n")
    interface = _RecordingInterface()

    await run_cuaworld_setup(
        SimpleNamespace(interface=interface),
        task,
        task_spec={
            "hooks": {
                "pre_task": "/workspace/tasks/mixed-hook/setup_task.sh",
                "post_task": "/workspace/tasks/mixed-hook/export_result.sh",
            }
        },
    )

    remote = "/tmp/cuaworld_task_sources/mixed-hook"
    setup_body = interface.writes[f"{remote}/setup_task.sh"].decode()
    export_body = interface.writes[f"{remote}/export_result.sh"].decode()
    assert "CUA-Lite Slicer startup modal normalization" in setup_body
    assert export_body == "wait_for_slicer 60\n"


@pytest.mark.asyncio
async def test_freecad_export_execution_fixup_uses_freecadcmd(tmp_path):
    """FreeCAD geometry modules must run under FreeCAD's command interpreter.

    `create_oring_groove/export_result.sh` wrote a FreeCAD script and then ran it
    with bare python3. In this image family `python3 -c "import Part"` can
    segfault, while freecadcmd is the intended entry point for the same script.
    This is an execution fixup, not a material/UI fixup, so it still applies to
    the export hook path where setup-only material rewrites stay disabled.
    """
    task = tmp_path / "create_oring_groove"
    task.mkdir()
    (task / "setup_task.sh").write_text("echo setup\n")
    (task / "export_result.sh").write_text(
        "cat > /tmp/analyze_geometry.py << 'PYEOF'\n"
        "import FreeCAD\n"
        "import Part\n"
        "PYEOF\n"
        "if python3 /tmp/analyze_geometry.py > /tmp/geo_out.json 2>/dev/null; then\n"
        "    GEOMETRY_JSON=$(cat /tmp/geo_out.json)\n"
        "else\n"
        "    GEOMETRY_JSON='{\"error\": \"Analysis script failed\"}'\n"
        "fi\n"
    )
    interface = _RecordingInterface()

    await run_cuaworld_setup(
        SimpleNamespace(interface=interface),
        task,
        task_spec={
            "hooks": {
                "pre_task": "/workspace/tasks/create_oring_groove/setup_task.sh",
                "post_task": "/workspace/tasks/create_oring_groove/export_result.sh",
            }
        },
    )

    remote = "/tmp/cuaworld_task_sources/create_oring_groove"
    export_body = interface.writes[f"{remote}/export_result.sh"].decode()
    assert "freecadcmd /tmp/analyze_geometry.py" in export_body
    assert "if python3 /tmp/analyze_geometry.py" not in export_body


@pytest.mark.asyncio
async def test_astro_align_streaming_fixup_reaches_uploaded_pinned_hook(tmp_path):
    """Regression proof for `astroimagej/align_and_crop_image_sequence`.

    The local `.cache` tree is gitignored and may already contain a hand-edited
    streaming script from a live debug session. This test builds a fresh fake
    materials tree from the pinned slow hook shape and checks the file content
    uploaded to `/tmp/cuaworld_task_sources`, which is the production path.
    """
    task = (
        tmp_path
        / "astroimagej"
        / ".cache"
        / "astroimagej_env"
        / "tasks"
        / "align_and_crop_image_sequence"
    )
    task.mkdir(parents=True)
    (task / "setup_task.sh").write_text(
        textwrap.dedent(
            """\
            #!/bin/bash
            echo "=== Setting up Align and Crop Image Sequence Task ==="

            source /workspace/scripts/task_utils.sh

            RAW_DIR="/home/ga/AstroImages/raw_sequence"
            OUT_DIR="/home/ga/AstroImages/aligned_sequence"

            rm -rf "$RAW_DIR" "$OUT_DIR"
            mkdir -p "$RAW_DIR" "$OUT_DIR"

            python3 << 'PYEOF'
            import os
            import glob
            import tarfile
            import shutil
            import numpy as np
            from astropy.io import fits

            WASP12_CACHE = "/opt/fits_samples/WASP-12b_calibrated.tar.gz"
            RAW_DIR = "/home/ga/AstroImages/raw_sequence"

            if not os.path.exists(WASP12_CACHE):
                print(f"ERROR: Cached data not found at {WASP12_CACHE}")
                exit(1)

            # Extract first 20 FITS files to a temporary directory
            TMP_EXTRACT = "/tmp/wasp_extract"
            os.makedirs(TMP_EXTRACT, exist_ok=True)

            print("Extracting frames from archive...")
            with tarfile.open(WASP12_CACHE, 'r:gz') as tar:
                members = [m for m in tar.getmembers() if m.name.endswith('.fits')]
                members = sorted(members, key=lambda x: x.name)[:20]
                tar.extractall(path=TMP_EXTRACT, members=members)

            extracted_files = sorted(glob.glob(f"{TMP_EXTRACT}/**/*.fits", recursive=True))

            print(f"Injecting severe tracking drift into {len(extracted_files)} frames...")
            for i, fpath in enumerate(extracted_files):
                # Read data
                with fits.open(fpath) as hdul:
                    data = hdul[0].data
                    hdr = hdul[0].header

                    # Inject artificial drift (dx = 3px/frame, dy = 2px/frame)
                    dy, dx = i * 2, i * 3

                    shifted = np.zeros_like(data)
                    if dy > 0 and dx > 0:
                        shifted[dy:, dx:] = data[:-dy, :-dx]
                    else:
                        shifted = data.copy()

                    # Write to raw_sequence directory
                    out_name = f"drifting_frame_{i:02d}.fits"
                    out_path = os.path.join(RAW_DIR, out_name)
                    fits.writeto(out_path, shifted, hdr, overwrite=True)

            # Cleanup temp
            shutil.rmtree(TMP_EXTRACT)
            print("Drifting sequence prepared successfully.")
            PYEOF
            """
        )
    )
    interface = _RecordingInterface()

    await run_cuaworld_setup(
        SimpleNamespace(interface=interface),
        task,
        task_spec={
            "hooks": {
                "pre_task": (
                    "/workspace/tasks/align_and_crop_image_sequence/setup_task.sh"
                )
            }
        },
    )

    remote = "/tmp/cuaworld_task_sources/align_and_crop_image_sequence"
    body = interface.writes[f"{remote}/setup_task.sh"].decode()
    assert "tar.getmembers()" not in body
    assert "tar.extractall" not in body
    assert "shutil.rmtree" not in body
    assert "tarfile.open(WASP12_CACHE, 'r|gz')" in body
    assert "Expected 20 FITS frames" in body
    assert "io.BytesIO(src.read())" in body


def test_empty_slot_guard_population_over_the_pinned_materials():
    """Pin the population the empty-slot rule covers, on the ON-DISK basis (every
    `.sh` under the task dirs) with the registered ∧ non-excluded subset spelled out.

    This shape — a whole line whose value is exactly one bare `$VAR` after a
    double-quoted key — is what makes the count detector-sensitive: restricting it to
    python heredocs (the `_HOOK_SITECUSTOMIZE` reading) gives a far smaller figure
    than counting every JSON heredoc, and both are the same defect.
    """
    from lite.gym.envs.lite.cuaworld.src.adapter import _EMPTY_SLOT_RE

    root = _materials_root()
    _require_full_pinned_materials()
    scripts = _task_dir_shell_scripts()

    sites = 0
    tasks: set[tuple[str, str]] = set()
    for source in scripts:
        software, _, _, task = source.relative_to(root).parts[:4]
        text = source.read_text(encoding="utf-8", errors="surrogateescape")
        found = len(_EMPTY_SLOT_RE.findall(text))
        if found:
            sites += found
            tasks.add((software, task))

    # `sites` and `tasks` are properties of the PINNED materials, so pinning them is
    # right — they move only when the materials revision moves.
    assert sites == 13939
    assert len(tasks) == 2090
    # The live count is NOT: it is a function of validation_excludes.json and drops
    # with every correct new exclusion (1510 -> 1499 when the forged-artifact sweep
    # landed 11). Pinning it turns a correct exclusion into a spurious test failure.
    # What IS invariant: the guard reaches a large, non-trivial share of them.
    live = sum(_registered_non_excluded(*k) for k in tasks)
    assert 0 < live <= len(tasks)
    assert live > len(tasks) // 2, (
        f"only {live} of {len(tasks)} empty-slot tasks are live — if exclusions "
        "really grew that much, re-derive this guard's value before relaxing this"
    )


def test_workspace_tasks_path_rewrite_population_over_the_pinned_materials():
    """Pin the population of the `/workspace/tasks/<task>` content rewrite.

    This measures the MATERIALS only; it deliberately does not re-implement the
    rewrite. The predecessor of this test did — it ran `str.replace` itself and then
    asserted on its own output, so it would have passed with the production rewrite
    deleted. The rewrite is proved where production actually performs it, in
    `test_task_hook_content_rewrites_hardcoded_workspace_tasks_path`.
    """
    root = _materials_root()
    _require_full_pinned_materials()
    scripts = _task_dir_shell_scripts()

    hardcoded: set[tuple[str, str]] = set()
    for source in scripts:
        software, _, _, task = source.relative_to(root).parts[:4]
        text = source.read_text(encoding="utf-8", errors="surrogateescape")
        if f"/workspace/tasks/{task}" in text:
            hardcoded.add((software, task))

    # 11 `.sh` files hardcode /workspace/tasks, one per task dir; 10 are registered
    # ∧ non-excluded (diagrams_net/microservices_outage_rca is excluded).
    assert len(hardcoded) == 11
    assert {software for software, _ in hardcoded} == {
        "diagrams_net",
        "imagej",
        "webots",
    }
    assert sum(_registered_non_excluded(*k) for k in hardcoded) == 10


def test_post_start_rewrite_shares_one_source_of_truth_with_the_host_guard():
    """The container-side post_start rewrite injects the SAME compiled patterns AND
    the helper guard's own SOURCE, so the two paths cannot drift apart.

    The helper guard cannot travel as a rendered pattern: its name set is
    per-software (`wait_for_window` is silent in ardour and `❌ Timeout` in vlc) and
    `run_cuaworld_post_start` only ever holds a container. Shipping the code lets the
    container derive its own set from the `task_utils.sh` it baked.
    """
    import ast
    import subprocess

    from lite.gym.envs.lite.cuaworld.src.adapter import (
        _COUNT_FALLBACK_RE,
        _EMPTY_SLOT_RE,
        _SEARCH_NOMATCH_RE,
        _guard_helper_calls,
        _guard_post_start_body,
        _guardable_helper_names,
        _helper_guard_source,
        _post_start_command,
    )

    command = _post_start_command()
    for pattern in (_COUNT_FALLBACK_RE, _EMPTY_SLOT_RE, _SEARCH_NOMATCH_RE):
        assert pattern.pattern in command
    for placeholder in ("__COUNT_RE__", "__SLOT_RE__", "__SEARCH_RE__",
                        "__HELPER_GUARD_SRC__"):
        assert placeholder not in command
    # This whole command is passed as `bash -c <command>`. Keep it free of
    # product names that an upstream `pkill -f <product>` cleanup can match.
    for pkill_pattern in ("librecad", "solvespace", "pymol", "ugene"):
        assert pkill_pattern not in command.lower()

    # The injected code must be pure ASCII: it crosses a shell heredoc into the
    # container's python3, and the corpus's two severity pictographs are spelled as
    # escapes for exactly that reason.
    injected = _helper_guard_source()
    assert injected.isascii()
    assert injected in command
    # It has to be WIRED UP, not merely present.
    # The command runs two python heredocs; the SECOND one is the rewrite.
    body = command.rsplit("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    ast.parse(body)
    assert "_guard_helper_calls(text, _guardable_helper_names(library))" in body
    assert '"/workspace/scripts/task_utils.sh"' in body

    # Run the injected definitions in a namespace that has ONLY `re` — the container
    # gets nothing else — and prove they are byte-identical to the host functions
    # over every pinned library and every hook the host guard rewrites.
    shipped: dict = {"re": re}
    exec(compile(injected, "<injected>", "exec"), shipped)
    root = _materials_root()
    _require_full_pinned_materials()
    libraries = sorted(root.glob("*/*/scripts/task_utils.sh"))
    compared = 0
    for library in libraries:
        text = library.read_text(errors="surrogateescape")
        names = _guardable_helper_names(text)
        assert shipped["_guardable_helper_names"](text) == names
        for hook in sorted(library.parent.parent.glob("tasks/*/**/*.sh")):
            source = hook.read_text(encoding="utf-8", errors="surrogateescape")
            assert shipped["_guard_helper_calls"](source, names) == _guard_helper_calls(
                source, names
            )
            compared += 1
    assert compared > 5000

    # End to end: the real heredoc, run by a real python3. `/workspace` does not
    # exist on the host, so this also pins the fail-safe — a missing library means
    # "guard nothing", never a crash that would take post_start down with it.
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.sh", Path(tmp) / "out.sh"
        src.write_text(
            "set -euo pipefail\n"
            'WID=$(xdotool search --name "GMAT" 2>/dev/null | head -1)\n'
            "wait_for_slew_complete 20\n"
            'N=$(grep -c PATTERN /tmp/f || echo "0")\n'
        )
        done = subprocess.run(
            [sys.executable, "-c", body, str(src), str(dst)],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        rewritten = dst.read_text()
        assert "|| true )" in rewritten
        assert "awk 'NR==1'" in rewritten
        assert "wait_for_slew_complete 20\n" in rewritten  # no library -> no guard

    # post_start gets everything a task hook gets, plus the xdotool guard.
    guarded = _guard_post_start_body(
        "set -euo pipefail\n"
        'WID=$(xdotool search --name "GMAT" 2>/dev/null | head -1)\n'
        "wait_for_slew_complete 20\n"
        'N=$(grep -c PATTERN /tmp/f || echo "0")\n',
        ("wait_for_slew_complete",),
    )
    assert "|| true )" in guarded
    assert "wait_for_slew_complete 20 || true" in guarded
    assert "awk 'NR==1'" in guarded


def test_openemr_native_schema_sets_default_for_pc_multiple():
    """OpenEMR native-LAMP migration must keep upstream appointment setup SQL valid.

    Several pinned setup hooks insert into ``openemr_postcalendar_events`` without
    the ``pc_multiple`` column. Upstream's compose image tolerates that; our native
    MariaDB path must set the same effective default rather than rewriting task SQL.
    """
    script = (_cuaworld_root() / "scripts/install.sh").read_text()

    assert "MODIFY pc_multiple int(11) NOT NULL DEFAULT 0" in script
    assert "ALTER TABLE openemr_postcalendar_events" in script
    assert "ALTER TABLE openemr_postcalendar_events\n  MODIFY" in script
    assert (
        "ALTER TABLE openemr_postcalendar_events\n"
        "  MODIFY pc_multiple int(11) NOT NULL DEFAULT 0;\n"
        "SQL\npc_multiple_default="
    ) in script
    assert "OpenEMR pc_multiple default normalization failed" in script
    assert "INSERT INTO product_registration (email, opt_out)" in script
    assert "CUA-Lite: suppress OpenEMR Product Registration modal" in script
    assert "OpenEMR setup normalization anchor missing" in script


def test_astroimagej_updater_modal_is_suppressed_before_first_frame():
    """The prefs key upstream writes is not the key the AstroImageJ updater reads.

    The image build must patch the generated setup script, not task materials, so
    first screenshots are not covered by the updater modal.
    """
    script = (_cuaworld_root() / "scripts/install.sh").read_text()

    assert "CUA-Lite: suppress AstroImageJ updater preferences" in script
    assert ".aij.update.updateCheckOnStartup=false" in script
    assert "CUA-Lite: close AstroImageJ updater dialog" in script
    assert 'xdotool search --name "AstroImageJ Updater"' in script
    assert "CUA-Lite: wrap /usr/local/bin/aij with updater suppression" in script
    assert "rm -f /usr/local/bin/aij" in script
    assert "aij.cua-lite-real" in script
    assert "CUA-Lite: task_utils close AstroImageJ updater" in script
    assert (
        "cua_lite_close_aij_updater "
        ">/tmp/aij_updater_suppression_${USER}.log 2>&1 || true"
    ) in script
    assert "CUA_LITE_AIJ_UPDATER_MIN_POLLS" in script
    assert "CUA_LITE_AIJ_UPDATER_QUIET_POLLS" in script
    assert 'grep -qi "AstroImageJ"' not in script
    assert 'stable=$((stable + 1))' not in script
    assert '[ "$stable" -ge 4 ]' not in script
    assert ") >/tmp/aij_updater_suppression_${USER}.log 2>&1 &" not in script
    assert "AstroImageJ prefs anchor missing" in script
    assert "AstroImageJ launch anchor missing" in script
    assert "AstroImageJ wrapper anchor missing" in script
    assert "AstroImageJ task_utils anchor missing" in script

    adapter = (_cuaworld_root() / "src/adapter.py").read_text()
    assert "_ASTROIMAGEJ_UPDATER_CLEANUP_COMMAND" in adapter
    assert "_cleanup_astroimagej_updater_after_setup" in adapter
    assert "env_id.startswith(\"astroimagej_env\")" in adapter
    assert "CUA_LITE_AIJ_ADAPTER_MIN_POLLS" in adapter
    assert "run_command(\n            _ASTROIMAGEJ_UPDATER_CLEANUP_COMMAND" in adapter


def test_post_task_settle_matches_upstream_contract(monkeypatch):
    from lite.gym.envs.lite.cuaworld.src.adapter import _post_task_settle_seconds

    task_spec = {"hooks": {"post_task": "/workspace/tasks/x/export_result.sh"}}
    assert _post_task_settle_seconds(task_spec) == 15.0
    monkeypatch.setenv("GYM_ANYTHING_POST_TASK_SETTLE_SEC", "0.25")
    assert _post_task_settle_seconds(task_spec) == 0.25
    assert _post_task_settle_seconds({"hooks": {}}) == 0.0
