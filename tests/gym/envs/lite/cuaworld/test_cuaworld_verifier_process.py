"""CUAWorld tests split from _cuaworld_support.py: verifier process."""
from __future__ import annotations

import asyncio
import multiprocessing
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lite.gym.envs.lite.cuaworld.src import software
from lite.gym.envs.lite.cuaworld.src.adapter import (
    _call_in_process_before_deadline,
    _final_reward,
    _write_bytes_before_deadline,
    run_cuaworld_verify,
)
from lite.gym.errors import CuaWorldVerifierError
from tests.gym.envs.lite._cuaworld_support import (
    _delayed_marker,
    _FakeInterface,
    _partial_request_worker,
    _png_bytes,
    _slow_read_regular_text,
    _slow_validated_image_mime,
    _slow_write_bytes,
    _verifier_task,
    _verify_error,
)


def test_verifier_timeout_covers_provider_retries(monkeypatch):
    from lite.gym.envs.lite.cuaworld.src.adapter import _resolved_verifier_timeout

    monkeypatch.delenv("LITE_CUAWORLD_VERIFIER_TIMEOUT", raising=False)
    monkeypatch.setenv("VLM_TIMEOUT", "400")
    monkeypatch.setenv("VLM_MAX_RETRIES", "3")
    assert _resolved_verifier_timeout(None) == 1233.0

    monkeypatch.setenv("LITE_CUAWORLD_VERIFIER_TIMEOUT", "77")
    assert _resolved_verifier_timeout(None) == 77.0


def test_verifier_budget_is_clamped_below_step_timeout(monkeypatch):
    monkeypatch.delenv("LITE_CUAWORLD_VERIFIER_TIMEOUT", raising=False)
    monkeypatch.setenv("VLM_TIMEOUT", "400")
    monkeypatch.setenv("VLM_MAX_RETRIES", "3")

    verifier_timeout, preparation_timeout = software._verification_budgets_for_step(
        1080.0
    )

    assert preparation_timeout == 180.0
    assert verifier_timeout == 870.0
    assert verifier_timeout + preparation_timeout < 1080.0


@pytest.mark.asyncio
async def test_verifiers_are_process_isolated_under_concurrency(tmp_path):
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    for root, score in ((left_root, "10"), (right_root, "90")):
        (root / "utils").mkdir(parents=True)
        (root / "utils" / "helper.py").write_text(f"SCORE = {score}\n")
    left = _verifier_task(left_root, "from helper import SCORE; return {'score': SCORE}")
    right = _verifier_task(right_root, "from helper import SCORE; return {'score': SCORE}")
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")

    results = await asyncio.gather(
        run_cuaworld_verify(computer, left, "verifier.py::verify"),
        run_cuaworld_verify(computer, right, "verifier.py::verify"),
    )

    assert [result[0] for result in results] == [0.1, 0.9]


@pytest.mark.asyncio
async def test_verifier_process_bridges_container_io(tmp_path):
    destination = tmp_path / "copied-value"
    task = _verifier_task(
        tmp_path / "bridge",
        f"path = {str(destination)!r}; "
        # `copied` is the host path, NOT None: two pinned verifiers truthiness-test
        # this return (sumo/evaluate_phased_evacuation, astroimagej/
        # extract_linear_shock_profile), and returning None made sumo's
        # get_env_file unable to ever produce a file — a guaranteed reward 0.
        "copied = env_info['copy_from_env']('/tmp/value', path); "
        "out = env_info['exec_capture']('echo captured'); "
        "return {'score': 100 if copied == path "
        "and open(path, 'rb').read() == b'payload' "
        "and out == 'captured' else 0}",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")
    reward, _ = await run_cuaworld_verify(
        computer, task, "verifier.py::verify"
    )
    assert reward == 1.0


@pytest.mark.asyncio
async def test_verifier_copy_bridge_serves_upstream_one_argument_contract(
    tmp_path,
):
    """The one-argument form must work and RETURN a readable host path.

    This test previously asserted the opposite — that `copy_from_env(src)` raises
    TypeError and scores 0 — on the premise that upstream's contract is
    two-argument. The locked materials say otherwise. An AST sweep of every `*.py`
    under the 3216 on-disk task dirs finds **67 one-argument call sites in 67 tasks,
    all sweet_home_3d** (all 67 are registered AND non-excluded, so the count is the
    same on every denominator). A previous revision of this docstring claimed "91
    call sites across 84 tasks (sweet_home_3d alone has 67)"; no basis reproduces
    that — 67/67 is the measured value. Every one of them crashed the verifier
    before it could score, so the agent's work was thrown away.
    """
    task = _verifier_task(
        tmp_path / "one-argument-copy",
        "p = env_info['copy_from_env']('/tmp/value'); "
        "return {'score': 100 if open(p, 'rb').read() == b'payload' else 0}",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")

    reward, info = await run_cuaworld_verify(
        computer, task, "verifier.py::verify"
    )

    assert reward == 1.0
    assert not info.get("error")


@pytest.mark.asyncio
async def test_verifier_can_import_upstream_top_level_vlm_utils(tmp_path):
    task = _verifier_task(
        tmp_path / "vlm-utils",
        "from vlm_utils import sample_trajectory_frames; "
        "frames = sample_trajectory_frames(traj, n=1); "
        "return {'score': 100 if len(frames) == 1 else 0}",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")

    reward, _ = await run_cuaworld_verify(
        computer, task, "verifier.py::verify"
    )

    assert reward == 1.0


@pytest.mark.asyncio
async def test_verifier_can_use_gym_anything_vlm_n_keyword(tmp_path):
    task = _verifier_task(
        tmp_path / "gym-anything-vlm",
        "from gym_anything.vlm import sample_trajectory_frames; "
        "frames = sample_trajectory_frames(traj, n=1); "
        "return {'score': 100 if len(frames) == 1 else 0}",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")

    reward, _ = await run_cuaworld_verify(
        computer, task, "verifier.py::verify"
    )

    assert reward == 1.0


@pytest.mark.asyncio
async def test_verifier_exec_capture_combines_stdout_and_stderr(tmp_path):
    class CombinedInterface(_FakeInterface):
        async def run_command(self, command, timeout=None):
            return SimpleNamespace(returncode=1, stdout="out", stderr="err")

    task = _verifier_task(
        tmp_path / "combined-output",
        "out = env_info['exec_capture']('failing command'); "
        "return {'score': 100 if out == 'outerr' else 0}",
    )
    computer = SimpleNamespace(
        interface=CombinedInterface(),
        container_name="fake",
    )

    reward, _ = await run_cuaworld_verify(
        computer, task, "verifier.py::verify"
    )

    assert reward == 1.0


@pytest.mark.parametrize(
    ("reward_type", "expected"),
    [
        ("sparse", 1.0),
        ("partial", 75.0),
        ("rubric", 75.0),
        ("continuous", 0.75),
        # `dense`/`weighted` have no reward_shaping in the locked materials. A PASS
        # is worth 1.0 like sparse: every registered non-excluded dense verifier
        # passes below 100 (thresholds 17-75), so `score/100` would pay 0.17-0.75
        # for a task the verifier certifies as solved. The FAILING side keeps
        # score/100 — see test_final_reward_failure_keeps_partial_credit.
        ("dense", 1.0),
    ],
)
def test_final_reward_matches_upstream_reward_type(reward_type, expected):
    reward, actual_type = _final_reward(
        {"init": {"reward_type": reward_type}},
        raw_score=75,
        passed=True,
    )

    assert reward == expected
    assert actual_type == reward_type


@pytest.mark.parametrize("reward_type", ["dense", "weighted", "unknown_future_type"])
def test_final_reward_failure_keeps_partial_credit(reward_type):
    """The `score/100` reading survives on the FAILING side only, as shaped credit.
    Without this, the `dense` row above would be indistinguishable from `sparse`."""
    assert _final_reward(
        {"init": {"reward_type": reward_type}}, raw_score=75, passed=False
    ) == (0.75, reward_type)


@pytest.mark.asyncio
async def test_verifier_receives_official_task_and_env_ids(tmp_path):
    task = _verifier_task(
        tmp_path / "official-ids",
        "ok = (task_info['task_id'] == 'task@1' "
        "and env_info['env_id'] == 'upstream_env@0.1' "
        "and env_info['lite_env_id'] == 'lite.cuaworld.demo'); "
        "return {'score': 100 if ok else 0, 'passed': ok}",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")

    reward, info = await run_cuaworld_verify(
        computer,
        task,
        "verifier.py::verify",
        task_spec={
            "id": "task@1",
            "env_id": "upstream_env@0.1",
            "init": {"reward_type": "sparse"},
        },
        env_id="upstream_env@0.1",
        lite_env_id="lite.cuaworld.demo",
    )

    assert reward == 1.0
    assert info["passed"] is True


@pytest.mark.asyncio
async def test_verifier_concurrent_rpc_calls_are_serialized(tmp_path):
    task = _verifier_task(
        tmp_path / "concurrent-rpc",
        "import concurrent.futures\n"
        "def call(index):\n"
        "    return env_info['exec_capture'](f'echo {index}')\n"
        "with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:\n"
        "    outputs = list(pool.map(call, range(32)))\n"
        "return {'score': 100 if outputs == ['captured'] * 32 else 0}",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")

    reward, info = await run_cuaworld_verify(
        computer, task, "verifier.py::verify", timeout=5.0
    )

    assert reward == 1.0, info


@pytest.mark.asyncio
async def test_verifier_timeout_terminates_host_code(tmp_path):
    marker = tmp_path / "late-write"
    task = _verifier_task(
        tmp_path / "timeout",
        f"import time; time.sleep(1); open({str(marker)!r}, 'w').write('late'); "
        "return {'score': 100}",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")

    err = await _verify_error(
        computer, task, "verifier.py::verify", timeout=0.1
    )
    await asyncio.sleep(1.1)

    assert err.kind == "timeout"
    assert not marker.exists()


@pytest.mark.asyncio
async def test_verifier_timeout_terminates_spawned_process_tree(tmp_path):
    marker = tmp_path / "descendant-write"
    child = (
        "import time; time.sleep(2); "
        f"open({str(marker)!r}, 'w').write('late')"
    )
    task = _verifier_task(
        tmp_path / "process-tree",
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        "time.sleep(5); return {'score': 100}",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")

    err = await _verify_error(
        computer, task, "verifier.py::verify", timeout=0.2
    )
    await asyncio.sleep(2.2)

    assert err.kind == "timeout"
    assert not marker.exists()


@pytest.mark.asyncio
async def test_verifier_timeout_terminates_detached_process_tree(tmp_path):
    marker = tmp_path / "detached-descendant-write"
    child = (
        "import os, time; os.setsid(); time.sleep(2); "
        f"open({str(marker)!r}, 'w').write('late')"
    )
    task = _verifier_task(
        tmp_path / "detached-process-tree",
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        "time.sleep(5); return {'score': 100}",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")

    err = await _verify_error(
        computer, task, "verifier.py::verify", timeout=0.2
    )
    await asyncio.sleep(2.2)

    assert err.kind == "timeout"
    assert not marker.exists()


@pytest.mark.asyncio
async def test_completed_verifier_does_not_leave_spawned_descendants(tmp_path):
    marker = tmp_path / "completed-descendant-write"
    child = (
        "import time; time.sleep(2); "
        f"open({str(marker)!r}, 'w').write('late')"
    )
    task = _verifier_task(
        tmp_path / "completed-process-tree",
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        "return {'score': 100}",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")

    reward, _ = await run_cuaworld_verify(
        computer, task, "verifier.py::verify"
    )
    await asyncio.sleep(2.2)

    assert reward == 1.0
    assert not marker.exists()


@pytest.mark.asyncio
async def test_completed_verifier_does_not_leave_detached_descendants(tmp_path):
    marker = tmp_path / "completed-detached-descendant-write"
    child = (
        "import os, time; os.setsid(); time.sleep(2); "
        f"open({str(marker)!r}, 'w').write('late')"
    )
    task = _verifier_task(
        tmp_path / "completed-detached-process-tree",
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        "return {'score': 100}",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")

    reward, _ = await run_cuaworld_verify(
        computer, task, "verifier.py::verify"
    )
    await asyncio.sleep(2.2)

    assert reward == 1.0
    assert not marker.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [None, 0.4])
async def test_verifier_terminates_detached_child_spawned_by_thread(
    tmp_path, timeout
):
    marker = tmp_path / f"thread-descendant-{timeout}"
    child = (
        "import os, time; os.setsid(); time.sleep(2); "
        f"open({str(marker)!r}, 'w').write('late')"
    )
    finish = (
        "time.sleep(5); return {'score': 100}"
        if timeout is not None
        else "time.sleep(0.3); return {'score': 100}"
    )
    task = _verifier_task(
        tmp_path / f"thread-process-tree-{timeout}",
        "import subprocess, sys, threading, time\n"
        "def launch():\n"
        f"    subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        "    time.sleep(5)\n"
        "threading.Thread(target=launch, daemon=True).start()\n"
        f"{finish}",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")

    # Both arms must reap the detached child; only the outcome differs. With a budget
    # the verifier never finishes, so there is no score and it raises; without one it
    # completes and scores normally.
    if timeout is None:
        reward, _ = await run_cuaworld_verify(computer, task, "verifier.py::verify")
        assert reward == 1.0
    else:
        err = await _verify_error(
            computer, task, "verifier.py::verify", timeout=timeout
        )
        assert err.kind == "timeout"
    await asyncio.sleep(2.2)

    assert not marker.exists()


@pytest.mark.asyncio
async def test_completed_verifier_cannot_spawn_after_result_cleanup(tmp_path):
    marker = tmp_path / "post-result-descendant"
    child = (
        "import time; time.sleep(0.4); "
        f"open({str(marker)!r}, 'w').write('late')"
    )
    task = _verifier_task(
        tmp_path / "post-result-race",
        "import subprocess, sys, threading, time\n"
        "def launch():\n"
        "    time.sleep(0.01)\n"
        f"    subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        "threading.Thread(target=launch).start()\n"
        "return {'score': 100}",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")

    reward, _ = await run_cuaworld_verify(
        computer, task, "verifier.py::verify"
    )
    await asyncio.sleep(0.7)

    assert reward == 1.0
    assert not marker.exists()


@pytest.mark.asyncio
async def test_tiny_deadlines_do_not_stall_event_loop(tmp_path):
    task = _verifier_task(
        tmp_path / "tiny-deadline", "return {'score': 100}"
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")
    start = asyncio.get_running_loop().time()

    results = await asyncio.gather(
        *(
            run_cuaworld_verify(
                computer, task, "verifier.py::verify", timeout=0.001
            )
            for _ in range(32)
        ),
        return_exceptions=True,
    )

    # Each one RAISES rather than scoring 0: a 1 ms budget means the verifier never
    # ran, so there is no reward to report. What this test is really about is that
    # 32 of them do it without blocking the loop. Keep the wall-clock threshold
    # loose enough for xdist workers on loaded hosts; a synchronous/blocking
    # verifier-spawn regression takes seconds here, not a few hundred ms.
    assert all(isinstance(r, CuaWorldVerifierError) for r in results)
    assert {r.kind for r in results} == {"timeout"}
    assert asyncio.get_running_loop().time() - start < 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_slow_process_start_is_bounded_and_cleaned(
    tmp_path, monkeypatch, cancel
):
    from multiprocessing import popen_spawn_posix

    original_init = popen_spawn_posix.Popen.__init__

    def slow_init(self, *args, **kwargs):
        time.sleep(0.2)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(popen_spawn_posix.Popen, "__init__", slow_init)
    task_dir = _verifier_task(
        tmp_path / f"slow-start-{cancel}", "return {'score': 100}"
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")
    baseline = {child.pid for child in multiprocessing.active_children()}
    start = asyncio.get_running_loop().time()
    task = asyncio.create_task(
        run_cuaworld_verify(
            computer,
            task_dir,
            "verifier.py::verify",
            timeout=0.5 if cancel else 0.06,
        )
    )
    if cancel:
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(CuaWorldVerifierError) as excinfo:
            await task
        assert excinfo.value.kind == "timeout"
    elapsed = asyncio.get_running_loop().time() - start
    await asyncio.sleep(0.6)

    leaked = {
        child.pid
        for child in multiprocessing.active_children()
        if child.pid not in baseline
    }
    assert elapsed < 0.15
    assert leaked == set()


def test_process_start_cleanup_survives_event_loop_close(
    tmp_path, monkeypatch
):
    from multiprocessing import popen_spawn_posix

    original_init = popen_spawn_posix.Popen.__init__

    def delayed_return(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        time.sleep(0.5)

    monkeypatch.setattr(popen_spawn_posix.Popen, "__init__", delayed_return)
    marker = tmp_path / "after-loop-close"
    baseline = {child.pid for child in multiprocessing.active_children()}

    async def run():
        with pytest.raises(TimeoutError):
            await _call_in_process_before_deadline(
                _delayed_marker,
                (marker, 0.2),
                time.monotonic() + 0.06,
            )

    asyncio.run(run())
    time.sleep(0.8)

    leaked = {
        child.pid
        for child in multiprocessing.active_children()
        if child.pid not in baseline
    }
    assert not marker.exists()
    assert leaked == set()


@pytest.mark.asyncio
async def test_verifier_rpc_respects_overall_deadline(tmp_path):
    class SlowInterface(_FakeInterface):
        async def run_command(self, command, timeout=None):
            await asyncio.sleep(5)
            return await super().run_command(command, timeout)

    task = _verifier_task(
        tmp_path / "slow-rpc",
        "env_info['exec_capture']('sleep'); return {'score': 100}",
    )
    computer = SimpleNamespace(interface=SlowInterface(), container_name="fake")
    start = asyncio.get_running_loop().time()
    err = await _verify_error(
        computer, task, "verifier.py::verify", timeout=1.0
    )
    elapsed = asyncio.get_running_loop().time() - start
    assert err.kind == "timeout"
    assert elapsed < 3.0


@pytest.mark.asyncio
async def test_screenshot_preparation_timeout_raises_instead_of_scoring(
    tmp_path,
):
    class SlowScreenshot(_FakeInterface):
        async def screenshot(self):
            await asyncio.sleep(5)
            return _png_bytes()

    task = _verifier_task(
        tmp_path / "slow-screenshot", "return {'score': 100}"
    )
    computer = SimpleNamespace(interface=SlowScreenshot(), container_name="fake")
    start = asyncio.get_running_loop().time()

    err = await _verify_error(
        computer, task, "verifier.py::verify", timeout=1.0
    )

    assert err.phase == "screenshot"
    assert err.kind == "timeout"
    assert asyncio.get_running_loop().time() - start < 3.0


@pytest.mark.asyncio
async def test_export_timeout_raises_instead_of_scoring(tmp_path):
    class SlowExport(_FakeInterface):
        async def write_bytes(self, _path, _data):
            await asyncio.sleep(5)

    task = _verifier_task(
        tmp_path / "slow-export",
        "return {'score': 100}",
        export=True,
    )
    computer = SimpleNamespace(interface=SlowExport(), container_name="fake")
    start = asyncio.get_running_loop().time()

    err = await _verify_error(
        computer, task, "verifier.py::verify", timeout=1.0
    )

    assert err.phase == "post_task"
    assert err.kind == "timeout"
    assert asyncio.get_running_loop().time() - start < 3.0


@pytest.mark.asyncio
async def test_export_read_respects_overall_deadline(
    tmp_path, monkeypatch
):
    from lite.gym.envs.lite.cuaworld.src import adapter

    task = _verifier_task(
        tmp_path / "slow-export-read",
        "return {'score': 100}",
        export=True,
    )
    monkeypatch.setattr(
        adapter, "_read_optional_regular_text", _slow_read_regular_text
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")
    start = asyncio.get_running_loop().time()

    err = await _verify_error(
        computer, task, "verifier.py::verify", timeout=1.0
    )

    assert err.kind == "timeout"
    assert asyncio.get_running_loop().time() - start < 3.0


@pytest.mark.asyncio
async def test_export_fifo_raises_without_blocking(tmp_path):
    task = _verifier_task(
        tmp_path / "fifo-export",
        "return {'score': 100}",
    )
    os.mkfifo(task / "export_result.sh")
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")
    start = asyncio.get_running_loop().time()

    err = await _verify_error(
        computer, task, "verifier.py::verify", timeout=1.0
    )

    assert "regular file" in str(err)
    # The point of the test is the second assertion: opening a FIFO must not block
    # until the budget expires. It refuses immediately, and now says so by raising.
    assert asyncio.get_running_loop().time() - start < 0.5


def test_cuaworld_process_helper_threads_do_not_block_interpreter_exit():
    from lite.gym.envs.lite.cuaworld.src import adapter

    source = Path(adapter.__file__).read_text()

    assert 'name="cuaworld-process-start",\n        daemon=True' in source
    assert 'name="cuaworld-process-cleanup",\n        daemon=True' in source
    assert 'name="cuaworld-deadline-call",\n        daemon=True' in source


@pytest.mark.asyncio
async def test_screenshot_decode_timeout_raises_instead_of_scoring(
    tmp_path, monkeypatch
):
    from lite.gym.envs.lite.cuaworld.src import vlm

    monkeypatch.setattr(
        vlm, "validated_image_mime", _slow_validated_image_mime
    )
    task = _verifier_task(
        tmp_path / "slow-screenshot-decode", "return {'score': 100}"
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")
    start = asyncio.get_running_loop().time()

    err = await _verify_error(
        computer, task, "verifier.py::verify", timeout=1.0
    )

    assert err.phase == "screenshot"
    assert err.kind == "timeout"
    assert asyncio.get_running_loop().time() - start < 3.0


@pytest.mark.asyncio
async def test_partial_rpc_frame_respects_overall_deadline(
    tmp_path, monkeypatch
):
    from lite.gym.envs.lite.cuaworld.src import adapter

    monkeypatch.setattr(adapter, "_verifier_worker", _partial_request_worker)
    task = _verifier_task(
        tmp_path / "partial-rpc-frame", "return {'score': 100}"
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")
    start = asyncio.get_running_loop().time()

    err = await _verify_error(
        computer, task, "verifier.py::verify", timeout=1.0
    )

    assert err.kind == "timeout"
    assert asyncio.get_running_loop().time() - start < 3.0


@pytest.mark.asyncio
async def test_verifier_large_rpc_response_does_not_block_timeout_cleanup(tmp_path):
    class LargeInterface(_FakeInterface):
        async def read_bytes(self, _path):
            return b"x" * (32 * 1024 * 1024)

    task = _verifier_task(
        tmp_path / "large-rpc",
        f"env_info['copy_from_env']('/tmp/value', "
        f"{str(tmp_path / 'large-copy')!r}); "
        "import time; time.sleep(1); return {'score': 100}",
    )
    computer = SimpleNamespace(interface=LargeInterface(), container_name="fake")

    # The outer wait_for is the real assertion: a large in-flight RPC response must not
    # keep the verifier's own 0.08s budget from firing. It fires by raising now.
    with pytest.raises(CuaWorldVerifierError) as excinfo:
        await asyncio.wait_for(
            run_cuaworld_verify(
                computer, task, "verifier.py::verify", timeout=0.08
            ),
            timeout=1.0,
        )

    assert excinfo.value.kind == "timeout"


@pytest.mark.asyncio
async def test_slow_rpc_host_write_respects_deadline(tmp_path, monkeypatch):
    from lite.gym.envs.lite.cuaworld.src import adapter

    monkeypatch.setattr(
        adapter, "_write_bytes_before_deadline", _slow_write_bytes
    )
    task = _verifier_task(
        tmp_path / "slow-host-write",
        f"env_info['copy_from_env']('/tmp/value', "
        f"{str(tmp_path / 'slow-copy')!r}); "
        "return {'score': 100}",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")
    start = asyncio.get_running_loop().time()

    err = await _verify_error(
        computer, task, "verifier.py::verify", timeout=0.3
    )

    assert err.kind == "timeout"
    assert asyncio.get_running_loop().time() - start < 0.8


@pytest.mark.asyncio
async def test_verifier_abrupt_exit_raises(tmp_path):
    task = _verifier_task(
        tmp_path / "abrupt-exit",
        "import os; os._exit(17)",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")

    with pytest.raises(CuaWorldVerifierError) as excinfo:
        await run_cuaworld_verify(computer, task, "verifier.py::verify")

    assert excinfo.value.kind == "verifier_raised"
    assert "pipe closed" in str(excinfo.value) or "code 17" in str(excinfo.value)


@pytest.mark.asyncio
async def test_missing_verifier_score_uses_passed_fallback(tmp_path):
    task = _verifier_task(
        tmp_path / "missing-score-fallback",
        "return {'passed': True}",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")

    reward, info = await run_cuaworld_verify(
        computer, task, "verifier.py::verify"
    )

    assert reward == 1.0
    assert info["raw_score"] == 100.0


@pytest.mark.asyncio
async def test_verifier_target_path_is_used(tmp_path):
    task = _verifier_task(
        tmp_path / "alternate-verifier",
        "return {'score': 0}",
    )
    checks = task / "checks"
    checks.mkdir()
    (checks / "alt_verifier.py").write_text(
        "def verify(traj, env_info, task_info):\n"
        "    return {'score': 100}\n"
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")

    reward, info = await run_cuaworld_verify(
        computer, task, "checks/alt_verifier.py::verify"
    )

    assert reward == 1.0
    assert info["raw_score"] == 100.0


@pytest.mark.parametrize("score_expression", ["None", "'not-a-number'"])
@pytest.mark.asyncio
async def test_explicit_nonnumeric_verifier_score_raises(
    tmp_path, score_expression
):
    task = _verifier_task(
        tmp_path / f"bad-score-{score_expression!r}",
        f"return {{'passed': True, 'score': {score_expression}}}",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")

    with pytest.raises(CuaWorldVerifierError) as excinfo:
        await run_cuaworld_verify(computer, task, "verifier.py::verify")

    assert excinfo.value.kind == "bad_result"
    assert "nonnumeric score" in str(excinfo.value)


@pytest.mark.asyncio
async def test_verifier_exit_during_rpc_raises(tmp_path):
    class SlowInterface(_FakeInterface):
        async def run_command(self, command, timeout=None):
            await asyncio.sleep(0.2)
            return await super().run_command(command, timeout)

    task = _verifier_task(
        tmp_path / "exit-during-rpc",
        "import os, threading; "
        "threading.Timer(0.05, lambda: os._exit(17)).start(); "
        "env_info['exec_capture']('slow'); return {'score': 100}",
    )
    computer = SimpleNamespace(interface=SlowInterface(), container_name="fake")

    with pytest.raises(CuaWorldVerifierError) as excinfo:
        await run_cuaworld_verify(computer, task, "verifier.py::verify")

    assert excinfo.value.kind == "verifier_raised"
    assert "pipe closed" in str(excinfo.value) or "code 17" in str(excinfo.value)


def test_expired_rpc_write_removes_partial_file(tmp_path):
    destination = tmp_path / "late"
    with pytest.raises(TimeoutError):
        _write_bytes_before_deadline(destination, b"payload", 0)
    assert not destination.exists()


def test_rpc_write_rejects_fifo_without_blocking(tmp_path):
    destination = tmp_path / "fifo"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.touch()
    destination.unlink()
    import os
    os.mkfifo(destination)

    start = time.monotonic()
    with pytest.raises(ValueError, match="not a regular file"):
        _write_bytes_before_deadline(
            destination, b"payload", time.monotonic() + 1
        )
    assert time.monotonic() - start < 0.2
    assert destination.is_fifo()


@pytest.mark.asyncio
@pytest.mark.parametrize("score", ["float('nan')", "float('inf')"])
async def test_non_finite_verifier_scores_raise(tmp_path, score):
    task = _verifier_task(
        tmp_path / score.replace("'", "").replace("(", "").replace(")", ""),
        f"return {{'score': {score}}}",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")
    with pytest.raises(CuaWorldVerifierError) as excinfo:
        await run_cuaworld_verify(computer, task, "verifier.py::verify")
    assert excinfo.value.kind == "bad_result"
    assert "finite" in str(excinfo.value)


@pytest.mark.asyncio
async def test_explicit_unparseable_failed_score_raises(tmp_path):
    task = _verifier_task(
        tmp_path / "bad-score",
        "return {'passed': False, 'score': 'bad'}",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")

    with pytest.raises(CuaWorldVerifierError) as excinfo:
        await run_cuaworld_verify(computer, task, "verifier.py::verify")

    assert excinfo.value.kind == "bad_result"
    assert "nonnumeric score" in str(excinfo.value)


@pytest.mark.asyncio
async def test_export_hook_failure_raises_instead_of_scoring(tmp_path):
    task = _verifier_task(
        tmp_path / "export", "return {'score': 100}", export=True
    )
    computer = SimpleNamespace(
        interface=_FakeInterface(export_rc=17), container_name="fake"
    )
    err = await _verify_error(
        computer, task, "verifier.py::verify"
    )
    assert err.phase == "post_task"
    assert err.kind == "command_failed"
    assert "export failed" in str(err)


@pytest.mark.asyncio
async def test_verifier_baseexception_raises_with_original_message(tmp_path):
    task = _verifier_task(
        tmp_path / "infra-baseexception",
        "class JudgeDown(BaseException): pass\nraise JudgeDown('judge timeout')",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")
    err = await _verify_error(
        computer, task, "verifier.py::verify"
    )
    assert err.phase == "verify"
    assert err.kind == "verifier_raised"
    assert "JudgeDown: judge timeout" in str(err)


@pytest.mark.asyncio
async def test_verifier_bare_except_cannot_hide_vlm_provider_failure(tmp_path):
    task = _verifier_task(
        tmp_path / "bare-except-vlm",
        "try:\n"
        "    env_info['query_vlm']('judge', images=['/tmp/does-not-exist.png'])\n"
        "except:\n"
        "    return {'score': 100}\n"
        "return {'score': 0}",
    )
    computer = SimpleNamespace(interface=_FakeInterface(), container_name="fake")

    err = await _verify_error(
        computer, task, "verifier.py::verify"
    )

    assert err.phase == "verify"
    assert err.kind == "verifier_raised"
    assert "VLMProviderError" in str(err)
    assert "no valid VLM images" in str(err)


@pytest.mark.asyncio
async def test_export_hook_transport_failure_raises_instead_of_scoring(
    tmp_path,
):
    class BrokenInterface(_FakeInterface):
        async def write_bytes(self, _path, _data):
            raise ConnectionError("transport down")

    task = _verifier_task(
        tmp_path / "export-transport", "return {'score': 100}", export=True
    )
    computer = SimpleNamespace(interface=BrokenInterface(), container_name="fake")
    err = await _verify_error(
        computer, task, "verifier.py::verify"
    )
    assert err.phase == "post_task"
    assert err.kind == "spawn"
    assert "transport down" in str(err)


@pytest.mark.asyncio
async def test_verification_screenshot_failure_raises_instead_of_scoring(
    tmp_path,
):
    class BrokenScreenshotInterface(_FakeInterface):
        async def screenshot(self):
            raise ConnectionError("display down")

    task = _verifier_task(
        tmp_path / "broken-screenshot", "return {'score': 100}"
    )
    computer = SimpleNamespace(
        interface=BrokenScreenshotInterface(), container_name="fake"
    )

    err = await _verify_error(
        computer, task, "verifier.py::verify"
    )
    assert err.phase == "screenshot"
    assert err.kind == "screenshot_failed"
    assert "display down" in str(err)
