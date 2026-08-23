"""Unit + integration tests for the admission module.

Covers:
  * L1 emergency sensor checks (RAM percent, RAM absolute free, load).
  * L2 in-flight cap with admit/release accounting.
  * Plugin hook (``register_emergency_check``).
  * Cluster auto-derivation (``derive_max_live_envs``).
  * Docker create semaphore (sync + async variants).

Run::

    uv run pytest tests/gym/remote/test_admission.py -xvs
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from unittest.mock import patch

import pytest
from gym.remote.conftest import (
    permissive_admission_config as _permissive_config,
)

import lite.gym.remote.admission as admission_mod
from lite.gym.errors import CapacityExhausted
from lite.gym.remote.admission import (
    EMERGENCY_RETRY_AFTER_S,
    AdmissionConfig,
    AdmissionGate,
    _reset_emergency_checks_for_tests,
    derive_max_live_envs,
    docker_create_slot,
    docker_create_slot_async,
    register_emergency_check,
)
from lite.gym.utils.server.capacity import HostCapacity


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Reset module-level state (docker sema + emergency checks) between
    tests so AdmissionGate construction with different
    docker_create_concurrency values doesn't trip the idempotency
    guard.
    """
    admission_mod._docker_sema = None
    admission_mod._docker_sema_capacity = 0
    admission_mod._docker_sema_timeout_total = 0
    _reset_emergency_checks_for_tests()
    yield
    admission_mod._docker_sema = None
    admission_mod._docker_sema_capacity = 0
    admission_mod._docker_sema_timeout_total = 0
    _reset_emergency_checks_for_tests()


# ---------------------------------------------------------------------------
# Cluster derivation — auto-adapt to host vCPU
# ---------------------------------------------------------------------------


class TestDeriveMaxLiveEnvs:
    """Tabular per-cluster check.

    ``ram_gb`` is required in production (:func:`derive_admission_config`
    always passes host RAM). Tests below that mean to isolate the
    vCPU-bound math pass ``_AMPLE_RAM_GB`` so the RAM bound never binds;
    ``test_ram_bound_wins_when_lower`` covers the RAM-bound branch
    directly.
    """

    #: Large enough that ``ram_gb / per_env_ram_gb`` never becomes the
    #: limiting bound for any vcpu value exercised below.
    _AMPLE_RAM_GB = 1_000_000.0

    @pytest.mark.parametrize(
        "vcpu,expected",
        [
            (1, 16),  # floor — even tiny CI containers get >= 16
            (4, 16),  # floor — laptop
            (8, 32),  # linear region
            (32, 128),
            (96, 384),  # production small
            (384, 1536),  # production big
            (1024, 4096),  # ceiling — huge research box
            (5000, 4096),  # ceiling — pathological
        ],
    )
    def test_per_cluster_value(self, vcpu, expected):
        assert (
            derive_max_live_envs(
                vcpu=vcpu,
                ram_gb=self._AMPLE_RAM_GB,
                per_vcpu=4,
                floor=16,
                ceiling=4096,
            )
            == expected
        )

    def test_env_var_override_per_vcpu(self):
        with patch.dict(os.environ, {"CUA_LITE_MAX_LIVE_PER_VCPU": "2"}, clear=False):
            # vCPU 100 × 2 = 200 (not 400)
            assert derive_max_live_envs(vcpu=100, ram_gb=self._AMPLE_RAM_GB) == 200

    def test_env_var_override_floor(self):
        with patch.dict(os.environ, {"CUA_LITE_MAX_LIVE_FLOOR": "64"}, clear=False):
            assert derive_max_live_envs(vcpu=1, ram_gb=self._AMPLE_RAM_GB) == 64

    def test_env_var_override_ceiling(self):
        with patch.dict(os.environ, {"CUA_LITE_MAX_LIVE_CEILING": "256"}, clear=False):
            assert derive_max_live_envs(vcpu=1000, ram_gb=self._AMPLE_RAM_GB) == 256

    def test_zero_vcpu_falls_back_to_floor(self):
        # vcpu=0 would multiply to 0; floor wins.
        assert (
            derive_max_live_envs(
                vcpu=0,
                ram_gb=self._AMPLE_RAM_GB,
                per_vcpu=4,
                floor=16,
                ceiling=4096,
            )
            == 16
        )

    def test_ram_bound_wins_when_lower(self):
        # 32 vCPU * 4/vcpu = 128 vCPU-bound, but 64 GB / 4 GB-per-env = 16
        # RAM-bound — the RAM bound must win.
        assert (
            derive_max_live_envs(
                vcpu=32,
                ram_gb=64.0,
                per_vcpu=4,
                floor=1,
                ceiling=4096,
            )
            == 16
        )


class TestDeriveAdmissionConfig:
    def _host(self) -> HostCapacity:
        return HostCapacity(
            vcpu=32,
            ram_total_gb=128.0,
            disk_total_gb=1000.0,
            has_nonrotational_storage=True,
        )

    def test_env_var_escape_hatches_apply(self, monkeypatch):
        monkeypatch.setenv("CUA_LITE_EMERGENCY_RAM_PCT", "88.5")
        monkeypatch.setenv("CUA_LITE_EMERGENCY_RAM_FREE_MIN_GB", "12")
        monkeypatch.setenv("CUA_LITE_EMERGENCY_LOAD_PER_CPU", "3.25")
        monkeypatch.setenv("CUA_LITE_EMERGENCY_DISK_FREE_MIN_GB", "40")
        monkeypatch.setenv("CUA_LITE_EMERGENCY_SWAP_PCT", "5")
        monkeypatch.setenv("CUA_LITE_DOCKER_CREATE_CONCURRENCY", "17")

        cfg = admission_mod.derive_admission_config(self._host())

        assert cfg.emergency_ram_pct == 88.5
        assert cfg.emergency_ram_free_min_bytes == 12 * (1024**3)
        assert cfg.emergency_load_per_cpu == 3.25
        assert cfg.emergency_disk_free_min_bytes == 40 * (1024**3)
        assert cfg.emergency_swap_pct == 5.0
        assert cfg.docker_create_concurrency == 17

    def test_explicit_overrides_win_over_env_vars(self, monkeypatch):
        monkeypatch.setenv("CUA_LITE_DOCKER_CREATE_CONCURRENCY", "17")

        cfg = admission_mod.derive_admission_config(
            self._host(),
            docker_create_concurrency=4,
            max_live_envs=9,
        )

        assert cfg.docker_create_concurrency == 4
        assert cfg.max_live_envs == 9


# ---------------------------------------------------------------------------
# L1 sensors
# ---------------------------------------------------------------------------


class TestL1Sensors:
    """Direct check of built-in RAM + load sensors via monkey-patched psutil."""

    def setup_method(self):
        _reset_emergency_checks_for_tests()
        # ``_read_host_sensors`` caches RAM/load/disk readings for 0.5s.
        # Without busting it here, a previous test's mocked values (or a
        # real-load read from another pytest worker landing in the same
        # process) survive across tests and make the patch below a no-op,
        # producing flaky "DID NOT RAISE" failures under parallel runs.
        import lite.gym.remote.admission as _adm

        _adm._host_sensor_cache = None

    def test_ram_percent_above_threshold_fires(self):
        cfg = AdmissionConfig(
            max_live_envs=100,
            emergency_ram_pct=80.0,
            emergency_ram_free_min_bytes=0,  # disable absolute check
            emergency_load_per_cpu=9999.0,
            docker_create_concurrency=8,
        )
        gate = AdmissionGate(cfg)
        with patch("lite.gym.remote.admission.psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.percent = 85.0
            mock_vm.return_value.available = 999_000_000_000  # plenty free
            with pytest.raises(CapacityExhausted) as exc_info:
                gate.require_capacity()
            assert "85.0%" in exc_info.value.what
            assert "> 80%" in exc_info.value.what
            assert exc_info.value.retry_after_s == EMERGENCY_RETRY_AFTER_S

    def test_ram_absolute_free_below_threshold_fires(self):
        cfg = AdmissionConfig(
            max_live_envs=100,
            emergency_ram_pct=99.9,
            emergency_ram_free_min_bytes=2 * (1024**3),  # 2 GB
            emergency_load_per_cpu=9999.0,
            docker_create_concurrency=8,
        )
        gate = AdmissionGate(cfg)
        with patch("lite.gym.remote.admission.psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.percent = 50.0  # percent fine
            mock_vm.return_value.available = 1 * (1024**3)  # only 1 GB free
            with pytest.raises(CapacityExhausted) as exc_info:
                gate.require_capacity()
            assert "1.0 GB < 2.0 GB" in exc_info.value.what

    def test_load_per_cpu_above_threshold_fires(self):
        cfg = AdmissionConfig(
            max_live_envs=100,
            emergency_ram_pct=99.9,
            emergency_ram_free_min_bytes=0,
            emergency_load_per_cpu=1.0,
            docker_create_concurrency=8,
        )
        gate = AdmissionGate(cfg)
        # Load must exceed threshold × cpu_count → use a number larger
        # than any realistic vCPU count so the test is host-independent.
        cpu = os.cpu_count() or 1
        with patch("lite.gym.remote.admission.os.getloadavg") as mock_load:
            mock_load.return_value = (cpu * 3.0, 0.0, 0.0)
            with pytest.raises(CapacityExhausted) as exc_info:
                gate.require_capacity()
            assert "load" in exc_info.value.what.lower()

    def test_all_sensors_healthy_no_raise(self):
        gate = AdmissionGate(_permissive_config())
        gate.require_capacity()  # must NOT raise

    def test_emergency_counter_increments(self):
        cfg = AdmissionConfig(
            max_live_envs=100,
            emergency_ram_pct=0.0,  # always fires
            emergency_ram_free_min_bytes=0,
            emergency_load_per_cpu=9999.0,
            docker_create_concurrency=8,
        )
        gate = AdmissionGate(cfg)
        for _ in range(3):
            with pytest.raises(CapacityExhausted):
                gate.require_capacity()
        assert gate.snapshot()["emergency_503_total"] == 3


# ---------------------------------------------------------------------------
# L2 in-flight cap
# ---------------------------------------------------------------------------


class TestL2InFlightCap:
    def setup_method(self):
        _reset_emergency_checks_for_tests()

    def test_admit_increments_in_flight(self):
        gate = AdmissionGate(_permissive_config(max_live_envs=3))
        assert gate.in_flight == 0
        gate.require_capacity()
        gate.admit()
        assert gate.in_flight == 1

    def test_cap_reached_rejects_with_typed_503(self):
        gate = AdmissionGate(_permissive_config(max_live_envs=2))
        for _ in range(2):
            gate.require_capacity()
            gate.admit()
        with pytest.raises(CapacityExhausted) as exc_info:
            gate.require_capacity()
        assert "live env cap" in exc_info.value.what
        assert "2/2" in exc_info.value.what
        assert gate.snapshot()["capacity_503_total"] == 1

    def test_release_makes_room(self):
        gate = AdmissionGate(_permissive_config(max_live_envs=1))
        gate.require_capacity()
        gate.admit()
        with pytest.raises(CapacityExhausted):
            gate.require_capacity()
        gate.release()
        # Now a new admit succeeds.
        gate.require_capacity()
        gate.admit()
        assert gate.in_flight == 1

    def test_capacity_check_after_emergency_check(self):
        """L1 fires BEFORE L2 — RAM emergency 503 even if cap has room."""
        cfg = AdmissionConfig(
            max_live_envs=100,
            emergency_ram_pct=0.0,  # always fires
            emergency_ram_free_min_bytes=0,
            emergency_load_per_cpu=9999.0,
            docker_create_concurrency=8,
        )
        gate = AdmissionGate(cfg)
        with pytest.raises(CapacityExhausted):
            gate.require_capacity()
        assert gate.snapshot()["emergency_503_total"] == 1
        assert gate.snapshot()["capacity_503_total"] == 0


# ---------------------------------------------------------------------------
# L1 plugin extensibility
# ---------------------------------------------------------------------------


class TestPluginHook:
    def setup_method(self):
        _reset_emergency_checks_for_tests()

    def teardown_method(self):
        _reset_emergency_checks_for_tests()

    def test_plugin_check_fires(self):
        def my_check():
            raise CapacityExhausted("custom check failed", retry_after_s=5)

        register_emergency_check(my_check)
        gate = AdmissionGate(_permissive_config())
        with pytest.raises(CapacityExhausted) as exc_info:
            gate.require_capacity()
        assert "custom check failed" in exc_info.value.what
        assert exc_info.value.retry_after_s == 5

    def test_plugin_check_passes_no_effect(self):
        # A no-op plugin doesn't interfere with the normal happy path.
        register_emergency_check(lambda: None)
        gate = AdmissionGate(_permissive_config())
        gate.require_capacity()
        gate.admit()
        assert gate.in_flight == 1

    def test_multiple_plugins_evaluated_in_order(self):
        log = []

        def check1():
            log.append("check1")

        def check2():
            log.append("check2")
            raise CapacityExhausted("from check2", retry_after_s=1)

        register_emergency_check(check1)
        register_emergency_check(check2)
        gate = AdmissionGate(_permissive_config())
        with pytest.raises(CapacityExhausted) as exc_info:
            gate.require_capacity()
        # check1 ran (no exception), check2 ran and raised.
        assert log == ["check1", "check2"]
        assert "check2" in exc_info.value.what


# ---------------------------------------------------------------------------
# Docker semaphore
# ---------------------------------------------------------------------------


class TestDockerSema:
    def setup_method(self):
        # Reset module-level docker semaphore state for clean tests.
        admission_mod._docker_sema = None
        admission_mod._docker_sema_capacity = 0
        admission_mod._docker_sema_timeout_total = 0

    def teardown_method(self):
        admission_mod._docker_sema = None
        admission_mod._docker_sema_capacity = 0
        admission_mod._docker_sema_timeout_total = 0

    def test_sync_acquire_release(self):
        AdmissionGate(_permissive_config())  # configures sema
        with docker_create_slot():
            pass
        assert admission_mod._docker_in_flight() == 0

    def test_sync_timeout_raises_capacity_exhausted(self):
        AdmissionGate(
            AdmissionConfig(
                max_live_envs=100,
                emergency_ram_pct=99.9,
                emergency_ram_free_min_bytes=1,
                emergency_load_per_cpu=9999.0,
                docker_create_concurrency=1,
            )
        )
        # Hold the only slot; subsequent acquire should time out.
        acquired = admission_mod._docker_sema.acquire(blocking=False)
        assert acquired is True
        try:
            # Patch the timeout so the test doesn't take 5s.
            with patch.object(admission_mod, "DOCKER_CREATE_WAIT_TIMEOUT_S", 0.05):
                with pytest.raises(CapacityExhausted) as exc_info:
                    with docker_create_slot():
                        pass
            assert "docker daemon slot unavailable" in exc_info.value.what
            assert admission_mod._docker_sema_timeout_total == 1
        finally:
            admission_mod._docker_sema.release()

    def test_async_acquire_release(self):
        AdmissionGate(_permissive_config())

        async def go():
            async with docker_create_slot_async():
                pass

        asyncio.run(go())
        assert admission_mod._docker_in_flight() == 0

    def test_unconfigured_is_noop(self):
        # No AdmissionGate ever created → semaphore stays None → both
        # context managers yield immediately without serializing.
        assert admission_mod._docker_sema is None
        with docker_create_slot():
            pass

        async def go():
            async with docker_create_slot_async():
                pass

        asyncio.run(go())

    def test_double_configure_same_capacity_is_idempotent(self):
        AdmissionGate(_permissive_config())
        # Second gate with the same docker concurrency should NOT raise.
        AdmissionGate(_permissive_config())

    def test_double_configure_different_capacity_raises(self):
        cfg1 = _permissive_config()
        AdmissionGate(cfg1)
        cfg2 = AdmissionConfig(
            max_live_envs=100,
            emergency_ram_pct=99.9,
            emergency_ram_free_min_bytes=1,
            emergency_load_per_cpu=9999.0,
            docker_create_concurrency=cfg1.docker_create_concurrency + 1,
        )
        with pytest.raises(RuntimeError, match="already configured"):
            AdmissionGate(cfg2)

    def test_sync_and_async_share_resource(self):
        """Sync acquire blocks async acquire via the same threading.Semaphore."""
        AdmissionGate(
            AdmissionConfig(
                max_live_envs=100,
                emergency_ram_pct=99.9,
                emergency_ram_free_min_bytes=1,
                emergency_load_per_cpu=9999.0,
                docker_create_concurrency=1,
            )
        )

        # Hold the only slot in a background thread.
        held = threading.Event()
        release = threading.Event()

        def hold_sync():
            with docker_create_slot():
                held.set()
                release.wait(timeout=5)

        t = threading.Thread(target=hold_sync, daemon=True)
        t.start()
        held.wait(timeout=5)

        # Async acquire should now time out because sync holds the slot.
        async def go():
            with patch.object(admission_mod, "DOCKER_CREATE_WAIT_TIMEOUT_S", 0.05):
                async with docker_create_slot_async():
                    pass

        with pytest.raises(CapacityExhausted):
            asyncio.run(go())

        release.set()
        t.join(timeout=5)


# ---------------------------------------------------------------------------
# Snapshot output (used by /metrics)
# ---------------------------------------------------------------------------


class TestDockerSemaCancellation:
    """Async docker semaphore must release the slot even if the outer
    task is cancelled mid-acquire. Without this the worker thread
    eventually wins the semaphore but the async context never released
    it → slot leak."""

    def test_async_acquire_cancelled_releases_slot(self):
        AdmissionGate(
            AdmissionConfig(
                max_live_envs=10,
                emergency_ram_pct=99.9,
                emergency_ram_free_min_bytes=1,
                emergency_load_per_cpu=9999.0,
                docker_create_concurrency=1,
            )
        )

        async def hold_then_yield():
            async with docker_create_slot_async():
                # Yield so the test driver can cancel us while we hold.
                await asyncio.sleep(60)

        async def go():
            t = asyncio.create_task(hold_then_yield())
            await asyncio.sleep(0.05)  # let it acquire
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
            # If release-on-cancel worked, the next acquire succeeds.
            async with docker_create_slot_async():
                pass

        asyncio.run(go())
        # After both runs, semaphore should be fully free.
        assert admission_mod._docker_in_flight() == 0


class TestPluginContract:
    """A plugin violating the contract (raises non-CapacityExhausted)
    propagates as-is — the operator should see the real bug, not a 503."""

    def test_non_capacity_exception_propagates(self):
        def buggy_check():
            raise ValueError("simulated GPU query failed")

        register_emergency_check(buggy_check)
        gate = AdmissionGate(_permissive_config())
        with pytest.raises(ValueError, match="simulated GPU query failed"):
            gate.require_capacity()
        # No 503 attributed: the bug surfaces as 500 to the client.
        assert gate.snapshot()["emergency_503_total"] == 0


class TestCgroupCpuCount:
    """The cgroup-aware CPU count is what derive_max_live_envs uses on
    Linux. Verify all three detection paths via monkey-patched file I/O.
    """

    def test_cgroup_v2_quota_period(self, monkeypatch):
        from lite.gym.remote.admission import effective_cpu_count

        # cgroup v2: ``<quota_us> <period_us>``
        v2_content = "400000 100000"  # 4 CPUs

        def fake_open(path, mode="r"):
            if path == "/sys/fs/cgroup/cpu.max":
                from io import StringIO

                return StringIO(v2_content)
            raise FileNotFoundError(path)

        # Hide os.process_cpu_count (Python 3.13+ path) and patch open.
        monkeypatch.setattr(admission_mod.os, "cpu_count", lambda: 96)
        if hasattr(admission_mod.os, "process_cpu_count"):
            monkeypatch.delattr(admission_mod.os, "process_cpu_count")
        monkeypatch.setattr("builtins.open", fake_open)
        assert effective_cpu_count() == 4

    def test_cgroup_v1_quota_period(self, monkeypatch):
        from lite.gym.remote.admission import effective_cpu_count

        files = {
            "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "800000",  # 8 CPUs
            "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000",
        }

        def fake_open(path, mode="r"):
            if path in files:
                from io import StringIO

                return StringIO(files[path])
            raise FileNotFoundError(path)

        monkeypatch.setattr(admission_mod.os, "cpu_count", lambda: 96)
        if hasattr(admission_mod.os, "process_cpu_count"):
            monkeypatch.delattr(admission_mod.os, "process_cpu_count")
        monkeypatch.setattr("builtins.open", fake_open)
        assert effective_cpu_count() == 8

    def test_fallback_to_os_cpu_count(self, monkeypatch):
        from lite.gym.remote.admission import effective_cpu_count

        def fake_open(path, mode="r"):
            raise FileNotFoundError(path)

        monkeypatch.setattr(admission_mod.os, "cpu_count", lambda: 16)
        if hasattr(admission_mod.os, "process_cpu_count"):
            monkeypatch.delattr(admission_mod.os, "process_cpu_count")
        monkeypatch.setattr("builtins.open", fake_open)
        assert effective_cpu_count() == 16

    def test_cgroup_v2_unlimited_falls_through(self, monkeypatch):
        from lite.gym.remote.admission import effective_cpu_count

        def fake_open(path, mode="r"):
            if path == "/sys/fs/cgroup/cpu.max":
                from io import StringIO

                return StringIO("max 100000")  # unlimited
            raise FileNotFoundError(path)

        monkeypatch.setattr(admission_mod.os, "cpu_count", lambda: 12)
        if hasattr(admission_mod.os, "process_cpu_count"):
            monkeypatch.delattr(admission_mod.os, "process_cpu_count")
        monkeypatch.setattr("builtins.open", fake_open)
        assert effective_cpu_count() == 12


class TestConcurrentAdmitRelease:
    """Under uvicorn's single-thread asyncio model, admit/release are
    already safe. Verify the counter stays consistent even when
    interleaved via cooperative multitasking on the event loop."""

    def test_burst_admit_release_converges_to_zero(self):
        gate = AdmissionGate(_permissive_config(max_live_envs=100))

        async def admit_then_release():
            gate.require_capacity()
            gate.admit()
            await asyncio.sleep(0)  # yield to peers
            gate.release()

        async def run():
            await asyncio.gather(*(admit_then_release() for _ in range(50)))

        asyncio.run(run())
        assert gate.in_flight == 0, (
            f"in_flight leaked under concurrent admit/release: {gate.in_flight}"
        )

    def test_snapshot_during_admit_release_storm(self):
        """Snapshot reads while admit/release fire — single-thread asyncio
        means each call runs atomically between awaits, so snapshot
        never observes an intermediate state."""
        gate = AdmissionGate(_permissive_config(max_live_envs=100))
        seen_in_flight: set[int] = set()

        async def worker():
            for _ in range(20):
                gate.require_capacity()
                gate.admit()
                seen_in_flight.add(gate.snapshot()["in_flight"])
                await asyncio.sleep(0)
                gate.release()
                seen_in_flight.add(gate.snapshot()["in_flight"])

        async def run():
            await asyncio.gather(*(worker() for _ in range(10)))

        asyncio.run(run())
        # Every snapshot value must be within [0, max_live_envs].
        assert all(0 <= v <= 100 for v in seen_in_flight), (
            f"snapshot saw invalid in_flight values: {sorted(seen_in_flight)}"
        )
        assert gate.in_flight == 0


class TestSnapshot:
    def test_snapshot_keys(self):
        gate = AdmissionGate(_permissive_config(max_live_envs=42))
        snap = gate.snapshot()
        expected_keys = {
            "max_live_envs",
            "in_flight",
            "emergency_503_total",
            "capacity_503_total",
            "docker_sema_503_total",
            "env_internal_503_total",
            "docker_in_flight",
            "docker_sema_pending",
            "docker_create_p50_s",
            "host_ram_percent",
            "host_ram_free_bytes",
            "host_swap_percent",
            "host_disk_free_bytes",
            "host_load_per_cpu",
        }
        assert set(snap.keys()) == expected_keys
        assert snap["max_live_envs"] == 42
        assert snap["in_flight"] == 0

    def test_snapshot_under_1ms(self):
        gate = AdmissionGate(_permissive_config())
        # First call may pay one-time costs; measure the steady state.
        gate.snapshot()
        t0 = time.monotonic()
        gate.snapshot()
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        # 5 ms ceiling is generous — psutil queries on Linux are
        # typically < 0.5 ms.
        assert elapsed_ms < 5.0, f"snapshot took {elapsed_ms:.2f} ms"


class TestLayerTagging:
    """`CapacityExhausted.layer` correctly identifies which gate fired."""

    def test_l1_ram_raise_tags_emergency(self):
        # Force RAM check failure by setting an impossibly low threshold.
        cfg = AdmissionConfig(
            max_live_envs=4,
            emergency_ram_pct=0.001,  # any usage > this triggers
            emergency_ram_free_min_bytes=1,
            emergency_load_per_cpu=99.0,
            docker_create_concurrency=2,
        )
        gate = AdmissionGate(cfg)
        with pytest.raises(CapacityExhausted) as exc_info:
            gate.require_capacity()
        assert exc_info.value.layer == "emergency"
        assert gate.snapshot()["emergency_503_total"] == 1

    def test_l2_cap_raise_tags_capacity(self):
        gate = AdmissionGate(_permissive_config(max_live_envs=2))
        gate.require_capacity()
        gate.admit()
        gate.require_capacity()
        gate.admit()
        with pytest.raises(CapacityExhausted) as exc_info:
            gate.require_capacity()
        assert exc_info.value.layer == "capacity"
        assert gate.snapshot()["capacity_503_total"] == 1

    def test_docker_sema_timeout_tags_docker_sema(self):
        cfg = AdmissionConfig(
            max_live_envs=4,
            emergency_ram_pct=99.9,
            emergency_ram_free_min_bytes=1,
            emergency_load_per_cpu=99.0,
            docker_create_concurrency=1,
        )
        AdmissionGate(cfg)  # configures the sema
        # Hold the only slot in a worker thread so the next acquire times out.
        held = threading.Event()
        done = threading.Event()

        def hog():
            with docker_create_slot():
                held.set()
                done.wait(timeout=10)

        t = threading.Thread(target=hog, daemon=True)
        t.start()
        held.wait(timeout=5)
        try:
            with pytest.raises(CapacityExhausted) as exc_info:
                with docker_create_slot():
                    pass
            assert exc_info.value.layer == "docker_sema"
        finally:
            done.set()
            t.join(timeout=5)

    def test_env_internal_default_layer_when_not_specified(self):
        # An env that raises without setting layer must default to
        # "env_internal" so the handler attributes it correctly.
        e = CapacityExhausted("browser pool exhausted", retry_after_s=60)
        assert e.layer == "env_internal"

    def test_record_env_internal_503_increments_counter(self):
        gate = AdmissionGate(_permissive_config())
        before = gate.snapshot()["env_internal_503_total"]
        gate.record_env_internal_503()
        gate.record_env_internal_503()
        after = gate.snapshot()["env_internal_503_total"]
        assert after - before == 2
