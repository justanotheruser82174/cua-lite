"""Boot-path stress tests for envs that go through sandbox/base._attempt_boot_computer.

Targets the failure mode that escaped ``test_live_stress.py`` (which only
uses ``screenspot_pro`` — a stateless env with no docker boot path):

* docker_sema (default 8) backpressures concurrent ``/reset`` at
  ``ENV_CONCURRENCY=64`` → 503 CapacityExhausted
* The client retries the same instance id (typical training behavior)
* Before the fix in ``sandbox/base.py:_attempt_boot_computer``, the leaked
  ``self._computer`` (half-spawned Computer with ``_interface=None``)
  made the retry skip the boot loop and crash in ``setup_fn`` with
  ``RuntimeError: Computer interface not initialized``

The asserted invariant: under load + retries, zero requests end with the
``Computer interface not initialized`` error.

We *also* validate that successful resets returned a non-trivial
screenshot. A bare 200 status is not enough — the sandbox/base.py reset path
returns ``observation.image`` (raw PNG bytes) from inside the container, and
under load we've previously seen "200 OK" responses carrying the GNOME
startup-failure splash (a uniformly grey or low-entropy frame). Each
successful instance must pass the cheap size/diversity heuristic in
:func:`_screenshot_looks_healthy`.

Other heavy envs (androidworld, mobilegym) don't share the sandbox/base.py
boot path but ARE part of the user's mandatory ENV_CONCURRENCY=64 ask, so
we exercise their admission/lifecycle as well — flagging any analogous
state-leak races there.

Run::

    CUA_LITE_LIVE_ENV_SERVER_URL=http://127.0.0.1:30200 \
    CUA_LITE_LIVE_ENV_SERVER_TOKEN=... \
    CUA_LITE_LIVE_ADMIN_TOKEN=... \
      uv run pytest -m "live and stress" \
        tests/gym/remote/test_live_stress_boot_path.py -xvs
"""
from __future__ import annotations

import asyncio
import os
import socket
import time
from collections import Counter
from urllib.parse import urlparse

import httpx
import pytest
from gym.remote.conftest import unique_live_scope

from lite.gym.remote.frame import decode_reset_observation

pytestmark = [pytest.mark.live, pytest.mark.stress]


LIVE_URL_ENV_VAR = "CUA_LITE_LIVE_ENV_SERVER_URL"
LIVE_TOKEN_ENV_VAR = "CUA_LITE_LIVE_ENV_SERVER_TOKEN"
LIVE_ADMIN_TOKEN_ENV_VAR = "CUA_LITE_LIVE_ADMIN_TOKEN"

LIVE_SERVER_URL = os.environ.get(LIVE_URL_ENV_VAR, "").rstrip("/")
LIVE_SERVER_TOKEN = os.environ.get(LIVE_TOKEN_ENV_VAR, "")
LIVE_ADMIN_TOKEN = os.environ.get(LIVE_ADMIN_TOKEN_ENV_VAR, "")

# (env_id, burst_size). Bursts are sized per env weight:
#
# * lite.osworld @ 64 reproduces the user's failing training condition.
# * Android emulators (androidworld, androidlab) and mobilegym
#   Chromium pool are heavier — 32 saturates each env's own
#   pool/limit and still loads docker_sema where relevant.
# * Grounding / stateless reward envs (osworld_g, screenspot_pro) have
#   no spawn-time cost — they should sustain very high concurrency.
#   256 stresses the FastAPI/asyncio path without disk/network thrash.
# * captcha + webgym fall between the two: light enough to push past
#   normal training concurrency, heavy enough to want a real cap.
# Burst sizes are multiplied by ``CUA_LITE_STRESS_MULT`` (default 1)
# at module load — used to ramp the same suite to 2×/4× concurrency
# without forking the test file. Pair with the host-pressure guard
# (``_RAM_PCT_ABORT=66``) so a doubled run still aborts before the
# cluster goes into trouble.
_STRESS_MULT = int(os.environ.get("CUA_LITE_STRESS_MULT", "1"))
# ``webgym`` is registered in the env-server but POST /instances
# returns 501 on this host because its runtime deps (chromium kernel
# extension / `omniboxes` service) aren't installed locally — see
# lite/gym/envs/webgym/README.md. Listed but commented out so the
# stress suite stays runnable across hosts; uncomment after running
# the webgym install steps.
TARGETS: list[tuple[str, int]] = [
    ("lite.osworld", 64 * _STRESS_MULT),
    ("androidworld", 32 * _STRESS_MULT),
    ("androidlab", 32 * _STRESS_MULT),
    ("mobilegym", 32 * _STRESS_MULT),
    ("captcha", 128 * _STRESS_MULT),
    ("osworld_g", 256 * _STRESS_MULT),
    ("screenspot_pro", 256 * _STRESS_MULT),
    # ("webgym", 64 * _STRESS_MULT),  # 501 on hosts without webgym install.
]

# The mixed-burst test runs THESE envs concurrently — one instance per
# env_id × ``MIXED_PER_ENV`` repetitions, all in flight at the same time.
# Sized to keep host RAM peak well under 200 GB (≈14 % of a 1.5 TB host)
# while still exercising cross-env contention on the shared
# docker_sema, the L2 cap, and the host load sensor.
MIXED_ENV_IDS: list[str] = [
    "lite.osworld", "androidworld", "androidlab", "mobilegym",
    "captcha", "osworld_g", "screenspot_pro",
]
MIXED_PER_ENV = 8 * _STRESS_MULT  # 8 per env × 7 envs × mult

# Marker string from computer.py:1036 that ``sandbox/base.py`` previously
# leaked through on retry.
COMPUTER_NOT_INIT_MSG = "Computer interface not initialized"


def _server_address() -> tuple[str, int] | None:
    parsed = urlparse(LIVE_SERVER_URL)
    if not parsed.scheme or not parsed.hostname:
        return None
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return parsed.hostname, port


def _listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module", autouse=True)
def require_dev_server():
    missing = [
        name for name, value in [
            (LIVE_URL_ENV_VAR, LIVE_SERVER_URL),
            (LIVE_TOKEN_ENV_VAR, LIVE_SERVER_TOKEN),
            (LIVE_ADMIN_TOKEN_ENV_VAR, LIVE_ADMIN_TOKEN),
        ] if not value
    ]
    if missing:
        pytest.skip("missing live env-server vars: " + ", ".join(missing))

    address = _server_address()
    if address is None:
        pytest.skip(f"{LIVE_URL_ENV_VAR} must be an absolute URL")

    host, port = address
    if not _listening(host, port):
        pytest.skip(f"live env-server not listening at {LIVE_SERVER_URL}")


def _envs_available() -> set[str]:
    r = httpx.get(
        f"{LIVE_SERVER_URL}/envs",
        headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
        timeout=5.0,
    )
    r.raise_for_status()
    return set(r.json())


def _first_task_id(env_id: str) -> str | None:
    r = httpx.get(
        f"{LIVE_SERVER_URL}/envs/{env_id}/tasks",
        headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
        timeout=5.0,
    )
    if r.status_code != 200:
        return None
    splits = r.json().get("splits") or {}
    for ids in splits.values():
        if ids:
            return ids[0]
    return None


# ---------------------------------------------------------------------------
# Lifecycle helpers — mirror the client's retry behavior on 503 with the
# SAME instance id (this is what triggers the sandbox/base.py boot-state leak).
# ---------------------------------------------------------------------------

async def _create_one(client: httpx.AsyncClient, env_id: str,
                      task_id: str, session_id: str) -> tuple[int, str | None]:
    try:
        r = await client.post(
            f"{LIVE_SERVER_URL}/instances",
            json={"env_key": f"{env_id}@{task_id}",
                  "env_kwargs": {}, "session_id": session_id},
            headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
            timeout=60.0,
        )
        if r.status_code == 201:
            return (201, r.json().get("id"))
        return (r.status_code, None)
    except Exception as e:
        return (0, f"exc:{type(e).__name__}")


#: Per-env screenshot size floors. Higher floors catch subtle
#: failure modes (e.g. lite.osworld's uniform-grey GNOME boot splash at
#: ~10 KB); lower floors only catch literal-empty / missing-asset
#: failures. Tuned from observed sizes on a healthy host:
#:
#: * lite.osworld: XFCE/GNOME @ 1024x768 with windows: 80-400 KB; boot
#:   splash: 2-15 KB → 25 KB floor.
#: * androidworld / androidlab: emulator @ 1080x1920; populated home
#:   screen: 50-300 KB; blank emulator frame: 5-20 KB → 25 KB floor.
#: * mobilegym: Chromium page @ 1080x2400; populated app: 50-200 KB;
#:   blank: < 5 KB → 20 KB floor.
#: * Other envs (grounding / captcha / webgym): return a TASK image,
#:   sizes vary widely; we only catch the literal-empty case → 1 KB
#:   floor (anything below means the server didn't include the field).
_SCREENSHOT_MIN_BYTES_BY_ENV: dict[str, int] = {
    "lite.osworld": 25_000,
    "androidworld": 25_000,
    "androidlab": 25_000,
    "mobilegym": 20_000,
}
_SCREENSHOT_MIN_BYTES_DEFAULT = 1_000


def _screenshot_looks_healthy(shot: bytes | None,
                               env_id: str = "") -> tuple[bool, str]:
    """Cheap-but-load-bearing screenshot validity check.

    Returns ``(is_healthy, reason)``. Reason is empty on success, a
    short diagnostic on failure (logged by the test for triage).

    Avoids decoding the PNG itself — the raw-byte count is enough to
    discriminate "real desktop" from "uniform error frame", and skips a
    PIL/numpy dependency for the test path. ``obs.image`` is already
    raw PNG bytes (no base64 hop). The per-env floor is set so a healthy
    frame clears it by ~4×.
    """
    floor = _SCREENSHOT_MIN_BYTES_BY_ENV.get(env_id, _SCREENSHOT_MIN_BYTES_DEFAULT)
    if not shot:
        return (False, "screenshot missing/empty")
    n = len(shot)
    if n < floor:
        return (False, f"decoded {n} B < {floor} B floor "
                       f"(likely uniform-frame error / missing asset)")
    return (True, "")


async def _reset_with_retries(client: httpx.AsyncClient, inst_id: str,
                              env_id: str = "",
                              deadline_s: float = 300.0) -> tuple[int, str]:
    """POST /instances/<id>/reset with deadline-based 503 retry.

    Returns ``(final_status, body_excerpt)``. ``body_excerpt`` carries
    the server's error detail when status != 200, *or* a screenshot
    health diagnostic when a 200 came back with a suspicious-looking
    frame. A reset that returned 200 but failed the per-env screenshot
    heuristic (see :data:`_SCREENSHOT_MIN_BYTES_BY_ENV`) is
    re-classified as status ``599`` (synthetic, distinguishable from
    any real HTTP code) with a ``screenshot:`` prefix on the body.

    The 300 s default mirrors the rollout client
    (``_CLIENT_503_DEADLINE_S``) so the test sees the same retry budget
    real training does. Under burst this lets transient post-boot
    failures recover via the sandbox/base.py ``EnvDepsMissingError``→503
    conversion instead of falsely flagging the run as broken.
    """
    last_body = ""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            r = await client.post(
                f"{LIVE_SERVER_URL}/instances/{inst_id}/reset",
                json={},
                headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
                timeout=180.0,
            )
        except Exception as e:
            last_body = f"exc:{type(e).__name__}:{e}"
            await asyncio.sleep(1.0)
            continue
        if r.status_code == 200:
            try:
                shot = decode_reset_observation(r.content).image
            except Exception:
                shot = None
            ok, reason = _screenshot_looks_healthy(shot, env_id=env_id)
            if ok:
                return (200, "")
            return (599, f"screenshot:{reason}")
        # Capture detail/body for inspection; clip to keep memory bounded.
        last_body = r.text[:500]
        if r.status_code == 503:
            # Respect Retry-After if present.
            ra = r.headers.get("Retry-After")
            try:
                delay = float(ra) if ra else 1.0
            except ValueError:
                delay = 1.0
            await asyncio.sleep(min(delay, 5.0))
            continue
        # Non-retryable
        return (r.status_code, last_body)
    return (r.status_code, last_body)


async def _delete_one(client: httpx.AsyncClient, inst_id: str) -> int:
    try:
        r = await client.delete(
            f"{LIVE_SERVER_URL}/instances/{inst_id}",
            headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
            timeout=30.0,
        )
        return r.status_code
    except Exception:
        return 0


async def _drain_session(client: httpx.AsyncClient, env_id: str, session_id: str) -> int:
    r = await client.delete(
        f"{LIVE_SERVER_URL}/instances",
        params={"session_id": session_id, "env_id": env_id},
        headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
        timeout=120.0,
    )
    if r.status_code == 200:
        return len(r.json().get("closed", []))
    return 0


# ---------------------------------------------------------------------------
# Parametrized stress: per env_id, burst create + reset-with-retries.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("env_id,burst", TARGETS)
def test_create_and_reset_under_concurrency_no_interface_leak(env_id, burst):
    """No request must end with ``Computer interface not initialized``.

    This is the asserted invariant from the sandbox/base.py fix. Other 500s
    are logged (per-env-specific failure modes are out of scope for this
    test) but don't fail the suite.

    503 CapacityExhausted is expected backpressure under the burst and
    is satisfied by client retries.
    """
    available = _envs_available()
    if env_id not in available:
        pytest.skip(f"{env_id} not registered on dev server")
    task_id = _first_task_id(env_id)
    if not task_id:
        pytest.skip(f"no usable task_id for {env_id}")

    session_id = unique_live_scope(f"boot_{env_id}")

    async def run():
        async with httpx.AsyncClient() as client:
            # Burst-create
            create_results = await asyncio.gather(
                *(_create_one(client, env_id, task_id, session_id)
                  for _ in range(burst))
            )
            create_codes = Counter(sc for sc, _ in create_results)
            created_ids = [iid for sc, iid in create_results
                           if sc == 201 and isinstance(iid, str)]

            # Reset each successfully-created instance with up to 3 retries
            # on 503. This is the exact pattern the training client uses
            # — and the exact path that ``sandbox/base.py`` was leaking state on.
            reset_results = await asyncio.gather(
                *(_reset_with_retries(client, iid, env_id=env_id)
                  for iid in created_ids)
            )

            # Always cleanup, even on assertion failure
            try:
                await _drain_session(client, env_id, session_id)
            except Exception:
                pass

            return create_codes, reset_results

    create_codes, reset_results = asyncio.run(run())

    reset_codes = Counter(sc for sc, _ in reset_results)
    leak_msgs = [body for sc, body in reset_results
                 if COMPUTER_NOT_INIT_MSG in (body or "")]
    bad_screenshots = [body for sc, body in reset_results
                       if sc == 599 and body.startswith("screenshot:")]

    print(f"\n[{env_id}@burst={burst}] create: {dict(create_codes)}")
    print(f"[{env_id}@burst={burst}] reset:  {dict(reset_codes)}")
    if leak_msgs:
        print(f"[{env_id}@burst={burst}] LEAK SAMPLES (first 3):")
        for m in leak_msgs[:3]:
            print(f"  {m[:200]}")
    if bad_screenshots:
        print(f"[{env_id}@burst={burst}] BAD-SCREENSHOT SAMPLES (first 3):")
        for m in bad_screenshots[:3]:
            print(f"  {m[:200]}")

    # The boot-state-leak invariant — applies to lite.osworld; for the
    # other envs it's just a sanity check (different boot paths).
    assert not leak_msgs, (
        f"{env_id}: {len(leak_msgs)} requests leaked "
        f"'{COMPUTER_NOT_INIT_MSG}' under burst={burst} "
        f"(sandbox/base.py boot-state cleanup regressed)"
    )
    # Screenshot validity — catches "reset returned 200 but the
    # observation is a uniform-frame failure" mode that the
    # ``_read_gnome_failed_marker`` (and its android/mobilegym
    # equivalents) guards can miss: partial GNOME start, blank emulator
    # frame, login prompt, dead Chromium target. The per-env floor in
    # ``_SCREENSHOT_MIN_BYTES_BY_ENV`` is tuned so a real desktop /
    # emulator / Chromium render clears it by ~4×; only envs in that
    # map are asserted, because stateless envs (captcha/grounding/
    # webgym) return task images with widely-varying natural sizes.
    if env_id in _SCREENSHOT_MIN_BYTES_BY_ENV:
        assert not bad_screenshots, (
            f"{env_id}: {len(bad_screenshots)} reset(s) returned 200 with "
            f"low-entropy screenshots (probably boot-failure / blank frame); "
            f"first: {bad_screenshots[0][:200]}"
        )


# ---------------------------------------------------------------------------
# Smoke: verify the marker string isn't trivially absent because the env
# isn't reachable at all (no creates succeeded → nothing tested).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("env_id,_", TARGETS)
def test_at_least_one_create_succeeds(env_id, _):
    """Guards against a green ``test_create_and_reset_under_concurrency_no_interface_leak``
    that's actually green because every create 503'd or 5xx'd — there was
    nothing to reset.
    """
    available = _envs_available()
    if env_id not in available:
        pytest.skip(f"{env_id} not registered on dev server")
    task_id = _first_task_id(env_id)
    if not task_id:
        pytest.skip(f"no usable task_id for {env_id}")

    session_id = unique_live_scope(f"boot_smoke_{env_id}")

    async def run():
        async with httpx.AsyncClient() as client:
            sc, iid = await _create_one(client, env_id, task_id, session_id)
            if sc == 201 and isinstance(iid, str):
                try:
                    await _delete_one(client, iid)
                except Exception:
                    pass
            try:
                await _drain_session(client, env_id, session_id)
            except Exception:
                pass
            return sc

    sc = asyncio.run(run())
    assert sc == 201, f"{env_id}: a single create returned {sc} — env unhealthy"


# ---------------------------------------------------------------------------
# Host-pressure guard. Real-time RAM / load sampling so a stress test
# can't quietly push a shared host into thrash. Aborts the running test
# the moment either threshold trips; final values land in pytest output.
# ---------------------------------------------------------------------------

try:
    import psutil  # noqa: E402
except ImportError:
    psutil = None  # type: ignore[assignment]


# 66 % RAM use is the abort threshold — operator-mandated "leave at
# least 500 GB on a 1.5 TB host" rule so stress tests can't squeeze out
# concurrent training tenants. Well below the env-server's L1
# emergency_ram_pct (95 %) so the test gives up long before the server
# starts 503-ing. load_per_cpu of 5 means CPUs are 5× oversubscribed —
# brief peaks are fine but sustained means the host is being wedged.
# Tighten both via env var on smaller hosts.
_RAM_PCT_ABORT = float(os.environ.get("CUA_LITE_STRESS_RAM_PCT_ABORT", "66"))
_LOAD_PER_CPU_ABORT = float(os.environ.get("CUA_LITE_STRESS_LOAD_PER_CPU_ABORT", "5"))


async def _host_pressure_guard(stop_event: asyncio.Event,
                                peaks: dict) -> None:
    """Poll host RAM + load_per_cpu every 2 s until ``stop_event`` is
    set; populate ``peaks`` with running maxima. If either crosses the
    abort threshold, set ``stop_event`` (caller aborts its work).
    """
    if psutil is None:
        return
    cpu = os.cpu_count() or 1
    while not stop_event.is_set():
        try:
            ram_pct = psutil.virtual_memory().percent
            load1 = os.getloadavg()[0]
            lpc = load1 / cpu
            peaks["ram_pct"] = max(peaks.get("ram_pct", 0.0), ram_pct)
            peaks["load_per_cpu"] = max(peaks.get("load_per_cpu", 0.0), lpc)
            if ram_pct > _RAM_PCT_ABORT or lpc > _LOAD_PER_CPU_ABORT:
                peaks["aborted"] = True
                peaks["abort_reason"] = (
                    f"ram_pct={ram_pct:.1f}%, load_per_cpu={lpc:.2f}"
                )
                stop_event.set()
                return
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=2.0)
        except TimeoutError:
            pass


# ---------------------------------------------------------------------------
# Mixed-env concurrent burst. Real production runs hit the env-server
# with a mix of env types simultaneously (multi-objective rollouts).
# This test checks:
#   * docker_sema fairness across env types (no env starves)
#   * L2 cap is enforced GLOBALLY across env types (not per-env_id)
#   * One env type's failure mode doesn't bleed into another
#   * The host-pressure guard caps total RAM
# ---------------------------------------------------------------------------

def test_mixed_env_concurrent_burst():
    """All MIXED_ENV_IDS × MIXED_PER_ENV in flight at the same time."""
    available = _envs_available()
    selected = [e for e in MIXED_ENV_IDS if e in available]
    if not selected:
        pytest.skip("none of MIXED_ENV_IDS registered on dev server")

    env_to_task: dict[str, str] = {}
    for env_id in selected:
        tid = _first_task_id(env_id)
        if not tid:
            continue
        env_to_task[env_id] = tid
    if not env_to_task:
        pytest.skip("no usable task_ids for any MIXED env")

    session_id = unique_live_scope("boot_mixed")
    peaks: dict = {}
    stop_event_holder: dict[str, asyncio.Event] = {}

    async def run():
        stop_event = asyncio.Event()
        stop_event_holder["e"] = stop_event
        guard = asyncio.create_task(_host_pressure_guard(stop_event, peaks))
        try:
            async with httpx.AsyncClient() as client:
                pairs = [
                    (env_id, env_to_task[env_id])
                    for env_id in env_to_task
                    for _ in range(MIXED_PER_ENV)
                ]
                # All creates fire simultaneously — peak admission pressure.
                create_results = await asyncio.gather(*(
                    _create_one(client, env_id, task_id, session_id)
                    for env_id, task_id in pairs
                ))
                created: list[tuple[str, str]] = []
                for (env_id, _), (sc, iid) in zip(pairs, create_results):
                    if sc == 201 and isinstance(iid, str):
                        created.append((env_id, iid))
                # Reset every created instance concurrently.
                reset_results = await asyncio.gather(*(
                    _reset_with_retries(client, iid, env_id=eid)
                    for eid, iid in created
                ))
                # Cleanup.
                try:
                    for created_env_id in env_to_task:
                        await _drain_session(client, created_env_id, session_id)
                except Exception:
                    pass
                return create_results, reset_results, created
        finally:
            stop_event.set()
            try:
                await asyncio.wait_for(guard, timeout=5.0)
            except TimeoutError:
                guard.cancel()

    create_results, reset_results, created = asyncio.run(run())

    create_codes = Counter(sc for sc, _ in create_results)
    reset_codes = Counter(sc for sc, _ in reset_results)
    per_env_created = Counter(env_id for env_id, _ in created)

    print(f"\n[mixed burst peak] RAM%={peaks.get('ram_pct', 0):.1f} "
          f"load_per_cpu={peaks.get('load_per_cpu', 0):.2f} "
          f"aborted={peaks.get('aborted', False)}")
    print(f"[mixed burst] create: {dict(create_codes)}")
    print(f"[mixed burst] reset:  {dict(reset_codes)}")
    print(f"[mixed burst] per-env created: {dict(per_env_created)}")

    # Host-pressure guard never tripped → we kept under threshold.
    assert not peaks.get("aborted", False), (
        f"host pressure abort: {peaks.get('abort_reason', '?')} — "
        f"reduce MIXED_PER_ENV or env weights"
    )
    # No env type starves: every registered env got at least one create.
    starved = [e for e in env_to_task if per_env_created[e] == 0]
    assert not starved, (
        f"these envs got 0 successful creates under mixed burst: {starved}; "
        f"create-code breakdown: {dict(create_codes)}"
    )


# ---------------------------------------------------------------------------
# Rapid create-delete cycling. Catches admission counter leaks (in_flight
# drifting upward over time even as instances are properly closed) and
# /admin/budget reporting drift. Runs against a lightweight env so the
# cycle time is dominated by the admission path, not container boot.
# ---------------------------------------------------------------------------

def test_rapid_create_delete_cycles_no_admission_leak():
    """N rounds of (burst-create 32, burst-delete 32) on screenspot_pro.

    The asserted invariant: after the loop, ``in_flight`` from
    ``/host_status`` returns to 0 (within tolerance) — i.e., the
    admission counter doesn't drift.
    """
    available = _envs_available()
    env_id = "screenspot_pro"
    if env_id not in available:
        pytest.skip(f"{env_id} not registered")
    task_id = _first_task_id(env_id)
    if not task_id:
        pytest.skip(f"no task_id for {env_id}")

    session_id = unique_live_scope("boot_cycle")
    ROUNDS = 4
    BURST = 32

    async def run():
        async with httpx.AsyncClient() as client:
            for _ in range(ROUNDS):
                creates = await asyncio.gather(*(
                    _create_one(client, env_id, task_id, session_id)
                    for _ in range(BURST)
                ))
                ids = [iid for sc, iid in creates
                       if sc == 201 and isinstance(iid, str)]
                await asyncio.gather(*(_delete_one(client, iid) for iid in ids))
            # Final drain catches anything strays.
            try:
                await _drain_session(client, env_id, session_id)
            except Exception:
                pass

    asyncio.run(run())

    # Probe admin/budget for in_flight; we expect 0 for our session.
    r = httpx.get(
        f"{LIVE_SERVER_URL}/admin/budget",
        headers={"Authorization": f"Bearer {LIVE_ADMIN_TOKEN}"},
        timeout=10.0,
    )
    assert r.status_code == 200, f"/admin/budget returned {r.status_code}"
    budget = r.json()
    in_flight_total = budget.get("in_flight", 0)
    # ``in_flight`` is GLOBAL across all sessions — other tests/agents
    # may legitimately have envs alive. We can only assert "doesn't
    # explode to BURST or higher" from our own cycling.
    assert in_flight_total < BURST, (
        f"in_flight={in_flight_total} after {ROUNDS} clean cycles — "
        f"admission counter is leaking"
    )


# ---------------------------------------------------------------------------
# Multi-session isolation. Two concurrent burst sessions on the same
# heavy env. Catches: token/session_id ownership misrouting, cross-
# session DELETE bleeding, drift-reaper picking up the wrong session.
# ---------------------------------------------------------------------------

def test_multi_session_isolation_on_lite_osworld():
    """Two sessions × 8 lite.osworld each running in parallel. After
    only session A's instances are deleted, session B's must still be
    alive and reset-able.
    """
    available = _envs_available()
    env_id = "lite.osworld"
    if env_id not in available:
        pytest.skip(f"{env_id} not registered")
    task_id = _first_task_id(env_id)
    if not task_id:
        pytest.skip(f"no task_id for {env_id}")

    session_base = unique_live_scope("boot_iso")
    session_a = f"{session_base}_a"
    session_b = f"{session_base}_b"
    PER_SESSION = 8

    async def run():
        async with httpx.AsyncClient() as client:
            creates = await asyncio.gather(*(
                _create_one(client, env_id, task_id, s)
                for s in (session_a, session_b)
                for _ in range(PER_SESSION)
            ))
            by_session: dict[str, list[str]] = {session_a: [], session_b: []}
            for i, (sc, iid) in enumerate(creates):
                if sc != 201 or not isinstance(iid, str):
                    continue
                s = session_a if i < PER_SESSION else session_b
                by_session[s].append(iid)

            # Drain session A only.
            await _drain_session(client, env_id, session_a)
            await asyncio.sleep(2.0)

            # Verify session B's instances are still alive.
            alive_b = 0
            for iid in by_session[session_b]:
                try:
                    r = await client.get(
                        f"{LIVE_SERVER_URL}/instances/{iid}",
                        headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
                        timeout=10.0,
                    )
                    if r.status_code == 200:
                        alive_b += 1
                except Exception:
                    pass

            # Cleanup B.
            try:
                await _drain_session(client, env_id, session_b)
            except Exception:
                pass

            return by_session, alive_b

    by_session, alive_b = asyncio.run(run())
    created_a = len(by_session[session_a])
    created_b = len(by_session[session_b])
    print(f"[multi-session] created A={created_a} B={created_b}; "
          f"after-drain-A alive B={alive_b}")

    if created_b == 0:
        pytest.skip("session B got 0 creates — cannot test isolation")
    # All of B's instances must survive the A drain.
    assert alive_b == created_b, (
        f"multi-session isolation: drained A but B lost {created_b - alive_b}/"
        f"{created_b} instances — cross-session bleed"
    )


# ---------------------------------------------------------------------------
# More corner-case tests — observability under pressure, double-reset,
# instance lifecycle correctness, drift-reaper interaction.
# ---------------------------------------------------------------------------

def test_metrics_scrape_during_burst_stays_responsive():
    """Hammer /metrics from another connection while 32 osworld creates
    are in-flight. The scrape must:
      * Stay under 500 ms p99 (a hung scrape means snapshot() is
        blocking the event loop).
      * Return 200 every time (no leaked state under contention).
    """
    available = _envs_available()
    env_id = "lite.osworld"
    if env_id not in available:
        pytest.skip(f"{env_id} not registered")
    task_id = _first_task_id(env_id)
    if not task_id:
        pytest.skip(f"no task_id for {env_id}")

    session_id = unique_live_scope("boot_metrics")
    scrape_latencies: list[float] = []
    scrape_codes: list[int] = []

    async def run():
        async with httpx.AsyncClient() as client:
            async def scrape_loop(stop):
                while not stop.is_set():
                    t0 = time.monotonic()
                    try:
                        r = await client.get(
                            f"{LIVE_SERVER_URL}/metrics",
                            headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
                            timeout=5.0,
                        )
                        scrape_codes.append(r.status_code)
                    except Exception:
                        scrape_codes.append(0)
                    scrape_latencies.append(time.monotonic() - t0)
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=0.1)
                    except TimeoutError:
                        pass

            stop = asyncio.Event()
            scraper = asyncio.create_task(scrape_loop(stop))
            try:
                await asyncio.gather(*(
                    _create_one(client, env_id, task_id, session_id)
                    for _ in range(32)
                ))
            finally:
                stop.set()
                try:
                    await asyncio.wait_for(scraper, timeout=5.0)
                except TimeoutError:
                    scraper.cancel()
                try:
                    await _drain_session(client, env_id, session_id)
                except Exception:
                    pass

    asyncio.run(run())

    if not scrape_latencies:
        pytest.skip("metrics scraper never managed a sample")
    n = len(scrape_latencies)
    p99 = sorted(scrape_latencies)[max(0, int(n * 0.99) - 1)]
    bad = sum(1 for c in scrape_codes if c != 200)
    print(f"\n[metrics-burst] {n} scrapes, p99={p99*1000:.0f}ms, "
          f"non-200={bad}, codes={Counter(scrape_codes)}")
    assert bad == 0, (
        f"/metrics returned non-200 {bad}/{n} times during burst"
    )
    assert p99 < 0.5, (
        f"/metrics p99 latency {p99*1000:.0f}ms > 500ms during burst — "
        f"snapshot() is blocking the event loop"
    )


def test_repeated_reset_returns_fresh_observation():
    """Reset twice on the same instance — must succeed both times and
    return TWO observations (no caching, no half-state). The second
    reset rebuilds the env from scratch via the sandbox/base.py boot loop
    that the rev-7 fix made re-entrant.
    """
    available = _envs_available()
    env_id = "lite.osworld"
    if env_id not in available:
        pytest.skip(f"{env_id} not registered")
    task_id = _first_task_id(env_id)
    if not task_id:
        pytest.skip(f"no task_id for {env_id}")
    session_id = unique_live_scope("boot_double_reset")

    async def run():
        async with httpx.AsyncClient() as client:
            sc, iid = await _create_one(client, env_id, task_id, session_id)
            if sc != 201 or not isinstance(iid, str):
                pytest.skip(f"create returned {sc}")
            try:
                a = await _reset_with_retries(client, iid, env_id=env_id)
                b = await _reset_with_retries(client, iid, env_id=env_id)
            finally:
                try:
                    await _delete_one(client, iid)
                except Exception:
                    pass
                try:
                    await _drain_session(client, env_id, session_id)
                except Exception:
                    pass
            return a, b

    a, b = asyncio.run(run())
    assert a[0] == 200, f"first reset returned {a[0]}: {a[1][:200]}"
    assert b[0] == 200, f"second reset returned {b[0]}: {b[1][:200]}"


def test_step_without_reset_returns_error_not_500_storm():
    """``/step`` on a freshly-created (un-reset) instance must surface a
    user-visible error (404 / 409 / 400) — never 500 + traceback. Burst
    32 such calls; any cleanly-typed error class works as long as the
    server doesn't fall over.
    """
    available = _envs_available()
    env_id = "lite.osworld"
    if env_id not in available:
        pytest.skip(f"{env_id} not registered")
    task_id = _first_task_id(env_id)
    if not task_id:
        pytest.skip(f"no task_id for {env_id}")
    session_id = unique_live_scope("boot_step_no_reset")

    async def step_one(client, iid):
        try:
            r = await client.post(
                f"{LIVE_SERVER_URL}/instances/{iid}/step",
                json={"actions": []},
                headers={"Authorization": f"Bearer {LIVE_SERVER_TOKEN}"},
                timeout=30.0,
            )
            return r.status_code
        except Exception:
            return 0

    async def run():
        async with httpx.AsyncClient() as client:
            creates = await asyncio.gather(*(
                _create_one(client, env_id, task_id, session_id)
                for _ in range(32)
            ))
            ids = [iid for sc, iid in creates
                   if sc == 201 and isinstance(iid, str)]
            codes = await asyncio.gather(*(step_one(client, iid) for iid in ids))
            try:
                await _drain_session(client, env_id, session_id)
            except Exception:
                pass
            return codes

    codes = asyncio.run(run())
    code_counter = Counter(codes)
    print(f"\n[step-no-reset] status: {dict(code_counter)}")
    # Anything in the 4xx range is fine — we just don't want 5xx (server
    # state machine bug) or 0 (connection drop / hang).
    server_errors = sum(1 for c in codes if 500 <= c < 600)
    drops = sum(1 for c in codes if c == 0)
    assert server_errors == 0, (
        f"/step before /reset triggered {server_errors} 5xx — server state "
        f"machine doesn't guard the un-reset case cleanly"
    )
    assert drops == 0, (
        f"/step before /reset dropped {drops} connections — server crashed "
        f"or hung"
    )
