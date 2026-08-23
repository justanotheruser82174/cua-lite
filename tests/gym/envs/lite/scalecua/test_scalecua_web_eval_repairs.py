"""ScaleCUA web-result evaluator repair tests."""

from __future__ import annotations

import base64
from datetime import datetime
from types import SimpleNamespace

import pytest

from lite.gym.envs.lite.scalecua.src.osworld import judges
from lite.gym.envs.lite.scalecua.src.osworld import verify as scalecua_verify
from lite.gym.envs.lite.scalecua.src.utils import dataset


def _overlays_ready() -> bool:
    """Judge-overlay tests need the imported getters/metrics modules too."""
    return all(
        (root / f"{name}.py").is_file()
        for split in dataset.RUNTIME_SPLITS
        if (root := judges.overlay_dir(split)) is not None
        for name in ("getters", "metrics")
    )


class _FakeInterface:
    def __init__(
        self,
        stdout: str | list[str] = "stdout",
        files: dict[str, bytes] | None = None,
    ):
        self.commands: list[str] = []
        self.command_calls: list[dict[str, object]] = []
        self.hotkeys: list[tuple[str, ...]] = []
        self.typed_text: list[str] = []
        self.stdout = stdout
        self.files = files or {}

    async def read_bytes(self, path: str) -> bytes:
        if path in self.files:
            return self.files[path]
        return f"bytes:{path}".encode()

    async def screenshot(self) -> bytes:
        return b"png"

    async def get_screen_size(self):
        return {"width": 800, "height": 600}

    async def run_command(self, command: str, timeout=None):
        self.commands.append(command)
        self.command_calls.append({"command": command, "timeout": timeout})
        if isinstance(self.stdout, list):
            stdout = self.stdout.pop(0) if self.stdout else ""
        else:
            stdout = self.stdout
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    async def hotkey(self, *keys: str):
        self.hotkeys.append(tuple(keys))

    async def type_text(self, text: str):
        self.typed_text.append(text)


class _FakeComputer:
    def __init__(
        self,
        stdout: str = "stdout",
        files: dict[str, bytes] | None = None,
    ):
        self.interface = _FakeInterface(stdout=stdout, files=files)


@pytest.mark.asyncio
async def test_scalecua_recreation_devilsgarden_uses_local_cdp_parser(monkeypatch, tmp_path):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))

    async def fake_cdp_pages(computer):
        return [
            {
                "type": "page",
                "title": "Devil's Garden Campground",
                "url": "https://www.recreation.gov/camping/campgrounds/232449",
            }
        ]

    monkeypatch.setattr(scalecua_verify, "_get_cdp_pages", fake_cdp_pages)

    async def fake_page_info(computer, target_url):
        return {
            "title": "Devil's Garden Campground",
            "url": target_url,
            "content": """
                <html>
                  <head><title>Devil's Garden Campground</title></head>
                  <body>
                    <h1>Devil's Garden</h1>
                    <div class="camp-sortable-column-header">Site</div>
                    <div class="camp-sortable-column-header">Date</div>
                    <div class="campsite-row">Available Jul 20, 2026</div>
                  </body>
                </html>
            """,
        }

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_page_info_via_cdp", fake_page_info)
    monkeypatch.setattr(
        scalecua_verify.judges,
        "resolve_getter",
        lambda result_type, runtime_split: (_ for _ in ()).throw(
            AssertionError("recreation Devil's Garden should use local CDP parser")
        ),
    )

    out = await scalecua_verify._get_result(
        env,
        {
            "type": "recreation_devilsgarden_html__fa1e76c31141f93d38de11c4bb8239cf",
            "selector": "class",
            "class": "camp-sortable-column-header",
            "order": "2",
        },
        str(tmp_path),
        "train",
    )

    assert out["location_verified"] is True
    assert out["reservation_table_present"] is True
    assert out["has_availability_data"] is True
    assert out["reservation_dates"] == ["Jul 20, 2026"]
    assert out["dates_sorted"] is True
    assert out["earliest_reservation_identified"] is True
    assert out["url"] == "https://www.recreation.gov/camping/campgrounds/232449"


@pytest.mark.asyncio
@pytest.mark.skipif(not _overlays_ready(), reason="lite.scalecua judge overlays not imported")
async def test_scalecua_recreation_devilsgarden_detects_current_grid_markup(monkeypatch, tmp_path):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))

    async def fake_cdp_pages(computer):
        return [
            {
                "type": "page",
                "title": "Devils Garden Campground",
                "url": "https://www.recreation.gov/camping/campgrounds/234059",
            }
        ]

    monkeypatch.setattr(scalecua_verify, "_get_cdp_pages", fake_cdp_pages)

    async def fake_page_info(computer, target_url):
        return {
            "title": "Devils Garden Campground, Arches National Park - Recreation.gov",
            "url": target_url,
            "content": """
                <html>
                  <head>
                    <title>Devils Garden Campground, Arches National Park</title>
                  </head>
                  <body>
                    <h1>Devils Garden Campground</h1>
                    <nav>Campsite List Seasons & Fees Rules & Cancellations</nav>
                    <button>Next Available</button>
                    <section class="current-recreation-grid">
                      <div>Sites</div><div>Loop</div>
                      <div>JULY</div>
                      <div>TUE 14</div><div>WED 15</div><div>THU 16</div>
                      <div>CANYON WREN</div><div>Devils Gar...</div>
                      <span>R</span><span>R</span><span>A</span>
                      <div>JUNIPER BASIN</div><div>Devils Gar...</div>
                      <span>R</span><span>A</span><span>A</span>
                    </section>
                  </body>
                </html>
            """,
        }

    monkeypatch.setattr(
        scalecua_verify.base_runner,
        "_get_page_info_via_cdp",
        fake_page_info,
    )

    out = await scalecua_verify._get_result(
        env,
        {
            "type": "recreation_devilsgarden_html__fa1e76c31141f93d38de11c4bb8239cf",
            "selector": "class",
            "class": "camp-sortable-column-header",
            "order": "2",
        },
        str(tmp_path),
        "train",
    )

    assert out["location_verified"] is True
    assert out["reservation_table_present"] is True
    assert out["has_availability_data"] is True
    assert out["reservation_dates"] == ["TUE 14", "WED 15", "THU 16"]
    assert out["dates_sorted"] is True
    assert out["earliest_reservation_identified"] is True
    metric = judges.resolve_metric(
        "check_recreation_html_element__fa1e76c31141f93d38de11c4bb8239cf",
        "train",
    )
    assert (
        metric(
            out,
            {
                "location": "Devil's Garden",
                "location_verified": True,
                "reservation_table_present": True,
                "has_availability_data": True,
            },
        )
        == 1.0
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not _overlays_ready(), reason="lite.scalecua judge overlays not imported")
async def test_scalecua_recreation_devilsgarden_falls_back_to_accessibility_grid(
    monkeypatch, tmp_path
):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))

    async def fake_cdp_pages(computer):
        return [
            {
                "type": "page",
                "title": "Devils Garden Campground",
                "url": "https://www.recreation.gov/camping/campgrounds/234059",
            }
        ]

    async def fake_page_info(computer, target_url):
        return {
            "title": "Devils Garden Campground, Arches National Park - Recreation.gov",
            "url": target_url,
            "content": """
                <html>
                  <head><title>Devils Garden Campground</title></head>
                  <body>
                    <h1>Devils Garden Campground</h1>
                    <button>Next Available</button>
                  </body>
                </html>
            """,
        }

    async def fake_at_xml(computer):
        return """
            <desktop-frame>
              <application name="Google Chrome">
                <frame name="Devils Garden Campground">
                  <document-frame name="Devils Garden Campground">
                    <section name="Campsite List">
                      <label>Sites</label>
                      <label>Loop</label>
                      <label>Next Available</label>
                      <label>JULY</label>
                      <label>TUE 14</label>
                      <label>WED 15</label>
                      <label>THU 16</label>
                      <label>CANYON WREN</label>
                      <label>R</label>
                      <label>A</label>
                      <label>R</label>
                      <label>JUNIPER BASIN</label>
                      <label>R</label>
                      <label>R</label>
                      <label>A</label>
                    </section>
                  </document-frame>
                </frame>
              </application>
            </desktop-frame>
        """

    monkeypatch.setattr(scalecua_verify, "_get_cdp_pages", fake_cdp_pages)
    monkeypatch.setattr(
        scalecua_verify.base_runner,
        "_get_page_info_via_cdp",
        fake_page_info,
    )
    monkeypatch.setattr(scalecua_verify.base_runner, "_get_at_xml", fake_at_xml)

    out = await scalecua_verify._get_result(
        env,
        {
            "type": "recreation_devilsgarden_html__fa1e76c31141f93d38de11c4bb8239cf",
            "selector": "class",
            "class": "camp-sortable-column-header",
            "order": "2",
        },
        str(tmp_path),
        "train",
    )

    assert out["location_verified"] is True
    assert out["reservation_table_present"] is True
    assert out["has_availability_data"] is True
    assert out["reservation_dates"] == ["TUE 14", "WED 15", "THU 16"]
    assert out["dates_sorted"] is True
    assert out["earliest_reservation_identified"] is True
    metric = judges.resolve_metric(
        "check_recreation_html_element__fa1e76c31141f93d38de11c4bb8239cf",
        "train",
    )
    assert (
        metric(
            out,
            {
                "location": "Devil's Garden",
                "location_verified": True,
                "reservation_table_present": True,
                "has_availability_data": True,
            },
        )
        == 1.0
    )


@pytest.mark.asyncio
async def test_scalecua_clipboard_command_uses_backend_fallback_when_xsel_empty(
    monkeypatch, tmp_path
):
    clipboard = "/home/user/Data2/List2/report.pdf"
    encoded = base64.b64encode(clipboard.encode("utf-8")).decode("ascii")
    computer = _FakeComputer(stdout=f"__SCALECUA_CLIPBOARD_B64__{encoded}\n")
    env = judges.make_eval_env(computer, str(tmp_path))

    async def base_get_result(*args, **kwargs):
        return ""

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_result", base_get_result)

    out = await scalecua_verify._get_result(
        env,
        {
            "type": "vm_command_line",
            "command": "xsel --clipboard --output",
            "shell": "true",
        },
        str(tmp_path),
        "train",
    )

    assert out == clipboard
    assert any("__SCALECUA_CLIPBOARD_B64__" in cmd for cmd in computer.interface.commands)


@pytest.mark.asyncio
async def test_scalecua_repairs_generated_clipboard_dict_result(tmp_path):
    clipboard = "/home/user/Data3/List3/secret.docx"
    encoded = base64.b64encode(clipboard.encode("utf-8")).decode("ascii")
    computer = _FakeComputer(stdout=f"__SCALECUA_CLIPBOARD_B64__{encoded}\n")
    env = judges.make_eval_env(computer, str(tmp_path))

    out = await scalecua_verify._repair_clipboard_result(
        env,
        {"type": "clipboard_content__241466d2f7566981ed09035f56716bb4"},
        {"clipboard": "", "error": ""},
    )

    assert out == {"clipboard": clipboard, "error": ""}


def test_scalecua_unwraps_generated_pptx_shape_text_value_rule():
    expected = scalecua_verify._normalize_scalecua_expected_rules(
        "check_pptx_shape_text__8b4cb395", {"value": "Game Instructions"}
    )
    untouched = scalecua_verify._normalize_scalecua_expected_rules(
        "check_other_metric", {"value": "Game Instructions"}
    )

    assert expected == "Game Instructions"
    assert untouched == {"value": "Game Instructions"}


def test_scalecua_url_pattern_metric_accepts_current_live_site_aliases():
    metric = judges.resolve_metric("is_expected_url_pattern_match", "train")

    assert (
        metric(
            "https://dmv.virginia.gov/vehicles/registration/exemp-disc-chart",
            {"expected": ["^https://(www\\.)?dmv\\.virginia\\.gov/vehicles/.*fees"]},
        )
        == 1.0
    )
    assert (
        metric(
            {"url": "https://www.flightaware.com/live/airport/delays", "title": "Airport Delays"},
            {"expected": ["https://www\\.flightaware\\.com/miserymap/"]},
        )
        == 1.0
    )
    assert (
        metric(
            "https://mileageplustravel.united.com/explore/car-rentals",
            {"expected": ["united.com/en/us/book/cars"]},
        )
        == 1.0
    )
    assert (
        metric(
            "https://evilunited.com/explore/car-rentals",
            {"expected": ["united.com/en/us/book/cars"]},
        )
        == 0.0
    )
    assert (
        metric(
            {
                "url": "https://www.united.com/en/us/fly/travel/airport/flight-status",
                "title": "United Airlines",
                "content": "",
            },
            {"expected": ["united.com/en/us/.*flight.*status"]},
        )
        == 1.0
    )
    assert (
        metric(
            {
                "url": "https://www.united.com/en/us",
                "title": "United Airlines",
                "content": "Book Flight status Check in My trips",
            },
            {"expected": ["united.com/en/us/.*flight.*status"]},
        )
        == 0.0
    )
    assert (
        metric(
            {
                "url": "https://www.united.com/en/us/account/join-now/about-me",
                "title": "Create account",
                "content": "Start your MileagePlus adventure today.",
            },
            {"expected": ["united.com", "mileageplus"]},
        )
        == 1.0
    )
    assert (
        metric(
            {
                "url": "https://evilunited.com/en/us/account/join-now/about-me",
                "title": "Create account",
                "content": "MileagePlus",
            },
            {"expected": ["united.com", "mileageplus"]},
        )
        == 0.0
    )
    assert (
        metric(
            "https://dmv.virginia.gov/licenses-ids/real-id",
            {"expected": ["^https://(www\\.)?dmv\\.virginia\\.gov/vehicles/.*fees"]},
        )
        == 0.0
    )


@pytest.mark.skipif(not _overlays_ready(), reason="lite.scalecua judge overlays not imported")
def test_scalecua_recreation_loaded_override_accepts_current_campground_page():
    result = judges._recreation_loaded_result_from_page_state(
        "https://www.recreation.gov/camping/campgrounds/232447",
        "Upper Pines Campground, Yosemite National Park - Recreation.gov",
        ("Upper Pines Campground Campsites Reserve Availability Amenities Overview Recreation.gov"),
        '<button aria-label="Search campgrounds">Search</button>',
    )
    metric = judges.resolve_metric(
        "check_recreation_loaded__df352c0b3200dce606c52356619c2699",
        "train",
    )

    assert result["is_valid"] is True
    assert result["checks_passed"] >= 3
    assert metric(result, {"has_elements": True}) == 1.0


@pytest.mark.skipif(not _overlays_ready(), reason="lite.scalecua judge overlays not imported")
def test_scalecua_recreation_loaded_override_rejects_non_recreation_page():
    result = judges._recreation_loaded_result_from_page_state(
        "https://example.com/camping/campgrounds/232447",
        "Example campground",
        "Reserve campsite availability",
    )
    metric = judges.resolve_metric(
        "check_recreation_loaded__df352c0b3200dce606c52356619c2699",
        "train",
    )

    assert result["is_recreation_domain"] is False
    assert metric(result, {"has_elements": True}) == 0.0


@pytest.mark.asyncio
async def test_scalecua_recreation_url_check_uses_active_access_tree_url(
    monkeypatch,
    tmp_path,
):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))

    async def fake_at_xml(computer):
        return "<xml/>"

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_at_xml", fake_at_xml)
    monkeypatch.setattr(
        scalecua_verify.base_runner,
        "_extract_address_bar_url",
        lambda at_xml: "www.recreation.gov/camping/campgrounds/232449",
    )

    out = await scalecua_verify._get_result(
        env,
        {"type": "recreation_url_check__6dc3893f96ccd943c500af1756962de6"},
        str(tmp_path),
        "train",
    )

    assert out == {"url": "https://www.recreation.gov/camping/campgrounds/232449"}
    assert not env.computer.interface.commands


def test_scalecua_google_flights_text_fallback_parses_selected_date():
    parsed = scalecua_verify._parse_google_flights_context_text(
        "Travel on Jul 18 for $105 Change dates Track prices Jul 17 Any dates "
        "Other flights 6:55 AM - 8:15 AM 1 hr 20 min Nonstop LAX-SFO $155",
        now=datetime(2026, 7, 16, 12, 0, 0),
    )

    assert parsed == {
        "start": "LAX",
        "end": "SFO",
        "time": "Fri, Jul 17, 2026",
    }


@pytest.mark.asyncio
async def test_scalecua_google_flights_html_parse_repairs_empty_class_result(monkeypatch, tmp_path):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))

    async def fake_inner_text(computer, active_url):
        assert active_url == "https://www.google.com/travel/flights/search"
        return "Track prices Jul 17 Any dates Other flights 7:15 AM - 8:39 AM Nonstop LAX-SFO $155"

    monkeypatch.setattr(scalecua_verify, "_get_cdp_page_inner_text", fake_inner_text)
    monkeypatch.setattr(
        scalecua_verify,
        "_parse_google_flights_context_text",
        lambda text, now=None: {"start": "LAX", "end": "SFO", "time": "Fri, Jul 17, 2026"},
    )

    out = await scalecua_verify._repair_google_flights_html_parse(
        env,
        "https://www.google.com/travel/flights/search",
        {
            "type": "active_tab_html_parse",
            "category": "class",
            "class_multiObject_child": {
                "mach-flight-context-info__wrapper__info--separator": {
                    "0": "start",
                    "1": "end",
                },
            },
            "class_singleObject": {
                "mach-flight-context-info__wrapper--date": "time",
            },
        },
        {"start": "", "end": "", "time": ""},
    )

    assert out == {"start": "LAX", "end": "SFO", "time": "Fri, Jul 17, 2026"}


@pytest.mark.asyncio
async def test_scalecua_google_flights_date_uses_vm_clock_not_host(monkeypatch, tmp_path):
    # VM wall clock is Jan 2, 2027 (via `date +...`); agent selected "Dec 31".
    # Grader must anchor on the VM year (2027), NOT the host clock.
    env = judges.make_eval_env(_FakeComputer(stdout="2027-01-02T09:00:00+0000"), str(tmp_path))

    async def fake_inner_text(computer, active_url):
        return "Track prices Dec 31 Any dates Other flights 7:15 AM - 8:39 AM Nonstop LAX-SFO $155"

    monkeypatch.setattr(scalecua_verify, "_get_cdp_page_inner_text", fake_inner_text)

    out = await scalecua_verify._repair_google_flights_html_parse(
        env,
        "https://www.google.com/travel/flights/search",
        {
            "type": "active_tab_html_parse",
            "category": "class",
            "class_multiObject_child": {
                "mach-flight-context-info__wrapper__info--separator": {
                    "0": "start",
                    "1": "end",
                },
            },
            "class_singleObject": {
                "mach-flight-context-info__wrapper--date": "time",
            },
        },
        {"start": "", "end": "", "time": ""},
    )

    # VM clock => 2027; host clock (unfixed) would give ...2026 -> FN.
    assert out == {"start": "LAX", "end": "SFO", "time": "Fri, Dec 31, 2027"}


@pytest.mark.asyncio
async def test_scalecua_google_flights_date_host_fallback_on_vm_read_failure(monkeypatch, tmp_path):
    # VM `date` read yields nothing => _get_vm_now returns None => host fallback,
    # byte-identical to prior behavior (zero regression / neg control).
    env = judges.make_eval_env(_FakeComputer(stdout=""), str(tmp_path))

    async def fake_inner_text(computer, active_url):
        return "Track prices Jul 17 Any dates Other flights 7:15 AM - 8:39 AM Nonstop LAX-SFO $155"

    monkeypatch.setattr(scalecua_verify, "_get_cdp_page_inner_text", fake_inner_text)
    monkeypatch.setattr(
        scalecua_verify,
        "datetime",
        type(
            "D",
            (datetime,),
            {"now": classmethod(lambda cls, tz=None: datetime(2026, 7, 16, 12, 0, 0))},
        ),
    )

    out = await scalecua_verify._repair_google_flights_html_parse(
        env,
        "https://www.google.com/travel/flights/search",
        {
            "type": "active_tab_html_parse",
            "category": "class",
            "class_multiObject_child": {
                "mach-flight-context-info__wrapper__info--separator": {"0": "start", "1": "end"},
            },
            "class_singleObject": {"mach-flight-context-info__wrapper--date": "time"},
        },
        {"start": "", "end": "", "time": ""},
    )

    assert out == {"start": "LAX", "end": "SFO", "time": "Fri, Jul 17, 2026"}


@pytest.mark.asyncio
async def test_scalecua_vlc_playing_info_uses_official_password(tmp_path):
    env = judges.make_eval_env(
        _FakeComputer(stdout=["", "<root><state>playing</state></root>"]),
        str(tmp_path),
    )

    out = await scalecua_verify._get_result(
        env,
        {"type": "vlc_playing_info", "dest": "status.xml"},
        str(tmp_path),
        "train",
    )

    assert out == str(tmp_path / "status.xml")
    assert (tmp_path / "status.xml").read_text() == "<root><state>playing</state></root>"
    assert env.computer.interface.commands == [
        "curl -s http://localhost:8080/requests/status.xml",
        "curl -s --user :password http://localhost:8080/requests/status.xml",
    ]


@pytest.mark.asyncio
async def test_scalecua_vlc_playing_info_keeps_vlc_password_fallback(tmp_path):
    env = judges.make_eval_env(
        _FakeComputer(stdout=["", "", "<root><state>playing</state></root>"]),
        str(tmp_path),
    )

    out = await scalecua_verify._get_result(
        env,
        {"type": "vlc_playing_info", "dest": "status.xml"},
        str(tmp_path),
        "train",
    )

    assert out == str(tmp_path / "status.xml")
    assert env.computer.interface.commands == [
        "curl -s http://localhost:8080/requests/status.xml",
        "curl -s --user :password http://localhost:8080/requests/status.xml",
        "curl -s --user :vlc http://localhost:8080/requests/status.xml",
    ]
