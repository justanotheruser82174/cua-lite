"""Static contracts for the lite.cuagym image."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lite.gym.envs.lite.cuagym import main as M
from lite.gym.errors import EnvDepsMissingError

_DOCKERFILE = (
    Path(__file__).resolve().parents[5]  # tests/gym/envs/lite/cuagym/<file> -> repo root
    / "lite"
    / "gym"
    / "envs"
    / "lite"
    / "cuagym"
    / "docker"
    / "Dockerfile"
)
_INSTALL_SCRIPT = _DOCKERFILE.parent.parent / "scripts" / "install.sh"
_BROWSER_SCRIPTS = _DOCKERFILE.parent.parent / "src" / "browser" / "scripts.py"
_CLEANUP_SCRIPT = _DOCKERFILE.parent.parent / "scripts" / "cleanup.sh"


def _require_fresh_task_cache() -> None:
    try:
        M._register_tasks()
    except EnvDepsMissingError as exc:
        pytest.skip(str(exc))


def test_web_mocks_are_readable_and_writable_by_the_task_user() -> None:
    dockerfile = _DOCKERFILE.read_text()
    install = _INSTALL_SCRIPT.read_text()

    assert "COPY --chmod=0644 mocks-package.json /opt/mocks/package.json" in dockerfile
    assert "RUN chmod 0755 /opt/mocks" in dockerfile
    assert "COPY --chown=user:user mocks/ /opt/mocks/" in dockerfile
    assert 'chmod -R u+rwX,go+rX "$CTX_DIR/mocks"' in install


def test_mock_dist_is_rebuilt_and_optional_plugin_is_not_staged() -> None:
    install = _INSTALL_SCRIPT.read_text()
    build_apps = install.split("build_apps() {", 1)[1].split("\n}\n", 1)[0]

    assert "Github as XitHub" not in install
    assert "Trello as Xrello" not in install
    assert '[ -f "$d/dist/index.html" ] ||' not in build_apps
    assert "npm run build" in build_apps
    assert "'/secureMockApiPlugin.*from/d'" in install
    assert "'s/secureMockApiPlugin(),[[:space:]]*//g'" in install


def test_pinned_hub_contains_github_and_trello_alias_fixes() -> None:
    _require_fresh_task_cache()
    hub = _DOCKERFILE.parent.parent / ".cache" / "web" / "cua-gym-hub"
    assert (
        "Github as XitHub" in (hub / "websites/github_mock/src/components/Layout.jsx").read_text()
    )
    assert (
        "Trello as Xrello" in (hub / "websites/trello_mock/src/components/Navbar.jsx").read_text()
    )


def test_cuagym_installs_libreoffice_base_with_embedded_hsqldb_support() -> None:
    source = _DOCKERFILE.read_text()

    assert "libreoffice-base libreoffice-sdbc-hsqldb" in source


def test_cuagym_keeps_uno_document_dependencies_env_facing() -> None:
    source = _DOCKERFILE.read_text()

    assert "openpyxl==3.1.5 python-pptx==1.0.2 python-docx==1.1.2" in source
    assert "--python /usr/bin/python3 --system-site-packages" in source
    assert "/opt/env/uno-venv" in source
    assert "--python /opt/env/uno-venv/bin/python" in source
    assert "openpyxl==3.1.5 python-docx==1.1.2 odfpy==1.4.1" in source


def test_cuagym_shell_setup_pip_stays_env_facing() -> None:
    source = _DOCKERFILE.read_text()

    assert '(site_dir / "pip.py").write_text' in source
    assert '"--user"' in source
    assert '"--disable-pip-version-check"' in source
    assert '"--no-input"' in source
    assert '"--timeout"' in source
    assert (
        '_TRANSLATE = {"--no-cache-dir": "--no-cache", "--constraint": "--constraints"}'
    ) in source
    assert '_SELF_REQUIREMENTS = {"pip"}' in source
    assert '_TARGET_WITH_VALUE = {"-r", "--requirements", "-e", "--editable"}' in source
    assert '"--constraints"' in source
    assert "def _has_install_target(" in source
    assert "if _is_self_requirement(arg):" in source
    assert "if not _has_install_target(rest):" in source
    assert 'cmd = [_UV, "pip", command, "--python", _PYTHON, *rest]' in source
    assert 'for name in ("pip", "pip3", "pip3.12")' in source
    assert "exec /opt/env/venv/bin/python -m pip" in source
    assert '(site_dir / "python_docx.py").write_text' in source
    assert '(site_dir / "python_pptx.py").write_text' in source


def test_cuagym_keeps_agent_and_env_cli_surfaces_intentional() -> None:
    source = _DOCKERFILE.read_text()

    assert "openssh-client rustc cargo" in source
    assert "libimage-exiftool-perl" not in source
    assert "--divert /opt/env/bin/ssh-keygen /usr/bin/ssh-keygen" not in source


def test_cuagym_keeps_reward_only_xcftools_env_facing() -> None:
    source = _DOCKERFILE.read_text()

    assert (
        "COPY --from=xcftools-builder /tmp/xcftools/xcf2png /tmp/xcftools/xcf2pnm /opt/env/bin/"
    ) in source
    assert "/tmp/xcftools/xcf2pnm /usr/local/bin/" not in source


def test_cleanup_sanitizes_session_id_like_container_naming() -> None:
    source = _CLEANUP_SCRIPT.read_text()

    assert 'safe_session="${SESSION_ID//[^[:alnum:]_]/_}"' in source
    assert "name=${safe_session}-lite.cuagym-" in source


def test_cuagym_waits_for_main_desktop_readiness_contract() -> None:
    runtime = (_DOCKERFILE.parent.parent / "src" / "utils" / "display.py").read_text()

    assert "/tmp/gnome-ready" in runtime
    assert "/tmp/osworld-settings-ready" not in runtime


def test_cuagym_owns_vlc_first_run_state() -> None:
    source = _DOCKERFILE.read_text()

    assert "qt-privacy-ask=0" in source
    assert "metadata-network-access=0" in source
    assert "> /home/user/.config/vlc/vlcrc" in source


def test_cuagym_inherits_the_shared_desktop_theme() -> None:
    source = _DOCKERFILE.read_text()

    assert "cat >> /usr/local/bin/apply-settings.sh" in source
    assert "touch /tmp/lite-cuagym-ready" in source
    assert "gsettings set org.gnome.desktop.interface gtk-theme" not in source
    assert "gtk-application-prefer-dark-theme" not in source


def test_browser_runtime_rejects_unrendered_react_pages() -> None:
    dockerfile = _DOCKERFILE.read_text()
    install = _INSTALL_SCRIPT.read_text()
    runtime = _BROWSER_SCRIPTS.read_text()

    assert "websocket-client==1.9.0" in dockerfile
    assert ("COPY --chmod=0755 page_health.py /opt/env/bin/lite-cuagym-page-health") in dockerfile
    assert '"$DOCKER_DIR/page_health.py"' in install
    chrome_wrapper = (_DOCKERFILE.parent / "google-chrome").read_text()
    executable = "\n".join(
        line.strip()
        for line in chrome_wrapper.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    assert re.search(r"(?m)^exec\s+/opt/google/chrome/chrome\b", executable)
    assert not re.search(
        r"(?m)^exec\s+/(?:usr/local|opt/env)/bin/google-chrome\b",
        executable,
    )
    assert "--remote-debugging-port=9222" in chrome_wrapper
    assert "--remote-allow-origins=*" in chrome_wrapper
    assert "COPY --chmod=0755 google-chrome /usr/local/bin/google-chrome" in dockerfile
    assert "/opt/env/bin/lite-cuagym-page-health --timeout 20" in runtime
