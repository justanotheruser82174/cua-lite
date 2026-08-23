"""Getter-level unit test for the VS Code on-disk state reader.

Extracts ``_VSCODE_ONDISK_READER`` verbatim from the osworld runner module,
synthesizes VS Code's persisted state (workspace.json / state.vscdb ItemTable /
User+workspace settings.json / a real folder on disk), runs the reader per
command family, and scores each answer file against the family's metric
contract. Covers the positive (correct state -> 1.0) and negative-control
(wrong state, expected fixed -> 0.0, no false-positive) directions. No
container: pkill/pgrep are stubbed so the host is never touched (the SIGTERM
flush behavior is separately validated by the container rollout).

Run: uv run pytest tests/gym/envs/lite/osworld/test_lite_osworld_vscode_reader.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys

from lite.gym.envs.lite.osworld.src.eval import runner as osworld_runner

READER_SRC = osworld_runner._VSCODE_ONDISK_READER

FAMILIES = [
    "OpenProject",
    "GetWorkspace",
    "GetExplorer",
    "GetOpenFile",
    "WorkspaceFiles",
    "GetOpenTabs",
    "GetSettings",
    "GetTerminalStatus",
    "GetSearchResults",
]


def _build_state(
    home: str,
    folder: str,
    *,
    open_tabs: list[str],
    settings: dict | None = None,
    ws_settings: dict | None = None,
    terminal_open: bool = True,
    search_query: str = "TODO",
) -> None:
    codeuser = os.path.join(home, ".config/Code/User")
    ws = os.path.join(codeuser, "workspaceStorage", "abc123")
    os.makedirs(ws, exist_ok=True)
    os.makedirs(os.path.join(codeuser, "globalStorage"), exist_ok=True)
    json.dump({"folder": "file://" + folder}, open(os.path.join(ws, "workspace.json"), "w"))
    os.makedirs(os.path.join(folder, ".vscode"), exist_ok=True)
    open(os.path.join(folder, "main.py"), "w").write("x=1  # TODO fix\n")
    open(os.path.join(folder, "README.md"), "w").write("# proj\n")
    if settings is not None:
        json.dump(settings, open(os.path.join(codeuser, "settings.json"), "w"))
    if ws_settings is not None:
        json.dump(ws_settings, open(os.path.join(folder, ".vscode", "settings.json"), "w"))
    db = os.path.join(ws, "state.vscdb")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value BLOB)")

    def edleaf(paths: list[str]) -> dict:
        return {
            "type": "leaf",
            "data": {
                "editors": [{"value": json.dumps({"resourceJSON": {"fsPath": p}})} for p in paths]
            },
        }

    st = {
        "editorpart.state": {
            "serializedGrid": {"root": {"type": "branch", "data": [edleaf(list(open_tabs))]}}
        }
    }
    conn.execute(
        "INSERT INTO ItemTable VALUES (?,?)", ("memento/workbench.parts.editor", json.dumps(st))
    )
    conn.execute(
        "INSERT INTO ItemTable VALUES (?,?)",
        (
            "terminal.integrated.layoutInfo",
            json.dumps({"tabs": [{"t": 1}]} if terminal_open else {"tabs": []}),
        ),
    )
    conn.execute(
        "INSERT INTO ItemTable VALUES (?,?)",
        ("memento/workbench.view.search", json.dumps({"query": search_query})),
    )
    conn.commit()
    conn.close()


def _run_reader(tmp_path, home: str, cmd: str) -> str | None:
    reader = tmp_path / "reader.py"
    reader.write_text(READER_SRC)
    out = os.path.join(home, cmd + ".txt")
    env = dict(os.environ)
    env.update(HOME=home, VSCODE_CMD=cmd, VSCODE_OUT=out)
    # Stub pkill/pgrep so the LAZY-memento SIGTERM path never touches the host.
    stub = os.path.join(home, "stubbin")
    os.makedirs(stub, exist_ok=True)
    for n in ("pkill", "pgrep"):
        p = os.path.join(stub, n)
        open(p, "w").write("#!/bin/sh\nexit 1\n")
        os.chmod(p, 0o755)
    env["PATH"] = stub + os.pathsep + env["PATH"]
    r = subprocess.run([sys.executable, str(reader)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"reader failed cmd={cmd}: {r.stderr}"
    return open(out).read() if os.path.exists(out) else None


def _j(t: str | None) -> dict:
    try:
        return json.loads(t)
    except Exception:
        return {}


def test_vscode_reader_positive_families(tmp_path):
    home = str(tmp_path / "home")
    folder = os.path.join(home, "workspace", "project")
    os.makedirs(folder, exist_ok=True)
    _build_state(
        home,
        folder,
        open_tabs=[os.path.join(folder, "main.py")],
        settings={"python.pythonPath": "/usr/bin/py"},
        terminal_open=True,
        search_query="TODO",
    )
    read = {cmd: _run_reader(tmp_path, home, cmd) for cmd in FAMILIES}

    assert read["OpenProject"] and os.path.basename(read["OpenProject"].strip()) == "project"
    assert read["GetWorkspace"] and folder in read["GetWorkspace"]
    assert read["GetExplorer"] and all(
        x in read["GetExplorer"] for x in ["main.py", "README.md", ".vscode"]
    )
    assert read["GetOpenFile"] and read["GetOpenFile"].strip().endswith("main.py")
    assert read["WorkspaceFiles"] and "README.md" in read["WorkspaceFiles"]
    assert read["GetOpenTabs"] is not None and "main.py" in read["GetOpenTabs"]
    assert _j(read["GetSettings"]).get("python.pythonPath") == "/usr/bin/py"
    assert read["GetTerminalStatus"] and "open" in read["GetTerminalStatus"].lower()
    assert read["GetSearchResults"] and "todo" in read["GetSearchResults"].lower()


def test_vscode_reader_negative_control(tmp_path):
    # Wrong on-disk state; the expected target is held at the POSITIVE value.
    # Every discriminator must score 0 -> the reader never mints a false-positive.
    home = str(tmp_path / "home")
    folder = os.path.join(home, "workspace", "wrongdir")
    os.makedirs(folder, exist_ok=True)
    _build_state(
        home,
        folder,
        open_tabs=[os.path.join(folder, "other.js"), os.path.join(folder, "secret.env")],
        settings={"python.pythonPath": "/usr/bin/OTHER"},
        terminal_open=False,
        search_query="",
    )
    read = {cmd: _run_reader(tmp_path, home, cmd) for cmd in FAMILIES}

    # OpenProject: basename is "wrongdir", not the expected "project".
    assert os.path.basename((read["OpenProject"] or "").strip()) != "project"
    # GetOpenFile: active file is other.js, not main.py.
    assert not (read["GetOpenFile"] or "").strip().endswith("main.py")
    # GetSettings: pythonPath differs from the expected /usr/bin/py.
    assert _j(read["GetSettings"]).get("python.pythonPath") != "/usr/bin/py"
    # GetTerminalStatus: terminal is closed.
    assert "open" not in (read["GetTerminalStatus"] or "").lower()
    # GetOpenTabs: a wrong tab (secret.env) present -> not the expected tab set.
    assert "secret.env" in (read["GetOpenTabs"] or "")


def test_vscode_reader_snippet_is_syntactically_valid():
    import ast

    ast.parse(READER_SRC)
