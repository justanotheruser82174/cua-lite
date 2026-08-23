"""Capture the official OSWorld VM agent-facing surface manifest.

This is a one-time, KVM-required helper for the VM surface parity artifact. It boots the
private official osworld VM-in-Docker image, executes scrubbed-PATH probes inside
the guest via the guest `/execute` service, and writes `vm_surface.json`.

Example:
    uv run python devs/envs/lite.osworld/validate/vm_surface_capture.py \\
      --image cua-lite/osworld:mine \\
      --qcow2 "$HOME/.cache/cua-lite/immutable-store/osworld/Ubuntu.qcow2" \\
      --output devs/envs/lite.osworld/validate/vm_surface.json \\
      --hf-revision <pinned-hf-commit-sha> \\
      --sha256 <known-qcow2-sha256>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

AGENT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
DEFAULT_HF_REPO = "xlangai/ubuntu_osworld"
DEFAULT_SESSION_ID = "vm_surface_capture"
_HF_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


def agent_shell(command: str) -> str:
    """Run a guest command with the exact scrubbed PATH required by §8c(i)."""
    return f"env -i PATH={shlex.quote(AGENT_PATH)} bash -c {shlex.quote(command)}"


def _guest_execute(container_name: str, command: str, *, timeout: int = 180) -> dict[str, Any]:
    script = """
import json
import requests
import sys

payload = json.loads(sys.stdin.read())
response = requests.post("http://20.20.20.21:5000/execute", json=payload, timeout=150)
response.raise_for_status()
print(json.dumps(response.json()))
"""
    out = subprocess.run(
        ["docker", "exec", "-i", container_name, "/opt/venv/bin/python", "-c", script],
        input=json.dumps({"shell": True, "command": command}),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    data = json.loads(out.stdout)
    if data.get("status") != "success":
        raise RuntimeError(f"guest execute failed: {data}")
    return data


def _guest_stdout(container_name: str, command: str, *, timeout: int = 180) -> str:
    data = _guest_execute(container_name, command, timeout=timeout)
    if data.get("returncode") != 0:
        raise RuntimeError(
            f"guest command rc={data.get('returncode')}: {command}\n"
            f"stdout={data.get('output', '')}\nstderr={data.get('error', '')}"
        )
    return str(data.get("output") or "")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_capture_inputs(args: argparse.Namespace) -> None:
    if not args.qcow2.exists():
        raise SystemExit(f"qcow2 not found: {args.qcow2}")
    image_name = args.image.rsplit("/", 1)[-1]
    if args.image.endswith(":latest") or (":" not in image_name and "@" not in image_name):
        raise SystemExit("Use a private osworld image tag or digest, not :latest.")
    if not _HF_COMMIT_RE.fullmatch(args.hf_revision):
        raise SystemExit("--hf-revision must be a pinned 40-character commit SHA, not a branch.")
    if args.sha256 is not None and not _SHA256_RE.fullmatch(args.sha256):
        raise SystemExit("--sha256 must be a 64-character hex digest.")


def _python_probe_command() -> str:
    return """python3 - <<'PY'
import json
import os
import shutil
import subprocess
import sys

version = subprocess.run(["python3", "-VV"], capture_output=True, text=True, check=True).stdout.strip()
print(json.dumps({
    "executable_realpath": os.path.realpath(shutil.which("python3") or ""),
    "version_verbose": version,
    "has_opt_env_venv_sys_path": any("/opt/env/venv" in p for p in sys.path),
    "sys_path": sys.path,
}, sort_keys=True))
PY"""


def _parse_packages(output: str) -> list[dict[str, str]]:
    packages = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        name, version, arch = parts
        packages.append({"name": name, "version": version, "arch": arch})
    return packages


def capture_manifest(args: argparse.Namespace) -> dict[str, Any]:
    from lite.gym.envs.osworld.container import OSWorldContainerFactory

    qcow2 = args.qcow2.resolve()
    factory = OSWorldContainerFactory(
        qcow2_path=str(qcow2),
        image=args.image,
        session_id=args.session_id,
        task_id="vm-surface-capture",
        boot_timeout=args.boot_timeout,
    )
    container = factory.build()
    try:
        container.start()
        identity = _guest_stdout(
            container.name,
            agent_shell("id -u; whoami"),
            timeout=60,
        ).splitlines()
        commands = sorted(set(
            _guest_stdout(container.name, agent_shell("compgen -c"), timeout=120).splitlines()
        ))
        package_output = _guest_stdout(
            container.name,
            agent_shell("dpkg-query -W -f='${binary:Package}\\t${Version}\\t${Architecture}\\n' || true"),
            timeout=120,
        )
        python_surface = json.loads(
            _guest_stdout(container.name, agent_shell(_python_probe_command()), timeout=120)
        )
    finally:
        container.destroy()

    stat = qcow2.stat()
    sha = args.sha256
    if args.compute_sha256:
        started = time.time()
        sha = _sha256(qcow2)
        elapsed = time.time() - started
    else:
        elapsed = None

    return {
        "schema_version": 1,
        "capture": {
            "agent_path": AGENT_PATH,
            "method": "docker exec into osworld qemu container -> guest /execute",
            "guest_execute_uid": int(identity[0]) if identity and identity[0].isdigit() else None,
            "guest_execute_user": identity[1] if len(identity) > 1 else None,
            "image": args.image,
            "session_id": args.session_id,
        },
        "provenance": {
            "hf_repo": args.hf_repo,
            "hf_revision": args.hf_revision,
            "filename": qcow2.name,
            "path": str(qcow2),
            "size": stat.st_size,
            "sha256": sha,
            "sha256_elapsed_s": elapsed,
        },
        "commands": {
            "probe": agent_shell("compgen -c"),
            "names": commands,
        },
        "python": python_surface,
        "packages": {
            "informational_only": True,
            "items": _parse_packages(package_output),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="cua-lite/osworld:mine")
    parser.add_argument("--qcow2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID)
    parser.add_argument("--boot-timeout", type=float, default=300.0)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-revision", required=True, help="Pinned HF dataset commit SHA.")
    sha_group = parser.add_mutually_exclusive_group(required=True)
    sha_group.add_argument("--sha256", default=None, help="Known qcow2 sha256 to record.")
    sha_group.add_argument(
        "--compute-sha256",
        action="store_true",
        help="Hash the qcow2 before writing the manifest. This reads ~23 GiB.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_capture_inputs(args)

    manifest = capture_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
