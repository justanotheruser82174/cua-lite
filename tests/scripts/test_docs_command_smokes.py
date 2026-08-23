from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from lite.core.utils.filters import parse_filter
from lite.data.hf.download import _expand_alternations
from lite.infer.cli import make_infer_parser

ROOT = Path(__file__).resolve().parents[2]
DOC_PATHS = (
    ROOT / "docs" / "sft.md",
    ROOT / "docs" / "examples" / "rollout_to_hf.md",
)

_BASH_BLOCK_RE = re.compile(r"```bash\n(.*?)\n```", re.S)
_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


@dataclass(frozen=True)
class DocCommand:
    doc: Path
    block_idx: int
    command_idx: int
    text: str
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class PythonCommand:
    command: DocCommand
    kind: str
    target: str
    args: tuple[str, ...]


def _logical_commands(block: str) -> list[str]:
    commands: list[str] = []
    pending: list[str] = []
    for raw in block.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        continued = line.endswith("\\")
        if continued:
            line = line[:-1].rstrip()
        pending.append(line.strip())
        if not continued:
            commands.append(" ".join(pending))
            pending = []
    if pending:
        commands.append(" ".join(pending))
    return commands


def _doc_commands() -> list[DocCommand]:
    commands: list[DocCommand] = []
    for doc in DOC_PATHS:
        blocks = _BASH_BLOCK_RE.findall(doc.read_text())
        assert blocks, doc
        for block_idx, block in enumerate(blocks):
            for command_idx, text in enumerate(_logical_commands(block)):
                tokens = shlex.split(text, comments=True)
                if tokens and tokens[-1] == "&":
                    tokens = tokens[:-1]
                if tokens:
                    commands.append(
                        DocCommand(
                            doc=doc,
                            block_idx=block_idx,
                            command_idx=command_idx,
                            text=text,
                            tokens=tuple(tokens),
                        )
                    )
    return commands


def _strip_env_prefix(tokens: tuple[str, ...]) -> tuple[dict[str, str], tuple[str, ...]]:
    env: dict[str, str] = {}
    idx = 0
    while idx < len(tokens) and _ENV_ASSIGNMENT_RE.match(tokens[idx]):
        name, value = tokens[idx].split("=", 1)
        env[name] = value
        idx += 1
    return env, tokens[idx:]


def _python_commands(commands: list[DocCommand]) -> list[PythonCommand]:
    out: list[PythonCommand] = []
    for command in commands:
        _env, tokens = _strip_env_prefix(command.tokens)
        if tokens[:3] != ("uv", "run", "python"):
            continue
        if tokens[3:4] == ("-m",):
            out.append(PythonCommand(command, "module", tokens[4], tokens[5:]))
        else:
            out.append(PythonCommand(command, "script", tokens[3], tokens[4:]))
    return out


def _flags(args: tuple[str, ...]) -> set[str]:
    flags: set[str] = set()
    for arg in args:
        if arg.startswith("--"):
            flags.add(arg.split("=", 1)[0])
        elif re.fullmatch(r"-[A-Za-z]", arg):
            flags.add(arg)
    return flags


def _arg_value(args: tuple[str, ...], flag: str) -> str | None:
    for idx, arg in enumerate(args):
        if arg == flag and idx + 1 < len(args):
            return args[idx + 1]
    return None


def _arg_values(args: tuple[str, ...], flag: str) -> list[str]:
    values: list[str] = []
    for idx, arg in enumerate(args):
        if arg == flag and idx + 1 < len(args):
            values.append(args[idx + 1])
    return values


def _help_text(kind: str, target: str) -> str:
    if kind == "module":
        argv = [sys.executable, "-m", target, "--help"]
    else:
        argv = [sys.executable, str(ROOT / target), "--help"]
    proc = subprocess.run(
        argv,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    return proc.stdout


def _write(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(0o755)


def _stage_fake_sft_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    train_dir = root / "scripts" / "train"
    utils_dir = train_dir / "utils"
    model_dir = train_dir / "models"
    bin_dir = root / "bin"
    train_dir.mkdir(parents=True)
    utils_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    bin_dir.mkdir(parents=True)
    (root / "slime").mkdir()

    (train_dir / "run_sft.sh").write_text((ROOT / "scripts" / "train" / "run_sft.sh").read_text())
    (utils_dir / "runtime_env.sh").write_text(
        (ROOT / "scripts" / "train" / "utils" / "runtime_env.sh").read_text()
    )
    _write(utils_dir / "cleanup.sh", 'echo cleanup >> "$FAKE_TRACE"\n')
    _write(utils_dir / "nvlink.sh", 'echo nvlink >> "$FAKE_TRACE"\nHAS_NVLINK=0\n')
    _write(
        utils_dir / "ckpt_args.sh",
        """
echo ckpt_args >> "$FAKE_TRACE"
CKPT_ARGS=(--fake-ckpt "$SAVE_DIR" "$SAVE_HF_DIR" "$HF_CKPT")
""".lstrip(),
    )
    _write(
        utils_dir / "models.sh",
        """
echo models >> "$FAKE_TRACE"
case "$MODEL_ID" in
  Qwen/Qwen2.5-VL-3B-Instruct|Qwen/Qwen3-VL-2B-Instruct)
    MODEL_ARGS_FILE=fake_model
    MODEL_FAMILY=qwen3_vl
    MODEL_SLUG="${MODEL_ID//\\//_}"
    ;;
  *)
    echo "unexpected model $MODEL_ID" >&2
    exit 1
    ;;
esac
resolve_tp() {
  local tp=${1:-1} n=$2
  if [ "$(( n % tp ))" -ne 0 ]; then
    echo "bad TP" >&2
    exit 1
  fi
  echo "$tp"
}
""".lstrip(),
    )
    _write(model_dir / "fake_model.sh", 'MODEL_ARGS=(--fake-model-arg "$MODEL_ID")\n')
    _write(
        utils_dir / "ray.sh",
        'start_ray() { echo start_ray >> "$FAKE_TRACE"; RAY_PORT=18080; }\n',
    )
    _write(
        bin_dir / "python",
        f"""
#!/usr/bin/env bash
if [ "$1" = "-c" ]; then
  echo python_preflight >> "$FAKE_TRACE"
  exit 0
fi
exec {shlex.quote(sys.executable)} "$@"
""".lstrip(),
        executable=True,
    )
    _write(
        bin_dir / "mkdir",
        """
#!/usr/bin/env bash
if [ "$1" = "-p" ] && [ "$2" = "/root/models" ]; then
  exit 0
fi
exec /usr/bin/mkdir "$@"
""".lstrip(),
        executable=True,
    )
    _write(
        bin_dir / "ray",
        f"""
#!/usr/bin/env bash
{shlex.quote(sys.executable)} - "$FAKE_RAY_CAPTURE" "$@" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))
PY
exit "${{FAKE_RAY_EXIT:-44}}"
""".lstrip(),
        executable=True,
    )
    return root


def _run_sft_commands(commands: list[DocCommand]) -> list[tuple[DocCommand, dict[str, str]]]:
    out: list[tuple[DocCommand, dict[str, str]]] = []
    for command in commands:
        env, tokens = _strip_env_prefix(command.tokens)
        if tokens == ("bash", "/workspaces/cua-lite/scripts/train/run_sft.sh"):
            out.append((command, env))
    return out


def test_documented_python_commands_expose_their_flags_in_help() -> None:
    commands = _doc_commands()
    python_commands = _python_commands(commands)
    assert python_commands
    assert {path.relative_to(ROOT).as_posix() for path in DOC_PATHS} == {
        command.doc.relative_to(ROOT).as_posix() for command in commands
    }

    flags_by_entrypoint: dict[tuple[str, str], set[str]] = {}
    for command in python_commands:
        flags_by_entrypoint.setdefault((command.kind, command.target), set()).update(
            _flags(command.args)
        )

    for (kind, target), flags in sorted(flags_by_entrypoint.items()):
        help_text = _help_text(kind, target)
        missing = sorted(flag for flag in flags if flag not in help_text)
        assert not missing, f"{target} help is missing documented flags: {missing}"


def test_documented_rollout_commands_parse_without_running_rollout() -> None:
    parser = make_infer_parser()
    rollout_commands = [
        command
        for command in _python_commands(_doc_commands())
        if command.target == "scripts/rollout.py"
    ]
    assert rollout_commands

    for command in rollout_commands:
        parsed = parser.parse_args(list(command.args))
        assert parsed.log_root
        assert parsed.model_id
        assert parsed.env_id


def test_documented_json_filters_and_config_paths_are_valid() -> None:
    commands = _python_commands(_doc_commands())
    checked_configs: set[Path] = set()
    checked_json = 0
    checked_filters = 0

    for command in commands:
        for flag in (
            "--sampling-kwargs",
            "--api-kwargs",
            "--engine-kwargs",
            "--agent-kwargs",
            "--env-kwargs",
        ):
            for raw in _arg_values(command.args, flag):
                parsed = json.loads(raw)
                assert isinstance(parsed, dict)
                checked_json += 1

        for flag in ("--config", "--config-path"):
            for raw in _arg_values(command.args, flag):
                path = ROOT / raw
                assert path.is_file(), f"{command.command.doc}: missing config {raw}"
                loaded = yaml.safe_load(path.read_text())
                assert isinstance(loaded, dict)
                checked_configs.add(path)

        for raw in _arg_values(command.args, "--filter"):
            keep = parse_filter(raw)
            result = keep(
                type(
                    "Metadata",
                    (),
                    {
                        "others": {
                            "complexity": 1.0,
                            "episode_return": 1.0,
                            "exclude_reason": None,
                        }
                    },
                )()
            )
            assert isinstance(result, bool)
            checked_filters += 1

    allow_patterns = [
        _arg_value(command.args, "--allow-patterns")
        for command in commands
        if command.target == "lite.data.hf.download"
    ]
    allow_patterns = [value for value in allow_patterns if value is not None]
    assert "(desktop|browser|mobile)/grounding.action/**" in allow_patterns
    assert _expand_alternations("(desktop|browser|mobile)/grounding.action/**") == [
        "desktop/grounding.action/**",
        "browser/grounding.action/**",
        "mobile/grounding.action/**",
    ]
    assert len(checked_configs) >= 5
    assert checked_json >= 3
    assert checked_filters >= 3


def test_documented_sft_train_commands_reach_fake_ray_boundary(tmp_path: Path) -> None:
    subprocess.run(
        ["bash", "-n", str(ROOT / "scripts" / "train" / "run_sft.sh")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )

    run_sft_commands = _run_sft_commands(_doc_commands())
    assert len(run_sft_commands) >= 3

    for idx, (_command, doc_env) in enumerate(run_sft_commands):
        root = _stage_fake_sft_repo(tmp_path / f"sft_{idx}")
        prompt = root / "prompt.parquet"
        prompt.write_text("fake")
        hf_ckpt = root / "hf_ckpt"
        hf_ckpt.mkdir()

        env = os.environ.copy()
        env.update(doc_env)
        env.update(
            {
                "PROMPT_DATA": str(prompt),
                "HF_CKPT": str(hf_ckpt),
                "SESSION_ID": f"docs-sft-smoke-{idx}",
                "FAKE_TRACE": str(root / "trace.log"),
                "FAKE_RAY_CAPTURE": str(root / "ray.args.json"),
                "FAKE_RAY_EXIT": "44",
                "PATH": f"{root / 'bin'}:{env['PATH']}",
            }
        )

        result = subprocess.run(
            ["bash", str(root / "scripts" / "train" / "run_sft.sh")],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
        )

        assert result.returncode == 44, result.stderr
        assert (root / "trace.log").read_text().splitlines() == [
            "cleanup",
            "models",
            "nvlink",
            "python_preflight",
            "ckpt_args",
            "start_ray",
        ]
        ray_args = json.loads((root / "ray.args.json").read_text())
        runtime_arg = next(arg for arg in ray_args if arg.startswith("--runtime-env-json="))
        runtime_env = json.loads(runtime_arg.removeprefix("--runtime-env-json="))["env_vars"]

        assert runtime_env["SESSION_ID"] == f"docs-sft-smoke-{idx}"
        assert runtime_env["PYTHONPATH"] == f"/root/Megatron-LM/:{root}:{root / 'slime'}"
        assert (
            ray_args[ray_args.index("--actor-num-gpus-per-node") + 1] == doc_env["NUM_TRAIN_GPUS"]
        )
        assert ray_args[ray_args.index("--prompt-data") + 1] == str(prompt)
        assert ray_args[ray_args.index("--num-epoch") + 1] == doc_env.get("NUM_EPOCH", "2")
        assert ray_args[ray_args.index("--global-batch-size") + 1] == doc_env.get(
            "GLOBAL_BATCH_SIZE", "4"
        )
        assert "--fake-model-arg" in ray_args
