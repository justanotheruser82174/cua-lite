"""Geo3K environment for geometry QA."""

from __future__ import annotations

import base64
import hashlib
import io
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from lite.core.metadata import LiteBaseMetadata, LiteGenericMetadata
from lite.core.tools.calls import RuntimeEnvAction
from lite.core.tools.results import make_tool_result
from lite.core.tools.schemas import tool_schema_name
from lite.gym.base import LiteBaseEnv
from lite.gym.registry import _clear_env_registration, register, registry
from lite.gym.services import BackendFamily, register_family
from lite.gym.types import EXECUTED_ACTIONS_INFO_KEY, LiteEnvObservation, LiteEnvStepResult
from lite.gym.utils.feedback.ingress import (
    prepare_env_tool_calls,
    standalone_tool_call_feedback,
)
from lite.utils.image import encode_png

ENV_ID = "geo3k"
TASK_ID = "unit_square_diagonal"
GEO3K_SOURCE_ENV = "GEO3K_SOURCE"


@dataclass(frozen=True)
class Geo3KTask:
    """One geometry QA task."""

    question: str
    answer: str
    image: bytes | None = None
    split: str = "train"


@dataclass(frozen=True)
class Geo3KSourceTask:
    """A task backed by one row in a Geo3K parquet."""

    source_path: str
    row_index: int
    source_fingerprint: str | None = None
    split: str = "train"


SMOKE_TASKS: dict[str, Geo3KTask] = {
    TASK_ID: Geo3KTask(
        question="A unit square has side length 1. What is the length of its diagonal?",
        answer="sqrt(2)",
    ),
    "triangle_area": Geo3KTask(
        question="A triangle has base 6 and height 4. What is its area?",
        answer="12",
    ),
    "circle_circumference": Geo3KTask(
        question="A circle has radius 3. What is its circumference in terms of pi?",
        answer="6*pi",
    ),
}


def _task_metadata(
    *, extra_tool_schemas: list[dict[str, Any]] | None = None
) -> LiteGenericMetadata:
    return LiteGenericMetadata(dims=(), extra_tool_schemas=extra_tool_schemas or [])


class Geo3KEnv(LiteBaseEnv):
    """Geometry QA env with optional response-feedback turns."""

    @classmethod
    def extra_tool_schemas(
        cls,
        selection: list[str] | None = None,
        **ctor_args: Any,
    ) -> list[dict[str, Any]]:
        schemas = super().extra_tool_schemas(selection, **ctor_args)
        unsupported = sorted(
            tool_schema_name(schema)
            for schema in schemas
            if tool_schema_name(schema) != "response"
        )
        if unsupported:
            raise ValueError(
                f"{cls.__name__} cannot execute extra_tools {unsupported}; "
                "only ['response'] is wired"
            )
        return schemas

    def __init__(
        self,
        task_id: str = TASK_ID,
        *,
        source_path: str | None = None,
        row_index: int | None = None,
        source_fingerprint: str | None = None,
        question: str | None = None,
        answer: str | None = None,
        image: bytes | None = None,
        max_turns: int = 1,
        extra_tools: list[str] | None = None,
    ) -> None:
        if not isinstance(max_turns, int) or isinstance(max_turns, bool):
            raise TypeError(f"Geo3KEnv max_turns must be an integer, got {max_turns!r}")
        if max_turns < 1:
            raise ValueError(f"Geo3KEnv max_turns must be >= 1, got {max_turns}")
        self._task_id = task_id
        self.max_turns = max_turns
        self._extra_tools = list(extra_tools or [])
        self._extra_tool_schemas = type(self).extra_tool_schemas(self._extra_tools)
        self._attempts = 0
        self._row: Geo3KTask | None = None
        if source_path is not None:
            if row_index is None:
                raise ValueError("Geo3KEnv source_path requires row_index")
            actual_fingerprint = geo3k_source_fingerprint(source_path)
            if (
                source_fingerprint is not None
                and source_fingerprint != actual_fingerprint
            ):
                raise ValueError(
                    "Geo3K source fingerprint mismatch: prompt data was exported "
                    f"from {source_fingerprint}, but GEO3K_SOURCE is "
                    f"{actual_fingerprint}"
                )
            self._source = Geo3KSourceTask(
                source_path=source_path,
                row_index=row_index,
                source_fingerprint=actual_fingerprint,
            )
            return
        if question is not None and answer is not None:
            self._source = None
            self._row = Geo3KTask(question=question, answer=answer, image=image)
            return
        if task_id not in SMOKE_TASKS:
            raise KeyError(f"unknown Geo3K task_id: {task_id!r}")
        self._source = None
        self._row = SMOKE_TASKS[task_id]

    def _runtime_metadata(self) -> LiteBaseMetadata:
        return _task_metadata(extra_tool_schemas=self._extra_tool_schemas)

    def _task(self) -> Geo3KTask:
        if self._row is None:
            assert self._source is not None
            self._row = _load_source_task(self._source)
        return self._row

    async def reset(self) -> LiteEnvObservation:
        self._attempts = 0
        task = self._task()
        return LiteEnvObservation(
            image=task.image,
            text=task.question,
            metadata={"env_id": ENV_ID, "task_id": self._task_id},
        )

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        if self._attempts >= self.max_turns:
            return LiteEnvStepResult(
                results=[
                    make_tool_result(None, error="task already finished"),
                ],
                reward=0.0,
                terminated=True,
                info={
                    EXECUTED_ACTIONS_INFO_KEY: [],
                    "attempt": self._attempts,
                    "max_turns": self.max_turns,
                },
            )

        task = self._task()
        metadata = self.metadata
        prepared, feedback = prepare_env_tool_calls(actions, metadata)
        results = [
            make_tool_result(call_id, error=item.message)
            for call_id, item in feedback.items()
        ]
        executed: list[dict[str, Any]] = []
        reward = 0.0
        self._attempts += 1
        response_seen = False
        for action, call_id in prepared:
            name = action["name"]
            args = action["arguments"]
            executed.append({"call": name, "args": dict(args)})
            if tool_feedback := standalone_tool_call_feedback(
                action,
                type(self).known_standalone_tool_names(),
                metadata.extra_tool_schemas,
            ):
                results.append(make_tool_result(call_id, error=tool_feedback.message))
                continue
            if name != "response":
                results.append(
                    make_tool_result(call_id, error=f"unknown tool: {name}")
                )
                continue
            if response_seen:
                results.append(
                    make_tool_result(call_id, error="only one response is accepted per attempt")
                )
                continue
            response_seen = True
            submitted = str(args.get("text", ""))
            correct = _answer_matches(submitted, task.answer)
            reward = 1.0 if correct else 0.0
            final_attempt = self._attempts >= self.max_turns
            if correct:
                feedback = f"expected={task.answer}; correct"
            elif final_attempt:
                feedback = f"expected={task.answer}; incorrect; no attempts remain"
            else:
                feedback = "incorrect; revise your reasoning and submit another response"
            results.append(
                make_tool_result(
                    call_id,
                    text=f"submitted={submitted}; {feedback}",
                )
            )

        terminated = reward >= 1.0 or self._attempts >= self.max_turns
        return LiteEnvStepResult(
            results=results,
            reward=reward,
            terminated=terminated,
            info={
                EXECUTED_ACTIONS_INFO_KEY: executed,
                "attempt": self._attempts,
                "max_turns": self.max_turns,
                "correct": reward >= 1.0,
            },
        )

    async def close(self) -> None:
        return None


def _load_source_task(source: Geo3KSourceTask) -> Geo3KTask:
    row = _read_parquet_row(source.source_path, source.row_index)
    source_dir = Path(source.source_path).resolve().parent
    raw_images = row.get("preprocessed_images") or row.get("images")
    return Geo3KTask(
        question=str(row["problem"]),
        answer=str(row["answer"]),
        image=_geo3k_image_to_png(raw_images, source_dir=source_dir),
        split=source.split,
    )


def _read_parquet_row(path: str, row_index: int) -> dict[str, Any]:
    import pyarrow.parquet as pq

    if row_index < 0:
        raise IndexError(f"Geo3K row_index must be non-negative, got {row_index}")
    parquet = pq.ParquetFile(path)
    remaining = row_index
    for group_idx in range(parquet.num_row_groups):
        group_rows = parquet.metadata.row_group(group_idx).num_rows
        if remaining >= group_rows:
            remaining -= group_rows
            continue
        table = parquet.read_row_group(group_idx).slice(remaining, 1)
        return table.to_pylist()[0]
    raise IndexError(f"Geo3K row_index {row_index} out of range for {path}")


def geo3k_source_fingerprint(path: str | os.PathLike[str]) -> str:
    """Fast content fingerprint for binding row ids to a source parquet."""
    source = Path(path)
    rows = _parquet_num_rows(source)
    stat = source.stat()
    digest = hashlib.sha256()
    digest.update(f"geo3k:{stat.st_size}:{rows}:".encode())
    chunk_size = 1024 * 1024
    with source.open("rb") as f:
        digest.update(f.read(chunk_size))
        if stat.st_size > chunk_size:
            f.seek(max(0, stat.st_size - chunk_size))
            digest.update(f.read(chunk_size))
    return digest.hexdigest()


def _geo3k_image_to_png(raw: Any, *, source_dir: Path) -> bytes | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        if not raw:
            return None
        return _geo3k_image_to_png(raw[0], source_dir=source_dir)
    if isinstance(raw, dict):
        if raw.get("bytes") is not None:
            return _image_bytes_to_png(raw["bytes"])
        if raw.get("path"):
            return _image_path_to_png(str(raw["path"]), source_dir=source_dir)
        return None
    if isinstance(raw, (bytes, bytearray)):
        return _image_bytes_to_png(bytes(raw))
    if isinstance(raw, str):
        if raw.startswith("data:image/") and ";base64," in raw:
            return _image_bytes_to_png(base64.b64decode(raw.split(";base64,", 1)[1]))
        return _image_path_to_png(raw, source_dir=source_dir)
    if isinstance(raw, Image.Image):
        return encode_png(raw.convert("RGB"))
    raise TypeError(f"unsupported Geo3K image value: {type(raw).__name__}")


def _image_bytes_to_png(data: bytes) -> bytes:
    with Image.open(io.BytesIO(data)) as image:
        return encode_png(image.convert("RGB"))


def _image_path_to_png(path: str, *, source_dir: Path) -> bytes:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = source_dir / resolved
    with Image.open(resolved) as image:
        return encode_png(image.convert("RGB"))


def _boxed_values(text: str) -> list[str]:
    values: list[str] = []
    cursor = 0
    while True:
        start = text.find("\\boxed", cursor)
        if start < 0:
            return values
        brace = text.find("{", start + len("\\boxed"))
        if brace < 0:
            return values
        depth = 0
        for idx in range(brace, len(text)):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    values.append(text[brace + 1:idx])
                    cursor = idx + 1
                    break
        else:
            return values


def _answer_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    boxed = _boxed_values(text)
    if boxed:
        candidates.append(boxed[-1])
    else:
        for line in reversed(text.splitlines()):
            lower = line.lower()
            if "answer" in lower or "therefore" in lower or "so" in lower:
                candidates.append(line.split(":", 1)[-1])
                break
    if not candidates:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            candidates.append(lines[-1])
        else:
            candidates.append(text)
    for segment in list(candidates):
        candidates.extend(
            re.findall(r"\\sqrt\s*\{\s*[^{}]+\s*\}|sqrt\s*\([^()]+\)|√\s*\d+", segment)
        )
        candidates.extend(
            re.findall(r"[-+]?\d+(?:\.\d+)?\s*(?:\\?pi|π)?", segment, flags=re.I)
        )
    return candidates


def _normalize_answer(text: str) -> str:
    compact = text.strip().lower()
    compact = compact.replace("$", "").replace(" ", "")
    compact = compact.replace("\\left", "").replace("\\right", "")
    compact = compact.replace("\\pi", "pi").replace("π", "pi")
    compact = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", compact)
    compact = compact.strip(".,;:。")
    if compact in {"sqrt(2)", "√2"}:
        return "sqrt(2)"
    if compact in {"6pi", "6*pi", "6×pi"}:
        return "6*pi"
    try:
        value = float(compact)
    except ValueError:
        return compact
    return "sqrt(2)" if math.isclose(value, math.sqrt(2), rel_tol=1e-4, abs_tol=1e-4) else compact


def _answer_matches(submitted: str, expected: str) -> bool:
    expected_norm = _normalize_answer(expected)
    for candidate in _answer_candidates(submitted):
        if _normalize_answer(candidate) == expected_norm:
            return True
    return False


def _resolve_source_path(source_path: str | os.PathLike[str] | None = None) -> Path | None:
    if source_path is not None:
        return Path(source_path)
    env_source = os.environ.get(GEO3K_SOURCE_ENV)
    if env_source:
        return Path(env_source)
    return None


def _parquet_num_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).metadata.num_rows


def register_geo3k(
    *,
    source_path: str | os.PathLike[str] | None = None,
    split: str = "train",
    limit: int | None = None,
) -> None:
    """Register Geo3K tasks.

    When ``source_path`` (or ``GEO3K_SOURCE``) points at a Geo3K parquet, each
    row becomes one task id. Without a source, a tiny text-only smoke catalog is
    registered so docs and unit tests can run offline.
    """
    source = _resolve_source_path(source_path)
    n_rows: int | None = None
    if source is not None:
        n_rows = _parquet_num_rows(source)
        if limit is not None:
            n_rows = min(n_rows, limit)

    # The selected source owns the whole Geo3K catalog for this process.
    _clear_env_registration(ENV_ID)
    register_family(ENV_ID, BackendFamily.PURE)
    registry.set_env_supported_kwargs(ENV_ID, {"extra_tools", "max_turns"})
    if source is not None:
        assert n_rows is not None
        source_fingerprint = geo3k_source_fingerprint(source)
        for row_index in range(n_rows):
            task_id = f"row_{row_index:06d}"
            register(
                f"{ENV_ID}@{task_id}",
                Geo3KEnv,
                split=split,
                metadata=_task_metadata(),
                task_id=task_id,
                source_path=str(source),
                row_index=row_index,
                source_fingerprint=source_fingerprint,
            )
        return

    for task_id, task in SMOKE_TASKS.items():
        register(
            f"{ENV_ID}@{task_id}",
            Geo3KEnv,
            split=task.split,
            metadata=_task_metadata(),
            task_id=task_id,
            question=task.question,
            answer=task.answer,
            image=task.image,
        )


__all__ = [
    "ENV_ID",
    "GEO3K_SOURCE_ENV",
    "SMOKE_TASKS",
    "TASK_ID",
    "Geo3KEnv",
    "Geo3KSourceTask",
    "Geo3KTask",
    "geo3k_source_fingerprint",
    "register_geo3k",
]
