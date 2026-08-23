from __future__ import annotations

import asyncio
import copy
import logging
import math
import re
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw

from lite.agents.core.agent.hooks import SampleHook, SampleStepData
from lite.agents.core.agent.utils.loop import mark_steps_truncated
from lite.agents.models.qwen3_5.adapter import Qwen3_5DesktopGroundingPointAdapter
from lite.agents.models.qwen3_5.agent import Qwen3_5DesktopGroundingPointAgent
from lite.agents.models.qwen3_vl.adapter import Qwen3VLDesktopGroundingPointAdapter
from lite.agents.models.qwen3_vl.agent import Qwen3VLDesktopGroundingPointAgent
from lite.agents.types import PredictResult
from lite.core import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_TRUNCATED,
    LiteMessage,
    LiteRLSample,
    LiteRLStep,
    LiteSample,
)
from lite.core.messages.final import mark_model_output_error
from lite.core.messages.image_refs import (
    referenced_image_indices_in_message_order,
    referenced_images_in_message_order,
)
from lite.core.messages.turns import count_sample_turns, group_into_turns
from lite.core.tools.action_space.geometry import strict_norm_to_pixel
from lite.core.tools.calls import (
    stamp_message_tool_call_ids,
    tool_call_arguments,
    tool_call_name,
)
from lite.gym.types import EXECUTED_ACTIONS_INFO_KEY, LiteEnvStepResult
from lite.utils.image import decode_image
from lite.utils.timer import Timer

logger = logging.getLogger(__name__)

_REFINE_CROP_RATIOS: tuple[tuple[float, float], ...] = (
    (0.5, 0.5),
    (0.3, 0.3),
    (0.4, 0.8),
    (0.8, 0.4),
)

_JUDGE_SYSTEM_TEXT = (
    "Judge whether the highlighted point is correct "
    "for the instruction. Reply with YES or NO first, "
    "then one short reason."
)
_JUDGE_PROMPT_MARKER = (
    "The magenta marker shows the predicted point. "
    "Reply with YES if it is correct and precise enough; otherwise reply with NO."
)
_AGGREGATE_SYSTEM_TEXT = "You are a precise UI grounding selector."
_AGGREGATE_PROMPT_MARKER = (
    "Several candidate points are marked and numbered. Select the single best point. "
    "Reply exactly with: Selected point: N"
)


def _judge_prompt_text(instruction: str) -> str:
    return f'Instruction: "{instruction}". {_JUDGE_PROMPT_MARKER}'


def _judge_prompt_messages(prompt_text: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": _JUDGE_SYSTEM_TEXT}]},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
        },
    ]


def _aggregate_prompt_text(instruction: str) -> str:
    return f'Instruction: "{instruction}". {_AGGREGATE_PROMPT_MARKER}'


def _aggregate_prompt_messages(prompt_text: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": _AGGREGATE_SYSTEM_TEXT}]},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
        },
    ]


def _regionfocus_system_text_for_prompt(prompt_text: str) -> str | None:
    if _JUDGE_PROMPT_MARKER in prompt_text:
        return _JUDGE_SYSTEM_TEXT
    if _AGGREGATE_PROMPT_MARKER in prompt_text:
        return _AGGREGATE_SYSTEM_TEXT
    return None

def _extract_point_from_lite_message(message: LiteMessage) -> list[int] | None:
    for tool_call in message.get("tool_calls", []):
        if tool_call_name(tool_call) != "point":
            continue
        coordinate = tool_call_arguments(tool_call).get("coordinate")
        if (
            isinstance(coordinate, list)
            and len(coordinate) == 2
            and all(isinstance(v, (int, float)) for v in coordinate)
        ):
            return [int(round(coordinate[0])), int(round(coordinate[1]))]
    return None


def _judge_accepted_from_response(response: str) -> bool:
    """Decide YES/NO from the leading token only, not a whole-response search.

    The judge prompt asks for "YES/NO first, then a short reason," so reasons
    routinely contain the word "no" (e.g. "no issues"); searching the whole
    response for \\bNO\\b flips valid YES verdicts to NO.
    """
    match = re.match(r"[^A-Za-z]*([A-Za-z]+)", response.strip())
    token = match.group(1).upper() if match else ""
    return token == "YES"


def _set_point_on_lite_message(message: dict[str, Any], point: list[int]) -> dict[str, Any]:
    patched = copy.deepcopy(message)
    for tool_call in patched.get("tool_calls", []):
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)
        if name == "point":
            args["coordinate"] = list(point)
        if name == "computer_use" and args.get("action") == "left_click":
            args["coordinate"] = list(point)
    return patched


def _norm_to_pixel(point: list[int], image: Image.Image) -> tuple[int, int]:
    return strict_norm_to_pixel(point, *image.size, clamp=True)


def _coerce_norm_value(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _project_crop_point(
    local_point: list[int],
    norm_box: tuple[int, int, int, int],
) -> list[int]:
    local_x = _coerce_norm_value(local_point[0], label="crop point x")
    local_y = _coerce_norm_value(local_point[1], label="crop point y")
    if not (0 <= local_x <= 1000 and 0 <= local_y <= 1000):
        raise ValueError(
            "crop-refine point outside normalized [0,1000] before projection: "
            f"{local_point!r}"
        )

    left_norm, top_norm, crop_w_norm, crop_h_norm = norm_box
    projected_x = left_norm + local_x / 1000.0 * crop_w_norm
    projected_y = top_norm + local_y / 1000.0 * crop_h_norm
    if not (0 <= projected_x <= 1000 and 0 <= projected_y <= 1000):
        raise ValueError(
            "crop-refine projected point outside normalized [0,1000]: "
            f"local={local_point!r} crop_norm_box={norm_box!r} "
            f"projected={[projected_x, projected_y]!r}"
        )
    return [int(round(projected_x)), int(round(projected_y))]


def _point_correct_from_bbox(
    point: list[int] | None,
    *,
    metadata: Any | None = None,
    step_info: dict[str, Any] | None = None,
) -> bool | None:
    from examples.grounding.utils import rf_metrics

    return rf_metrics.point_correct_from_sources(
        point,
        metadata=metadata,
        step_info=step_info,
    )


def _draw_marker(image: Image.Image, point: list[int], *, label: str | None = None) -> Image.Image:
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    px, py = _norm_to_pixel(point, output)
    radius = max(10, round(min(output.size) * 0.015))
    draw.ellipse(
        (px - radius, py - radius, px + radius, py + radius),
        fill=(255, 0, 255),
        outline=(255, 255, 255),
        width=2,
    )
    draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(0, 255, 0))
    if label:
        draw.text((px + radius + 2, py - radius), label, fill=(255, 0, 255))
    return output


def _make_step(payload: dict[str, Any]) -> LiteRLStep:
    finish_reason = payload.get("finish_reason", "stop")
    status = (
        STATUS_TRUNCATED
        if finish_reason in ("length", "max_tokens", "context_length_exceeded")
        else STATUS_COMPLETED
    )
    return LiteRLStep(
        prompt=payload["prompt"],
        image_indices=tuple(payload["image_indices"]),
        response=payload["response"],
        response_tokens=list(payload.get("response_tokens") or []),
        response_log_probs=list(payload.get("response_log_probs") or []),
        reward=0.0,
        status=status,
    )


def _user_message(text: str, image_index: int | None = None) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if image_index is not None:
        content.append({"type": "image", "index": image_index})
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def _assistant_text_message(text: str) -> dict[str, Any]:
    content = [{"type": "text", "text": text}] if text else []
    return {"role": "assistant", "content": content}


def _append_trainable_turn(
    messages: list[LiteMessage],
    *,
    image_index: int | None,
    prompt_text: str,
    assistant_message: LiteMessage,
) -> None:
    messages.append(_user_message(prompt_text, image_index))
    messages.append(copy.deepcopy(assistant_message))


def _replace_last_assistant_message(
    messages: list[LiteMessage],
    message: LiteMessage,
) -> None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            messages[index] = copy.deepcopy(message)
            return


def _attach_trace_to_last_assistant(
    messages: list[LiteMessage],
    trace: dict[str, Any],
) -> None:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            message["regionfocus_trace"] = copy.deepcopy(trace)
            return


class _RegionFocusGroundingAdapterMixin:
    """Render RegionFocus-only judge/aggregate turns with their online prompts."""

    def render_step(
        self,
        sample: LiteSample,
        k: int,
        processed,
        **kwargs,
    ):
        turns = group_into_turns(sample.messages)
        if 1 <= k <= len(turns):
            rendered = self._render_regionfocus_special_turn(turns[k - 1])
            if rendered is not None:
                return rendered
        return super().render_step(sample, k, processed, **kwargs)

    def _render_regionfocus_special_turn(self, turn: dict[str, Any]):
        assistant = turn.get("assistant")
        if assistant is None:
            return None

        user_message = None
        for message in turn.get("observations") or []:
            if message.get("role") == "user":
                user_message = message
                break
        if user_message is None:
            return None

        prompt_text = ""
        user_content: list[dict[str, Any]] = []
        for part in user_message.get("content") or []:
            if part.get("type") == "image":
                user_content.append(copy.deepcopy(part))
            elif part.get("type") == "text":
                text = str(part.get("text") or "")
                prompt_text += text
                user_content.append({"type": "text", "text": text})

        system_text = _regionfocus_system_text_for_prompt(prompt_text)
        if system_text is None:
            return None

        return [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": user_content},
            self.convert_message_to_agent(assistant),
        ]


@dataclass
class Qwen3VLRegionFocusGroundingAdapter(
    _RegionFocusGroundingAdapterMixin,
    Qwen3VLDesktopGroundingPointAdapter,
    key=r"qwen3_vl\.regionfocus@(desktop|browser)@grounding\.point",
):
    """Qwen3-VL wire-format registration for the RegionFocus agent."""


@dataclass
class Qwen3_5RegionFocusGroundingAdapter(
    _RegionFocusGroundingAdapterMixin,
    Qwen3_5DesktopGroundingPointAdapter,
    key=r"qwen3_5\.regionfocus@(desktop|browser)@grounding\.point",
):
    """Qwen3.5 XML-wire adapter entry for the RegionFocus agent."""


@dataclass
class _RegionFocusGroundingAgentMixin:
    """RegionFocus grounding loop that trains on every generate call.

    Flow:
      1. initial grounding
      2. self-judge
      3. if judge=no: reseed + crop refine + aggregate

    sample() emits one LiteRLStep per generate call along the adopted path
    (initial, judge, seed, every valid crop-refine, and the aggregate
    selection), all sharing the episode reward via
    LiteRLSample.episode_return. predict() / _predict_with_details() run
    the same pipeline without per-turn step tracking, for standalone eval.
    """

    judge_initial_point: bool = True
    region_focus_temperatures: tuple[float, ...] = (0.0, 0.3, 0.5)

    async def _append_bound_image(
        self,
        lite_sample: LiteSample | None,
        processed_images: list[Image.Image],
        image: Image.Image,
        *,
        processed_image: Image.Image | None = None,
    ) -> int:
        if lite_sample is not None and len(lite_sample.images) != len(processed_images):
            raise RuntimeError(
                "LiteSample.images and processed_images index drift before image append"
            )
        image_index = len(processed_images)
        if lite_sample is not None:
            lite_sample.images.append(image)
        processed_images.append(
            processed_image
            if processed_image is not None
            else await asyncio.to_thread(self.adapter.process_image, image)
        )
        return image_index

    async def _bind_payload_image(
        self,
        payload: dict[str, Any],
        lite_sample: LiteSample | None,
        processed_images: list[Image.Image],
        image: Image.Image,
    ) -> int:
        image_index = await self._append_bound_image(
            lite_sample,
            processed_images,
            image,
            processed_image=image,
        )
        payload["image_indices"] = (image_index,)
        return image_index

    async def _predict_with_details(
        self,
        lite_sample: LiteSample,
        *,
        processed_images: list[Image.Image],
    ) -> PredictResult:
        trace: dict[str, Any] = {
            "instruction": self._extract_instruction(lite_sample),
            "path": "initial_only",
        }
        initial = await self._generate_grounding_once(lite_sample, processed_images)
        initial_point = _extract_point_from_lite_message(initial["lite_message"])
        trace["initial"] = self._trace_entry(initial, point=initial_point)
        if initial_point is None:
            trace["path"] = "initial_invalid"
            return self._build_predict_result(initial, preserve_raw_response=True, trace=trace)

        if not self.judge_initial_point:
            return self._build_predict_result(initial, preserve_raw_response=True, trace=trace)

        original_image = processed_images[initial["image_indices"][-1]]
        judge_result = await self._judge_point(
            original_image,
            trace["instruction"],
            initial_point,
        )
        judged_yes = judge_result["accepted"]
        trace["judge"] = {
            "accepted": judged_yes,
            "prompt": judge_result["prompt"],
            "response": judge_result["response"],
        }
        if judged_yes:
            trace["path"] = "judge_yes"
            return self._build_predict_result(initial, preserve_raw_response=True, trace=trace)

        original_image_index = initial["image_indices"][-1]
        seed = await self._region_focus_seed(
            original_image,
            trace["instruction"],
            image_index=original_image_index,
        )
        seed_point = seed["point"] if seed is not None else None
        trace["seed"] = self._trace_entry(seed, point=seed_point)
        if seed_point is None:
            trace["path"] = "seed_invalid"
            return self._build_predict_result(initial, preserve_raw_response=True, trace=trace)

        candidates = []
        seen_points: set[tuple[int, int]] = set()
        trace["path"] = "regionfocus"
        trace["refine_candidates"] = []
        for crop_idx, crop in enumerate(self._crop_and_upsample(original_image, seed_point)):
            refined = await self._refine_on_crop(crop, trace["instruction"])
            trace["refine_candidates"].append({
                "crop_index": crop_idx,
                "crop_norm_box": list(crop["norm_box"]),
                "result": self._trace_entry(
                    refined,
                    point=None if refined is None else refined.get("point"),
                ),
            })
            if refined is not None:
                refined_point = refined.get("point")
                if refined_point is None:
                    continue
                point_key = (int(refined_point[0]), int(refined_point[1]))
                if point_key in seen_points:
                    continue
                seen_points.add(point_key)
                await self._bind_payload_image(
                    refined,
                    None,
                    processed_images,
                    refined["train_image"],
                )
                candidates.append(refined)

        if not candidates:
            trace["path"] = "refine_empty_seed"
            trace["selected"] = self._trace_entry(seed, point=seed_point)
            return self._build_predict_result(seed, preserve_raw_response=True, trace=trace)

        selected = await self._aggregate_points(
            original_image,
            trace["instruction"],
            candidates,
        )
        selected_point = selected.get("point")
        trace["selected"] = self._trace_entry(selected, point=selected_point)
        return self._build_predict_result(selected, preserve_raw_response=False, trace=trace)

    async def _run_judge_as_step(
        self,
        original_image: Image.Image,
        instruction: str,
        point: list[int],
        lite_sample: LiteSample | None,
        processed_images: list[Image.Image],
    ) -> tuple[LiteRLStep, bool, dict[str, Any]]:
        """Run judge, add highlighted image to processed_images, return (step, accepted, trace)."""
        highlighted = _draw_marker(original_image, point)

        prompt_text = _judge_prompt_text(instruction)
        prompt = self.build_generation_prompt(_judge_prompt_messages(prompt_text))
        gen = await self.generate_fn(prompt=prompt, images=[highlighted], messages=None)
        response = gen["response"].strip()
        accepted = _judge_accepted_from_response(response)
        image_index = await self._append_bound_image(
            lite_sample,
            processed_images,
            highlighted,
            processed_image=highlighted,
        )

        step = LiteRLStep(
            prompt=prompt,
            image_indices=(image_index,),
            response=response,
            response_tokens=list(gen.get("response_tokens") or []),
            response_log_probs=list(gen.get("response_log_probs") or []),
            reward=0.0,
            status=STATUS_COMPLETED,
        )
        trace = {
            "accepted": accepted,
            "prompt": prompt,
            "response": response,
            "prompt_text": prompt_text,
            "image_index": image_index,
        }
        return step, accepted, trace

    async def _run_aggregate_as_step(
        self,
        original_image: Image.Image,
        instruction: str,
        candidates: list[dict[str, Any]],
        lite_sample: LiteSample | None,
        processed_images: list[Image.Image],
    ) -> tuple[LiteRLStep, dict[str, Any]]:
        """Run aggregate selection and return the step plus selected payload."""
        labeled: Image.Image = original_image.copy().convert("RGB")
        for idx, candidate in enumerate(candidates, start=1):
            labeled = _draw_marker(labeled, candidate["point"], label=str(idx))

        prompt_text = _aggregate_prompt_text(instruction)
        prompt = self.build_generation_prompt(_aggregate_prompt_messages(prompt_text))
        gen = await self.generate_fn(prompt=prompt, images=[labeled], messages=None)
        response = gen["response"]
        image_index = await self._append_bound_image(
            lite_sample,
            processed_images,
            labeled,
            processed_image=labeled,
        )

        match = re.search(r"Selected point:\s*(\d+)", response, re.IGNORECASE)
        selected_idx = 0
        if match:
            i = int(match.group(1)) - 1
            if 0 <= i < len(candidates):
                selected_idx = i

        selected = copy.deepcopy(candidates[selected_idx])
        selected["aggregation_response"] = response
        selected["aggregation_selected_index"] = selected_idx
        selected["aggregation_prompt_text"] = prompt_text
        selected["aggregation_image_index"] = image_index

        step = LiteRLStep(
            prompt=prompt,
            image_indices=(image_index,),
            response=response,
            response_tokens=list(gen.get("response_tokens") or []),
            response_log_probs=list(gen.get("response_log_probs") or []),
            reward=0.0,
            status=STATUS_COMPLETED,
        )
        return step, selected

    async def sample(
        self,
        env: Any,
        max_steps: int = 10_000,
        hooks: list[SampleHook] | None = None,
    ) -> LiteRLSample:
        """Multi-turn sample: every generate call becomes a trainable LiteRLStep.

        For each episode, emits one LiteRLStep per generate call along the
        adopted path:
          1. initial coordinate (always)
          2. judge YES/NO (only if judge_initial_point)
          3. seed coordinate (only if judge=NO and seed is valid)
          4. every valid crop-refine coordinate (only if seed was valid)
          5. aggregate selection (only if refine yielded >1 candidate)

        All steps share the same episode reward via LiteRLSample.episode_return.
        """
        hooks = hooks or []
        lite_rl_sample: LiteRLSample | None = None
        completed = False

        try:
            t = Timer()
            reset_observation = await env.reset()
            t.mark("reset")

            screenshot_png = reset_observation.image
            instruction = reset_observation.text
            if not screenshot_png and not instruction:
                raise RuntimeError("Env returned neither screenshot nor text on env.reset()")
            if not instruction:
                raise RuntimeError(
                    "No instruction returned on env.reset() "
                    "(observation.text is None/empty)"
                )

            lite_sample = LiteSample(metadata=env.metadata)
            lite_rl_sample = LiteRLSample(lite_sample=lite_sample)
            reset_image_index: int | None = None

            if screenshot_png:
                screenshot = await asyncio.to_thread(decode_image, screenshot_png)
                reset_image_index = await self._append_bound_image(
                    lite_sample,
                    lite_rl_sample.processed_images,
                    screenshot,
                )
                original_image = lite_rl_sample.processed_images[reset_image_index]
            else:
                screenshot = None
                original_image = None

            lite_sample.messages.append(_user_message(instruction, reset_image_index))
            t.mark("observe")

            steps: list[LiteRLStep] = []
            trace: dict[str, Any] = {"instruction": instruction, "path": "initial_only"}
            _judge_ran = False
            _judged_yes = False
            _initial_point_correct: bool | None = None
            next_call_id = 0

            # 1. Initial grounding
            initial = await self._generate_grounding_once(
                lite_sample,
                lite_rl_sample.processed_images,
                call_id_start=next_call_id,
            )
            next_call_id += len(initial["lite_message"].get("tool_calls") or [])
            initial_point = _extract_point_from_lite_message(initial["lite_message"])
            trace["initial"] = self._trace_entry(initial, point=initial_point)
            steps.append(_make_step(initial))
            lite_sample.messages.append(copy.deepcopy(initial["lite_message"]))
            final_payload = initial

            if initial_point is None:
                trace["path"] = "initial_invalid"
            elif self.judge_initial_point and original_image is not None:
                # 2. Judge — also a training step
                judge_step, judged_yes, judge_trace = await self._run_judge_as_step(
                    original_image, instruction, initial_point,
                    lite_sample,
                    lite_rl_sample.processed_images,
                )
                _judge_ran = True
                _judged_yes = judged_yes
                steps.append(judge_step)
                trace["judge"] = judge_trace
                _append_trainable_turn(
                    lite_sample.messages,
                    image_index=judge_trace["image_index"],
                    prompt_text=judge_trace["prompt_text"],
                    assistant_message=_assistant_text_message(judge_trace["response"]),
                )

                if judged_yes:
                    trace["path"] = "judge_yes"
                else:
                    # 3. Seed grounding
                    seed = await self._region_focus_seed(
                        original_image,
                        instruction,
                        image_index=reset_image_index or 0,
                        call_id_start=next_call_id,
                    )
                    seed_point = seed["point"] if seed is not None else None
                    trace["seed"] = self._trace_entry(seed, point=seed_point)

                    if seed is None or seed_point is None:
                        trace["path"] = "seed_invalid"
                    else:
                        next_call_id += len(seed["lite_message"].get("tool_calls") or [])
                        steps.append(_make_step(seed))
                        _append_trainable_turn(
                            lite_sample.messages,
                            image_index=seed["image_indices"][-1],
                            prompt_text=seed["prompt_text"],
                            assistant_message=seed["lite_message"],
                        )
                        trace["path"] = "regionfocus"
                        trace["refine_candidates"] = []

                        # 4. Crop-refine (sequential): ALL valid results become steps
                        candidates: list[dict[str, Any]] = []
                        seen_points: set[tuple[int, int]] = set()
                        for crop_idx, crop in enumerate(
                            self._crop_and_upsample(original_image, seed_point)
                        ):
                            refined = await self._refine_on_crop(
                                crop,
                                instruction,
                                call_id_start=next_call_id,
                            )
                            trace["refine_candidates"].append({
                                "crop_index": crop_idx,
                                "crop_norm_box": list(crop["norm_box"]),
                                "result": self._trace_entry(
                                    refined,
                                    point=None if refined is None else refined.get("point"),
                                ),
                            })
                            if refined is None:
                                continue
                            refined_point = refined.get("point")
                            if refined_point is None:
                                continue
                            point_key = (int(refined_point[0]), int(refined_point[1]))
                            if point_key in seen_points:
                                continue
                            seen_points.add(point_key)
                            crop_image_index = await self._bind_payload_image(
                                refined,
                                lite_sample,
                                lite_rl_sample.processed_images,
                                refined["train_image"],
                            )
                            next_call_id += len(refined["lite_message"].get("tool_calls") or [])
                            candidates.append(refined)
                            steps.append(_make_step(refined))  # every valid crop is a step
                            _append_trainable_turn(
                                lite_sample.messages,
                                image_index=crop_image_index,
                                prompt_text=refined["prompt_text"],
                                assistant_message=refined["lite_message"],
                            )

                        if not candidates:
                            trace["path"] = "refine_empty_seed"
                            trace["selected"] = self._trace_entry(seed, point=seed_point)
                            final_payload = seed
                        elif len(candidates) == 1:
                            # Single candidate: no aggregate generate needed
                            selected = copy.deepcopy(candidates[0])
                            selected["aggregation_response"] = "Selected point: 1"
                            selected["aggregation_selected_index"] = 0
                            trace["selected"] = self._trace_entry(
                                selected,
                                point=selected.get("point"),
                            )
                            final_payload = selected
                        else:
                            # 5. Aggregate — also a training step
                            agg_step, selected = await self._run_aggregate_as_step(
                                original_image, instruction, candidates,
                                lite_sample,
                                lite_rl_sample.processed_images,
                            )
                            trace["selected"] = self._trace_entry(
                                selected,
                                point=selected.get("point"),
                            )
                            steps.append(agg_step)
                            _append_trainable_turn(
                                lite_sample.messages,
                                image_index=selected["aggregation_image_index"],
                                prompt_text=selected["aggregation_prompt_text"],
                                assistant_message=_assistant_text_message(
                                    selected["aggregation_response"]
                                ),
                            )
                            final_payload = selected

            for step in steps:
                lite_rl_sample.steps.append(step)
            t.mark("predict")

            final_lite_message = copy.deepcopy(final_payload["lite_message"])
            final_point = final_payload.get("point")
            if final_point is not None:
                final_lite_message = _set_point_on_lite_message(final_lite_message, final_point)
            stamp_message_tool_call_ids(final_lite_message, start=next_call_id, preserve=False)
            actions = final_lite_message.get("tool_calls", [])
            if not actions:
                error = "RegionFocus final response produced no tool_calls"
                logger.warning("%s; ending as terminal model-output failure", error)
                mark_model_output_error(final_lite_message, error)
                final_lite_message["regionfocus_trace"] = copy.deepcopy(trace)
                _replace_last_assistant_message(lite_sample.messages, final_lite_message)
                if lite_rl_sample.steps:
                    lite_rl_sample.steps[-1].status = STATUS_FAILED
                lite_rl_sample.terminated = True
                t.mark("act")
                step_result = LiteEnvStepResult(
                    reward=0.0,
                    terminated=True,
                    truncated=False,
                    info={
                        EXECUTED_ACTIONS_INFO_KEY: [],
                        "model_output_error": error,
                    },
                )
                predict_result = self._build_predict_result(
                    final_payload, preserve_raw_response=True, trace=trace
                )
                predict_result.lite_message = final_lite_message
                predict_result.step.status = STATUS_FAILED
                predict_result.model_output_error = error
                logger.info(
                    "⏱ sample step 0: %s  n_rl_steps=%d  path=%s",
                    t, len(steps), trace.get("path"),
                )
                step_data = SampleStepData(
                    step_idx=0,
                    image=screenshot,
                    predict_result=predict_result,
                    step_result=step_result,
                    actions=[],
                    current_image_index=reset_image_index,
                )
                for hook in hooks:
                    await asyncio.to_thread(hook.on_step, step_data)
                completed = True
                return lite_rl_sample
            step_result = await env.step(actions)
            t.mark("act")

            logger.info(
                "⏱ sample step 0: %s  n_rl_steps=%d  path=%s",
                t, len(steps), trace.get("path"),
            )

            if step_result.reward is not None:
                lite_rl_sample.episode_return += step_result.reward
                lite_rl_sample.steps[-1].reward = float(step_result.reward)
            _initial_point_correct = _point_correct_from_bbox(
                initial_point,
                metadata=lite_sample.metadata,
                step_info=step_result.info,
            )
            if trace.get("initial") is not None:
                trace["initial"]["bbox_correct"] = _initial_point_correct
            final_lite_message["regionfocus_trace"] = copy.deepcopy(trace)
            _attach_trace_to_last_assistant(lite_sample.messages, trace)
            if len(lite_sample.images) != len(lite_rl_sample.processed_images):
                raise RuntimeError(
                    "LiteSample.images and processed_images index drift before completion"
                )
            lite_rl_sample.terminated = step_result.terminated
            lite_rl_sample.truncated = step_result.truncated
            if lite_rl_sample.truncated:
                mark_steps_truncated(lite_rl_sample.steps)

            predict_result = self._build_predict_result(
                final_payload, preserve_raw_response=True, trace=trace
            )
            predict_result.lite_message = final_lite_message
            step_data = SampleStepData(
                step_idx=0,
                image=screenshot,
                predict_result=predict_result,
                step_result=step_result,
                actions=actions,
                current_image_index=reset_image_index,
            )
            for hook in hooks:
                await asyncio.to_thread(hook.on_step, step_data)

            if _judge_ran:
                from examples.grounding.utils import rf_metrics
                rf_metrics.record({
                    "judge_accepted": _judged_yes,
                    "initial_point_correct": _initial_point_correct,
                    "reward": float(step_result.reward) if step_result.reward is not None else 0.0,
                })

            completed = True
            return lite_rl_sample

        finally:
            for hook in hooks:
                await asyncio.to_thread(hook.on_complete, lite_rl_sample if completed else None)
            try:
                await env.close()
            except Exception as e:
                logger.warning("env.close() failed: %s", e)

    async def _generate_grounding_once(
        self,
        lite_sample: LiteSample,
        processed_images: list[Image.Image],
        *,
        call_id_start: int = 0,
    ) -> dict[str, Any]:
        k = count_sample_turns(lite_sample)
        agent_step = await asyncio.to_thread(
            self.adapter.render_step,
            lite_sample,
            k,
            processed_images,
        )
        step_images = referenced_images_in_message_order(agent_step, processed_images)
        image_indices = referenced_image_indices_in_message_order(agent_step)
        prompt = self.build_generation_prompt(agent_step)
        image_pad_count = prompt.count("<|image_pad|>")
        if image_pad_count != len(image_indices):
            raise RuntimeError(
                "grounding image/template alignment broke: "
                f"image_indices={len(image_indices)} image_pad={image_pad_count}"
            )
        gen_response = await self.generate_fn(
            prompt=prompt,
            images=step_images,
            messages=agent_step,
        )
        response = gen_response["response"]
        agent_message = self.adapter.parse_raw_assistant_response(response)
        lite_message = self.adapter.convert_message_from_agent(agent_message)
        stamp_message_tool_call_ids(lite_message, start=call_id_start, preserve=False)
        return {
            "prompt": prompt,
            "image_indices": image_indices,
            "response": response,
            "agent_message": agent_message,
            "lite_message": lite_message,
            "response_tokens": list(gen_response.get("response_tokens") or []),
            "response_log_probs": list(gen_response.get("response_log_probs") or []),
            "finish_reason": (gen_response.get("finish_reason") or "stop").lower(),
        }

    async def _judge_point(
        self,
        image: Image.Image,
        instruction: str,
        point: list[int],
    ) -> dict[str, Any]:
        highlighted = _draw_marker(image, point)
        prompt_text = _judge_prompt_text(instruction)
        prompt = self.build_generation_prompt(_judge_prompt_messages(prompt_text))
        gen_response = await self.generate_fn(prompt=prompt, images=[highlighted], messages=None)
        response = gen_response["response"].strip()
        accepted = _judge_accepted_from_response(response)
        return {"accepted": accepted, "prompt": prompt, "response": response}

    async def _region_focus_seed(
        self,
        image: Image.Image,
        instruction: str,
        image_index: int = 0,
        call_id_start: int = 0,
    ) -> dict[str, Any] | None:
        for temperature in self.region_focus_temperatures:
            parsed = await self._generate_point_on_image(
                image=image,
                prompt_text=(
                    "Locate the most relevant point in the image for the "
                    f'instruction "{instruction}".'
                ),
                image_indices=(image_index,),
                call_id_start=call_id_start,
                temperature=temperature,
                top_p=0.9,
            )
            if parsed is None:
                continue
            point = _extract_point_from_lite_message(parsed["lite_message"])
            if point is None:
                continue
            parsed["point"] = point
            parsed["seed_temperature"] = temperature
            return parsed
        return None

    def _crop_and_upsample(
        self,
        image: Image.Image,
        seed_point: list[int],
    ) -> list[dict[str, Any]]:
        width, height = image.size
        crops: list[dict[str, Any]] = []
        for ratio_x, ratio_y in _REFINE_CROP_RATIOS:
            crop_w_norm = int(round(1000 * ratio_x))
            crop_h_norm = int(round(1000 * ratio_y))
            left_norm = max(0, min(1000 - crop_w_norm, int(round(seed_point[0] - crop_w_norm / 2))))
            top_norm = max(0, min(1000 - crop_h_norm, int(round(seed_point[1] - crop_h_norm / 2))))

            left_px = int(round(left_norm / 1000 * width))
            top_px = int(round(top_norm / 1000 * height))
            right_px = int(round((left_norm + crop_w_norm) / 1000 * width))
            bottom_px = int(round((top_norm + crop_h_norm) / 1000 * height))
            right_px = max(left_px + 1, min(width, right_px))
            bottom_px = max(top_px + 1, min(height, bottom_px))

            crop_img = image.crop((left_px, top_px, right_px, bottom_px))
            zoom = min(width / max(1, crop_img.width), height / max(1, crop_img.height))
            cropped = crop_img.resize(
                (max(1, round(crop_img.width * zoom)), max(1, round(crop_img.height * zoom))),
                Image.Resampling.LANCZOS,
            )
            crops.append({
                "image": cropped,
                "norm_box": (left_norm, top_norm, crop_w_norm, crop_h_norm),
            })
        return crops

    async def _refine_on_crop(
        self,
        crop: dict[str, Any],
        instruction: str,
        *,
        call_id_start: int = 0,
    ) -> dict[str, Any] | None:
        parsed = await self._generate_point_on_image(
            image=crop["image"],
            prompt_text=(
                "For this zoomed-in screenshot, identify the precise point "
                f'for the instruction "{instruction}".'
            ),
            call_id_start=call_id_start,
        )
        if parsed is None:
            return None

        local_point = _extract_point_from_lite_message(parsed["lite_message"])
        if local_point is None:
            return None

        left_norm, top_norm, crop_w_norm, crop_h_norm = crop["norm_box"]
        try:
            projected_point = _project_crop_point(
                local_point,
                (left_norm, top_norm, crop_w_norm, crop_h_norm),
            )
        except ValueError as e:
            parsed["local_point"] = local_point
            parsed["projection_error"] = str(e)
            return parsed

        parsed["point"] = projected_point
        parsed["local_point"] = local_point
        parsed["train_image"] = crop["image"]
        return parsed

    async def _aggregate_points(
        self,
        image: Image.Image,
        instruction: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if len(candidates) == 1:
            selected = copy.deepcopy(candidates[0])
            selected["aggregation_response"] = "Selected point: 1"
            selected["aggregation_selected_index"] = 0
            return selected

        labeled = image.copy().convert("RGB")
        for idx, candidate in enumerate(candidates, start=1):
            labeled = _draw_marker(labeled, candidate["point"], label=str(idx))

        prompt_text = _aggregate_prompt_text(instruction)
        prompt = self.build_generation_prompt(_aggregate_prompt_messages(prompt_text))
        gen_response = await self.generate_fn(prompt=prompt, images=[labeled], messages=None)
        response = gen_response["response"]
        match = re.search(r"Selected point:\s*(\d+)", response, re.IGNORECASE)
        if match:
            selected_idx = int(match.group(1)) - 1
            if 0 <= selected_idx < len(candidates):
                selected = copy.deepcopy(candidates[selected_idx])
                selected["aggregation_response"] = response
                selected["aggregation_selected_index"] = selected_idx
                return selected
        selected = copy.deepcopy(candidates[0])
        selected["aggregation_response"] = response
        selected["aggregation_selected_index"] = 0
        return selected

    async def _generate_point_on_image(
        self,
        *,
        image: Image.Image,
        prompt_text: str,
        image_indices: tuple[int, ...] = (),
        call_id_start: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> dict[str, Any] | None:
        lite_sample = LiteSample(metadata=self.adapter.metadata)
        lite_sample.images.append(image)
        lite_sample.messages.append(_user_message(prompt_text, 0))
        agent_step = await asyncio.to_thread(
            self.adapter.render_step,
            lite_sample,
            1,
            [image],
        )
        prompt = self.build_generation_prompt(agent_step)
        gen_response = await self.generate_fn(
            prompt=prompt,
            images=[image],
            messages=agent_step,
            temperature=temperature,
            top_p=top_p,
        )
        response = gen_response["response"]
        agent_message = self.adapter.parse_raw_assistant_response(response)
        lite_message = self.adapter.convert_message_from_agent(agent_message)
        stamp_message_tool_call_ids(lite_message, start=call_id_start, preserve=False)
        return {
            "prompt": prompt,
            "prompt_text": prompt_text,
            "image_indices": image_indices,
            "response": response,
            "agent_message": agent_message,
            "lite_message": lite_message,
            "response_tokens": list(gen_response.get("response_tokens") or []),
            "response_log_probs": list(gen_response.get("response_log_probs") or []),
            "finish_reason": (gen_response.get("finish_reason") or "stop").lower(),
        }

    def _build_predict_result(
        self,
        selected: dict[str, Any],
        *,
        preserve_raw_response: bool,
        trace: dict[str, Any] | None = None,
    ) -> PredictResult:
        lite_message = copy.deepcopy(selected["lite_message"])
        if preserve_raw_response and self.preserve_raw_response:
            lite_message["raw_response"] = {
                "text": selected["response"],
                "adapter_key": self.adapter._registry_key,
            }
        if trace is not None:
            lite_message["regionfocus_trace"] = copy.deepcopy(trace)

        finish_reason = selected.get("finish_reason", "stop")
        status = (
            STATUS_TRUNCATED
            if finish_reason in ("length", "max_tokens", "context_length_exceeded")
            else STATUS_COMPLETED
        )
        step = LiteRLStep(
            prompt=selected["prompt"],
            image_indices=tuple(selected["image_indices"]),
            response=selected["response"],
            response_tokens=list(selected.get("response_tokens") or []),
            response_log_probs=list(selected.get("response_log_probs") or []),
            reward=0.0,
            status=status,
        )
        return PredictResult(
            lite_message=lite_message,
            agent_message=self._attach_trace(selected["agent_message"], trace),
            step=step,
        )

    def _extract_instruction(self, lite_sample: LiteSample) -> str:
        for message in lite_sample.messages:
            if message.get("role") != "user":
                continue
            for part in message.get("content") or []:
                if part.get("type") == "text" and part.get("text"):
                    return str(part["text"])
        return ""

    def _attach_trace(
        self,
        agent_message: dict[str, Any],
        trace: dict[str, Any] | None,
    ) -> dict[str, Any]:
        patched = copy.deepcopy(agent_message)
        if trace is not None:
            patched["regionfocus_trace"] = copy.deepcopy(trace)
        return patched

    def _trace_entry(
        self,
        payload: dict[str, Any] | None,
        *,
        point: list[int] | None,
    ) -> dict[str, Any] | None:
        if payload is None:
            return None
        entry = {
            "point": point,
            "prompt": payload.get("prompt"),
            "response": payload.get("response"),
        }
        if "seed_temperature" in payload:
            entry["seed_temperature"] = payload.get("seed_temperature")
        if "aggregation_response" in payload:
            entry["aggregation_response"] = payload.get("aggregation_response")
            entry["aggregation_selected_index"] = payload.get("aggregation_selected_index")
        if "local_point" in payload:
            entry["local_point"] = payload.get("local_point")
        if "projection_error" in payload:
            entry["projection_error"] = payload.get("projection_error")
        return entry


@dataclass
class Qwen3VLRegionFocusGroundingAgent(
    _RegionFocusGroundingAgentMixin,
    Qwen3VLDesktopGroundingPointAgent,
    key=r"qwen3_vl\.regionfocus@(desktop|browser)@grounding\.point",
):
    """Qwen3-VL RegionFocus grounding agent."""


@dataclass
class Qwen3_5RegionFocusGroundingAgent(
    _RegionFocusGroundingAgentMixin,
    Qwen3_5DesktopGroundingPointAgent,
    key=r"qwen3_5\.regionfocus@(desktop|browser)@grounding\.point",
):
    """Qwen3.5 RegionFocus grounding agent."""
