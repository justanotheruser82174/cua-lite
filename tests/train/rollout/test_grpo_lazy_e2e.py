"""End-to-end tests for the lazy multimodal patch, exercising the real
GRPO dataflow:

  RolloutManager (build trajectory_images dict with PNG-encoded bytes)
    → multimodal_lazy_payloads (per-Sample, shared dict refs)
    → ray.put(rollout_data)        ← cross-Sample dedup HERE
    → MegatronTrainRayActor.ray.get
    → materialize_lazy_payloads calls expand(p, args=args)  ← runs HF processor
        (once per RL iter, inside actor._get_rollout_data)
    → multimodal_train_inputs with pixel_values
    → slime get_batch slices micro-batch + cats per-key across Samples

The patch's correctness hinges on three production-only behaviors that
unit tests with stub processors can't catch:

  1. ray.put cross-Sample dedup actually fires for ``image_data`` (uint8
     torch.Tensor) but NOT for raw ``bytes``. The patch wraps bytes in
     uint8 tensors specifically for this. Verified by measuring
     ``ray.put`` serialized size for shared-trajectory patterns.
  2. The PNG round-trip through ``torch.frombuffer`` → bytes →
     ``Image.open`` → ``image_processor`` produces ``pixel_values``
     BYTE-IDENTICAL to a direct (PIL → processor) baseline.
  3. The full pipeline (build → ray.put → ray.get → expand) recovers
     identical tensors to what slime's get_batch expects downstream.

Skipped outside the slime training container — needs ``ray`` and a
Qwen3VL processor checkpoint. The processor path is read from the
``CUA_LITE_E2E_HF_CHECKPOINT`` env var (defaults to
``/root/models/Tongyi-MAI/MAI-UI-2B`` which is the mai_ui-2B path
inside the slime container).

Run::

    # Inside the isolated slime container with the processor mounted:
    docker exec lite.slime-raytest python -m pytest \\
        /workspaces/cua-lite/tests/train/rollout/test_grpo_lazy_e2e.py -v
"""
from __future__ import annotations

import io
import os

import numpy as np
import pytest

ray = pytest.importorskip("ray", reason="needs ray (training container)")
torch = pytest.importorskip("torch", reason="needs torch")
PIL = pytest.importorskip("PIL", reason="needs Pillow")
transformers = pytest.importorskip("transformers", reason="needs transformers")
from PIL import Image  # noqa: E402  (must follow importorskip)
from transformers import AutoProcessor  # noqa: E402

CKPT = os.environ.get("CUA_LITE_E2E_HF_CHECKPOINT", "/root/models/Tongyi-MAI/MAI-UI-2B")
if not os.path.isdir(CKPT):
    pytest.skip(f"processor checkpoint not present at {CKPT}", allow_module_level=True)


# ----------------------------------------------------------------------------
# Module-level fixtures: real Qwen3VL processor + ray cluster.
# Both are expensive to set up (HF load ~3s, ray init ~3s) so we share
# them across the whole module.
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_processor():
    """The actual mai_ui-2B Qwen3VLProcessor used in production."""
    return AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)


@pytest.fixture(scope="module")
def ray_cluster():
    """Local ray cluster for serialize / put / get round-trips."""
    ray.init(num_cpus=2, object_store_memory=int(8e9), include_dashboard=False)
    yield
    ray.shutdown()


# ----------------------------------------------------------------------------
# Realistic source data
# ----------------------------------------------------------------------------

def _make_screen_pil(seed: int = 0) -> Image.Image:
    """Mobile-UI-shaped screenshot: solid color bands + noise block. Matches
    the actual content distribution of mobilegym/androidworld screens so
    PNG compression behavior is representative."""
    rng = np.random.default_rng(seed)
    arr = np.zeros((1600, 720, 3), dtype=np.uint8)
    arr[:400, :, :] = [40 + seed % 20, 40, 80]                  # header
    arr[400:800, :, :] = [220 + seed % 10, 220, 220]            # body
    arr[800:1200, 100:620, :] = [255, 255, 255]                 # card
    arr[200:300, 100:620, :] = rng.integers(80, 200, (100, 520, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _encode_to_tensor(pil: Image.Image) -> torch.Tensor:
    """Producer-side encoding: PIL → PNG bytes → uint8 tensor (the same
    wrap ``lite/train/rollout/core/segmenter.py::build_segment_samples`` does)."""
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8)


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

def test_e2e_png_pixel_values_byte_identical_to_legacy(real_processor):
    """The trainer-side pixel_values produced by the lazy path
    (uint8 tensor → bytes → PIL → image_processor) must be BYTE-IDENTICAL
    to the direct path (PIL → image_processor). This is the load-bearing
    fidelity invariant — without it the patch silently changes the data
    the model trains on."""
    from lite.train.utils import multimodal_expand

    class _Args:
        hf_checkpoint = CKPT

    pil = _make_screen_pil(seed=42)

    # Direct baseline (no lazy path)
    legacy_out = real_processor.image_processor(images=[pil], return_tensors="pt")
    pv_legacy = legacy_out["pixel_values"].to(torch.bfloat16)

    multimodal_expand._processor = real_processor  # avoid HF reload via stub
    try:
        payload = {
            "images":  {0: {"image_data": _encode_to_tensor(pil)}},
            "indices": (0,),
        }
        out = multimodal_expand.expand(payload, args=_Args())
        assert torch.equal(out["pixel_values"], pv_legacy), (
            "PNG round-trip changed pixel_values — patch is NOT a transparent "
            "drop-in for the direct PIL → processor path"
        )
        assert torch.equal(
            out["image_grid_thw"], legacy_out["image_grid_thw"]
        )
    finally:
        multimodal_expand._processor = None


def test_e2e_ray_put_dedup_for_image_data_tensor(ray_cluster):
    """Cross-Sample dedup — the load-bearing reason bytes are wrapped in
    uint8 tensors. With shared per-trajectory ``trajectory_images`` dict
    and N samples per trajectory, ray.put must serialize the underlying
    encoded data ONCE, not N times."""
    sc = ray._private.worker.global_worker.get_serialization_context()
    N_TRAJ, N_TURNS = 4, 10
    pil = Image.fromarray(np.random.RandomState(1).randint(0, 255, (80, 80, 3), dtype=np.uint8))
    encoded_tensors = [
        [_encode_to_tensor(pil) for _ in range(N_TURNS)]
        for _ in range(N_TRAJ)
    ]
    unique_size = sum(t.numel() for trajs in encoded_tensors for t in trajs)

    payloads = []
    for t in range(N_TRAJ):
        traj_images = {i: {"image_data": encoded_tensors[t][i]} for i in range(N_TURNS)}
        for k in range(N_TURNS):
            payloads.append({"images": traj_images, "indices": (k,)})

    train_data = {"multimodal_lazy_payloads": payloads}
    sd = sc.serialize(train_data)
    serialized_size = sd.total_bytes if hasattr(sd, "total_bytes") else len(sd.inband)

    assert serialized_size < unique_size * 2, (
        f"ray.put dedup FAILED for image_data tensors: serialized {serialized_size} bytes "
        f"vs {unique_size} bytes unique. Patch's dedup hypothesis is broken."
    )


def test_e2e_grpo_full_pipeline(real_processor, ray_cluster):
    """The full producer → ray → consumer pipeline at GRPO scale (scaled
    down for the test ray's 8 GB plasma). Verifies:
      * Sample structure survives ray.put / ray.get
      * expand() processes each Sample independently and correctly
      * Output dtype + shape match what slime's get_batch expects
      * Cross-Sample concat downstream produces one batched pixel_values tensor.
    """
    from lite.train.utils import multimodal_expand

    class _Args:
        hf_checkpoint = CKPT

    multimodal_expand._processor = real_processor
    try:
        N_TRAJ, N_TURNS = 2, 4
        train_data = {"multimodal_lazy_payloads": []}
        for t in range(N_TRAJ):
            traj_images = {
                i: {"image_data": _encode_to_tensor(_make_screen_pil(t * 100 + i))}
                for i in range(N_TURNS)
            }
            for k in range(N_TURNS):
                train_data["multimodal_lazy_payloads"].append({
                    "images": traj_images, "indices": (k,),
                })

        ref = ray.put(train_data)
        received = ray.get(ref)
        assert len(received["multimodal_lazy_payloads"]) == N_TRAJ * N_TURNS

        expanded = [
            multimodal_expand.expand(p, args=_Args())
            for p in received["multimodal_lazy_payloads"]
        ]
        assert len(expanded) == N_TRAJ * N_TURNS
        for mm in expanded:
            assert "pixel_values" in mm
            assert "image_grid_thw" in mm
            assert mm["pixel_values"].dtype == torch.bfloat16

        all_pv = torch.cat([mm["pixel_values"] for mm in expanded], dim=0)
        all_grid = torch.cat([mm["image_grid_thw"] for mm in expanded], dim=0)
        assert all_grid.shape[0] == N_TRAJ * N_TURNS
        rows_per_image = (all_grid[:, 1] * all_grid[:, 2]).sum().item()
        assert all_pv.shape[0] == rows_per_image
    finally:
        multimodal_expand._processor = None
