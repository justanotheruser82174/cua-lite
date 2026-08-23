"""Dual-clip PPO (arXiv:1912.09729) regression tests.

History: slime's ``loss.py`` called ``compute_policy_loss`` WITHOUT the
``eps_clip_c`` argument, so dual-clip was silently inert even with
``--eps-clip-c 3.0`` configured. The migration fixed the call site. These
tests pin (a) the dual-clip math itself and (b) that the loss call site keeps
threading ``args.eps_clip_c`` (the exact silent regression we had).

Slime-required.

Usage:
    pytest tests/train/utils/test_dual_clip.py -v
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("slime.utils", reason="slime not installed")

from slime.utils.ppo_utils import compute_policy_loss  # noqa: E402

EPS, EPS_HIGH = 0.2, 0.28


def _loss(ratio, adv, c):
    """compute_policy_loss takes ppo_kl = old - new = -log(ratio)."""
    ppo_kl = -torch.log(torch.tensor([float(ratio)]))
    advantages = torch.tensor([float(adv)])
    pg, _ = compute_policy_loss(ppo_kl, advantages, EPS, EPS_HIGH, c)
    return pg.item()


def test_positive_advantage_unaffected_by_dual_clip():
    """Dual-clip only gates A<0; positive-advantage losses must be identical."""
    for ratio in (0.5, 1.0, 10.0):
        assert _loss(ratio, +1.0, None) == pytest.approx(_loss(ratio, +1.0, 3.0))


def test_negative_advantage_large_ratio_is_capped_at_c():
    """The dual-clip region: A<0 and ratio >> 1+eps_high. Single-clip explodes
    with ratio; dual-clip caps the loss at -c*A = c*|A|."""
    ratio, adv, c = 10.0, -1.0, 3.0
    uncapped = _loss(ratio, adv, None)
    capped = _loss(ratio, adv, c)
    assert uncapped == pytest.approx(ratio * abs(adv))  # -ratio*A = 10
    assert capped == pytest.approx(c * abs(adv))        # -c*A = 3
    assert capped < uncapped


def test_negative_advantage_small_ratio_unchanged():
    """Inside the trust region (ratio ≈ 1), dual-clip must not kick in:
    min(-c*A, clip1) keeps clip1 because clip1 < c*|A|."""
    assert _loss(1.0, -1.0, 3.0) == pytest.approx(_loss(1.0, -1.0, None))


def test_eps_clip_c_must_exceed_one():
    with pytest.raises(AssertionError):
        _loss(1.0, -1.0, 0.9)


def test_loss_call_site_threads_eps_clip_c():
    """Regression pin for the original bug: the policy-loss call site must
    pass args.eps_clip_c. Source-level check (the full function needs a live
    megatron context, so we assert on the call expression instead)."""
    import inspect

    loss_mod = pytest.importorskip(
        "slime.backends.megatron_utils.loss", reason="needs megatron (training image)"
    )
    src = inspect.getsource(loss_mod.policy_loss_function)
    call = src.split("compute_policy_loss(", 1)[1].split(")", 1)[0]
    assert "eps_clip_c" in call, (
        "policy_loss_function no longer passes args.eps_clip_c to "
        "compute_policy_loss — dual-clip would be silently disabled again"
    )
