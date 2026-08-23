"""ScaleCUA LibreOffice decimal display parity tests."""

from __future__ import annotations

import pytest

from lite.gym.envs.lite.scalecua.src.osworld import judges
from lite.gym.envs.lite.scalecua.src.utils import dataset


def _overlays_ready() -> bool:
    """Judge-overlay tests need the imported getters/metrics modules too."""
    return all(
        (root / f"{name}.py").is_file()
        for split in dataset.RUNTIME_SPLITS
        if (root := judges.overlay_dir(split)) is not None
        for name in ("getters", "metrics")
    )

# (value, decimals, what real LibreOffice 7.3.7.2 rendered)
#
# BOUNDARY block: the shortest repr needs 16/17 significant digits, so it
# straddles the tie from BELOW while LibreOffice's 15-digit view sits exactly ON
# it and rounds away from zero. `Decimal(str(v))` rounds these DOWN -> FN.
_LIBREOFFICE_TEXT_BOUNDARY = [
    (65 * 1.085, 2, "70.53"),  # f_calc_76 E7  (double 70.52499999999999)
    (38 * 1.0875, 2, "41.33"),  # f_calc_40 E12
    (54.8 * 1.0875, 2, "59.60"),  # f_calc_40 E15 - trailing zero kept
    (36.4 * 1.0875, 2, "39.59"),  # f_calc_40 E17
    (8.2 / (32.0 / 60.0), 2, "15.38"),  # f_calc_44 E17
    (-(65 * 1.085), 2, "-70.53"),  # half-AWAY-from-zero, not half-up
    (863.4314999999999, 3, "863.432"),
    (1867.3244999999997, 3, "1867.325"),
    (332.77549999999997, 3, "332.776"),
    (1033.7694999999999, 3, "1033.770"),  # trailing zero kept
    (1987.3149999999998, 2, "1987.32"),
    (946.5749999999999, 2, "946.58"),
    (226.17499999999998, 2, "226.18"),
    (471.09499999999997, 2, "471.10"),
]

# CONTROL block: the 15-digit snap must not perturb anything that was already
# right. Both the old and the current form produce these.
_LIBREOFFICE_TEXT_CONTROL = [
    (1412.175, 2, "1412.18"),
    (4.2345, 3, "4.235"),
    (4.2345, 2, "4.23"),
    (2.675, 2, "2.68"),  # double is 2.67499999999999982, still "2.68"
    (1.005, 2, "1.01"),  # `1.0049999999999999` is the SAME double
    (0.145, 2, "0.15"),
    (2.5, 2, "2.50"),
    (2.5, 3, "2.500"),
    (1.25, 2, "1.25"),
    (2.0, 2, "2.00"),
    (33.33333333333333 * 3, 2, "100.00"),  # 99.99999999999999 -> LO sees 100
]


@pytest.mark.parametrize("value,decimals,rendered", _LIBREOFFICE_TEXT_BOUNDARY)
def test_calc_decimal_display_matches_libreoffice_at_the_15_digit_boundary(
    value, decimals, rendered
):
    """The judge must render what Calc renders, or it false-fails a correct agent."""
    assert judges._calc_decimal_display(value, decimals) == rendered


@pytest.mark.parametrize("value,decimals,rendered", _LIBREOFFICE_TEXT_CONTROL)
def test_calc_decimal_display_leaves_non_boundary_values_alone(value, decimals, rendered):
    assert judges._calc_decimal_display(value, decimals) == rendered


def test_calc_decimal_display_snaps_to_15_significant_digits_not_the_repr():
    value = 65 * 1.085
    assert repr(value) == "70.52499999999999"
    assert f"{value:.15g}" == "70.525"
    assert judges._calc_decimal_display(value, 2) == "70.53"


def test_calc_decimal_display_is_the_declared_mirror_of_the_gold_shims():
    from decimal import ROUND_HALF_UP, Decimal

    def gold(x, n):
        return str(Decimal(f"{x:.15g}").quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP))

    for value, decimals, rendered in _LIBREOFFICE_TEXT_BOUNDARY + _LIBREOFFICE_TEXT_CONTROL:
        assert judges._calc_decimal_display(value, decimals) == gold(value, decimals) == rendered


@pytest.mark.skipif(not _overlays_ready(), reason="lite.scalecua judge overlays not imported")
def test_calc_decimal_display_is_libreoffice_half_up():
    # eval-side rounding class: LibreOffice's TEXT/FIXED display rounds
    # half-AWAY-from-zero; Python f"{v:.Nf}" uses banker's (half-to-even), which
    # false-failed a correct agent (2.5 -> "2" vs LO/gold "3"). The display helper
    # must match LibreOffice; trailing zeros preserved.
    assert judges._calc_decimal_display(2.5, 0) == "3"  # banker's would give "2"
    assert judges._calc_decimal_display(1.25, 1) == "1.3"  # banker's -> "1.2"
    assert judges._calc_decimal_display(4.2345, 3) == "4.235"
    assert judges._calc_decimal_display(2.0, 2) == "2.00"
