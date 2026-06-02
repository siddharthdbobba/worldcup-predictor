# tests/test_blend.py
import math
from worldcup.blend import blend


def test_blend_sums_to_one():
    model = {"A": 0.5, "B": 0.3, "C": 0.2}
    market = {"A": 0.4, "B": 0.4, "C": 0.2}
    liq = {"A": 1e6, "B": 1e6, "C": 1e6}
    out = blend(model, market, liq, w_cap=0.7, K=1e6)
    assert math.isclose(sum(out.values()), 1.0, abs_tol=1e-9)


def test_unpriced_team_falls_back_to_model_then_renormalizes():
    model = {"A": 0.6, "B": 0.4}
    market = {"A": 0.5}                       # B unpriced
    liq = {"A": 1e9}                          # A fully liquid
    out = blend(model, market, liq, w_cap=0.7, K=1e6)
    # A pulled toward market (0.5), B stays at model pre-normalization.
    assert out["A"] < 0.6
    assert math.isclose(sum(out.values()), 1.0, abs_tol=1e-9)


def test_zero_liquidity_team_is_model_only_before_norm():
    model = {"A": 0.5, "B": 0.5}
    market = {"A": 0.9, "B": 0.1}
    liq = {"A": 0.0, "B": 0.0}               # no confidence => model only
    out = blend(model, market, liq, w_cap=0.7, K=1e6)
    assert math.isclose(out["A"], out["B"], abs_tol=1e-9)


def test_fixed_weight_mode():
    model = {"A": 0.5, "B": 0.5}
    market = {"A": 1.0, "B": 0.0}
    out = blend(model, market, {"A": 1.0, "B": 1.0}, fixed_w=0.5)
    # pre-norm: A=0.75, B=0.25 -> normalized identical ratio
    assert math.isclose(out["A"], 0.75, abs_tol=1e-9)
    assert math.isclose(out["B"], 0.25, abs_tol=1e-9)
