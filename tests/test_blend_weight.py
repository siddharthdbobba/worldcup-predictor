from worldcup.blend import market_weight, blend


def test_market_weight_matches_inline_formula():
    # w = w_cap * L/(L+K); defaults w_cap=0.7, K=0.5
    assert abs(market_weight(0.5) - 0.7 * (0.5 / 1.0)) < 1e-12
    assert market_weight(0.0) == 0.0
    assert market_weight(-5.0) == 0.0          # clamped at 0


def test_blend_still_uses_the_helper_equivalently():
    model = {"A": 0.6, "B": 0.4}
    market = {"A": 0.5, "B": 0.5}
    conf = {"A": 0.5, "B": 0.5}
    out = blend(model, market, conf, w_cap=0.7, K=0.5)
    w = market_weight(0.5)                       # 0.7 * 0.5/1.0 = 0.35
    raw_a = w * 0.5 + (1 - w) * 0.6
    raw_b = w * 0.5 + (1 - w) * 0.4
    total = raw_a + raw_b
    assert abs(out["A"] - raw_a / total) < 1e-9
    assert abs(out["B"] - raw_b / total) < 1e-9
