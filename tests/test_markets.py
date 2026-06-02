import json
import math
from pathlib import Path
from worldcup.markets import parse_polymarket, parse_kalshi, combine_markets

FIX = Path(__file__).parent / "fixtures"


def test_parse_polymarket():
    data = json.loads((FIX / "polymarket_sample.json").read_text())
    lines = parse_polymarket(data)
    assert math.isclose(lines["France"]["prob"], 0.17)
    assert math.isclose(lines["France"]["ask"], 0.18)
    assert lines["France"]["liquidity"] == 250000


def test_parse_kalshi_converts_cents():
    data = json.loads((FIX / "kalshi_sample.json").read_text())
    lines = parse_kalshi(data)
    assert math.isclose(lines["France"]["prob"], 0.17)
    assert math.isclose(lines["France"]["ask"], 0.19)


def test_combine_devigs_and_liquidity_weights():
    poly = {"France": {"prob": 0.6, "ask": 0.61, "liquidity": 100.0},
            "Spain": {"prob": 0.6, "ask": 0.61, "liquidity": 100.0}}  # sums to 1.2
    kalshi = {"France": {"prob": 0.5, "ask": 0.51, "liquidity": 300.0},
              "Spain": {"prob": 0.5, "ask": 0.51, "liquidity": 300.0}}
    prob, ask, liq = combine_markets(poly, kalshi)
    assert math.isclose(prob["France"] + prob["Spain"], 1.0, abs_tol=1e-9)  # de-vigged
    assert liq["France"] == 400.0                                           # summed
