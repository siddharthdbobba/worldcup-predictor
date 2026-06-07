import json
import math
from pathlib import Path

from worldcup.markets import combine_markets, parse_kalshi, parse_polymarket

FIX = Path(__file__).parent / "fixtures"


def test_parse_polymarket():
    data = json.loads((FIX / "polymarket_sample.json").read_text())
    lines = parse_polymarket(data)
    assert math.isclose(lines["France"]["prob"], 0.17)
    assert math.isclose(lines["France"]["ask"], 0.18)
    assert lines["France"]["vol"] == 8000000
    assert lines["France"]["depth"] == 250000


def test_parse_polymarket_missing_liquidity_is_zero():
    # gamma-api omits liquidityNum on some markets; must not crash.
    payload = {"markets": [{"groupItemTitle": "Spain", "lastTradePrice": 0.16,
                            "bestAsk": 0.17, "volumeNum": 1000}]}
    lines = parse_polymarket(payload)
    assert lines["Spain"]["depth"] == 0.0
    assert lines["Spain"]["vol"] == 1000


def test_parse_kalshi_uses_dollar_fields():
    data = json.loads((FIX / "kalshi_sample.json").read_text())
    lines = parse_kalshi(data)
    assert math.isclose(lines["France"]["prob"], 0.17)   # dollars, no /100
    assert math.isclose(lines["France"]["ask"], 0.19)
    assert lines["France"]["vol"] == 5000000


def test_combine_devigs_and_volume_weights_to_consensus():
    poly = {"France": {"prob": 0.6, "ask": 0.61, "vol": 100.0, "depth": 100.0},
            "Spain": {"prob": 0.6, "ask": 0.61, "vol": 100.0, "depth": 100.0}}  # sums 1.2
    kalshi = {"France": {"prob": 0.5, "ask": 0.51, "vol": 300.0, "depth": 0.0},
              "Spain": {"prob": 0.5, "ask": 0.51, "vol": 300.0, "depth": 0.0}}
    prob, ask, conf, depth = combine_markets(poly, kalshi)
    assert math.isclose(sum(prob.values()), 1.0, abs_tol=1e-9)   # de-vigged + renormalized
    assert math.isclose(ask["France"], 0.51)                     # cheaper ask wins
    assert math.isclose(conf["France"], 2.0)                     # both books at their max vol
    assert depth["France"] == 100.0                              # summed dollar depth


def test_combine_normalizes_under_asymmetric_volume():
    poly = {"France": {"prob": 0.6, "ask": 0.61, "vol": 100.0, "depth": 0.0},
            "Spain": {"prob": 0.4, "ask": 0.41, "vol": 300.0, "depth": 0.0}}
    kalshi = {"France": {"prob": 0.5, "ask": 0.51, "vol": 300.0, "depth": 0.0},
              "Spain": {"prob": 0.5, "ask": 0.51, "vol": 100.0, "depth": 0.0}}
    prob, _, _, _ = combine_markets(poly, kalshi)
    assert abs(sum(prob.values()) - 1.0) < 1e-9


def test_combine_intersects_valid_teams():
    # A stale, non-qualified market priced at 1.0 must be excluded by valid_teams.
    poly = {"France": {"prob": 0.5, "ask": 0.5, "vol": 100.0, "depth": 0.0},
            "Peru": {"prob": 1.0, "ask": 1.0, "vol": 1.0, "depth": 0.0}}
    prob, _, _, _ = combine_markets(poly, {}, valid_teams={"France"})
    assert set(prob) == {"France"}
    assert math.isclose(prob["France"], 1.0)


def test_parse_polymarket_empty_list_returns_empty():
    assert parse_polymarket([]) == {}


def test_parse_polymarket_captures_bid_and_derived_no():
    data = json.loads((FIX / "polymarket_sample.json").read_text())
    lines = parse_polymarket(data)
    assert math.isclose(lines["France"]["yes_bid"], 0.16)
    assert math.isclose(lines["France"]["no_ask"], 1 - 0.16)  # derived 1 - bestBid


def test_parse_kalshi_captures_bid_and_quoted_no():
    data = json.loads((FIX / "kalshi_sample.json").read_text())
    lines = parse_kalshi(data)
    assert math.isclose(lines["France"]["yes_bid"], 0.16)
    assert math.isclose(lines["France"]["no_ask"], 0.84)   # quoted directly
    assert math.isclose(lines["France"]["no_bid"], 0.81)
