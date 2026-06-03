from worldcup.models import BetRec, Forecast
from worldcup.snapshot import build_snapshot


def _inputs():
    groups = {"A": ["Spain", "Brazil", "Japan", "Ghana"]}
    ratings = {"Spain": 2090.0, "Brazil": 2050.0, "Japan": 1850.0, "Ghana": 1700.0,
               "Italy": 1990.0}  # extra (undrawn) team must be dropped
    market = {"Spain": 0.18, "Brazil": 0.16}
    ask = {"Spain": 0.19, "Brazil": 0.17}
    confidence = {"Spain": 0.5, "Brazil": 0.4}
    depth = {"Spain": 4000.0, "Brazil": 3000.0}
    forecasts = [Forecast("Spain", 0.20, 0.18, 0.19), Forecast("Brazil", 0.15, 0.16, 0.155)]
    advancement = {t: {"r32": 1.0, "r16": 0.5, "qf": 0.3, "sf": 0.2, "final": 0.1,
                       "champion": 0.05} for t in ["Spain", "Brazil", "Japan", "Ghana"]}
    bets = [BetRec("Spain", 0.19, 0.20, 0.18, 0.02, 0.05, 12.0, 51.2)]
    return (groups, ratings, market, ask, confidence, depth, forecasts, advancement, bets)


def test_build_snapshot_shape_with_bankroll():
    (groups, ratings, market, ask, conf, depth, forecasts, adv, bets) = _inputs()
    snap = build_snapshot(
        groups=groups, ratings=ratings, market=market, ask=ask, confidence=conf,
        depth=depth, forecasts=forecasts, advancement=adv, bets=bets,
        bankroll=100.0, n_sims=20000, seed=42, kelly_fraction=0.5, min_edge=0.05,
        generated_at="2026-06-02T00:00:00Z")
    assert set(snap) == {"meta", "draw", "ratings", "forecasts", "market",
                         "blend_weights", "advancement", "bankroll", "bets",
                         "staked", "reserve"}
    assert snap["draw"] == groups
    assert set(snap["ratings"]) == {"Spain", "Brazil", "Japan", "Ghana"}  # Italy dropped
    assert snap["forecasts"][0]["team"] == "Spain"
    assert set(snap["market"]["Spain"]) == {"prob", "ask", "confidence", "depth"}
    assert 0.0 <= snap["blend_weights"]["Spain"] <= 0.7
    assert snap["meta"]["n_sims"] == 20000
    assert snap["bankroll"] == 100.0
    assert snap["staked"] == 12.0
    assert snap["reserve"] == 88.0


def test_build_snapshot_without_bankroll_omits_card():
    (groups, ratings, market, ask, conf, depth, forecasts, adv, _bets) = _inputs()
    snap = build_snapshot(
        groups=groups, ratings=ratings, market=market, ask=ask, confidence=conf,
        depth=depth, forecasts=forecasts, advancement=adv, bets=[],
        bankroll=None, n_sims=10, seed=1, kelly_fraction=0.5, min_edge=0.05,
        generated_at="2026-06-02T00:00:00Z")
    assert snap["bankroll"] is None
    assert snap["bets"] == []
    assert snap["staked"] is None
    assert snap["reserve"] is None
