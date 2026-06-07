# tests/test_stake.py
import math
from worldcup.stake import find_value_bets


def test_only_positive_ev_teams_kept_and_edge_uses_model_minus_market():
    model = {"A": 0.30, "B": 0.10, "C": 0.20}
    market = {"A": 0.22, "B": 0.12, "C": 0.20}
    ask = {"A": 0.25, "B": 0.13, "C": 0.20}      # +EV only where model > ask
    bets = find_value_bets(model, market, ask)
    teams = {b["team"] for b in bets}
    assert teams == {"A"}                          # B and C are not +EV
    a = bets[0]
    assert math.isclose(a["edge"], 0.30 - 0.22, abs_tol=1e-9)   # model - market
    assert math.isclose(a["ev"], 0.30 / 0.25 - 1, abs_tol=1e-9)


def test_no_value_returns_empty():
    model = {"A": 0.20}
    market = {"A": 0.25}
    ask = {"A": 0.26}
    assert find_value_bets(model, market, ask) == []


def test_min_edge_gate_filters_small_edges_but_default_shows_all():
    model = {"Big": 0.30, "Tiny": 0.05}      # edges vs market: 0.08 and 0.02
    market = {"Big": 0.22, "Tiny": 0.03}
    ask = {"Big": 0.25, "Tiny": 0.035}       # both +EV at the ask
    # Default (min_edge=0): both qualify.
    assert {b["team"] for b in find_value_bets(model, market, ask)} == {"Big", "Tiny"}
    # With a 5pp gate, only the meaningful disagreement survives.
    gated = find_value_bets(model, market, ask, min_edge=0.05)
    assert {b["team"] for b in gated} == {"Big"}


from worldcup.stake import kelly_allocate, recommend_bets


def test_single_bet_matches_closed_form_kelly():
    # One +EV bet; full Kelly should approach (p-a)/(1-a).
    bets = [{"team": "A", "ask": 0.40, "p": 0.60, "market": 0.45,
             "edge": 0.15, "ev": 0.5}]
    recs = kelly_allocate(bets, bankroll=1000.0, kelly_fraction=1.0)
    expected_fraction = (0.60 - 0.40) / (1 - 0.40)   # = 1/3
    assert abs(recs[0].stake - expected_fraction * 1000.0) < 15.0


def test_total_stake_never_exceeds_bankroll():
    bets = [
        {"team": "A", "ask": 0.20, "p": 0.40, "market": 0.25, "edge": 0.15, "ev": 1.0},
        {"team": "B", "ask": 0.10, "p": 0.25, "market": 0.12, "edge": 0.13, "ev": 1.5},
    ]
    recs = kelly_allocate(bets, bankroll=500.0, kelly_fraction=0.5)
    assert sum(r.stake for r in recs) <= 500.0 + 1e-6


def test_liquidity_caps_stake():
    bets = [{"team": "A", "ask": 0.40, "p": 0.90, "market": 0.45,
             "edge": 0.45, "ev": 1.25}]
    recs = kelly_allocate(bets, bankroll=1000.0, kelly_fraction=1.0,
                          liquidity={"A": 25.0})
    assert recs[0].stake <= 25.0 + 1e-9


def test_recommend_bets_empty_when_no_value():
    recs = recommend_bets({"A": 0.20}, {"A": 0.25}, {"A": 0.26},
                          bankroll=100.0)
    assert recs == []


def test_incoherent_probabilities_above_one_raise():
    import pytest
    bets = [
        {"team": "A", "ask": 0.30, "p": 0.60, "market": 0.35, "edge": 0.25, "ev": 1.0},
        {"team": "B", "ask": 0.30, "p": 0.60, "market": 0.35, "edge": 0.25, "ev": 1.0},
    ]  # p sums to 1.2 -> incoherent
    with pytest.raises(ValueError):
        kelly_allocate(bets, bankroll=1000.0)
