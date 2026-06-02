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
