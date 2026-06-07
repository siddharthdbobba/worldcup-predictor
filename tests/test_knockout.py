# tests/test_knockout.py
import numpy as np
from worldcup.models import MatchModelParams
from worldcup.simulator import bracket_seed_order, play_match_ko, play_knockout


def test_bracket_seed_order_small():
    assert bracket_seed_order(4) == [1, 4, 2, 3]
    order = bracket_seed_order(32)
    assert len(order) == 32 and sorted(order) == list(range(1, 33))
    assert order[0] == 1 and order[1] == 32  # top seed meets bottom seed


def test_play_match_ko_returns_a_participant():
    p = MatchModelParams()
    ratings = {"A": 2000.0, "B": 1900.0}
    w = play_match_ko("A", "B", ratings, np.random.default_rng(3), p)
    assert w in ("A", "B")


def test_play_knockout_strongest_seed_usually_wins():
    p = MatchModelParams()
    teams = [f"S{i}" for i in range(32)]
    ratings = {t: 1800.0 for t in teams}
    ratings["S0"] = 2600.0  # dominant seed-1
    wins = sum(play_knockout(teams, ratings, np.random.default_rng(i), p) == "S0"
               for i in range(200))
    assert wins > 100  # dominant team wins a clear majority
