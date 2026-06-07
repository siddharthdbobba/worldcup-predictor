# tests/test_groups.py
import numpy as np
from worldcup.models import MatchModelParams
from worldcup.simulator import simulate_group, rank_thirds


def test_simulate_group_returns_four_ranked_and_stats():
    p = MatchModelParams()
    teams = ["A", "B", "C", "D"]
    ratings = {"A": 2200.0, "B": 1900.0, "C": 1850.0, "D": 1700.0}
    ranked, stats = simulate_group(teams, ratings, np.random.default_rng(1), p)
    assert len(ranked) == 4
    assert set(ranked) == set(teams)
    for t in teams:
        assert {"pts", "gf", "ga"} <= set(stats[t])


def test_group_ranking_orders_by_points_then_gd_then_gf():
    # Hand-built stats; verify the sort key via the helper.
    from worldcup.simulator import _rank_key
    stats = {
        "A": {"pts": 6, "gf": 4, "ga": 1},
        "B": {"pts": 6, "gf": 5, "ga": 1},   # same pts, better GD
        "C": {"pts": 3, "gf": 9, "ga": 2},
        "D": {"pts": 0, "gf": 0, "ga": 9},
    }
    rng = np.random.default_rng(0)
    order = sorted(stats, key=lambda t: _rank_key(stats[t], rng))
    assert order == ["B", "A", "C", "D"]


def test_rank_thirds_takes_best_eight():
    rng = np.random.default_rng(0)
    thirds = [(f"T{i}", {"pts": i, "gf": i, "ga": 0}) for i in range(12)]
    best = rank_thirds(thirds, rng)
    assert len(best) == 8
    assert "T11" in best and "T0" not in best
