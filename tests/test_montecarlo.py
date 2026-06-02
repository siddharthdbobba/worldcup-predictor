# tests/test_montecarlo.py
import numpy as np
from worldcup.models import MatchModelParams
from worldcup.simulator import simulate_once, run_simulation


def _synthetic_world():
    """12 groups of 4 = 48 teams; one dominant team."""
    groups, ratings = {}, {}
    labels = "ABCDEFGHIJKL"
    n = 0
    for g in labels:
        members = []
        for _ in range(4):
            name = f"T{n}"
            ratings[name] = 1800.0
            members.append(name)
            n += 1
        groups[g] = members
    ratings["T0"] = 2500.0  # dominant
    return groups, ratings


def test_simulate_once_returns_a_real_team():
    p = MatchModelParams()
    groups, ratings = _synthetic_world()
    champ = simulate_once(ratings, groups, np.random.default_rng(0), p)
    assert champ in ratings


def test_run_simulation_probabilities_sum_to_one_and_are_seed_stable():
    groups, ratings = _synthetic_world()
    r1 = run_simulation(ratings, groups, n=300, seed=42)
    r2 = run_simulation(ratings, groups, n=300, seed=42)
    assert r1 == r2                                  # deterministic
    assert abs(sum(r1.values()) - 1.0) < 1e-9        # normalized
    assert set(r1) == set(ratings)                   # every team present


def test_dominant_team_has_highest_probability():
    groups, ratings = _synthetic_world()
    res = run_simulation(ratings, groups, n=500, seed=1)
    assert max(res, key=res.get) == "T0"
