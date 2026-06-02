# tests/test_simulator.py
import numpy as np
from worldcup.models import MatchModelParams
from worldcup.simulator import expected_goals, sample_goals, elo_winprob


def test_equal_ratings_give_equal_expected_goals():
    p = MatchModelParams()
    la, lb = expected_goals(2000, 2000, p)
    assert la == lb == p.base


def test_stronger_team_has_higher_expected_goals():
    p = MatchModelParams()
    la, lb = expected_goals(2200, 1800, p)
    assert la > lb


def test_sample_goals_is_deterministic_with_seed():
    p = MatchModelParams()
    ratings = {"A": 2000.0, "B": 1900.0}
    g1 = sample_goals("A", "B", ratings, np.random.default_rng(7), p)
    g2 = sample_goals("A", "B", ratings, np.random.default_rng(7), p)
    assert g1 == g2
    assert all(isinstance(x, int) for x in g1)


def test_elo_winprob_symmetry():
    assert abs(elo_winprob(2000, 2000) - 0.5) < 1e-9
    assert elo_winprob(2200, 1800) > 0.5
