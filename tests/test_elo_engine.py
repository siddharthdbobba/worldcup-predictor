# tests/test_elo_engine.py
from worldcup.elo_engine import Match, MatchElo, replay, _k_factor


def _m(date, h, a, gh, ga, neutral=True, tournament="Friendly"):
    return Match(date=date, home=h, away=a, goals_home=gh, goals_away=ga,
                 neutral=neutral, tournament=tournament)


def test_replay_emits_pre_match_elo_then_updates():
    out = replay([_m("2020-01-01", "A", "B", 2, 0),
                  _m("2020-02-01", "A", "B", 0, 1)], init=1500.0)
    assert len(out) == 2 and isinstance(out[0], MatchElo)
    # first match: both teams start at init
    assert out[0].elo_home == 1500.0 and out[0].elo_away == 1500.0
    # A won the first → by the second match A is rated above B
    assert out[1].elo_home > 1500.0 and out[1].elo_away < 1500.0


def test_winner_gains_equal_loser_loss_zero_sum():
    out = replay([_m("2020-01-01", "A", "B", 3, 0),
                  _m("2020-01-02", "A", "B", 0, 0)], init=1500.0)
    # zero-sum update: A's gain == B's loss (pre-elo of the 2nd match sums to 2*init)
    assert abs((out[1].elo_home + out[1].elo_away) - 3000.0) < 1e-9
    assert out[1].elo_home > 1500.0


def test_new_team_seeds_at_init():
    out = replay([_m("2020-01-01", "X", "Y", 1, 1)], init=1500.0)
    assert out[0].elo_home == 1500.0 and out[0].elo_away == 1500.0


def test_replay_sorts_by_date():
    out = replay([_m("2021-01-01", "A", "B", 1, 0),
                  _m("2020-01-01", "A", "B", 1, 0)])
    # chronological: the 2020 match is processed first, both at init
    assert out[0].elo_home == 1500.0 and out[0].elo_away == 1500.0


def test_goal_difference_amplifies_update():
    big = replay([_m("2020-01-01", "A", "B", 5, 0), _m("2020-01-02", "A", "B", 0, 0)])
    small = replay([_m("2020-01-01", "A", "B", 1, 0), _m("2020-01-02", "A", "B", 0, 0)])
    assert (big[1].elo_home - 1500.0) > (small[1].elo_home - 1500.0)


def test_home_advantage_applied_when_not_neutral():
    # Equal ratings, home win by 1: a non-neutral home win moves the rating LESS than a
    # neutral one (the home team was already expected to do better, so beats expectation by less).
    neutral = replay([_m("2020-01-01", "A", "B", 1, 0, neutral=True),
                      _m("2020-01-02", "A", "B", 0, 0)])
    home = replay([_m("2020-01-01", "A", "B", 1, 0, neutral=False),
                   _m("2020-01-02", "A", "B", 0, 0)])
    assert (home[1].elo_home - 1500.0) < (neutral[1].elo_home - 1500.0)


def test_k_factor_importance_ordering():
    assert _k_factor("FIFA World Cup") > _k_factor("FIFA World Cup qualification")
    assert _k_factor("FIFA World Cup qualification") > _k_factor("Friendly")
