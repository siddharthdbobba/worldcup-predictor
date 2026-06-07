from worldcup.agent import run_pipeline
from worldcup.models import MatchModelParams


def test_run_pipeline_end_to_end_with_injected_data():
    # 12x4 synthetic groups + ratings (no network).
    groups, ratings = {}, {}
    n = 0
    for g in "ABCDEFGHIJKL":
        members = []
        for _ in range(4):
            name = f"T{n}"; ratings[name] = 1800.0; members.append(name); n += 1
        groups[g] = members
    ratings["T0"] = 2400.0
    market = {"T0": 0.30, "T1": 0.10}
    ask = {"T0": 0.20, "T1": 0.12}
    confidence = {"T0": 1.5, "T1": 1.5}   # normalized [0, 2] market confidence
    depth = {"T0": 1e6, "T1": 1e6}        # dollar order-book depth (stake cap)

    result = run_pipeline(ratings, groups, market, ask, confidence, depth,
                          bankroll=100.0, n_sims=200, seed=1)
    assert abs(sum(f.blended_pct for f in result.forecasts) - 1.0) < 1e-6
    assert len(result.forecasts) == 48                      # only drawn teams
    assert result.forecasts[0].team == "T0"                 # sorted, dominant first
    # T0 is +EV (model >> ask) so it should be recommended.
    assert any(b.team == "T0" for b in result.bets)


def test_run_pipeline_raises_on_unrated_drawn_team():
    groups = {"A": ["X", "Y", "Z", "W"]}
    ratings = {"X": 1800.0, "Y": 1800.0, "Z": 1800.0}  # W missing
    import pytest
    with pytest.raises(ValueError):
        run_pipeline(ratings, groups, {}, {}, {}, bankroll=None, n_sims=10)


def test_run_pipeline_threads_calibrated_params():
    # A very flat scale makes even a big Elo gap barely matter → the dominant team's
    # model probability is much lower than under the default (top-heavy) params.
    groups, ratings = {}, {}
    n = 0
    for g in "ABCDEFGHIJKL":
        members = []
        for _ in range(4):
            name = f"T{n}"; ratings[name] = 1800.0; members.append(name); n += 1
        groups[g] = members
    ratings["T0"] = 2400.0
    default = run_pipeline(ratings, groups, {}, {}, {}, bankroll=None, n_sims=400, seed=7)
    flat = run_pipeline(ratings, groups, {}, {}, {}, bankroll=None, n_sims=400, seed=7,
                        params=MatchModelParams(base=1.35, scale=6000.0))
    p_default = next(f.model_pct for f in default.forecasts if f.team == "T0")
    p_flat = next(f.model_pct for f in flat.forecasts if f.team == "T0")
    assert p_flat < p_default            # flatter scale → dominant team less dominant
