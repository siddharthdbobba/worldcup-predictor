from worldcup.agent import run_pipeline


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
    liquidity = {"T0": 1e6, "T1": 1e6}

    result = run_pipeline(ratings, groups, market, ask, liquidity,
                          bankroll=100.0, n_sims=200, seed=1)
    assert abs(sum(f.blended_pct for f in result.forecasts) - 1.0) < 1e-6
    assert result.forecasts[0].team == "T0"                 # sorted, dominant first
    # T0 is +EV (model >> ask) so it should be recommended.
    assert any(b.team == "T0" for b in result.bets)
