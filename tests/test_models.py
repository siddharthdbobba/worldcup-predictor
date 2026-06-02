from worldcup.models import Team, MatchModelParams, Forecast, BetRec


def test_team_holds_name_and_rating():
    t = Team(name="France", rating=2100.0)
    assert t.name == "France" and t.rating == 2100.0


def test_default_match_params_have_hosts():
    p = MatchModelParams()
    assert set(p.hosts) == {"United States", "Canada", "Mexico"}
    assert p.base > 0 and p.scale > 0


def test_forecast_and_betrec_fields():
    f = Forecast(team="Spain", model_pct=0.15, market_pct=0.14, blended_pct=0.145)
    assert f.blended_pct == 0.145
    b = BetRec(team="Spain", ask=0.16, model_pct=0.15, market_pct=0.14,
               edge=0.01, ev_pct=-0.0625, stake=10.0, potential_profit=52.5)
    assert b.stake == 10.0
