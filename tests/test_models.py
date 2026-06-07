import json

from worldcup.models import Team, MatchModelParams, Forecast, BetRec, load_params


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


def test_matchmodelparams_dict_roundtrip():
    p = MatchModelParams(base=1.5, scale=1234.0, host_bump=50.0)
    p2 = MatchModelParams.from_dict(p.to_dict())
    assert (p2.base, p2.scale, p2.host_bump) == (1.5, 1234.0, 50.0)
    assert set(p2.hosts) == set(p.hosts)


def test_from_dict_uses_defaults_for_missing_keys():
    p = MatchModelParams.from_dict({"scale": 999.0})
    d = MatchModelParams()
    assert p.scale == 999.0 and p.base == d.base and set(p.hosts) == set(d.hosts)


def test_load_params_absent_present_and_corrupt(tmp_path):
    assert load_params(tmp_path / "nope.json") == MatchModelParams()        # absent → defaults
    good = tmp_path / "calibrated_params.json"
    good.write_text(json.dumps({"base": 1.42, "scale": 1100.0}))
    p = load_params(good)
    assert p.base == 1.42 and p.scale == 1100.0                              # present → loaded
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert load_params(bad) == MatchModelParams()                           # corrupt → defaults
