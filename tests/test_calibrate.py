# tests/test_calibrate.py
import numpy as np
import pytest

from worldcup.calibrate import CalibrationResult, fit_params, rescale_to_live
from worldcup.elo_engine import MatchElo


def _synth(n, base, scale, seed=0, neutral=True):
    """Synthesize MatchElo rows whose goals are Poisson(base*10^(Δ/scale))."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        ea = rng.uniform(1500, 2100)
        eb = rng.uniform(1500, 2100)
        la = base * 10 ** ((ea - eb) / scale)
        lb = base * 10 ** ((eb - ea) / scale)
        rows.append(MatchElo(elo_home=ea, elo_away=eb,
                             goals_home=int(rng.poisson(la)),
                             goals_away=int(rng.poisson(lb)), neutral=neutral))
    return rows


def test_fit_recovers_known_params():
    rows = _synth(4000, base=1.5, scale=1200.0, seed=42)
    res = fit_params(rows, neutral_only=True)
    assert isinstance(res, CalibrationResult)
    assert res.n_matches == 4000
    assert abs(res.base - 1.5) < 0.15
    assert abs(res.scale - 1200.0) < 200.0


def test_fit_improves_loglik_vs_defaults():
    rows = _synth(2000, base=1.5, scale=1200.0, seed=1)
    res = fit_params(rows)                       # x0 defaults to current (1.35, 2000)
    assert res.ll_after >= res.ll_before


def test_fit_empty_raises():
    with pytest.raises(ValueError):
        fit_params([])


def test_neutral_only_filters_non_neutral():
    rows = _synth(50, 1.4, 1500.0, seed=2, neutral=True) + \
           _synth(50, 1.4, 1500.0, seed=3, neutral=False)
    res = fit_params(rows, neutral_only=True)
    assert res.n_matches == 50                   # only the neutral half is used


def test_rescale_to_live_by_spread_ratio():
    replay_elos = [1400, 1600, 1400, 1600]            # std 100
    live = {"a": 1300, "b": 1700, "c": 1300, "d": 1700}  # std 200
    assert abs(rescale_to_live(1000.0, replay_elos, live) - 2000.0) < 1e-6   # ×(200/100)
    assert rescale_to_live(1000.0, [1500, 1500], live) == 1000.0             # degenerate → unchanged


def test_run_calibrate_cli_writes_params(monkeypatch, tmp_path, capsys):
    import json
    import main as cli
    from worldcup.elo_engine import Match
    matches = [Match("2020-01-01", "A", "B", 2, 0, neutral=True, tournament="Friendly"),
               Match("2020-02-01", "B", "A", 1, 0, neutral=True, tournament="Friendly"),
               Match("2020-03-01", "A", "B", 3, 1, neutral=True, tournament="Friendly")]
    monkeypatch.setattr(cli, "fetch_matches", lambda *, since_year=None, url=None: matches)
    monkeypatch.setattr(cli, "fetch_ratings", lambda: {"A": 1800.0, "B": 1500.0})
    out = tmp_path / "calibrated_params.json"
    monkeypatch.setattr(cli, "CALIBRATED_PARAMS_PATH", out)
    cli.run_calibrate_cli(since_year=2019, all_matches=False, history_url=None)
    assert out.exists()
    d = json.loads(out.read_text())
    assert "base" in d and "scale" in d and d["_meta"]["n_matches"] == 3
    assert "scale_fit_replay_units" in d["_meta"]
    assert "Calibration" in capsys.readouterr().out
