# src/worldcup/calibrate.py
"""Maximum-likelihood calibration of the goals model's `base`/`scale`.

Given matches annotated with PRE-match Elo (`elo_engine.MatchElo`), fit the two
parameters of `λ = base · 10^((Δelo)/scale)` by maximizing the Poisson likelihood
of the observed scorelines. Mirrors the scipy `minimize` pattern in `stake.py`.

Independence assumption: home and away goals are modeled as independent Poisson
(no Dixon-Coles low-score correction) — a deliberate simplification.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from worldcup.elo_engine import HOME_ADV, MatchElo

# Plausible search bounds for (base goals/team, Elo->goals scale).
_BOUNDS = [(0.1, 5.0), (200.0, 6000.0)]
_DEFAULT_X0 = (1.35, 2000.0)        # the current hand-tuned defaults


@dataclass(frozen=True)
class CalibrationResult:
    base: float
    scale: float
    n_matches: int
    ll_before: float        # total log-likelihood at x0 (current defaults)
    ll_after: float         # total log-likelihood at the fitted params
    converged: bool


def _arrays(rows: list[MatchElo], neutral_only: bool, home_adv_elo: float):
    """Return (eff_elo_home, elo_away, goals_home, goals_away) numpy arrays.

    Non-neutral matches (when included) get `home_adv_elo` added to the home side,
    consistent with how the simulator applies the host bump.
    """
    if neutral_only:
        rows = [r for r in rows if r.neutral]
    ea = np.array([r.elo_home + (0.0 if r.neutral else home_adv_elo) for r in rows], float)
    eb = np.array([r.elo_away for r in rows], float)
    ga = np.array([r.goals_home for r in rows], float)
    gb = np.array([r.goals_away for r in rows], float)
    return ea, eb, ga, gb


def fit_params(rows: list[MatchElo], *, neutral_only: bool = True,
               x0: tuple[float, float] = _DEFAULT_X0,
               home_adv_elo: float = HOME_ADV) -> CalibrationResult:
    """Fit (base, scale) by Poisson MLE over the supplied matches."""
    ea, eb, ga, gb = _arrays(rows, neutral_only, home_adv_elo)
    n = len(ea)
    if n == 0:
        raise ValueError("no matches to calibrate on (after filtering)")

    def neg_ll(theta: np.ndarray) -> float:
        base, scale = float(theta[0]), float(theta[1])
        la = base * 10.0 ** ((ea - eb) / scale)
        lb = base * 10.0 ** ((eb - ea) / scale)
        return -(poisson.logpmf(ga, la).sum() + poisson.logpmf(gb, lb).sum())

    res = minimize(neg_ll, x0=np.array(x0, float), method="SLSQP",
                   bounds=_BOUNDS, options={"maxiter": 1000, "ftol": 1e-9})
    return CalibrationResult(
        base=float(res.x[0]), scale=float(res.x[1]), n_matches=n,
        ll_before=-neg_ll(np.array(x0, float)), ll_after=-neg_ll(res.x),
        converged=bool(res.success),
    )


def rescale_to_live(fitted_scale: float, replay_elos, live_ratings: dict[str, float]) -> float:
    """Convert a `scale` fitted on replayed Elo to the live ratings' Elo spread.

    `scale` is relative to the spread of whatever Elo it was fit on: `λ ∝ 10^(Δ/scale)`.
    Our replay produces a NARROWER spread than eloratings.net's published ratings (which
    the forecaster consumes), so the raw fit would over-dramatize live Elo gaps. Multiply
    by `std(live) / std(replay)` so the fitted sensitivity transfers to live units.
    Returns the fitted scale unchanged if the replay spread is degenerate.
    """
    rep_std = float(np.std(np.asarray(list(replay_elos), float))) if len(replay_elos) else 0.0
    live_std = float(np.std(np.asarray(list(live_ratings.values()), float))) if live_ratings else 0.0
    if rep_std <= 0 or live_std <= 0:
        return fitted_scale
    return fitted_scale * (live_std / rep_std)
