# src/worldcup/stake.py
"""Pure value detection + fractional-Kelly bankroll allocation.

Betting edge is model% - market% (NOT the liquidity-weighted blend) to avoid
circularity: blending in the same market we bet against would steer stakes onto
thin longshots. The +EV test and Kelly math use the ASK (the executable price).
"""
from __future__ import annotations


def find_value_bets(model: dict[str, float], market: dict[str, float],
                    ask: dict[str, float]) -> list[dict]:
    """Return +EV candidate bets (buying at ask), each with edge and EV.

    A bet is +EV when model% > ask. `edge` is the headline signal model - market;
    `ev` is expected value per $1 staked at the ask.
    """
    bets = []
    for team, p in model.items():
        a = ask.get(team)
        if a is None or a <= 0 or a >= 1:
            continue
        if p > a:
            bets.append({
                "team": team,
                "ask": a,
                "p": p,
                "market": market.get(team, a),
                "edge": p - market.get(team, a),
                "ev": p / a - 1.0,
            })
    return bets


import numpy as np
from scipy.optimize import minimize

from worldcup.models import BetRec


def kelly_allocate(bets: list[dict], bankroll: float, kelly_fraction: float = 0.5,
                   liquidity: dict[str, float] | None = None) -> list[BetRec]:
    """Allocate bankroll across mutually-exclusive +EV bets via fractional Kelly.

    Maximizes expected log-wealth; scales the optimum by `kelly_fraction`; caps
    each stake by available liquidity. Returns BetRec rows (zero-stake bets dropped).
    """
    if not bets or bankroll <= 0:
        return []
    liquidity = liquidity or {}
    p = np.array([b["p"] for b in bets])
    a = np.array([b["ask"] for b in bets])
    q0 = max(0.0, 1.0 - float(p.sum()))           # prob none of the bet teams win

    def neg_log_wealth(f: np.ndarray) -> float:
        spent = float(f.sum())
        if spent >= 1.0:
            return 1e9
        base = 1.0 - spent
        win_wealth = base + f / a                  # wealth/B if team i wins
        if np.any(win_wealth <= 0) or base <= 0:
            return 1e9
        return -(float((p * np.log(win_wealth)).sum()) + q0 * np.log(base))

    n = len(bets)
    res = minimize(
        neg_log_wealth, x0=np.full(n, 0.01),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "ineq", "fun": lambda f: 1.0 - f.sum() - 1e-6}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    fractions = np.clip(res.x, 0.0, 1.0) * kelly_fraction

    recs: list[BetRec] = []
    for b, frac in zip(bets, fractions):
        stake = frac * bankroll
        cap = liquidity.get(b["team"])
        if cap is not None:
            stake = min(stake, cap)
        if stake <= 1e-6:
            continue
        profit = stake * (1.0 - b["ask"]) / b["ask"]
        recs.append(BetRec(
            team=b["team"], ask=b["ask"], model_pct=b["p"], market_pct=b["market"],
            edge=b["edge"], ev_pct=b["ev"], stake=round(stake, 2),
            potential_profit=round(profit, 2),
        ))
    recs.sort(key=lambda r: r.stake, reverse=True)
    return recs


def recommend_bets(model: dict[str, float], market: dict[str, float],
                   ask: dict[str, float], bankroll: float,
                   kelly_fraction: float = 0.5,
                   liquidity: dict[str, float] | None = None) -> list[BetRec]:
    """End-to-end: detect +EV bets, then Kelly-allocate the bankroll across them."""
    return kelly_allocate(find_value_bets(model, market, ask),
                          bankroll, kelly_fraction, liquidity)
