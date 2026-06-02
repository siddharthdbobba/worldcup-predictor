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
