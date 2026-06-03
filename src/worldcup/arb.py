"""Method A — cross-book arbitrage between Polymarket and Kalshi.

Pure, deterministic, fully tested. Operates on the RAW per-book price dicts
(not the blended consensus) because an arb must see the two books separately.

Lock condition (per team): buy YES on the cheaper book at ask `a`, NO on the
dearer book at ask `b` (NO ask = 1 - that book's YES bid for Polymarket; quoted
directly for Kalshi). Profit per equal contract-pair = 1 - a - b - fees. Buying
equal YES and NO contracts pays out identically whoever wins.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

VENUE_POLY = "polymarket"
VENUE_KALSHI = "kalshi"


@dataclass
class _Cand:
    """Internal lock candidate (one team, one orientation)."""
    team: str
    yes_venue: str
    no_venue: str
    a: float            # YES leg price
    b: float            # NO leg price
    fee_poly: float     # fee per pair attributable to the Polymarket leg
    fee_kalshi: float   # fee per pair attributable to the Kalshi leg
    fee: float          # total fee per pair
    profit: float       # 1 - a - b - fee  (per pair)
    yes_cap: float      # max $ stake on the YES leg
    no_cap: float       # max $ stake on the NO leg


def _fee(rate: float, price: float) -> float:
    """Per-contract trading fee ~ rate * p * (1 - p) (Kalshi's shape)."""
    return rate * price * (1.0 - price)


def _cap(depth: float, max_leg_stake: float) -> float:
    """Per-leg $ cap: the book's depth if it reports a positive one, else the
    user's manual max. (Kalshi depth is ~0 in practice; Poly's is whole-book.)"""
    return min(max_leg_stake, depth) if depth and depth > 0 else max_leg_stake


def _eval(team, yes_venue, a, yes_rate, yes_depth,
          no_venue, b, no_rate, no_depth, max_leg_stake) -> _Cand | None:
    """Build a candidate for one orientation, or None if prices are unusable."""
    if not (0.0 < a < 1.0) or not (0.0 < b < 1.0):
        return None
    fee_yes = _fee(yes_rate, a)
    fee_no = _fee(no_rate, b)
    fee_poly = fee_yes if yes_venue == VENUE_POLY else (fee_no if no_venue == VENUE_POLY else 0.0)
    fee_kalshi = fee_yes if yes_venue == VENUE_KALSHI else (fee_no if no_venue == VENUE_KALSHI else 0.0)
    fee = fee_yes + fee_no
    return _Cand(
        team=team, yes_venue=yes_venue, no_venue=no_venue, a=a, b=b,
        fee_poly=fee_poly, fee_kalshi=fee_kalshi, fee=fee,
        profit=1.0 - a - b - fee,
        yes_cap=_cap(yes_depth, max_leg_stake),
        no_cap=_cap(no_depth, max_leg_stake),
    )


def find_locks(poly: dict[str, dict], kalshi: dict[str, dict], *,
               kalshi_fee_rate: float = 0.07, poly_fee_rate: float = 0.0,
               min_profit: float = 0.02,
               max_leg_stake: float = math.inf) -> list[_Cand]:
    """Per-team lock candidates that clear `min_profit` after fees.

    Evaluates both orientations and keeps the more profitable positive one.
    """
    out: list[_Cand] = []
    for team in sorted(set(poly) & set(kalshi)):
        p, k = poly[team], kalshi[team]
        cand_a = _eval(team, VENUE_POLY, p.get("ask", 0.0), poly_fee_rate, p.get("depth", 0.0),
                       VENUE_KALSHI, k.get("no_ask", 0.0), kalshi_fee_rate, k.get("depth", 0.0),
                       max_leg_stake)
        cand_b = _eval(team, VENUE_KALSHI, k.get("ask", 0.0), kalshi_fee_rate, k.get("depth", 0.0),
                       VENUE_POLY, p.get("no_ask", 0.0), poly_fee_rate, p.get("depth", 0.0),
                       max_leg_stake)
        viable = [c for c in (cand_a, cand_b) if c and c.profit >= min_profit]
        if viable:
            out.append(max(viable, key=lambda c: c.profit))
    return out


from scipy.optimize import linprog

from worldcup.models import ArbRec


def size_locks(cands: list[_Cand], *, poly_balance: float, kalshi_balance: float,
               days_to_settlement: int) -> list[ArbRec]:
    """Global LP: maximize Σ xᵢ·profitᵢ over contract-pairs xᵢ ≥ 0, subject to
    Polymarket and Kalshi balance limits; per-leg depth caps become bounds.

    Fees are charged on the venue that levies them, so they consume that venue's
    balance too. The LP is continuous; we floor each xᵢ to whole contracts after.
    """
    if not cands:
        return []

    c = [-cand.profit for cand in cands]                 # minimize -profit
    poly_row, kalshi_row, bounds = [], [], []
    for cand in cands:
        if cand.yes_venue == VENUE_POLY:                 # YES on Poly, NO on Kalshi
            poly_cost, kalshi_cost = cand.a, cand.b
        else:                                            # YES on Kalshi, NO on Poly
            poly_cost, kalshi_cost = cand.b, cand.a
        poly_row.append(poly_cost + cand.fee_poly)
        kalshi_row.append(kalshi_cost + cand.fee_kalshi)
        ub_yes = cand.yes_cap / cand.a if cand.a > 0 else math.inf
        ub_no = cand.no_cap / cand.b if cand.b > 0 else math.inf
        bounds.append((0.0, min(ub_yes, ub_no)))

    res = linprog(c, A_ub=[poly_row, kalshi_row],
                  b_ub=[poly_balance, kalshi_balance],
                  bounds=bounds, method="highs")
    if not res.success:
        return []

    recs: list[ArbRec] = []
    for cand, x in zip(cands, res.x):
        contracts = int(math.floor(x + 1e-9))
        if contracts <= 0:
            continue
        stake_yes = contracts * cand.a
        stake_no = contracts * cand.b
        fee_cost = contracts * cand.fee
        total_cost = stake_yes + stake_no + fee_cost
        profit = contracts * cand.profit
        roc = profit / total_cost if total_cost > 0 else 0.0
        if days_to_settlement and days_to_settlement > 0:
            annual = (1.0 + roc) ** (365.0 / days_to_settlement) - 1.0
        else:
            annual = roc
        recs.append(ArbRec(
            team=cand.team, yes_venue=cand.yes_venue, no_venue=cand.no_venue,
            yes_ask=cand.a, no_ask=cand.b, contracts=contracts,
            stake_yes=round(stake_yes, 2), stake_no=round(stake_no, 2),
            total_cost=total_cost,  # kept at full precision; rounding breaks rel_tol=1e-6 invariant
            guaranteed_profit=round(profit, 2), roc=roc, annual_roc=annual))
    recs.sort(key=lambda r: r.guaranteed_profit, reverse=True)
    return recs
