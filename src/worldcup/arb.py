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
    """Per-contract trading fee, conservatively rounded UP to the next cent so a
    'risk-free' lock never understates fees: ceil(rate * p * (1-p)) to $0.01.
    (Kalshi rounds the whole order up to a cent; per-contract ceil is a safe
    over-estimate — we'd rather miss a marginal arb than claim a false one.)"""
    return math.ceil(rate * price * (1.0 - price) * 100.0) / 100.0


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


from worldcup.models import EvBetRec, DutchBook


def find_ev_bets(poly: dict[str, dict], kalshi: dict[str, dict],
                 consensus: dict[str, float], *, poly_balance: float,
                 kalshi_balance: float, max_leg_stake: float = math.inf,
                 kelly_fraction: float = 0.5, min_ev: float = 0.0) -> list[EvBetRec]:
    """+EV (not risk-free) YES buys on whichever book underprices a team vs.
    `consensus`. Sized by fractional Kelly f* = (p - ask)/(1 - ask) in descending
    EV order, depleting a per-venue running budget so that the total stakes for any
    venue never exceed the provided balance. `poly_balance`/`kalshi_balance` should
    already reflect capital already committed to locks upstream.
    """
    # Phase 1: collect all eligible candidates without sizing yet.
    candidates: list[tuple[float, str, str, float, float]] = []  # (ev, team, venue, ask, p)
    for team in sorted(set(poly) | set(kalshi)):
        p = consensus.get(team)
        if not p or p <= 0:
            continue
        # cheapest YES ask across the books that price this team
        options = []
        if team in poly and 0.0 < poly[team].get("ask", 0.0) < 1.0:
            options.append((VENUE_POLY, poly[team]["ask"]))
        if team in kalshi and 0.0 < kalshi[team].get("ask", 0.0) < 1.0:
            options.append((VENUE_KALSHI, kalshi[team]["ask"]))
        if not options:
            continue
        venue, ask = min(options, key=lambda o: o[1])
        ev = p / ask - 1.0
        if ev < min_ev or p <= ask:
            continue
        candidates.append((ev, team, venue, ask, p))

    # Phase 2: size in descending EV order, depleting per-venue running budget.
    candidates.sort(key=lambda c: c[0], reverse=True)
    remaining = {VENUE_POLY: poly_balance, VENUE_KALSHI: kalshi_balance}
    bets: list[EvBetRec] = []
    for ev, team, venue, ask, p in candidates:
        if remaining[venue] <= 1e-9:
            continue
        kelly = (p - ask) / (1.0 - ask)                  # full-Kelly fraction
        stake = max(0.0, kelly) * kelly_fraction * remaining[venue]
        stake = min(stake, max_leg_stake)
        if stake <= 1e-6:
            continue
        remaining[venue] -= stake
        bets.append(EvBetRec(
            team=team, venue=venue, side="YES", ask=ask, fair=p,
            ev_pct=ev, stake=round(stake, 2),
            potential_profit=round(stake * (1.0 - ask) / ask, 2)))
    # bets already in ev_pct-descending order from the sorted candidates
    return bets


def dutch_book(poly: dict[str, dict], kalshi: dict[str, dict], *,
               kalshi_fee_rate: float = 0.07, poly_fee_rate: float = 0.0) -> DutchBook:
    """Cover the whole field: cheapest YES (incl. fee) per team across both books.
    Risk-free iff the field sum is below $1."""
    legs: list = []
    field_sum = 0.0
    for team in sorted(set(poly) | set(kalshi)):
        options = []
        if team in poly and 0.0 < poly[team].get("ask", 0.0) < 1.0:
            a = poly[team]["ask"]
            options.append((a + _fee(poly_fee_rate, a), VENUE_POLY, a))
        if team in kalshi and 0.0 < kalshi[team].get("ask", 0.0) < 1.0:
            a = kalshi[team]["ask"]
            options.append((a + _fee(kalshi_fee_rate, a), VENUE_KALSHI, a))
        if not options:
            continue
        cost, venue, raw = min(options, key=lambda o: o[0])
        field_sum += cost
        legs.append((team, venue, raw))
    gap = 1.0 - field_sum
    return DutchBook(field_sum=field_sum, gap=gap, is_arb=gap > 0.0, legs=legs)


@dataclass
class ArbResult:
    locks: list            # list[ArbRec]
    ev_bets: list          # list[EvBetRec]
    dutch: object | None   # DutchBook | None


def run_arb(poly: dict[str, dict], kalshi: dict[str, dict],
            consensus: dict[str, float], *, poly_balance: float,
            kalshi_balance: float, max_leg_stake: float = math.inf,
            days_to_settlement: int = 0, kalshi_fee_rate: float = 0.07,
            poly_fee_rate: float = 0.0, min_profit: float = 0.02,
            kelly_fraction: float = 0.5, enable_ev: bool = True,
            enable_dutch: bool = True) -> ArbResult:
    """Run method A end-to-end on two raw books: locks (LP-sized) first, then the
    thin EV fallback on teams without a lock, then the optional Dutch-book check."""
    cands = find_locks(poly, kalshi, kalshi_fee_rate=kalshi_fee_rate,
                       poly_fee_rate=poly_fee_rate, min_profit=min_profit,
                       max_leg_stake=max_leg_stake)
    locks = size_locks(cands, poly_balance=poly_balance,
                       kalshi_balance=kalshi_balance,
                       days_to_settlement=days_to_settlement)

    ev_bets: list = []
    if enable_ev:
        locked = {r.team for r in locks}
        ev_poly = {t: v for t, v in poly.items() if t not in locked}
        ev_kalshi = {t: v for t, v in kalshi.items() if t not in locked}

        # Compute how much capital the locks already consumed per venue.
        # For each lock rec we need the corresponding _Cand for its fee split.
        cand_by_team = {c.team: c for c in cands}
        poly_spent = 0.0
        kalshi_spent = 0.0
        for rec in locks:
            cand = cand_by_team.get(rec.team)
            if cand is None:
                continue
            # Which leg sits on Poly vs Kalshi?
            if rec.yes_venue == VENUE_POLY:
                poly_spent += rec.stake_yes + rec.contracts * cand.fee_poly
                kalshi_spent += rec.stake_no + rec.contracts * cand.fee_kalshi
            else:
                kalshi_spent += rec.stake_yes + rec.contracts * cand.fee_kalshi
                poly_spent += rec.stake_no + rec.contracts * cand.fee_poly

        ev_poly_balance = max(0.0, poly_balance - poly_spent)
        ev_kalshi_balance = max(0.0, kalshi_balance - kalshi_spent)

        ev_bets = find_ev_bets(ev_poly, ev_kalshi, consensus,
                               poly_balance=ev_poly_balance,
                               kalshi_balance=ev_kalshi_balance,
                               max_leg_stake=max_leg_stake, kelly_fraction=kelly_fraction)

    dutch = None
    if enable_dutch:
        dutch = dutch_book(poly, kalshi, kalshi_fee_rate=kalshi_fee_rate,
                           poly_fee_rate=poly_fee_rate)
    return ArbResult(locks=locks, ev_bets=ev_bets, dutch=dutch)
