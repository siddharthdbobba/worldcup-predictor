# Cross-Book Arbitrage (Method A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a cross-book arbitrage strategy ("method A") that finds risk-free locks between Polymarket and Kalshi on the World Cup outright-winner market, sizes them with a global LP across both venue balances, and reports exact per-leg stakes — plus a thin +EV fallback and an optional whole-field Dutch-book check.

**Architecture:** A new pure module `src/worldcup/arb.py` does all detection and sizing on raw per-book price dicts. `markets.py` gains bid/NO-side fields and a `fetch_books` helper that returns the two books separately (arb must see them un-combined). `report.py` renders an arb card; `main.py` gets an `--arb` mode that bypasses the LLM agent (no LLM in the numerical path). Everything numeric is deterministic and unit-tested against crafted fixtures.

**Tech Stack:** Python · `scipy.optimize.linprog` (the LP) · existing fractional-Kelly math for the EV tier · `numpy` · `rich` · `pytest`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-02-cross-book-arbitrage-design.md`

**Key conventions reused from the codebase:**
- Per-book parsed dict is `{team: {"prob","ask","vol","depth", ...}}` where `ask` is the YES ask. We add `yes_bid`, `no_ask`, `no_bid`.
- Venue string constants: `"polymarket"`, `"kalshi"`.
- Money fields rounded to 2 dp in dataclasses; rates/probabilities kept full precision.
- Tests live in `tests/`, fixtures in `tests/fixtures/`, run with `pytest`.

---

## Task 1: Extend the market parsers with bid / NO-side fields

The NO leg needs the dear book's bid. Kalshi quotes NO directly; Polymarket's NO is derived as `1 − bestBid`. Changes are **additive** — existing keys (`prob`, `ask`, `vol`, `depth`) stay, so existing tests keep passing.

**Files:**
- Modify: `src/worldcup/markets.py` (`parse_polymarket`, `parse_kalshi`)
- Modify: `tests/fixtures/polymarket_sample.json`, `tests/fixtures/kalshi_sample.json`
- Test: `tests/test_markets.py`

- [ ] **Step 1: Add bid fields to the existing fixtures**

Edit `tests/fixtures/polymarket_sample.json` to add `bestBid` to each market (existing values unchanged):

```json
{
  "title": "World Cup Winner",
  "markets": [
    {"groupItemTitle": "France", "lastTradePrice": 0.17, "bestAsk": 0.18, "bestBid": 0.16, "liquidityNum": 250000, "volumeNum": 8000000},
    {"groupItemTitle": "Spain",  "lastTradePrice": 0.16, "bestAsk": 0.17, "bestBid": 0.15, "liquidityNum": 240000, "volumeNum": 7000000}
  ]
}
```

Edit `tests/fixtures/kalshi_sample.json` to add `yes_bid_dollars`, `no_ask_dollars`, `no_bid_dollars`:

```json
{
  "markets": [
    {"yes_sub_title": "France", "last_price_dollars": 0.17, "yes_ask_dollars": 0.19, "yes_bid_dollars": 0.16, "no_ask_dollars": 0.84, "no_bid_dollars": 0.81, "liquidity_dollars": 0, "volume_fp": 5000000, "open_interest_fp": 3000000},
    {"yes_sub_title": "England", "last_price_dollars": 0.11, "yes_ask_dollars": 0.12, "yes_bid_dollars": 0.10, "no_ask_dollars": 0.90, "no_bid_dollars": 0.88, "liquidity_dollars": 0, "volume_fp": 3000000, "open_interest_fp": 1800000}
  ]
}
```

- [ ] **Step 2: Write the failing test**

Add to `tests/test_markets.py`:

```python
def test_parse_polymarket_captures_bid_and_derived_no():
    data = json.loads((FIX / "polymarket_sample.json").read_text())
    lines = parse_polymarket(data)
    assert math.isclose(lines["France"]["yes_bid"], 0.16)
    assert math.isclose(lines["France"]["no_ask"], 1 - 0.16)  # derived 1 - bestBid


def test_parse_kalshi_captures_bid_and_quoted_no():
    data = json.loads((FIX / "kalshi_sample.json").read_text())
    lines = parse_kalshi(data)
    assert math.isclose(lines["France"]["yes_bid"], 0.16)
    assert math.isclose(lines["France"]["no_ask"], 0.84)   # quoted directly
    assert math.isclose(lines["France"]["no_bid"], 0.81)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_markets.py::test_parse_polymarket_captures_bid_and_derived_no tests/test_markets.py::test_parse_kalshi_captures_bid_and_quoted_no -v`
Expected: FAIL with `KeyError: 'yes_bid'`.

- [ ] **Step 4: Extend `parse_polymarket`**

In `src/worldcup/markets.py`, replace the dict built inside `parse_polymarket`'s loop:

```python
        bid = _f(m.get("bestBid"))
        out[canonical(title)] = {
            "prob": _f(price),
            "ask": _f(m.get("bestAsk") or price),
            "yes_bid": bid,
            "no_ask": (1.0 - bid) if bid > 0 else 0.0,  # derived; Poly has no direct NO quote
            "vol": _f(m.get("volumeNum")),
            "depth": _f(m.get("liquidityNum")),
        }
```

- [ ] **Step 5: Extend `parse_kalshi`**

In `src/worldcup/markets.py`, replace the dict built inside `parse_kalshi`'s loop:

```python
        out[canonical(title)] = {
            "prob": _f(price),
            "ask": _f(m.get("yes_ask_dollars") or price),
            "yes_bid": _f(m.get("yes_bid_dollars")),
            "no_ask": _f(m.get("no_ask_dollars")),
            "no_bid": _f(m.get("no_bid_dollars")),
            "vol": _f(m.get("volume_fp")),
            "depth": _f(m.get("liquidity_dollars")),
        }
```

- [ ] **Step 6: Run the full markets test file**

Run: `pytest tests/test_markets.py -v`
Expected: PASS — the two new tests pass and all pre-existing markets tests still pass (they never asserted the new keys).

- [ ] **Step 7: Commit**

```bash
git add src/worldcup/markets.py tests/test_markets.py tests/fixtures/polymarket_sample.json tests/fixtures/kalshi_sample.json
git commit -m "feat: capture bid + NO-side quotes in market parsers"
```

---

## Task 2: Add the arb dataclasses

**Files:**
- Modify: `src/worldcup/models.py`
- Test: `tests/test_arb.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_arb.py`:

```python
from worldcup.models import ArbRec, EvBetRec, DutchBook


def test_dataclasses_construct():
    a = ArbRec(team="France", yes_venue="polymarket", no_venue="kalshi",
               yes_ask=0.18, no_ask=0.71, contracts=10,
               stake_yes=1.80, stake_no=7.10, total_cost=9.04,
               guaranteed_profit=0.96, roc=0.106, annual_roc=1.2)
    assert a.team == "France" and a.contracts == 10
    e = EvBetRec(team="Spain", venue="polymarket", side="YES", ask=0.10,
                 fair=0.14, ev_pct=0.4, stake=5.0, potential_profit=45.0)
    assert e.side == "YES"
    d = DutchBook(field_sum=1.03, gap=-0.03, is_arb=False, legs=[("France", "kalshi", 0.17)])
    assert d.is_arb is False and d.legs[0][0] == "France"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_arb.py::test_dataclasses_construct -v`
Expected: FAIL with `ImportError: cannot import name 'ArbRec'`.

- [ ] **Step 3: Add the dataclasses**

Append to `src/worldcup/models.py`:

```python
@dataclass
class ArbRec:
    """One sized risk-free lock: buy `contracts` YES on yes_venue and the same
    number of NO on no_venue. Payout is identical whoever wins."""
    team: str
    yes_venue: str          # "polymarket" | "kalshi"
    no_venue: str
    yes_ask: float          # a: price paid per YES contract
    no_ask: float           # b: price paid per NO contract
    contracts: int          # floored from the LP solution
    stake_yes: float
    stake_no: float
    total_cost: float       # stake_yes + stake_no + fees
    guaranteed_profit: float
    roc: float              # guaranteed_profit / total_cost
    annual_roc: float       # roc annualized to settlement


@dataclass
class EvBetRec:
    """A +EV (not risk-free) cross-book bet: the cheaper book underprices a team
    vs. the two-book consensus. Sized by fractional Kelly."""
    team: str
    venue: str
    side: str               # "YES"
    ask: float
    fair: float             # consensus probability
    ev_pct: float           # fair/ask - 1
    stake: float
    potential_profit: float


@dataclass
class DutchBook:
    """Whole-field check: buy the cheapest YES per team across both books."""
    field_sum: float        # Σ cheapest YES ask (incl. fee) over all teams
    gap: float              # 1 - field_sum  (positive => risk-free arb)
    is_arb: bool
    legs: list              # [(team, venue, price), ...]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_arb.py::test_dataclasses_construct -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/models.py tests/test_arb.py
git commit -m "feat: add ArbRec/EvBetRec/DutchBook dataclasses"
```

---

## Task 3: Lock detection (`find_locks`)

Detect per-team locks in both orientations, net fees, and gate by `min_profit`. Returns internal `_Cand` objects consumed by the LP in Task 4.

**Files:**
- Create: `src/worldcup/arb.py`
- Test: `tests/test_arb.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_arb.py`:

```python
import math
from worldcup.arb import find_locks, VENUE_POLY, VENUE_KALSHI

# Crafted book pair with a clear orientation-A lock on France.
# Poly YES 0.18 (bid 0.16); Kalshi YES 0.30 (bid 0.28), NO 0.71.
# Orientation A: YES Poly @0.18 + NO Kalshi @0.71 = 0.89 < 1, fires (k_bid .28 > p_ask .18).
POLY = {
    "France": {"prob": 0.17, "ask": 0.18, "yes_bid": 0.16, "no_ask": 0.84, "depth": 250000},
    "Spain":  {"prob": 0.10, "ask": 0.10, "yes_bid": 0.08, "no_ask": 0.92, "depth": 250000},
    "Brazil": {"prob": 0.20, "ask": 0.20, "yes_bid": 0.19, "no_ask": 0.81, "depth": 250000},
}
KALSHI = {
    "France": {"prob": 0.30, "ask": 0.30, "yes_bid": 0.28, "no_ask": 0.71, "no_bid": 0.70, "depth": 0},
    "Spain":  {"prob": 0.20, "ask": 0.20, "yes_bid": 0.18, "no_ask": 0.81, "no_bid": 0.80, "depth": 0},
    "Brazil": {"prob": 0.21, "ask": 0.21, "yes_bid": 0.20, "no_ask": 0.80, "no_bid": 0.79, "depth": 0},
}


def test_find_locks_picks_orientation_a():
    cands = find_locks(POLY, KALSHI, min_profit=0.02)
    france = next(c for c in cands if c.team == "France")
    assert france.yes_venue == VENUE_POLY and france.no_venue == VENUE_KALSHI
    assert math.isclose(france.a, 0.18) and math.isclose(france.b, 0.71)
    # raw 1-0.18-0.71 = 0.11; kalshi fee 0.07*0.71*0.29 ≈ 0.0144; profit ≈ 0.0956
    assert 0.09 < france.profit < 0.10


def test_find_locks_min_profit_gate_filters_thin():
    # Brazil: a=0.20, b=0.80 -> raw 0.0, fee>0 -> negative; never a lock.
    cands = find_locks(POLY, KALSHI, min_profit=0.02)
    assert all(c.team != "Brazil" for c in cands)


def test_find_locks_none_when_spreads_overlap():
    poly = {"France": {"prob": 0.5, "ask": 0.52, "yes_bid": 0.48, "no_ask": 0.52, "depth": 100}}
    kalshi = {"France": {"prob": 0.5, "ask": 0.52, "yes_bid": 0.48, "no_ask": 0.52, "no_bid": 0.46, "depth": 0}}
    assert find_locks(poly, kalshi, min_profit=0.0) == []


def test_find_locks_depth_caps_from_book_and_max_leg():
    cands = find_locks(POLY, KALSHI, min_profit=0.02, max_leg_stake=500.0)
    france = next(c for c in cands if c.team == "France")
    # YES leg on Poly: depth 250000 vs cap 500 -> 500; NO leg on Kalshi: depth 0 -> cap 500
    assert math.isclose(france.yes_cap, 500.0)
    assert math.isclose(france.no_cap, 500.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_arb.py -k find_locks -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'worldcup.arb'`.

- [ ] **Step 3: Create `arb.py` with detection**

Create `src/worldcup/arb.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_arb.py -k find_locks -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/arb.py tests/test_arb.py
git commit -m "feat: cross-book lock detection (both orientations, fee-aware)"
```

---

## Task 4: LP sizing (`size_locks`)

Maximize total locked profit across all candidates subject to each venue's balance; per-leg depth caps become variable upper bounds. Floor to whole contracts.

**Files:**
- Modify: `src/worldcup/arb.py`
- Test: `tests/test_arb.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_arb.py`:

```python
from worldcup.arb import size_locks


def test_size_locks_equal_payout_per_arb():
    cands = find_locks(POLY, KALSHI, min_profit=0.02, max_leg_stake=1000.0)
    recs = size_locks(cands, poly_balance=10_000, kalshi_balance=10_000,
                      days_to_settlement=47)
    france = next(r for r in recs if r.team == "France")
    # Equal contracts on both legs => identical payout (= contracts) either way.
    payout_if_france_wins = france.contracts          # YES pays 1 each
    payout_if_france_loses = france.contracts         # NO pays 1 each
    assert payout_if_france_wins == payout_if_france_loses
    assert france.guaranteed_profit > 0
    assert math.isclose(france.total_cost,
                        france.stake_yes + france.stake_no
                        + france.contracts * (next(c.fee for c in cands if c.team == "France")),
                        rel_tol=1e-6)


def test_size_locks_respects_venue_balances():
    cands = find_locks(POLY, KALSHI, min_profit=0.02, max_leg_stake=1e9)
    recs = size_locks(cands, poly_balance=18.0, kalshi_balance=1e9,
                      days_to_settlement=47)
    # Whichever locks the LP funds, total $ drawn from Polymarket must respect $18.
    poly_spent = sum(r.stake_yes if r.yes_venue == VENUE_POLY else r.stake_no
                     for r in recs)
    assert poly_spent <= 18.0 + 1e-6


def test_size_locks_respects_depth_cap():
    # max_leg_stake applies to BOTH legs; the pricier NO leg (0.71) binds first:
    # 7.10 / 0.71 = 10 contracts (YES leg would allow 7.10/0.18 = 39).
    cands = find_locks(POLY, KALSHI, min_profit=0.02, max_leg_stake=7.10)
    recs = size_locks(cands, poly_balance=1e9, kalshi_balance=1e9,
                      days_to_settlement=47)
    france = next(r for r in recs if r.team == "France")
    assert france.contracts == 10
    assert france.stake_no <= 7.10 + 1e-6


def test_size_locks_contracts_are_integers():
    cands = find_locks(POLY, KALSHI, min_profit=0.02, max_leg_stake=1000.0)
    recs = size_locks(cands, poly_balance=10_000, kalshi_balance=10_000,
                      days_to_settlement=47)
    assert all(isinstance(r.contracts, int) and r.contracts >= 1 for r in recs)


def test_size_locks_annualizes_roc():
    cands = find_locks(POLY, KALSHI, min_profit=0.02, max_leg_stake=1000.0)
    recs = size_locks(cands, poly_balance=10_000, kalshi_balance=10_000,
                      days_to_settlement=47)
    r = recs[0]
    expected = (1.0 + r.roc) ** (365.0 / 47) - 1.0
    assert math.isclose(r.annual_roc, expected, rel_tol=1e-9)


def test_size_locks_empty():
    assert size_locks([], poly_balance=100, kalshi_balance=100,
                      days_to_settlement=47) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_arb.py -k size_locks -v`
Expected: FAIL with `ImportError: cannot import name 'size_locks'`.

- [ ] **Step 3: Implement `size_locks`**

Append to `src/worldcup/arb.py`:

```python
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
            total_cost=round(total_cost, 2),
            guaranteed_profit=round(profit, 2), roc=roc, annual_roc=annual))
    recs.sort(key=lambda r: r.guaranteed_profit, reverse=True)
    return recs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_arb.py -k size_locks -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/arb.py tests/test_arb.py
git commit -m "feat: global LP sizing for cross-book locks"
```

---

## Task 5: Thin +EV fallback (`find_ev_bets`)

When no lock exists, flag the side a book underprices vs. the two-book consensus. Independent fractional Kelly, capped — **not** the existing `kelly_allocate` (which assumes mutually-exclusive outcomes and raises otherwise).

**Files:**
- Modify: `src/worldcup/arb.py`
- Test: `tests/test_arb.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_arb.py`:

```python
from worldcup.arb import find_ev_bets


def test_find_ev_bets_flags_underpriced_book():
    # No lock, but Poly YES for Spain (0.10) sits below consensus 0.14 -> +EV.
    poly = {"Spain": {"ask": 0.10, "yes_bid": 0.08, "no_ask": 0.92, "depth": 1000}}
    kalshi = {"Spain": {"ask": 0.15, "yes_bid": 0.13, "no_ask": 0.86, "depth": 0}}
    consensus = {"Spain": 0.14}
    bets = find_ev_bets(poly, kalshi, consensus, poly_balance=1000,
                        kalshi_balance=1000, max_leg_stake=1000, kelly_fraction=0.5)
    spain = next(b for b in bets if b.team == "Spain")
    assert spain.venue == "polymarket" and spain.side == "YES"
    assert math.isclose(spain.ev_pct, 0.14 / 0.10 - 1.0)
    assert 0 < spain.stake <= 1000


def test_find_ev_bets_none_when_no_edge():
    poly = {"Spain": {"ask": 0.20, "yes_bid": 0.18, "no_ask": 0.80, "depth": 1000}}
    kalshi = {"Spain": {"ask": 0.21, "yes_bid": 0.19, "no_ask": 0.79, "depth": 0}}
    consensus = {"Spain": 0.15}            # both asks above fair -> no +EV YES buy
    assert find_ev_bets(poly, kalshi, consensus, poly_balance=1000,
                        kalshi_balance=1000, max_leg_stake=1000) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_arb.py -k find_ev_bets -v`
Expected: FAIL with `ImportError: cannot import name 'find_ev_bets'`.

- [ ] **Step 3: Implement `find_ev_bets`**

Append to `src/worldcup/arb.py`:

```python
from worldcup.models import EvBetRec


def find_ev_bets(poly: dict[str, dict], kalshi: dict[str, dict],
                 consensus: dict[str, float], *, poly_balance: float,
                 kalshi_balance: float, max_leg_stake: float = math.inf,
                 kelly_fraction: float = 0.5, min_ev: float = 0.0) -> list[EvBetRec]:
    """+EV (not risk-free) YES buys on whichever book underprices a team vs.
    `consensus`. Each sized independently by fractional Kelly f* = (p - ask)/(1 - ask),
    capped by that venue's balance and `max_leg_stake`. Deliberately thin/secondary.
    """
    balances = {VENUE_POLY: poly_balance, VENUE_KALSHI: kalshi_balance}
    bets: list[EvBetRec] = []
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
        kelly = (p - ask) / (1.0 - ask)                  # full-Kelly fraction
        stake = max(0.0, kelly) * kelly_fraction * balances[venue]
        stake = min(stake, max_leg_stake)
        if stake <= 1e-6:
            continue
        bets.append(EvBetRec(
            team=team, venue=venue, side="YES", ask=ask, fair=p,
            ev_pct=ev, stake=round(stake, 2),
            potential_profit=round(stake * (1.0 - ask) / ask, 2)))
    bets.sort(key=lambda b: b.ev_pct, reverse=True)
    return bets
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_arb.py -k find_ev_bets -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/arb.py tests/test_arb.py
git commit -m "feat: thin +EV cross-book fallback (independent fractional Kelly)"
```

---

## Task 6: Whole-field Dutch-book check (`dutch_book`)

Buy the cheapest YES per team across both books; if the field sum (incl. fee) is below $1, the whole field is covered for under a dollar.

**Files:**
- Modify: `src/worldcup/arb.py`
- Test: `tests/test_arb.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_arb.py`:

```python
from worldcup.arb import dutch_book


def test_dutch_book_flags_when_field_below_one():
    # Two-team toy world; cheapest YES per team sums under 1.
    poly = {"A": {"ask": 0.40}, "B": {"ask": 0.55}}
    kalshi = {"A": {"ask": 0.45}, "B": {"ask": 0.50}}
    db = dutch_book(poly, kalshi, kalshi_fee_rate=0.0, poly_fee_rate=0.0)
    # min(0.40,0.45)=0.40 (poly A) + min(0.55,0.50)=0.50 (kalshi B) = 0.90
    assert math.isclose(db.field_sum, 0.90)
    assert math.isclose(db.gap, 0.10) and db.is_arb is True
    assert ("A", "polymarket", 0.40) in db.legs and ("B", "kalshi", 0.50) in db.legs


def test_dutch_book_not_arb_for_normal_field():
    poly = {"A": {"ask": 0.60}, "B": {"ask": 0.60}}
    kalshi = {"A": {"ask": 0.62}, "B": {"ask": 0.61}}
    db = dutch_book(poly, kalshi, kalshi_fee_rate=0.0, poly_fee_rate=0.0)
    assert db.field_sum > 1.0 and db.is_arb is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_arb.py -k dutch_book -v`
Expected: FAIL with `ImportError: cannot import name 'dutch_book'`.

- [ ] **Step 3: Implement `dutch_book`**

Append to `src/worldcup/arb.py`:

```python
from worldcup.models import DutchBook


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_arb.py -k dutch_book -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/arb.py tests/test_arb.py
git commit -m "feat: optional whole-field Dutch-book check"
```

---

## Task 7: Orchestrator (`run_arb`) + `fetch_books` helper

`run_arb` is pure: it takes the two raw books + consensus and returns everything. `fetch_books` is the thin I/O wrapper (mirrors `fetch_market_probabilities` but returns the books separately and requires **both**).

**Files:**
- Modify: `src/worldcup/arb.py` (add `ArbResult`, `run_arb`)
- Modify: `src/worldcup/markets.py` (add `fetch_books`)
- Test: `tests/test_arb.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_arb.py`:

```python
from worldcup.arb import run_arb, ArbResult


def test_run_arb_combines_all_tiers():
    consensus = {"France": 0.22, "Spain": 0.13, "Brazil": 0.205}
    res = run_arb(POLY, KALSHI, consensus, poly_balance=10_000,
                  kalshi_balance=10_000, max_leg_stake=1000.0,
                  days_to_settlement=47)
    assert isinstance(res, ArbResult)
    assert any(r.team == "France" for r in res.locks)     # France lock sized
    assert isinstance(res.ev_bets, list)
    assert res.dutch is not None


def test_run_arb_can_disable_extras():
    res = run_arb(POLY, KALSHI, {"France": 0.22}, poly_balance=10_000,
                  kalshi_balance=10_000, max_leg_stake=1000.0,
                  days_to_settlement=47, enable_ev=False, enable_dutch=False)
    assert res.ev_bets == [] and res.dutch is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_arb.py -k run_arb -v`
Expected: FAIL with `ImportError: cannot import name 'run_arb'`.

- [ ] **Step 3: Implement `ArbResult` and `run_arb`**

Append to `src/worldcup/arb.py`:

```python
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
        ev_bets = find_ev_bets(ev_poly, ev_kalshi, consensus,
                               poly_balance=poly_balance, kalshi_balance=kalshi_balance,
                               max_leg_stake=max_leg_stake, kelly_fraction=kelly_fraction)

    dutch = None
    if enable_dutch:
        dutch = dutch_book(poly, kalshi, kalshi_fee_rate=kalshi_fee_rate,
                           poly_fee_rate=poly_fee_rate)
    return ArbResult(locks=locks, ev_bets=ev_bets, dutch=dutch)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_arb.py -k run_arb -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Add `fetch_books` to `markets.py`**

Append to `src/worldcup/markets.py`:

```python
def fetch_books(valid_teams=None, timeout: float = 20.0):
    """Fetch BOTH books as separate raw dicts for cross-book arbitrage.

    Unlike `fetch_market_probabilities`, arb needs both venues, so this raises if
    either is missing. Optionally restricts to `valid_teams` (drops stale
    non-qualified markets that can carry a bogus 1.0 price).
    """
    poly, kalshi, errors = {}, {}, []
    try:
        r = httpx.get(POLYMARKET_URL, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
        poly = parse_polymarket(r.json())
    except Exception as e:  # noqa: BLE001
        errors.append(f"polymarket: {e}")
    try:
        r = httpx.get(KALSHI_URL, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
        kalshi = parse_kalshi(r.json())
    except Exception as e:  # noqa: BLE001
        errors.append(f"kalshi: {e}")
    if not poly or not kalshi:
        raise RuntimeError("cross-book arbitrage needs BOTH books; "
                           + ("; ".join(errors) or "one book returned no markets"))
    if valid_teams is not None:
        vt = set(valid_teams)
        poly = {t: v for t, v in poly.items() if t in vt}
        kalshi = {t: v for t, v in kalshi.items() if t in vt}
    return poly, kalshi
```

- [ ] **Step 6: Run the full arb + markets suites**

Run: `pytest tests/test_arb.py tests/test_markets.py -v`
Expected: PASS (all tasks 1–7 tests green).

- [ ] **Step 7: Commit**

```bash
git add src/worldcup/arb.py src/worldcup/markets.py tests/test_arb.py
git commit -m "feat: run_arb orchestrator + fetch_books (both-books) helper"
```

---

## Task 8: Arb report (terminal + markdown)

**Files:**
- Modify: `src/worldcup/report.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_report.py`:

```python
from worldcup.arb import ArbResult
from worldcup.models import ArbRec, EvBetRec, DutchBook
from worldcup.report import build_arb_markdown


def test_build_arb_markdown_renders_locks_and_disclaimer():
    locks = [ArbRec(team="France", yes_venue="polymarket", no_venue="kalshi",
                    yes_ask=0.18, no_ask=0.71, contracts=100,
                    stake_yes=18.0, stake_no=71.0, total_cost=90.44,
                    guaranteed_profit=9.56, roc=0.1057, annual_roc=1.23)]
    ev = [EvBetRec(team="Spain", venue="polymarket", side="YES", ask=0.10,
                   fair=0.14, ev_pct=0.40, stake=20.0, potential_profit=180.0)]
    dutch = DutchBook(field_sum=1.03, gap=-0.03, is_arb=False, legs=[])
    res = ArbResult(locks=locks, ev_bets=ev, dutch=dutch)
    md = build_arb_markdown(res, poly_balance=10_000, kalshi_balance=10_000)
    assert "France" in md and "polymarket" in md and "kalshi" in md
    assert "$18.00" in md and "$71.00" in md            # per-leg stakes
    assert "9.56" in md                                  # guaranteed profit
    assert "Spain" in md                                 # EV section
    assert "never stake more than you can afford" in md.lower()


def test_build_arb_markdown_handles_no_locks():
    res = ArbResult(locks=[], ev_bets=[], dutch=None)
    md = build_arb_markdown(res, poly_balance=100, kalshi_balance=100)
    assert "no risk-free arbitrage" in md.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report.py -k arb -v`
Expected: FAIL with `ImportError: cannot import name 'build_arb_markdown'`.

- [ ] **Step 3: Implement the report functions**

Append to `src/worldcup/report.py` (add the needed imports at the top of the file: `from worldcup.arb import ArbResult`):

```python
ARB_DISCLAIMER = (
    "Risk-free only if BOTH legs fill at the quoted prices — place them near-"
    "simultaneously (leg-timing risk). Capital is locked on both venues until the "
    "tournament settles. Depth caps are user-set, not measured. For education only — "
    "never stake more than you can afford to lose.")


def build_arb_markdown(res: ArbResult, poly_balance: float,
                       kalshi_balance: float) -> str:
    """Markdown for the arb card: locks, EV fallback, Dutch-book line, disclaimer."""
    lines = ["# Method A — Cross-Book Arbitrage", "",
             f"_Polymarket balance: ${poly_balance:,.2f} · "
             f"Kalshi balance: ${kalshi_balance:,.2f}_", ""]

    lines += ["## Risk-free locks", ""]
    if not res.locks:
        lines.append("No risk-free arbitrage at current prices.")
    else:
        lines += ["| Team | YES @ | NO @ | Contracts | Stake YES | Stake NO | Cost | Profit | ROC | Annual |",
                  "|------|-------|------|----------:|----------:|---------:|-----:|-------:|----:|-------:|"]
        for r in res.locks:
            lines.append(
                f"| {r.team} | {r.yes_venue} {r.yes_ask:.2f} | {r.no_venue} {r.no_ask:.2f} | "
                f"{r.contracts} | ${r.stake_yes:,.2f} | ${r.stake_no:,.2f} | ${r.total_cost:,.2f} | "
                f"${r.guaranteed_profit:,.2f} | {r.roc:.2%} | {r.annual_roc:.1%} |")
        tot = sum(r.guaranteed_profit for r in res.locks)
        lines += ["", f"_Total guaranteed profit: ${tot:,.2f}_"]

    if res.ev_bets:
        lines += ["", "## +EV cross-book bets (NOT risk-free)", "",
                  "| Team | Venue | Side | Ask | Fair | EV | Stake | Profit |",
                  "|------|-------|------|----:|-----:|---:|------:|-------:|"]
        for b in res.ev_bets:
            lines.append(
                f"| {b.team} | {b.venue} | {b.side} | {b.ask:.2f} | {b.fair:.1%} | "
                f"{b.ev_pct:+.1%} | ${b.stake:,.2f} | ${b.potential_profit:,.2f} |")

    if res.dutch is not None:
        d = res.dutch
        verdict = f"ARB (gap {d.gap:+.3f})" if d.is_arb else f"no arb (gap {d.gap:+.3f})"
        lines += ["", "## Whole-field Dutch book",
                  f"Cheapest-YES field sum: {d.field_sum:.3f} — {verdict}."]

    lines += ["", f"> {ARB_DISCLAIMER}"]
    return "\n".join(lines) + "\n"


def print_arb_report(res: ArbResult, poly_balance: float, kalshi_balance: float,
                     console: Console | None = None) -> None:
    """Pretty-print the arb card to the terminal."""
    console = console or Console()
    if not res.locks:
        console.print("[bold]No risk-free arbitrage[/bold] at current prices.")
    else:
        t = Table(title="Method A — risk-free locks")
        for col in ("Team", "YES @", "NO @", "Contracts", "Stake YES",
                    "Stake NO", "Cost", "Profit", "ROC", "Annual"):
            t.add_column(col, justify="left" if col == "Team" else "right")
        for r in res.locks:
            t.add_row(r.team, f"{r.yes_venue} {r.yes_ask:.2f}",
                      f"{r.no_venue} {r.no_ask:.2f}", str(r.contracts),
                      f"${r.stake_yes:,.2f}", f"${r.stake_no:,.2f}",
                      f"${r.total_cost:,.2f}", f"${r.guaranteed_profit:,.2f}",
                      f"{r.roc:.2%}", f"{r.annual_roc:.1%}")
        console.print(t)
    if res.ev_bets:
        et = Table(title="+EV cross-book bets (NOT risk-free)")
        for col in ("Team", "Venue", "Side", "Ask", "Fair", "EV", "Stake", "Profit"):
            et.add_column(col, justify="left" if col == "Team" else "right")
        for b in res.ev_bets:
            et.add_row(b.team, b.venue, b.side, f"{b.ask:.2f}", f"{b.fair:.1%}",
                       f"{b.ev_pct:+.1%}", f"${b.stake:,.2f}", f"${b.potential_profit:,.2f}")
        console.print(et)
    if res.dutch is not None:
        d = res.dutch
        verdict = f"ARB (gap {d.gap:+.3f})" if d.is_arb else f"no arb (gap {d.gap:+.3f})"
        console.print(f"[dim]Dutch-book field sum {d.field_sum:.3f} — {verdict}[/dim]")
    console.print(f"\n[dim]{ARB_DISCLAIMER}[/dim]")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_report.py -k arb -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/report.py tests/test_report.py
git commit -m "feat: cross-book arbitrage report (terminal + markdown)"
```

---

## Task 9: CLI wiring (`--arb` mode)

Add an `--arb` mode that bypasses the LLM agent: fetch the draw (for valid teams) + both books + consensus, run `run_arb`, print the card. Keeps the LLM out of the numerical path.

**Files:**
- Modify: `main.py`
- Test: `tests/test_arb.py` (CLI driver smoke test via monkeypatched fetches)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_arb.py`:

```python
def test_run_arb_cli_prints_card(monkeypatch, capsys):
    import main as cli
    monkeypatch.setattr(cli, "fetch_group_draw", lambda: {"A": ["France", "Spain", "Brazil", "Japan"]})
    monkeypatch.setattr(cli, "fetch_books", lambda valid_teams=None: (POLY, KALSHI))
    monkeypatch.setattr(cli, "fetch_market_probabilities",
                        lambda valid_teams=None: ({"France": 0.22, "Spain": 0.13, "Brazil": 0.205},
                                                  {}, {}, {}))
    cli.run_arb_cli(poly_balance=10_000, kalshi_balance=10_000, kalshi_fee_rate=0.07,
                    poly_fee_rate=0.0, min_profit=0.02, max_leg_stake=1000.0,
                    days_to_settlement=47, kelly_fraction=0.5,
                    enable_ev=True, enable_dutch=True)
    out = capsys.readouterr().out
    assert "France" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_arb.py::test_run_arb_cli_prints_card -v`
Expected: FAIL with `AttributeError: module 'main' has no attribute 'run_arb_cli'`.

- [ ] **Step 3: Add imports and the driver to `main.py`**

At the top of `main.py`, add imports:

```python
from datetime import date

from worldcup.arb import run_arb
from worldcup.draw import fetch_group_draw
from worldcup.markets import fetch_books, fetch_market_probabilities
from worldcup.report import print_arb_report
```

Add the driver function to `main.py`:

```python
def run_arb_cli(*, poly_balance, kalshi_balance, kalshi_fee_rate, poly_fee_rate,
                min_profit, max_leg_stake, days_to_settlement, kelly_fraction,
                enable_ev, enable_dutch):
    """Fetch live data and run method A (no LLM in the numerical path)."""
    groups = fetch_group_draw()
    valid = {t for ts in groups.values() for t in ts}
    poly, kalshi = fetch_books(valid_teams=valid)
    consensus, _ask, _conf, _depth = fetch_market_probabilities(valid_teams=valid)
    res = run_arb(poly, kalshi, consensus, poly_balance=poly_balance,
                  kalshi_balance=kalshi_balance, max_leg_stake=max_leg_stake,
                  days_to_settlement=days_to_settlement, kalshi_fee_rate=kalshi_fee_rate,
                  poly_fee_rate=poly_fee_rate, min_profit=min_profit,
                  kelly_fraction=kelly_fraction, enable_ev=enable_ev,
                  enable_dutch=enable_dutch)
    print_arb_report(res, poly_balance, kalshi_balance)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_arb.py::test_run_arb_cli_prints_card -v`
Expected: PASS.

- [ ] **Step 5: Add the `--arb` flags and branch to `parse_args` / `main`**

In `parse_args` in `main.py`, add:

```python
    ap.add_argument("--arb", action="store_true",
                    help="Run method A: cross-book arbitrage (Polymarket vs Kalshi)")
    ap.add_argument("--poly-balance", type=float, default=0.0,
                    help="[--arb] Available USDC balance on Polymarket")
    ap.add_argument("--kalshi-balance", type=float, default=0.0,
                    help="[--arb] Available USD balance on Kalshi")
    ap.add_argument("--kalshi-fee-rate", type=float, default=0.07,
                    help="[--arb] Kalshi fee rate in fee ~ rate*p*(1-p) (default 0.07)")
    ap.add_argument("--poly-fee-rate", type=float, default=0.0,
                    help="[--arb] Polymarket fee rate (default 0)")
    ap.add_argument("--min-profit", type=float, default=0.02,
                    help="[--arb] Min profit per contract-pair after fees (default 0.02)")
    ap.add_argument("--max-leg-stake", type=float, default=1000.0,
                    help="[--arb] Max $ stake per leg (manual depth cap; default 1000)")
    ap.add_argument("--settlement-date", type=str, default="2026-07-19",
                    help="[--arb] Settlement date YYYY-MM-DD for annualized ROC")
    ap.add_argument("--no-ev", action="store_true", help="[--arb] Disable the +EV fallback")
    ap.add_argument("--no-dutch", action="store_true", help="[--arb] Disable the Dutch-book check")
```

In `main` in `main.py`, add this branch **before** the existing `asyncio.run(...)` call:

```python
    if args.arb:
        if args.poly_balance <= 0 or args.kalshi_balance <= 0:
            raise SystemExit("error: --arb requires positive --poly-balance and --kalshi-balance")
        y, m, d = (int(x) for x in args.settlement_date.split("-"))
        days = (date(y, m, d) - date.today()).days
        run_arb_cli(poly_balance=args.poly_balance, kalshi_balance=args.kalshi_balance,
                    kalshi_fee_rate=args.kalshi_fee_rate, poly_fee_rate=args.poly_fee_rate,
                    min_profit=args.min_profit, max_leg_stake=args.max_leg_stake,
                    days_to_settlement=days, kelly_fraction=args.kelly_fraction,
                    enable_ev=not args.no_ev, enable_dutch=not args.no_dutch)
        return
```

- [ ] **Step 6: Verify CLI arg validation**

Run: `python main.py --arb` (no balances)
Expected: exits with `error: --arb requires positive --poly-balance and --kalshi-balance`.

- [ ] **Step 7: Run the full test suite**

Run: `pytest -q`
Expected: PASS — all pre-existing tests plus the new arb/report/markets tests.

- [ ] **Step 8: Commit**

```bash
git add main.py tests/test_arb.py
git commit -m "feat: --arb CLI mode for cross-book arbitrage"
```

---

## Task 10: README + spec status update

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-06-02-cross-book-arbitrage-design.md`

- [ ] **Step 1: Document the `--arb` mode in the README**

Add a "Method A — cross-book arbitrage" section to `README.md` with a usage example:

```markdown
## Method A — cross-book arbitrage

Find risk-free locks between Polymarket and Kalshi on the outright-winner market:

```bash
python main.py --arb --poly-balance 5000 --kalshi-balance 5000 \
  --min-profit 0.02 --max-leg-stake 1000
```

It buys YES on the cheaper book and NO on the dearer book, sizes both legs with a
global LP across your two balances, and prints exact per-leg stakes plus guaranteed
profit (and annualized ROC). Risk-free **only** if both legs fill at the quoted
prices and you hold the position to settlement. `--min-profit` filters thin/phantom
arbs; raise it to be more conservative. `--no-ev` / `--no-dutch` turn off the extras.
```

- [ ] **Step 2: Mark the spec status as implemented**

In `docs/superpowers/specs/2026-06-02-cross-book-arbitrage-design.md`, change the
`**Status:**` line to `Status: Implemented 2026-06-02`.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/superpowers/specs/2026-06-02-cross-book-arbitrage-design.md
git commit -m "docs: document --arb cross-book arbitrage mode"
```

---

## Self-Review

**Spec coverage:**
- Lock detection, both orientations, fee-aware → Task 3 ✓
- Global LP sizing with venue balances + depth caps + integer flooring → Task 4 ✓
- Annualized ROC → Task 4 (`size_locks`) + Task 9 (date math) ✓
- Five honesty points: depth caps as manual knob (Task 3 `_cap`), capital-lock/annual ROC (Task 4 + report), Polymarket NO derived + `min_profit` gate (Tasks 1, 3), fees netted pre-lock (Task 3), thin EV via independent Kelly not `kelly_allocate` (Task 5) ✓
- Whole-field Dutch book → Task 6 ✓
- `markets.py` bid/NO-side fields + `fetch_books` (both books required) → Tasks 1, 7 ✓
- `ArbRec`/`EvBetRec`/`DutchBook` → Task 2 ✓
- Report (locks + EV + Dutch + disclaimer incl. leg-timing/lock warnings) → Task 8 ✓
- `--arb` CLI with all flags, no LLM in numeric path → Task 9 ✓
- Error handling: both-books-required raise (Task 7), positive-balance check (Task 9), LP failure → empty (Task 4), no-locks message (Task 8) ✓

**Placeholder scan:** none — every code step shows complete code; every run step shows the exact command and expected result.

**Type consistency:** `ArbRec`/`EvBetRec`/`DutchBook` fields defined in Task 2 are used identically in Tasks 4/5/6/8; `_Cand` fields (`a`, `b`, `fee`, `fee_poly`, `fee_kalshi`, `profit`, `yes_cap`, `no_cap`, venues) defined in Task 3 are consumed unchanged in Task 4; `VENUE_POLY`/`VENUE_KALSHI` constants used consistently; `run_arb` signature in Task 7 matches the `run_arb_cli` call in Task 9.

**Out of scope (per spec):** order execution, per-match markets, multi-level depth, live polling — none included. ✓
