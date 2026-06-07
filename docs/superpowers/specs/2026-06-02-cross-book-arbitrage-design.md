# Method A — Cross-Book Arbitrage (Polymarket × Kalshi)

**Date:** 2026-06-02
**Status:** Implemented 2026-06-02
**Target market:** 2026 FIFA World Cup **outright-winner** market (48 teams), the
same market `markets.py` already fetches from Polymarket and Kalshi.

## Goal

A new staking strategy ("method A") that exploits **price differences between the
two books on the same outcome**. Each team is a binary YES/NO contract on each book.
Where the two books disagree enough, you can take **opposing sides across the books**
and lock a profit independent of who actually wins the Cup.

Two tiers, **arb-first**:

1. **Risk-free arbitrage (primary):** when a true lock exists (after fees), size it
   with a **global LP** that maximizes total locked profit subject to your balance on
   each venue and a per-leg stake cap. This is the part the user explicitly asked for:
   *calculate the amount of money to bet on each leg.*
2. **+EV cross-book bets (secondary, thin):** when no lock exists but the books still
   disagree, flag the side a book underprices vs. the two-book consensus. **Positive
   expected value, NOT guaranteed** — sized small with independent fractional Kelly.

Plus an optional cheap **whole-field Dutch-book** check.

Like the rest of the repo: **detect + size + report only.** No order execution.
Pure, deterministic, fully tested math; no LLM in the numerical path.

## Why a price difference alone is not enough (the crux)

A gap between the two books' percentages is *necessary but not sufficient* for
guaranteed profit. Per team, each book offers a YES contract (pays $1 if the team wins
the Cup). To lock profit you buy **YES on the cheaper book** and **NO on the dearer
book**. NO costs `1 − that book's YES bid`. So the lock condition is:

```
cheap_book_YES_ask  <  dear_book_YES_bid          (after fees)
  ⇔  a + b < 1,  where a = YES ask (cheap),  b = NO ask (dear) = 1 − YES bid (dear)
```

The spreads must **not overlap** — a 40% vs 30% mid-price gap can still be *no* arb if
the spreads are wide. This is the difference between a real lock and a mirage.

## The lock: orientations and profit

For each team priced on **both** books, evaluate two orientations:

| Orientation | YES leg | NO leg | Fires iff |
|-------------|---------|--------|-----------|
| **A** | YES on Polymarket @ `a` | NO on Kalshi @ `b` | `kalshi_yes_bid > poly_yes_ask` |
| **B** | YES on Kalshi @ `a` | NO on Polymarket @ `b` | `poly_yes_bid > kalshi_yes_ask` |

At most one is positive (unless books are crossed). Evaluate both, keep the positive
one. With **equal contracts on both legs**, payout is identical whoever wins:

- Buy `N` YES + `N` NO → payout `= N` in **both** outcomes.
- Cost `= N·(a + b) + fees`. **Guaranteed profit `= N·(1 − a − b) − fees`.**
- **Return on capital `= (1 − a − b − fee_per_pair) / (a + b + fee_per_pair)`** — the
  denominator is total capital outlaid including fees (matches `profit / total_cost` in code).

Given a budget `C` for one opportunity: `N = C/(a+b)`, money on YES leg `= a·C/(a+b)`,
money on NO leg `= b·C/(a+b)`. (The LP below generalizes this across all teams and
both venue balances rather than per-opportunity.)

## Sizing — the global LP

Decision variables: `xᵢ ≥ 0` = number of **contract-pairs** for arb candidate `i`
(one pair = one YES + one NO, so each `xᵢ` is hedged by construction). For each `i`,
`yes_venue(i)` and `no_venue(i)` are opposite books.

```
maximize    Σᵢ  xᵢ · (1 − aᵢ − bᵢ − feeᵢ)                 # total locked profit
subject to  Σ_{i: yes on Poly}  aᵢ·xᵢ  +  Σ_{i: no on Poly}  bᵢ·xᵢ  ≤  poly_balance
            Σ_{i: yes on Kalshi} aᵢ·xᵢ +  Σ_{i: no on Kalshi} bᵢ·xᵢ ≤  kalshi_balance
            aᵢ·xᵢ ≤ yes_leg_depth_capᵢ      (per leg)
            bᵢ·xᵢ ≤ no_leg_depth_capᵢ        (per leg)
            xᵢ ≥ 0
```

Solved with `scipy.optimize.linprog` (the repo already depends on scipy; `linprog`
minimizes `−profit`). Per-team hedges are **genuinely independent** — `NOᵢ` pays
whenever team `i` loses, regardless of who wins — so there is no cross-coupling and the
LP is valid. After solving, **floor each `xᵢ` to whole contracts** and recompute exact
stakes and profit from the integer counts.

**Output per arb (the betting card):** `team, yes_venue, no_venue, contracts,
yes_ask a, no_ask b, stake_yes $, stake_no $, total cost $, guaranteed profit $,
ROC %, annualized ROC %`.

## Five honesty points (built in, not footnoted)

1. **Depth caps are a manual safety knob, not measured liquidity.** Kalshi's liquidity
   field reads 0 in practice and Polymarket's `liquidityNum` is whole-book, not
   top-of-book. So the per-leg cap defaults to a configurable `--max-leg-stake`; the LP
   respects it but it is *your* cap, not a real order-book measurement. The arb price is
   only valid up to top-of-book — beyond it you walk the book and the quoted ask is wrong.
   Where Polymarket per-team `depth` is present and smaller than `--max-leg-stake`, use it.
2. **Capital-lock reality, shown in the report.** A lock is typically a few percent
   *total*, with capital **frozen on both venues until the tournament settles** (months).
   The report shows **annualized ROC** beside the raw % so it is never dressed up as free
   money. (Annualization uses days-to-settlement from a configurable settlement date.)
3. **Polymarket NO is derived** as `1 − bestBid`, an approximation (separate YES/NO CLOB
   books are not perfectly complementary); Kalshi NO is quoted directly
   (`no_ask_dollars`) and used as-is. Mitigation: a **`--min-profit` gate defaulted above
   zero** (e.g. require profit-per-pair > 0.02 after fees); raw `>0` arbs are filtered as
   likely phantom.
4. **Fees netted before the lock test.** Kalshi fee per contract `≈ fee_rate·p·(1−p)`
   (configurable `--kalshi-fee-rate`, default 0.07, where `p` is the Kalshi leg's price);
   Polymarket fee configurable, default 0. `feeᵢ` per pair = sum of both legs' per-contract
   fees.
5. **EV fallback is deliberately thin.** When no lock exists, surface +EV cross-book bets
   (a book's YES ask below the volume-weighted two-book consensus from `combine_markets`).
   Sized by **independent per-bet fractional Kelly with a global stake cap** — **not** the
   existing `kelly_allocate`, which assumes mutually-exclusive championship outcomes
   summing to ≤ 1 and *raises* otherwise (EV-fallback bets mix YES-on-X and NO-on-Y and do
   not fit that structure). Clearly a secondary report section.

## Optional add-on — whole-field Dutch book

Buy the **cheapest YES per team across both books**; if `Σ min(poly_ask, kalshi_ask) < 1`
(after fees), the entire field is covered for under $1 → guaranteed $1 payout. Rarely
fires with 48 de-vigged teams and is a different portfolio shape (covers all teams at
once), so it is a **standalone check** that reports the field sum and the gap to 1.0, not
part of the per-team LP.

## Data-layer changes (`markets.py`)

The current parsers capture only `prob`, `ask` (YES ask), `vol`, `depth`. The NO leg
needs the **bid** (and, on Kalshi, the directly-quoted NO side). **Additive** changes —
keep the existing `ask` key as an alias so existing code and tests do not break:

- **Polymarket** (`parse_polymarket`): add `yes_bid` from `bestBid`; derive
  `no_ask = 1 − bestBid`. Keep `ask = bestAsk`.
- **Kalshi** (`parse_kalshi`): add `yes_bid` from `yes_bid_dollars`,
  `no_ask = no_ask_dollars`, `no_bid = no_bid_dollars`. Keep `ask = yes_ask_dollars`.
- Per-team dict gains `yes_bid`, `no_ask` (both books) and `no_bid` (Kalshi). `combine_markets`
  is unchanged for the forecast path; method A reads the **raw per-book dicts** (it must
  see the two books separately, not the consensus), so `arb.py` consumes the outputs of
  `parse_polymarket` / `parse_kalshi` directly via a small fetch helper.

## Files

| File | Change | Responsibility |
|------|--------|----------------|
| `src/worldcup/arb.py` | **new** | Lock detection (both orientations, fee-aware), global LP sizing, thin EV fallback, optional Dutch-book check |
| `src/worldcup/markets.py` | modify | Capture `yes_bid`/`no_ask`/`no_bid`; add a helper returning the two raw per-book dicts for `arb.py` |
| `src/worldcup/models.py` | modify | Add `ArbRec` dataclass (and `EvBetRec` if the EV tier needs its own row) |
| `src/worldcup/report.py` | modify | Arb betting-card formatter (locks + EV section + Dutch-book line) with disclaimer |
| `main.py` | modify | `--arb` mode; flags: `--poly-balance`, `--kalshi-balance`, `--kalshi-fee-rate`, `--poly-fee-rate`, `--min-profit`, `--max-leg-stake`, `--settlement-date` |
| `tests/test_arb.py` | **new** | See Testing |
| `tests/fixtures/` | add | Crafted Poly/Kalshi JSON producing a known lock and a known no-arb |

## `ArbRec` (draft)

```python
@dataclass
class ArbRec:
    team: str
    yes_venue: str        # "polymarket" | "kalshi"
    no_venue: str
    yes_ask: float        # a
    no_ask: float         # b
    contracts: int        # floored xᵢ
    stake_yes: float
    stake_no: float
    total_cost: float
    guaranteed_profit: float
    roc: float            # profit / total_cost
    annual_roc: float
```

## Error handling (fail loud, never fabricate)

- **One book down** → no cross-book arb possible (need both); report that clearly, skip
  method A. (The forecast path still degrades to the surviving book as today.)
- **Missing bid** on a team → that team cannot form the NO leg → skip the team, note it.
- **Non-positive balance** on either venue → clear error.
- **LP infeasible / solver failure** → report no sized arbs with the solver message; never
  emit unsized "phantom" arbs.
- **No locks found** → not an error; report "no risk-free arbitrage at current prices"
  and fall through to the EV tier (if enabled).
- **Responsible-gambling + execution-risk disclaimer** on every betting card: prices move
  between the two legs (leg-timing risk); capital is locked until settlement; depth caps
  are user-set, not measured.

## Testing (pytest, all offline)

`tests/test_arb.py`:

- **Lock detection** — crafted bid/ask where `kalshi_yes_bid > poly_yes_ask` → orientation
  A flagged with correct `a`, `b`; the reverse case → orientation B; overlapping spreads →
  no lock.
- **Fee gate** — an arb that is +profit on raw prices but ≤ 0 after the Kalshi fee → not
  flagged; the `--min-profit` threshold filters a thin (<2%) lock.
- **Equal-payout sizing** — for a single arb, `N` YES + `N` NO yields identical payout in
  both outcomes; `stake_yes = a·C/(a+b)`, `stake_no = b·C/(a+b)`.
- **LP, binding balance** — small `kalshi_balance` caps total contracts; Poly/Kalshi
  capital used ≤ respective balances.
- **LP, binding depth** — small `--max-leg-stake` caps a leg before balance does.
- **Integer flooring** — fractional LP solution floored to whole contracts; recomputed
  profit matches the integer stakes.
- **Multi-arb** — two independent locks sized together respect both venue balances.
- **EV fallback** — no-lock input with a book below consensus → +EV bet flagged, Kelly
  stake ≤ cap; never raises the way `kelly_allocate` would on incoherent inputs.
- **Dutch-book** — crafted field where `Σ min-asks < 1` → flagged; normal field → not.
- **Data parsing** — `parse_polymarket`/`parse_kalshi` capture `yes_bid`/`no_ask` from
  fixtures; existing `ask`-based tests still pass.

## Stack

`numpy` · `scipy.optimize.linprog` (LP) + existing fractional Kelly (`scipy.optimize`) ·
`httpx` · `rich` · `pytest`. No new dependencies.

## Explicitly out of scope (this iteration)

- **Order execution** via venue APIs (auth, balance sync, leg-timing automation) — a
  later iteration; this one is detect + size + report.
- **Per-match (Team A vs Team B) markets** — a different data source; this targets the
  outright-winner market only.
- **Multi-level order-book depth modeling** — only top-of-book / user cap (the data does
  not expose reliable depth).
- **Live polling / continuous monitoring** — single-shot scan per invocation.
