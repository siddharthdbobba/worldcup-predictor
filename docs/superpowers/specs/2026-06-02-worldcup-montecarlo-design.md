# World Cup Predictor — Monte Carlo + Prediction-Market Ensemble

**Date:** 2026-06-02
**Status:** Design approved, pending spec review
**Target tournament:** 2026 FIFA World Cup (48 teams; kicks off 2026-06-11)

## Goal

A Claude Agent SDK application (Python) that forecasts each nation's probability of
winning the 2026 World Cup by **blending two independent signals**:

1. A **Monte Carlo simulation** of the full tournament driven by team strength (Elo).
2. **Prediction-market** implied probabilities (Polymarket + Kalshi).

It then acts as a **staking advisor**: given a user-supplied bankroll, it finds
positive-expected-value bets and recommends how much to stake on each using
fractional Kelly across mutually-exclusive outcomes.

The agent *orchestrates and wrangles data*; pure, deterministic, tested Python does
all numerical work. The LLM is never in the critical path of the math.

## Architecture (Approach A — agent orchestrates, deterministic tools compute)

The agent uses the Claude Agent SDK and is given a small set of **custom tools**. Its
job is the part agents are good at: deciding which sources to hit, parsing messy
web/API responses into clean structured data, adapting when a source is down, and
composing the final report. The simulation and blending are deterministic code.

Approaches B (agent writes/runs the sim each time) and C (plain script, thin agent
wrapper) were rejected: B puts the LLM in the numerical critical path
(non-reproducible, untestable); C barely uses the SDK.

## Project layout

```
worldcup-predictor/
  main.py                  # CLI entrypoint → runs the agent (accepts --bankroll)
  pyproject.toml           # deps: claude-agent-sdk, numpy, scipy, httpx, rich, pytest
  .env.example             # ANTHROPIC_API_KEY
  README.md
  src/worldcup/
    agent.py               # Agent SDK setup: registers tools, system prompt, runs loop
    models.py              # dataclasses: Team, Group, MatchModelParams, Forecast, BetRec
    teamnames.py           # canonical team-name normalization
    ratings.py             # [I/O] fetch + parse Elo/strength ratings  → tool
    markets.py             # [I/O] fetch + parse Polymarket + Kalshi (price, ask, liquidity) → tool
    draw.py                # [I/O] fetch + validate the 12×4 group draw → tool
    simulator.py           # [PURE] deterministic, seedable Monte Carlo core → tool
    blend.py               # [PURE] combine model% + market% → final forecast → tool
    stake.py               # [PURE] value detection + fractional Kelly bankroll allocation → tool
    report.py              # render rich terminal table + markdown report (forecast + betting card)
  tests/
    test_simulator.py  test_groups.py  test_blend.py  test_teamnames.py  test_stake.py
    fixtures/              # saved JSON so data-parsing is testable offline
```

## Components

| Component | Pure/IO | Responsibility |
|-----------|---------|----------------|
| `draw.py` | I/O | Fetch live group draw; **validate exactly 12 groups × 4 teams**, fail loudly if any slot unresolved |
| `ratings.py` | I/O | Fetch current international Elo strength for all 48 teams; normalize names |
| `markets.py` | I/O | Fetch Polymarket + Kalshi prices, **ask price/spread**, and liquidity/volume; strip vig → implied prob per priced team |
| `teamnames.py` | pure | Canonical name map so ratings/markets/draw align |
| `simulator.py` | **pure** | Seedable Monte Carlo: group stage → knockout → champion counts. No I/O. |
| `blend.py` | **pure** | Liquidity-weighted blend of model% + market%, with unpriced fallback + renormalize |
| `stake.py` | **pure** | Detect +EV bets (`edge = model% − market%`); allocate bankroll via fractional Kelly across mutually-exclusive outcomes; cap by liquidity |
| `report.py` | I/O | Terminal tables (forecast: Model/Market/Blended; betting card) + saved markdown report |
| `agent.py` | — | Registers tools; system prompt drives orchestration |

**Agent-facing tools:** `fetch_group_draw`, `fetch_team_ratings`,
`fetch_market_probabilities`, `run_simulation`, `blend_forecast`, `recommend_bets`.

## Data flow

```
1. fetch_group_draw()            → validate 12×4, else STOP loudly
2. fetch_team_ratings()          → {team: rating} for all 48 (names normalized)
3. fetch_market_probabilities()  → {team: (prob, ask, liquidity)} for priced teams (~40 of 48)
4. run_simulation(ratings, draw, seed=42, n=20000)  → model% per team
5. blend_forecast(model%, market%, liquidity, w_cap, K)  → final forecast
6. recommend_bets(model%, market%, ask, liquidity, bankroll, kelly_fraction)  → bet card
7. report()                      → forecast table + betting card + markdown file
```

Bankroll comes from the CLI (`--bankroll`, default e.g. $100; the agent may also
prompt for it). If no bankroll is supplied, steps 1–5/7 still run — the betting card
is simply omitted.

## Match model (shared by both stages)

Group tiebreakers need **goal difference and goals scored**, so the model produces
*scorelines*, not just W/D/L. Both stages share a goals-based model:

- Expected goals from the Elo difference:
  `λ_A = BASE · 10^((R_A − R_B)/S)`, `λ_B = BASE · 10^((R_B − R_A)/S)`
  (BASE ≈ average goals/team; S a scale constant; both calibrated to sane defaults).
- Each match: `G_A ~ Poisson(λ_A)`, `G_B ~ Poisson(λ_B)`.
- **Host bump (v1-on):** USA / Canada / Mexico get a small Elo boost in home matches.

- **Group match:** scoreline stands → 3 / 1 / 0 points.
- **Knockout match:** if level after 90', resolve extra-time/penalties as a
  strength-weighted coin flip `P(A) = 1 / (1 + 10^(−(R_A−R_B)/400))`. No surviving draws.

## Group stage → advancement

- Simulate all 6 matches per group; build the table.
- **Tiebreakers (v1, explicitly simplified):** points → goal difference → goals scored
  → random draw. **Head-to-head mini-tables are deferred** (fiddly, rarely move a
  pre-tournament forecast). This simplification is stated, not silent.
- Advance **top 2 per group** (24) + **8 best third-placed teams** — a named sub-routine
  ranking all 12 third-place finishers by the same tiebreaker chain.
- Map the 32 into FIFA's **fixed Round-of-32 bracket**, then simulate
  R32 → R16 → QF → SF → Final.

## Monte Carlo

- `numpy` `default_rng(seed)` — reproducible. `n_simulations` default 20,000.
- Count championships per team → `model%` (sums to 100% across all 48).

## Prediction markets

- **Sources:** Polymarket `gamma-api` (`world-cup-winner` event, per-team Yes/No markets)
  and Kalshi (`KXMENWORLDCUP-26`). Hit the **JSON APIs directly**, never scrape rendered
  pages. Verified 2026-06-02: ~40+ of 48 teams are priced; aggregated outright volume
  ≈ $500M (liquid).
- **A price is a probability:** a "Yes" share at $0.17 ≈ 17%.
- **Strip the vig:** raw prices sum to >100% (overround); normalize across priced teams
  so they sum to 100%. Do this per market.
- **Dual-market consensus**, liquidity-weighted per team:
  `market_i = (L_poly·p_poly + L_kalshi·p_kalshi) / (L_poly + L_kalshi)`

## Blending (default: liquidity-weighted)

Each team gets its own market weight, driven by that market's liquidity/open interest
(the confidence signal; falls back to cumulative volume where liquidity is absent):

```
w_i        = w_cap · ( L_i / (L_i + K) )
blended_i  = w_i · market_i + (1 − w_i) · model_i
            then renormalize so Σ blended = 100%
```

- `L_i` = liquidity for team i (0 if unpriced)
- `K`   = half-saturation constant (configurable; at `L_i = K`, market gets half of `w_cap`)
- `w_cap` = max market weight, e.g. 0.70 (configurable) — the market never *fully*
  overrides the model

**Properties:**
- High-liquidity favorites → market dominates; thin longshots → model dominates.
- **Unpriced fallback falls out for free:** `L_i = 0 → w_i = 0 →` model-only.
- Longshot forecasts are therefore model-driven by design (stated, intended).
- A simple **fixed-`w` mode** is retained for comparison.
- Volume/liquidity affects only the *weight* of the market signal, never the price itself.

Output rows: `team, model%, market%, blended%`, sorted by blended%. Showing all three
columns surfaces where the model disagrees with the market.

## Staking advisor (`stake.py`)

Given a bankroll, recommend where to put money. **Pure, deterministic, fully tested.**

**Betting-truth estimate is separate from the forecast.** The forecast table is the
liquidity-weighted blend; the *betting edge* is **`edge_i = model%_i − market%_i`**.
Rationale: the simulation is the one signal the market price does not already contain,
so betting wagers that independent signal against the price. Using the
liquidity-weighted blend here would be circular — it would shrink the edge on liquid
favorites and inflate it on thin longshots, steering bets toward exactly the markets
where both signals are least reliable and you can't get size on. We do not do that.

**Price you bet against:** the **ask price (incl. spread)** you would actually pay,
not the mid/last trade. EV and Kelly are computed against the ask.

**Value detection:** keep only teams where buying at the ask is +EV, i.e.
`model%_i > ask_i`. Expected value per $1 staked: `EV_i = model%_i / ask_i − 1`.

**Stake sizing — fractional Kelly across mutually-exclusive outcomes.**
Only one team wins the cup, so per-team independent Kelly over-stakes (correlated
outcomes). Instead, maximize expected log-wealth over the stake vector subject to
`Σ stake ≤ bankroll`:

```
maximize  Σ_outcomes  P(outcome) · log(wealth after that outcome)
over      stakes f_i ≥ 0,  Σ f_i ≤ 1   (fractions of bankroll)
```

This is a small **convex** problem — solve numerically with `scipy.optimize`
(robust and easy to unit-test) rather than a hand-coded closed form. Apply the chosen
**Kelly fraction** (default ½, configurable) to the optimal stakes to reduce variance.

**Liquidity is an execution cap, not just a weight:** cap each recommended stake by the
depth actually available at the ask, so the tool never tells you to bet more than the
book can absorb.

**Realistic expectations (stated, not a bug):** against a liquid, vig'd ~$500M market,
**very few teams clear +EV** after the overround. The honest output is often
**0–2 value bets, sometimes none** — not a 20-team portfolio. The report says so when
the list is short or empty.

**Output — the betting card:** rows of
`team, ask price, model%, market%, edge, EV%, recommended stake $, potential profit $`,
plus `total staked`, `bankroll held in reserve`, and a one-line **responsible-gambling
disclaimer** ("model-based estimate; markets are efficient; never stake more than you
can afford to lose").

## Error handling (fail loud, never fabricate)

- **Group draw** not exactly 12×4 / any unresolved slot → **STOP** with a clear message.
- **Ratings source down** → try an alternate; if none, **STOP** (never invent ratings).
- **One market down** → use the other. **Both down** → emit **model-only** forecast with
  a loud warning (blending requires a market).
- **Name mismatch** → `teamnames.py` canonicalizes; any draw team that fails to map to a
  rating is reported, not silently dropped.
- **Bankroll** missing → skip the betting card (forecast still produced); non-positive
  bankroll → clear error. **No market data** → no betting card (staking requires prices).
- **No +EV bets found** → not an error; report "no value bets at current prices."

## Testing (pytest, all offline)

- `test_simulator.py` — tiny synthetic 8-team / 2-group world + fixed seed: deterministic
  output, probabilities sum ~100%, stronger team wins more often, equal ratings → ~equal
  outcomes.
- `test_groups.py` — crafted tables exercise points, GD/goals tiebreakers, and the
  8-best-thirds ranking.
- `test_blend.py` — liquidity-weighting math, unpriced fallback, fixed-`w` mode,
  renormalization sums to 100%.
- `test_stake.py` — `edge = model − market` (never the blend); only +EV teams
  recommended; Kelly stakes never exceed bankroll; liquidity cap respected; a no-edge
  input returns **zero** bets; a clear single-edge input returns one sane stake.
- `test_teamnames.py` — known aliases map correctly (USA↔United States,
  South Korea↔Korea Republic, …).
- Data parsing validated against **saved fixture JSON** (live network not in unit tests).

## Stack

`claude-agent-sdk` (Python) · `numpy` · `scipy` (Kelly optimization) · `httpx` ·
`rich` · `pytest`.

## Explicitly out of scope (v1)

- Head-to-head group tiebreakers (deferred to v2).
- Live/in-tournament re-forecasting on a schedule (possible v2).
- Per-match xG / lineup-level modeling beyond Elo-derived goals.
- Web UI (CLI + markdown report only).

## Open items — status after live-data wiring (2026-06)

Resolved:
- **Elo source:** `eloratings.net/World.tsv` joined to `en.teams.tsv` (code→name).
  46/48 teams matched on canonical names automatically; 2 aliases added
  (Congo DR→DR Congo, Curacao→Curaçao). Parse is sanity-asserted.
- **Market field names:** Polymarket `gamma-api` uses `groupItemTitle`,
  `lastTradePrice`, `bestAsk`, `liquidityNum`, `volumeNum` (0–1 prices). Kalshi
  uses `*_dollars` fields (0–1, **not** cents) and `volume_fp`/`open_interest_fp`;
  `liquidity_dollars` is 0 in practice. Confidence metric switched to **volume,
  normalized per-book to [0,1]** (scale-free); blend `K` recalibrated to 0.5 for
  that [0,2] confidence range (favorites land at w≈0.45–0.55). Polymarket
  `liquidityNum` is used as the dollar stake-depth cap.
- **Draw source:** derived from Kalshi `KXWCGROUPWIN` group-winner markets
  (`event_ticker` → group letter, `yes_sub_title` → team). Committed to
  `data/draw_2026.json`; live-fetch-with-committed-fallback (group markets close
  at kickoff). Stale non-qualified markets excluded by intersecting with the 48.
- **Model calibration:** `scale` calibrated to 2000 against the live field so the
  Elo favorite lands at a realistic ~20% (was ~53% at 600). `base`=1.35, `w_cap`
  =0.7, host_bump=60, Kelly fraction=½.

Still open (v2):
- Tail calibration: the one-knob Elo–Poisson model still over-rates longshots vs
  the sharp market, yielding more "value bets" than market efficiency supports
  (Kelly sizes them tiny). A min-edge/EV gate or richer model would tighten this.
- The exact FIFA Round-of-32 bracket mapping (v1 uses a seeded approximation).
