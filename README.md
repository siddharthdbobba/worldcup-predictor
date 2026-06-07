# World Cup Predictor

A Claude Agent SDK app that forecasts each nation's 2026 World Cup title odds by
**blending a Monte Carlo simulation with prediction-market prices**, then recommends
Kelly-sized value bets from a bankroll you choose.

## How it works
- **Simulation** (`simulator.py`): Elo-driven Poisson scorelines → group stage (with
  tiebreakers + 8 best thirds) → seeded knockout → 20k-run championship probabilities.
- **Markets** (`markets.py`): Polymarket + Kalshi outright prices, de-vigged and
  liquidity-weighted into a consensus.
- **Blend** (`blend.py`): liquidity-weighted mix of model + market (configurable).
- **Staking** (`stake.py`): bets the *model vs market* edge at the ask, sized by
  fractional Kelly across mutually-exclusive outcomes, capped by liquidity.

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env   # add ANTHROPIC_API_KEY
```
The key in `.env` is loaded automatically at startup (an already-exported
`ANTHROPIC_API_KEY` env var takes precedence, for CI/containers). It's only needed for
the forecast/betting-card path below — **`--arb` (method A) needs no API key**.

## Run
```bash
.venv/bin/python main.py --bankroll 100 --sims 20000
```
Omit `--bankroll` for a forecast with no betting card.

By default the betting card only lists bets where the model disagrees with the
market by at least 5 percentage points (`--min-edge 0.05`) — this filters the long
tail of tiny longshot edges that are usually model noise. Loosen or tighten it:
```bash
.venv/bin/python main.py --bankroll 100 --min-edge 0.03   # 3pp gate
.venv/bin/python main.py --bankroll 100 --all-bets        # show every +EV bet
```

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

Additional flags:
- `--kalshi-fee-rate` / `--poly-fee-rate` — override the per-venue fee model
- `--settlement-date YYYY-MM-DD` — used for annualized-ROC calculation (default `2026-07-19`)

## Calibrating the model (`--calibrate`)

The match model's `base`/`scale` default to a hand-tuned guess. To fit them to data
instead, replay historical international results through an Elo engine and
maximum-likelihood-fit the Poisson goals model:

```bash
.venv/bin/python main.py --calibrate --history-years 15
```

This fetches a public results dataset (martj42), recomputes each match's pre-match Elo,
fits `base`/`scale`, and writes `src/worldcup/data/calibrated_params.json` (gitignored —
it's machine-generated; regenerate as data updates). Subsequent forecasts **load it
automatically**; pass `--no-calibrated` to ignore it. The fitted `scale` is rescaled to
eloratings.net's live rating spread so it transfers to the forecast — sanity-check that a
forecast's favorite still lands in a sensible range. Flags: `--all-matches` (default is
neutral-site only, to avoid home-advantage confounding), `--history-url` to override the
source.

## Test
```bash
.venv/bin/pytest -q
```

## Disclaimer
Educational tool. Prediction markets are highly efficient; expect **few or zero**
value bets. Never stake more than you can afford to lose.
