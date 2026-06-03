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

## Test
```bash
.venv/bin/pytest -q
```

## Disclaimer
Educational tool. Prediction markets are highly efficient; expect **few or zero**
value bets. Never stake more than you can afford to lose.
