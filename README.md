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

## Test
```bash
.venv/bin/pytest -q
```

## Disclaimer
Educational tool. Prediction markets are highly efficient; expect **few or zero**
value bets. Never stake more than you can afford to lose.
