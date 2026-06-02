"""CLI entrypoint for the World Cup predictor agent."""
import argparse
import asyncio

from worldcup.agent import run_agent


def parse_args():
    ap = argparse.ArgumentParser(description="2026 World Cup forecaster + staking advisor")
    ap.add_argument("--bankroll", type=float, default=None,
                    help="Bankroll in dollars; omit to skip the betting card")
    ap.add_argument("--sims", type=int, default=20000, help="Monte Carlo simulations")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (reproducibility)")
    ap.add_argument("--kelly-fraction", type=float, default=0.5,
                    help="Fraction of full Kelly (default 0.5)")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.bankroll is not None and args.bankroll <= 0:
        raise SystemExit("error: --bankroll must be positive")
    asyncio.run(run_agent(args.bankroll, args.sims, args.seed, args.kelly_fraction))


if __name__ == "__main__":
    main()
