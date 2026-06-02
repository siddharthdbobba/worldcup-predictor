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
    ap.add_argument("--min-edge", type=float, default=0.05,
                    help="Min model-vs-market edge (probability points) for a value "
                         "bet; e.g. 0.05 = 5pp. Use 0 (or --all-bets) to show every "
                         "+EV bet. Default 0.05.")
    ap.add_argument("--all-bets", action="store_true",
                    help="Show every +EV value bet (equivalent to --min-edge 0)")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.bankroll is not None and args.bankroll <= 0:
        raise SystemExit("error: --bankroll must be positive")
    if args.min_edge < 0:
        raise SystemExit("error: --min-edge must be >= 0")
    min_edge = 0.0 if args.all_bets else args.min_edge
    asyncio.run(run_agent(args.bankroll, args.sims, args.seed,
                          args.kelly_fraction, min_edge))


if __name__ == "__main__":
    main()
