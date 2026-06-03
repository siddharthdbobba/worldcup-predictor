"""CLI entrypoint for the World Cup predictor agent."""
import argparse
import asyncio
import sys
from datetime import date

from rich.console import Console

from worldcup.agent import run_agent
from worldcup.arb import run_arb
from worldcup.draw import fetch_group_draw
from worldcup.markets import combine_markets, fetch_books
from worldcup.report import print_arb_report


def run_arb_cli(*, poly_balance, kalshi_balance, kalshi_fee_rate, poly_fee_rate,
                min_profit, max_leg_stake, days_to_settlement, kelly_fraction,
                enable_ev, enable_dutch):
    """Fetch live data and run method A (no LLM in the numerical path)."""
    groups = fetch_group_draw()
    valid = {t for ts in groups.values() for t in ts}
    poly, kalshi = fetch_books(valid_teams=valid)
    # Derive the consensus from the SAME books we fetched (one round-trip per venue),
    # rather than re-fetching — avoids extra latency and quote-skew between the lock
    # prices and the EV-tier consensus.
    consensus, _ask, _conf, _depth = combine_markets(poly, kalshi, valid_teams=valid)
    res = run_arb(poly, kalshi, consensus, poly_balance=poly_balance,
                  kalshi_balance=kalshi_balance, max_leg_stake=max_leg_stake,
                  days_to_settlement=days_to_settlement, kalshi_fee_rate=kalshi_fee_rate,
                  poly_fee_rate=poly_fee_rate, min_profit=min_profit,
                  kelly_fraction=kelly_fraction, enable_ev=enable_ev,
                  enable_dutch=enable_dutch)
    console = None if sys.stdout.isatty() else Console(width=200)
    print_arb_report(res, poly_balance, kalshi_balance, console=console)


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
    return ap.parse_args()


def main():
    args = parse_args()
    if args.bankroll is not None and args.bankroll <= 0:
        raise SystemExit("error: --bankroll must be positive")
    if args.min_edge < 0:
        raise SystemExit("error: --min-edge must be >= 0")
    if args.arb:
        if args.poly_balance <= 0 or args.kalshi_balance <= 0:
            raise SystemExit("error: --arb requires positive --poly-balance and --kalshi-balance")
        try:
            settlement = date.fromisoformat(args.settlement_date)
        except ValueError:
            raise SystemExit("error: --settlement-date must be ISO format YYYY-MM-DD")
        days = (settlement - date.today()).days
        run_arb_cli(poly_balance=args.poly_balance, kalshi_balance=args.kalshi_balance,
                    kalshi_fee_rate=args.kalshi_fee_rate, poly_fee_rate=args.poly_fee_rate,
                    min_profit=args.min_profit, max_leg_stake=args.max_leg_stake,
                    days_to_settlement=days, kelly_fraction=args.kelly_fraction,
                    enable_ev=not args.no_ev, enable_dutch=not args.no_dutch)
        return
    min_edge = 0.0 if args.all_bets else args.min_edge
    asyncio.run(run_agent(args.bankroll, args.sims, args.seed,
                          args.kelly_fraction, min_edge))


if __name__ == "__main__":
    main()
