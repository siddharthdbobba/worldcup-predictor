"""CLI entrypoint for the World Cup predictor agent."""
import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from worldcup.agent import run_agent
from worldcup.arb import run_arb
from worldcup.calibrate import fit_params, rescale_to_live
from worldcup.draw import fetch_group_draw
from worldcup.elo_engine import replay
from worldcup.history import RESULTS_URL, fetch_matches
from worldcup.markets import combine_markets, fetch_books
from worldcup.models import CALIBRATED_PARAMS_PATH, MatchModelParams
from worldcup.ratings import fetch_ratings
from worldcup.report import print_arb_report


def run_arb_cli(*, poly_balance, kalshi_balance, kalshi_fee_rate, poly_fee_rate,
                min_profit, max_leg_stake, days_to_settlement, kelly_fraction,
                enable_ev, enable_dutch, min_ev=0.0, min_ask=0.02):
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
                  kelly_fraction=kelly_fraction, min_ev=min_ev, min_ask=min_ask,
                  enable_ev=enable_ev, enable_dutch=enable_dutch)
    console = None if sys.stdout.isatty() else Console(width=200)
    print_arb_report(res, poly_balance, kalshi_balance, console=console)


def run_calibrate_cli(*, since_year, all_matches, history_url):
    """Fetch results, replay Elo, MLE-fit base/scale, write calibrated_params.json."""
    console = Console()
    console.print(f"[dim]Fetching international results since {since_year} and "
                  f"replaying Elo…[/dim]")
    matches = fetch_matches(since_year=since_year, url=history_url or RESULTS_URL)
    rows = replay(matches)
    res = fit_params(rows, neutral_only=not all_matches)

    # The fitted scale is relative to our replayed-Elo spread, which is narrower than
    # eloratings.net's live ratings (what forecasts use). Rescale it to live units so it
    # transfers; otherwise the raw fit over-dramatizes live Elo gaps (favorite too high).
    fit_elos = [e for r in rows if (r.neutral or all_matches)
                for e in (r.elo_home, r.elo_away)]
    live = fetch_ratings()
    scale_live = rescale_to_live(res.scale, fit_elos, live)

    old = MatchModelParams()
    new = MatchModelParams(base=res.base, scale=scale_live,
                           host_bump=old.host_bump, hosts=old.hosts)
    payload = new.to_dict()
    payload["_meta"] = {"n_matches": res.n_matches, "ll_before": res.ll_before,
                        "ll_after": res.ll_after, "neutral_only": not all_matches,
                        "since_year": since_year, "converged": res.converged,
                        "scale_fit_replay_units": res.scale}
    CALIBRATED_PARAMS_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    t = Table(title="Calibration — base/scale MLE fit")
    for col in ("Param", "Old (default)", "Fitted"):
        t.add_column(col, justify="left" if col == "Param" else "right")
    t.add_row("base", f"{old.base:.3f}", f"{new.base:.3f}")
    t.add_row("scale", f"{old.scale:.0f}", f"{new.scale:.0f}")
    console.print(t)
    matchset = "neutral-site only" if not all_matches else "all matches (+home bump)"
    console.print(f"matches used: {res.n_matches:,} ({matchset}) · "
                  f"logLik {res.ll_before:,.0f} → {res.ll_after:,.0f} "
                  f"(Δ {res.ll_after - res.ll_before:+,.0f}) · converged={res.converged}")
    console.print(f"[dim]scale fitted on replayed Elo = {res.scale:.0f}; rescaled to live "
                  f"eloratings spread → {scale_live:.0f}[/dim]")
    console.print(f"[green]Wrote {CALIBRATED_PARAMS_PATH}[/green] — forecasts now use it "
                  f"(disable with --no-calibrated).")
    console.print("[yellow]Sanity-check[/yellow] a forecast's favorite still lands ~15–25%.")


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
    ap.add_argument("--no-calibrated", action="store_true",
                    help="Ignore a saved calibration and use the default base/scale")
    ap.add_argument("--calibrate", action="store_true",
                    help="Calibrate base/scale by backtest, write calibrated_params.json, exit")
    ap.add_argument("--history-years", type=int, default=15,
                    help="[--calibrate] How many years of results to fit on (default 15)")
    ap.add_argument("--all-matches", action="store_true",
                    help="[--calibrate] Fit on all matches (default: neutral-site only)")
    ap.add_argument("--history-url", type=str, default=None,
                    help="[--calibrate] Override the results CSV URL")
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
    ap.add_argument("--min-ev", type=float, default=0.0,
                    help="[--arb] Min EV over consensus for an +EV-fallback bet (default 0.0)")
    ap.add_argument("--min-ask", type=float, default=0.02,
                    help="[--arb] Drop +EV-fallback bets priced below this ask — filters "
                         "near-zero longshot noise (default 0.02 = 2c)")
    ap.add_argument("--max-leg-stake", type=float, default=1000.0,
                    help="[--arb] Max $ stake per leg (manual depth cap; default 1000)")
    ap.add_argument("--settlement-date", type=str, default="2026-07-19",
                    help="[--arb] Settlement date YYYY-MM-DD for annualized ROC")
    ap.add_argument("--no-ev", action="store_true", help="[--arb] Disable the +EV fallback")
    ap.add_argument("--no-dutch", action="store_true", help="[--arb] Disable the Dutch-book check")
    return ap.parse_args()


def main():
    # Load ANTHROPIC_API_KEY (and any other vars) from a project-local .env if present.
    # Does NOT override an already-exported env var, so CI/containers still win.
    load_dotenv(Path(__file__).resolve().parent / ".env")
    args = parse_args()
    if args.bankroll is not None and args.bankroll <= 0:
        raise SystemExit("error: --bankroll must be positive")
    if args.min_edge < 0:
        raise SystemExit("error: --min-edge must be >= 0")
    if args.calibrate:
        if args.history_years <= 0:
            raise SystemExit("error: --history-years must be positive")
        since = date.today().year - args.history_years
        run_calibrate_cli(since_year=since, all_matches=args.all_matches,
                          history_url=args.history_url)
        return
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
                    min_ev=args.min_ev, min_ask=args.min_ask,
                    enable_ev=not args.no_ev, enable_dutch=not args.no_dutch)
        return
    min_edge = 0.0 if args.all_bets else args.min_edge
    asyncio.run(run_agent(args.bankroll, args.sims, args.seed,
                          args.kelly_fraction, min_edge,
                          use_calibrated=not args.no_calibrated))


if __name__ == "__main__":
    main()
