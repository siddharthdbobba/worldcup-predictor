"""Export the predictor's full pipeline output as one JSON snapshot for the
portfolio site to visualize. `build_snapshot` is pure (no I/O); the CLI below
fetches live data and writes the file."""
from __future__ import annotations

from worldcup.blend import market_weight
from worldcup.models import BetRec, Forecast, MatchModelParams


def build_snapshot(*, groups, ratings, market, ask, confidence, depth,
                   forecasts, advancement, bets, bankroll, n_sims, seed,
                   kelly_fraction, min_edge, generated_at,
                   poly_pct=None, kalshi_pct=None,
                   params: MatchModelParams | None = None) -> dict:
    """Assemble the snapshot dict. Only the 48 drawn teams are emitted; `ratings`
    may contain extras. `confidence` is the per-team market confidence (the blend
    weight input); `depth` is dollar order-book depth. `poly_pct`/`kalshi_pct` are
    each book's de-vigged win prob per team (None => team not priced on that book)."""
    p = params or MatchModelParams()
    poly_pct = poly_pct or {}
    kalshi_pct = kalshi_pct or {}
    drawn = sorted({t for ts in groups.values() for t in ts})
    staked = round(sum(b.stake for b in bets), 2) if bankroll else None
    reserve = round(bankroll - staked, 2) if bankroll else None
    return {
        "meta": {
            "generated_at": generated_at,
            "n_sims": n_sims, "seed": seed, "kelly_fraction": kelly_fraction,
            "min_edge": min_edge,
            "model_params": {"base": p.base, "scale": p.scale,
                             "host_bump": p.host_bump, "hosts": list(p.hosts)},
            "sources": ["Polymarket", "Kalshi", "eloratings.net"],
        },
        "draw": {label: list(teams) for label, teams in groups.items()},
        "ratings": {t: ratings[t] for t in drawn},
        "forecasts": [
            {"team": f.team, "model_pct": f.model_pct,
             "market_pct": f.market_pct, "blended_pct": f.blended_pct}
            for f in forecasts
        ],
        "market": {
            t: {"prob": market.get(t, 0.0), "ask": ask.get(t, 0.0),
                "confidence": confidence.get(t, 0.0), "depth": depth.get(t, 0.0),
                "polymarket": poly_pct.get(t), "kalshi": kalshi_pct.get(t)}
            for t in drawn
        },
        "blend_weights": {
            t: market_weight(confidence.get(t, 0.0)) for t in drawn if t in market
        },
        "advancement": {t: advancement[t] for t in drawn if t in advancement},
        "bankroll": bankroll,
        "bets": [
            {"team": b.team, "ask": b.ask, "model_pct": b.model_pct,
             "market_pct": b.market_pct, "edge": b.edge, "ev_pct": b.ev_pct,
             "stake": b.stake, "potential_profit": b.potential_profit}
            for b in bets
        ],
        "staked": staked,
        "reserve": reserve,
    }


# --- CLI: fetch live data, run the pipeline, write JSON -----------------------
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from worldcup.blend import blend
from worldcup.draw import fetch_group_draw, validate_draw
from worldcup.markets import combine_markets, devigged_per_book, fetch_books
from worldcup.models import Forecast
from worldcup.ratings import fetch_ratings
from worldcup.simulator import run_simulation_detailed
from worldcup.stake import recommend_bets

DEFAULT_OUT = "../portfolio/src/data/worldcup-snapshot.json"


def generate_snapshot(*, bankroll, n_sims, seed, kelly_fraction, min_edge) -> dict:
    """Fetch live draw/ratings/markets, run the detailed sim + pipeline, and build
    the snapshot dict. Raises loudly if markets are unavailable (no stale data)."""
    groups = fetch_group_draw()
    validate_draw(groups)
    ratings = fetch_ratings()
    valid = {t for ts in groups.values() for t in ts}
    missing = sorted(t for t in valid if t not in ratings)
    if missing:
        raise SystemExit(f"error: no Elo rating for drawn teams: {missing}")
    # One fetch of both books; derive the combined consensus AND each book's
    # de-vigged prob from the same pull (no re-fetch, no quote skew between them).
    poly, kalshi = fetch_books(valid_teams=valid)
    market, ask, confidence, depth = combine_markets(poly, kalshi, valid_teams=valid)
    if not market:
        raise SystemExit("error: market fetch returned no priced teams; aborting "
                         "(refusing to ship a snapshot without live prices)")
    poly_pct, kalshi_pct = devigged_per_book(poly, kalshi, valid_teams=valid)
    sub_ratings = {t: ratings[t] for t in valid}
    champ, advancement = run_simulation_detailed(sub_ratings, groups,
                                                 n=n_sims, seed=seed)
    blended = blend(champ, market, confidence, w_cap=0.7, K=0.5)
    forecasts = sorted(
        [Forecast(t, champ[t], market.get(t, 0.0), blended[t]) for t in champ],
        key=lambda f: f.blended_pct, reverse=True)
    bets = recommend_bets(champ, market, ask, bankroll,
                          kelly_fraction=kelly_fraction, liquidity=depth,
                          min_edge=min_edge) if bankroll else []
    return build_snapshot(
        groups=groups, ratings=ratings, market=market, ask=ask,
        confidence=confidence, depth=depth, poly_pct=poly_pct, kalshi_pct=kalshi_pct,
        forecasts=forecasts, advancement=advancement, bets=bets, bankroll=bankroll,
        n_sims=n_sims, seed=seed, kelly_fraction=kelly_fraction, min_edge=min_edge,
        generated_at=datetime.now(timezone.utc).isoformat())


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a World Cup predictor snapshot")
    ap.add_argument("--bankroll", type=float, default=100.0,
                    help="Bankroll in dollars; pass 0 for a forecast-only snapshot")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kelly-fraction", type=float, default=0.5)
    ap.add_argument("--min-edge", type=float, default=0.05)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT,
                    help=f"Output path (default {DEFAULT_OUT})")
    args = ap.parse_args()
    bankroll = args.bankroll if args.bankroll and args.bankroll > 0 else None
    snap = generate_snapshot(bankroll=bankroll, n_sims=args.sims, seed=args.seed,
                             kelly_fraction=args.kelly_fraction,
                             min_edge=args.min_edge)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, indent=2) + "\n")
    n_bets = len(snap["bets"])
    print(f"wrote {out} — {len(snap['forecasts'])} teams, {n_bets} value bets, "
          f"generated_at {snap['meta']['generated_at']}")


if __name__ == "__main__":
    main()
