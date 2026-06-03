"""Export the predictor's full pipeline output as one JSON snapshot for the
portfolio site to visualize. `build_snapshot` is pure (no I/O); the CLI below
fetches live data and writes the file."""
from __future__ import annotations

from worldcup.blend import market_weight
from worldcup.models import BetRec, Forecast, MatchModelParams


def build_snapshot(*, groups, ratings, market, ask, confidence, depth,
                   forecasts, advancement, bets, bankroll, n_sims, seed,
                   kelly_fraction, min_edge, generated_at,
                   params: MatchModelParams | None = None) -> dict:
    """Assemble the snapshot dict. Only the 48 drawn teams are emitted; `ratings`
    may contain extras. `confidence` is the per-team market confidence (the blend
    weight input); `depth` is dollar order-book depth."""
    p = params or MatchModelParams()
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
                "confidence": confidence.get(t, 0.0), "depth": depth.get(t, 0.0)}
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
