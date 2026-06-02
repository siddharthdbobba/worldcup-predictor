"""Claude Agent SDK wiring + the deterministic pipeline the agent orchestrates.

`run_pipeline` is pure (no network, no LLM) so it is fully unit-testable. The SDK
tools are thin wrappers that fetch live data and then call the same pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

from worldcup.blend import blend
from worldcup.models import BetRec, Forecast
from worldcup.simulator import run_simulation
from worldcup.stake import recommend_bets


@dataclass
class PipelineResult:
    forecasts: list[Forecast]
    bets: list[BetRec]


def run_pipeline(ratings, groups, market, ask, confidence, depth=None, *, bankroll,
                 n_sims=20000, seed=42, w_cap=0.7, K=0.5,
                 kelly_fraction=0.5) -> PipelineResult:
    """Simulate, blend, and (if bankroll given) recommend bets.

    `ratings` may cover more teams than the draw; only the 48 drawn teams are
    simulated. `confidence` is the market's normalized confidence in [0, 2] (the
    blend weight; hence the K~0.5 default); `depth` is dollar order-book depth used
    only to cap stake sizes.
    """
    teams = {t for ts in groups.values() for t in ts}
    missing = sorted(t for t in teams if t not in ratings)
    if missing:
        raise ValueError(f"no Elo rating for drawn teams: {missing}")
    model = run_simulation({t: ratings[t] for t in teams}, groups, n=n_sims, seed=seed)
    blended = blend(model, market, confidence, w_cap=w_cap, K=K)
    forecasts = [Forecast(t, model[t], market.get(t, 0.0), blended[t]) for t in model]
    forecasts.sort(key=lambda f: f.blended_pct, reverse=True)
    bets: list[BetRec] = []
    if bankroll:
        bets = recommend_bets(model, market, ask, bankroll,
                              kelly_fraction=kelly_fraction, liquidity=depth)
    return PipelineResult(forecasts=forecasts, bets=bets)


# --- Claude Agent SDK integration -------------------------------------------
# Verified against installed claude-agent-sdk 0.2.87: tool, create_sdk_mcp_server,
# ClaudeAgentOptions (system_prompt/mcp_servers/allowed_tools), and query all match.
import json

from claude_agent_sdk import (  # type: ignore
    ClaudeAgentOptions, create_sdk_mcp_server, query, tool,
)

from worldcup.draw import fetch_group_draw
from worldcup.markets import fetch_market_probabilities
from worldcup.ratings import fetch_ratings
from worldcup.report import build_markdown, print_report

SYSTEM_PROMPT = """You are a World Cup forecasting agent. Run these steps in order
and narrate each: (1) fetch_group_draw — STOP if it is not 12 groups of 4;
(2) fetch_team_ratings; (3) fetch_market_probabilities; (4) run_forecast with the
user's bankroll. Never invent ratings or prices. If a step fails, report the error
plainly. Finish by presenting the forecast table and betting card."""


@tool("fetch_group_draw", "Fetch and validate the 2026 group draw (12x4)", {})
async def _t_draw(args):
    groups = fetch_group_draw()
    return {"content": [{"type": "text", "text": json.dumps(groups)}]}


@tool("fetch_team_ratings", "Fetch international Elo ratings for all teams", {})
async def _t_ratings(args):
    return {"content": [{"type": "text", "text": json.dumps(fetch_ratings())}]}


@tool("fetch_market_probabilities", "Fetch + de-vig Polymarket and Kalshi prices", {})
async def _t_markets(args):
    prob, ask, conf, depth = fetch_market_probabilities()
    return {"content": [{"type": "text", "text": json.dumps(
        {"prob": prob, "ask": ask, "confidence": conf, "depth": depth})}]}


@tool("run_forecast", "Simulate, blend, and recommend bets",
      {"bankroll": float, "n_sims": int, "seed": int, "kelly_fraction": float})
async def _t_forecast(args):
    groups = fetch_group_draw()
    ratings = fetch_ratings()
    valid = {t for ts in groups.values() for t in ts}
    prob, ask, conf, depth = fetch_market_probabilities(valid_teams=valid)
    result = run_pipeline(ratings, groups, prob, ask, conf, depth,
                          bankroll=args.get("bankroll"),
                          n_sims=int(args.get("n_sims", 20000)),
                          seed=int(args.get("seed", 42)),
                          kelly_fraction=float(args.get("kelly_fraction", 0.5)))
    print_report(result.forecasts, result.bets, args.get("bankroll"))
    md = build_markdown(result.forecasts, result.bets, args.get("bankroll"))
    return {"content": [{"type": "text", "text": md}]}


def build_options() -> ClaudeAgentOptions:
    server = create_sdk_mcp_server(
        name="worldcup", version="0.1.0",
        tools=[_t_draw, _t_ratings, _t_markets, _t_forecast])
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"worldcup": server},
        allowed_tools=[
            "mcp__worldcup__fetch_group_draw",
            "mcp__worldcup__fetch_team_ratings",
            "mcp__worldcup__fetch_market_probabilities",
            "mcp__worldcup__run_forecast",
        ],
    )


async def run_agent(bankroll: float | None, n_sims: int, seed: int,
                    kelly_fraction: float) -> None:
    prompt = (f"Forecast the 2026 World Cup. Bankroll: "
              f"{bankroll if bankroll else 'none (skip betting card)'}. "
              f"Use n_sims={n_sims}, seed={seed}, kelly_fraction={kelly_fraction}.")
    async for message in query(prompt=prompt, options=build_options()):
        print(message)
