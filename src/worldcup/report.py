# src/worldcup/report.py
"""Render the forecast table and betting card to the terminal and markdown."""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

from worldcup.models import BetRec, Forecast

DISCLAIMER = ("Model-based estimate; prediction markets are highly efficient. "
              "For education only — never stake more than you can afford to lose.")


def build_markdown(forecasts: list[Forecast], bets: list[BetRec],
                   bankroll: float | None) -> str:
    """Build the markdown report string (forecast table + optional betting card)."""
    lines = ["# 2026 World Cup Forecast", "",
             "| Team | Model | Market | Blended |", "|------|------:|-------:|--------:|"]
    for f in sorted(forecasts, key=lambda x: x.blended_pct, reverse=True):
        lines.append(f"| {f.team} | {f.model_pct:.1%} | {f.market_pct:.1%} | {f.blended_pct:.1%} |")

    if bankroll is not None:
        lines += ["", "## Betting card", f"_Bankroll: ${bankroll:,.2f}_", ""]
        if not bets:
            lines.append("No value bets at current prices.")
        else:
            lines += ["| Team | Ask | Model | Market | Edge | EV | Stake | Profit |",
                      "|------|----:|------:|-------:|-----:|---:|------:|-------:|"]
            for b in bets:
                lines.append(
                    f"| {b.team} | {b.ask:.2f} | {b.model_pct:.1%} | {b.market_pct:.1%} | "
                    f"{b.edge:+.1%} | {b.ev_pct:+.1%} | ${b.stake:,.2f} | ${b.potential_profit:,.2f} |")
            staked = sum(b.stake for b in bets)
            lines += ["", f"_Total staked: ${staked:,.2f} · Reserve: ${bankroll - staked:,.2f}_"]
        lines += ["", f"> {DISCLAIMER}"]
    return "\n".join(lines) + "\n"


def print_report(forecasts: list[Forecast], bets: list[BetRec],
                 bankroll: float | None, console: Console | None = None) -> None:
    """Pretty-print the forecast and betting card to the terminal."""
    console = console or Console()
    ft = Table(title="2026 World Cup Forecast")
    for col in ("Team", "Model", "Market", "Blended"):
        ft.add_column(col, justify="right" if col != "Team" else "left")
    for f in sorted(forecasts, key=lambda x: x.blended_pct, reverse=True):
        ft.add_row(f.team, f"{f.model_pct:.1%}", f"{f.market_pct:.1%}", f"{f.blended_pct:.1%}")
    console.print(ft)

    if bankroll is None:
        return
    if not bets:
        console.print(f"\n[bold]No value bets[/bold] at current prices (bankroll ${bankroll:,.2f}).")
    else:
        bt = Table(title=f"Betting card (bankroll ${bankroll:,.2f})")
        for col in ("Team", "Ask", "Model", "Market", "Edge", "EV", "Stake", "Profit"):
            bt.add_column(col, justify="right" if col != "Team" else "left")
        for b in bets:
            bt.add_row(b.team, f"{b.ask:.2f}", f"{b.model_pct:.1%}", f"{b.market_pct:.1%}",
                       f"{b.edge:+.1%}", f"{b.ev_pct:+.1%}", f"${b.stake:,.2f}", f"${b.potential_profit:,.2f}")
        console.print(bt)
    console.print(f"\n[dim]{DISCLAIMER}[/dim]")
