# src/worldcup/report.py
"""Render the forecast table and betting card to the terminal and markdown."""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

from worldcup.arb import ArbResult
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


ARB_DISCLAIMER = (
    "Risk-free only if BOTH legs fill at the quoted prices — place them near-"
    "simultaneously (leg-timing risk). Capital is locked on both venues until the "
    "tournament settles. Depth caps are user-set, not measured. For education only — "
    "never stake more than you can afford to lose.")


def build_arb_markdown(res: ArbResult, poly_balance: float,
                       kalshi_balance: float) -> str:
    """Markdown for the arb card: locks, EV fallback, Dutch-book line, disclaimer."""
    lines = ["# Method A — Cross-Book Arbitrage", "",
             f"_Polymarket balance: ${poly_balance:,.2f} · "
             f"Kalshi balance: ${kalshi_balance:,.2f}_", ""]

    lines += ["## Risk-free locks", ""]
    if not res.locks:
        lines.append("No risk-free arbitrage at current prices.")
    else:
        lines += ["| Team | YES @ | NO @ | Contracts | Stake YES | Stake NO | Cost | Profit | ROC | Annual |",
                  "|------|-------|------|----------:|----------:|---------:|-----:|-------:|----:|-------:|"]
        for r in res.locks:
            lines.append(
                f"| {r.team} | {r.yes_venue} {r.yes_ask:.2f} | {r.no_venue} {r.no_ask:.2f} | "
                f"{r.contracts} | ${r.stake_yes:,.2f} | ${r.stake_no:,.2f} | ${r.total_cost:,.2f} | "
                f"${r.guaranteed_profit:,.2f} | {r.roc:.2%} | {r.annual_roc:.1%} |")
        tot = sum(r.guaranteed_profit for r in res.locks)
        lines += ["", f"_Total guaranteed profit: ${tot:,.2f}_",
                  "",
                  "> **Funding note:** Place the listed number of contracts on each leg"
                  " — fund by contract count, not by the per-leg $ stakes"
                  " (those exclude trading fees, which are included in Cost)."
                  " Equal contracts on both legs is what makes the payout identical either way."]

    if res.ev_bets:
        lines += ["", "## +EV cross-book bets (NOT risk-free)", "",
                  "| Team | Venue | Side | Ask | Fair | EV | Stake | Profit |",
                  "|------|-------|------|----:|-----:|---:|------:|-------:|"]
        for b in res.ev_bets:
            lines.append(
                f"| {b.team} | {b.venue} | {b.side} | {b.ask:.2f} | {b.fair:.1%} | "
                f"{b.ev_pct:+.1%} | ${b.stake:,.2f} | ${b.potential_profit:,.2f} |")

    if res.dutch is not None:
        d = res.dutch
        verdict = f"ARB (gap {d.gap:+.3f})" if d.is_arb else f"no arb (gap {d.gap:+.3f})"
        lines += ["", "## Whole-field Dutch book",
                  f"Cheapest-YES field sum: {d.field_sum:.3f} — {verdict}."]

    lines += ["", f"> {ARB_DISCLAIMER}"]
    return "\n".join(lines) + "\n"


def print_arb_report(res: ArbResult, poly_balance: float, kalshi_balance: float,
                     console: Console | None = None) -> None:
    """Pretty-print the arb card to the terminal."""
    console = console or Console()
    if not res.locks:
        console.print(f"[bold]No risk-free arbitrage[/bold] at current prices "
                      f"(Poly ${poly_balance:,.2f} · Kalshi ${kalshi_balance:,.2f}).")
    else:
        t = Table(title=f"Method A — risk-free locks "
                        f"(Poly ${poly_balance:,.2f} · Kalshi ${kalshi_balance:,.2f})")
        for col in ("Team", "YES @", "NO @", "Contracts", "Stake YES",
                    "Stake NO", "Cost", "Profit", "ROC", "Annual"):
            t.add_column(col, justify="left" if col == "Team" else "right")
        for r in res.locks:
            t.add_row(r.team, f"{r.yes_venue} {r.yes_ask:.2f}",
                      f"{r.no_venue} {r.no_ask:.2f}", str(r.contracts),
                      f"${r.stake_yes:,.2f}", f"${r.stake_no:,.2f}",
                      f"${r.total_cost:,.2f}", f"${r.guaranteed_profit:,.2f}",
                      f"{r.roc:.2%}", f"{r.annual_roc:.1%}")
        console.print(t)
        console.print(
            "\n[bold]Funding note:[/bold] Place the listed number of contracts on each leg"
            " — fund by contract count, not by the per-leg $ stakes"
            " (those exclude trading fees, which are included in Cost)."
            " Equal contracts on both legs is what makes the payout identical either way.")
    if res.ev_bets:
        et = Table(title="+EV cross-book bets (NOT risk-free)")
        for col in ("Team", "Venue", "Side", "Ask", "Fair", "EV", "Stake", "Profit"):
            et.add_column(col, justify="left" if col == "Team" else "right")
        for b in res.ev_bets:
            et.add_row(b.team, b.venue, b.side, f"{b.ask:.2f}", f"{b.fair:.1%}",
                       f"{b.ev_pct:+.1%}", f"${b.stake:,.2f}", f"${b.potential_profit:,.2f}")
        console.print(et)
    if res.dutch is not None:
        d = res.dutch
        verdict = f"ARB (gap {d.gap:+.3f})" if d.is_arb else f"no arb (gap {d.gap:+.3f})"
        console.print(f"[dim]Dutch-book field sum {d.field_sum:.3f} — {verdict}[/dim]")
    console.print(f"\n[dim]{ARB_DISCLAIMER}[/dim]")
