from worldcup.models import Forecast, BetRec
from worldcup.report import build_markdown

DISCLAIMER = "never stake more than you can afford to lose"


def test_markdown_has_forecast_rows_sorted_and_betting_card():
    forecasts = [
        Forecast("Spain", 0.15, 0.14, 0.145),
        Forecast("France", 0.12, 0.16, 0.14),
    ]
    bets = [BetRec("Spain", 0.16, 0.15, 0.14, 0.01, -0.0625, 20.0, 105.0)]
    md = build_markdown(forecasts, bets, bankroll=100.0)
    assert "Spain" in md and "France" in md
    assert "## Betting card" in md
    assert DISCLAIMER in md


def test_markdown_handles_no_value_bets():
    forecasts = [Forecast("Spain", 0.15, 0.14, 0.145)]
    md = build_markdown(forecasts, [], bankroll=100.0)
    assert "no value bets" in md.lower()


def test_markdown_omits_card_when_no_bankroll():
    md = build_markdown([Forecast("Spain", 0.15, 0.14, 0.145)], [], bankroll=None)
    assert "Betting card" not in md


from worldcup.arb import ArbResult
from worldcup.models import ArbRec, EvBetRec, DutchBook
from worldcup.report import build_arb_markdown


def test_build_arb_markdown_renders_locks_and_disclaimer():
    locks = [ArbRec(team="France", yes_venue="polymarket", no_venue="kalshi",
                    yes_ask=0.18, no_ask=0.71, contracts=100,
                    stake_yes=18.0, stake_no=71.0, total_cost=90.44,
                    guaranteed_profit=9.56, roc=0.1057, annual_roc=1.23)]
    ev = [EvBetRec(team="Spain", venue="polymarket", side="YES", ask=0.10,
                   fair=0.14, ev_pct=0.40, stake=20.0, potential_profit=180.0)]
    dutch = DutchBook(field_sum=1.03, gap=-0.03, is_arb=False, legs=[])
    res = ArbResult(locks=locks, ev_bets=ev, dutch=dutch)
    md = build_arb_markdown(res, poly_balance=10_000, kalshi_balance=10_000)
    assert "France" in md and "polymarket" in md and "kalshi" in md
    assert "$18.00" in md and "$71.00" in md            # per-leg stakes
    assert "9.56" in md                                  # guaranteed profit
    assert "Spain" in md                                 # EV section
    assert "never stake more than you can afford" in md.lower()


def test_build_arb_markdown_handles_no_locks():
    res = ArbResult(locks=[], ev_bets=[], dutch=None)
    md = build_arb_markdown(res, poly_balance=100, kalshi_balance=100)
    assert "no risk-free arbitrage" in md.lower()


import io
from rich.console import Console
from worldcup.report import print_arb_report


def test_print_arb_report_smoke():
    locks = [ArbRec(team="France", yes_venue="polymarket", no_venue="kalshi",
                    yes_ask=0.18, no_ask=0.71, contracts=100,
                    stake_yes=18.0, stake_no=71.0, total_cost=90.44,
                    guaranteed_profit=9.56, roc=0.1057, annual_roc=1.23)]
    ev = [EvBetRec(team="Spain", venue="polymarket", side="YES", ask=0.10,
                   fair=0.14, ev_pct=0.40, stake=20.0, potential_profit=180.0)]
    dutch = DutchBook(field_sum=1.03, gap=-0.03, is_arb=False, legs=[])
    res = ArbResult(locks=locks, ev_bets=ev, dutch=dutch)

    buf = io.StringIO()
    print_arb_report(res, poly_balance=10_000, kalshi_balance=10_000,
                     console=Console(file=buf, width=200))
    output = buf.getvalue()

    assert "France" in output
    assert "10,000" in output
