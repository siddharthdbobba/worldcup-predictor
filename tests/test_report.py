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
