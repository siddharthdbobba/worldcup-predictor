# tests/test_history.py
from pathlib import Path

from worldcup.elo_engine import Match
from worldcup.history import parse_results

FIX = Path(__file__).parent / "fixtures"


def _text():
    return (FIX / "results_sample.csv").read_text()


def test_parse_results_drops_blank_scores_and_canonicalizes():
    rows = parse_results(_text())
    assert len(rows) == 3 and all(isinstance(r, Match) for r in rows)  # blank-score row dropped
    kr = [r for r in rows if r.away == "South Korea"]                  # "Korea Republic" canonicalized
    assert len(kr) == 1 and kr[0].home == "Brazil"


def test_parse_results_neutral_and_scores():
    rows = parse_results(_text())
    final = next(r for r in rows if r.home == "Argentina")
    assert final.neutral is True and (final.goals_home, final.goals_away) == (3, 3)
    opener = next(r for r in rows if r.home == "Russia")
    assert opener.neutral is False and opener.tournament == "FIFA World Cup"


def test_parse_results_since_year_filter():
    rows = parse_results(_text(), since_year=2020)
    assert len(rows) == 1 and rows[0].home == "Argentina"


def test_parse_results_sorted_by_date():
    rows = parse_results(_text())
    assert [r.date for r in rows] == sorted(r.date for r in rows)
