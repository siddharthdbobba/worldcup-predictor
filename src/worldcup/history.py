# src/worldcup/history.py
"""Fetch + parse historical international results (live), name-canonicalized.

Thin I/O mirroring `ratings.py`. Source: the public `martj42/international_results`
`results.csv`. Feeds the Elo replay (`elo_engine.replay`) → calibration backtest.
"""
from __future__ import annotations

import csv
import io

import httpx

from worldcup.elo_engine import Match
from worldcup.teamnames import canonical

RESULTS_URL = ("https://raw.githubusercontent.com/martj42/"
               "international_results/master/results.csv")


def _int_or_none(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_results(csv_text: str, *, since_year: int | None = None) -> list[Match]:
    """Parse results.csv text into canonicalized `Match` rows, sorted by date.

    Rows with a missing/blank score are dropped. `since_year` keeps matches whose
    year is >= the given year (string-comparison on the ISO date).
    """
    rows: list[Match] = []
    for r in csv.DictReader(io.StringIO(csv_text)):
        gh, ga = _int_or_none(r.get("home_score")), _int_or_none(r.get("away_score"))
        if gh is None or ga is None:
            continue
        date = (r.get("date") or "").strip()
        if since_year is not None and (len(date) < 4 or date[:4] < str(since_year)):
            continue
        rows.append(Match(
            date=date,
            home=canonical(r.get("home_team", "")),
            away=canonical(r.get("away_team", "")),
            goals_home=gh, goals_away=ga,
            neutral=(r.get("neutral", "").strip().lower() == "true"),
            tournament=(r.get("tournament") or "").strip(),
        ))
    rows.sort(key=lambda m: m.date)
    return rows


def fetch_results(url: str = RESULTS_URL, timeout: float = 30.0) -> str:
    """Fetch the raw results CSV text. Raises on network/HTTP error."""
    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def fetch_matches(*, since_year: int | None = None, url: str = RESULTS_URL,
                  timeout: float = 30.0) -> list[Match]:
    """Fetch + parse in one call (live)."""
    return parse_results(fetch_results(url=url, timeout=timeout), since_year=since_year)
