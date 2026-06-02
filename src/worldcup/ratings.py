"""Fetch + parse international team strength (Elo) ratings from eloratings.net.

Two files are joined on eloratings' 2-letter code: `en.teams.tsv` (code -> name,
with extra alias columns) and `World.tsv` (rank, rank, code, rating, ...). The
result is `{canonical_team_name: elo}`. Parsing is kept separate from fetching so
it stays unit-testable offline.
"""
from __future__ import annotations

import httpx

from worldcup.teamnames import canonical

TEAMS_URL = "https://www.eloratings.net/en.teams.tsv"
WORLD_URL = "https://www.eloratings.net/World.tsv"

# Column indices in World.tsv rows: rank, rank, CODE, RATING, ...
_CODE_COL = 2
_RATING_COL = 3


def parse_team_codes(teams_tsv: str) -> dict[str, str]:
    """Map eloratings code -> primary English name (first name column)."""
    out: dict[str, str] = {}
    for line in teams_tsv.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip():
            out[parts[0]] = parts[1]
    return out


def parse_ratings(world_tsv: str, code_to_name: dict[str, str],
                  min_count: int = 100) -> dict[str, float]:
    """Parse World.tsv into {canonical_team_name: elo}, joined via code_to_name.

    Sanity-asserts the parse so a future column-layout change on eloratings fails
    loud instead of feeding garbage strengths into the simulator. `min_count` is
    lowered in unit tests that use tiny fixtures.
    """
    out: dict[str, float] = {}
    for line in world_tsv.splitlines():
        parts = line.split("\t")
        if len(parts) <= _RATING_COL:
            continue
        code = parts[_CODE_COL]
        if code not in code_to_name:
            continue
        try:
            rating = float(parts[_RATING_COL])
        except ValueError:
            continue
        out[canonical(code_to_name[code])] = rating
    if len(out) < min_count:
        raise ValueError(f"parsed only {len(out)} ratings; eloratings format may have changed")
    top = max(out.values())
    if not (1800.0 < top < 2400.0):
        raise ValueError(f"implausible top Elo {top}; eloratings column layout may have changed")
    return out


def fetch_ratings(teams_url: str = TEAMS_URL, world_url: str = WORLD_URL,
                  timeout: float = 20.0) -> dict[str, float]:
    """Fetch and parse live Elo ratings. Raises on network/HTTP/format error."""
    r_teams = httpx.get(teams_url, timeout=timeout, follow_redirects=True)
    r_teams.raise_for_status()
    r_world = httpx.get(world_url, timeout=timeout, follow_redirects=True)
    r_world.raise_for_status()
    return parse_ratings(r_world.text, parse_team_codes(r_teams.text))
