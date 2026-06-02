"""Fetch + parse international team strength (Elo) ratings."""
from __future__ import annotations

import httpx

from worldcup.teamnames import canonical

RATINGS_URL = "https://example.com/elo.json"  # OPEN ITEM: set real source


def parse_ratings(payload: dict) -> dict[str, float]:
    """Parse a ratings payload into {canonical_team: elo}."""
    out: dict[str, float] = {}
    for row in payload.get("ratings", []):
        out[canonical(row["team"])] = float(row["elo"])
    return out


def fetch_ratings(url: str = RATINGS_URL, timeout: float = 20.0) -> dict[str, float]:
    """Fetch and parse live ratings. Raises on network/HTTP error (fail loud)."""
    resp = httpx.get(url, timeout=timeout)
    resp.raise_for_status()
    return parse_ratings(resp.json())
