"""Canonical team-name normalization so ratings/markets/draw align.

Extend _ALIASES as new source spellings are encountered during implementation.
"""

_ALIASES = {
    "usa": "United States",
    "us": "United States",
    "united states": "United States",
    "united states of america": "United States",
    "korea republic": "South Korea",
    "korea, republic of": "South Korea",
    "republic of korea": "South Korea",
    "south korea": "South Korea",
    "ir iran": "Iran",
    "iran": "Iran",
    "türkiye": "Turkey",
    "turkiye": "Turkey",
    "turkey": "Turkey",
    "côte d'ivoire": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    # canonical target spellings follow eloratings.net (the ratings join key)
    "congo dr": "DR Congo",
    "dr congo": "DR Congo",
    "democratic republic of the congo": "DR Congo",
    "curacao": "Curaçao",
    "curaçao": "Curaçao",
    "czech republic": "Czechia",
    "czechia": "Czechia",
    "cabo verde": "Cape Verde",
    "cape verde": "Cape Verde",
}


def canonical(name: str) -> str:
    """Return the canonical spelling of a team name (trimmed passthrough if unknown)."""
    key = name.strip().lower()
    return _ALIASES.get(key, name.strip())
