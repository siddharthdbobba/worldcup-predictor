"""Fetch + parse + validate the 2026 group draw (12 groups of 4)."""
from __future__ import annotations

import httpx

from worldcup.teamnames import canonical

DRAW_URL = "https://example.com/wc2026-draw.json"  # OPEN ITEM: set real source


def parse_draw(payload: dict) -> dict[str, list[str]]:
    """Parse a draw payload into {group_label: [4 canonical team names]}."""
    return {label: [canonical(t) for t in teams]
            for label, teams in payload.get("groups", {}).items()}


def validate_draw(groups: dict[str, list[str]]) -> None:
    """Fail loudly unless the draw is exactly 12 groups of 4 with no blanks."""
    if len(groups) != 12:
        raise ValueError(f"expected 12 groups, got {len(groups)}")
    for label, teams in groups.items():
        if len(teams) != 4:
            raise ValueError(f"group {label} has {len(teams)} teams, expected 4")
        if any(not t.strip() for t in teams):
            raise ValueError(f"group {label} has an unresolved/blank slot")


def fetch_group_draw(url: str = DRAW_URL, timeout: float = 20.0) -> dict[str, list[str]]:
    """Fetch, parse, and validate the live draw. Raises on any problem (fail loud)."""
    resp = httpx.get(url, timeout=timeout)
    resp.raise_for_status()
    groups = parse_draw(resp.json())
    validate_draw(groups)
    return groups
