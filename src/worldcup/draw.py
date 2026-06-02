"""Source + validate the 2026 group draw (12 groups of 4).

The draw is derived from Kalshi's group-winner markets (`KXWCGROUPWIN`): each
market's `event_ticker` ends in the group letter and its `yes_sub_title` is a team
in that group. Those markets close at kickoff, so the draw — a fixed fact — is also
committed as `data/draw_2026.json`. `fetch_group_draw` tries live first (refreshing
the committed copy on success) and falls back to the committed copy if the live
source is unavailable.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import httpx

from worldcup.teamnames import canonical

KALSHI_GROUPS_URL = ("https://api.elections.kalshi.com/trade-api/v2/markets"
                     "?series_ticker=KXWCGROUPWIN&limit=200&status=open")
FIXTURE_PATH = Path(__file__).parent / "data" / "draw_2026.json"


def parse_draw(payload: dict) -> dict[str, list[str]]:
    """Parse either the committed `{"groups": {...}}` form or a Kalshi markets
    payload (`{"markets": [...]}`) into {group_label: [4 canonical team names]}."""
    if "groups" in payload:
        return {label: [canonical(t) for t in teams]
                for label, teams in payload["groups"].items()}
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for m in payload.get("markets", []):
        event = m.get("event_ticker")
        team = m.get("yes_sub_title")
        if event and team:
            groups[event[-1]].append(canonical(team))  # ...-26A -> "A"
    return {label: groups[label] for label in sorted(groups)}


def validate_draw(groups: dict[str, list[str]]) -> None:
    """Fail loudly unless the draw is exactly 12 groups of 4, no blanks/dupes."""
    if len(groups) != 12:
        raise ValueError(f"expected 12 groups, got {len(groups)}")
    for label, teams in groups.items():
        if len(teams) != 4:
            raise ValueError(f"group {label} has {len(teams)} teams, expected 4")
        for t in teams:
            if not isinstance(t, str) or not t.strip():
                raise ValueError(f"group {label} has an unresolved/blank slot")
    all_teams = [t for ts in groups.values() for t in ts]
    if len(all_teams) != len(set(all_teams)):
        raise ValueError("draw contains duplicate team assignments")


def _save_fixture(groups: dict[str, list[str]]) -> None:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps({"groups": groups}, indent=2) + "\n")


def fetch_group_draw(url: str = KALSHI_GROUPS_URL, timeout: float = 20.0
                     ) -> dict[str, list[str]]:
    """Return the validated 12x4 draw: live if available (refreshing the committed
    copy), else the committed fixture. Raises only if both paths fail."""
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        groups = parse_draw(resp.json())
        validate_draw(groups)
        _save_fixture(groups)
        return groups
    except Exception as live_err:  # noqa: BLE001 - fall back to committed draw
        if FIXTURE_PATH.exists():
            groups = parse_draw(json.loads(FIXTURE_PATH.read_text()))
            validate_draw(groups)
            return groups
        raise RuntimeError(
            f"live draw failed ({live_err}) and no committed fixture at {FIXTURE_PATH}"
        ) from live_err
