# src/worldcup/elo_engine.py
"""Chronological Elo replay to recover each match's PRE-match Elo.

eloratings.net publishes only *current* ratings, so to calibrate the goals model
against history we reconstruct ratings ourselves by replaying past results through
the **World Football Elo** update rule — the same algorithm eloratings.net uses.
Matching their formula (1/400 logistic, +100 home advantage, goal-difference
multiplier, importance-weighted K) keeps the replayed ratings on the same scale as
the live ratings the forecaster consumes, so a `scale` fitted here transfers.

Pure and deterministic: no I/O, no randomness.
"""
from __future__ import annotations

from dataclasses import dataclass

INIT_RATING = 1500.0   # provisional rating for a team's first appearance
HOME_ADV = 100.0       # Elo points added to the home side when not on neutral ground


@dataclass(frozen=True)
class Match:
    """A historical result (team names already canonicalized)."""
    date: str            # ISO yyyy-mm-dd (string-sortable)
    home: str
    away: str
    goals_home: int
    goals_away: int
    neutral: bool
    tournament: str = ""


@dataclass(frozen=True)
class MatchElo:
    """A match annotated with the two sides' PRE-match Elo (for calibration)."""
    elo_home: float
    elo_away: float
    goals_home: int
    goals_away: int
    neutral: bool


def _k_factor(tournament: str) -> float:
    """Importance weight K by competition (eloratings-style tiers)."""
    t = tournament.lower()
    if "qualif" in t:
        return 40.0
    if "world cup" in t:
        return 60.0
    if any(s in t for s in ("euro", "copa am", "african cup", "afcon", "asian cup",
                            "gold cup", "confederations", "nations league")):
        return 50.0
    if "friendly" in t:
        return 20.0
    return 30.0


def _gd_multiplier(goal_diff: int) -> float:
    """Goal-difference index: bigger wins move ratings more."""
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0


def replay(matches: list[Match], *, init: float = INIT_RATING) -> list[MatchElo]:
    """Replay matches in date order; return each annotated with PRE-match Elo.

    Ratings are zero-sum per match (the winner gains exactly what the loser loses).
    Teams seen for the first time enter at `init`.
    """
    ratings: dict[str, float] = {}
    out: list[MatchElo] = []
    for m in sorted(matches, key=lambda x: x.date):
        ea = ratings.get(m.home, init)
        eb = ratings.get(m.away, init)
        out.append(MatchElo(elo_home=ea, elo_away=eb, goals_home=m.goals_home,
                            goals_away=m.goals_away, neutral=m.neutral))
        dr = ea - eb + (0.0 if m.neutral else HOME_ADV)
        we = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))          # expected score for home
        if m.goals_home > m.goals_away:
            w = 1.0
        elif m.goals_home < m.goals_away:
            w = 0.0
        else:
            w = 0.5
        delta = _k_factor(m.tournament) * _gd_multiplier(m.goals_home - m.goals_away) * (w - we)
        ratings[m.home] = ea + delta
        ratings[m.away] = eb - delta
    return out
