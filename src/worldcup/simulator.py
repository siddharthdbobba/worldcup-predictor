# src/worldcup/simulator.py
"""Pure, deterministic Monte Carlo simulation of the 2026 World Cup.

No I/O. All randomness flows through an injected numpy Generator so runs are
reproducible. Two stages share one goals-based match model; the knockout stage
additionally resolves draws via a strength-weighted coin flip.
"""
from __future__ import annotations

import itertools
from collections import Counter

import numpy as np

from worldcup.models import MatchModelParams


def expected_goals(r_a: float, r_b: float, p: MatchModelParams) -> tuple[float, float]:
    """Expected goals for each side from the Elo difference."""
    lam_a = p.base * 10 ** ((r_a - r_b) / p.scale)
    lam_b = p.base * 10 ** ((r_b - r_a) / p.scale)
    return lam_a, lam_b


def _eff_rating(team: str, ratings: dict[str, float], p: MatchModelParams) -> float:
    return ratings[team] + (p.host_bump if team in p.hosts else 0.0)


def sample_goals(a: str, b: str, ratings: dict[str, float],
                 rng: np.random.Generator, p: MatchModelParams) -> tuple[int, int]:
    """Sample a single scoreline (Poisson goals), applying the host bump."""
    lam_a, lam_b = expected_goals(_eff_rating(a, ratings, p), _eff_rating(b, ratings, p), p)
    return int(rng.poisson(lam_a)), int(rng.poisson(lam_b))


def elo_winprob(r_a: float, r_b: float) -> float:
    """Standard Elo win probability for A over B."""
    return 1.0 / (1.0 + 10 ** (-(r_a - r_b) / 400.0))


def _rank_key(s: dict, rng: np.random.Generator) -> tuple:
    """Sort key (ascending => best first): points, goal diff, goals for, random."""
    return (-s["pts"], -(s["gf"] - s["ga"]), -s["gf"], rng.random())


def simulate_group(teams: list[str], ratings: dict[str, float],
                   rng: np.random.Generator, p: MatchModelParams):
    """Round-robin a 4-team group; return (ranked_team_names, stats_by_team).

    Tiebreakers (v1, simplified): points -> goal difference -> goals for ->
    random draw. Head-to-head mini-tables are deferred to v2.
    """
    stats = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}
    for a, b in itertools.combinations(teams, 2):
        ga, gb = sample_goals(a, b, ratings, rng, p)
        stats[a]["gf"] += ga; stats[a]["ga"] += gb
        stats[b]["gf"] += gb; stats[b]["ga"] += ga
        if ga > gb:
            stats[a]["pts"] += 3
        elif gb > ga:
            stats[b]["pts"] += 3
        else:
            stats[a]["pts"] += 1; stats[b]["pts"] += 1
    ranked = sorted(teams, key=lambda t: _rank_key(stats[t], rng))
    return ranked, stats


def rank_thirds(thirds: list[tuple[str, dict]], rng: np.random.Generator) -> list[str]:
    """Rank all third-placed teams and return the best 8 (2026 format)."""
    ordered = sorted(thirds, key=lambda x: _rank_key(x[1], rng))
    return [name for name, _ in ordered[:8]]


def bracket_seed_order(n: int) -> list[int]:
    """Standard single-elimination seed order for a bracket of size n (power of 2).

    n=4 -> [1, 4, 2, 3]. Adjacent pairs are first-round matchups.
    """
    order = [1, 2]
    while len(order) < n:
        m = len(order) * 2 + 1
        nxt = []
        for s in order:
            nxt.append(s)
            nxt.append(m - s)
        order = nxt
    return order


def play_match_ko(a: str, b: str, ratings: dict[str, float],
                  rng: np.random.Generator, p: MatchModelParams) -> str:
    """Knockout match: scoreline, then resolve any tie via strength-weighted flip."""
    ga, gb = sample_goals(a, b, ratings, rng, p)
    if ga > gb:
        return a
    if gb > ga:
        return b
    return a if rng.random() < elo_winprob(ratings[a], ratings[b]) else b


def play_knockout(seeded_teams: list[str], ratings: dict[str, float],
                  rng: np.random.Generator, p: MatchModelParams) -> str:
    """Play a fixed seeded bracket to a single champion.

    seeded_teams[i] is seed i+1 (index 0 = top seed).
    """
    order = bracket_seed_order(len(seeded_teams))
    bracket = [seeded_teams[s - 1] for s in order]
    while len(bracket) > 1:
        bracket = [play_match_ko(bracket[i], bracket[i + 1], ratings, rng, p)
                   for i in range(0, len(bracket), 2)]
    return bracket[0]
