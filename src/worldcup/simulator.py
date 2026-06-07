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
    # Tie -> extra time/penalties: resolve with a strength-weighted flip whose bias
    # comes from the SAME goals model (`scale`) as regulation — A's share of expected
    # goals, lam_a / (lam_a + lam_b). Using the standard Elo /400 here instead (as a
    # prior version did) would snap drawn knockouts back to a much steeper favorite
    # bias than the calibrated `scale` implies, inflating favorites' title odds — the
    # exact distortion the 600->2000 scale recalibration removed. Host bump is applied
    # via _eff_rating so hosts keep their edge in shootouts too.
    lam_a, lam_b = expected_goals(_eff_rating(a, ratings, p),
                                  _eff_rating(b, ratings, p), p)
    return a if rng.random() < lam_a / (lam_a + lam_b) else b


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


DEFAULT_SIMS = 20000


def simulate_once(ratings: dict[str, float], groups: dict[str, list[str]],
                  rng: np.random.Generator, p: MatchModelParams) -> str:
    """Simulate one full tournament; return the champion's name."""
    qualifiers: list[tuple[str, dict, int]] = []   # (team, stats, tier 1/2/3)
    thirds: list[tuple[str, dict]] = []
    for teams in groups.values():
        ranked, stats = simulate_group(teams, ratings, rng, p)
        qualifiers.append((ranked[0], stats[ranked[0]], 1))
        qualifiers.append((ranked[1], stats[ranked[1]], 2))
        thirds.append((ranked[2], stats[ranked[2]]))
    best_thirds = set(rank_thirds(thirds, rng))
    for name, stats in thirds:
        if name in best_thirds:
            qualifiers.append((name, stats, 3))
    # Seed 1..32: tier first (winners, runners-up, thirds), then pts/GD/GF.
    seeded = [q[0] for q in sorted(
        qualifiers, key=lambda q: (q[2], *_rank_key(q[1], rng)))]
    return play_knockout(seeded, ratings, rng, p)


def run_simulation(ratings: dict[str, float], groups: dict[str, list[str]],
                   n: int = DEFAULT_SIMS, seed: int = 42,
                   params: MatchModelParams | None = None) -> dict[str, float]:
    """Run n tournaments; return {team: championship_probability} over ALL teams."""
    p = params or MatchModelParams()
    rng = np.random.default_rng(seed)
    counts: Counter[str] = Counter()
    for _ in range(n):
        counts[simulate_once(ratings, groups, rng, p)] += 1
    return {team: counts.get(team, 0) / n for team in ratings}
