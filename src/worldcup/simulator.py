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
