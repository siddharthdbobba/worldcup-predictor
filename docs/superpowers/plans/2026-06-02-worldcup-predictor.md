# World Cup Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Claude Agent SDK (Python) app that forecasts each nation's 2026 World Cup title probability by blending a Monte Carlo tournament simulation with prediction-market prices, then recommends Kelly-sized value bets from a user bankroll.

**Architecture:** The agent (Claude Agent SDK) orchestrates data fetching and report composition; pure, deterministic, seed-controlled Python modules do all numerical work (simulation, blending, staking) and are exposed to the agent as in-process MCP tools. Pure modules are unit-tested offline; I/O modules are tested against saved fixtures.

**Tech Stack:** Python 3.12+, `claude-agent-sdk`, `numpy`, `scipy`, `httpx`, `rich`, `pytest`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `pyproject.toml` | Package metadata + dependencies |
| `src/worldcup/models.py` | Pure dataclasses: `Team`, `MatchModelParams`, `Forecast`, `BetRec` |
| `src/worldcup/teamnames.py` | Canonical team-name normalization |
| `src/worldcup/simulator.py` | Pure Monte Carlo: match model → group stage → knockout → champion %s |
| `src/worldcup/blend.py` | Pure liquidity-weighted blend of model% + market% |
| `src/worldcup/stake.py` | Pure value detection + fractional-Kelly bankroll allocation |
| `src/worldcup/ratings.py` | I/O: fetch + parse team Elo ratings |
| `src/worldcup/markets.py` | I/O: fetch + parse Polymarket + Kalshi prices/ask/liquidity |
| `src/worldcup/draw.py` | I/O: fetch + parse + validate the 12×4 group draw |
| `src/worldcup/report.py` | Render rich terminal tables + markdown report |
| `src/worldcup/agent.py` | Agent SDK: wrap the above as tools, system prompt, run loop |
| `main.py` | CLI entrypoint (`--bankroll`, `--sims`, `--seed`, `--kelly-fraction`) |
| `tests/*` | pytest suite; `tests/fixtures/*.json` saved API responses |

**Build order:** scaffolding → pure core (models, teamnames, simulator, blend, stake) → I/O (ratings, markets, draw) → report → agent → CLI → README. Pure-before-I/O so most logic is testable without network.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `src/worldcup/__init__.py`, `tests/__init__.py`, `tests/fixtures/.gitkeep`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "worldcup-predictor"
version = "0.1.0"
description = "Monte Carlo + prediction-market ensemble forecaster for the 2026 World Cup"
requires-python = ">=3.12"
dependencies = [
    "claude-agent-sdk>=0.1.0",
    "numpy>=2.0",
    "scipy>=1.13",
    "httpx>=0.27",
    "rich>=13.7",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
.env
*.egg-info/
.pytest_cache/
reports/
```

- [ ] **Step 3: Create `.env.example`**

```dotenv
# Anthropic API key for the Claude Agent SDK
ANTHROPIC_API_KEY=sk-ant-...
```

- [ ] **Step 4: Create empty package files**

Create `src/worldcup/__init__.py` (empty), `tests/__init__.py` (empty), and an empty `tests/fixtures/.gitkeep`.

- [ ] **Step 5: Create venv and install**

Run:
```bash
cd ~/projects/worldcup-predictor
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```
Expected: installs without error. If `claude-agent-sdk` version pin fails, run `.venv/bin/pip index versions claude-agent-sdk` and pin to the latest available.

- [ ] **Step 6: Verify pytest runs (no tests yet)**

Run: `.venv/bin/pytest -q`
Expected: "no tests ran" (exit 0 or 5), no import errors.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: scaffold worldcup-predictor package and tooling"
```

---

## Task 2: Data models (`models.py`)

**Files:**
- Create: `src/worldcup/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from worldcup.models import Team, MatchModelParams, Forecast, BetRec


def test_team_holds_name_and_rating():
    t = Team(name="France", rating=2100.0)
    assert t.name == "France" and t.rating == 2100.0


def test_default_match_params_have_hosts():
    p = MatchModelParams()
    assert set(p.hosts) == {"United States", "Canada", "Mexico"}
    assert p.base > 0 and p.scale > 0


def test_forecast_and_betrec_fields():
    f = Forecast(team="Spain", model_pct=0.15, market_pct=0.14, blended_pct=0.145)
    assert f.blended_pct == 0.145
    b = BetRec(team="Spain", ask=0.16, model_pct=0.15, market_pct=0.14,
               edge=0.01, ev_pct=-0.0625, stake=10.0, potential_profit=52.5)
    assert b.stake == 10.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'worldcup.models'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/worldcup/models.py
from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    name: str          # canonical name
    rating: float      # international Elo


@dataclass(frozen=True)
class MatchModelParams:
    base: float = 1.35              # baseline expected goals per team
    scale: float = 600.0           # Elo scale for goal expectation
    host_bump: float = 60.0        # Elo added to a host in its own match
    hosts: tuple = ("United States", "Canada", "Mexico")


@dataclass
class Forecast:
    team: str
    model_pct: float
    market_pct: float
    blended_pct: float


@dataclass
class BetRec:
    team: str
    ask: float
    model_pct: float
    market_pct: float
    edge: float
    ev_pct: float
    stake: float
    potential_profit: float
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_models.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/models.py tests/test_models.py
git commit -m "feat: add core data models"
```

---

## Task 3: Team-name normalization (`teamnames.py`)

**Files:**
- Create: `src/worldcup/teamnames.py`
- Test: `tests/test_teamnames.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_teamnames.py
from worldcup.teamnames import canonical


def test_known_aliases_map():
    assert canonical("USA") == "United States"
    assert canonical("United States") == "United States"
    assert canonical("Korea Republic") == "South Korea"
    assert canonical("south korea") == "South Korea"
    assert canonical("IR Iran") == "Iran"


def test_unknown_name_is_trimmed_passthrough():
    assert canonical("  France ") == "France"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_teamnames.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

```python
# src/worldcup/teamnames.py
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
}


def canonical(name: str) -> str:
    """Return the canonical spelling of a team name (trimmed passthrough if unknown)."""
    key = name.strip().lower()
    return _ALIASES.get(key, name.strip())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_teamnames.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/teamnames.py tests/test_teamnames.py
git commit -m "feat: add team-name normalization"
```

---

## Task 4: Match model (`simulator.py` part 1)

**Files:**
- Create: `src/worldcup/simulator.py`
- Test: `tests/test_simulator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_simulator.py
import numpy as np
from worldcup.models import MatchModelParams
from worldcup.simulator import expected_goals, sample_goals, elo_winprob


def test_equal_ratings_give_equal_expected_goals():
    p = MatchModelParams()
    la, lb = expected_goals(2000, 2000, p)
    assert la == lb == p.base


def test_stronger_team_has_higher_expected_goals():
    p = MatchModelParams()
    la, lb = expected_goals(2200, 1800, p)
    assert la > lb


def test_sample_goals_is_deterministic_with_seed():
    p = MatchModelParams()
    ratings = {"A": 2000.0, "B": 1900.0}
    g1 = sample_goals("A", "B", ratings, np.random.default_rng(7), p)
    g2 = sample_goals("A", "B", ratings, np.random.default_rng(7), p)
    assert g1 == g2
    assert all(isinstance(x, int) for x in g1)


def test_elo_winprob_symmetry():
    assert abs(elo_winprob(2000, 2000) - 0.5) < 1e-9
    assert elo_winprob(2200, 1800) > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_simulator.py -v`
Expected: FAIL — cannot import `expected_goals`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_simulator.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/simulator.py tests/test_simulator.py
git commit -m "feat: add Elo-based match model"
```

---

## Task 5: Group stage + tiebreakers + best thirds (`simulator.py` part 2)

**Files:**
- Modify: `src/worldcup/simulator.py`
- Test: `tests/test_groups.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_groups.py
import numpy as np
from worldcup.models import MatchModelParams
from worldcup.simulator import simulate_group, rank_thirds


def test_simulate_group_returns_four_ranked_and_stats():
    p = MatchModelParams()
    teams = ["A", "B", "C", "D"]
    ratings = {"A": 2200.0, "B": 1900.0, "C": 1850.0, "D": 1700.0}
    ranked, stats = simulate_group(teams, ratings, np.random.default_rng(1), p)
    assert len(ranked) == 4
    assert set(ranked) == set(teams)
    for t in teams:
        assert {"pts", "gf", "ga"} <= set(stats[t])


def test_group_ranking_orders_by_points_then_gd_then_gf():
    # Hand-built stats; verify the sort key via the helper.
    from worldcup.simulator import _rank_key
    stats = {
        "A": {"pts": 6, "gf": 4, "ga": 1},
        "B": {"pts": 6, "gf": 5, "ga": 1},   # same pts, better GD
        "C": {"pts": 3, "gf": 9, "ga": 2},
        "D": {"pts": 0, "gf": 0, "ga": 9},
    }
    rng = np.random.default_rng(0)
    order = sorted(stats, key=lambda t: _rank_key(stats[t], rng))
    assert order == ["B", "A", "C", "D"]


def test_rank_thirds_takes_best_eight():
    rng = np.random.default_rng(0)
    thirds = [(f"T{i}", {"pts": i, "gf": i, "ga": 0}) for i in range(12)]
    best = rank_thirds(thirds, rng)
    assert len(best) == 8
    assert "T11" in best and "T0" not in best
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_groups.py -v`
Expected: FAIL — cannot import `simulate_group`

- [ ] **Step 3: Write minimal implementation (append to `simulator.py`)**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_groups.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/simulator.py tests/test_groups.py
git commit -m "feat: add group stage, tiebreakers, and best-thirds ranking"
```

---

## Task 6: Knockout bracket (`simulator.py` part 3)

**Files:**
- Modify: `src/worldcup/simulator.py`
- Test: `tests/test_knockout.py`

> **Note (stated simplification):** v1 builds a fixed single-elimination bracket from a standard seed order (seed 1 vs seed 32, etc.), where the 32 qualifiers are seeded winners (12) → runners-up (12) → best thirds (8), each tier ordered by pts/GD/GF. This approximates FIFA's official positional bracket and the third-place lookup table, which are an open item for v2.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_knockout.py
import numpy as np
from worldcup.models import MatchModelParams
from worldcup.simulator import bracket_seed_order, play_match_ko, play_knockout


def test_bracket_seed_order_small():
    assert bracket_seed_order(4) == [1, 4, 2, 3]
    order = bracket_seed_order(32)
    assert len(order) == 32 and sorted(order) == list(range(1, 33))
    assert order[0] == 1 and order[1] == 32  # top seed meets bottom seed


def test_play_match_ko_returns_a_participant():
    p = MatchModelParams()
    ratings = {"A": 2000.0, "B": 1900.0}
    w = play_match_ko("A", "B", ratings, np.random.default_rng(3), p)
    assert w in ("A", "B")


def test_play_knockout_strongest_seed_usually_wins():
    p = MatchModelParams()
    teams = [f"S{i}" for i in range(32)]
    ratings = {t: 1800.0 for t in teams}
    ratings["S0"] = 2600.0  # dominant seed-1
    wins = sum(play_knockout(teams, ratings, np.random.default_rng(i), p) == "S0"
               for i in range(200))
    assert wins > 100  # dominant team wins a clear majority
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_knockout.py -v`
Expected: FAIL — cannot import `bracket_seed_order`

- [ ] **Step 3: Write minimal implementation (append to `simulator.py`)**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_knockout.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/simulator.py tests/test_knockout.py
git commit -m "feat: add seeded knockout bracket simulation"
```

---

## Task 7: Full Monte Carlo (`simulator.py` part 4)

**Files:**
- Modify: `src/worldcup/simulator.py`
- Test: `tests/test_montecarlo.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_montecarlo.py
import numpy as np
from worldcup.models import MatchModelParams
from worldcup.simulator import simulate_once, run_simulation


def _synthetic_world():
    """12 groups of 4 = 48 teams; one dominant team."""
    groups, ratings = {}, {}
    labels = "ABCDEFGHIJKL"
    n = 0
    for g in labels:
        members = []
        for _ in range(4):
            name = f"T{n}"
            ratings[name] = 1800.0
            members.append(name)
            n += 1
        groups[g] = members
    ratings["T0"] = 2500.0  # dominant
    return groups, ratings


def test_simulate_once_returns_a_real_team():
    p = MatchModelParams()
    groups, ratings = _synthetic_world()
    champ = simulate_once(ratings, groups, np.random.default_rng(0), p)
    assert champ in ratings


def test_run_simulation_probabilities_sum_to_one_and_are_seed_stable():
    groups, ratings = _synthetic_world()
    r1 = run_simulation(ratings, groups, n=300, seed=42)
    r2 = run_simulation(ratings, groups, n=300, seed=42)
    assert r1 == r2                                  # deterministic
    assert abs(sum(r1.values()) - 1.0) < 1e-9        # normalized
    assert set(r1) == set(ratings)                   # every team present


def test_dominant_team_has_highest_probability():
    groups, ratings = _synthetic_world()
    res = run_simulation(ratings, groups, n=500, seed=1)
    assert max(res, key=res.get) == "T0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_montecarlo.py -v`
Expected: FAIL — cannot import `simulate_once`

- [ ] **Step 3: Write minimal implementation (append to `simulator.py`)**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_montecarlo.py -v`
Expected: PASS (3 passed). If slow, lower `n` in tests only; production default stays 20000.

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/simulator.py tests/test_montecarlo.py
git commit -m "feat: add full Monte Carlo tournament simulation"
```

---

## Task 8: Forecast blending (`blend.py`)

**Files:**
- Create: `src/worldcup/blend.py`
- Test: `tests/test_blend.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_blend.py
import math
from worldcup.blend import blend


def test_blend_sums_to_one():
    model = {"A": 0.5, "B": 0.3, "C": 0.2}
    market = {"A": 0.4, "B": 0.4, "C": 0.2}
    liq = {"A": 1e6, "B": 1e6, "C": 1e6}
    out = blend(model, market, liq, w_cap=0.7, K=1e6)
    assert math.isclose(sum(out.values()), 1.0, abs_tol=1e-9)


def test_unpriced_team_falls_back_to_model_then_renormalizes():
    model = {"A": 0.6, "B": 0.4}
    market = {"A": 0.5}                       # B unpriced
    liq = {"A": 1e9}                          # A fully liquid
    out = blend(model, market, liq, w_cap=0.7, K=1e6)
    # A pulled toward market (0.5), B stays at model pre-normalization.
    assert out["A"] < 0.6
    assert math.isclose(sum(out.values()), 1.0, abs_tol=1e-9)


def test_zero_liquidity_team_is_model_only_before_norm():
    model = {"A": 0.5, "B": 0.5}
    market = {"A": 0.9, "B": 0.1}
    liq = {"A": 0.0, "B": 0.0}               # no confidence => model only
    out = blend(model, market, liq, w_cap=0.7, K=1e6)
    assert math.isclose(out["A"], out["B"], abs_tol=1e-9)


def test_fixed_weight_mode():
    model = {"A": 0.5, "B": 0.5}
    market = {"A": 1.0, "B": 0.0}
    out = blend(model, market, {"A": 1.0, "B": 1.0}, fixed_w=0.5)
    # pre-norm: A=0.75, B=0.25 -> normalized identical ratio
    assert math.isclose(out["A"], 0.75, abs_tol=1e-9)
    assert math.isclose(out["B"], 0.25, abs_tol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_blend.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

```python
# src/worldcup/blend.py
"""Pure liquidity-weighted blend of model% and market% into the final forecast."""
from __future__ import annotations


def blend(model: dict[str, float], market: dict[str, float],
          liquidity: dict[str, float], w_cap: float = 0.7, K: float = 1e6,
          fixed_w: float | None = None) -> dict[str, float]:
    """Combine model and market probabilities per team, then renormalize to sum 1.

    For a priced team, market weight w = w_cap * L/(L+K) (or `fixed_w` if given).
    Unpriced teams (or zero liquidity) fall back to the model probability.
    """
    out: dict[str, float] = {}
    for team, m in model.items():
        if team in market:
            if fixed_w is not None:
                w = fixed_w
            else:
                L = max(0.0, liquidity.get(team, 0.0))
                w = w_cap * (L / (L + K)) if (L + K) > 0 else 0.0
            out[team] = w * market[team] + (1 - w) * m
        else:
            out[team] = m
    total = sum(out.values())
    if total <= 0:
        return out
    return {t: v / total for t, v in out.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_blend.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/blend.py tests/test_blend.py
git commit -m "feat: add liquidity-weighted forecast blending"
```

---

## Task 9: Value detection (`stake.py` part 1)

**Files:**
- Create: `src/worldcup/stake.py`
- Test: `tests/test_stake.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stake.py
import math
from worldcup.stake import find_value_bets


def test_only_positive_ev_teams_kept_and_edge_uses_model_minus_market():
    model = {"A": 0.30, "B": 0.10, "C": 0.20}
    market = {"A": 0.22, "B": 0.12, "C": 0.20}
    ask = {"A": 0.25, "B": 0.13, "C": 0.20}      # +EV only where model > ask
    bets = find_value_bets(model, market, ask)
    teams = {b["team"] for b in bets}
    assert teams == {"A"}                          # B and C are not +EV
    a = bets[0]
    assert math.isclose(a["edge"], 0.30 - 0.22, abs_tol=1e-9)   # model - market
    assert math.isclose(a["ev"], 0.30 / 0.25 - 1, abs_tol=1e-9)


def test_no_value_returns_empty():
    model = {"A": 0.20}
    market = {"A": 0.25}
    ask = {"A": 0.26}
    assert find_value_bets(model, market, ask) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_stake.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

```python
# src/worldcup/stake.py
"""Pure value detection + fractional-Kelly bankroll allocation.

Betting edge is model% - market% (NOT the liquidity-weighted blend) to avoid
circularity: blending in the same market we bet against would steer stakes onto
thin longshots. The +EV test and Kelly math use the ASK (the executable price).
"""
from __future__ import annotations


def find_value_bets(model: dict[str, float], market: dict[str, float],
                    ask: dict[str, float]) -> list[dict]:
    """Return +EV candidate bets (buying at ask), each with edge and EV.

    A bet is +EV when model% > ask. `edge` is the headline signal model - market;
    `ev` is expected value per $1 staked at the ask.
    """
    bets = []
    for team, p in model.items():
        a = ask.get(team)
        if a is None or a <= 0 or a >= 1:
            continue
        if p > a:
            bets.append({
                "team": team,
                "ask": a,
                "p": p,
                "market": market.get(team, a),
                "edge": p - market.get(team, a),
                "ev": p / a - 1.0,
            })
    return bets
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_stake.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/stake.py tests/test_stake.py
git commit -m "feat: add value-bet detection (model vs market, priced at ask)"
```

---

## Task 10: Fractional-Kelly allocation (`stake.py` part 2)

**Files:**
- Modify: `src/worldcup/stake.py`
- Test: `tests/test_stake.py`

> **Math:** Outcomes are mutually exclusive (one champion). Maximize expected log-wealth over stake fractions `f_i` (of bankroll), buying `f_i·B/ask_i` shares paying $1 if team i wins. Wealth/B if team j wins = `1 − Σf + f_j/ask_j`; if no bet team wins = `1 − Σf`. Solve the convex problem with `scipy.optimize.minimize` (SLSQP), then scale by the Kelly fraction and cap each stake by liquidity. A single-bet solution must approach the closed form `f* = (p − ask)/(1 − ask)`.

- [ ] **Step 1: Write the failing test (append to `tests/test_stake.py`)**

```python
from worldcup.stake import kelly_allocate, recommend_bets


def test_single_bet_matches_closed_form_kelly():
    # One +EV bet; full Kelly should approach (p-a)/(1-a).
    bets = [{"team": "A", "ask": 0.40, "p": 0.60, "market": 0.45,
             "edge": 0.15, "ev": 0.5}]
    recs = kelly_allocate(bets, bankroll=1000.0, kelly_fraction=1.0)
    expected_fraction = (0.60 - 0.40) / (1 - 0.40)   # = 1/3
    assert abs(recs[0].stake - expected_fraction * 1000.0) < 15.0


def test_total_stake_never_exceeds_bankroll():
    bets = [
        {"team": "A", "ask": 0.20, "p": 0.40, "market": 0.25, "edge": 0.15, "ev": 1.0},
        {"team": "B", "ask": 0.10, "p": 0.25, "market": 0.12, "edge": 0.13, "ev": 1.5},
    ]
    recs = kelly_allocate(bets, bankroll=500.0, kelly_fraction=0.5)
    assert sum(r.stake for r in recs) <= 500.0 + 1e-6


def test_liquidity_caps_stake():
    bets = [{"team": "A", "ask": 0.40, "p": 0.90, "market": 0.45,
             "edge": 0.45, "ev": 1.25}]
    recs = kelly_allocate(bets, bankroll=1000.0, kelly_fraction=1.0,
                          liquidity={"A": 25.0})
    assert recs[0].stake <= 25.0 + 1e-9


def test_recommend_bets_empty_when_no_value():
    recs = recommend_bets({"A": 0.20}, {"A": 0.25}, {"A": 0.26},
                          bankroll=100.0)
    assert recs == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_stake.py -v`
Expected: FAIL — cannot import `kelly_allocate`

- [ ] **Step 3: Write minimal implementation (append to `stake.py`)**

```python
import numpy as np
from scipy.optimize import minimize

from worldcup.models import BetRec


def kelly_allocate(bets: list[dict], bankroll: float, kelly_fraction: float = 0.5,
                   liquidity: dict[str, float] | None = None) -> list[BetRec]:
    """Allocate bankroll across mutually-exclusive +EV bets via fractional Kelly.

    Maximizes expected log-wealth; scales the optimum by `kelly_fraction`; caps
    each stake by available liquidity. Returns BetRec rows (zero-stake bets dropped).
    """
    if not bets or bankroll <= 0:
        return []
    liquidity = liquidity or {}
    p = np.array([b["p"] for b in bets])
    a = np.array([b["ask"] for b in bets])
    q0 = max(0.0, 1.0 - float(p.sum()))           # prob none of the bet teams win

    def neg_log_wealth(f: np.ndarray) -> float:
        spent = float(f.sum())
        if spent >= 1.0:
            return 1e9
        base = 1.0 - spent
        win_wealth = base + f / a                  # wealth/B if team i wins
        if np.any(win_wealth <= 0) or base <= 0:
            return 1e9
        return -(float((p * np.log(win_wealth)).sum()) + q0 * np.log(base))

    n = len(bets)
    res = minimize(
        neg_log_wealth, x0=np.full(n, 0.01),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "ineq", "fun": lambda f: 1.0 - f.sum() - 1e-6}],
    )
    fractions = np.clip(res.x, 0.0, 1.0) * kelly_fraction

    recs: list[BetRec] = []
    for b, frac in zip(bets, fractions):
        stake = frac * bankroll
        cap = liquidity.get(b["team"])
        if cap is not None:
            stake = min(stake, cap)
        if stake <= 1e-6:
            continue
        profit = stake * (1.0 - b["ask"]) / b["ask"]
        recs.append(BetRec(
            team=b["team"], ask=b["ask"], model_pct=b["p"], market_pct=b["market"],
            edge=b["edge"], ev_pct=b["ev"], stake=round(stake, 2),
            potential_profit=round(profit, 2),
        ))
    recs.sort(key=lambda r: r.stake, reverse=True)
    return recs


def recommend_bets(model: dict[str, float], market: dict[str, float],
                   ask: dict[str, float], bankroll: float,
                   kelly_fraction: float = 0.5,
                   liquidity: dict[str, float] | None = None) -> list[BetRec]:
    """End-to-end: detect +EV bets, then Kelly-allocate the bankroll across them."""
    return kelly_allocate(find_value_bets(model, market, ask),
                          bankroll, kelly_fraction, liquidity)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_stake.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/stake.py tests/test_stake.py
git commit -m "feat: add fractional-Kelly bankroll allocation"
```

---

## Task 11: Team ratings I/O (`ratings.py`)

**Files:**
- Create: `src/worldcup/ratings.py`, `tests/fixtures/ratings_sample.json`
- Test: `tests/test_ratings.py`

> **Open item:** confirm the live Elo source + endpoint during this task (e.g. an eloratings.net export or an Elo JSON API). The parser below targets a simple `{"ratings": [{"team","elo"}]}` shape; capture a real response into the fixture and adjust field names if they differ. Keep `parse_ratings` separate from `fetch_ratings` so parsing stays unit-testable.

- [ ] **Step 1: Create the fixture**

```json
// tests/fixtures/ratings_sample.json
{
  "ratings": [
    {"team": "France", "elo": 2090},
    {"team": "USA", "elo": 1825},
    {"team": "Korea Republic", "elo": 1740}
  ]
}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_ratings.py
import json
from pathlib import Path
from worldcup.ratings import parse_ratings

FIX = Path(__file__).parent / "fixtures" / "ratings_sample.json"


def test_parse_ratings_normalizes_names_and_floats():
    data = json.loads(FIX.read_text())
    out = parse_ratings(data)
    assert out["United States"] == 1825.0       # USA normalized
    assert out["South Korea"] == 1740.0         # Korea Republic normalized
    assert isinstance(out["France"], float)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ratings.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Write minimal implementation**

```python
# src/worldcup/ratings.py
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_ratings.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: Commit**

```bash
git add src/worldcup/ratings.py tests/test_ratings.py tests/fixtures/ratings_sample.json
git commit -m "feat: add team ratings fetch + parse"
```

---

## Task 12: Prediction-market I/O (`markets.py`)

**Files:**
- Create: `src/worldcup/markets.py`, `tests/fixtures/polymarket_sample.json`, `tests/fixtures/kalshi_sample.json`
- Test: `tests/test_markets.py`

> **Open item:** confirm exact field names for Polymarket `gamma-api` (`world-cup-winner` event) and Kalshi (`KXMENWORLDCUP-26`), and whether ask/order-book depth is exposed. The parsers below target documented-looking shapes; capture real responses into the fixtures and adjust. Polymarket prices are 0–1; Kalshi prices are cents (0–100).

- [ ] **Step 1: Create fixtures**

```json
// tests/fixtures/polymarket_sample.json
{
  "title": "World Cup Winner",
  "markets": [
    {"groupItemTitle": "France", "lastTradePrice": 0.17, "bestAsk": 0.18, "liquidityNum": 250000, "volumeNum": 8000000},
    {"groupItemTitle": "Spain",  "lastTradePrice": 0.16, "bestAsk": 0.17, "liquidityNum": 240000, "volumeNum": 7000000}
  ]
}
```

```json
// tests/fixtures/kalshi_sample.json
{
  "markets": [
    {"yes_sub_title": "France", "last_price": 17, "yes_ask": 19, "liquidity": 200000, "volume": 5000000},
    {"yes_sub_title": "England", "last_price": 11, "yes_ask": 12, "liquidity": 150000, "volume": 3000000}
  ]
}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_markets.py
import json
import math
from pathlib import Path
from worldcup.markets import parse_polymarket, parse_kalshi, combine_markets

FIX = Path(__file__).parent / "fixtures"


def test_parse_polymarket():
    data = json.loads((FIX / "polymarket_sample.json").read_text())
    lines = parse_polymarket(data)
    assert math.isclose(lines["France"]["prob"], 0.17)
    assert math.isclose(lines["France"]["ask"], 0.18)
    assert lines["France"]["liquidity"] == 250000


def test_parse_kalshi_converts_cents():
    data = json.loads((FIX / "kalshi_sample.json").read_text())
    lines = parse_kalshi(data)
    assert math.isclose(lines["France"]["prob"], 0.17)
    assert math.isclose(lines["France"]["ask"], 0.19)


def test_combine_devigs_and_liquidity_weights():
    poly = {"France": {"prob": 0.6, "ask": 0.61, "liquidity": 100.0},
            "Spain": {"prob": 0.6, "ask": 0.61, "liquidity": 100.0}}  # sums to 1.2
    kalshi = {"France": {"prob": 0.5, "ask": 0.51, "liquidity": 300.0},
              "Spain": {"prob": 0.5, "ask": 0.51, "liquidity": 300.0}}
    prob, ask, liq = combine_markets(poly, kalshi)
    assert math.isclose(prob["France"] + prob["Spain"], 1.0, abs_tol=1e-9)  # de-vigged
    assert liq["France"] == 400.0                                           # summed
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_markets.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Write minimal implementation**

```python
# src/worldcup/markets.py
"""Fetch + parse Polymarket and Kalshi outright-winner markets.

Returns, per team: implied probability (vig-stripped at combine time), the ask
price you would pay, and a liquidity figure used for blend weighting + stake caps.
"""
from __future__ import annotations

import httpx

from worldcup.teamnames import canonical

POLYMARKET_URL = "https://gamma-api.polymarket.com/events?slug=world-cup-winner"
KALSHI_URL = "https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXMENWORLDCUP&status=open"


def parse_polymarket(payload: dict) -> dict[str, dict]:
    """Parse a gamma-api event payload into {team: {prob, ask, liquidity}}."""
    event = payload[0] if isinstance(payload, list) else payload
    out: dict[str, dict] = {}
    for m in event.get("markets", []):
        team = canonical(m["groupItemTitle"])
        out[team] = {
            "prob": float(m["lastTradePrice"]),
            "ask": float(m.get("bestAsk", m["lastTradePrice"])),
            "liquidity": float(m.get("liquidityNum", 0.0)),
        }
    return out


def parse_kalshi(payload: dict) -> dict[str, dict]:
    """Parse a Kalshi markets payload (cent prices) into {team: {prob, ask, liquidity}}."""
    out: dict[str, dict] = {}
    for m in payload.get("markets", []):
        team = canonical(m["yes_sub_title"])
        out[team] = {
            "prob": float(m["last_price"]) / 100.0,
            "ask": float(m.get("yes_ask", m["last_price"])) / 100.0,
            "liquidity": float(m.get("liquidity", 0.0)),
        }
    return out


def _devig(lines: dict[str, dict]) -> dict[str, float]:
    total = sum(v["prob"] for v in lines.values())
    if total <= 0:
        return {t: 0.0 for t in lines}
    return {t: v["prob"] / total for t, v in lines.items()}


def combine_markets(poly: dict[str, dict], kalshi: dict[str, dict]):
    """De-vig each market, then liquidity-weight the two into one consensus.

    Returns (prob, ask, liquidity) dicts keyed by team. `ask` is the cheaper of
    the two available asks (best executable price).
    """
    poly_p, kalshi_p = _devig(poly), _devig(kalshi)
    teams = set(poly) | set(kalshi)
    prob, ask, liq = {}, {}, {}
    for t in teams:
        lp = poly.get(t, {}).get("liquidity", 0.0)
        lk = kalshi.get(t, {}).get("liquidity", 0.0)
        total_l = lp + lk
        if total_l > 0:
            prob[t] = (lp * poly_p.get(t, 0.0) + lk * kalshi_p.get(t, 0.0)) / total_l
        else:
            prob[t] = poly_p.get(t, kalshi_p.get(t, 0.0))
        asks = [x["ask"] for x in (poly.get(t), kalshi.get(t)) if x]
        ask[t] = min(asks) if asks else 1.0
        liq[t] = total_l
    return prob, ask, liq


def fetch_market_probabilities(timeout: float = 20.0):
    """Fetch both markets and combine. Returns (prob, ask, liquidity).

    If one source fails, proceed with the other. Raises only if BOTH fail.
    """
    poly, kalshi = {}, {}
    errors = []
    try:
        r = httpx.get(POLYMARKET_URL, timeout=timeout); r.raise_for_status()
        poly = parse_polymarket(r.json())
    except Exception as e:  # noqa: BLE001 - degrade gracefully, record reason
        errors.append(f"polymarket: {e}")
    try:
        r = httpx.get(KALSHI_URL, timeout=timeout); r.raise_for_status()
        kalshi = parse_kalshi(r.json())
    except Exception as e:  # noqa: BLE001
        errors.append(f"kalshi: {e}")
    if not poly and not kalshi:
        raise RuntimeError("both market sources failed: " + "; ".join(errors))
    return combine_markets(poly, kalshi)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_markets.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/worldcup/markets.py tests/test_markets.py tests/fixtures/polymarket_sample.json tests/fixtures/kalshi_sample.json
git commit -m "feat: add Polymarket + Kalshi fetch, de-vig, and consensus"
```

---

## Task 13: Group-draw I/O + validation (`draw.py`)

**Files:**
- Create: `src/worldcup/draw.py`, `tests/fixtures/draw_sample.json`
- Test: `tests/test_draw.py`

> **Open item:** confirm the live group-draw source during this task. The parser targets `{"groups": {"A": [4 names], ...}}`; capture a real response into the fixture and adjust. The validator (12 groups × 4) is the important, source-independent part.

- [ ] **Step 1: Create the fixture (12 groups × 4; names illustrative)**

```json
// tests/fixtures/draw_sample.json
{
  "groups": {
    "A": ["Mexico", "T2", "T3", "T4"],
    "B": ["Canada", "T6", "T7", "T8"],
    "C": ["USA", "T10", "T11", "T12"],
    "D": ["T13", "T14", "T15", "T16"],
    "E": ["T17", "T18", "T19", "T20"],
    "F": ["T21", "T22", "T23", "T24"],
    "G": ["T25", "T26", "T27", "T28"],
    "H": ["T29", "T30", "T31", "T32"],
    "I": ["T33", "T34", "T35", "T36"],
    "J": ["T37", "T38", "T39", "T40"],
    "K": ["T41", "T42", "T43", "T44"],
    "L": ["T45", "T46", "T47", "T48"]
  }
}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_draw.py
import json
import pytest
from pathlib import Path
from worldcup.draw import parse_draw, validate_draw

FIX = Path(__file__).parent / "fixtures" / "draw_sample.json"


def test_parse_and_validate_ok():
    groups = parse_draw(json.loads(FIX.read_text()))
    validate_draw(groups)                       # must not raise
    assert len(groups) == 12
    assert all(len(v) == 4 for v in groups.values())
    assert groups["C"][0] == "United States"    # normalized


def test_validate_rejects_wrong_group_count():
    with pytest.raises(ValueError):
        validate_draw({"A": ["a", "b", "c", "d"]})


def test_validate_rejects_short_group():
    bad = {chr(65 + i): ["a", "b", "c", "d"] for i in range(12)}
    bad["A"] = ["a", "b", "c"]                   # only 3
    with pytest.raises(ValueError):
        validate_draw(bad)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_draw.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Write minimal implementation**

```python
# src/worldcup/draw.py
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_draw.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/worldcup/draw.py tests/test_draw.py tests/fixtures/draw_sample.json
git commit -m "feat: add group-draw fetch, parse, and 12x4 validation"
```

---

## Task 14: Reporting (`report.py`)

**Files:**
- Create: `src/worldcup/report.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
from worldcup.models import Forecast, BetRec
from worldcup.report import build_markdown

DISCLAIMER = "never stake more than you can afford to lose"


def test_markdown_has_forecast_rows_sorted_and_betting_card():
    forecasts = [
        Forecast("Spain", 0.15, 0.14, 0.145),
        Forecast("France", 0.12, 0.16, 0.14),
    ]
    bets = [BetRec("Spain", 0.16, 0.15, 0.14, 0.01, -0.0625, 20.0, 105.0)]
    md = build_markdown(forecasts, bets, bankroll=100.0)
    assert "Spain" in md and "France" in md
    assert "## Betting card" in md
    assert DISCLAIMER in md


def test_markdown_handles_no_value_bets():
    forecasts = [Forecast("Spain", 0.15, 0.14, 0.145)]
    md = build_markdown(forecasts, [], bankroll=100.0)
    assert "no value bets" in md.lower()


def test_markdown_omits_card_when_no_bankroll():
    md = build_markdown([Forecast("Spain", 0.15, 0.14, 0.145)], [], bankroll=None)
    assert "Betting card" not in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_report.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

```python
# src/worldcup/report.py
"""Render the forecast table and betting card to the terminal and markdown."""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

from worldcup.models import BetRec, Forecast

DISCLAIMER = ("Model-based estimate; prediction markets are highly efficient. "
              "For education only — never stake more than you can afford to lose.")


def build_markdown(forecasts: list[Forecast], bets: list[BetRec],
                   bankroll: float | None) -> str:
    """Build the markdown report string (forecast table + optional betting card)."""
    lines = ["# 2026 World Cup Forecast", "",
             "| Team | Model | Market | Blended |", "|------|------:|-------:|--------:|"]
    for f in sorted(forecasts, key=lambda x: x.blended_pct, reverse=True):
        lines.append(f"| {f.team} | {f.model_pct:.1%} | {f.market_pct:.1%} | {f.blended_pct:.1%} |")

    if bankroll is not None:
        lines += ["", "## Betting card", f"_Bankroll: ${bankroll:,.2f}_", ""]
        if not bets:
            lines.append("No value bets at current prices.")
        else:
            lines += ["| Team | Ask | Model | Market | Edge | EV | Stake | Profit |",
                      "|------|----:|------:|-------:|-----:|---:|------:|-------:|"]
            for b in bets:
                lines.append(
                    f"| {b.team} | {b.ask:.2f} | {b.model_pct:.1%} | {b.market_pct:.1%} | "
                    f"{b.edge:+.1%} | {b.ev_pct:+.1%} | ${b.stake:,.2f} | ${b.potential_profit:,.2f} |")
            staked = sum(b.stake for b in bets)
            lines += ["", f"_Total staked: ${staked:,.2f} · Reserve: ${bankroll - staked:,.2f}_"]
        lines += ["", f"> {DISCLAIMER}"]
    return "\n".join(lines) + "\n"


def print_report(forecasts: list[Forecast], bets: list[BetRec],
                 bankroll: float | None, console: Console | None = None) -> None:
    """Pretty-print the forecast and betting card to the terminal."""
    console = console or Console()
    ft = Table(title="2026 World Cup Forecast")
    for col in ("Team", "Model", "Market", "Blended"):
        ft.add_column(col, justify="right" if col != "Team" else "left")
    for f in sorted(forecasts, key=lambda x: x.blended_pct, reverse=True):
        ft.add_row(f.team, f"{f.model_pct:.1%}", f"{f.market_pct:.1%}", f"{f.blended_pct:.1%}")
    console.print(ft)

    if bankroll is None:
        return
    if not bets:
        console.print(f"\n[bold]No value bets[/bold] at current prices (bankroll ${bankroll:,.2f}).")
    else:
        bt = Table(title=f"Betting card (bankroll ${bankroll:,.2f})")
        for col in ("Team", "Ask", "Model", "Market", "Edge", "EV", "Stake", "Profit"):
            bt.add_column(col, justify="right" if col != "Team" else "left")
        for b in bets:
            bt.add_row(b.team, f"{b.ask:.2f}", f"{b.model_pct:.1%}", f"{b.market_pct:.1%}",
                       f"{b.edge:+.1%}", f"{b.ev_pct:+.1%}", f"${b.stake:,.2f}", f"${b.potential_profit:,.2f}")
        console.print(bt)
    console.print(f"\n[dim]{DISCLAIMER}[/dim]")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_report.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/report.py tests/test_report.py
git commit -m "feat: add terminal + markdown reporting"
```

---

## Task 15: Agent wiring (`agent.py`)

**Files:**
- Create: `src/worldcup/agent.py`
- Test: `tests/test_agent_pipeline.py` (tests the pure pipeline helper, not the LLM)

> **Open item:** verify the Claude Agent SDK tool/MCP API against current docs before running live. Use the `context7` MCP (`resolve-library-id` → `query-docs` for "claude-agent-sdk") or the `agent-sdk-dev:agent-sdk-verifier-py` skill. The structure below (`@tool`, `create_sdk_mcp_server`, `ClaudeAgentOptions`, `query`) matches the documented Python SDK; adjust import/return shapes if the installed version differs. To keep logic testable without the LLM, all orchestration lives in a plain `run_pipeline()` function that the tools and tests both call.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_pipeline.py
from worldcup.agent import run_pipeline


def test_run_pipeline_end_to_end_with_injected_data():
    # 12x4 synthetic groups + ratings (no network).
    groups, ratings = {}, {}
    n = 0
    for g in "ABCDEFGHIJKL":
        members = []
        for _ in range(4):
            name = f"T{n}"; ratings[name] = 1800.0; members.append(name); n += 1
        groups[g] = members
    ratings["T0"] = 2400.0
    market = {"T0": 0.30, "T1": 0.10}
    ask = {"T0": 0.20, "T1": 0.12}
    liquidity = {"T0": 1e6, "T1": 1e6}

    result = run_pipeline(ratings, groups, market, ask, liquidity,
                          bankroll=100.0, n_sims=200, seed=1)
    assert abs(sum(f.blended_pct for f in result.forecasts) - 1.0) < 1e-6
    assert result.forecasts[0].team == "T0"                 # sorted, dominant first
    # T0 is +EV (model >> ask) so it should be recommended.
    assert any(b.team == "T0" for b in result.bets)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_agent_pipeline.py -v`
Expected: FAIL — cannot import `run_pipeline`

- [ ] **Step 3: Write minimal implementation**

```python
# src/worldcup/agent.py
"""Claude Agent SDK wiring + the deterministic pipeline the agent orchestrates.

`run_pipeline` is pure (no network, no LLM) so it is fully unit-testable. The SDK
tools are thin wrappers that fetch live data and then call the same pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

from worldcup.blend import blend
from worldcup.models import BetRec, Forecast
from worldcup.simulator import run_simulation
from worldcup.stake import recommend_bets


@dataclass
class PipelineResult:
    forecasts: list[Forecast]
    bets: list[BetRec]


def run_pipeline(ratings, groups, market, ask, liquidity, *, bankroll,
                 n_sims=20000, seed=42, w_cap=0.7, K=1e6,
                 kelly_fraction=0.5) -> PipelineResult:
    """Simulate, blend, and (if bankroll given) recommend bets."""
    model = run_simulation(ratings, groups, n=n_sims, seed=seed)
    blended = blend(model, market, liquidity, w_cap=w_cap, K=K)
    forecasts = [Forecast(t, model[t], market.get(t, 0.0), blended[t]) for t in model]
    forecasts.sort(key=lambda f: f.blended_pct, reverse=True)
    bets: list[BetRec] = []
    if bankroll:
        bets = recommend_bets(model, market, ask, bankroll,
                              kelly_fraction=kelly_fraction, liquidity=liquidity)
    return PipelineResult(forecasts=forecasts, bets=bets)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_agent_pipeline.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Add the SDK agent entrypoint (append to `agent.py`)**

```python
# --- Claude Agent SDK integration -------------------------------------------
# Verify against installed claude-agent-sdk docs before first live run.
import json

from claude_agent_sdk import (  # type: ignore
    ClaudeAgentOptions, create_sdk_mcp_server, query, tool,
)

from worldcup.draw import fetch_group_draw
from worldcup.markets import fetch_market_probabilities
from worldcup.ratings import fetch_ratings
from worldcup.report import build_markdown, print_report

SYSTEM_PROMPT = """You are a World Cup forecasting agent. Run these steps in order
and narrate each: (1) fetch_group_draw — STOP if it is not 12 groups of 4;
(2) fetch_team_ratings; (3) fetch_market_probabilities; (4) run_forecast with the
user's bankroll. Never invent ratings or prices. If a step fails, report the error
plainly. Finish by presenting the forecast table and betting card."""


@tool("fetch_group_draw", "Fetch and validate the 2026 group draw (12x4)", {})
async def _t_draw(args):
    groups = fetch_group_draw()
    return {"content": [{"type": "text", "text": json.dumps(groups)}]}


@tool("fetch_team_ratings", "Fetch international Elo ratings for all teams", {})
async def _t_ratings(args):
    return {"content": [{"type": "text", "text": json.dumps(fetch_ratings())}]}


@tool("fetch_market_probabilities", "Fetch + de-vig Polymarket and Kalshi prices", {})
async def _t_markets(args):
    prob, ask, liq = fetch_market_probabilities()
    return {"content": [{"type": "text", "text": json.dumps({"prob": prob, "ask": ask, "liquidity": liq})}]}


@tool("run_forecast", "Simulate, blend, and recommend bets",
      {"bankroll": float, "n_sims": int, "seed": int, "kelly_fraction": float})
async def _t_forecast(args):
    groups = fetch_group_draw()
    ratings = fetch_ratings()
    prob, ask, liq = fetch_market_probabilities()
    result = run_pipeline(ratings, groups, prob, ask, liq,
                          bankroll=args.get("bankroll"),
                          n_sims=int(args.get("n_sims", 20000)),
                          seed=int(args.get("seed", 42)),
                          kelly_fraction=float(args.get("kelly_fraction", 0.5)))
    print_report(result.forecasts, result.bets, args.get("bankroll"))
    md = build_markdown(result.forecasts, result.bets, args.get("bankroll"))
    return {"content": [{"type": "text", "text": md}]}


def build_options() -> ClaudeAgentOptions:
    server = create_sdk_mcp_server(
        name="worldcup", version="0.1.0",
        tools=[_t_draw, _t_ratings, _t_markets, _t_forecast])
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"worldcup": server},
        allowed_tools=[
            "mcp__worldcup__fetch_group_draw",
            "mcp__worldcup__fetch_team_ratings",
            "mcp__worldcup__fetch_market_probabilities",
            "mcp__worldcup__run_forecast",
        ],
    )


async def run_agent(bankroll: float | None, n_sims: int, seed: int,
                    kelly_fraction: float) -> None:
    prompt = (f"Forecast the 2026 World Cup. Bankroll: "
              f"{bankroll if bankroll else 'none (skip betting card)'}. "
              f"Use n_sims={n_sims}, seed={seed}, kelly_fraction={kelly_fraction}.")
    async for message in query(prompt=prompt, options=build_options()):
        print(message)
```

- [ ] **Step 6: Verify the package still imports**

Run: `.venv/bin/python -c "import worldcup.agent"`
Expected: no error. If the SDK import or signatures differ from the installed version, consult the docs (context7 `query-docs`) and fix imports/return shapes, then re-run.

- [ ] **Step 7: Run the full test suite**

Run: `.venv/bin/pytest -q`
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/worldcup/agent.py tests/test_agent_pipeline.py
git commit -m "feat: add orchestration pipeline and Claude Agent SDK wiring"
```

---

## Task 16: CLI entrypoint (`main.py`) + smoke run

**Files:**
- Create: `main.py`

- [ ] **Step 1: Write `main.py`**

```python
# main.py
"""CLI entrypoint for the World Cup predictor agent."""
import argparse
import asyncio

from worldcup.agent import run_agent


def parse_args():
    ap = argparse.ArgumentParser(description="2026 World Cup forecaster + staking advisor")
    ap.add_argument("--bankroll", type=float, default=None,
                    help="Bankroll in dollars; omit to skip the betting card")
    ap.add_argument("--sims", type=int, default=20000, help="Monte Carlo simulations")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (reproducibility)")
    ap.add_argument("--kelly-fraction", type=float, default=0.5,
                    help="Fraction of full Kelly (default 0.5)")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.bankroll is not None and args.bankroll <= 0:
        raise SystemExit("error: --bankroll must be positive")
    asyncio.run(run_agent(args.bankroll, args.sims, args.seed, args.kelly_fraction))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the CLI parses (no network/LLM)**

Run: `.venv/bin/python main.py --help`
Expected: prints usage with `--bankroll`, `--sims`, `--seed`, `--kelly-fraction`.

- [ ] **Step 3: Verify bankroll guard**

Run: `.venv/bin/python main.py --bankroll -5`
Expected: exits with "error: --bankroll must be positive".

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: add CLI entrypoint"
```

> **Live smoke test (manual, after the open-item data sources are wired):** set `ANTHROPIC_API_KEY` in `.env`, confirm the real ratings/draw URLs in `ratings.py`/`draw.py`, then run `.venv/bin/python main.py --bankroll 100 --sims 20000`. Expect a forecast table topped by genuine contenders and a betting card with 0–2 value bets (often none). Do not commit secrets.

---

## Task 17: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

````markdown
# World Cup Predictor

A Claude Agent SDK app that forecasts each nation's 2026 World Cup title odds by
**blending a Monte Carlo simulation with prediction-market prices**, then recommends
Kelly-sized value bets from a bankroll you choose.

## How it works
- **Simulation** (`simulator.py`): Elo-driven Poisson scorelines → group stage (with
  tiebreakers + 8 best thirds) → seeded knockout → 20k-run championship probabilities.
- **Markets** (`markets.py`): Polymarket + Kalshi outright prices, de-vigged and
  liquidity-weighted into a consensus.
- **Blend** (`blend.py`): liquidity-weighted mix of model + market (configurable).
- **Staking** (`stake.py`): bets the *model vs market* edge at the ask, sized by
  fractional Kelly across mutually-exclusive outcomes, capped by liquidity.

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env   # add ANTHROPIC_API_KEY
```

## Run
```bash
.venv/bin/python main.py --bankroll 100 --sims 20000
```
Omit `--bankroll` for a forecast with no betting card.

## Test
```bash
.venv/bin/pytest -q
```

## Disclaimer
Educational tool. Prediction markets are highly efficient; expect **few or zero**
value bets. Never stake more than you can afford to lose.
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README"
```

---

## Self-Review (completed by plan author)

**Spec coverage:** Every spec section maps to a task — match model (T4), groups/tiebreakers/best-thirds (T5), knockout (T6), Monte Carlo (T7), markets de-vig/consensus (T12), liquidity-weighted blend + fixed-w (T8), staking with model−market edge / ask pricing / mutually-exclusive Kelly / liquidity cap (T9–T10), error handling (T7/T12/T13/T16), reporting + disclaimer + "no value bets" (T14), CLI bankroll (T16), team-name normalization (T3), offline fixture tests (T11–T13), agent orchestration (T15).

**Open items** from the spec are carried as explicit task notes where they bite: real ratings URL (T11), real market field names + ask depth (T12), real draw source (T13), exact FIFA bracket mapping → stated v1 seeding simplification (T6), SDK API verification (T15), calibration defaults (live smoke, T16).

**Placeholder scan:** No "TBD/implement later" in code steps; every code step is complete. The only `example.com` URLs are deliberate, called-out open items with working parsers + fixtures behind them.

**Type consistency:** `BetRec`/`Forecast` fields (T2) match their use in `stake.py` (T10), `report.py` (T14), `agent.py` (T15). Function names are stable across tasks: `run_simulation`, `blend`, `find_value_bets`, `kelly_allocate`, `recommend_bets`, `run_pipeline`, `parse_*`/`fetch_*`.
