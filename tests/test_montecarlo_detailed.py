import numpy as np
from worldcup.models import MatchModelParams
from worldcup.simulator import (
    simulate_once, simulate_once_detailed, run_simulation, run_simulation_detailed,
    STAGES,
)


def _synthetic_world():
    groups, ratings = {}, {}
    n = 0
    for g in "ABCDEFGHIJKL":
        members = []
        for _ in range(4):
            name = f"T{n}"; ratings[name] = 1800.0; members.append(name); n += 1
        groups[g] = members
    ratings["T0"] = 2500.0  # dominant
    return groups, ratings


def test_detailed_champion_matches_simulate_once_rng_for_rng():
    groups, ratings = _synthetic_world()
    p = MatchModelParams()
    r1 = np.random.default_rng(3); r2 = np.random.default_rng(3)
    for _ in range(50):
        c1 = simulate_once(ratings, groups, r1, p)
        c2, reached = simulate_once_detailed(ratings, groups, r2, p)
        assert c1 == c2
        assert reached[c2] == 5                      # champion reached final stage


def test_advancement_is_monotone_and_normalized():
    groups, ratings = _synthetic_world()
    champ, adv = run_simulation_detailed(ratings, groups, n=300, seed=42)
    for team, stages in adv.items():
        seq = [stages[s] for s in STAGES]
        for earlier, later in zip(seq, seq[1:]):
            assert earlier + 1e-9 >= later           # r32 >= r16 >= ... >= champion
        for v in seq:
            assert 0.0 <= v <= 1.0
        assert abs(stages["champion"] - champ[team]) < 1e-9   # champion stage == champ prob
    # 32 teams reach the KO round each sim => total r32 mass ~= 32
    assert abs(sum(s["r32"] for s in adv.values()) - 32.0) < 1.0


def test_detailed_champion_probs_equal_run_simulation():
    groups, ratings = _synthetic_world()
    base = run_simulation(ratings, groups, n=300, seed=7)
    champ, _ = run_simulation_detailed(ratings, groups, n=300, seed=7)
    assert champ == base                              # identical, same rng order
