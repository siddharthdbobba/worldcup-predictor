import math

from worldcup.models import ArbRec, EvBetRec, DutchBook


def test_dataclasses_construct():
    a = ArbRec(team="France", yes_venue="polymarket", no_venue="kalshi",
               yes_ask=0.18, no_ask=0.71, contracts=10,
               stake_yes=1.80, stake_no=7.10, total_cost=9.04,
               guaranteed_profit=0.96, roc=0.106, annual_roc=1.2)
    assert a.team == "France" and a.contracts == 10
    e = EvBetRec(team="Spain", venue="polymarket", side="YES", ask=0.10,
                 fair=0.14, ev_pct=0.4, stake=5.0, potential_profit=45.0)
    assert e.side == "YES"
    d = DutchBook(field_sum=1.03, gap=-0.03, is_arb=False, legs=[("France", "kalshi", 0.17)])
    assert d.is_arb is False and d.legs[0][0] == "France"


from worldcup.arb import find_locks, VENUE_POLY, VENUE_KALSHI

# Crafted book pair with a clear orientation-A lock on France.
# Poly YES 0.18 (bid 0.16); Kalshi YES 0.30 (bid 0.28), NO 0.71.
# Orientation A: YES Poly @0.18 + NO Kalshi @0.71 = 0.89 < 1, fires (k_bid .28 > p_ask .18).
POLY = {
    "France": {"prob": 0.17, "ask": 0.18, "yes_bid": 0.16, "no_ask": 0.84, "depth": 250000},
    "Spain":  {"prob": 0.10, "ask": 0.10, "yes_bid": 0.08, "no_ask": 0.92, "depth": 250000},
    "Brazil": {"prob": 0.20, "ask": 0.20, "yes_bid": 0.19, "no_ask": 0.81, "depth": 250000},
}
KALSHI = {
    "France": {"prob": 0.30, "ask": 0.30, "yes_bid": 0.28, "no_ask": 0.71, "no_bid": 0.70, "depth": 0},
    "Spain":  {"prob": 0.20, "ask": 0.20, "yes_bid": 0.18, "no_ask": 0.81, "no_bid": 0.80, "depth": 0},
    "Brazil": {"prob": 0.21, "ask": 0.21, "yes_bid": 0.20, "no_ask": 0.80, "no_bid": 0.79, "depth": 0},
}


def test_find_locks_picks_orientation_a():
    cands = find_locks(POLY, KALSHI, min_profit=0.02)
    france = next(c for c in cands if c.team == "France")
    assert france.yes_venue == VENUE_POLY and france.no_venue == VENUE_KALSHI
    assert math.isclose(france.a, 0.18) and math.isclose(france.b, 0.71)
    # raw 1-0.18-0.71 = 0.11; kalshi fee 0.07*0.71*0.29 ≈ 0.0144; profit ≈ 0.0956
    assert 0.09 < france.profit < 0.10


def test_find_locks_min_profit_gate_filters_thin():
    # Brazil: a=0.20, b=0.80 -> raw 0.0, fee>0 -> negative; never a lock.
    cands = find_locks(POLY, KALSHI, min_profit=0.02)
    assert all(c.team != "Brazil" for c in cands)


def test_find_locks_none_when_spreads_overlap():
    poly = {"France": {"prob": 0.5, "ask": 0.52, "yes_bid": 0.48, "no_ask": 0.52, "depth": 100}}
    kalshi = {"France": {"prob": 0.5, "ask": 0.52, "yes_bid": 0.48, "no_ask": 0.52, "no_bid": 0.46, "depth": 0}}
    assert find_locks(poly, kalshi, min_profit=0.0) == []


def test_find_locks_depth_caps_from_book_and_max_leg():
    cands = find_locks(POLY, KALSHI, min_profit=0.02, max_leg_stake=500.0)
    france = next(c for c in cands if c.team == "France")
    # YES leg on Poly: depth 250000 vs cap 500 -> 500; NO leg on Kalshi: depth 0 -> cap 500
    assert math.isclose(france.yes_cap, 500.0)
    assert math.isclose(france.no_cap, 500.0)


from worldcup.arb import size_locks


def test_size_locks_equal_payout_per_arb():
    cands = find_locks(POLY, KALSHI, min_profit=0.02, max_leg_stake=1000.0)
    recs = size_locks(cands, poly_balance=10_000, kalshi_balance=10_000,
                      days_to_settlement=47)
    france = next(r for r in recs if r.team == "France")
    # Equal contracts on both legs => identical payout (= contracts) either way.
    payout_if_france_wins = france.contracts          # YES pays 1 each
    payout_if_france_loses = france.contracts         # NO pays 1 each
    assert payout_if_france_wins == payout_if_france_loses
    assert france.guaranteed_profit > 0
    assert math.isclose(france.total_cost,
                        france.stake_yes + france.stake_no
                        + france.contracts * (next(c.fee for c in cands if c.team == "France")),
                        rel_tol=1e-6)


def test_size_locks_respects_venue_balances():
    cands = find_locks(POLY, KALSHI, min_profit=0.02, max_leg_stake=1e9)
    recs = size_locks(cands, poly_balance=18.0, kalshi_balance=1e9,
                      days_to_settlement=47)
    # Whichever locks the LP funds, total $ drawn from Polymarket must respect $18.
    poly_spent = sum(r.stake_yes if r.yes_venue == VENUE_POLY else r.stake_no
                     for r in recs)
    assert poly_spent <= 18.0 + 1e-6


def test_size_locks_respects_depth_cap():
    # max_leg_stake applies to BOTH legs; the pricier NO leg (0.71) binds first:
    # 7.10 / 0.71 = 10 contracts (YES leg would allow 7.10/0.18 = 39).
    cands = find_locks(POLY, KALSHI, min_profit=0.02, max_leg_stake=7.10)
    recs = size_locks(cands, poly_balance=1e9, kalshi_balance=1e9,
                      days_to_settlement=47)
    france = next(r for r in recs if r.team == "France")
    assert france.contracts == 10
    assert france.stake_no <= 7.10 + 1e-6


def test_size_locks_contracts_are_integers():
    cands = find_locks(POLY, KALSHI, min_profit=0.02, max_leg_stake=1000.0)
    recs = size_locks(cands, poly_balance=10_000, kalshi_balance=10_000,
                      days_to_settlement=47)
    assert all(isinstance(r.contracts, int) and r.contracts >= 1 for r in recs)


def test_size_locks_annualizes_roc():
    cands = find_locks(POLY, KALSHI, min_profit=0.02, max_leg_stake=1000.0)
    recs = size_locks(cands, poly_balance=10_000, kalshi_balance=10_000,
                      days_to_settlement=47)
    r = recs[0]
    expected = (1.0 + r.roc) ** (365.0 / 47) - 1.0
    assert math.isclose(r.annual_roc, expected, rel_tol=1e-9)


def test_size_locks_empty():
    assert size_locks([], poly_balance=100, kalshi_balance=100,
                      days_to_settlement=47) == []


from worldcup.arb import find_ev_bets


def test_find_ev_bets_flags_underpriced_book():
    # No lock, but Poly YES for Spain (0.10) sits below consensus 0.14 -> +EV.
    poly = {"Spain": {"ask": 0.10, "yes_bid": 0.08, "no_ask": 0.92, "depth": 1000}}
    kalshi = {"Spain": {"ask": 0.15, "yes_bid": 0.13, "no_ask": 0.86, "depth": 0}}
    consensus = {"Spain": 0.14}
    bets = find_ev_bets(poly, kalshi, consensus, poly_balance=1000,
                        kalshi_balance=1000, max_leg_stake=1000, kelly_fraction=0.5)
    spain = next(b for b in bets if b.team == "Spain")
    assert spain.venue == "polymarket" and spain.side == "YES"
    assert math.isclose(spain.ev_pct, 0.14 / 0.10 - 1.0)
    assert 0 < spain.stake <= 1000


def test_find_ev_bets_none_when_no_edge():
    poly = {"Spain": {"ask": 0.20, "yes_bid": 0.18, "no_ask": 0.80, "depth": 1000}}
    kalshi = {"Spain": {"ask": 0.21, "yes_bid": 0.19, "no_ask": 0.79, "depth": 0}}
    consensus = {"Spain": 0.15}            # both asks above fair -> no +EV YES buy
    assert find_ev_bets(poly, kalshi, consensus, poly_balance=1000,
                        kalshi_balance=1000, max_leg_stake=1000) == []
