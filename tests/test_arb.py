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


from worldcup.arb import find_locks, VENUE_POLY, VENUE_KALSHI, _fee


def test_fee_is_ceiled_up_to_the_cent():
    """_fee must round UP to the nearest cent (conservative: never understate)."""
    # 0.07 * 0.71 * 0.29 = 0.014441 → ceil to 0.02
    assert math.isclose(_fee(0.07, 0.71), 0.02), (
        f"Expected 0.02 (ceiled cent), got {_fee(0.07, 0.71)}"
    )
    # zero rate → zero fee regardless of price
    assert _fee(0.0, 0.40) == 0.0
    # result must always be >= the raw continuous value (never understates)
    raw = 0.07 * 0.5 * 0.5
    assert _fee(0.07, 0.5) >= raw

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
    # raw 1-0.18-0.71 = 0.11; kalshi fee ceil(0.07*0.71*0.29*100)/100 = 0.02; profit = 0.09
    assert math.isclose(france.profit, 0.09)


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
    assert math.isclose(spain.stake, 22.22, rel_tol=1e-3)
    assert math.isclose(spain.potential_profit, 199.98, rel_tol=1e-3)


def test_find_ev_bets_none_when_no_edge():
    poly = {"Spain": {"ask": 0.20, "yes_bid": 0.18, "no_ask": 0.80, "depth": 1000}}
    kalshi = {"Spain": {"ask": 0.21, "yes_bid": 0.19, "no_ask": 0.79, "depth": 0}}
    consensus = {"Spain": 0.15}            # both asks above fair -> no +EV YES buy
    assert find_ev_bets(poly, kalshi, consensus, poly_balance=1000,
                        kalshi_balance=1000, max_leg_stake=1000) == []


from worldcup.arb import dutch_book


def test_dutch_book_flags_when_field_below_one():
    # Two-team toy world; cheapest YES per team sums under 1.
    poly = {"A": {"ask": 0.40}, "B": {"ask": 0.55}}
    kalshi = {"A": {"ask": 0.45}, "B": {"ask": 0.50}}
    db = dutch_book(poly, kalshi, kalshi_fee_rate=0.0, poly_fee_rate=0.0)
    # min(0.40,0.45)=0.40 (poly A) + min(0.55,0.50)=0.50 (kalshi B) = 0.90
    assert math.isclose(db.field_sum, 0.90)
    assert math.isclose(db.gap, 0.10) and db.is_arb is True
    assert ("A", "polymarket", 0.40) in db.legs and ("B", "kalshi", 0.50) in db.legs


def test_dutch_book_not_arb_for_normal_field():
    poly = {"A": {"ask": 0.60}, "B": {"ask": 0.60}}
    kalshi = {"A": {"ask": 0.62}, "B": {"ask": 0.61}}
    db = dutch_book(poly, kalshi, kalshi_fee_rate=0.0, poly_fee_rate=0.0)
    assert db.field_sum > 1.0 and db.is_arb is False


def test_dutch_book_fee_flips_venue_choice():
    """Fee-adjusted cost determines venue selection, not raw ask.

    Single-team world, team "A":
      Poly:   ask=0.40, poly_fee_rate=0.0
              cost = 0.40 + 0.0*0.40*0.60 = 0.40
      Kalshi: ask=0.39, kalshi_fee_rate=0.50
              fee  = ceil(0.50 * 0.39 * 0.61 * 100) / 100
                   = ceil(11.895) / 100 = 12 / 100 = 0.12
              cost = 0.39 + 0.12 = 0.51

    Kalshi has the cheaper RAW ask (0.39 < 0.40), but after fees its cost
    (0.51) exceeds Poly's (0.40), so Poly must be chosen.
    field_sum = 0.40 (exact, since poly_fee_rate=0).
    """
    poly = {"A": {"ask": 0.40}}
    kalshi = {"A": {"ask": 0.39}}
    db = dutch_book(poly, kalshi, kalshi_fee_rate=0.50, poly_fee_rate=0.0)

    # Poly cost:   0.40 + 0.0*0.40*0.60 = 0.40
    # Kalshi cost: 0.39 + ceil(0.50*0.39*0.61*100)/100 = 0.39 + 0.12 = 0.51
    # Poly is cheaper after fees despite the higher raw ask.
    assert ("A", "polymarket", 0.40) in db.legs
    assert math.isclose(db.field_sum, 0.40, rel_tol=1e-9)


from worldcup.arb import run_arb, ArbResult


def test_run_arb_combines_all_tiers():
    consensus = {"France": 0.22, "Spain": 0.13, "Brazil": 0.205}
    res = run_arb(POLY, KALSHI, consensus, poly_balance=10_000,
                  kalshi_balance=10_000, max_leg_stake=1000.0,
                  days_to_settlement=47)
    assert isinstance(res, ArbResult)
    assert any(r.team == "France" for r in res.locks)     # France lock sized
    assert isinstance(res.ev_bets, list)
    assert res.dutch is not None


def test_run_arb_can_disable_extras():
    res = run_arb(POLY, KALSHI, {"France": 0.22}, poly_balance=10_000,
                  kalshi_balance=10_000, max_leg_stake=1000.0,
                  days_to_settlement=47, enable_ev=False, enable_dutch=False)
    assert res.ev_bets == [] and res.dutch is None


def test_arb_cli_rejects_bad_settlement_date(monkeypatch):
    import sys
    import main as cli
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--arb",
        "--poly-balance", "100",
        "--kalshi-balance", "100",
        "--settlement-date", "not-a-date",
    ])
    import pytest
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert "YYYY-MM-DD" in str(exc_info.value)


def test_run_arb_cli_prints_card(monkeypatch, capsys):
    import main as cli
    monkeypatch.setattr(cli, "fetch_group_draw", lambda: {"A": ["France", "Spain", "Brazil", "Japan"]})
    monkeypatch.setattr(cli, "fetch_books", lambda valid_teams=None: (POLY, KALSHI))
    monkeypatch.setattr(cli, "fetch_market_probabilities",
                        lambda valid_teams=None: ({"France": 0.22, "Spain": 0.13, "Brazil": 0.205},
                                                  {}, {}, {}))
    cli.run_arb_cli(poly_balance=10_000, kalshi_balance=10_000, kalshi_fee_rate=0.07,
                    poly_fee_rate=0.0, min_profit=0.02, max_leg_stake=1000.0,
                    days_to_settlement=47, kelly_fraction=0.5,
                    enable_ev=True, enable_dutch=True)
    out = capsys.readouterr().out
    assert "France" in out


# ---------------------------------------------------------------------------
# New tests for capital over-allocation bug fix
# ---------------------------------------------------------------------------

def test_find_ev_bets_depletes_balance_across_multiple_bets():
    """Two +EV teams both cheapest on Polymarket; at kelly_fraction=1.0 with
    strong edges (Σkelly > 1), the pre-fix code lets both bets size against the
    full balance, producing a total stake > poly_balance.  The fix must cap the
    second bet to the remaining balance after the first, so the sum ≤ poly_balance.

    team "Alpha": ask=0.05, consensus=0.60  → kelly=(0.60-0.05)/(1-0.05)≈0.5789
    team "Beta":  ask=0.10, consensus=0.65  → kelly=(0.65-0.10)/(1-0.10)≈0.6111
    Σkelly ≈ 1.19 → pre-fix total stake ≈ 119 on a $100 balance (BAD).
    EV ranks by p/ask-1: Alpha 0.60/0.05-1=11.0 > Beta 0.65/0.10-1=5.5, so Alpha
    sizes FIRST. Post-fix: Alpha stakes 0.5789*100≈57.89 (leaving ≈42.11); Beta
    sizes against the remainder → 0.6111*42.11≈25.73; total ≈83.62 ≤ 100.
    """
    poly = {
        "Alpha": {"ask": 0.05},
        "Beta":  {"ask": 0.10},
    }
    kalshi = {}  # neither team on Kalshi → both cheapest on Poly
    consensus = {"Alpha": 0.60, "Beta": 0.65}

    poly_balance = 100.0
    bets = find_ev_bets(poly, kalshi, consensus,
                        poly_balance=poly_balance, kalshi_balance=1000.0,
                        max_leg_stake=math.inf, kelly_fraction=1.0)

    poly_bets = [b for b in bets if b.venue == VENUE_POLY]
    assert len(poly_bets) == 2, "expected both Alpha and Beta to be +EV on Poly"
    total_poly_stake = sum(b.stake for b in poly_bets)
    assert total_poly_stake <= poly_balance + 1e-6, (
        f"Poly stakes {total_poly_stake:.4f} exceed poly_balance {poly_balance}"
    )


def test_find_ev_bets_single_bet_stake_unchanged():
    """Single-bet regression: with only one +EV team, the computed stake must
    equal the pre-fix value max(0, kelly)*kelly_fraction*balance, capped by
    max_leg_stake. The depletion logic must not alter a lone bet's sizing.

    Spain: ask=0.10, consensus=0.14, kelly_fraction=0.5, poly_balance=1000
    kelly = (0.14-0.10)/(1-0.10) = 0.04/0.90 ≈ 0.04444
    stake = 0.04444 * 0.5 * 1000 ≈ 22.22
    """
    poly = {"Spain": {"ask": 0.10}}
    kalshi = {}
    consensus = {"Spain": 0.14}
    kelly_fraction = 0.5
    poly_balance = 1000.0

    bets = find_ev_bets(poly, kalshi, consensus,
                        poly_balance=poly_balance, kalshi_balance=1000.0,
                        max_leg_stake=math.inf, kelly_fraction=kelly_fraction)

    assert len(bets) == 1
    spain = bets[0]
    kelly = (0.14 - 0.10) / (1.0 - 0.10)
    expected_stake = round(kelly * kelly_fraction * poly_balance, 2)
    assert math.isclose(spain.stake, expected_stake, rel_tol=1e-6), (
        f"Single-bet stake {spain.stake} ≠ expected {expected_stake}"
    )


def test_run_arb_lock_plus_ev_within_venue_balance():
    """End-to-end: France lock commits Poly capital (YES leg on Poly @0.18),
    then Brazil gets a Poly +EV bet.  Total Poly commitment must stay ≤ poly_balance.

    We use a SMALL poly_balance so that without the fix, the EV bet would size
    against the full balance and the combined spend would exceed it.

    France (and Spain) lock via YES-on-Poly, so the LP spends nearly the whole
    $200 Poly balance on locks. Pre-fix, the Brazil +EV bet then sizes against the
    FULL $200 (0.25*200 = $50), pushing combined Poly spend to ~$250 > $200 (BAD).
    Post-fix, Brazil sizes against the residual budget left after locks, so
    lock+EV ≤ $200.

    Note: the residual here is only a few cents, so Brazil's EV stake is tiny —
    the INVARIANT (lock+EV ≤ balance), not the EV magnitude, is what this asserts.
    Then assert:
        poly_lock_spend + poly_ev_spend ≤ poly_balance + 1e-6
    and that at least one EV bet on Poly was generated.
    """
    import math as _math
    from worldcup.arb import run_arb, VENUE_POLY

    # Keep existing POLY/KALSHI book fixtures (France locks, Spain locks, Brazil EV-only).
    # Brazil is not a lock: a=0.20, b=0.80, profit≈0 after fees → excluded from locks.
    # With consensus[Brazil]=0.40 >> 0.20 ask, it's a strong +EV Poly bet.
    consensus = {"France": 0.22, "Spain": 0.13, "Brazil": 0.40}
    poly_balance = 200.0
    kalshi_balance = 10_000.0

    res = run_arb(POLY, KALSHI, consensus,
                  poly_balance=poly_balance,
                  kalshi_balance=kalshi_balance,
                  max_leg_stake=1000.0,
                  days_to_settlement=47,
                  kelly_fraction=1.0,   # amplify stakes to stress-test
                  enable_ev=True, enable_dutch=False)

    # Compute actual Poly spend from locks
    from worldcup.arb import find_locks, VENUE_POLY as VP
    cands = find_locks(POLY, KALSHI, min_profit=0.02, max_leg_stake=1000.0)
    cand_by_team = {c.team: c for c in cands}

    poly_lock_spend = 0.0
    for rec in res.locks:
        cand = cand_by_team.get(rec.team)
        if cand is None:
            continue
        leg_stake = rec.stake_yes if rec.yes_venue == VP else rec.stake_no
        fee_poly = rec.contracts * cand.fee_poly
        poly_lock_spend += leg_stake + fee_poly

    poly_ev_spend = sum(b.stake for b in res.ev_bets if b.venue == VP)

    # Core invariant: combined spend must not exceed the balance
    total = poly_lock_spend + poly_ev_spend
    assert total <= poly_balance + 1e-6, (
        f"Total Poly spend {total:.4f} exceeds balance {poly_balance}: "
        f"lock={poly_lock_spend:.4f}, ev={poly_ev_spend:.4f}"
    )

    # Verify both tiers actually fired on Poly (otherwise the test proves nothing)
    assert poly_lock_spend > 0, "Expected at least one Poly lock to fund"
    assert poly_ev_spend > 0, "Expected at least one Poly EV bet to fund"
