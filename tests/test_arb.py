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
