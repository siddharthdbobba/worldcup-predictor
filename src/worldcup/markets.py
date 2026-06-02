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
    if isinstance(payload, list):
        if not payload:
            return {}
        event = payload[0]
    else:
        event = payload
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
    total_p = sum(prob.values())
    if total_p > 0:
        prob = {t: v / total_p for t, v in prob.items()}
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
