"""Fetch + parse Polymarket and Kalshi outright-winner markets.

Per team we extract: implied probability (vig-stripped at combine time), the ask
you would pay, traded **volume** (the cross-book confidence signal — both books
populate it; Kalshi's liquidity field is 0 in practice), and dollar **depth**
(order-book liquidity, used only as a stake-size cap).

`combine_markets` intersects with the tournament's actual teams (so stale markets
for non-qualified teams, which can carry a bogus price of 1.0, don't corrupt the
de-vig), de-vigs each book, then weights the two into one consensus by each book's
volume normalized to [0, 1] (scale-free, so the two books' differing units don't
matter). The returned `confidence` is the sum of the two normalized volumes
(range [0, 2]) and feeds the blend's market weight.
"""
from __future__ import annotations

import httpx

from worldcup.teamnames import canonical

POLYMARKET_URL = "https://gamma-api.polymarket.com/events?slug=world-cup-winner"
KALSHI_URL = ("https://api.elections.kalshi.com/trade-api/v2/markets"
              "?series_ticker=KXMENWORLDCUP&limit=200&status=open")


def _f(value) -> float:
    """Best-effort float; missing/blank/None -> 0.0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_polymarket(payload) -> dict[str, dict]:
    """Parse a gamma-api event payload into {team: {prob, ask, vol, depth}}."""
    if isinstance(payload, list):
        if not payload:
            return {}
        event = payload[0]
    else:
        event = payload
    out: dict[str, dict] = {}
    for m in event.get("markets", []):
        title = m.get("groupItemTitle")
        price = m.get("lastTradePrice")
        if not title or price is None:
            continue
        bid = _f(m.get("bestBid"))
        out[canonical(title)] = {
            "prob": _f(price),
            "ask": _f(m.get("bestAsk") or price),
            "yes_bid": bid,
            "no_ask": (1.0 - bid) if bid > 0 else 0.0,  # derived; Poly has no direct NO quote
            "vol": _f(m.get("volumeNum")),
            "depth": _f(m.get("liquidityNum")),
        }
    return out


def parse_kalshi(payload: dict) -> dict[str, dict]:
    """Parse a Kalshi markets payload (dollar prices, 0-1) into {team: {...}}."""
    out: dict[str, dict] = {}
    for m in payload.get("markets", []):
        title = m.get("yes_sub_title")
        price = m.get("last_price_dollars")
        if not title or price is None:
            continue
        out[canonical(title)] = {
            "prob": _f(price),
            "ask": _f(m.get("yes_ask_dollars") or price),
            "yes_bid": _f(m.get("yes_bid_dollars")),
            "no_ask": _f(m.get("no_ask_dollars")),
            "no_bid": _f(m.get("no_bid_dollars")),
            "vol": _f(m.get("volume_fp")),
            "depth": _f(m.get("liquidity_dollars")),
        }
    return out


def _devig(probs: dict[str, float]) -> dict[str, float]:
    """Normalize a single book's probabilities to sum to 1 (strip the overround)."""
    total = sum(probs.values())
    if total <= 0:
        return {t: 0.0 for t in probs}
    return {t: p / total for t, p in probs.items()}


def _normalize(values: dict[str, float]) -> dict[str, float]:
    """Scale values to [0, 1] by the max (scale-free confidence per book)."""
    top = max(values.values()) if values else 0.0
    if top <= 0:
        return {t: 0.0 for t in values}
    return {t: v / top for t, v in values.items()}


def combine_markets(poly: dict[str, dict], kalshi: dict[str, dict],
                    valid_teams=None):
    """Combine the two books into (prob, ask, confidence, depth) dicts.

    `valid_teams`, when given, restricts BOTH books to those teams before
    de-vigging (keeps stale non-qualified markets out of the normalization).
    """
    if valid_teams is not None:
        vt = set(valid_teams)
        poly = {t: v for t, v in poly.items() if t in vt}
        kalshi = {t: v for t, v in kalshi.items() if t in vt}

    poly_p = _devig({t: v["prob"] for t, v in poly.items()})
    kalshi_p = _devig({t: v["prob"] for t, v in kalshi.items()})
    poly_c = _normalize({t: v["vol"] for t, v in poly.items()})
    kalshi_c = _normalize({t: v["vol"] for t, v in kalshi.items()})

    teams = set(poly) | set(kalshi)
    prob, ask, confidence, depth = {}, {}, {}, {}
    for t in teams:
        wp, wk = poly_c.get(t, 0.0), kalshi_c.get(t, 0.0)
        wsum = wp + wk
        if wsum > 0:
            prob[t] = (wp * poly_p.get(t, 0.0) + wk * kalshi_p.get(t, 0.0)) / wsum
        else:
            avail = [p for p in (poly_p.get(t), kalshi_p.get(t)) if p]
            prob[t] = sum(avail) / len(avail) if avail else 0.0
        asks = [v["ask"] for v in (poly.get(t), kalshi.get(t)) if v and v["ask"] > 0]
        ask[t] = min(asks) if asks else 1.0
        confidence[t] = wp + wk  # [0, 2]
        depth[t] = poly.get(t, {}).get("depth", 0.0) + kalshi.get(t, {}).get("depth", 0.0)

    total_p = sum(prob.values())
    if total_p > 0:
        prob = {t: p / total_p for t, p in prob.items()}
    return prob, ask, confidence, depth


def fetch_market_probabilities(valid_teams=None, timeout: float = 20.0):
    """Fetch both books and combine. Returns (prob, ask, confidence, depth).

    If one source fails, proceed with the other. Raises only if BOTH fail.
    """
    poly, kalshi, errors = {}, {}, []
    try:
        r = httpx.get(POLYMARKET_URL, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
        poly = parse_polymarket(r.json())
    except Exception as e:  # noqa: BLE001 - degrade gracefully, record reason
        errors.append(f"polymarket: {e}")
    try:
        r = httpx.get(KALSHI_URL, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
        kalshi = parse_kalshi(r.json())
    except Exception as e:  # noqa: BLE001
        errors.append(f"kalshi: {e}")
    if not poly and not kalshi:
        raise RuntimeError("both market sources failed: " + "; ".join(errors))
    return combine_markets(poly, kalshi, valid_teams)


def fetch_books(valid_teams=None, timeout: float = 20.0):
    """Fetch BOTH books as separate raw dicts for cross-book arbitrage.

    Unlike `fetch_market_probabilities`, arb needs both venues, so this raises if
    either is missing. Optionally restricts to `valid_teams` (drops stale
    non-qualified markets that can carry a bogus 1.0 price).
    """
    poly, kalshi, errors = {}, {}, []
    try:
        r = httpx.get(POLYMARKET_URL, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
        poly = parse_polymarket(r.json())
    except Exception as e:  # noqa: BLE001
        errors.append(f"polymarket: {e}")
    try:
        r = httpx.get(KALSHI_URL, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
        kalshi = parse_kalshi(r.json())
    except Exception as e:  # noqa: BLE001
        errors.append(f"kalshi: {e}")
    if not poly or not kalshi:
        raise RuntimeError("cross-book arbitrage needs BOTH books; "
                           + ("; ".join(errors) or "one book returned no markets"))
    if valid_teams is not None:
        vt = set(valid_teams)
        poly = {t: v for t, v in poly.items() if t in vt}
        kalshi = {t: v for t, v in kalshi.items() if t in vt}
    return poly, kalshi
