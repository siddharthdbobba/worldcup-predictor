# src/worldcup/blend.py
"""Pure liquidity-weighted blend of model% and market% into the final forecast."""
from __future__ import annotations


def market_weight(liquidity: float, w_cap: float = 0.7, K: float = 0.5) -> float:
    """Market blend weight for a priced team: w_cap * L/(L+K), clamped at L>=0.

    `liquidity` here is the market's normalized confidence (sum of the two books'
    [0,1] volumes, range [0,2]); K=0.5 is the blend's default half-saturation.
    """
    L = max(0.0, liquidity)
    return w_cap * (L / (L + K)) if (L + K) > 0 else 0.0


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
                w = market_weight(liquidity.get(team, 0.0), w_cap=w_cap, K=K)
            out[team] = w * market[team] + (1 - w) * m
        else:
            out[team] = m
    total = sum(out.values())
    if total <= 0:
        return out
    return {t: v / total for t, v in out.items()}
