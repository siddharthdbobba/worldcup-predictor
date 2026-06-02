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
