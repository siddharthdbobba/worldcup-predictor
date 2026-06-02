from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    name: str          # canonical name
    rating: float      # international Elo


@dataclass(frozen=True)
class MatchModelParams:
    # scale was calibrated against the live 2026 field so the Elo favorite lands
    # at a realistic ~20% championship probability (scale=600 gave a nonsensical
    # ~53%). Higher scale => flatter, more upset-prone; lower => more top-heavy.
    # Still a coarse single-knob fit; refine against historical results in v2.
    base: float = 1.35              # baseline expected goals per team
    scale: float = 2000.0          # Elo->goals sensitivity (calibrated 2026-06)
    host_bump: float = 60.0        # Elo added to a host in its own match
    hosts: tuple[str, ...] = ("United States", "Canada", "Mexico")


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
