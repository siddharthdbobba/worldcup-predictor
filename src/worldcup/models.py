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
