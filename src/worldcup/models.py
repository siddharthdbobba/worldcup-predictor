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
    host_bump: float = 60.0        # Elo added to a host in ALL its matches (v1
                                   # simplification: per-match venue isn't modeled,
                                   # so neutral-site late-stage games get it too)
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


@dataclass
class ArbRec:
    """One sized risk-free lock: buy `contracts` YES on yes_venue and the same
    number of NO on no_venue. Payout is identical whoever wins."""
    team: str
    yes_venue: str          # "polymarket" | "kalshi"
    no_venue: str
    yes_ask: float          # a: price paid per YES contract
    no_ask: float           # b: price paid per NO contract
    contracts: int          # floored from the LP solution
    stake_yes: float
    stake_no: float
    total_cost: float       # stake_yes + stake_no + fees
    guaranteed_profit: float
    roc: float              # guaranteed_profit / total_cost
    annual_roc: float       # roc annualized to settlement


@dataclass
class EvBetRec:
    """A +EV (not risk-free) cross-book bet: the cheaper book underprices a team
    vs. the two-book consensus. Sized by fractional Kelly."""
    team: str
    venue: str
    side: str               # "YES"
    ask: float
    fair: float             # consensus probability
    ev_pct: float           # fair/ask - 1
    stake: float
    potential_profit: float


@dataclass
class DutchBook:
    """Whole-field check: buy the cheapest YES per team across both books."""
    field_sum: float        # Σ cheapest YES ask (incl. fee) over all teams
    gap: float              # 1 - field_sum  (positive => risk-free arb)
    is_arb: bool
    legs: list              # [(team, venue, price), ...]
