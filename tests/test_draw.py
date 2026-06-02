import json
import pytest
from pathlib import Path
from worldcup.draw import parse_draw, validate_draw

FIX = Path(__file__).parent / "fixtures" / "draw_sample.json"


def test_parse_and_validate_ok():
    groups = parse_draw(json.loads(FIX.read_text()))
    validate_draw(groups)                       # must not raise
    assert len(groups) == 12
    assert all(len(v) == 4 for v in groups.values())
    assert groups["C"][0] == "United States"    # normalized


def test_parse_draw_from_kalshi_markets_form():
    payload = {"markets": [
        {"event_ticker": "KXWCGROUPWIN-26A", "yes_sub_title": "Mexico"},
        {"event_ticker": "KXWCGROUPWIN-26A", "yes_sub_title": "Korea Republic"},
        {"event_ticker": "KXWCGROUPWIN-26B", "yes_sub_title": "USA"},
    ]}
    groups = parse_draw(payload)
    assert groups["A"] == ["Mexico", "South Korea"]   # grouped by letter, canonicalized
    assert groups["B"] == ["United States"]


def test_validate_rejects_wrong_group_count():
    with pytest.raises(ValueError):
        validate_draw({"A": ["a", "b", "c", "d"]})


def test_validate_rejects_short_group():
    bad = {chr(65 + i): ["a", "b", "c", "d"] for i in range(12)}
    bad["A"] = ["a", "b", "c"]                   # only 3
    with pytest.raises(ValueError):
        validate_draw(bad)


def test_validate_rejects_none_slot():
    bad = {chr(65 + i): ["a", "b", "c", "d"] for i in range(12)}
    bad["A"] = ["a", "b", "c", None]
    with pytest.raises(ValueError):
        validate_draw(bad)


def test_validate_rejects_duplicate_teams():
    bad = {chr(65 + i): [f"{chr(65+i)}{j}" for j in range(4)] for i in range(12)}
    bad["B"][0] = "A0"  # duplicate of a team in group A
    with pytest.raises(ValueError):
        validate_draw(bad)
