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


def test_validate_rejects_wrong_group_count():
    with pytest.raises(ValueError):
        validate_draw({"A": ["a", "b", "c", "d"]})


def test_validate_rejects_short_group():
    bad = {chr(65 + i): ["a", "b", "c", "d"] for i in range(12)}
    bad["A"] = ["a", "b", "c"]                   # only 3
    with pytest.raises(ValueError):
        validate_draw(bad)
