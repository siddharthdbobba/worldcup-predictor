import json
from pathlib import Path
from worldcup.ratings import parse_ratings

FIX = Path(__file__).parent / "fixtures" / "ratings_sample.json"


def test_parse_ratings_normalizes_names_and_floats():
    data = json.loads(FIX.read_text())
    out = parse_ratings(data)
    assert out["United States"] == 1825.0       # USA normalized
    assert out["South Korea"] == 1740.0         # Korea Republic normalized
    assert isinstance(out["France"], float)
