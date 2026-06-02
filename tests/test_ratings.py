import pytest

from worldcup.ratings import parse_ratings, parse_team_codes

# Minimal eloratings-shaped fixtures (tab-separated), matching the live format.
TEAMS_TSV = (
    "ES\tSpain\n"
    "AR\tArgentina\n"
    "US\tUnited States\n"
    "KR\tSouth Korea\n"
    "CD\tDR Congo\n"
)
WORLD_TSV = (
    "1\t1\tES\t2165\textra\n"
    "2\t2\tAR\t2113\textra\n"
    "3\t3\tUS\t1825\textra\n"
    "4\t4\tKR\t1740\textra\n"
    "5\t5\tCD\t1500\textra\n"
)


def test_parse_team_codes_uses_first_name_column():
    codes = parse_team_codes(TEAMS_TSV)
    assert codes["ES"] == "Spain"
    assert codes["KR"] == "South Korea"


def test_parse_ratings_joins_on_code_and_canonicalizes():
    codes = parse_team_codes(TEAMS_TSV)
    r = parse_ratings(WORLD_TSV, codes, min_count=2)
    assert r["Spain"] == 2165.0
    assert r["United States"] == 1825.0
    assert r["DR Congo"] == 1500.0
    assert isinstance(r["Argentina"], float)


def test_parse_ratings_rejects_implausible_layout():
    bad = "1\t1\tES\t50\tx\n2\t2\tAR\t40\tx\n"  # ratings far below plausible band
    with pytest.raises(ValueError):
        parse_ratings(bad, {"ES": "Spain", "AR": "Argentina"}, min_count=1)
