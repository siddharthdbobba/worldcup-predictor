from worldcup.teamnames import canonical


def test_known_aliases_map():
    assert canonical("USA") == "United States"
    assert canonical("United States") == "United States"
    assert canonical("Korea Republic") == "South Korea"
    assert canonical("south korea") == "South Korea"
    assert canonical("IR Iran") == "Iran"


def test_unknown_name_is_trimmed_passthrough():
    assert canonical("  France ") == "France"
