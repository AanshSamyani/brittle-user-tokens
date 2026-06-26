from brittle_user_tokens.utils.naming import gen_name, parse_name, run_name


def test_run_name_roundtrip():
    p = parse_name(run_name("arc_sycophancy", "polite_deferential", 3))
    assert p == {"dataset": "arc_sycophancy", "arm": "polite_deferential", "seed": 3, "test_axis": None}


def test_gen_name_roundtrip():
    name = gen_name("arc_sycophancy", "assertive", 0, "warm") + ".jsonl"
    p = parse_name(name)
    assert p["dataset"] == "arc_sycophancy"
    assert p["arm"] == "assertive"
    assert p["seed"] == 0
    assert p["test_axis"] == "warm"
