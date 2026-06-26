from brittle_user_tokens.utils.config import get, load_config


def test_load_and_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("data:\n  n_train: 100\nmodel:\n  name: foo\n", encoding="utf-8")
    cfg = load_config(str(p), ["data.n_train=5", "model.name=bar", "train.lr=1e-5"])
    assert get(cfg, "data.n_train") == 5          # coerced to int
    assert isinstance(get(cfg, "data.n_train"), int)
    assert get(cfg, "model.name") == "bar"
    assert get(cfg, "train.lr") == 1e-5            # coerced to float
    assert get(cfg, "missing.key", "default") == "default"
