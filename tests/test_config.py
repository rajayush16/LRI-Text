from src.utils.config import load_config


def test_config_loads_and_has_required_languages():
    cfg = load_config()
    assert "hindi" in cfg["languages"]
    assert "bengali" in cfg["languages"]


def test_canonical_label_scheme_consistent():
    cfg = load_config()
    assert cfg["canonical_label_names"] == ["negative", "neutral", "positive"]
    for name, idx in cfg["canonical_name_to_id"].items():
        assert cfg["canonical_label_names"][idx] == name


def test_raw_label_maps_are_opposite_between_languages():
    # Regression test for the verified Hindi/Bengali label-order mismatch.
    cfg = load_config()
    hi_map = cfg["languages"]["hindi"]["raw_label_to_name"]
    bn_map = cfg["languages"]["bengali"]["raw_label_to_name"]
    assert hi_map != bn_map


def test_models_config_loads():
    models_cfg = load_config("models.yaml")
    for key in models_cfg["active_models"]:
        assert key in models_cfg["transformers"]
