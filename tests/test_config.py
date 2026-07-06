from zeod.config import DEFAULT_CONFIG_PATH, load_config


def test_default_config_loads():
    config = load_config()
    assert config.labeling.system_prompt
    assert config.dataset.head_crop_max_aspect_ratio == 1.8


def test_set_override_numeric():
    config = load_config(DEFAULT_CONFIG_PATH, overrides=["train.epochs=5"])
    assert config.train.epochs == 5


def test_set_override_float():
    config = load_config(DEFAULT_CONFIG_PATH, overrides=["dataset.val_split=0.3"])
    assert config.dataset.val_split == 0.3


def test_set_override_bool():
    config = load_config(DEFAULT_CONFIG_PATH, overrides=["llama_cpp.n_gpu_layers=-1"])
    assert config.llama_cpp.n_gpu_layers == -1


def test_set_override_null_clears_optional_field():
    config = load_config(DEFAULT_CONFIG_PATH, overrides=["dataset.head_crop_max_aspect_ratio=null"])
    assert config.dataset.head_crop_max_aspect_ratio is None


def test_set_override_invalid_format_raises():
    import pytest

    with pytest.raises(ValueError):
        load_config(DEFAULT_CONFIG_PATH, overrides=["not-a-key-value-pair"])
