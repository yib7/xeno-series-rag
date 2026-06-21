from xeno_rag.config import load_config


def test_load_config_reads_expected_keys():
    cfg = load_config("config.yaml")
    assert cfg["base_url"].endswith("/api.php")
    assert cfg["batch_size"] == 50
    assert cfg["maxlag"] == 5
    assert isinstance(cfg["paths"], dict)
    assert "titles" in cfg["paths"]
    assert "vectorstore" in cfg["paths"]


def test_load_config_missing_file_raises():
    import pytest

    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")
