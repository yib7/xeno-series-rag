def test_package_imports():
    from importlib.metadata import version

    import xeno_rag

    # Single-sourced from installed metadata (see test_version.py); not the stale scaffold value.
    assert xeno_rag.__version__ == version("xeno-rag")
