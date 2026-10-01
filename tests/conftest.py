"""Suite-wide isolation: no test may see a real API key or reload the developer's ``.env``.

``load_config()`` (called by ``cli.main()``, ``load_config("config.yaml")`` in several tests, and the
eval ``main()``s) runs ``load_env()``, which copies a local ``.env`` into ``os.environ`` for the rest of
the process. On a machine with a real ``.env`` that leaks a live ``TYPESAFE_API_KEY`` / Gemini key into
every later test, so any test that routes or generates without its own stub would make a paid call, and
the result would differ between a developer machine and CI. This fixture blanks the keys before each
test and turns ``load_env`` into a no-op for ``load_config``; tests that exercise ``load_env`` itself
call the function they imported directly, so they are unaffected.
"""

import pytest

_SECRET_ENV_VARS = ("TYPESAFE_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "HF_TOKEN",
                    "HUGGING_FACE_HUB_TOKEN")


@pytest.fixture(autouse=True)
def _hermetic_environment(monkeypatch):
    for name in _SECRET_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("xeno_rag.config.load_env", lambda path=".env": None)
