"""MediaWiki API client with etiquette baked in.

Every request carries a descriptive User-Agent, `format=json`, `formatversion=2`, and `maxlag`.
Requests are serial; the client sleeps `request_delay_seconds` after each success and backs off on
HTTP 429 / `maxlag` errors, honoring `Retry-After`. On exhausted retries it raises so the caller can
checkpoint and stop.
"""

import time

import requests


class WikiClient:
    def __init__(self, cfg: dict, session=None):
        self.base = cfg["base_url"]
        self.delay = cfg["request_delay_seconds"]
        self.maxlag = cfg["maxlag"]
        self.s = session if session is not None else requests.Session()
        self.s.headers["User-Agent"] = cfg["user_agent"]

    def get(self, params: dict, max_retries: int = 6) -> dict:
        params = {
            **params,
            "format": "json",
            "formatversion": 2,
            "maxlag": self.maxlag,
        }
        backoff = 3
        for _ in range(max_retries):
            r = self.s.get(self.base, params=params, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", backoff))
                time.sleep(wait)
                backoff *= 2
                continue
            data = r.json()
            if isinstance(data, dict) and data.get("error", {}).get("code") == "maxlag":
                time.sleep(backoff)
                backoff *= 2
                continue
            time.sleep(self.delay)
            return data
        raise RuntimeError("API retries exhausted")
