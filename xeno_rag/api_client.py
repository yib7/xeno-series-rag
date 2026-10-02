"""MediaWiki API client that follows the wiki's request etiquette.

Every request carries a descriptive User-Agent, `format=json`, `formatversion=2`, and `maxlag`.
Requests are serial; the client sleeps `request_delay_seconds` after each success and backs off on
HTTP 429 / 5xx / `maxlag` errors and transient network failures (connection reset, timeout),
honoring `Retry-After` when it is delta-seconds (an HTTP-date header falls back to the computed
backoff) and clamping every wait to MAX_RETRY_WAIT_SECONDS. Terminal 4xx errors are raised via
`raise_for_status`. On exhausted retries it raises so the caller can checkpoint and stop.
"""

import time

import requests

# Never sleep longer than this on a single retry, whatever Retry-After says. A misbehaving (or
# malicious) header must not park a 19h pull for hours; the wiki sheds load through maxlag.
MAX_RETRY_WAIT_SECONDS = 120


def _retry_wait(retry_after, backoff: float) -> float:
    """Seconds to wait before the next retry, clamped to MAX_RETRY_WAIT_SECONDS.

    ``Retry-After`` may legally be an HTTP-date rather than delta-seconds; parsing it as a date is
    not worth the complexity for one wiki, so any non-integer header falls back to the computed
    exponential backoff instead of crashing a multi-hour pull on a header format."""
    if retry_after is not None:
        try:
            wait = int(retry_after)
        except (TypeError, ValueError):
            wait = backoff
    else:
        wait = backoff
    return min(wait, MAX_RETRY_WAIT_SECONDS)


class RetriesExhausted(RuntimeError):
    """Every retry hit a transient failure (timeout, 429, 5xx, bad body, maxlag). A RuntimeError
    subclass, so existing ``except RuntimeError`` callers are unaffected; fetchers catch it by type
    to tag the page as retryable rather than as a permanent error."""


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
            # A transient network failure (connection reset, DNS blip, timeout) gets the same
            # backoff treatment as a 5xx: long unattended pulls should degrade to a wait, not die
            # on the first blip. HTTPError from raise_for_status below is deliberately NOT caught
            # here. Terminal 4xx must still surface immediately.
            try:
                r = self.s.get(self.base, params=params, timeout=30)
            except requests.RequestException:
                time.sleep(_retry_wait(None, backoff))
                backoff *= 2
                continue
            # Back off on rate-limit (429) and transient server errors (5xx) alike,
            # honoring Retry-After when present. A 5xx may carry a non-JSON body
            # (e.g. an HTML gateway page), so retry before ever calling r.json().
            if r.status_code == 429 or 500 <= r.status_code < 600:
                time.sleep(_retry_wait(r.headers.get("Retry-After"), backoff))
                backoff *= 2
                continue
            # Surface terminal 4xx clearly instead of returning it as a success dict.
            r.raise_for_status()
            # A 200 OK can still carry a non-JSON body (captive portal, corporate-proxy
            # interstitial, Cloudflare challenge page). json.JSONDecodeError is a ValueError
            # subclass, not a requests.RequestException, so it would otherwise escape uncaught
            # and abort a multi-hour pull; treat it as transient like the other backoff paths.
            try:
                data = r.json()
            except ValueError:
                time.sleep(_retry_wait(None, backoff))
                backoff *= 2
                continue
            if isinstance(data, dict) and data.get("error", {}).get("code") == "maxlag":
                time.sleep(_retry_wait(None, backoff))
                backoff *= 2
                continue
            time.sleep(self.delay)
            return data
        raise RetriesExhausted("API retries exhausted")
