"""Errors whose message is written for the person running the app.

A ``SetupError`` is a first-run or environment problem the user can fix themselves (no config file, no
API key, no vector store). Its message is safe to show verbatim: it names the missing thing and the
command or setting that fixes it, and never carries a secret or a stack trace. The CLI prints it as
``error: ...`` and the web stream relays it as the ``error`` event, where any other exception becomes a
generic "something went wrong" because its text may be an internal detail.
"""


class SetupError(Exception):
    """A user-fixable setup problem; ``str(exc)`` is the message to show."""
