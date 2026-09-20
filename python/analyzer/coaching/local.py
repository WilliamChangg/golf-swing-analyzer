"""Talking to a language model running on this machine, and nothing else.

One provider, one request shape, one hard constraint: **the endpoint must be
loopback**. A configuration pointing anywhere else is refused rather than
followed, because "local-first" is either enforced somewhere or it is a sentence
in a README. The check is here, at the only place in this system that opens a
socket for a finding.

The request shape is the one a local runner exposes for a single completion --
a JSON body carrying the model name and the prompt, a JSON reply carrying the
text. Written against the standard library rather than a client package: the
whole exchange is one POST, and a dependency whose only job is to construct one
would be a dependency to audit for no gain.

Nothing about this module has been exercised against a real model in this
repository. See `phrasing.py` for what that means for the numbers it reports.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from analyzer.coaching.phrasing import PhrasingError
from analyzer.contracts.coaching import Finding

_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


class LocalPhrasing:
    """A `PhrasingProvider` backed by a model server on this machine."""

    def __init__(self, endpoint: str, model: str | None, timeout_s: float) -> None:
        self._endpoint = _require_loopback(endpoint)
        self._model = model or "llama3"
        self._timeout_s = timeout_s

    @property
    def name(self) -> str:
        return f"local:{self._model}"

    def propose(self, finding: Finding, prompt: str) -> str | None:
        """One completion, or None if the server answered with nothing usable.

        A transport failure raises rather than returning None. The two are
        different facts -- a model that declined to answer and a model that was
        never reached -- and `PhrasingReport` records them in different fields.
        """
        del finding  # the prompt is the whole of what is sent
        body = json.dumps(
            {"model": self._model, "prompt": prompt, "stream": False},
        ).encode()
        request = urllib.request.Request(  # noqa: S310 - scheme checked in _require_loopback
            self._endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:  # noqa: S310
                payload = json.loads(response.read().decode())
        except urllib.error.URLError as exc:
            raise PhrasingError(
                f"No local model answered at {self._endpoint}: {exc.reason}.",
                "Start the model server, or leave phrasing off -- every finding "
                "already carries the engine's own sentence.",
            ) from exc
        except json.JSONDecodeError as exc:
            raise PhrasingError(
                f"The server at {self._endpoint} replied with something that is not JSON."
            ) from exc

        text = payload.get("response")
        if isinstance(text, str) and text.strip():
            return text.strip()
        return None


def _require_loopback(endpoint: str) -> str:
    """Refuse any endpoint that is not on this machine.

    Checked on the parsed host rather than by searching the string, so that a
    URL whose path or query happens to contain "localhost" cannot pass. A
    hostname that resolves to a loopback address elsewhere in DNS is not
    accepted either: the allowed set is the three literals, which is the only
    form that cannot be changed by something outside this process.
    """
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in {"http", "https"}:
        raise PhrasingError(f"A phrasing endpoint must be an http or https URL; got {endpoint!r}.")
    if (parsed.hostname or "") not in _LOOPBACK:
        raise PhrasingError(
            f"{endpoint} is not on this machine. The phrasing layer is handed a "
            "measurement of somebody's body, and this engine sends that to "
            "loopback or to nowhere.",
            "Run the model locally and point the endpoint at 127.0.0.1.",
        )
    return endpoint
