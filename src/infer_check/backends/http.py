"""Retrying POST shared by the HTTP backends."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

# Statuses worth retrying: the server is busy or restarting.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
# Seconds to wait before each retry. Module-level so tests can skip the waits.
RETRY_DELAYS: tuple[float, ...] = (0.5, 1.0, 2.0)


async def post(client: httpx.AsyncClient, path: str, payload: dict[str, Any]) -> httpx.Response:
    """POST ``payload``, retrying transient failures, then ``raise_for_status()``.

    Retried: timeouts, refused or dropped connections, and 429/5xx responses. A
    refused connection is retried because a server that is still starting refuses
    for a moment. Once the retries run out the last error propagates unchanged, so
    callers translate it as before; a refused connection then stops the run (see
    ``infer_check.errors``) and anything else counts as an error for this prompt.
    """
    delays = RETRY_DELAYS
    for attempt in range(len(delays) + 1):
        last = attempt == len(delays)
        try:
            response = await client.post(path, json=payload)
        except httpx.TransportError:
            if last:
                raise
        else:
            if last or response.status_code not in RETRY_STATUS:
                response.raise_for_status()
                return response
        await asyncio.sleep(delays[attempt])
    raise AssertionError("unreachable")
