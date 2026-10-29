"""HTTP backends retry transient failures before counting a prompt as failed."""

import asyncio
from collections.abc import Callable

import httpx
import pytest

from infer_check.backends import http
from infer_check.backends.openai_compat import OpenAICompatBackend
from infer_check.errors import systemic_reason
from infer_check.types import Prompt

OK = {"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}]}


def _client(responses: list[Callable[[httpx.Request], httpx.Response]]) -> tuple[httpx.AsyncClient, list[int]]:
    """A client whose Nth request is answered by ``responses[N]`` (the last repeats)."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return responses[min(len(calls), len(responses)) - 1](request)

    return httpx.AsyncClient(base_url="http://server", transport=httpx.MockTransport(handler)), calls


def _status(code: int) -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(code, json=OK if code == 200 else {"error": "x"})


def _raise(exc: type[httpx.TransportError]) -> Callable[[httpx.Request], httpx.Response]:
    def respond(request: httpx.Request) -> httpx.Response:
        raise exc("transient", request=request)

    return respond


@pytest.mark.parametrize("first", [_status(503), _status(429), _raise(httpx.ReadTimeout), _raise(httpx.ReadError)])
def test_transient_failure_then_success(first: Callable[[httpx.Request], httpx.Response]) -> None:
    client, calls = _client([first, _status(200)])
    response = asyncio.run(http.post(client, "/v1/chat/completions", {}))
    assert response.status_code == 200 and len(calls) == 2


def test_client_errors_are_not_retried() -> None:
    client, calls = _client([_status(404)])
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(http.post(client, "/v1/chat/completions", {}))
    assert len(calls) == 1


def test_gives_up_after_the_retries() -> None:
    client, calls = _client([_status(503)])
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(http.post(client, "/v1/chat/completions", {}))
    assert len(calls) == 1 + len(http.RETRY_DELAYS)


def test_refused_connection_after_retries_stops_the_run() -> None:
    backend = OpenAICompatBackend(base_url="http://server", model_id="m")
    backend._client, calls = _client([_raise(httpx.ConnectError)])
    with pytest.raises(RuntimeError, match="Cannot connect") as exc_info:
        asyncio.run(backend.generate(Prompt(id="p0", text="2+2?")))
    assert len(calls) == 1 + len(http.RETRY_DELAYS)
    assert systemic_reason(exc_info.value) == "the server refused the connection"
