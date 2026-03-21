"""Generic OpenAI-compatible backend adapter.

Works with any server that implements the ``/v1/completions`` endpoint
(vLLM, SGLang, Ollama, vllm-mlx, etc.).
"""

from __future__ import annotations

import asyncio
import math
import time

import httpx

from infer_check.types import InferenceResult, Prompt

__all__ = ["OpenAICompatBackend"]


class OpenAICompatBackend:
    """Backend adapter for any OpenAI-compatible completion server.

    Supports both ``/v1/completions`` (raw) and ``/v1/chat/completions``
    (chat-formatted) endpoints.  Use ``chat=True`` for instruct/chat models
    served by vLLM, SGLang, vllm-mlx, Ollama, etc. — this ensures the
    server applies the model's chat template, matching what local backends
    like mlx-lm do automatically.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        api_key: str | None = None,
        chat: bool = False,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._api_key = api_key
        self._chat = chat

        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=120.0,
        )

    # ------------------------------------------------------------------
    # BackendAdapter protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "openai-compat"

    async def generate(self, prompt: Prompt) -> InferenceResult:
        """POST to the completions or chat/completions endpoint."""
        if self._chat:
            return await self._generate_chat(prompt)
        return await self._generate_completions(prompt)

    async def _generate_chat(self, prompt: Prompt) -> InferenceResult:
        """Use ``/v1/chat/completions`` with proper message formatting."""
        payload = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt.text}],
            "max_tokens": prompt.max_tokens,
            "temperature": prompt.metadata.get("temperature", 0.0) if prompt.metadata else 0.0,
        }

        start = time.perf_counter()
        try:
            response = await self._client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot connect to server at {self._base_url}. "
                "Ensure the server is running and the base_url is correct."
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Request to {self._base_url}/v1/chat/completions timed out after 120s."
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            body = exc.response.text[:500]
            raise RuntimeError(f"Server returned HTTP {status}: {body}") from exc

        elapsed_s = time.perf_counter() - start

        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError(f"Server returned non-JSON response: {response.text[:200]}") from exc

        if "choices" not in data or not data["choices"]:
            raise RuntimeError(f"Server returned empty or malformed response: {data}")

        choice = data["choices"][0]
        message = choice.get("message", {})
        text: str = message.get("content", "")
        tokens = text.split()

        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", len(tokens))
        tps = completion_tokens / elapsed_s if elapsed_s > 0 and completion_tokens else None

        return InferenceResult(
            prompt_id=prompt.id,
            backend_name=self.name,
            model_id=self._model_id,
            tokens=tokens,
            logprobs=None,
            text=text,
            latency_ms=elapsed_s * 1000,
            tokens_per_second=tps,
        )

    async def _generate_completions(self, prompt: Prompt) -> InferenceResult:
        """Use the legacy ``/v1/completions`` endpoint for raw logprobs."""
        payload = {
            "model": self._model_id,
            "prompt": prompt.text,
            "max_tokens": prompt.max_tokens,
            "temperature": prompt.metadata.get("temperature", 0.0) if prompt.metadata else 0.0,
            "logprobs": 5,
        }

        start = time.perf_counter()
        try:
            response = await self._client.post("/v1/completions", json=payload)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot connect to OpenAI-compatible server at {self._base_url}. "
                "Ensure the server is running and the base_url is correct."
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Request to {self._base_url}/v1/completions timed out after 120s. "
                f"The model may be too large or the server overloaded."
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            body = exc.response.text[:500]
            if status == 404:
                raise RuntimeError(
                    f"Server at {self._base_url} returned 404. "
                    f"Check that /v1/completions endpoint exists. "
                    f"Some servers (Ollama) use /api/generate instead."
                ) from exc
            elif status == 401 or status == 403:
                raise RuntimeError(
                    f"Authentication failed at {self._base_url} (HTTP {status}). "
                    f"Check your API key."
                ) from exc
            else:
                raise RuntimeError(f"Server returned HTTP {status}: {body}") from exc

        elapsed_s = time.perf_counter() - start

        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError(f"Server returned non-JSON response: {response.text[:200]}") from exc

        if "choices" not in data or not data["choices"]:
            raise RuntimeError(f"Server returned empty or malformed response: {data}")

        choice = data["choices"][0]
        text: str = choice.get("text", "")

        # Parse logprobs ---------------------------------------------------
        tokens: list[str] = []
        logprobs_list: list[float] | None = None
        distributions: list[list[float]] | None = None

        lp_data = choice.get("logprobs")
        if lp_data and lp_data.get("tokens"):
            tokens = lp_data["tokens"]
            raw_logprobs = lp_data.get("token_logprobs", [])
            logprobs_list = [
                float(v) if v is not None and not math.isnan(v) else 0.0 for v in raw_logprobs
            ]

            top_logprobs = lp_data.get("top_logprobs")
            if top_logprobs:
                distributions = []
                for entry in top_logprobs:
                    # entry is a dict of token: logprob
                    distributions.append(list(entry.values()))
        else:
            tokens = text.split()

        # Timing -----------------------------------------------------------
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", len(tokens))
        tps = completion_tokens / elapsed_s if elapsed_s > 0 and completion_tokens else None

        return InferenceResult(
            prompt_id=prompt.id,
            backend_name=self.name,
            model_id=self._model_id,
            tokens=tokens,
            logprobs=logprobs_list,
            distributions=distributions,
            text=text,
            latency_ms=elapsed_s * 1000,
            tokens_per_second=tps,
        )

    async def generate_batch(self, prompts: list[Prompt]) -> list[InferenceResult]:
        return list(await asyncio.gather(*(self.generate(p) for p in prompts)))

    async def health_check(self) -> bool:
        """Verify the server responds to a lightweight models listing."""
        try:
            resp = await self._client.get("/v1/models")
            return resp.status_code == 200
        except httpx.ConnectError:
            return False

    async def cleanup(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
