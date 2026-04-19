"""Generic OpenAI-compatible backend adapter.

Works with any server that implements the ``/v1/completions`` endpoint
(vLLM, SGLang, Ollama, vllm-mlx, etc.).
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any

import httpx

from infer_check.types import InferenceResult, Prompt
from infer_check.utils import format_prompt, strip_thinking_tokens

__all__ = ["OpenAICompatBackend"]


class _ServerHTTPError(RuntimeError):
    """Internal exception carrying the HTTP status code from the server."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        super().__init__(f"Server returned HTTP {status_code}: {body}")


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
        revision: str | None = None,
        disable_thinking: bool = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._api_key = api_key
        self._chat = chat
        self._revision = revision
        self._disable_thinking = disable_thinking
        # Ollama listens on :11434 by default. When we're talking to Ollama and
        # thinking is disabled, we prepend "/no_think" to the user message — a
        # directive that Qwen3 and some Gemma/Ollama templates honour even when
        # the top-level `think` field is ignored.
        self._is_ollama = ":11434" in self._base_url

        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=120.0,
        )

        # Logprobs support: assume yes until a server rejects it.
        self._chat_logprobs_supported: bool = True
        # Thinking-disable keys are opportunistic: most servers accept them,
        # some reject unknown params with 400/422. We drop them on first failure.
        self._thinking_keys_supported: bool = True

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

    async def _post_chat(self, payload: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        """POST to /v1/chat/completions with consistent error handling.

        Returns (elapsed_seconds, response_json).
        """
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
            raise RuntimeError(f"Request to {self._base_url}/v1/chat/completions timed out after 120s.") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            body = exc.response.text[:500]
            raise _ServerHTTPError(status, body) from exc

        elapsed_s = time.perf_counter() - start

        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError(f"Server returned non-JSON response: {response.text[:200]}") from exc

        if "choices" not in data or not data["choices"]:
            raise RuntimeError(f"Server returned empty or malformed response: {data}")

        return elapsed_s, data

    async def _generate_chat(self, prompt: Prompt) -> InferenceResult:
        """Use ``/v1/chat/completions`` with proper message formatting.

        Requests logprobs when the server supports them.  If the first
        request fails with 400 or 422 (unsupported parameter), the backend
        automatically retries without logprobs and disables them for
        all subsequent requests.

        When ``disable_thinking`` is set and the server is Ollama, we route
        through Ollama's native ``/api/chat`` endpoint instead — it's the only
        one that reliably honours ``think: false`` across Ollama's model zoo
        (gpt-oss, Qwen3, Gemma-thinking, …). Logprobs are requested via the
        native ``logprobs`` and ``top_logprobs`` fields.
        """
        if self._disable_thinking and self._is_ollama:
            return await self._generate_ollama_native(prompt)

        user_text = strip_thinking_tokens(prompt.text) if self._disable_thinking else prompt.text
        messages: list[dict[str, str]] = []
        if self._disable_thinking:
            # Empty system message overrides any server-side SYSTEM default
            # (Ollama Modelfile SYSTEM, hosted-template system prompts, …)
            # that might re-inject a thinking trigger token.
            messages.append({"role": "system", "content": ""})
        messages.append({"role": "user", "content": user_text})
        payload: dict[str, object] = {
            "model": self._model_id,
            "messages": messages,
            "max_tokens": prompt.max_tokens,
            "temperature": prompt.metadata.get("temperature", 0.0) if prompt.metadata else 0.0,
        }
        if self._chat_logprobs_supported:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5
        if self._disable_thinking and self._thinking_keys_supported:
            # Cross-backend hints for disabling reasoning/thinking mode:
            #   chat_template_kwargs.enable_thinking — vLLM / SGLang / Qwen3 family
            #   chat_template_kwargs.thinking        — some DeepSeek / HunYuan templates
            #   think                                — Ollama native flag
            #   reasoning.enabled / reasoning_effort — OpenRouter / OpenAI-style
            payload["chat_template_kwargs"] = {"enable_thinking": False, "thinking": False}
            payload["think"] = False
            payload["reasoning"] = {"enabled": False}
            payload["reasoning_effort"] = "minimal"

        try:
            elapsed_s, data = await self._post_chat(payload)
        except _ServerHTTPError as exc:
            # Retry shedding unsupported params only on 400/422.
            if exc.status_code in (400, 422) and (self._chat_logprobs_supported or self._thinking_keys_supported):
                if self._chat_logprobs_supported:
                    self._chat_logprobs_supported = False
                    payload.pop("logprobs", None)
                    payload.pop("top_logprobs", None)
                if self._thinking_keys_supported:
                    self._thinking_keys_supported = False
                    payload.pop("chat_template_kwargs", None)
                    payload.pop("think", None)
                    payload.pop("reasoning", None)
                    payload.pop("reasoning_effort", None)
                elapsed_s, data = await self._post_chat(payload)
            else:
                raise

        choice = data["choices"][0]
        message = choice.get("message", {})
        text: str = message.get("content", "")
        if not text:
            text = message.get("reasoning_content", "")

        # Parse logprobs (chat completions format) -------------------------
        tokens: list[str] = []
        logprobs_list: list[float] | None = None
        distributions: list[list[float]] | None = None
        distribution_metadata: list[dict[str, int | str]] | None = None

        lp_data = choice.get("logprobs")
        if lp_data and lp_data.get("content"):
            content_logprobs = lp_data["content"]
            tokens = [entry.get("token", "") for entry in content_logprobs]
            logprobs_list = []
            for entry in content_logprobs:
                raw = entry.get("logprob")
                try:
                    fv = float(raw) if raw is not None else -9999.0
                except (TypeError, ValueError):
                    fv = -9999.0
                if math.isnan(fv):
                    fv = -9999.0
                logprobs_list.append(fv)

            distributions = []
            distribution_metadata = []
            for entry in content_logprobs:
                top = entry.get("top_logprobs", [])
                if not top:
                    distributions.append([])
                    distribution_metadata.append({})
                    continue
                sorted_items = sorted(top, key=lambda x: x.get("token", ""))
                cleaned: list[tuple[str, float]] = []
                for item in sorted_items:
                    try:
                        fv = float(item["logprob"]) if item.get("logprob") is not None else -9999.0
                    except (TypeError, ValueError):
                        fv = -9999.0
                    if math.isnan(fv):
                        fv = -9999.0
                    cleaned.append((item.get("token", ""), fv))
                distributions.append([fv for _, fv in cleaned])
                meta: dict[str, int | str] = {}
                for i, (tok, _) in enumerate(cleaned):
                    meta[f"id_{i}"] = tok
                distribution_metadata.append(meta)
        else:
            tokens = text.split()

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
            distribution_metadata=distribution_metadata,
            text=text,
            latency_ms=elapsed_s * 1000,
            tokens_per_second=tps,
        )

    async def _generate_ollama_native(self, prompt: Prompt) -> InferenceResult:
        """POST to Ollama's native ``/api/chat`` with ``think: false``.

        Unlike ``/v1/chat/completions``, Ollama's native endpoint consistently
        respects the ``think`` flag — vital for Gemma-thinking, gpt-oss, and
        other variants whose Modelfile TEMPLATE hardcodes the thinking trigger.
        Logprobs are requested via the ``logprobs`` and ``top_logprobs`` fields.
        """
        user_text = strip_thinking_tokens(prompt.text)
        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": user_text},
            ],
            "stream": False,
            "think": False,
            "logprobs": True,
            "top_logprobs": 5,
            "options": {
                "temperature": prompt.metadata.get("temperature", 0.0) if prompt.metadata else 0.0,
                "num_predict": prompt.max_tokens,
            },
        }

        start = time.perf_counter()
        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._base_url}. Ensure `ollama serve` is running."
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(f"Request to {self._base_url}/api/chat timed out after 120s.") from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text[:500]}") from exc

        elapsed_s = time.perf_counter() - start

        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError(f"Ollama returned non-JSON response: {response.text[:200]}") from exc

        message = data.get("message") or {}
        text: str = message.get("content", "") or ""

        # Parse logprobs (Ollama native format) ----------------------------
        # Ollama returns logprobs at the top level as a list of token entries,
        # each with {token, logprob, top_logprobs: [{token, logprob}, ...]}.
        tokens: list[str] = []
        logprobs_list: list[float] | None = None
        distributions: list[list[float]] | None = None
        distribution_metadata: list[dict[str, int | str]] | None = None

        lp_data = data.get("logprobs")
        if lp_data and isinstance(lp_data, list) and len(lp_data) > 0:
            tokens = [entry.get("token", "") for entry in lp_data]
            logprobs_list = []
            for entry in lp_data:
                raw = entry.get("logprob")
                try:
                    fv = float(raw) if raw is not None else -9999.0
                except (TypeError, ValueError):
                    fv = -9999.0
                if math.isnan(fv):
                    fv = -9999.0
                logprobs_list.append(fv)

            distributions = []
            distribution_metadata = []
            for entry in lp_data:
                top = entry.get("top_logprobs", [])
                if not top:
                    distributions.append([])
                    distribution_metadata.append({})
                    continue
                sorted_items = sorted(top, key=lambda x: x.get("token", ""))
                cleaned: list[tuple[str, float]] = []
                for item in sorted_items:
                    try:
                        fv = float(item["logprob"]) if item.get("logprob") is not None else -9999.0
                    except (TypeError, ValueError):
                        fv = -9999.0
                    if math.isnan(fv):
                        fv = -9999.0
                    cleaned.append((item.get("token", ""), fv))
                distributions.append([fv for _, fv in cleaned])
                meta: dict[str, int | str] = {}
                for i, (tok, _) in enumerate(cleaned):
                    meta[f"id_{i}"] = tok
                distribution_metadata.append(meta)
        else:
            tokens = text.split()

        completion_tokens = data.get("eval_count", len(tokens))
        tps = completion_tokens / elapsed_s if elapsed_s > 0 and completion_tokens else None

        return InferenceResult(
            prompt_id=prompt.id,
            backend_name=self.name,
            model_id=self._model_id,
            tokens=tokens,
            logprobs=logprobs_list,
            distributions=distributions,
            distribution_metadata=distribution_metadata,
            text=text,
            latency_ms=elapsed_s * 1000,
            tokens_per_second=tps,
        )

    async def _generate_completions(self, prompt: Prompt) -> InferenceResult:
        """Use the legacy ``/v1/completions`` endpoint for raw logprobs."""
        formatted = format_prompt(
            prompt.text,
            model_id=self._model_id,
            revision=self._revision,
            disable_thinking=self._disable_thinking,
        )
        payload = {
            "model": self._model_id,
            "prompt": formatted,
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
                    f"Authentication failed at {self._base_url} (HTTP {status}). Check your API key."
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
        distribution_metadata: list[dict[str, int | str]] | None = None

        lp_data = choice.get("logprobs")
        if lp_data and lp_data.get("tokens"):
            tokens = lp_data["tokens"]
            raw_logprobs = lp_data.get("token_logprobs", [])
            logprobs_list = [float(v) if v is not None and not math.isnan(v) else -9999.0 for v in raw_logprobs]

            top_logprobs = lp_data.get("top_logprobs")
            if top_logprobs:
                distributions = []
                distribution_metadata = []
                for entry in top_logprobs:
                    # entry is a dict of token: logprob
                    # Sort by token string to ensure deterministic order
                    if not entry:
                        distributions.append([])
                        distribution_metadata.append({})
                        continue
                    sorted_items = sorted(entry.items())
                    cleaned_items: list[tuple[str, float]] = []
                    for tok, v in sorted_items:
                        # Mirror token_logprobs sanitization: treat None/NaN/invalid as -9999.0
                        try:
                            fv = float(v) if v is not None else float("nan")
                        except (TypeError, ValueError):
                            fv = float("nan")
                        if math.isnan(fv):
                            fv = -9999.0
                        cleaned_items.append((tok, fv))
                    distributions.append([fv for _, fv in cleaned_items])
                    meta: dict[str, int | str] = {}
                    for i, (tok, _) in enumerate(cleaned_items):
                        meta[f"id_{i}"] = tok
                    distribution_metadata.append(meta)
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
            distribution_metadata=distribution_metadata,
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
