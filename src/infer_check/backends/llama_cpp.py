"""llama.cpp backend adapter — HTTP client for llama-server."""

from __future__ import annotations

import asyncio
import math
import time

import httpx

from infer_check.types import InferenceResult, Prompt

__all__ = ["LlamaCppBackend"]


class LlamaCppBackend:
    """Backend adapter for llama.cpp's built-in HTTP server (``llama-server``).

    Communicates via the ``/completion`` endpoint.
    """

    def __init__(self, base_url: str = "http://localhost:8080") -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=120.0)

    # ------------------------------------------------------------------
    # BackendAdapter protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "llama-cpp"

    async def generate(self, prompt: Prompt) -> InferenceResult:
        """Send a completion request and parse the response."""
        payload = {
            "prompt": prompt.text,
            "n_predict": prompt.max_tokens,
            "temperature": prompt.metadata.get("temperature", 0.0) if prompt.metadata else 0.0,
            "n_probs": 10,  # Request top 10 probabilities for KL divergence
        }

        start = time.perf_counter()
        try:
            response = await self._client.post("/completion", json=payload)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot connect to llama-server at {self._base_url}. "
                "Start it with: llama-server -m <model.gguf> --port 8080\n"
                "Or use Ollama: ollama serve"
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Request to {self._base_url}/completion timed out after 120s. "
                f"The model may be too large or the prompt too long."
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            body = exc.response.text[:500]
            raise RuntimeError(f"llama-server returned HTTP {status}: {body}") from exc

        elapsed_s = time.perf_counter() - start

        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"llama-server returned non-JSON response: {response.text[:200]}"
            ) from exc

        content: str = data.get("content", "")

        # Extract per-token data ------------------------------------------
        tokens: list[str] = []
        logprobs: list[float] | None = None
        distributions: list[list[float]] | None = None
        distribution_metadata: list[dict[str, int]] | None = None

        completion_probs = data.get("completion_probabilities")
        if completion_probs:
            logprobs = []
            distributions = []
            distribution_metadata = []
            for entry in completion_probs:
                tok_str = entry.get("content", "")
                tokens.append(tok_str)
                # Top prob entry (index 0) contains the chosen token linear prob.
                probs = entry.get("probs", [])
                if probs:
                    # llama-server returns linear probabilities (0..1).
                    # We convert to log-probabilities (log-space) to match the rest of the codebase.
                    # Epsilon 1e-10 matches kl_divergence epsilon in the analyzer.
                    epsilon = 1e-10
                    logprobs.append(math.log(max(float(probs[0].get("prob", 0.0)), epsilon)))

                    dist_logprobs = []
                    for p in probs:
                        p_val = max(float(p.get("prob", 0.0)), epsilon)
                        dist_logprobs.append(math.log(p_val))
                    distributions.append(dist_logprobs)

                    # Store token IDs to allow alignment if needed.
                    dist_meta = {}
                    for i, p in enumerate(probs):
                        if "id" in p:
                            dist_meta[f"id_{i}"] = int(p["id"])
                    distribution_metadata.append(dist_meta)
        else:
            tokens = content.split()

        # Timing -----------------------------------------------------------
        timing = data.get("timings", {})
        tps = timing.get("predicted_per_second")
        if tps is None and tokens:
            tps = len(tokens) / elapsed_s if elapsed_s > 0 else None

        return InferenceResult(
            prompt_id=prompt.id,
            backend_name=self.name,
            model_id=data.get("model", "unknown"),
            tokens=tokens,
            logprobs=logprobs,
            distributions=distributions,
            distribution_metadata=distribution_metadata,
            text=content,
            latency_ms=elapsed_s * 1000,
            tokens_per_second=tps,
        )

    async def generate_batch(self, prompts: list[Prompt]) -> list[InferenceResult]:
        return list(await asyncio.gather(*(self.generate(p) for p in prompts)))

    async def health_check(self) -> bool:
        """Call ``GET /health`` on the llama-server."""
        try:
            resp = await self._client.get("/health")
            return resp.status_code == 200
        except httpx.ConnectError:
            return False

    async def cleanup(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
