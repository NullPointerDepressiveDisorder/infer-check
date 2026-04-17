"""Tests for infer_check.resolve — model spec resolution."""

from pathlib import Path

import pytest

from infer_check.resolve import resolve_model

# ── Explicit prefix tests ────────────────────────────────────────────


class TestExplicitPrefix:
    def test_ollama_prefix(self) -> None:
        r = resolve_model("ollama:llama3.1:8b-instruct-q4_K_M")
        assert r.backend == "openai-compat"
        assert r.model_id == "llama3.1:8b-instruct-q4_K_M"
        assert r.base_url == "http://127.0.0.1:11434"
        assert r.label == "llama3.1:8b-instruct-q4_K_M"

    def test_mlx_prefix(self) -> None:
        r = resolve_model("mlx:mlx-community/Llama-3.1-8B-Instruct-4bit")
        assert r.backend == "mlx-lm"
        assert r.model_id == "mlx-community/Llama-3.1-8B-Instruct-4bit"
        assert r.base_url is None  # mlx-lm has no default URL

    def test_gguf_prefix(self) -> None:
        r = resolve_model("gguf:/path/to/model.gguf")
        assert r.backend == "llama-cpp"
        assert r.model_id == "/path/to/model.gguf"
        assert r.base_url == "http://127.0.0.1:8080"

    def test_vllm_mlx_prefix(self) -> None:
        r = resolve_model("vllm-mlx:mlx-community/Llama-3.1-8B-Instruct-4bit")
        assert r.backend == "vllm-mlx"
        assert r.model_id == "mlx-community/Llama-3.1-8B-Instruct-4bit"
        assert r.base_url == "http://127.0.0.1:8000"

    def test_prefix_case_insensitive(self) -> None:
        r = resolve_model("OLLAMA:llama3.1:8b")
        assert r.backend == "openai-compat"
        assert r.model_id == "llama3.1:8b"


# ── Heuristic detection tests ────────────────────────────────────────


class TestHeuristicDetection:
    def test_mlx_community_repo(self) -> None:
        r = resolve_model("mlx-community/Llama-3.1-8B-Instruct-4bit")
        assert r.backend == "mlx-lm"
        assert r.base_url is None
        assert r.label == "Llama-3.1-8B-Instruct-4bit"

    def test_gguf_repo_heuristic(self) -> None:
        r = resolve_model("bartowski/Llama-3.1-8B-Instruct-GGUF")
        assert r.backend == "llama-cpp"
        assert r.label == "Llama-3.1-8B-Instruct-GGUF"

    def test_maziyarpanahi_gguf_heuristic(self) -> None:
        r = resolve_model("MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF")
        assert r.backend == "llama-cpp"

    def test_mlx_keyword_heuristic(self) -> None:
        r = resolve_model("some-user/my-special-mlx-model")
        assert r.backend == "mlx-lm"

    def test_ollama_style_tag(self) -> None:
        r = resolve_model("llama3.1:8b-instruct-q4_K_M")
        assert r.backend == "openai-compat"
        assert r.base_url == "http://127.0.0.1:11434"

    def test_local_gguf_path(self, tmp_path: Path) -> None:
        gguf_file = tmp_path / "model-q4.gguf"
        gguf_file.touch()
        r = resolve_model(str(gguf_file))
        assert r.backend == "llama-cpp"
        assert r.label == "model-q4"

    def test_nonexistent_gguf_path(self) -> None:
        r = resolve_model("/does/not/exist/model.gguf")
        assert r.backend == "llama-cpp"
        assert r.model_id == "/does/not/exist/model.gguf"

    def test_fallback_to_mlx(self) -> None:
        r = resolve_model("meta-llama/Llama-3.1-8B-Instruct")
        assert r.backend == "mlx-lm"
        assert r.base_url is None
        assert r.label == "Llama-3.1-8B-Instruct"


# ── Override tests ───────────────────────────────────────────────────


class TestOverrides:
    def test_custom_label(self) -> None:
        r = resolve_model(
            "mlx-community/Llama-3.1-8B-Instruct-4bit",
            label="llama-4bit",
        )
        assert r.label == "llama-4bit"
        assert r.backend == "mlx-lm"

    def test_custom_base_url(self) -> None:
        r = resolve_model(
            "ollama:llama3.1:8b",
            base_url="http://my-server:11434/v1",
        )
        assert r.base_url == "http://my-server:11434/v1"

    def test_base_url_override_on_gguf_repo(self) -> None:
        r = resolve_model(
            "bartowski/Llama-3.1-8B-Instruct-GGUF",
            base_url="http://custom:9999/v1",
        )
        assert r.base_url == "http://custom:9999/v1"


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_spec_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty model spec"):
            resolve_model("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty model spec"):
            resolve_model("   ")

    def test_spec_with_leading_trailing_whitespace(self) -> None:
        r = resolve_model("  mlx-community/Llama-3.1-8B-Instruct-4bit  ")
        assert r.backend == "mlx-lm"
        assert r.model_id == "mlx-community/Llama-3.1-8B-Instruct-4bit"

    def test_resolved_model_str(self) -> None:
        r = resolve_model("mlx-community/Llama-3.1-8B-Instruct-4bit")
        assert str(r) == "Llama-3.1-8B-Instruct-4bit (mlx-lm)"
