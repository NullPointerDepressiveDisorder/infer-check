# Backends

`infer-check` supports four inference backends. Each backend implements a common protocol for generation, health checks, and cleanup.

## Overview

| Backend           | Type | Default URL | Use case |
|-------------------|------|-------------|----------|
| **mlx-lm**        | In-process | (local) | Local Apple Silicon inference with logprobs |
| **llama-cpp**     | HTTP | `http://127.0.0.1:8080` | llama-server via `/completion` endpoint |
| **vllm-mlx**      | HTTP | `http://127.0.0.1:8000` | Continuous batching on Apple Silicon |
| **openai-compat** | HTTP | `http://127.0.0.1:11434` | Any OpenAI-compatible server (vLLM, SGLang, Ollama) |

## mlx-lm

In-process inference using Apple's MLX framework. Runs directly on Apple Silicon with no server required.

**Install**: `pip install "infer-check[mlx]"` (requires `mlx` and `mlx-lm` packages)

**Features**:

- Generates per-token logprobs when available via `generate_step()`
- Falls back to simple generation if logprobs aren't supported
- Lazy model loading -- the model is downloaded and loaded on first use, not at import time
- Single-threaded sequential inference

**When to use**: Local testing on Mac. Best baseline for quantization sweeps since it runs natively with no serving layer overhead.

**Example**:

```bash
infer-check compare \
  mlx-community/Llama-3.1-8B-Instruct-bf16 \
  mlx-community/Llama-3.1-8B-Instruct-4bit
```

## llama.cpp

HTTP backend targeting [llama-server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server) (the built-in HTTP server from llama.cpp).

**Setup**: Start llama-server separately:

```bash
llama-server -m /path/to/model.gguf --port 8080
```

**Features**:

- Uses the `/completion` endpoint for text generation
- Requests top-10 token probabilities and converts them to logprobs
- Aligns token distributions by ID metadata for cross-backend comparison
- 120-second request timeout

**When to use**: Testing GGUF models served via llama.cpp. Good for comparing GGUF quantization formats against each other or against MLX native quantization.

**Example**:

```bash
infer-check determinism \
  --model my-model \
  --backend llama-cpp \
  --base-url http://127.0.0.1:8080 \
  --prompts determinism \
  --runs 20
```

## vllm-mlx

HTTP backend for [vllm-mlx](https://github.com/vllm-project/vllm-mlx), a continuous-batching inference server for Apple Silicon. Extends the OpenAI-compatible backend with model-aware health checks.

**Setup**: Start vllm-mlx separately:

```bash
vllm-mlx serve mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --port 8000
```

**Features**:

- Inherits all capabilities from the openai-compat backend
- Model-aware health check verifies the expected model is loaded
- Supports both `/v1/chat/completions` and `/v1/completions` endpoints

**When to use**: Testing continuous-batching serving layer correctness. Ideal for `diff` and `stress` commands to verify the serving layer doesn't introduce divergence.

**Example**:

```bash
infer-check diff \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --backends "mlx-lm,vllm-mlx" \
  --base-urls ",http://127.0.0.1:8000" \
  --prompts reasoning
```

## openai-compat

Generic backend for any server that implements the OpenAI API format. Works with vLLM, SGLang, Ollama, and others.

**Features**:

- Supports both `/v1/chat/completions` and `/v1/completions` endpoints
- Requests logprobs with graceful fallback if unsupported
- 120-second request timeout
- Detailed error messages for connection, timeout, and HTTP errors

**When to use**: Any OpenAI-compatible server. This is the most flexible backend and the default for Ollama-style model tags.

**Default URLs by resolution**:

| Model source | Default URL |
|-------------|-------------|
| Ollama tags (e.g., `llama3.1:8b`) | `http://127.0.0.1:11434` |
| Custom server | Use `--base-url` |

**Example with Ollama**:

```bash
infer-check compare \
  ollama:llama3.1:8b-instruct-q4_K_M \
  ollama:llama3.1:8b-instruct-q8_0
```

**Example with custom server**:

```bash
infer-check stress \
  --model my-model \
  --backend openai-compat \
  --base-url http://my-server:8000/v1 \
  --prompts reasoning \
  --concurrency 1,2,4,8
```

## Chat vs completions

HTTP backends support two endpoint modes:

- **Chat** (`--chat`, default) -- uses `/v1/chat/completions`. The server applies its chat template. Use this when the server is configured with the correct chat template for your model.
- **Completions** (`--no-chat`) -- uses `/v1/completions`. Sends raw text with no template. Use this for raw text generation or when you want to control the prompt format yourself.

The `--chat` / `--no-chat` flag applies to the `diff` command. The `compare` command always uses completions mode to avoid template differences between backends.

## Backend selection

Backends are selected in different ways depending on the command:

| Command | How backend is chosen |
|---------|----------------------|
| `compare` | Auto-detected from each model spec |
| `sweep` | `--backend` flag (shared across all models) or auto-detected |
| `diff` | `--backends` flag (explicit list) |
| `stress` | `--backend` flag or auto-detected from model |
| `determinism` | `--backend` flag or auto-detected from model |
| `report` | N/A (operates on saved results) |
