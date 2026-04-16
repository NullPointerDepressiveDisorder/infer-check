# infer-check

[![PyPI - Version](https://img.shields.io/pypi/v/infer-check?logo=PyPi&color=%233775A9)](https://pypi.org/project/infer-check/)
[![Run tests and upload coverage](https://github.com/NullPointerDepressiveDisorder/infer-check/actions/workflows/coverage.yml/badge.svg?branch=main)](https://github.com/NullPointerDepressiveDisorder/infer-check/actions/workflows/coverage.yml)
[![codecov](https://codecov.io/gh/NullPointerDepressiveDisorder/infer-check/graph/badge.svg?token=FWG0Z5YHUS)](https://codecov.io/gh/NullPointerDepressiveDisorder/infer-check)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://nullpointerdepressivedisorder.github.io/infer-check)

**Catches the correctness bugs that benchmarks miss in LLM inference engines.**

Quantization silently breaks arithmetic. Serving layers silently alter output. KV caches silently corrupt under load. Benchmarks like lm-evaluation-harness test whether models are smart — `infer-check` tests whether engines are correct.

> **[Read the full documentation](https://nullpointerdepressivedisorder.github.io/infer-check)**

## The problem

Every LLM inference engine has correctness bugs that benchmarks don't catch:

- **KV cache NaN pollution** in vLLM-Ascend permanently corrupts all subsequent requests
- **FP8 KV quantization** in vLLM causes repeated garbage output
- **32.5% element mismatches** in SGLang's FP8 DeepGEMM kernels on Blackwell GPUs
- **Batch-size-dependent output** where tokens change depending on concurrent request count

These aren't model quality problems — they're engine correctness failures. `infer-check` is a CLI tool that runs differential tests across backends, quantization levels, and concurrency conditions to surface them automatically.

## Installation

```
pip install infer-check

# With MLX backend support (Apple Silicon)
pip install "infer-check[mlx]"
```

## Quick start

Compare two quantizations head-to-head:

```
infer-check compare \
  mlx-community/Llama-3.1-8B-Instruct-4bit \
  mlx-community/Llama-3.1-8B-Instruct-8bit \
  --prompts adversarial-numerics
```

Run a full quantization sweep:

```
infer-check sweep \
  --models "bf16=mlx-community/Meta-Llama-3.1-8B-Instruct-bf16,\
            8bit=mlx-community/Meta-Llama-3.1-8B-Instruct-8bit,\
            4bit=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit" \
  --prompts reasoning
```

## Commands

| Command | Purpose | Docs |
| --- | --- | --- |
| `sweep` | Compare pre-quantized models against a baseline | [docs](https://nullpointerdepressivedisorder.github.io/infer-check/commands/sweep/) |
| `compare` | Head-to-head comparison of two models or quantizations | [docs](https://nullpointerdepressivedisorder.github.io/infer-check/commands/compare/) |
| `diff` | Compare outputs across different backends for the same model | [docs](https://nullpointerdepressivedisorder.github.io/infer-check/commands/diff/) |
| `determinism` | Test output reproducibility at temperature=0 | [docs](https://nullpointerdepressivedisorder.github.io/infer-check/commands/determinism/) |
| `stress` | Test correctness under concurrent load | [docs](https://nullpointerdepressivedisorder.github.io/infer-check/commands/stress/) |
| `report` | Generate HTML/JSON reports from saved results | [docs](https://nullpointerdepressivedisorder.github.io/infer-check/commands/report/) |

## Example results

Results from running `infer-check` on Llama-3.1-8B-Instruct on Apple Silicon using mlx-lm.

### Quantization sweep

```
                                 Sweep Summary
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ quant_level         ┃ identical ┃ minor ┃ moderate ┃ severe ┃ mean_similarity ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ bf16 (self-check)   │     50/50 │  0/50 │     0/50 │   0/50 │          1.0000 │
│ 8bit                │     20/50 │  9/50 │    12/50 │   9/50 │          0.8067 │
│ 4bit                │      1/50 │  3/50 │    11/50 │  35/50 │          0.3837 │
└─────────────────────┴───────────┴───────┴──────────┴────────┴─────────────────┘
```

A "severe" divergence means the quantized output is functionally wrong — not just worded differently, but giving incorrect answers to questions the bf16 baseline handles correctly.

### Cross-backend diff

mlx-lm vs vllm-mlx at temperature=0: 50/50 identical (reasoning) and 30/30 identical (numerics). Zero serving-layer divergence detected.

### Determinism & stress

100% determinism across 20 runs per prompt at temperature=0. 100% output consistency at concurrency levels 1/2/4/8.

## Supported backends

| Backend | Type | Use case |
| --- | --- | --- |
| **mlx-lm** | In-process | Local Apple Silicon inference with logprobs |
| **llama.cpp** | HTTP | `llama-server` via `/completion` endpoint |
| **vllm-mlx** | HTTP | Continuous batching on Apple Silicon |
| **openai-compat** | HTTP | Any OpenAI-compatible server (vLLM, SGLang, Ollama) |

See the [backends documentation](https://nullpointerdepressivedisorder.github.io/infer-check/backends/) for setup and configuration details.

## Prompt suites

Six curated suites ship with the package — no need to clone the repo:

| Suite | Count | Purpose |
| --- | --- | --- |
| `reasoning` | 50 | Multi-step math and logic |
| `code` | 49 | Python, JSON, SQL generation |
| `adversarial-numerics` | 30 | IEEE 754 edge cases, overflow, precision |
| `long-context` | 10 | Tables and transcripts with recall questions |
| `quant-sensitive` | 20 | Multi-digit arithmetic, long CoT, precise syntax |
| `determinism` | 50 | High-entropy continuations for determinism testing |

Custom suites are JSONL files with one object per line:

```json
{"id": "custom-001", "text": "Your prompt here", "category": "math", "max_tokens": 512}
```

## Roadmap

- [ ] GGUF backend (direct llama.cpp integration without HTTP)
- [ ] CUDA vLLM backend for GPU-based differential testing
- [ ] Logprobs-based divergence scoring where backends support it
- [ ] Automated regression CI mode (`infer-check ci` with pass/fail exit codes)
- [ ] Expanded prompt suites for tool use and multi-turn conversations

## Requirements

- Python >= 3.11
- macOS with Apple Silicon (for mlx-lm backend) or Linux
- At least one backend installed

## License

Apache 2.0
