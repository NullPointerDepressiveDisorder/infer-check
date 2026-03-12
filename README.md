# infer-check

[![PyPI - Version](https://img.shields.io/pypi/v/infer-check?logo=PyPi&color=%233775A9)](https://pypi.org/project/infer-check/)
[![Run tests and upload coverage](https://github.com/NullPointerDepressiveDisorder/infer-check/actions/workflows/coverage.yml/badge.svg?branch=main)](https://github.com/NullPointerDepressiveDisorder/infer-check/actions/workflows/coverage.yml)

**Catches the correctness bugs that benchmarks miss in LLM inference engines.**

Quantization silently breaks arithmetic. Serving layers silently alter output. KV caches silently corrupt under load. Benchmarks like lm-evaluation-harness test whether models are smart — `infer-check` tests whether engines are correct.

## The problem

Every LLM inference engine has correctness bugs that benchmarks don't catch:

- **KV cache NaN pollution** in vLLM-Ascend permanently corrupts all subsequent requests
- **FP8 KV quantization** in vLLM causes repeated garbage output
- **32.5% element mismatches** in SGLang's FP8 DeepGEMM kernels on Blackwell GPUs
- **Batch-size-dependent output** where tokens change depending on concurrent request count

These aren't model quality problems — they're engine correctness failures. `infer-check` is a CLI tool that runs differential tests across backends, quantization levels, and concurrency conditions to surface them automatically.

## Example results

Results from running `infer-check` on Llama-3.1-8B-Instruct and Qwen3.5-4B (MoE) on Apple Silicon using mlx-lm and vllm-mlx. These demonstrate what the tool catches — not a comprehensive benchmark.

### Quantization sweep

4-bit quantization on Llama-3.1-8B showed clear task-dependent degradation. Numerical tasks broke worst:

```
                       Llama-3.1-8B: bf16 vs 4bit
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ prompt suite          ┃ identical ┃ severe   ┃ mean_similarity ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ adversarial-numerics  │     0/30  │   23/30  │          0.311  │
│ reasoning             │     1/50  │   35/50  │          0.384  │
│ code                  │     0/49  │   30/49  │          0.452  │
└───────────────────────┴───────────┴──────────┴─────────────────┘
```

A "severe" divergence means the quantized output is functionally wrong — not just worded differently, but giving incorrect answers to questions the bf16 baseline handles correctly. This pattern is consistent with published research on quantization-induced degradation, reproduced here on MLX's native quantization scheme.

### Dense vs. MoE comparison

Qwen3.5-4B (Gated Delta Networks + sparse MoE) showed similar degradation rates to dense Llama-3.1-8B in our testing — 35/50 severe on reasoning at 4-bit. Small sample, but the tool picks up the signal clearly on both architectures.

### Cross-backend diff

mlx-lm vs vllm-mlx at temperature=0 on Llama-3.1-8B-4bit: 50/50 identical (reasoning) and 30/30 identical (numerics). In this test, the vllm-mlx serving layer introduced zero divergence — output differences in production would come from quantization, not from the serving layer itself.

### Determinism

Llama-3.1-8B-4bit and Qwen3.5-4B both scored 50/50 identical across 20 runs per prompt on single-request mlx-lm inference at temperature=0.

### Stress test

vllm-mlx at concurrency 1/2/4/8: zero errors, 100% output consistency at all levels. No KV cache corruption or batch-dependent divergence detected.

## Installation

```
pip install infer-check

# With MLX backend support (Apple Silicon)
pip install "infer-check[mlx]"
```

## Usage

### Quantization sweep

Compare pre-quantized models against a baseline. Each model is a separate HuggingFace repo.

```
infer-check sweep \
  --models "bf16=mlx-community/Meta-Llama-3.1-8B-Instruct-bf16,\
            8bit=mlx-community/Meta-Llama-3.1-8B-Instruct-8bit,\
            4bit=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit" \
  --backend mlx-lm \
  --prompts reasoning \
  --output ./results/sweep/
```

`--prompts` accepts either a bundled suite name (`reasoning`, `code`, `adversarial-numerics`, `determinism`, `long-context`) or a path to any `.jsonl` file.

The baseline is automatically run twice as a self-check — if it's not 50/50 identical, your comparison data is unreliable.

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

### Cross-backend diff

Same model, same quant, different inference paths. Catches serving-layer bugs.

```
# Start vllm-mlx in another terminal:
# vllm-mlx serve mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --port 8000

infer-check diff \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --backends "mlx-lm,openai-compat" \
  --base-urls ",http://localhost:8000" \
  --prompts reasoning \
  --output ./results/diff/
```

Uses `/v1/chat/completions` by default (`--chat`) so server-side chat templates match the local backend. Pass `--no-chat` for raw `/v1/completions`.

### Determinism

Same prompt N times at temperature=0. Output should be bit-identical every run.

```
infer-check determinism \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --backend mlx-lm \
  --prompts determinism \
  --runs 20 \
  --output ./results/determinism/
```

### Stress test

Concurrent requests through a serving backend. Tests KV cache correctness under load.

```
infer-check stress \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --backend openai-compat \
  --base-url http://localhost:8000 \
  --prompts reasoning \
  --concurrency 1,2,4,8 \
  --output ./results/stress/
```

### Report

Generate an HTML report from all saved results.

```
infer-check report ./results/ --format html
```

## Prompt suites

Curated prompts targeting known quantization failure modes:

| Suite | Count | Purpose |
| --- | --- | --- |
| `reasoning.jsonl` | 50 | Multi-step math and logic |
| `code.jsonl` | 49 | Python, JSON, SQL generation |
| `adversarial-numerics.jsonl` | 30 | IEEE 754 edge cases, overflow, precision |
| `long-context.jsonl` | 10 | Tables and transcripts with recall questions |
| `determinism.jsonl` | 50 | High-entropy continuations for determinism testing |

All suites ship with the package — no need to clone the repo. Custom suites are JSONL files with one object per line:

```json
{"id": "custom-001", "text": "Your prompt here", "category": "math", "max_tokens": 512}
```

## Supported backends

| Backend | Type | Use case |
| --- | --- | --- |
| **mlx-lm** | In-process | Local Apple Silicon inference with logprobs |
| **llama.cpp** | HTTP | `llama-server` via `/completion` endpoint |
| **vllm-mlx** | HTTP | Continuous batching on Apple Silicon |
| **openai-compat** | HTTP | Any OpenAI-compatible server (vLLM, SGLang, Ollama) |

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
