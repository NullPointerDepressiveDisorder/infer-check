# infer-check

**Correctness and reliability testing for LLM inference engines.**

Quantization silently breaks arithmetic. Serving layers silently alter output. KV caches silently corrupt under load. Benchmarks like lm-evaluation-harness test whether models are *smart* -- `infer-check` tests whether engines are *correct*.

## The problem

Every LLM inference engine has correctness bugs that benchmarks don't catch:

- **KV cache NaN pollution** in vLLM-Ascend permanently corrupts all subsequent requests
- **FP8 KV quantization** in vLLM causes repeated garbage output
- **32.5% element mismatches** in SGLang's FP8 DeepGEMM kernels on Blackwell GPUs
- **Batch-size-dependent output** where tokens change depending on concurrent request count

These aren't model quality problems -- they're engine correctness failures. `infer-check` runs differential tests across backends, quantization levels, and concurrency conditions to surface them automatically.

## What it does

`infer-check` provides six commands for testing inference correctness:

| Command | Purpose |
|---------|---------|
| [`sweep`](commands/sweep.md) | Compare pre-quantized models against a baseline |
| [`compare`](commands/compare.md) | Head-to-head comparison of two models or quantizations |
| [`diff`](commands/diff.md) | Compare outputs across different backends for the same model |
| [`determinism`](commands/determinism.md) | Test output reproducibility at temperature=0 |
| [`stress`](commands/stress.md) | Test correctness under concurrent load |
| [`report`](commands/report.md) | Generate HTML/JSON reports from saved results |

## Example results

Results from running `infer-check` on Llama-3.1-8B-Instruct and Qwen3.5-4B on Apple Silicon using mlx-lm and vllm-mlx.

### Quantization sweep

4-bit quantization on Llama-3.1-8B showed clear task-dependent degradation:

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

### Cross-backend diff

mlx-lm vs vllm-mlx at temperature=0 on Llama-3.1-8B-4bit: 50/50 identical (reasoning) and 30/30 identical (numerics). The vllm-mlx serving layer introduced zero divergence.

### Determinism

Llama-3.1-8B-4bit and Qwen3.5-4B both scored 50/50 identical across 20 runs per prompt on single-request mlx-lm inference at temperature=0.

### Stress test

vllm-mlx at concurrency 1/2/4/8: zero errors, 100% output consistency at all levels. No KV cache corruption or batch-dependent divergence detected.

## Quick start

```bash
pip install infer-check

# With MLX backend support (Apple Silicon)
pip install "infer-check[mlx]"
```

Then run your first comparison:

```bash
infer-check compare \
  mlx-community/Llama-3.1-8B-Instruct-4bit \
  mlx-community/Llama-3.1-8B-Instruct-8bit \
  --prompts adversarial-numerics
```

See the [Getting Started](getting-started.md) guide for a full walkthrough.
