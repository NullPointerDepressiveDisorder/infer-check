# stress

Stress-test a backend with varying concurrency levels. Tests whether concurrent requests cause output divergence, KV cache corruption, or errors.

## CLI Reference

::: mkdocs-click
    :module: infer_check.cli
    :command: main
    :prog_name: infer-check
    :subcommand: stress
    :style: table
    :show_subcommand_aliases:

## How it works

1. **Baseline pass** -- runs all prompts at concurrency=1 (the first level). These outputs become the reference.
2. **Concurrent passes** -- for each concurrency level, runs all prompts with that many concurrent requests using `asyncio.Semaphore`.
3. **Consistency check** -- compares each concurrent output against the baseline (concurrency=1) output for the same prompt.
4. **Error tracking** -- counts failed requests at each concurrency level.
5. **Summary** -- displays output consistency and error count per concurrency level.

## Example

```bash
infer-check stress \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --backend openai-compat \
  --base-url http://127.0.0.1:8000 \
  --prompts reasoning \
  --concurrency 1,2,4,8 \
  --output ./results/stress/
```

Output:

```
                    Stress Test Summary
┏━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ concurrency ┃ errors ┃ output_consistency  ┃
┡━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│           1 │      0 │              100.00% │
│           2 │      0 │              100.00% │
│           4 │      0 │              100.00% │
│           8 │      0 │              100.00% │
└─────────────┴────────┴─────────────────────┘
```

## What to look for

- **Errors at high concurrency** -- the backend is failing under load. Check server logs for OOM, timeout, or connection errors.
- **Dropping output consistency** -- concurrent requests are interfering with each other. This is a strong signal of KV cache corruption or batch-dependent computation bugs.
- **Consistency drop at a specific threshold** -- if consistency drops sharply at concurrency N, the backend likely has a fixed-size buffer or cache that overflows at that level.

!!! tip
    For HTTP backends (openai-compat, vllm-mlx, llama-cpp), make sure the server is running before starting the stress test. The `--base-url` option lets you point to any running server.

## Output format

Results are saved as a JSON array of `StressResult` objects, each containing:

- `concurrency_level` -- the concurrency level tested
- `results` -- all `InferenceResult` objects from that level
- `error_count` -- number of failed requests
- `output_consistency` -- fraction of outputs matching the baseline (concurrency=1)
