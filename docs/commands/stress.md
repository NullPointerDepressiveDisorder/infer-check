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

Batching servers compute each request together with whatever else is scheduled, and some GPU/Metal kernels give slightly different results depending on batch size and composition. Under greedy decoding a single flipped token changes the rest of the output. `stress` measures how often that happens.

1. **Reference (1 in flight):** every prompt runs alone, one at a time.
2. **Mixed batches (N in flight):** each prompt (the *target*) is sent at the same moment as N−1 *other* prompts drawn at random from the suite (seeded with `--seed`). The target's output is compared token-for-token with its solo run. `--trials` repeats this with freshly drawn co-scheduled prompts, so different batch compositions get tested.
3. **Summary:** per level, the fraction of targets identical to their solo run. Every diverged target is listed with the first differing token and the prompts it was sent with. Distinct error messages are reported too.

!!! warning "Needs an HTTP server"
    The in-process `mlx-lm` backend runs one request at a time, so concurrency can't affect it and the test could never fail. `stress` refuses to run on it. Point `--backend openai-compat --base-url ...` at llama-server, vllm-mlx, `mlx_lm.server`, vLLM, SGLang, etc. For llama-server, requests only batch when it is started with `--parallel N`.

## Example

```bash
mlx_lm.server --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --port 8181 &
infer-check stress \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --backend openai-compat --base-url http://127.0.0.1:8181 \
  --prompts reasoning --num-prompts 24 --concurrency 1,4,8
```

Measured on an M4 Max with `mlx_lm.server`, comparing its default batching against `--decode-concurrency 1 --prompt-concurrency 1` (24 reasoning prompts, 1 trial):

| In flight | Batching on: identical to solo | Batching off: identical to solo |
|---|---|---|
| 4 | 66.7% (8 diverged) | 91.7% (2 diverged) |
| 8 | 70.8% (7 diverged) | 87.5% (3 diverged) |

Batching roughly triples how often an output changes. The residual divergence with batching off is not run-to-run noise: the same prompt sent alone twice in a row came back identical. Some other server state, such as scheduling or prefix-cache reuse across different prompts, is likely involved but unconfirmed. At 8 in flight the server also refused some connections (`Cannot connect to server`), which shows up as errors.

## What to look for

- **Identical-to-solo below 100%:** outputs depend on what else the server is processing. Compare against the same server with batching disabled (or vLLM with `VLLM_BATCH_INVARIANT=1`) to separate batch effects from other causes.
- **Divergence at token 0:** the first token already differs, which points at prompt processing (prefill) rather than decoding.
- **Errors at high concurrency:** the server is failing under load; see `error_messages` in the results.

## Output format

Results are saved as a JSON array of `StressResult` objects, each containing:

- `concurrency_level` -- the concurrency level tested
- `results` -- all `InferenceResult` objects from that level
- `error_count` -- number of failed requests
- `output_consistency` -- fraction of outputs matching the baseline (concurrency=1)
