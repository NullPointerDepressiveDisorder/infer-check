# diff

Compare outputs across different backends for the same model and prompts. Catches serving-layer bugs by holding the model and quantization constant while varying the inference path.

## CLI Reference

::: mkdocs-click
    :module: infer_check.cli
    :command: main
    :prog_name: infer-check
    :subcommand: diff
    :style: table
    :show_subcommand_aliases:

## How it works

1. **Build backends** -- creates a backend instance for each entry in `--backends`, using the shared `--model` and optional `--quant`.
2. **Baseline pass** -- generates outputs for all prompts using the first backend.
3. **Test passes** -- generates outputs for all prompts using each remaining backend.
4. **Compare** -- each test backend's outputs are compared against the baseline, producing per-prompt `ComparisonResult` objects with severity, text similarity, and flip metadata.
5. **Summary table** -- groups results by test backend and displays failure rate, flip rate, and mean similarity.

## Base URL matching

The `--base-urls` option is positionally matched to `--backends`. Use an empty entry for backends that don't need a URL (e.g., mlx-lm):

```bash
--backends "mlx-lm,openai-compat" \
--base-urls ",http://127.0.0.1:8000"
```

This gives mlx-lm no URL (local inference) and openai-compat the vllm-mlx server URL.

## Examples

mlx-lm vs vllm-mlx serving layer:

```bash
# Start vllm-mlx in another terminal:
# vllm-mlx serve mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --port 8000

infer-check diff \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --backends "mlx-lm,openai-compat" \
  --base-urls ",http://127.0.0.1:8000" \
  --prompts reasoning \
  --output ./results/diff/
```

With raw completions endpoint (no chat template):

```bash
infer-check diff \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --backends "mlx-lm,openai-compat" \
  --base-urls ",http://127.0.0.1:8000" \
  --prompts reasoning \
  --no-chat
```

## Output

```
                              Diff Summary
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ test_backend  ┃ failures ┃ failure_rate ┃ flip_rate ┃ mean_similarity ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ openai-compat │        0 │        0.00% │      0.0% │          1.0000 │
└───────────────┴──────────┴──────────────┴───────────┴─────────────────┘
```

A 100% similarity with 0% flip rate means the serving layer introduces zero divergence -- any output differences in production come from quantization, not the backend itself.

## Output format

Results are saved as a JSON array of `ComparisonResult` objects, each containing:

- `baseline` / `test` -- the `InferenceResult` from each backend
- `kl_divergence` -- KL(baseline || test) if logprobs are available
- `token_divergence_index` -- first token where the outputs differ
- `text_similarity` -- 0-1 similarity score
- `is_failure` -- true if similarity < 0.5
- `metadata` -- includes severity, flip status, answers, extraction strategy
