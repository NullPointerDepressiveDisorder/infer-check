# diff

Compare outputs across different backends for the same model and prompts. Catches serving-layer bugs by holding the model and quantization constant while varying the inference path.

## Usage

```bash
infer-check diff \
  --model MODEL_ID \
  --backends "backend1,backend2" \
  --prompts SUITE
```

## Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--model` | yes | | Model ID or HuggingFace path. |
| `--backends` | yes | | Comma-separated backend names. The first is the baseline; all others are tested against it. |
| `--prompts` | yes | | Bundled suite name or path to a `.jsonl` file. |
| `--output` | no | `./results/diff/` | Output directory for result JSON. |
| `--quant` | no | | Quantization level applied to all backends. |
| `--base-urls` | no | | Comma-separated base URLs, positionally matched to `--backends`. Use empty entries for backends that don't need a URL. |
| `--chat` / `--no-chat` | no | `--chat` | Use `/v1/chat/completions` for HTTP backends (applies chat template server-side). Pass `--no-chat` for raw `/v1/completions`. |
| `--max-tokens` | no | | Override default max tokens for generation. |
| `--num-prompts` | no | | Limit number of prompts to use. |

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
