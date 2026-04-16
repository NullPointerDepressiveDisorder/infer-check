# compare

Head-to-head comparison of two models, quantizations, or backends. Auto-detects the backend from model specs, or accepts explicit prefixes.

## Usage

```bash
infer-check compare MODEL_A MODEL_B [OPTIONS]
```

## Arguments

| Argument | Description |
|----------|-------------|
| `MODEL_A` | Model spec for the first model -- HuggingFace repo, Ollama tag, or local GGUF path. |
| `MODEL_B` | Model spec for the second model. |

## Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--prompts` | no | `adversarial-numerics` | Bundled suite name or path to a `.jsonl` file. |
| `--output` | no | `./results/compare/` | Output directory for result JSON and reports. |
| `--base-url` | no | | Base URL override for HTTP backends. Applied to both models unless they resolve to mlx-lm. |
| `--label-a` | no | auto-derived | Custom label for model A. |
| `--label-b` | no | auto-derived | Custom label for model B. |
| `--report` / `--no-report` | no | `--report` | Generate an HTML comparison report after the run. |
| `--max-tokens` | no | | Override default max tokens for generation. |
| `--num-prompts` | no | | Limit number of prompts to use. |

## How it works

1. **Resolve models** -- each model spec is resolved to a backend type, model ID, and base URL using the [model resolution rules](../getting-started.md#model-resolution).
2. **Generate pass A** -- runs all prompts through the first backend.
3. **Generate pass B** -- runs all prompts through the second backend.
4. **Compare** -- for each prompt, computes text similarity, severity classification, and KL divergence (when logprobs are available).
5. **Answer extraction** -- extracts the functional answer from each response based on the prompt category (numeric, boolean, code, JSON, or raw text). Computes flip rate from extracted answers.
6. **Per-category breakdown** -- groups results by prompt category and computes per-category flip rate and mean similarity.
7. **Report** -- optionally generates an HTML report with detailed comparisons.

## Model spec prefixes

| Prefix | Backend | Example |
|--------|---------|---------|
| `mlx:` | mlx-lm | `mlx:mlx-community/Llama-3.1-8B-Instruct-4bit` |
| `ollama:` | openai-compat | `ollama:llama3.1:8b-instruct-q4_K_M` |
| `gguf:` | llama-cpp | `gguf:/path/to/model.gguf` |
| `vllm-mlx:` | vllm-mlx | `vllm-mlx:mlx-community/Llama-3.1-8B-Instruct-4bit` |

Without a prefix, the backend is auto-detected from the model path. See [Getting Started](../getting-started.md#model-resolution) for full resolution rules.

## Examples

Two MLX quantizations:

```bash
infer-check compare \
  mlx-community/Llama-3.1-8B-Instruct-4bit \
  mlx-community/Llama-3.1-8B-Instruct-8bit
```

MLX native vs Ollama GGUF:

```bash
infer-check compare \
  mlx-community/Llama-3.1-8B-Instruct-4bit \
  ollama:llama3.1:8b-instruct-q4_K_M
```

With custom labels and limited prompts:

```bash
infer-check --num-prompts 10 compare \
  mlx-community/Llama-3.1-8B-Instruct-4bit \
  mlx-community/Llama-3.1-8B-Instruct-8bit \
  --label-a "4bit" \
  --label-b "8bit" \
  --prompts reasoning \
  --no-report
```

## Output

The command displays three tables:

**Summary table** -- overall metrics:

| Metric | Description |
|--------|-------------|
| prompts | Total number of prompts tested |
| flip rate | Fraction of prompts where the extracted answer changed |
| mean KL divergence | Average KL(baseline \|\| test) across prompts |
| mean text similarity | Average text similarity (0-1) |
| identical / minor / moderate / severe | Severity tier counts |

**Per-category breakdown** -- flip rate and mean similarity by prompt category.

**Flipped prompts detail** -- for each prompt where the answer flipped, shows the prompt text, category, extraction strategy, both answers, and similarity score.

## Output format

Results are saved as a JSON file containing a `CompareResult` with:

- `model_a`, `model_b` -- model configuration labels
- `backend_a`, `backend_b` -- backend names
- `comparisons` -- all per-prompt `ComparisonResult` objects
- `flip_rate` -- fraction of prompts with answer flips
- `mean_kl_divergence` -- average KL divergence
- `mean_text_similarity` -- average text similarity
- `per_category_stats` -- per-category breakdown with flip rate, mean similarity, count
- `timestamp` -- when the comparison completed
