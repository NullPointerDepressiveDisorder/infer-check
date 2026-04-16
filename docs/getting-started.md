# Getting Started

## Requirements

- Python >= 3.11
- macOS with Apple Silicon (for mlx-lm backend) or Linux
- At least one supported backend

## Installation

```bash
pip install infer-check
```

For Apple Silicon users who want local inference via MLX:

```bash
pip install "infer-check[mlx]"
```

To verify the installation:

```bash
infer-check --version
```

## Global options

These options apply to all commands:

| Option | Default | Description |
|--------|---------|-------------|
| `--max-tokens` | `1024` | Default max tokens for generation. Applies to all prompts unless they specify their own. |
| `--num-prompts` | all | Limit the number of prompts to use from a suite. |
| `--version` | | Show version and exit. |

Global options go before the subcommand:

```bash
infer-check --max-tokens 512 --num-prompts 10 compare ...
```

## Your first test

The simplest way to start is with the `compare` command. It takes two model specs and runs them against a prompt suite:

```bash
infer-check compare \
  mlx-community/Llama-3.1-8B-Instruct-4bit \
  mlx-community/Llama-3.1-8B-Instruct-8bit \
  --prompts adversarial-numerics
```

This will:

1. Auto-detect the backend (mlx-lm for `mlx-community/` repos)
2. Load 30 adversarial-numerics prompts
3. Run each prompt through both models
4. Compare outputs and compute metrics (flip rate, KL divergence, text similarity)
5. Display a summary table and save JSON results to `./results/compare/`

## Prompt suites

`infer-check` ships with curated prompt suites targeting known quantization failure modes:

| Suite | Count | Purpose |
|-------|-------|---------|
| `reasoning` | 50 | Multi-step math and logic |
| `code` | 49 | Python, JSON, SQL generation |
| `adversarial-numerics` | 30 | IEEE 754 edge cases, overflow, precision |
| `long-context` | 10 | Tables and transcripts with recall questions |
| `quant-sensitive` | 20 | Multi-digit arithmetic, long CoT, precise syntax |
| `determinism` | 50 | High-entropy continuations for determinism testing |

All suites ship with the package -- no need to clone the repo. Use them by name:

```bash
--prompts reasoning
--prompts adversarial-numerics
```

### Custom prompt suites

Create a `.jsonl` file with one JSON object per line:

```json
{"id": "custom-001", "text": "What is 2^31 - 1?", "category": "math", "max_tokens": 512}
{"id": "custom-002", "text": "Write a Python function to sort a list.", "category": "code"}
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `id` | no | auto-generated UUID | Unique identifier |
| `text` | yes | | The prompt text |
| `category` | no | `"general"` | Task category (used for per-category breakdowns) |
| `max_tokens` | no | `1024` | Max generation tokens for this prompt |

Then pass the path:

```bash
--prompts ./my-custom-suite.jsonl
```

## Model resolution

The `compare` command auto-detects the backend from the model spec. You can also use explicit prefixes:

| Prefix | Backend | Example |
|--------|---------|---------|
| `mlx:` | mlx-lm | `mlx:mlx-community/Llama-3.1-8B-Instruct-4bit` |
| `ollama:` | openai-compat (Ollama) | `ollama:llama3.1:8b-instruct-q4_K_M` |
| `gguf:` | llama-cpp | `gguf:/path/to/model.gguf` |
| `vllm-mlx:` | vllm-mlx | `vllm-mlx:mlx-community/Llama-3.1-8B-Instruct-4bit` |

Without a prefix, resolution follows these rules:

1. Path ends in `.gguf` --> llama-cpp
2. Repo starts with `mlx-community/` or contains `-mlx` --> mlx-lm
3. Repo contains `gguf`, `bartowski`, `maziyarpanahi`, `mradermacher` --> llama-cpp
4. Contains `:` but no `/` (Ollama tag style) --> openai-compat
5. Fallback --> mlx-lm

## Output and results

All commands save JSON results to their `--output` directory (defaults to `./results/<command>/`). Result files include timestamps in their filenames to avoid overwrites.

Generate an HTML report from any results directory:

```bash
infer-check report ./results/ --format html
```

See [Interpreting Results](interpreting-results.md) for details on what the metrics mean.

## Next steps

- [Commands reference](commands/sweep.md) -- full details on every command
- [Backends](backends.md) -- supported backends and configuration
- [Interpreting Results](interpreting-results.md) -- understanding metrics and severity levels
