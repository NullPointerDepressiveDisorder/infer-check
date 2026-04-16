# sweep

Compare pre-quantized models against a baseline. Each model is a separate HuggingFace repo or local path. The first model (or `--baseline`) is the reference; all others are compared against it.

## CLI Reference

::: mkdocs-click
    :module: infer_check.cli
    :command: main
    :prog_name: infer-check
    :subcommand: sweep
    :style: table
    :show_subcommand_aliases:

## How it works

1. **Parse models** -- splits the `--models` string into label/path pairs and creates a backend for each.
2. **Baseline self-check** -- runs the baseline model twice on all prompts and compares the results. If the baseline isn't perfectly deterministic (50/50 identical), you'll see a warning. This tells you whether your comparison data is reliable.
3. **Test comparisons** -- runs every other quantization against the baseline and computes per-prompt metrics (text similarity, severity, KL divergence).
4. **Checkpoint saves** -- results are saved incrementally after each quantization level completes, so partial results survive interruptions.
5. **Summary table** -- displays a table grouped by quantization level with severity breakdowns.

## Model format

The `--models` option accepts comma-separated entries. Each entry can be:

- **Labeled**: `label=model_path` (e.g., `bf16=mlx-community/Meta-Llama-3.1-8B-Instruct-bf16`)
- **Unlabeled**: just the model path (the last path component becomes the label)

You need at least 2 models (one baseline + one test).

## Example

Full sweep across three quantization levels:

```bash
infer-check --max-tokens 512 --num-prompts 10 sweep \
  --models "bf16=mlx-community/Meta-Llama-3.1-8B-Instruct-bf16,\
            8bit=mlx-community/Meta-Llama-3.1-8B-Instruct-8bit,\
            4bit=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit" \
  --backend mlx-lm \
  --prompts reasoning \
  --output ./results/sweep/
```

Output:

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

The self-check row confirms the baseline is deterministic. The 4-bit row shows 35/50 severe divergences -- a clear signal of quantization-induced correctness degradation.

## Output format

Results are saved as a JSON file containing a `SweepResult` with:

- `model_id` -- the baseline model identifier
- `backend_name` -- the backend used
- `quantization_levels` -- list of quantization labels
- `comparisons` -- all per-prompt `ComparisonResult` objects
- `timestamp` -- when the sweep completed
- `summary` -- aggregate statistics (mean KL, failure counts)
