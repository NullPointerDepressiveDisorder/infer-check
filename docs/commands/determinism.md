# determinism

Test whether a backend produces identical outputs across repeated runs at temperature=0. A correctly implemented inference engine should produce bit-identical output for the same prompt and parameters every time.

## CLI Reference

::: mkdocs-click
    :module: infer_check.cli
    :command: main
    :prog_name: infer-check
    :subcommand: determinism
    :style: table
    :show_subcommand_aliases:

## How it works

1. **Force temperature=0** -- all prompts are run at temperature=0 to ensure deterministic sampling.
2. **Repeat runs** -- each prompt is sent to the backend N times (default 100).
3. **Count identical outputs** -- counts how many runs produced the exact same text as the most common output.
4. **Find divergence positions** -- for each pair of non-identical outputs, identifies the first token position where they diverge.
5. **Compute score** -- determinism score = identical_count / num_runs (1.0 = perfect).

## Example

```bash
infer-check determinism \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --backend mlx-lm \
  --prompts determinism \
  --runs 20 \
  --output ./results/determinism/
```

Output:

```
                          Determinism Summary
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ prompt_id                            ┃ runs ┃ identical ┃ determinism_score ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ d1a2b3c4-...                         │   20 │        20 │           100.00% │
│ e5f6a7b8-...                         │   20 │        20 │           100.00% │
│ ...                                  │   20 │        20 │           100.00% │
└──────────────────────────────────────┴──────┴───────────┴───────────────────┘

Overall determinism score: 100.00%
```

## What non-determinism means

A determinism score below 100% indicates that the backend is not producing consistent output at temperature=0. Common causes:

- **Floating-point non-determinism** in GPU kernels (different thread scheduling leads to different rounding)
- **KV cache bugs** that accumulate errors across requests
- **Batching interference** where concurrent requests affect each other's outputs
- **Buggy sampling implementations** that don't properly handle temperature=0

!!! warning
    Non-determinism at temperature=0 is always a bug in the inference engine, not a property of the model. A correct implementation must produce identical output for identical inputs.

## Output format

Results are saved as a JSON array of `DeterminismResult` objects, each containing:

- `prompt_id` -- reference to the prompt
- `num_runs` -- total number of runs
- `identical_count` -- how many runs matched the most common output
- `divergence_positions` -- token indices where any pair of runs diverged
- `determinism_score` -- identical_count / num_runs
