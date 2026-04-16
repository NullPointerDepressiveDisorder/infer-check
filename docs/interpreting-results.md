# Interpreting Results

This guide explains the metrics `infer-check` computes and how to interpret them.

## Severity tiers

Every per-prompt comparison is classified into one of four severity tiers based on text similarity:

| Severity | Similarity range | Meaning |
|----------|-----------------|---------|
| **identical** | 1.0 | Outputs are character-for-character the same. |
| **minor** | >= 0.8 | Small wording differences. The answer is functionally the same. |
| **moderate** | >= 0.5 | Significant differences, but some overlap. May or may not affect correctness. |
| **severe** | < 0.5 | Outputs are fundamentally different. Likely a correctness failure. |

Text similarity is computed using `difflib.SequenceMatcher`, which measures the ratio of matching characters between the two outputs.

## Flip rate

The flip rate is the fraction of prompts where the **functional answer** changed between the two models or backends. Unlike text similarity (which measures surface-level text overlap), flip rate uses answer extraction to determine whether the actual answer is different.

### Answer extraction strategies

`infer-check` extracts the functional answer from each response based on the prompt's category:

| Category | Strategy | What it extracts |
|----------|----------|-----------------|
| Numeric prompts | `numeric` | Last number in the response (integers, decimals, scientific notation) |
| Boolean prompts | `boolean` | Yes/no with negation detection |
| Code prompts | `code` | Fenced code blocks with whitespace normalization |
| JSON prompts | `json` | Parsed and canonicalized JSON |
| Everything else | `raw` | Full lowercased text |

A "flip" occurs when the extracted answers from two models don't match. For example:

- Model A answers "42", Model B answers "43" --> **flipped** (numeric)
- Model A answers "Yes", Model B answers "No" --> **flipped** (boolean)
- Model A and B give the same code but different commentary --> **not flipped** (code blocks match)

### Flip rate vs severity

These metrics capture different things:

- **Severity** measures how different the full text outputs are
- **Flip rate** measures whether the functional answer changed

A response can have "severe" text divergence but no flip (e.g., different reasoning paths reaching the same answer), or "minor" text divergence with a flip (e.g., nearly identical text except the final number is wrong).

Flip rate is generally the more actionable metric for assessing correctness.

## KL divergence

KL divergence (Kullback-Leibler divergence) measures how different the token probability distributions are between two backends. It's computed as KL(baseline || test) -- how much information is lost when using the test distribution to approximate the baseline.

| KL divergence | Interpretation |
|---------------|----------------|
| 0.0 | Identical distributions |
| < 0.01 | Very similar -- negligible difference |
| 0.01 - 0.1 | Small differences in token probabilities |
| 0.1 - 1.0 | Moderate divergence -- different confidence levels |
| > 1.0 | Large divergence -- fundamentally different predictions |

!!! note
    KL divergence is only available when both backends provide logprobs or token probability distributions. Not all backends support this. When unavailable, the field is `null` in the output.

## Text similarity

A 0-1 score from `difflib.SequenceMatcher` measuring character-level overlap. Used to classify severity tiers.

| Score | Interpretation |
|-------|----------------|
| 1.0 | Identical output |
| 0.9+ | Very similar -- minor rewording |
| 0.7-0.9 | Moderately similar -- different phrasing, same general content |
| 0.5-0.7 | Partially similar -- some shared content |
| < 0.5 | Mostly different -- classified as "severe" |

## Token divergence index

The index of the first token where the baseline and test outputs diverge. A low index (e.g., 0-5) means the outputs diverge early and are likely completely different. A high index means the outputs share a common prefix before diverging.

## Determinism score

For determinism tests, the score is:

```
determinism_score = identical_count / num_runs
```

- **1.0** (100%) -- all runs produced identical output. The backend is deterministic.
- **< 1.0** -- some runs produced different output at temperature=0. This is a bug.

The `divergence_positions` field lists the token indices where pairs of runs first diverged, helping locate where non-determinism creeps in.

## Output consistency (stress tests)

For stress tests, output consistency is:

```
output_consistency = identical_to_baseline / total_compared
```

Where the baseline is the output from concurrency=1. This measures whether increasing concurrency changes the outputs.

- **100%** -- concurrent requests don't affect output. The backend is correct under load.
- **< 100%** -- some outputs changed under concurrency. Investigate KV cache correctness and batch-dependent computation.

## Per-category stats

The `compare` command breaks down results by prompt category (as defined in the prompt suite). Each category gets:

| Stat | Description |
|------|-------------|
| `count` | Number of prompts in this category |
| `flip_rate` | Fraction of prompts with answer flips |
| `mean_similarity` | Average text similarity |

This helps identify which task types are most affected by quantization or backend differences. Numerical tasks typically show the highest degradation.

## Reading the summary tables

### Sweep table

```
┃ quant_level       ┃ identical ┃ minor ┃ moderate ┃ severe ┃ mean_similarity ┃
```

- **Self-check row**: The baseline compared against itself. Should be 100% identical. If not, your baseline isn't deterministic and all other comparisons are unreliable.
- **Test rows**: Each quantization level compared against the baseline. More severe divergences = more correctness degradation.

### Compare table

```
┃ metric                              ┃ value ┃
```

Look at flip rate first -- it's the most direct measure of correctness. Then check severity tiers for the distribution of divergence. The flipped prompts detail table shows exactly which prompts broke and what the answers changed to.

### Diff table

```
┃ test_backend ┃ failures ┃ failure_rate ┃ flip_rate ┃ mean_similarity ┃
```

Failures indicate the backend returned errors. Flip rate and mean similarity show whether the serving layer changes outputs. Ideally, a diff test shows 0 failures, 0% flip rate, and 1.0 similarity.

### Stress table

```
┃ concurrency ┃ errors ┃ output_consistency ┃
```

Look for errors and consistency drops at higher concurrency levels. A sudden drop at a specific concurrency level often indicates a buffer overflow or cache corruption bug in the backend.
