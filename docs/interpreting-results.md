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

## Correctness (answer keys)

Prompts can carry an answer key: an `answer_type` (`integer`, `decimal`, `boolean` or `choice`) and an `expected` value. For these prompts, infer-check grades each response against the key rather than comparing two outputs to each other. This is the most direct measure of whether a quantization or backend broke something.

The prompt sent to the model gets a fixed instruction appended: work step by step, then end with `Final answer: <answer>`. The grader reads the last `Final answer:` line, or the last `\boxed{...}` if there isn't one. Numbers may be written with commas, as `1.5e9`, `1.5 × 10^9`, `10^9`, or as a fraction `a/b`. Integers must match exactly, and are compared without any rounding: fractions are exact rationals, and `1.5 × 10^9`-style numbers are built straight from their digits. Decimals must be within a relative tolerance of 10⁻⁶ of the key; a key of 0 requires exactly 0.

Each response gets one of four grades:

| Grade | Meaning |
|-------|---------|
| `correct` | Parsed answer matches the key |
| `wrong` | Parsed answer doesn't match |
| `unparsed` | No `Final answer:` line or `\boxed{}`; counts as wrong, in accuracy and in the paired table |
| `truncated` | No answer because generation hit `max_tokens`; see the truncation rule below |

**Truncation rule.** When one configuration hits the token limit on a prompt that the other one finished, the truncation counts as wrong. This both lowers its accuracy and enters the paired table: a regression if the other side was right, both wrong otherwise. Running out of budget where the other model didn't is that model's own failure, typically a repetition loop. When both configurations hit the limit, the prompt is excluded (`both_truncated`), because the budget itself is the likely cause. In that case, raise `max_tokens`.

Comparisons are reported as a paired table: both right, **regressions** (A right, B wrong), **improvements** (A wrong, B right), and both wrong. The headline is "B regressed on *r* prompts and improved on *i*", with an exact McNemar test on those discordant pairs. A small p-value means the difference is unlikely to be chance. Each side's accuracy comes with a 95% Wilson confidence interval.

!!! tip "You need many prompts"
    With 100 prompts and a handful of regressions, McNemar can't detect anything. Use the `gsm8k` suite (1,319 problems) for quantization comparisons, and pick prompts the baseline mostly gets right. Two different wrong answers say nothing about whether quantization broke the model.

Bundled answer keys: all of `gsm8k`, and 43 of the 100 `adversarial-numerics` prompts (only those with a single unambiguous answer, computed by `scripts/build_answer_keys.py`).

## Flip rate

The flip rate is the fraction of **flip-eligible** prompts where the answer changed between the two models or backends.

For prompts with an answer key, a flip means one side is correct and the other wrong. An `unparsed` response and a one-sided `truncated` response are treated as wrong, so a correct answer becoming unparseable is a flip (and a regression); two-sided truncations are excluded. The correctness table reports separately how many regressions came from unparsed answers, so format failures stay distinguishable from wrong values.

For prompts without a key, infer-check falls back to extracting an answer based on the prompt's category:

| Category | Strategy | What it extracts |
|----------|----------|-----------------|
| Numeric prompts | `numeric` | Last number in the response (integers, decimals, scientific notation) |
| Boolean prompts | `boolean` | Yes/no with negation detection |
| Code prompts | `code` | Fenced code blocks with whitespace normalization |
| JSON prompts | `json` | Parsed and canonicalized JSON |
| Everything else | — | Not flip-eligible (`flipped` is `null`) |

Prompts in categories without an extractor used to be compared by raw text similarity, which made any rewording count as a "flip". They are now excluded from the flip rate.

A "flip" occurs when the extracted answers from two models don't match. For example:

- Model A answers "42", Model B answers "43" --> **flipped** (numeric)
- Model A answers "Yes", Model B answers "No" --> **flipped** (boolean)
- Model A and B give the same code but different commentary --> **not flipped** (code blocks match)

### Flip rate vs severity

These metrics capture different things:

- **Severity** measures how different the full text outputs are
- **Flip rate** measures whether the functional answer changed

A response can have "severe" text divergence but no flip (e.g., different reasoning paths reaching the same answer), or "minor" text divergence with a flip (e.g., nearly identical text except the final number is wrong).

For prompts with answer keys, prefer the correctness table over the flip rate: it says which side was right.

## KL divergence

KL divergence measures how different two models' next-token probability distributions are, reported as the mean per-token KL(baseline ‖ test) in nats. Unlike text similarity, it catches drift even when both models happen to generate the same text.

Every comparison records which of two methods produced its KL (`kl_method`). The two are not comparable and are never averaged together.

**Teacher-forced** (`teacher_forced`) is used when both backends can score, which today means both are mlx-lm. Both models are run over the baseline's exact output tokens, so at every position they predict the next token from identical context. The baseline's top 64 tokens are compared individually and all remaining probability is merged into one bucket. This makes the value a slight underestimate of the full-vocabulary KL, by a negligible amount in practice.

**Prefix** (`prefix`) is an approximate fallback for backends that can't score, such as HTTP servers. It uses the stored top-K distributions, but only at positions where both generations still share the same context. That means everything up to and including the first token where the outputs differ. Past that point the models are continuing different text, and comparing them is meaningless. Expect few positions and a rougher number.

| Mean KL (nats/token) | Interpretation |
|---------------|----------------|
| 0.0 | Identical distributions (expected for a baseline self-check) |
| < 0.01 | Negligible: typical of 8-bit vs. bf16 |
| 0.01 - 0.1 | Noticeable: typical of 4-bit, more so on small models |
| 0.1 - 1.0 | Large: the model's predictions are materially different |
| > 1.0 | Very large: likely broken |

As a reference point, Qwen2.5-0.5B-Instruct on 30 reasoning prompts measured 0.0006 for 8-bit and 0.092 for 4-bit against bf16. Small models are more sensitive to quantization than large ones.

Related fields:

- `top1_agreement`: the fraction of positions where both models' most likely token is the same.
- `kl_max` and `kl_p99`: localized spikes that the mean hides.
- `kl_per_token`: the per-position values.
- `kl_positions`: how many positions the KL was computed over.

!!! note
    Teacher-forced KL feeds the baseline's token IDs to the test model, so both must share a tokenizer, for example different quantizations of the same model. When neither method is possible, all KL fields are `null`.

!!! note "Stored distributions"
    The per-token top-K distributions used to compute KL are dropped from saved results once KL is computed, because they make files about 20× larger. Run with `infer-check --keep-distributions <command> ...` to keep them for your own analysis.

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
