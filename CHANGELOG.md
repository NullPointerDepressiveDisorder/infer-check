# Unreleased

- **Fixed: huge numbers in model answers no longer abort a run.** An answer like `10^369693099` (a plausible wrong answer to the 9^(9^9) digit-count prompt) raised `decimal.Overflow` and killed the whole compare/sweep. Parsing and grading now use a widened decimal range, so such answers are graded wrong. Exponents beyond even that range, or longer than Python's 4300-digit limit, are unparsed; the leading "10" of a failed `10^…` is never taken as the answer.
- **Fixed: fractions that can't be evaluated are unparsed.** `Final answer: 1/0` used to be graded as its numerator (correct against a key of 1), and fractions with parts over 4300 digits raised and aborted the run.
- **Fixed: fraction parsing.** `\\frac{a}{b}` (including inside `\\boxed{}`), signed operands (`1/-2`) and comma-grouped operands (`1,000/2`) are evaluated. Previously they parsed as a single component (1, 1 and 1000), marking wrong answers correct and correct answers wrong. Fractions that can't be evaluated (`\\frac{\\pi}{2}`) are unparsed rather than read as the number inside them.
- **Fixed: keyed prompts ask only for the graded quantity.** Several keyed `adversarial-numerics` prompts asked a second question ("Compute 170!. How many digits does it have?"), inviting a final answer the grader would compare against the wrong number. numerics-010, -032, -047, -049, -057, -067, -083 and -084 are reworded to a single question. numerics-075 (a bit string, which the numeric grader would read as decimal) is no longer keyed. `scripts/build_answer_keys.py` now removes keys from prompts it doesn't list, and a test guards against compound keyed prompts.
- **Fixed: positive-controls model cache** is keyed by the full repo ID, so `org-a/model` and `org-b/model` no longer share converted weights.
- **Documented: the KL tail floor.** Q's leftover mass (1 − Σ top-K) is floored, not treated as zero. Float32 top-K sums overshoot 1 by up to ~2e-6 (Q's computed leftover was ≤ 0 at 103 of 1310 positions, bf16 vs 4-bit), so "zero" there is indistinguishable from rounding. Treating it as exactly zero made KL infinite on ordinary runs. The floor keeps KL a lower bound.
- **Fixed: leading-dot decimals.** `.5`, `-.5` and `.5e2` parsed as 5, 5 and 500 (and `.5 × 10^3` as 5000), so correct decimal answers were graded wrong and `.5` matched an integer key of 5.
- **Fixed: answer-key canonicalization.** Keys are stored stripped, so a boolean key like `" yes "` no longer grades `Final answer: yes` wrong. Choice keys must be ASCII letters or digits, the only ones the grader can match; `"é"` previously loaded and could never be graded correct.
- **Fixed: positive controls with no gradable pairs.** When both sides truncated on every prompt, the script crashed formatting a missing accuracy. It now shows N/A and fails that check with an explanation (raise `max_tokens`).
- **Fixed: KL with masked tokens.** A `-inf` log-probability in the compared bins made the per-token KL `NaN` (0 × −∞), which poisoned every pooled mean. Zero-probability bins now contribute 0, by the 0·log 0 = 0 convention; mass where the test model has zero probability still gives `inf`.
- **Fixed: malformed answer keys fail when the suite loads**, not mid-run or silently. `integer` must be a whole number, `decimal` a finite number, `boolean` one of yes/no/true/false, `choice` a single letter or digit. Previously `boolean` + `"maybe"` loaded and graded every response wrong.
- **Fixed: token-level metrics across different tokenizers.**
  - Backends expose `tokenizer_fingerprint`: a hash of the vocabulary for mlx-lm, `None` (unknown) for HTTP backends.
  - When both sides are known and differ, KL, top-1 agreement and token divergence are skipped, with a warning; correctness and text similarity are still reported. Previously, comparing two model families on mlx-lm fed one model's token IDs to the other and reported meaningless KL.
- **Changed: positive controls check accuracy the right way.** The most-damaged level must show a statistically significant accuracy loss (exact McNemar p < 0.05). Accuracy is deliberately not required to fall monotonically, because small damage moves it a few prompts in either direction. Top-1 disagreement is now also checked for the noise sweep.
- **Docs:** the flip-rate section now matches the truncation rule: a one-sided truncation is a flip, a two-sided one is excluded.

- **Fixed: degradation cliff.**
  - It was computed by an unused function, walked quantization levels alphabetically ("3bit" < "4bit" < "8bit"), and was never written to the sweep summary, so the report never showed it.
  - It is now the first level, in the order you list them, with a statistically significant accuracy loss vs. the baseline (more regressions than improvements, exact McNemar p < 0.05). The sweep table and HTML report show it.
- **Changed: flip eligibility for prompts without answer keys.** When no answer can be extracted from one side, the prompt is no longer compared by raw text similarity; it's excluded from the flip rate.
- **Changed: `--max-tokens`.** Prompt-level `max_tokens` still takes precedence. The help text now says so, and a notice appears when the flag didn't apply to some or all prompts (every bundled suite sets its own).
- **Changed: HTML stress section** shows compared/diverged counts, the diverged prompts with their first differing token and co-scheduled prompts, and error messages.
- **Changed: `vllm-mlx` backend defaults to `chat=True`**, matching the CLI and `get_backend_for_model`.
- Unknown backend types now raise a clear `ValueError` instead of failing pydantic validation.
- **Removed: unused helpers** `token_level_divergence`, `sequence_similarity`, `find_divergence_onset`, `compute_sweep_statistics` and `_find_degradation_cliff` from `analysis.divergence`. Use `runner.first_divergence` for divergence onset.
- Cleared all bandit findings: narrowed exception handling in the JSON exporter, and marked false positives.

- **Changed: `stress` now tests batch composition, and can fail.**
  - Each prompt first runs alone as the reference. At each concurrency level it is then sent together with N−1 randomly drawn other prompts (`--seed`, `--trials`), and its output is compared token-for-token with the solo run.
  - Results list every diverged prompt (first differing token, co-scheduled prompts) and distinct error messages.
  - `stress` refuses to run on backends without real concurrency (in-process mlx-lm, where it could never fail). Backends gain a `supports_concurrency` check in the runner.
  - On `mlx_lm.server`, batching changed 8 of 24 outputs at 4 in flight, vs 2 of 24 with batching off.
- **Added: positive controls.**
  - `MLXBackend(logits_transform=...)` plus `backends.perturb.fixed_logit_noise(sigma)` builds a deliberately degraded model, the same in generation and scoring.
  - `scripts/positive_controls.py` runs two dose-response sweeps against bf16: 8→4→3→2-bit quantization, building 3/2-bit locally with `mlx_lm.convert`, and injected noise σ. It checks that KL and top-1 disagreement rise monotonically and that σ=0 gives KL exactly 0.

- **Changed: smaller result files.** Per-token top-K distributions are now dropped from saved results (and checkpoints) once KL has been computed. They were ~90% of file size, about 400 KB per prompt; a 12-prompt GSM8K compare went from 4.9 MB to 0.6 MB. Pass the global `--keep-distributions` flag to keep them.
- **Changed: HTML report leads with correctness.**
  - The executive summary shows graded comparisons, regressions and improvements. Its verdict names any statistically significant accuracy loss and no longer calls low text similarity "correctness issues".
  - Compare cards show accuracy with 95% intervals, the paired 2x2 table, McNemar p, and truncation/unparsed notes.
  - Sweeps get a per-level table (accuracy vs baseline, regressions/improvements, KL with method, top-1 agreement).
  - KL is labelled teacher-forced or prefix everywhere, and the cross-backend table shows each prompt's grades.

- **Added: correctness scoring against answer keys.** Prompts can carry `answer_type` (`integer` / `decimal` / `boolean` / `choice`) and `expected`. Graded prompts are sent with a fixed instruction to work step by step and end with `Final answer: <answer>`. Responses are graded `correct` / `wrong` / `unparsed` / `truncated`. Hitting the token limit on a prompt the other configuration finished counts as wrong; prompts where both hit the limit are excluded (`both_truncated`). `compare`, `diff` and `sweep` report a paired table (both right / regressions / improvements / both wrong) with an exact McNemar test and Wilson 95% intervals on accuracy (`CompareResult.correctness`, sweep `summary["correctness_by_quant"]`).
- **Added: `gsm8k` suite.** The 1,319-problem GSM8K test split, all answer-keyed, imported by `scripts/import_gsm8k.py` (MIT, Cobbe et al. 2021). 43 of the 100 `adversarial-numerics` prompts now have answer keys computed by `scripts/build_answer_keys.py`; only prompts with a single unambiguous answer are keyed.
- **Changed: flip rate.** For graded prompts, a flip means one side correct and the other wrong. Prompts whose category has no answer extractor were previously "flipped" whenever the text differed at all (raw similarity < 0.85). They are now excluded (`flipped = null`), and the flip rate is computed over eligible prompts only.
- **Fixed: README no longer claims "severe" means "functionally wrong".** It means low text similarity.

- **Fixed (KL divergence):** the old metric compared distributions at every position, including after the two generations had diverged, when the models were predicting continuations of different text. Combined with a 1e-10 floor for missing tokens, this produced meaningless values (~19 nats/token for 4-bit vs. bf16). KL is now either:
  - **teacher-forced** (mlx-lm): both models score the baseline's exact tokens, so every position compares the same context. The baseline's top 64 tokens are compared individually and all remaining probability is merged into one bucket, which makes the value a slight underestimate of the full-vocabulary KL.
  - **prefix** (fallback for other backends): stored top-K distributions, restricted to positions where both generations still share a context.
- New comparison fields: `kl_method`, `kl_max`, `kl_p99`, `kl_positions`, `kl_per_token`, `top1_agreement`. `CompareResult` gains `kl_method` and `mean_top1_agreement`. Sweep summaries report KL per quantization level (`kl_by_quant`) instead of one number pooled across levels.
- The CLI sweep and compare tables show mean KL (labelled by method) and top-1 agreement.
- Backends gain `supports_scoring` and `score()` (implemented for mlx-lm).
- Removed `analysis.divergence.kl_divergence`, which computed `p·log(p/q)` over chosen-token log-probs only and was not a KL divergence.

- **Fixed (MLX backend):** generation now uses `mlx_lm.stream_generate`. Previously the backend looped over `generate_step`, which (a) never stopped at the end-of-message token, so outputs ran on into invented extra turns, (b) was capped at 256 tokens regardless of `max_tokens`, and (c) decoded one token at a time, mangling multi-byte characters. Results produced by earlier versions of the MLX backend overstate divergence and should be regenerated.
- **Fixed (MLX backend):** chat-templated prompts no longer get a second BOS token.
- **Changed (MLX backend):** removed the silent fallback to `mlx_lm.generate`; generation errors now propagate.
- Token divergence is computed on token IDs when both results have them.
- All bundled prompt suites now carry stable IDs (`reasoning-001`, `numerics-001`, …) so results can be joined across runs. Prompts in custom suites without an `id` get one derived from their text instead of a random UUID; duplicate IDs are rejected. `scripts/add_prompt_ids.py` assigns IDs to new prompts.
- `InferenceResult` gains `token_ids`, and every backend now records `metadata["finish_reason"]` (`"stop"` / `"length"`) and `metadata["prompt_tokens"]` where the server exposes them. llama.cpp requests raw token IDs via `return_tokens`.

# 0.2.3 (2026-04-07)

- Added `--num-prompts` option to all CLI subcommands to limit the number of prompts used in a task.
- Added global `--max-tokens` flag (defaults to 1024) to the main CLI.
- Increased default `max_tokens` for all prompts from 256 to 1024.

# 0.1.0 (2026-03-11)

- Initial release
- Commands: sweep, diff, stress, determinism, report
- Backends: mlx-lm, llama.cpp, vllm-mlx, openai-compat
- Baseline self-check (run-twice determinism validation)
- Severity tiers: identical / minor / moderate / severe
- Chat template auto-detection for instruct models
- Comprehensive error handling across all backends
