# Unreleased

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
