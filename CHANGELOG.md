# Unreleased

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
