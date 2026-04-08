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
