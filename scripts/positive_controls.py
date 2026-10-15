"""Positive controls: show that infer-check detects damage it is known to be there.

Two dose-response experiments against a bf16 baseline:

1. Quantization: 8 -> 4 -> 3 -> 2 bits.
2. Injected logit noise: a fixed random bias of size sigma added to every logit
   (``infer_check.backends.perturb.fixed_logit_noise``).

Checks, per experiment:

- KL and top-1 disagreement must rise monotonically with damage. They are
  near-deterministic, so any inversion means the metrics are broken.
- sigma = 0 must give KL exactly 0.
- The most-damaged level must show a statistically significant accuracy loss
  (more regressions than improvements, exact McNemar p < 0.05).

Accuracy is deliberately *not* required to fall monotonically: at small damage
it moves by a few prompts in either direction (in one 60-prompt run, 8-bit went
39% -> 46% and sigma=0.25 39% -> 44%, both p > 0.25). That is sampling noise,
and a strict check would fail on honest results. Suites without answer keys
skip the accuracy check. The script prints a PASS/FAIL/SKIP line per check and
saves the raw results.

Usage (Apple Silicon):
    python scripts/positive_controls.py                      # Qwen2.5-0.5B, 100 GSM8K prompts
    python scripts/positive_controls.py --num-prompts 300
    python scripts/positive_controls.py --base mlx-community/Llama-3.2-1B-Instruct --prompts gsm8k

The 3- and 2-bit models are built locally with mlx_lm.convert (cached in
--models-dir); 8- and 4-bit use the published mlx-community conversions when
they exist.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from infer_check.backends.base import BackendAdapter
from infer_check.backends.mlx_lm import MLXBackend
from infer_check.backends.perturb import fixed_logit_noise
from infer_check.prompt_suites import get_suite_path
from infer_check.runner import TestRunner
from infer_check.suites.loader import load_suite


def local_quant(base_repo: str, bits: int, models_dir: Path) -> str:
    """Path to a locally converted ``bits``-bit model, building it if needed."""
    from mlx_lm import convert

    out = models_dir / f"{base_repo.split('/')[-1]}-{bits}bit"
    if not (out / "config.json").exists():
        print(f"building {bits}-bit model at {out} ...", flush=True)
        convert(base_repo, mlx_path=str(out), quantize=True, q_bits=bits)
    return str(out)


def published_or_local(base_repo: str, bits: int, models_dir: Path) -> str:
    """Prefer mlx-community's conversion of ``<base>-<bits>bit``; fall back to converting locally."""
    from huggingface_hub import repo_exists

    stem = base_repo.removesuffix("-bf16")
    candidate = f"{stem}-{bits}bit"
    with contextlib.suppress(Exception):  # offline or hub error: build locally instead
        if repo_exists(candidate):
            return candidate
    return local_quant(base_repo, bits, models_dir)


def fmt_row(label: str, kl: dict[str, Any], corr: dict[str, Any] | None) -> str:
    kl_s = f"{kl['mean_kl']:.4f}" if kl.get("mean_kl") is not None else "   N/A"
    top1 = f"{kl['mean_top1_agreement']:.2%}" if kl.get("mean_top1_agreement") is not None else "N/A"
    if corr:
        acc = f"{corr['a']['accuracy']:.1%} -> {corr['b']['accuracy']:.1%}"
        flips = f"{corr['regressions']:>3} / {corr['improvements']:<3} p={corr['mcnemar_p']:.2g}"
    else:
        acc, flips = "—", "—"
    return f"  {label:12} {kl_s:>8} {top1:>8}   {acc:>17}   {flips}"


def disagreement(stats: dict[str, Any]) -> float | None:
    """1 - top-1 agreement, or None when unavailable."""
    top1 = stats.get("mean_top1_agreement")
    return None if top1 is None else 1 - top1


def check_detects_damage(name: str, table: dict[str, Any] | None, alpha: float = 0.05) -> bool:
    """PASS if ``table`` (a paired-correctness table) shows a significant accuracy loss."""
    if not table:
        print(f"  SKIP: {name}: suite has no answer keys")
        return True
    ok = table["regressions"] > table["improvements"] and table["mcnemar_p"] < alpha
    print(
        f"  {'PASS' if ok else 'FAIL'}: {name}: {table['regressions']} regressions vs "
        f"{table['improvements']} improvements, p={table['mcnemar_p']:.2g}"
    )
    return ok


def check_monotonic(name: str, values: list[float | None], eps: float = 1e-9) -> bool:
    vals = [v for v in values if v is not None]
    ok = len(vals) == len(values) and all(b >= a - eps for a, b in zip(vals, vals[1:], strict=False))
    print(f"  {'PASS' if ok else 'FAIL'}: {name} non-decreasing: {', '.join(f'{v:.4f}' for v in vals)}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16", help="bf16 baseline repo")
    parser.add_argument("--prompts", default="gsm8k")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--bits", default="8,4,3,2", help="Quantization levels, in decreasing precision")
    parser.add_argument("--sigmas", default="0,0.25,0.5,1,2", help="Logit-noise levels, increasing")
    parser.add_argument("--models-dir", type=Path, default=Path.home() / ".cache" / "infer-check" / "models")
    parser.add_argument("--out", type=Path, default=Path("results/positive_controls"))
    args = parser.parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    prompts = load_suite(get_suite_path(args.prompts))[: args.num_prompts]
    runner = TestRunner(cache_dir=tempfile.mkdtemp())
    all_ok = True

    # ── 1. Quantization dose-response ──────────────────────────────
    bits = [int(b) for b in args.bits.split(",")]
    levels = ["bf16", *[f"{b}bit" for b in bits]]
    backends: dict[str, BackendAdapter] = {"bf16": MLXBackend(args.base, quantization="bf16")}
    for b in bits:
        backends[f"{b}bit"] = MLXBackend(published_or_local(args.base, b, args.models_dir), quantization=f"{b}bit")
    sweep = asyncio.run(runner.sweep(backends, prompts, levels, model_id=args.base))
    sweep.save(args.out / "quantization_sweep.json")

    print(f"\nQuantization dose-response ({len(prompts)} {args.prompts} prompts, vs bf16)")
    print(f"  {'level':12} {'mean KL':>8} {'top-1':>8}   {'accuracy':>17}   regr / impr")
    kl_by = sweep.summary["kl_by_quant"]
    corr_by = sweep.summary["correctness_by_quant"]
    for q in levels:
        print(fmt_row(q + (" (self)" if q == "bf16" else ""), kl_by[q], corr_by.get(q)))
    all_ok &= check_monotonic("KL with fewer bits", [kl_by[q]["mean_kl"] for q in levels])
    all_ok &= check_monotonic(
        "top-1 disagreement with fewer bits",
        [disagreement(kl_by[q]) for q in levels],
    )
    all_ok &= check_detects_damage(f"significant accuracy loss at {levels[-1]}", corr_by.get(levels[-1]))

    # ── 2. Injected-noise dose-response ────────────────────────────
    # One sweep: the bf16 baseline is generated once, every σ is scored against it.
    sigmas = [float(s) for s in args.sigmas.split(",")]
    labels = [f"σ={s:g}" for s in sigmas]
    noise_backends: dict[str, BackendAdapter] = {"bf16": MLXBackend(args.base, quantization="bf16")}
    for sigma, label in zip(sigmas, labels, strict=True):
        noise_backends[label] = MLXBackend(
            args.base, quantization=label, logits_transform=fixed_logit_noise(sigma, seed=0)
        )
    noise = asyncio.run(runner.sweep(noise_backends, prompts, ["bf16", *labels], model_id=args.base))
    noise.save(args.out / "noise_sweep.json")

    print("\nInjected logit noise (bf16 + fixed N(0, σ²) logit bias, vs bf16)")
    print(f"  {'σ':12} {'mean KL':>8} {'top-1':>8}   {'accuracy':>17}   regr / impr")
    nkl = noise.summary["kl_by_quant"]
    ncorr = noise.summary["correctness_by_quant"]
    for label in labels:
        print(fmt_row(label, nkl[label], ncorr.get(label)))
    kls = [nkl[label]["mean_kl"] for label in labels]
    if sigmas[0] == 0:
        zero_ok = kls[0] == 0.0
        print(f"  {'PASS' if zero_ok else 'FAIL'}: σ=0 gives KL exactly 0 (got {kls[0]})")
        all_ok &= zero_ok
    all_ok &= check_monotonic("KL with more noise", kls)
    all_ok &= check_monotonic(
        "top-1 disagreement with more noise",
        [disagreement(nkl[label]) for label in labels],
    )
    all_ok &= check_detects_damage(f"significant accuracy loss at {labels[-1]}", ncorr.get(labels[-1]))

    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "quantization": {"kl": kl_by, "correctness": corr_by},
                "noise": {"kl": nkl, "correctness": ncorr},
            },
            indent=2,
        )
    )
    print(f"\n{'ALL POSITIVE CONTROLS PASSED' if all_ok else 'SOME POSITIVE CONTROLS FAILED'}; results in {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
