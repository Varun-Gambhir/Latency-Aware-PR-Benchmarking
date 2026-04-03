"""
main.py
───────
Phase 6 orchestrator: runs the full benchmark pipeline end-to-end.

Steps:
  1. Load hft_benchmark.json
  2. Zero-shot evaluation (Gemini + 2 NIM models, 10 attempts each)
  3. Agentic evaluation (ReAct loop, 1 attempt each to control cost)
  4. Write results/comparison.csv + results/full_results.json

Usage:
    python main.py --benchmark hft_benchmark.json --output_dir results/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm

from agents import zero_shot_generate, agentic_generate
from metrics import evaluate_sample

load_dotenv()

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

ZERO_SHOT_MODELS = ["gemini", "llama3-70b", "mixtral-8x7b"]
N_ATTEMPTS_ZERO_SHOT = 10   # per prompt per model
N_ATTEMPTS_AGENTIC   = 1    # ReAct is expensive; 1 run per prompt

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _load_benchmark(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _safe_generate(fn, *args, retries: int = 3, **kwargs) -> list[str]:
    """Wrap a generation call with simple retry logic."""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            wait = 2 ** attempt
            print(f"   ⚠  Generation failed (attempt {attempt + 1}): {exc}. Retrying in {wait}s…")
            time.sleep(wait)
    return []   # empty list signals failure


def _metrics_row(
    sample_id: str,
    model: str,
    condition: str,
    metrics: dict,
) -> dict:
    return {
        "id":               sample_id,
        "model":            model,
        "condition":        condition,
        "pass_at_1":        round(metrics["pass_at_1"],           4),
        "pass_at_5":        round(metrics.get("pass_at_5", 0.0),  4),
        "mean_ext_bleu":    round(metrics["mean_extended_bleu"],  4),
        "max_ext_bleu":     round(metrics["max_extended_bleu"],   4),
    }


# ──────────────────────────────────────────────
# Core evaluation loops
# ──────────────────────────────────────────────

def run_zero_shot(
    samples: list[dict],
    api_keys: dict,
    n_attempts: int = N_ATTEMPTS_ZERO_SHOT,
) -> list[dict]:
    """
    Evaluate every model × every sample in zero-shot mode.
    Returns a flat list of metric row dicts.
    """
    rows: list[dict] = []

    for model in ZERO_SHOT_MODELS:
        print(f"\n{'='*60}")
        print(f"  Zero-Shot: {model}  ({n_attempts} attempts/prompt)")
        print(f"{'='*60}")

        for sample in tqdm(samples, desc=model, unit="prompt"):
            gens = _safe_generate(
                zero_shot_generate,
                prompt=sample["clean_prompt"],
                model_name=model,
                n_attempts=n_attempts,
                api_keys=api_keys,
            )
            if not gens:
                print(f"   ⚠  Skipping {sample['id']} (generation failed).")
                continue

            metrics = evaluate_sample(gens, sample["reference_code"])
            rows.append(
                _metrics_row(sample["id"], model, "zero_shot", metrics)
            )

    return rows


def run_agentic(
    samples: list[dict],
    api_keys: dict,
    n_attempts: int = N_ATTEMPTS_AGENTIC,
) -> list[dict]:
    """
    Evaluate every sample through the ReAct multi-agent pipeline.
    Runs for every model in ZERO_SHOT_MODELS.
    Returns a flat list of metric row dicts.
    """
    rows: list[dict] = []

    for model in ZERO_SHOT_MODELS:
        print(f"\n{'='*60}")
        print(f"  Agentic ReAct: {model}  (max {n_attempts} iterations / prompt)")
        print(f"{'='*60}")

        for sample in tqdm(samples, desc=f"agentic-{model}", unit="prompt"):
            gens = _safe_generate(
                agentic_generate,
                prompt=sample["clean_prompt"],
                model_name=model,
                n_attempts=n_attempts,
                api_keys=api_keys,
            )
            if not gens:
                print(f"   ⚠  Skipping {sample['id']} (generation failed).")
                continue

            metrics = evaluate_sample(gens, sample["reference_code"])
            rows.append(
                _metrics_row(sample["id"], model, "agentic_react", metrics)
            )

    return rows


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────

def _aggregate(rows: list[dict]) -> dict[tuple, dict]:
    """Compute mean metrics grouped by (model, condition)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["model"], r["condition"])
        groups.setdefault(key, []).append(r)

    summary: dict[tuple, dict] = {}
    for key, group in groups.items():
        n = len(group)
        summary[key] = {
            "model":         key[0],
            "condition":     key[1],
            "n_samples":     n,
            "mean_pass@1":   round(sum(r["pass_at_1"]     for r in group) / n, 4),
            "mean_pass@5":   round(sum(r["pass_at_5"]     for r in group) / n, 4),
            "mean_ext_bleu": round(sum(r["mean_ext_bleu"] for r in group) / n, 4),
        }
    return summary


def write_outputs(
    all_rows: list[dict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-sample CSV ──────────────────────────
    csv_path = output_dir / "full_results.csv"
    if all_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n📄  Per-sample results → {csv_path}")

    # ── Aggregated comparison CSV ───────────────
    summary = _aggregate(all_rows)
    cmp_path = output_dir / "comparison.csv"
    summary_rows = list(summary.values())
    if summary_rows:
        with open(cmp_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"📊  Aggregated comparison → {cmp_path}")

    # ── Pretty console table ────────────────────
    print("\n" + "─" * 70)
    print(f"  {'Model':<20} {'Condition':<16} {'Pass@1':>7} {'Pass@5':>7} {'ExtBLEU':>8}")
    print("─" * 70)
    for row in summary_rows:
        print(
            f"  {row['model']:<20} {row['condition']:<16} "
            f"{row['mean_pass@1']:>7.4f} {row['mean_pass@5']:>7.4f} "
            f"{row['mean_ext_bleu']:>8.4f}"
        )
    print("─" * 70)

    # ── Compute lift (agentic vs best zero-shot) ─
    zero_shot_bleu = [
        v["mean_ext_bleu"]
        for v in summary.values()
        if v["condition"] == "zero_shot"
    ]
    agentic_bleu = [
        v["mean_ext_bleu"]
        for v in summary.values()
        if v["condition"] == "agentic_react"
    ]
    if zero_shot_bleu and agentic_bleu:
        best_zs = max(zero_shot_bleu)
        best_ag = max(agentic_bleu)
        lift = (best_ag - best_zs) / best_zs * 100 if best_zs > 0 else float("inf")
        print(
            f"\n  🚀  Agentic lift over best zero-shot: "
            f"{best_ag:.4f} vs {best_zs:.4f}  ({lift:+.1f}%)"
        )


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="HFT Benchmark Orchestrator")
    parser.add_argument("--benchmark",  default="hft_benchmark.json", help="Path to benchmark JSON")
    parser.add_argument("--output_dir", default="results",            help="Directory for output CSVs")
    parser.add_argument("--max_samples", type=int, default=None,      help="Limit samples (useful for quick dev runs)")
    parser.add_argument("--skip_zero_shot",  action="store_true",     help="Skip zero-shot phase")
    parser.add_argument("--skip_agentic",    action="store_true",     help="Skip agentic phase")
    parser.add_argument("--gemini_api_key",  default=os.getenv("GEMINI_API_KEY"))
    parser.add_argument("--nvidia_api_key",  default=os.getenv("NVIDIA_NIM_API_KEY"))
    args = parser.parse_args()

    api_keys = {
        "gemini": args.gemini_api_key,
        "nvidia": args.nvidia_api_key,
    }

    # Load benchmark
    print(f"📂  Loading benchmark: {args.benchmark}")
    samples = _load_benchmark(args.benchmark)
    if args.max_samples:
        samples = samples[: args.max_samples]
        print(f"   ↳  Truncated to {len(samples)} samples (--max_samples)")
    print(f"   ↳  {len(samples)} benchmark samples loaded.")

    all_rows: list[dict] = []

    # Phase 4: Zero-shot
    if not args.skip_zero_shot:
        zs_rows = run_zero_shot(samples, api_keys)
        all_rows.extend(zs_rows)

    # Phase 5+6: Agentic
    if not args.skip_agentic:
        ag_rows = run_agentic(samples, api_keys)
        all_rows.extend(ag_rows)

    # Write outputs
    write_outputs(all_rows, Path(args.output_dir))


if __name__ == "__main__":
    main()
