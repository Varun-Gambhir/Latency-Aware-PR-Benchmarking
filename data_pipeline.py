"""
data_pipeline.py
────────────────
Phase 1 + 2: Mine merged HFT PRs from GitHub and convert raw PR bodies
into clean zero-shot coding prompts via an LLM call.

Usage:
    python data_pipeline.py --output hft_benchmark.json --max_prs 200
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import google.generativeai as genai
from github import Github, GithubException
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

HFT_REPOS: list[str] = [
    "quickfix/quickfix",
    "xerial/snappy-java",        # compression relevant to tick data
    "zeromq/libzmq",             # low-latency messaging
    "google/flatbuffers",        # zero-copy serialization
    "chronoxor/CppBenchmark",    # C++ latency benchmarks
    "martingale-cs/StrategyStudio",  # trading strategy infra
]

HFT_KEYWORDS: list[str] = [
    "lock-free",
    "lock free",
    "latency",
    "SPSC",
    "SIMD",
    "atomic",
    "memory_order",
    "wait-free",
    "cache line",
    "cacheline",
    "zero-copy",
    "busy-spin",
    "high frequency",
]

CLEANER_SYSTEM_PROMPT = """You are a senior C++ HFT engineer. 
Your job is to convert a messy GitHub PR description into a clean, 
self-contained, zero-shot coding instruction for an LLM.

Rules:
- Remove ALL GitHub metadata: issue refs (#42), review comments (LGTM, nit), CI status.
- Remove author opinions and merge noise.
- Preserve the *technical intent* of the change.
- Output a single, imperative instruction starting with a verb (e.g. "Implement…", "Optimise…", "Refactor…").
- Maximum 3 sentences.
- Output ONLY the instruction. No preamble, no quotes."""


# ──────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────

@dataclass
class BenchmarkSample:
    id: str
    repo: str
    pr_number: int
    raw_title: str
    clean_prompt: str
    reference_code: str          # unified diff from the PR
    pr_url: str


# ──────────────────────────────────────────────
# GitHub helpers
# ──────────────────────────────────────────────

def _pr_matches_keywords(pr_body: str, title: str) -> bool:
    """Return True if any HFT keyword appears in the PR title or body."""
    text = (title + " " + (pr_body or "")).lower()
    return any(kw.lower() in text for kw in HFT_KEYWORDS)


def _extract_diff(pr) -> str:
    """
    Pull the unified diff for a PR.
    Focuses on .cpp / .h / .hpp files and caps at 4 000 chars to
    stay within LLM context limits.
    """
    diff_parts: list[str] = []
    try:
        files = list(pr.get_files())
    except GithubException:
        return ""

    for f in files:
        if not any(f.filename.endswith(ext) for ext in (".cpp", ".cc", ".h", ".hpp", ".cxx")):
            continue
        if f.patch:
            diff_parts.append(f"// File: {f.filename}\n{f.patch}")

    combined = "\n\n".join(diff_parts)
    return combined[:4000] if combined else ""


def _append_raw_pr(record: dict, path: str) -> None:
    """
    Append a single raw-PR record to a newline-delimited JSON file
    (one JSON object per line).  Creates the file on first write.
    Safe to call after every match — if the run crashes, already-written
    records are preserved and can be replayed without re-hitting GitHub.
    """
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_prs(
    token: str,
    max_prs: int = 200,
    raw_output_path: str = "raw_prs.ndjson",
) -> list[dict]:
    """
    Query each repo for merged PRs that contain HFT keywords.
    Appends each match to *raw_output_path* immediately on discovery,
    then returns the full list for the cleaning phase.
    """
    g = Github(token)
    raw_samples: list[dict] = []

    # Truncate/create the file at the start of a fresh run
    open(raw_output_path, "w").close()
    print(f"📝  Streaming raw matches → {raw_output_path}")

    for repo_name in HFT_REPOS:
        print(f"\n🔍  Scanning {repo_name} …")
        try:
            repo = g.get_repo(repo_name)
        except GithubException as exc:
            print(f"   ⚠  Could not access {repo_name}: {exc}")
            continue

        pulls = repo.get_pulls(state="closed", sort="updated", direction="desc")

        for pr in pulls:
            if len(raw_samples) >= max_prs:
                break
            if not pr.merged:
                continue
            if not _pr_matches_keywords(pr.body or "", pr.title):
                continue

            diff = _extract_diff(pr)
            if not diff:
                continue

            record = {
                "repo": repo_name,
                "pr_number": pr.number,
                "raw_title": pr.title,
                "raw_body": (pr.body or "")[:2000],
                "diff": diff,
                "pr_url": pr.html_url,
            }

            # ── Persist immediately, before anything else ──
            _append_raw_pr(record, raw_output_path)

            raw_samples.append(record)
            print(f"   ✓  #{pr.number}  {pr.title[:70]}")

        if len(raw_samples) >= max_prs:
            break

    print(f"\n📦  Collected {len(raw_samples)} raw PR samples.")
    return raw_samples


# ──────────────────────────────────────────────
# LLM-based prompt cleaner  (Phase 2)
# ──────────────────────────────────────────────

def _init_gemini(api_key: str) -> genai.GenerativeModel:
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=CLEANER_SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(temperature=0.2, max_output_tokens=200),
    )


def clean_prompt_with_llm(
    model: genai.GenerativeModel,
    raw_title: str,
    raw_body: str,
    retries: int = 3,
) -> str:
    """
    Ask the LLM to distil the PR title + body into a concise coding task.
    Falls back to the raw title on repeated failure.
    """
    user_msg = f"PR Title: {raw_title}\n\nPR Body:\n{raw_body}"

    for attempt in range(retries):
        try:
            response = model.generate_content(user_msg)
            cleaned = response.text.strip()
            # Basic sanity check: non-empty and shorter than input
            if cleaned and len(cleaned) < len(raw_body) + 50:
                return cleaned
        except Exception as exc:
            print(f"   ⚠  LLM attempt {attempt + 1} failed: {exc}")
            time.sleep(2 ** attempt)

    # Graceful fallback
    return raw_title


def refine_dataset(
    raw_samples: list[dict],
    gemini_api_key: str,
) -> list[BenchmarkSample]:
    """
    Phase 2: iterate over raw samples, clean each prompt, assemble BenchmarkSample.
    """
    model = _init_gemini(gemini_api_key)
    refined: list[BenchmarkSample] = []

    print("\n🧹  Cleaning prompts with Gemini …")
    for idx, raw in enumerate(tqdm(raw_samples, unit="PR")):
        clean = clean_prompt_with_llm(model, raw["raw_title"], raw["raw_body"])
        sample = BenchmarkSample(
            id=f"HFT-{idx:04d}",
            repo=raw["repo"],
            pr_number=raw["pr_number"],
            raw_title=raw["raw_title"],
            clean_prompt=clean,
            reference_code=raw["diff"],
            pr_url=raw["pr_url"],
        )
        refined.append(sample)
        # Polite rate-limit: ~30 req/min on free tier
        time.sleep(0.5)

    return refined


# ──────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────

def save_benchmark(samples: list[BenchmarkSample], output_path: str) -> None:
    records = [asdict(s) for s in samples]
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    print(f"\n✅  Saved {len(records)} samples → {output_path}")


def load_benchmark(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ──────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="HFT Benchmark Data Pipeline")
    parser.add_argument("--output", default="hft_benchmark.json", help="Output JSON path")
    parser.add_argument("--raw_output", default="raw_prs.ndjson", help="Incremental raw-PR append file (NDJSON)")
    parser.add_argument("--max_prs", type=int, default=200, help="Max PRs to collect")
    parser.add_argument(
        "--github_token",
        default=os.getenv("GITHUB_TOKEN"),
        help="GitHub personal access token (or set GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--gemini_api_key",
        default=os.getenv("GEMINI_API_KEY"),
        help="Google Gemini API key (or set GEMINI_API_KEY env var)",
    )
    args = parser.parse_args()

    if not args.github_token:
        raise ValueError("GitHub token is required. Set GITHUB_TOKEN or pass --github_token.")
    if not args.gemini_api_key:
        raise ValueError("Gemini API key is required. Set GEMINI_API_KEY or pass --gemini_api_key.")

    # Phase 1 ──────────────────────────────────
    raw_samples = collect_prs(
        token=args.github_token,
        max_prs=args.max_prs,
        raw_output_path=args.raw_output,
    )

    if not raw_samples:
        print("⚠  No matching PRs found. Try expanding HFT_REPOS or HFT_KEYWORDS.")
        return

    # Phase 2 ──────────────────────────────────
    refined = refine_dataset(raw_samples, args.gemini_api_key)

    # Persist ──────────────────────────────────
    save_benchmark(refined, args.output)


if __name__ == "__main__":
    main()