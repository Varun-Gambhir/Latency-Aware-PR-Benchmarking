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
# Multi-token pool
# ──────────────────────────────────────────────

class TokenPool:
    """
    Round-robin GitHub token manager.

    Rotation strategy:
      - Every successful API call advances to the next token (even spread).
      - On a rate-limit / 503, the current token is marked exhausted and the
        next one is tried immediately — no sleep yet.
      - Only when ALL tokens are exhausted does the pool back off
        (exponential, capped at 5 min) then reset all tokens.

    Usage:
        pool = TokenPool(["ghp_aaa", "ghp_bbb"])
        g    = pool.github_client()   # Github() for current token
        pool.rotate()                 # advance after a successful fetch
        pool.mark_exhausted(exc)      # flag current token, switch immediately
    """

    def __init__(self, tokens: list[str]) -> None:
        if not tokens:
            raise ValueError("TokenPool requires at least one GitHub token.")
        self._tokens: list[str] = list(tokens)
        self._exhausted: list[bool] = [False] * len(self._tokens)
        self._dead: list[bool] = [False] * len(self._tokens)   # permanent, never reset
        self._index: int = 0
        self._backoff_step: int = 0
        self._clients: dict[int, object] = {}

    # ── public interface ─────────────────────

    @property
    def current_token(self) -> str:
        return self._tokens[self._index]

    @property
    def active_count(self) -> int:
        """Number of tokens not permanently dead."""
        return sum(1 for d in self._dead if not d)

    def github_client(self):
        """Return (cached) Github client for the current token."""
        if self._index not in self._clients:
            self._clients[self._index] = Github(self._tokens[self._index])
        return self._clients[self._index]

    def rotate(self) -> None:
        """Advance to the next live token (called after every successful fetch)."""
        for offset in range(1, len(self._tokens) + 1):
            nxt = (self._index + offset) % len(self._tokens)
            if not self._dead[nxt]:
                self._index = nxt
                return

    def mark_exhausted(self, exc: Optional[Exception] = None) -> None:
        """
        Rate-limited: flag current token temporarily exhausted, switch immediately.
        All temporarily-exhausted tokens are reset after the backoff sleep.
        """
        label = f"token[{self._index}] ...{self.current_token[-6:]}"
        print(f"   🔴  {label} rate-limited ({exc or 'exhausted'})")
        self._exhausted[self._index] = True

        next_idx = self._next_available()
        if next_idx is not None:
            self._index = next_idx
            print(f"   🔄  Switched to token[{self._index}] ...{self.current_token[-6:]}")
        else:
            self._full_pool_backoff()

    def mark_dead(self, exc: Optional[Exception] = None) -> None:
        """
        Bad credential / 401: permanently remove this token from the pool.
        Unlike mark_exhausted, this is NEVER undone by backoff reset.
        Raises RuntimeError if no live tokens remain.
        """
        label = f"token[{self._index}] ...{self.current_token[-6:]}"
        print(f"   💀  {label} permanently removed — bad credentials ({exc})")
        self._dead[self._index] = True
        self._exhausted[self._index] = True   # also block it from backoff reset

        if self.active_count == 0:
            raise RuntimeError(
                "All GitHub tokens are invalid (401). "
                "Please check your tokens and try again."
            )

        next_idx = self._next_available()
        if next_idx is not None:
            self._index = next_idx
            print(f"   🔄  Switched to token[{self._index}] ...{self.current_token[-6:]}")

    @staticmethod
    def is_bad_credentials(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "401" in msg or "bad credentials" in msg

    @staticmethod
    def is_rate_limit_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(k in msg for k in ("503", "429", "403", "rate limit", "retry", "max retries"))

    # ── private helpers ──────────────────────

    def _next_available(self) -> Optional[int]:
        """Next token that is neither dead nor temporarily exhausted."""
        for offset in range(1, len(self._tokens) + 1):
            idx = (self._index + offset) % len(self._tokens)
            if not self._exhausted[idx] and not self._dead[idx]:
                return idx
        return None

    def _full_pool_backoff(self) -> None:
        self._backoff_step += 1
        wait = min(30 * (2 ** (self._backoff_step - 1)), 300)  # 30 60 120 240 300s
        alive = self.active_count
        print(
            f"\n   ⏳  All {alive} remaining token(s) temporarily exhausted. "
            f"Backing off {wait}s (round {self._backoff_step}) ...\n"
        )
        time.sleep(wait)
        # Reset only non-dead tokens
        self._exhausted = [d for d in self._dead]   # dead stay exhausted; others freed
        self._backoff_step = 0
        # Move index to first live token
        for i in range(len(self._tokens)):
            if not self._dead[i]:
                self._index = i
                break
        print(f"   ✅  Token pool reset — {alive} token(s) resuming.")


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

HFT_REPOS: list[str] = [
    # ── Core HFT / trading infra ──────────────────────────────────────────
    "quickfix/quickfix",             # FIX protocol engine
    "OnixS/fix-antenna-cpp",         # FIX engine, latency-focused
    "aeron-io/aeron",                # ultra-low-latency messaging
    "real-logic/aeron",              # same project, main mirror
    "real-logic/SBE",                # Simple Binary Encoding (FIX SBE)

    # ── Lock-free / concurrent data structures ────────────────────────────
    "cameron314/concurrentqueue",    # lock-free MPMC queue
    "cameron314/readerwriterqueue",  # lock-free SPSC queue
    "rigtorp/SPSCQueue",             # SPSC ring buffer
    "rigtorp/MPMCQueue",             # MPMC ring buffer
    "rigtorp/Seqlock",               # seqlock implementation
    "rigtorp/HashMap",               # low-latency hash map
    "facebook/folly",                # includes lock-free structures
    "abseil/abseil-cpp",             # Swiss tables, low-latency utils
    "efficient/libcds",              # Concurrent Data Structures library
    "khizmax/libcds",                # same lib, main repo
    "mpoeter/xenium",                # lock-free data structures research

    # ── Low-latency messaging / networking ────────────────────────────────
    "zeromq/libzmq",                 # ZeroMQ C++ core
    "nanomsg/nng",                   # nanomsg-next-gen
    "chronoxor/CppServer",           # async C++ networking
    "MengRao/WFMPMC",                # wait-free MPMC queue

    # ── Serialisation (zero-copy / SIMD) ─────────────────────────────────
    "google/flatbuffers",            # zero-copy serialisation
    "EsotericSoftware/flatbuffers",  # flatbuffers fork
    "fmtlib/fmt",                    # fast format library
    "simdjson/simdjson",             # SIMD JSON parser
    "lemire/simdjson",               # same project
    "nicowillis/fast_float",         # SIMD float parsing
    "fastfloat/fast_float",          # fast float parsing

    # ── Memory / allocators ───────────────────────────────────────────────
    "microsoft/mimalloc",            # fast allocator
    "jemalloc/jemalloc",             # jemalloc
    "gperftools/gperftools",         # tcmalloc + profiler

    # ── Benchmarking / profiling ──────────────────────────────────────────
    "chronoxor/CppBenchmark",        # latency benchmark harness
    "google/benchmark",              # Google microbenchmark
    "DigitalInBlue/Celero",          # C++ benchmark framework
    "martinus/nanobench",            # fast microbenchmark

    # ── SIMD / numeric ────────────────────────────────────────────────────
    "xtensor-stack/xsimd",           # SIMD wrappers
    "VcDevel/Vc",                    # portable SIMD
    "highway/highway",               # Google Highway SIMD
    "google/highway",

    # ── Atomics / memory model ────────────────────────────────────────────
    "crossbeam-rs/crossbeam",        # lock-free in Rust (good C++ diffs too)
    "boostorg/atomic",               # Boost.Atomic
    "boostorg/lockfree",             # Boost.Lockfree

    # ── Time / clock ─────────────────────────────────────────────────────
    "google/cctz",                   # fast time-zone handling
    "HowardHinnant/date",            # C++ date/time library
]

HFT_KEYWORDS: list[str] = [
    # Concurrency primitives
    "lock-free", "lock free", "lockfree",
    "wait-free", "wait free",
    "SPSC", "MPMC", "MPSC", "SPMC",
    "atomic", "memory_order", "compare_exchange", "fetch_add",
    "CAS", "ABA",
    # Latency / performance
    "latency", "throughput", "nanosecond", "tick-to-trade",
    "busy-spin", "busy spin", "spin-wait",
    "high frequency", "HFT", "ultra-low",
    # Memory / cache
    "cache line", "cacheline", "false sharing",
    "alignas", "memory pool", "arena allocat",
    "prefetch", "NUMA",
    # SIMD / vectorisation
    "SIMD", "AVX", "SSE", "intrinsic", "vectori",
    # Serialisation
    "zero-copy", "zero copy", "flatbuffer", "SBE",
    # Networking
    "kernel bypass", "RDMA", "DPDK", "busy poll",
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
    reference_code: str          # added lines only — plain C++ (used by metrics)
    raw_diff: str                # full unified diff patch (for human inspection)
    pr_url: str


# ──────────────────────────────────────────────
# GitHub helpers
# ──────────────────────────────────────────────

def _pr_matches_keywords(pr_body: str, title: str) -> bool:
    """Return True if any HFT keyword appears in the PR title or body."""
    text = (title + " " + (pr_body or "")).lower()
    return any(kw.lower() in text for kw in HFT_KEYWORDS)


def _diff_to_reference_code(patch: str) -> str:
    """
    Extract only the added lines from a unified diff patch,
    stripping the leading '+' so the result is plain C++ code.
    Skips diff headers (lines starting with '@@', '---', '+++').
    This is what the model should reproduce — not the diff format itself.
    """
    lines = []
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            lines.append(line[1:])   # strip the leading '+'
    return "\n".join(lines)


def _extract_diff(pr) -> tuple[str, str]:
    """
    Pull the unified diff for a PR and return TWO strings:
      - raw_diff:        the full patch (for context / human inspection)
      - reference_code:  only the added lines stripped of diff markers
                         (used as the ground-truth for metric comparison)

    Focuses on .cpp / .h / .hpp files, caps each at 4 000 chars.
    """
    raw_parts: list[str] = []
    ref_parts: list[str] = []
    try:
        files = list(pr.get_files())
    except GithubException:
        return "", ""

    for f in files:
        if not any(f.filename.endswith(ext) for ext in (".cpp", ".cc", ".h", ".hpp", ".cxx")):
            continue
        if f.patch:
            raw_parts.append(f"// File: {f.filename}\n{f.patch}")
            ref_parts.append(f"// File: {f.filename}\n{_diff_to_reference_code(f.patch)}")

    raw  = "\n\n".join(raw_parts)[:4000]
    ref  = "\n\n".join(ref_parts)[:4000]
    return raw, ref


def _append_raw_pr(record: dict, path: str) -> None:
    """
    Append a single raw-PR record to a newline-delimited JSON file
    (one JSON object per line).  Creates the file on first write.
    Safe to call after every match — if the run crashes, already-written
    records are preserved and can be replayed without re-hitting GitHub.
    """
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_seen(path: str) -> set[tuple[str, int]]:
    """
    Read an existing NDJSON checkpoint and return a set of
    (repo, pr_number) pairs that have already been collected.
    Returns an empty set if the file does not exist yet.
    """
    seen: set[tuple[str, int]] = set()
    if not os.path.exists(path):
        return seen
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                seen.add((rec["repo"], rec["pr_number"]))
            except (json.JSONDecodeError, KeyError):
                pass   # corrupt line — skip silently
    return seen


def collect_prs(
    tokens: list[str],
    max_prs: int = 200,
    raw_output_path: str = "raw_prs.ndjson",
) -> list[dict]:
    """
    Query each repo for merged PRs that contain HFT keywords.

    tokens: one or more GitHub PATs — rotates between them on rate-limit,
            backs off only when the entire pool is exhausted simultaneously.
    Appends each match to *raw_output_path* immediately on discovery,
    then returns the full list for the cleaning phase.
    """
    pool = TokenPool(tokens)

    # ── Resume: load already-collected (repo, pr_number) pairs ──────────
    seen: set[tuple[str, int]] = _load_seen(raw_output_path)
    raw_samples: list[dict] = []          # holds only NEW records this run

    if seen:
        print(f"♻️   Resuming — {len(seen)} PR(s) already in {raw_output_path}, skipping them.")
    else:
        print(f"🆕  No existing checkpoint found, starting fresh.")
    print(f"📝  Appending new matches → {raw_output_path}")

    for repo_name in HFT_REPOS:
        print(f"\n🔍  Scanning {repo_name} …")

        # ── Get repo object, rotating token on failure ───────────────
        repo = None
        for _ in range(len(pool._tokens) + 1):
            try:
                repo = pool.github_client().get_repo(repo_name)
                break
            except GithubException as exc:
                if TokenPool.is_bad_credentials(exc):
                    pool.mark_dead(exc)          # permanent — remove token
                elif TokenPool.is_rate_limit_error(exc):
                    pool.mark_exhausted(exc)     # temporary — rotate & retry
                else:
                    print(f"   ⚠  Could not access {repo_name}: {exc}")
                    break
        if repo is None:
            continue

        def _fresh_iter():
            return iter(pool.github_client().get_repo(repo_name)
                        .get_pulls(state="closed", sort="updated", direction="desc"))

        MAX_PAGE_RETRIES = 5
        retry_count = 0
        pr_iter = _fresh_iter()

        while len(seen) + len(raw_samples) < max_prs:
            try:
                pr = next(pr_iter)
                retry_count = 0          # successful fetch — reset counter
                pool.rotate()            # spread load across tokens evenly
            except StopIteration:
                break                    # no more PRs in this repo
            except Exception as exc:
                if TokenPool.is_bad_credentials(exc):
                    pool.mark_dead(exc)          # permanent — remove token, no retry
                elif TokenPool.is_rate_limit_error(exc):
                    pool.mark_exhausted(exc)     # temporary — rotate & back off
                else:
                    # Transient non-rate-limit error: simple backoff
                    retry_count += 1
                    if retry_count > MAX_PAGE_RETRIES:
                        print(f"   ✖  {repo_name}: gave up after {MAX_PAGE_RETRIES} retries ({exc})")
                        break
                    wait = 2 ** retry_count
                    print(f"   ⚠  GitHub error (attempt {retry_count}/{MAX_PAGE_RETRIES}): {exc}")
                    print(f"      Backing off {wait}s …")
                    time.sleep(wait)
                # Either way, rebuild iterator with current (possibly new) token
                try:
                    pr_iter = _fresh_iter()
                except Exception:
                    pass
                continue

            if not pr.merged:
                continue
            # ── Skip already-collected PRs ──────────────────────────────
            if (repo_name, pr.number) in seen:
                continue
            if not _pr_matches_keywords(pr.body or "", pr.title):
                continue

            raw_diff, reference_code = _extract_diff(pr)
            if not reference_code:
                continue

            record = {
                "repo": repo_name,
                "pr_number": pr.number,
                "raw_title": pr.title,
                "raw_body": (pr.body or "")[:2000],
                "raw_diff": raw_diff,
                "reference_code": reference_code,
                "pr_url": pr.html_url,
            }

            # ── Persist immediately, before anything else ──
            _append_raw_pr(record, raw_output_path)

            raw_samples.append(record)
            print(f"   ✓  [token {pool._index}] #{pr.number}  {pr.title[:70]}")

        if len(seen) + len(raw_samples) >= max_prs:
            break

    print(f"\n📦  Added {len(raw_samples)} new PR(s) this run  |  {len(seen) + len(raw_samples)} total in checkpoint.")
    return raw_samples


# ──────────────────────────────────────────────
# LLM-based prompt cleaner  (Phase 2)
# ──────────────────────────────────────────────

def _init_gemini(api_key: str) -> genai.GenerativeModel:
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash",
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


def _load_cleaned(path: str) -> tuple[list[BenchmarkSample], set[tuple[str, int]]]:
    """
    Load already-cleaned samples from an NDJSON checkpoint.
    Returns (list_of_samples, set_of_(repo, pr_number)_already_done).
    """
    samples: list[BenchmarkSample] = []
    done: set[tuple[str, int]] = set()
    if not os.path.exists(path):
        return samples, done
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                samples.append(BenchmarkSample(**rec))
                done.add((rec["repo"], rec["pr_number"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass   # corrupt line — skip
    return samples, done


def _append_cleaned(sample: BenchmarkSample, path: str) -> None:
    """Write one cleaned sample to the NDJSON checkpoint immediately."""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def refine_dataset(
    raw_samples: list[dict],
    gemini_api_key: str,
    cleaned_checkpoint: str = "cleaned_prs.ndjson",
) -> list[BenchmarkSample]:
    """
    Phase 2: clean each raw prompt via Gemini and write to *cleaned_checkpoint*
    immediately after each call — crash-safe.

    On resume, already-cleaned (repo, pr_number) pairs are skipped so no
    Gemini call is wasted and progress is never lost.
    """
    # Load whatever was already cleaned in a previous run
    already_cleaned, done_set = _load_cleaned(cleaned_checkpoint)
    if already_cleaned:
        print(f"\n♻️   Resuming Phase 2 — {len(already_cleaned)} PR(s) already cleaned, skipping them.")

    # Filter to only the PRs that still need cleaning
    pending = [r for r in raw_samples if (r["repo"], r["pr_number"]) not in done_set]

    if not pending:
        print("✅  All raw PRs already cleaned — nothing to do.")
        return already_cleaned

    model = _init_gemini(gemini_api_key)

    print(f"\n🧹  Cleaning {len(pending)} prompt(s) with Gemini …  ({len(already_cleaned)} already done)")

    # Use a global counter so IDs are stable across resumed runs
    next_idx = len(already_cleaned)

    for raw in tqdm(pending, unit="PR"):
        clean = clean_prompt_with_llm(model, raw["raw_title"], raw["raw_body"])
        sample = BenchmarkSample(
            id=f"HFT-{next_idx:04d}",
            repo=raw["repo"],
            pr_number=raw["pr_number"],
            raw_title=raw["raw_title"],
            clean_prompt=clean,
            reference_code=raw["reference_code"],
            raw_diff=raw.get("raw_diff", ""),
            pr_url=raw["pr_url"],
        )
        # ── Write immediately — crash-safe ──────────────────────────────
        _append_cleaned(sample, cleaned_checkpoint)
        already_cleaned.append(sample)
        next_idx += 1
        time.sleep(0.5)   # ~30 req/min on free tier

    return already_cleaned


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
    parser.add_argument("--output", default="hft_benchmark.json", help="Final benchmark JSON path")
    parser.add_argument("--raw_output", default="raw_prs.ndjson", help="Phase 1 checkpoint — raw PRs (NDJSON)")
    parser.add_argument("--cleaned_output", default="cleaned_prs.ndjson", help="Phase 2 checkpoint — cleaned PRs (NDJSON)")
    parser.add_argument("--max_prs", type=int, default=200, help="Max PRs to collect")
    parser.add_argument(
        "--github_tokens",
        nargs="+",
        default=[t.strip() for t in os.getenv("GITHUB_TOKENS", os.getenv("GITHUB_TOKEN", "")).split(",") if t.strip()],
        help=(
            "One or more GitHub PATs. Pass as space-separated args: "
            "--github_tokens ghp_aaa ghp_bbb  "
            "Or set GITHUB_TOKENS=ghp_aaa,ghp_bbb in .env"
        ),
    )
    parser.add_argument(
        "--gemini_api_key",
        default=os.getenv("GEMINI_API_KEY"),
        help="Google Gemini API key (or set GEMINI_API_KEY env var)",
    )
    args = parser.parse_args()

    if not args.github_tokens:
        raise ValueError("At least one GitHub token is required. Set GITHUB_TOKENS or pass --github_tokens.")
    if not args.gemini_api_key:
        raise ValueError("Gemini API key is required. Set GEMINI_API_KEY or pass --gemini_api_key.")

    # Phase 1 ──────────────────────────────────
    print(f"🔑  Using {len(args.github_tokens)} GitHub token(s).")
    collect_prs(
        tokens=args.github_tokens,
        max_prs=args.max_prs,
        raw_output_path=args.raw_output,
    )

    # Load the FULL checkpoint (all runs combined), not just what was found above
    all_raw: list[dict] = []
    if os.path.exists(args.raw_output):
        with open(args.raw_output, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        all_raw.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        print(f"📂  Loaded {len(all_raw)} total raw PR(s) from {args.raw_output}")

    if not all_raw:
        print("⚠  No raw PRs found. Try expanding HFT_REPOS or HFT_KEYWORDS.")
        return

    # Phase 2 ──────────────────────────────────
    refined = refine_dataset(
        raw_samples=all_raw,
        gemini_api_key=args.gemini_api_key,
        cleaned_checkpoint=args.cleaned_output,
    )

    # Persist final JSON ────────────────────────
    save_benchmark(refined, args.output)


if __name__ == "__main__":
    main()