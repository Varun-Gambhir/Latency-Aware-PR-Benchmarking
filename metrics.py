"""
metrics.py
──────────
Phase 3: Custom evaluation metrics for the HFT C++ benchmark.

  • Pass@1  – probability that at least one of k=10 attempts is correct
  • Extended CodeBLUE – n-gram BLEU augmented with HFT-keyword multipliers
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence


# ──────────────────────────────────────────────
# HFT domain vocabulary & weights
# ──────────────────────────────────────────────

# Each entry: (regex_pattern, weight_bonus)
# Bonus is *additive* to the base [0,1] BLEU score, then the total is clipped to 1.
# Tune these weights as your evaluation progresses.
HFT_DOMAIN_FEATURES: list[tuple[str, float]] = [
    # Memory ordering / atomics
    (r"\bstd::atomic\b",                  0.08),
    (r"\bmemory_order_relaxed\b",         0.06),
    (r"\bmemory_order_acquire\b",         0.06),
    (r"\bmemory_order_release\b",         0.06),
    (r"\bmemory_order_acq_rel\b",         0.07),
    (r"\bmemory_order_seq_cst\b",         0.05),
    # Alignment & cache-line hygiene
    (r"\balignas\s*\(",                   0.07),
    (r"\b__attribute__\s*\(\s*\(aligned", 0.06),
    (r"\b_mm_prefetch\b",                 0.05),
    # Branch prediction hints
    (r"\b__builtin_expect\s*\(",          0.07),
    (r"\b\[\[likely\]\]\b",              0.05),
    (r"\b\[\[unlikely\]\]\b",            0.05),
    # Compile-time optimisation
    (r"\bconstexpr\b",                    0.04),
    (r"\binline\b",                       0.02),
    (r"\b__forceinline\b",               0.05),
    (r"\b__attribute__\s*\(\s*\(always_inline\)", 0.05),
    # SIMD intrinsics
    (r"\b_mm256_\w+\b",                   0.09),
    (r"\b_mm_\w+\b",                      0.07),
    # Lock-free patterns
    (r"\bcompare_exchange_\w+\b",         0.08),
    (r"\bfetch_add\b",                    0.06),
    (r"\bfetch_sub\b",                    0.06),
    # Avoid false positives from generic keywords
    # (No reward for std::mutex – it's a latency smell, not a bonus)
]


# ──────────────────────────────────────────────
# Internal n-gram helpers
# ──────────────────────────────────────────────

def _tokenise(code: str) -> list[str]:
    """
    Split C++ code into tokens. We split on whitespace AND punctuation
    so identifiers, operators, and brackets are separate tokens.
    """
    return re.findall(r"[A-Za-z_]\w*|[^\s\w]", code)


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _clipped_precision(hypothesis: list[str], reference: list[str], n: int) -> float:
    """
    Standard BLEU clipped precision for order n.
    Returns 0.0 when there are no hypothesis n-grams.
    """
    hyp_counts = _ngram_counts(hypothesis, n)
    ref_counts = _ngram_counts(reference, n)

    if not hyp_counts:
        return 0.0

    clipped = sum(min(cnt, ref_counts[gram]) for gram, cnt in hyp_counts.items())
    return clipped / sum(hyp_counts.values())


def _brevity_penalty(hyp_len: int, ref_len: int) -> float:
    if hyp_len >= ref_len:
        return 1.0
    if hyp_len == 0:
        return 0.0
    return math.exp(1 - ref_len / hyp_len)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def compute_bleu(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """
    Sentence-level BLEU (1-4 grams, uniform weights).
    Returns a value in [0, 1].
    """
    hyp_tokens = _tokenise(hypothesis)
    ref_tokens = _tokenise(reference)

    if not hyp_tokens or not ref_tokens:
        return 0.0

    log_sum = 0.0
    weight = 1.0 / max_n
    for n in range(1, max_n + 1):
        p_n = _clipped_precision(hyp_tokens, ref_tokens, n)
        if p_n == 0:
            return 0.0
        log_sum += weight * math.log(p_n)

    bp = _brevity_penalty(len(hyp_tokens), len(ref_tokens))
    return bp * math.exp(log_sum)


def domain_bonus(code: str) -> float:
    """
    Sum the weight bonuses for every HFT feature found in *code*.
    Caps at 0.35 so it can't overwhelm a near-zero BLEU base.
    """
    total = 0.0
    for pattern, bonus in HFT_DOMAIN_FEATURES:
        if re.search(pattern, code):
            total += bonus
    return min(total, 0.35)


def extended_codeblue(
    hypothesis: str,
    reference: str,
    domain_weight: float = 0.4,
) -> float:
    """
    Extended CodeBLUE score.

    Formula:
        score = (1 - domain_weight) * BLEU(hyp, ref)
              + domain_weight       * domain_bonus(hyp)

    The domain_weight parameter (default 0.4) controls how much
    the HFT-keyword bonus contributes relative to pure n-gram overlap.

    Returns a value in [0, 1].
    """
    bleu = compute_bleu(hypothesis, reference)
    bonus = domain_bonus(hypothesis)
    raw = (1 - domain_weight) * bleu + domain_weight * bonus
    return min(raw, 1.0)


def pass_at_k(
    generations: Sequence[str],
    reference: str,
    k: int = 1,
    threshold: float = 0.15,
) -> float:
    """
    Pass@k metric.

    A generation is considered "correct" if its extended_codeblue score
    against the reference exceeds `threshold`.

    For academic reporting, use the unbiased estimator from the Codex paper:
        Pass@k = 1 - C(n-c, k) / C(n, k)

    where:
        n = total number of generations
        c = number of correct generations
        k = how many we're sampling (default 1)

    Returns a probability in [0, 1].
    """
    n = len(generations)
    if n == 0:
        return 0.0

    c = sum(
        1 for g in generations
        if extended_codeblue(g, reference) >= threshold
    )

    if c == 0:
        return 0.0
    if c == n:
        return 1.0

    # Unbiased estimator: 1 - prod_{i=0}^{k-1} (n-c-i)/(n-i)
    numerator = math.comb(n - c, k)
    denominator = math.comb(n, k)
    if denominator == 0:
        return 0.0
    return 1.0 - numerator / denominator


def evaluate_sample(
    generations: Sequence[str],
    reference: str,
) -> dict:
    """
    Convenience wrapper: returns a dict with all metric values for one benchmark sample.
    """
    scores = [extended_codeblue(g, reference) for g in generations]
    return {
        "pass_at_1":           pass_at_k(generations, reference, k=1),
        "pass_at_5":           pass_at_k(generations, reference, k=5),
        "mean_extended_bleu":  sum(scores) / len(scores) if scores else 0.0,
        "max_extended_bleu":   max(scores, default=0.0),
        "individual_scores":   scores,
    }


# ──────────────────────────────────────────────
# Quick smoke-test (run this file directly)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    REF = """
    alignas(64) std::atomic<uint64_t> counter{0};

    inline void increment() noexcept {
        counter.fetch_add(1, std::memory_order_relaxed);
        if (__builtin_expect(counter.load() > 1000, 0)) {
            reset();
        }
    }
    """

    GOOD_GEN = """
    alignas(64) std::atomic<uint64_t> counter{0};

    __attribute__((always_inline))
    inline void increment() noexcept {
        counter.fetch_add(1, std::memory_order_relaxed);
        if (__builtin_expect(counter.load() > 1000, 0)) {
            reset();
        }
    }
    """

    BAD_GEN = """
    int counter = 0;
    std::mutex mtx;

    void increment() {
        std::lock_guard<std::mutex> lock(mtx);
        ++counter;
        if (counter > 1000) reset();
    }
    """

    print("=== Smoke Test ===")
    print(f"BLEU (good):          {compute_bleu(GOOD_GEN, REF):.4f}")
    print(f"BLEU (bad):           {compute_bleu(BAD_GEN,  REF):.4f}")
    print(f"Domain bonus (good):  {domain_bonus(GOOD_GEN):.4f}")
    print(f"Domain bonus (bad):   {domain_bonus(BAD_GEN):.4f}")
    print(f"Ext-CodeBLUE (good):  {extended_codeblue(GOOD_GEN, REF):.4f}")
    print(f"Ext-CodeBLUE (bad):   {extended_codeblue(BAD_GEN,  REF):.4f}")

    generations = [GOOD_GEN, BAD_GEN, GOOD_GEN, BAD_GEN, GOOD_GEN,
                   BAD_GEN,  GOOD_GEN, BAD_GEN, GOOD_GEN, BAD_GEN]
    result = evaluate_sample(generations, REF)
    print(f"\nPass@1:  {result['pass_at_1']:.4f}")
    print(f"Pass@5:  {result['pass_at_5']:.4f}")
    print(f"Mean Ext-BLEU: {result['mean_extended_bleu']:.4f}")
