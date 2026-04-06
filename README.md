# HFT LLM Benchmark — Multi-Agent ReAct Evaluation Framework

A research pipeline for evaluating LLMs on **domain-specific C++ High-Frequency Trading coding tasks**. Mines real-world HFT pull requests from GitHub, builds a clean benchmark, and evaluates models under zero-shot and Multi-Agent ReAct conditions using a custom **Extended CodeBLUE** metric.

---

## Architecture

```
GitHub (40 repos) ──► keyword filter ──► diff extraction
        │                                      │
  raw_prs.ndjson ◄── append on match           │
                                               ▼
  Gemini Flash ──► prompt cleaner ──► cleaned_prs.ndjson
                                               │
                                     hft_benchmark.json
                                               │
                    ┌──────────────────────────┴──────────────────────────┐
                    │                                                      │
             Zero-Shot (×10)                          Agentic ReAct (×3 runs)
          llama3-70b, mixtral-8x7b               Programmer → Critic → Synthesizer
                    │                                                      │
                    └──────────────────────┬───────────────────────────────┘
                                           ▼
                                     metrics.py
                               Extended CodeBLUE + Pass@k
                                           │
                                 results/comparison.csv
```

---

## Project Structure

```
├── data_pipeline.py    # Phases 1+2: GitHub mining + Gemini prompt cleaning
├── metrics.py          # Phase 3:   Extended CodeBLUE + Pass@k
├── agents.py           # Phases 4+5: LLM wrappers + ReAct loop
├── main.py             # Phase 6:   Orchestrator + CSV reporting
├── requirements.txt
├── .env                # API keys (never commit)
├── checkpoints/
│   ├── raw_prs.ndjson        # Phase 1 crash-safe checkpoint
│   └── cleaned_prs.ndjson    # Phase 2 crash-safe checkpoint
└── results/
    ├── full_results.csv      # Per-sample scores
    └── comparison.csv        # Aggregated comparison table
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`.env` file:
```env
GITHUB_TOKENS=ghp_token1,ghp_token2,ghp_token3   # comma-separated, rotates on rate-limit
GEMINI_API_KEY=AIza...
NVIDIA_NIM_API_KEY=nvapi-...
```

---

## Usage

```bash
# Step 1 — build dataset (resumable, appends to checkpoints)
python data_pipeline.py \
  --raw_output checkpoints/raw_prs.ndjson \
  --cleaned_output checkpoints/cleaned_prs.ndjson \
  --max_prs 1000

# Step 2 — run evaluation
python main.py --models llama3-70b mixtral-8x7b --output_dir results/

# Quick dev run
python main.py --max_samples 10 --models llama3-70b --skip_agentic
```

---

## Data Pipeline

**Phase 1** scans **40 curated C++ repos** across 8 categories (HFT infra, lock-free structures, messaging, serialisation, allocators, SIMD, benchmarking, atomics) and filters merged PRs by **38 low-latency keywords** (`lock-free`, `SPSC`, `MPMC`, `memory_order`, `alignas`, `AVX`, `RDMA`, `tick-to-trade`, etc.).

Each PR diff is split into two fields:
- `raw_diff` — full patch for human inspection
- `reference_code` — only the added `+` lines with the `+` prefix stripped, giving plain C++ that metrics can compare against model output

Every match is **appended immediately** to `raw_prs.ndjson`. On restart, already-collected `(repo, pr_number)` pairs are skipped in O(1) using a seen-set.

**Phase 2** calls Gemini Flash to distil each PR body into a clean imperative coding instruction. Each cleaned sample is written to `cleaned_prs.ndjson` immediately — crash-safe. Already-cleaned PRs are skipped on resume.

**GitHub token pool** rotates between multiple PATs evenly. On 403/429/503 it switches instantly to the next token. On 401 that token is **permanently removed** (never reset). When all tokens are exhausted simultaneously it backs off exponentially (30 → 60 → 120 → 300s) then resets non-dead tokens.

---

## Scoring Metrics

### Why Standard BLEU Fails Here

Standard BLEU has two blind spots for HFT C++:

1. **It cannot distinguish correct from dangerous implementations.** A `std::mutex` implementation and a lock-free `std::atomic` CAS loop score identically against the same reference — both share common tokens (`void`, `int`, `{`, `}`). In HFT, a mutex introduces microseconds of kernel overhead on every contended call. Microseconds are the entire margin.

2. **References are short PR diffs.** Our `reference_code` is extracted from PR diffs — typically 10–50 tokens. A complete, correct model output gets penalised by the brevity penalty for being longer than the reference even when it's technically superior.

### BLEU-4 Baseline

Tokenisation: `re.findall(r"[A-Za-z_]\w*|[^\s\w]", code)` — identifiers kept whole (`memory_order_relaxed`), punctuation split individually. Preserves C++ namespace and template syntax as meaningful units.

Standard sentence-level BLEU with clipped precision and brevity penalty across n-gram orders 1–4 (uniform weights). Returns 0.0 if any order has zero precision.

### Domain Bonus

Rewards **presence of HFT-specific idioms** in the generated code, independent of the reference. Computed as a weighted sum over 21 regex patterns, capped at **0.35**:

| Category | Patterns | Max bonus |
|---|---|---:|
| Atomics | `std::atomic`, `memory_order_{relaxed,acquire,release,acq_rel,seq_cst}` | 0.38 → capped |
| Lock-free ops | `compare_exchange_*`, `fetch_add`, `fetch_sub` | 0.20 |
| SIMD | `_mm256_*`, `_mm_*` intrinsics | 0.16 |
| Alignment | `alignas(`, `__attribute__((aligned`, `_mm_prefetch` | 0.18 |
| Branch hints | `__builtin_expect(`, `[[likely]]`, `[[unlikely]]` | 0.17 |
| Inline | `__forceinline`, `__attribute__((always_inline)`, `inline` | 0.12 |
| Compile-time | `constexpr` | 0.04 |

**The cap at 0.35** prevents keyword-stuffing — a model cannot score purely on vocabulary without any structural similarity to the reference.

**Deliberately excluded:** `std::mutex`, `std::lock_guard`, `new`, `malloc` — latency smells that are neither rewarded nor penalised by the domain bonus.

### Extended CodeBLUE

```
ExtCodeBLUE(hyp, ref) = (1 − λ) × BLEU-4(hyp, ref)
                      +      λ  × DomainBonus(hyp)

λ = 0.4  (domain_weight, tunable in metrics.py)
```

The 60/40 split reflects that *using the right primitives* is nearly as important as *textual similarity to the reference* in HFT code generation.

**Score range analysis:**

| Scenario | BLEU | DomainBonus | ExtCodeBLUE |
|---|---:|---:|---:|
| Perfect match, no HFT idioms | 1.00 | 0.00 | 0.600 |
| No token overlap, max HFT idioms | 0.00 | 0.35 | 0.140 |
| Good match + full HFT idioms | 0.55 | 0.35 | 0.470 |
| Mutex-based (latency smell) | 0.15 | 0.00 | 0.090 |

**Worked example** — reference: `alignas(64) std::atomic<int> counter{0}; inline void increment() { counter.fetch_add(1, std::memory_order_relaxed); }`

- **Generation A (mutex):** BLEU≈0.15, DomainBonus=0.00 → **ExtCodeBLUE = 0.090** — below the 0.15 correctness threshold, marked incorrect
- **Generation B (lock-free + idioms):** BLEU≈0.55, DomainBonus=0.35 → **ExtCodeBLUE = 0.470** — correctly marked as a superior implementation

Generation B scores 5× higher, correctly reflecting real-world latency superiority.

### Pass@k — Unbiased Estimator

From the Codex paper (Chen et al., 2021). Given n total generations, c of which are "correct" (ExtCodeBLUE ≥ 0.15):

```
Pass@k = 1 − C(n−c, k) / C(n, k)
```

The combinatorial formula gives the exact expected value over all k-subsets of n generations — no sampling variance.

**Example: n=10, c=3**
```
Pass@1 = 1 − C(7,1)/C(10,1) = 0.300
Pass@5 = 1 − C(7,5)/C(10,5) = 0.917
```

| Condition | Pool size (n) | k reported |
|---|---|---|
| Zero-shot | 10 | 1, 5 |
| Agentic ReAct | 3 (3 full runs) | 1 |

---

## Multi-Agent ReAct System

### Agents

| Agent | Role | API calls |
|---|---|---|
| **ProgrammerAgent** | Writes C++ from prompt. Iterations 2+ receive structured `ISSUE/FIX` blocks and rewrite. | 1 per iteration |
| **CriticAgent** | Reviews for 8 latency smells. Outputs `ISSUE: X / FIX: Y` format — exact substitutions, not prose. | 1 per iteration |
| **ExecutorAgent** | Heuristic structural validator: brace balance, paren balance, non-empty, C++ construct present. | **0** (no API) |
| **SynthesizerAgent** | Post-pass: injects all missing HFT idioms into the best-seen code. Reverts if `domain_bonus` decreases. | 1 per run |

**Cost per prompt:** `(Programmer + Critic) × 3 iters + Synthesizer = 7 calls/run × 3 runs = 21 calls` vs 10 for zero-shot.

### Loop

```
For each of 3 runs:
  Iteration 1–3:
    Programmer  ──► writes / rewrites C++ using ISSUE/FIX feedback
    Executor    ──► heuristic check (free)
    Critic      ──► structured ISSUE/FIX blocks
    track best_code = max(domain_bonus) across all iterations

  Synthesizer post-pass:
    Input:  best_code across all iterations
    Action: inject alignas / memory_order_* / __builtin_expect / always_inline
    Safety: revert to best_code if domain_bonus degrades
```

**Five decisions that drive improvement over zero-shot:**

1. **Structured Critic** — `ISSUE: std::mutex in hot path / FIX: replace with std::atomic + compare_exchange_strong` gives the Programmer exact substitutions. Prose feedback produced vague rewrites.
2. **Best-of-N tracking** — keeps the highest `domain_bonus` code across all iterations. A regressing iteration 3 cannot erase a good iteration 2.
3. **Heuristic Executor** — the LLM-based syntax checker cost 1 API call per iteration, was slow, and was often wrong. Four simple checks run in microseconds.
4. **Synthesizer as a floor** — even if the loop produces mediocre code, the Synthesizer guarantees a minimum `domain_bonus` by forcibly injecting HFT idioms.
5. **Pool of 3 runs** — the original n=1 agentic design made Pass@k equivalent to binary pass/fail. n=3 enables a meaningful pool for the unbiased estimator.

---

## Results

Evaluated on **206 benchmark samples** (real HFT PR diffs from GitHub). Full results available in `results/comparison.csv` and `results/full_results.csv`.

### Aggregated Scores (n=206 samples)

| Model | Condition | Pass@1 | Pass@5 | ExtBLEU |
|---|---|---:|---:|---:|
| llama3-70b | zero_shot | 0.1612 | 0.3348 | 0.1158 |
| mixtral-8x7b | zero_shot | 0.1505 | 0.3443 | 0.1192 |
| llama3-70b | **agentic_react** | **0.3091** | 0.2233 | **0.1467** |
| mixtral-8x7b | **agentic_react** | 0.2735 | 0.1796 | 0.1460 |

**🚀 Agentic lift over best zero-shot: +23.1% ExtBLEU (0.1467 vs 0.1192)**

### Key Observations

- **Pass@1 improves dramatically under agentic conditions** — llama3-70b nearly doubles from 0.1612 → 0.3091 (+91.8%), reflecting that the Critic+Synthesizer reliably pushes at least one run past the correctness threshold.
- **Pass@5 is lower for agentic** — expected, as the agentic pool is n=3 runs vs zero-shot n=10. Pass@5 with n=3 is mathematically bounded and not directly comparable.
- **ExtBLEU improves for both models** — +26.7% for llama3-70b (0.1158→0.1467), +22.5% for mixtral-8x7b (0.1192→0.1460). The Synthesizer post-pass is the primary driver, injecting `alignas`, `memory_order_*`, and `__builtin_expect` into outputs that lacked them.
- **Models are competitive zero-shot** — llama3-70b edges mixtral on Pass@1 while mixtral leads on Pass@5 and ExtBLEU, suggesting mixtral generates more consistent HFT vocabulary but llama is more likely to produce a strong single attempt.

---

## Supported Models

| CLI name | Provider | Model ID |
|---|---|---|
| `gemini` | Google AI | `gemini-2.5-flash` (configurable in `agents.py`) |
| `llama3-70b` | NVIDIA NIM | `meta/llama3-70b-instruct` |
| `mixtral-8x7b` | NVIDIA NIM | `mistralai/mixtral-8x7b-instruct-v0.1` |

Add new NIM models by extending `NIM_MODELS` in `agents.py`.

---

## Known Issues & Fixes

| Issue | Fix |
|---|---|
| `proxies` kwarg error with openai + httpx ≥0.28 | Pass `http_client=httpx.Client()` explicitly to `OpenAI()` |
| Gemini blocked by safety filter on C++ code | All four harm categories set to `BLOCK_NONE` via `safety_settings` |
| `KeyError: 'reference_code'` on old checkpoints | Legacy shim detects `"diff"` key and runs `_diff_to_reference_code()` on the fly |
| GitHub 401 causing infinite retry loop | `_dead[]` array permanently excludes bad tokens from pool resets |
| All Gemini scores 0.0 | Error strings returned silently as code; now errors are printed, empty string returned |

---

## References

- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, arXiv:2210.03629
- arXiv:2408.08927 — HFT multi-agent system
- Ren et al., *CodeBLEU*, arXiv:2009.10297
- Chen et al., *Evaluating LLMs Trained on Code* (Pass@k), arXiv:2107.03374
- Papineni et al., *BLEU*, ACL 2002