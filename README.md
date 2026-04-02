# HFT LLM Benchmark — Multi-Agent ReAct Evaluation Framework

A research pipeline for evaluating Large Language Models on **domain-specific C++ High-Frequency Trading coding tasks**, using a custom *Extended CodeBLUE* metric and a **Multi-Agent ReAct loop** for iterative code refinement.

---

## Architecture

```
GitHub PRs  ──►  data_pipeline.py  ──►  hft_benchmark.json
                                                │
                          ┌─────────────────────┤
                          │                     │
                    Zero-Shot                Agentic ReAct
               (Gemini / NIM x10)       (Programmer → Executor
                          │               → Critic loop x3)
                          │                     │
                          └────────┬────────────┘
                                   ▼
                            metrics.py
                      (Extended CodeBLUE + Pass@1)
                                   ▼
                         results/comparison.csv
```

---

## Project Structure

| File | Phase | Description |
|------|-------|-------------|
| `data_pipeline.py` | 1 + 2 | Mine GitHub PRs, clean prompts via Gemini |
| `metrics.py` | 3 | Extended CodeBLUE, Pass@k implementation |
| `agents.py` | 4 + 5 | LLM wrappers + ReAct multi-agent loop |
| `main.py` | 6 | Orchestrator, runs benchmark end-to-end |
| `requirements.txt` | — | Python dependencies |

---

## Setup

### 1. Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt  # only needed if you extend metrics with NLTK
```

### 2. Set API keys
Create a `.env` file (never commit this):
```
GITHUB_TOKEN=ghp_...
GEMINI_API_KEY=AIza...
NVIDIA_NIM_API_KEY=nvapi-...
```

---

## Usage

### Step 1 — Build the benchmark dataset
```bash
python data_pipeline.py --output hft_benchmark.json --max_prs 200
```
Output: `hft_benchmark.json` with fields `[id, repo, pr_number, clean_prompt, reference_code, pr_url]`.

---

### Step 2 — Run the full evaluation
```bash
python main.py \
  --benchmark hft_benchmark.json \
  --output_dir results/ \
  --max_samples 50          # remove for full run
```

#### Flags
| Flag | Default | Description |
|------|---------|-------------|
| `--max_samples` | all | Limit samples for quick dev runs |
| `--skip_zero_shot` | false | Skip Phase 4 |
| `--skip_agentic` | false | Skip Phase 5 |

---

### Step 3 — Inspect results
```
results/
├── full_results.csv      # per-sample scores for every model × condition
└── comparison.csv        # aggregated mean metrics table
```

---

## Metrics

### Extended CodeBLUE

```
ExtCodeBLUE = (1 - λ) × BLEU(hyp, ref)  +  λ × DomainBonus(hyp)
```

- **BLEU(hyp, ref)** — standard sentence-level BLEU-4 on tokenised C++ code.
- **DomainBonus(hyp)** — weighted sum of HFT-specific C++ features present in the hypothesis (e.g. `std::atomic`, `alignas`, `memory_order_release`, `__builtin_expect`, SIMD intrinsics). Capped at 0.35.
- **λ = 0.4** by default (tunable in `metrics.py`).

### Pass@k (unbiased estimator)

```
Pass@k = 1 − C(n−c, k) / C(n, k)
```
where `n=10` attempts, `c` = number of attempts whose ExtCodeBLUE ≥ threshold.

---

## Multi-Agent ReAct Loop

```
[Programmer] ──writes──► [Executor validates syntax]
      ▲                          │ FAIL
      │      feedback            ▼
      └──────────────── [Critic reviews latency smells]
                                 │ NO_ISSUES + PASS → ✓ done
```

- **Max iterations:** 3
- **Termination:** Executor passes AND Critic returns `NO_ISSUES`.
- **Critic latency smells checked:** heap allocations, `std::mutex`, virtual dispatch, false sharing, missing `__builtin_expect`.

---

## Supported Models

| Name (--model) | Provider | Notes |
|----------------|----------|-------|
| `gemini` | Google AI | `gemini-1.5-pro` |
| `llama3-70b` | NVIDIA NIM | `meta/llama3-70b-instruct` |
| `mixtral-8x7b` | NVIDIA NIM | `mistralai/mixtral-8x7b-instruct-v0.1` |

---

## References

- ReAct framework: Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, arXiv:2210.03629
- HFT agent system: arXiv:2408.08927
- CodeBLEU: Ren et al., arXiv:2009.10297
- Pass@k estimator: Chen et al. (Codex), arXiv:2107.03374
