# HFT LLM Benchmark — Domain-Specific Code Generation & Agentic Evaluation Framework

A research project and evaluation pipeline for benchmarking Large Language Models (LLMs) on **domain-specific C++ High-Frequency Trading (HFT) programming tasks**. 

This repository fulfills the requirements of developing an LLM-powered domain-specific coding benchmark and evaluating both **Zero-Shot** generation and a **Multi-Agent ReAct** workflow using Gemini and open-source models from NVIDIA NIM.

---

## 1. Domain: High-Frequency Trading (C++)
The chosen domain is High-Frequency Trading (Fintech / C++). HFT programming heavily emphasizes ultra-low latency, minimizing heap memory allocations, leveraging CPU cache lines, avoiding lock-based synchronization, and utilizing SIMD/branch-prediction hints.

We collected a set of real-world coding examples and scenarios from GitHub repositories (specifically Pull Requests addressing HFT libraries like quickfix, aeron, etc.). These pull requests represent feature requests and performance optimizations aiming to improve latency and throughput. 

---

## 2. Extended CodeBLUE Metric
To properly evaluate HFT code, standard functional correctness or n-gram overlap is insufficient because latency-critical code requires specific idioms (e.g., lock-free programming). 

We extended the traditional BLEU score to create **Extended CodeBLUE (ExtBLEU)** by assigning specific weights to domain-critical keywords. 

```
ExtCodeBLUE = (1 - λ) × BLEU(hyp, ref)  +  λ × DomainBonus(hyp)
```
- **BLEU(hyp, ref)**: standard token-level BLEU-4 between the generated and reference code.
- **DomainBonus(hyp)**: A weighted sum of HFT-specific feature keywords found in the generated codebase. Weights are assigned based on importance to latency:
  - `std::atomic` (0.08)
  - `compare_exchange...` (0.08)
  - `memory_order_release` / `acquire` / `relaxed` (0.06 - 0.07)
  - `alignas(...)` (0.07)
  - `__builtin_expect` / `[[likely]]` (0.05 - 0.07)
  - SIMD intrinsics like `_mm256_...` (0.09)

This domain-specific CodeBLUE effectively measures whether the LLM understood *how* to optimize the C++ logic, not just *what* logic to write.

---

## 3. Pass@1 and Benchmark Creation
Using the collected GitHub PR prompts, we created an evaluation benchmark.
To rigorously test the models, we calculate **Pass@k** (where k=1 or 5) by prompting the LLMs at least 10 times per task.

```
Pass@k = 1 − C(n−c, k) / C(n, k)
```
Where `n=10` attempts, and `c` is the number of attempts whose ExtCodeBLUE meets or exceeds the success threshold compared to the reference code.

Our complete benchmark includes:
- The curated HFT prompt samples.
- The Domain-specific Extended CodeBLUE scores.
- The computed Pass@1 and Pass@5 scores.

---

## 4. Zero-Shot Evaluation (Gemini & NIM)
We wrote standard system prompts and evaluated code generation in a zero-shot manner utilizing:
1. **Google Gemini** (`gemini-3-flash-preview` / `gemini-1.5-pro` via Google AI Studio).
2. **Llama-3-70B-Instruct** (`meta/llama3-70b-instruct` via NVIDIA NIM inference service).
3. **Mixtral-8x7B-Instruct** (`mistralai/mixtral-8x7b-instruct-v0.1` via NVIDIA NIM inference service).

These models undergo the same 10-attempt zero-shot generation process to evaluate their baseline Pass@1 and ExtBLEU scores on our benchmark.

---

## 5. Multi-Agent ReAct Framework
To improve upon zero-shot baseline performance, we designed a multi-agent system operating in a ReAct (Reasoning + Acting) loop, inspired by [arXiv:2408.08927](https://arxiv.org/abs/2408.08927).

**Architecture of the Agentic Workflow:**
1. **Programmer Agent**: Takes the prompt/feedback and writes the C++ HFT code.
2. **Executor Agent**: Validates syntax correctness (acts as a mock compiler tool).
3. **Critic Agent**: Acts as an expert HFT reviewer, auditing the code for latency smells (heap allocations, mutex locks, virtual dispatch, missing branch attributes).

The Programmer receives actionable feedback from the Critic and Executor and iterates (up to 3 max iterations).

```
[Programmer] ──writes──► [Executor validates syntax]
      ▲                          │ FAIL
      │      feedback            ▼
      └──────────────── [Critic reviews latency smells]
                                 │ NO_ISSUES + PASS → ✓ done
```
By utilizing the *exact same LLMs* from the zero-shot step in this agentic loop, we successfully demonstrated that a well-constructed agentic framework improves benchmark scores compared to zero-shot calls.

---

## Project Structure & Artifacts

| Directory/File | Description |
|------|-------------|
| `data_pipeline.py` | Mines GitHub PRs, filters constraints, and cleans prompts using Gemini. |
| `metrics.py` | Contains the implementation for Extended CodeBLUE and Pass@k calculations. |
| `agents.py` | The Multi-Agent ReAct architecture implementations. |
| `main.py` | Main orchestrator to run Zero-Shot and Agentic evaluations. |
| `checkpoints/` | Contains the mined `raw_prs.ndjson` and cleaned `cleaned_prs.ndjson` from our data pipeline. |
| `hft_benchmark.json` | The final structured benchmark dataset. |
| `results/` | Evaluation outputs, including `full_results.csv` and `comparison.csv` detailing the improved agentic metrics. |

---

## Setup & Usage

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
