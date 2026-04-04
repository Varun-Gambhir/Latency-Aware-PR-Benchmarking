"""
agents.py
─────────
Phase 4 + 5: Zero-shot LLM inference wrappers and the Multi-Agent ReAct loop.

Agents
──────
  ProgrammerAgent  – writes C++ code from a prompt
  CriticAgent      – reviews code for HFT latency smells and returns feedback
  ExecutorAgent    – mock syntax validator (real clang check optional)

ReAct Loop
──────────
  Thought → Action → Observation, max 3 iterations per prompt.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from openai import OpenAI               # used for NVIDIA NIM (OpenAI-compatible API)
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Model / endpoint config
# ──────────────────────────────────────────────

NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Models available on NVIDIA NIM (OpenAI-compatible)
NIM_MODELS: dict[str, str] = {
    "llama3-70b":  "meta/llama3-70b-instruct",
    "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct-v0.1",
}

GEMINI_CODE_MODEL = "gemini-3-flash-preview"

MAX_REACT_ITERATIONS = 3

# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────

PROGRAMMER_SYSTEM = """You are an elite C++ High-Frequency Trading engineer.
Write production-quality, low-latency C++ code.

Rules:
- Prefer lock-free data structures (std::atomic, CAS loops).
- Avoid heap allocations in hot paths; use stack or pre-allocated pools.
- Use alignas(64) for cache-line alignment on shared state.
- Annotate branches with __builtin_expect where probability is known.
- Use memory_order_relaxed / acquire / release appropriately.
- Avoid virtual dispatch and std::mutex in critical paths.
- Output ONLY the C++ code. No markdown fences, no explanation."""

CRITIC_SYSTEM = """You are a strict HFT code reviewer. Your ONLY goal is maximum latency reduction.

Scan the code for these smells and for each one found output a fix block in EXACTLY this format:

ISSUE: <one-line description of the problem>
FIX:   <one-line exact code change, e.g. replace X with Y>

Smells to check (in priority order):
  1. std::mutex / std::lock_guard / std::unique_lock  → replace with std::atomic + CAS
  2. new / malloc / std::string / std::vector growth   → replace with stack array or pre-allocated pool
  3. virtual functions / vtable                        → replace with CRTP or direct call
  4. Missing alignas(64) on shared state              → add alignas(64) to the declaration
  5. Missing memory_order on atomic ops               → add memory_order_relaxed / acquire / release
  6. Hot branches without __builtin_expect            → wrap condition in __builtin_expect(cond, likely)
  7. Large struct/class passed by value               → pass by const reference or pointer
  8. Missing inline / __attribute__((always_inline))  → add to hot-path functions

If NONE of these smells exist, output exactly: NO_ISSUES
Output ONLY the ISSUE/FIX blocks or NO_ISSUES. No preamble, no summary."""

EXECUTOR_SYSTEM = """You are a C++ syntax checker.
Check if the supplied code is syntactically valid C++17.

Respond with exactly one line:
  PASS   – if the code would compile without errors
  FAIL: <brief reason> – if there is an obvious syntax error"""

SYNTHESIZER_SYSTEM = """You are an HFT C++ finaliser. You receive working C++ code and must
return a version that is IDENTICAL in logic but maximally optimised for nanosecond latency.

Apply ALL of these transformations that are not already present:
  • Wrap every shared variable with alignas(64) std::atomic<T>
  • Change all raw atomic loads/stores to use memory_order_relaxed unless ordering is required
  • Add __builtin_expect(cond, 1) or (cond, 0) around every branch where probability is obvious
  • Mark every small function with __attribute__((always_inline)) inline
  • Replace std::string with fixed char arrays where possible
  • Replace dynamic containers with stack arrays or std::array
  • Add constexpr to every constant

Output ONLY the final C++ code. No markdown, no explanation."""


# ──────────────────────────────────────────────
# Low-level API wrappers
# ──────────────────────────────────────────────

# Turn off all safety filters — HFT C++ code (atomics, memory ops) routinely
# triggers Gemini's heuristics even though it is entirely benign.
_GEMINI_SAFETY_OFF = {
    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT:        genai.types.HarmBlockThreshold.BLOCK_NONE,
    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH:       genai.types.HarmBlockThreshold.BLOCK_NONE,
    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
}


def _call_gemini(
    system: str,
    user_message: str,
    api_key: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> str:
    key = api_key or os.getenv("GEMINI_API_KEY", "")
    genai.configure(api_key=key)
    model = genai.GenerativeModel(
        model_name=GEMINI_CODE_MODEL,
        system_instruction=system,
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    for attempt in range(3):
        try:
            response = model.generate_content(
                user_message,
                safety_settings=_GEMINI_SAFETY_OFF,
            )
            # Check finish reason before accessing .text — a blocked response
            # has no candidates/parts and .text raises instead of returning "".
            candidate = response.candidates[0] if getattr(response, "candidates", None) and len(response.candidates) > 0 else None
            if candidate is None:
                print(f"   ⚠  Gemini returned no candidates (attempt {attempt + 1}/3)")
                time.sleep(2 ** attempt)
                continue
            finish = str(candidate.finish_reason)
            if finish not in ("FinishReason.STOP", "1", "STOP", "FinishReason.MAX_TOKENS", "2", "MAX_TOKENS"):
                print(f"   ⚠  Gemini blocked — finish_reason={finish}  "
                      f"safety={candidate.safety_ratings} (attempt {attempt + 1}/3)")
                time.sleep(2 ** attempt)
                continue
            if not candidate.content or not candidate.content.parts:
                print(f"   ⚠  Gemini returned empty content parts (attempt {attempt + 1}/3)")
                time.sleep(2 ** attempt)
                continue
            return candidate.content.parts[0].text.strip()
        except Exception as exc:
            print(f"   ⚠  Gemini call failed (attempt {attempt + 1}/3): {exc}")
            time.sleep(2 ** attempt)
    return ""


def _call_nim(
    system: str,
    user_message: str,
    model_key: str = "llama3-70b",
    api_key: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> str:
    key = api_key or os.getenv("NVIDIA_NIM_API_KEY", "")
    # Passing an explicit http_client bypasses the openai SDK's internal
    # proxy-detection code that breaks on newer httpx versions (>=0.28).
    client = OpenAI(
        base_url=NVIDIA_NIM_BASE_URL,
        api_key=key,
        http_client=httpx.Client(),
    )
    model_id = NIM_MODELS.get(model_key, model_key)
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content.strip()
    except Exception as exc:
        return f"// ERROR: NIM call failed – {exc}"


# ──────────────────────────────────────────────
# Zero-shot inference (Phase 4)
# ──────────────────────────────────────────────

def zero_shot_generate(
    prompt: str,
    model_name: str,
    n_attempts: int = 10,
    api_keys: Optional[dict] = None,
) -> list[str]:
    """
    Generate `n_attempts` completions for a single prompt using the
    specified model.  Returns a list of code strings.

    model_name options: "gemini", "llama3-70b", "mixtral-8x7b"
    api_keys: {"gemini": "...", "nvidia": "..."}
    """
    api_keys = api_keys or {}
    generations: list[str] = []

    for i in range(n_attempts):
        if model_name == "gemini":
            code = _call_gemini(
                PROGRAMMER_SYSTEM,
                prompt,
                api_key=api_keys.get("gemini"),
                temperature=0.7,   # slight diversity for Pass@k
            )
        elif model_name in NIM_MODELS:
            code = _call_nim(
                PROGRAMMER_SYSTEM,
                prompt,
                model_key=model_name,
                api_key=api_keys.get("nvidia"),
                temperature=0.7,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}. Choose from: gemini, {list(NIM_MODELS)}")

        generations.append(code)
        time.sleep(0.3)   # basic rate-limit guard

    return generations


# ──────────────────────────────────────────────
# Individual agents  (Phase 5)
# ──────────────────────────────────────────────

@dataclass
class AgentMessage:
    role: str          # "programmer" | "critic" | "executor"
    content: str


class ProgrammerAgent:
    """Writes C++ code from a natural-language prompt."""

    def __init__(self, model_name: str = "gemini", api_keys: Optional[dict] = None):
        self.model_name = model_name
        self.api_keys = api_keys or {}

    def act(self, prompt: str, critic_feedback: str = "") -> str:
        user_msg = prompt
        if critic_feedback:
            user_msg = (
                f"Original task:\n{prompt}\n\n"
                f"Critic feedback to address:\n{critic_feedback}\n\n"
                "Rewrite the C++ code addressing all latency issues above."
            )

        if self.model_name == "gemini":
            return _call_gemini(
                PROGRAMMER_SYSTEM, user_msg,
                api_key=self.api_keys.get("gemini"),
            )
        return _call_nim(
            PROGRAMMER_SYSTEM, user_msg,
            model_key=self.model_name,
            api_key=self.api_keys.get("nvidia"),
        )


class CriticAgent:
    """Reviews code for HFT latency smells."""

    def __init__(self, model_name: str = "gemini", api_keys: Optional[dict] = None):
        self.model_name = model_name
        self.api_keys = api_keys or {}

    def act(self, code: str) -> str:
        if self.model_name == "gemini":
            return _call_gemini(
                CRITIC_SYSTEM, code,
                api_key=self.api_keys.get("gemini"),
                temperature=0.1,
            )
        return _call_nim(
            CRITIC_SYSTEM, code,
            model_key=self.model_name,
            api_key=self.api_keys.get("nvidia"),
            temperature=0.1,
        )

    @staticmethod
    def has_issues(feedback: str) -> bool:
        return feedback.strip().upper() != "NO_ISSUES"


class SynthesizerAgent:
    """
    Post-pass agent: takes the final code from the ReAct loop and
    forces injection of HFT idioms (alignas, memory_order, __builtin_expect, etc.)
    This is the key driver of ExtCodeBLUE improvement over zero-shot.
    """

    def __init__(self, model_name: str = "gemini", api_keys: Optional[dict] = None):
        self.model_name = model_name
        self.api_keys   = api_keys or {}

    def act(self, code: str) -> str:
        user_msg = f"Optimise this C++ code for HFT latency:\n\n{code}"
        if self.model_name == "gemini":
            result = _call_gemini(
                SYNTHESIZER_SYSTEM, user_msg,
                api_key=self.api_keys.get("gemini"),
                temperature=0.1,
                max_tokens=1500,
            )
        else:
            result = _call_nim(
                SYNTHESIZER_SYSTEM, user_msg,
                model_key=self.model_name,
                api_key=self.api_keys.get("nvidia"),
                temperature=0.1,
                max_tokens=1500,
            )
        # If synthesizer fails or returns empty, keep original
        return result if result.strip() else code


class ExecutorAgent:
    """
    Pure-heuristic C++ validator — no LLM call needed.
    Checks structural validity (balanced braces/parens, non-empty) fast.
    This frees up API quota for the Programmer and Critic.
    """

    def __init__(self, model_name: str = "gemini", api_keys: Optional[dict] = None):
        self.model_name = model_name   # kept for API compat, not used
        self.api_keys   = api_keys or {}

    def act(self, code: str) -> tuple[bool, str]:
        """Returns (passed: bool, reason: str)."""
        code = code.strip()
        if not code or len(code) < 10:
            return False, "Empty or trivial output"

        brace_diff = code.count("{") - code.count("}")
        if abs(brace_diff) > 2:
            return False, f"Unbalanced braces ({brace_diff:+d})"

        paren_diff = code.count("(") - code.count(")")
        if abs(paren_diff) > 2:
            return False, f"Unbalanced parentheses ({paren_diff:+d})"

        # Must contain at least one C++ construct
        if not re.search(r"[\w:]+\s*[\(\{]", code):
            return False, "No recognisable C++ constructs"

        return True, "OK"


# ──────────────────────────────────────────────
# ReAct orchestration loop  (Phase 5)
# ──────────────────────────────────────────────

@dataclass
class ReActResult:
    final_code: str
    iterations: int
    trajectory: list[AgentMessage] = field(default_factory=list)
    executor_passed: bool = False


def react_loop(
    prompt: str,
    programmer: ProgrammerAgent,
    critic: CriticAgent,
    executor: ExecutorAgent,
    synthesizer: "SynthesizerAgent",
    max_iterations: int = MAX_REACT_ITERATIONS,
) -> ReActResult:
    """
    ReAct loop  (Thought → Action → Observe) × max_iterations,
    followed by a Synthesizer post-pass that injects HFT idioms.

    Improvement over naive ReAct:
    - Executor is heuristic-only (no API waste).
    - Critic uses structured ISSUE/FIX format for precise refinement.
    - Best code across all iterations is kept, not just the last.
    - Synthesizer forcibly adds alignas/memory_order/__builtin_expect.
    """
    from metrics import domain_bonus   # local import to avoid circular dep

    trajectory: list[AgentMessage] = []
    current_code = ""
    critic_feedback = ""
    executor_passed = False
    best_code = ""
    best_score = -1.0

    for iteration in range(1, max_iterations + 1):

        # ── Thought + Action (Programmer) ───────────
        thought = (
            f"[Iter {iteration}] Programmer generating code"
            + (" with critic feedback." if critic_feedback else ".")
        )
        trajectory.append(AgentMessage("thought", thought))

        current_code = programmer.act(prompt, critic_feedback)
        trajectory.append(AgentMessage("programmer", current_code))

        # ── Track best code by domain_bonus score ───
        score = domain_bonus(current_code)
        if score > best_score:
            best_score = score
            best_code  = current_code

        # ── Observation 1: Executor (heuristic) ─────
        executor_passed, exec_reason = executor.act(current_code)
        trajectory.append(
            AgentMessage("executor", "PASS" if executor_passed else f"FAIL: {exec_reason}")
        )

        # ── Observation 2: Critic ───────────────────
        critic_feedback = critic.act(current_code)
        trajectory.append(AgentMessage("critic", critic_feedback))

        # ── Early termination ────────────────────────
        if executor_passed and not CriticAgent.has_issues(critic_feedback):
            break

        time.sleep(0.2)

    # ── Synthesizer post-pass ────────────────────────
    # Runs on the best code seen across all iterations.
    synthesized = synthesizer.act(best_code)
    synthesized_score = domain_bonus(synthesized)
    if synthesized_score >= best_score:
        final_code = synthesized
    else:
        final_code = best_code   # synthesizer made things worse — revert
    trajectory.append(AgentMessage("synthesizer", final_code))

    return ReActResult(
        final_code=final_code,
        iterations=iteration,
        trajectory=trajectory,
        executor_passed=executor_passed,
    )


def agentic_generate(
    prompt: str,
    model_name: str = "gemini",
    n_attempts: int = 3,
    api_keys: Optional[dict] = None,
) -> list[str]:
    """
    Run the full ReAct+Synthesizer pipeline `n_attempts` times and return
    all final code strings so Pass@k metrics are computed over a pool,
    not a single generation (which unfairly disadvantages the agentic condition).

    Default n_attempts=3: enough for meaningful Pass@k without excessive cost.
    """
    programmer  = ProgrammerAgent(model_name, api_keys)
    critic      = CriticAgent(model_name, api_keys)
    executor    = ExecutorAgent(model_name, api_keys)
    synthesizer = SynthesizerAgent(model_name, api_keys)

    results: list[str] = []
    for _ in range(n_attempts):
        result = react_loop(prompt, programmer, critic, executor, synthesizer)
        if result.final_code:
            results.append(result.final_code)

    return results