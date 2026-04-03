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

GEMINI_CODE_MODEL = "gemini-3.1-flash-preview-04-17"

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

CRITIC_SYSTEM = """You are a strict HFT code reviewer focused solely on latency.

Analyse the supplied C++ code for these latency smells:
  1. Heap allocations (new, malloc, std::vector without reserve, std::string)
  2. Locks (std::mutex, std::lock_guard, std::unique_lock)
  3. Virtual dispatch (virtual functions, vtable calls)
  4. Unnecessary copies of large objects
  5. Cache-unfriendly layouts (missing alignas, false sharing)
  6. Unpredicted branches (missing __builtin_expect)

For each smell found, give a one-line actionable fix.
If the code is already optimal, respond with exactly: "NO_ISSUES"
Output ONLY your numbered findings or "NO_ISSUES"."""

EXECUTOR_SYSTEM = """You are a C++ syntax checker.
Check if the supplied code is syntactically valid C++17.

Respond with exactly one line:
  PASS   – if the code would compile without errors
  FAIL: <brief reason> – if there is an obvious syntax error"""


# ──────────────────────────────────────────────
# Low-level API wrappers
# ──────────────────────────────────────────────

# Turn off all safety filters — HFT C++ code (atomics, memory ops) routinely
# triggers Gemini's heuristics even though it is entirely benign.
_GEMINI_SAFETY_OFF = {
    HarmCategory.HARM_CATEGORY_HARASSMENT:        HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH:       HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
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
            candidate = response.candidates[0] if response.candidates else None
            if candidate is None:
                print(f"   ⚠  Gemini returned no candidates (attempt {attempt + 1}/3)")
                time.sleep(2 ** attempt)
                continue
            finish = str(candidate.finish_reason)
            if finish not in ("FinishReason.STOP", "1", "STOP"):
                print(f"   ⚠  Gemini blocked — finish_reason={finish}  "
                      f"safety={candidate.safety_ratings} (attempt {attempt + 1}/3)")
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


class ExecutorAgent:
    """
    Mock C++ syntax validator.
    In production you'd shell out to:   clang++ -fsyntax-only -std=c++17 -x c++ -
    Here we use an LLM call + a fast regex pre-check.
    """

    # Fast heuristic pre-checks (no LLM needed)
    _OBVIOUS_ERRORS: list[tuple[str, str]] = [
        (r"\bmain\s*\(", "contains main() – should be a snippet"),
        (r"#include\s+\"",   "uses quoted includes – relative path may not resolve"),
    ]

    def __init__(self, model_name: str = "gemini", api_keys: Optional[dict] = None):
        self.model_name = model_name
        self.api_keys = api_keys or {}

    def act(self, code: str) -> tuple[bool, str]:
        """Returns (passed: bool, reason: str)."""
        # 1. Quick heuristic checks
        unmatched_braces = code.count("{") - code.count("}")
        if abs(unmatched_braces) > 2:
            return False, f"Unbalanced braces: {unmatched_braces:+d}"

        # 2. LLM syntax check
        if self.model_name == "gemini":
            verdict = _call_gemini(
                EXECUTOR_SYSTEM, code,
                api_key=self.api_keys.get("gemini"),
                temperature=0.0,
                max_tokens=64,
            )
        else:
            verdict = _call_nim(
                EXECUTOR_SYSTEM, code,
                model_key=self.model_name,
                api_key=self.api_keys.get("nvidia"),
                temperature=0.0,
                max_tokens=64,
            )

        verdict = verdict.strip()
        passed = verdict.upper().startswith("PASS")
        reason = verdict if not passed else "OK"
        return passed, reason


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
    max_iterations: int = MAX_REACT_ITERATIONS,
) -> ReActResult:
    """
    ReAct loop:
        Thought  – programmer decides what to write
        Action   – programmer generates code
        Observe  – executor checks syntax; critic identifies smells
        (repeat up to max_iterations)

    Returns the best final code + the full agent trajectory.
    """
    trajectory: list[AgentMessage] = []
    current_code = ""
    critic_feedback = ""
    executor_passed = False

    for iteration in range(1, max_iterations + 1):

        # ── Thought + Action (Programmer) ───────────
        thought = (
            f"[Iter {iteration}] Programmer generating code"
            + (" with critic feedback." if critic_feedback else ".")
        )
        trajectory.append(AgentMessage("thought", thought))

        current_code = programmer.act(prompt, critic_feedback)
        trajectory.append(AgentMessage("programmer", current_code))

        # ── Observation 1: Executor ──────────────────
        executor_passed, exec_reason = executor.act(current_code)
        trajectory.append(
            AgentMessage("executor", f"PASS" if executor_passed else f"FAIL: {exec_reason}")
        )

        # ── Observation 2: Critic ───────────────────
        critic_feedback = critic.act(current_code)
        trajectory.append(AgentMessage("critic", critic_feedback))

        # ── Termination condition ────────────────────
        if executor_passed and not CriticAgent.has_issues(critic_feedback):
            break   # Code passes validation and critic has no complaints

        time.sleep(0.2)   # avoid rate-limit bursts

    return ReActResult(
        final_code=current_code,
        iterations=iteration,
        trajectory=trajectory,
        executor_passed=executor_passed,
    )


def agentic_generate(
    prompt: str,
    model_name: str = "gemini",
    n_attempts: int = 1,
    api_keys: Optional[dict] = None,
) -> list[str]:
    """
    Convenience wrapper: run the full ReAct pipeline `n_attempts` times
    and return the list of final code strings.  (Usually n_attempts=1 for
    the agentic condition – it's expensive.)
    """
    programmer = ProgrammerAgent(model_name, api_keys)
    critic      = CriticAgent(model_name, api_keys)
    executor    = ExecutorAgent(model_name, api_keys)

    results: list[str] = []
    for _ in range(n_attempts):
        result = react_loop(prompt, programmer, critic, executor)
        results.append(result.final_code)

    return results