"""
llm_client.py
-------------
LLM provider abstraction with automatic failover, retry/backoff,
and JSON extraction.

Changes from v1:
  - Added exponential backoff retry (3 attempts per provider)
  - Added rate-limit detection (429 errors skip to next provider faster)
  - Improved JSON extraction (handles nested, malformed, partial JSON)
  - Added token usage logging for cost monitoring
  - Fixed: empty string response no longer silently retried as success
  - Added SYSTEM prompt support (separate system/user messages)
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from openai import OpenAI
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# ── Auto-load .env ─────────────────────────────────────────────────────────────
_env = Path(".env")
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Provider registry ─────────────────────────────────────────────────────────
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key":  "ollama",
        "model":    "llama3.2:latest",
        "label":    "Ollama/llama3.2 (local)",
        "key_env":  None,
        # Retry config per provider
        "max_retries": 2,
        "retry_delay": 1.0,
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key":  None,
        "model":    "llama-3.3-70b-versatile",
        "label":    "Groq",
        "key_env":  "GROQ_API_KEY",
        "max_retries": 3,
        "retry_delay": 2.0,  # Groq has rate limits
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key":  None,
        "model":    "mistralai/mistral-7b-instruct:free",
        "label":    "OpenRouter",
        "key_env":  "OPENROUTER_API_KEY",
        "max_retries": 2,
        "retry_delay": 1.5,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key":  None,
        "model":    "gpt-4o-mini",
        "label":    "OpenAI",
        "key_env":  "OPENAI_API_KEY",
        "max_retries": 3,
        "retry_delay": 1.0,
    },
}

# Provider priority — first available is tried first
PRIORITY = ["groq", "ollama", "openrouter", "openai"]


# ── Provider availability ─────────────────────────────────────────────────────

def _ollama_running() -> bool:
    """Check if local Ollama server is reachable."""
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def _is_available(name: str) -> bool:
    """Return True if provider has required credentials / is reachable."""
    cfg = PROVIDERS[name]
    if name == "ollama":
        return _ollama_running()
    key_env = cfg.get("key_env")
    return bool(key_env and os.environ.get(key_env, "").strip())


def _get_client(name: str):
    cfg  = PROVIDERS[name]
    api_key = cfg["api_key"] or os.environ.get(cfg.get("key_env", "") or "", "").strip()
    return OpenAI(api_key=api_key, base_url=cfg["base_url"])


# ── Core LLM call ─────────────────────────────────────────────────────────────

def call_llm(
    prompt:      str,
    max_tokens:  int            = 800,
    system:      Optional[str]  = None,
    temperature: float          = 0.1,
) -> str:
    """
    Call the first available LLM provider with automatic failover and retry.

    FIX v2:
      - Added exponential backoff retry per provider
      - Rate-limit (429) triggers immediate skip to next provider
      - Empty string response is treated as failure (not success)
      - System prompt supported via separate message role
      - Token usage logged at DEBUG level

    Args:
        prompt:      User message / full prompt text
        max_tokens:  Maximum tokens in response
        system:      Optional system prompt (sent as role="system")
        temperature: Sampling temperature (default 0.1 for consistency)

    Returns:
        Response text from first successful provider.

    Raises:
        RuntimeError: If all providers fail.
    """
    available = [p for p in PRIORITY if _is_available(p)]

    if not available:
        raise RuntimeError(
            "\n\nNo LLM provider available!\n\n"
            "Option A — Start Ollama:\n"
            "  ollama serve\n\n"
            "Option B — Add API key to .env:\n"
            "  GROQ_API_KEY=gsk_...   (free at console.groq.com)\n"
            "  OPENAI_API_KEY=sk_...\n"
        )

    # Build message list
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    all_errors = []

    for name in available:
        cfg         = PROVIDERS[name]
        max_retries = cfg.get("max_retries", 2)
        retry_delay = cfg.get("retry_delay", 1.0)

        for attempt in range(1, max_retries + 1):
            try:
                client = _get_client(name)
                log.debug(
                    "Calling %s (attempt %d/%d)...", cfg["label"], attempt, max_retries
                )

                resp = client.chat.completions.create(
                    model=cfg["model"],
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                text = resp.choices[0].message.content
                if text:
                    text = text.strip()

                # FIX: Empty string was silently returned as success before
                if not text:
                    raise ValueError("Provider returned empty response")

                # Log token usage if available
                if hasattr(resp, "usage") and resp.usage:
                    log.debug(
                        "%s tokens: prompt=%d completion=%d total=%d",
                        cfg["label"],
                        resp.usage.prompt_tokens,
                        resp.usage.completion_tokens,
                        resp.usage.total_tokens,
                    )

                log.info(
                    "LLM response via %s | %d chars | attempt %d",
                    cfg["label"], len(text), attempt,
                )
                return text

            except Exception as e:
                err_str = str(e)

                # Rate limit — skip remaining retries, go to next provider
                if "429" in err_str or "rate limit" in err_str.lower():
                    log.warning(
                        "%s rate-limited — skipping to next provider.", cfg["label"]
                    )
                    all_errors.append(f"{cfg['label']} [rate-limited]: {err_str[:80]}")
                    break

                log.warning(
                    "%s attempt %d/%d failed: %s",
                    cfg["label"], attempt, max_retries, err_str[:100],
                )
                all_errors.append(
                    f"{cfg['label']} attempt {attempt}: {err_str[:80]}"
                )

                # Exponential backoff before retry
                if attempt < max_retries:
                    sleep_time = retry_delay * (2 ** (attempt - 1))
                    log.debug("Retrying in %.1fs...", sleep_time)
                    time.sleep(sleep_time)

    raise RuntimeError(
        "All LLM providers failed after retries:\n"
        + "\n".join(f"  • {e}" for e in all_errors)
        + "\n\nDebug: run python test_llm.py"
    )


# ── JSON-mode call ────────────────────────────────────────────────────────────

def call_llm_json(
    prompt:     str,
    max_tokens: int = 600,
    system:     Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call LLM and parse response as JSON.

    FIX v2:
      - Multi-stage extraction: tries strict parse → regex object → regex array
      - Handles markdown code fences (```json ... ```)
      - Handles JSON embedded in prose
      - Returns {} on failure (never raises — callers handle missing keys)
      - Added truncation logging so parse failures are easier to diagnose

    Args:
        prompt:     User prompt (JSON instruction will be appended)
        max_tokens: Maximum response tokens
        system:     Optional system prompt

    Returns:
        Parsed dict/list, or {} on failure.
    """
    json_instruction = (
        "\n\nYou MUST respond with ONLY valid JSON. "
        "No markdown fences, no explanation, no extra text. "
        "Start your response with { or [."
    )

    raw = call_llm(
        prompt + json_instruction,
        max_tokens=max_tokens,
        system=system,
    )

    # Stage 1: Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    # Stage 2: Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Stage 3: Extract first JSON object
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Stage 4: Extract outermost JSON object (greedy, handles nesting)
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Stage 5: Extract JSON array
    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            # Wrap array in dict so callers get consistent type
            return {"items": result}
        except json.JSONDecodeError:
            pass

    log.warning(
        "call_llm_json: all parse stages failed. Raw (first 300 chars):\n%s",
        raw[:300],
    )
    return {}