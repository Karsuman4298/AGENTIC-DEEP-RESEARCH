
import logging
import os
import re
from pathlib import Path

log = logging.getLogger(__name__)

# Auto-load .env
_env = Path(".env")
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

PROVIDERS = {
    "ollama": {
        "base_url":    "http://localhost:11434/v1",
        "api_key":     "ollama",
        "model":       "llama3.2:latest",     
        "label":       "Ollama/llama3.2 (local)",
        "key_env":     None,                   
    },
    "groq": {
        "base_url":    "https://api.groq.com/openai/v1",
        "api_key":     None,                   
        "model":       "llama-3.3-70b-versatile",  
        "label":       "Groq (free)",
        "key_env":     "GROQ_API_KEY",
    },
    "openrouter": {
        "base_url":    "https://openrouter.ai/api/v1",
        "api_key":     None,
        "model":       "mistralai/mistral-7b-instruct:free",
        "label":       "OpenRouter (free)",
        "key_env":     "OPENROUTER_API_KEY",
    },
    "openai": {
        "base_url":    "https://api.openai.com/v1",
        "api_key":     None,
        "model":       "gpt-4o-mini",
        "label":       "OpenAI",
        "key_env":     "OPENAI_API_KEY",
    },
}

#set your own priority
PRIORITY = [ "groq","ollama","openrouter", "openai"]

def _ollama_running() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def _is_available(name: str) -> bool:
    cfg = PROVIDERS[name]
    if name == "ollama":
        return _ollama_running()
    key_env = cfg["key_env"]
    return bool(key_env and os.environ.get(key_env, "").strip())


def _get_client(name: str):
    from openai import OpenAI
    cfg = PROVIDERS[name]
    api_key = cfg["api_key"] or os.environ.get(cfg["key_env"], "").strip()
    return OpenAI(api_key=api_key, base_url=cfg["base_url"])


def call_llm(prompt: str, max_tokens: int = 800) -> str:
    available = [p for p in PRIORITY if _is_available(p)]

    if not available:
        raise RuntimeError(
            "\nNo LLM provider available!\n\n"
            "You have Ollama installed — just run:\n"
            "  ollama serve          (start Ollama if not running)\n\n"
            "Or add a key to .env:\n"
            "  GROQ_API_KEY=gsk_...  (free at console.groq.com)\n"
        )

    errors = []
    for name in available:
        cfg = PROVIDERS[name]
        try:
            client = _get_client(name)
            log.debug(f"Trying {cfg['label']}...")
            resp = client.chat.completions.create(
                model=cfg["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            text = resp.choices[0].message.content.strip()
            if text:
                log.info(f"LLM via {cfg['label']}")
                return text
        except Exception as e:
            err = str(e)[:120]
            errors.append(f"{cfg['label']}: {err}")
            log.warning(f"{cfg['label']} failed: {err}")

    raise RuntimeError(
        "All providers failed:\n" +
        "\n".join(f"  • {e}" for e in errors) +
        "\n\nRun: python test_llm.py"
    )


def call_llm_json(prompt: str, max_tokens: int = 600) -> dict:
    import json
    raw = call_llm(
        prompt + "\n\nRespond ONLY with valid JSON. No markdown, no explanation.",
        max_tokens,
    )
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    log.warning(f"JSON parse failed:\n{raw[:200]}")
    return {}