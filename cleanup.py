"""Transcript cleanup via local Ollama. Falls back to the raw transcript on any failure."""
import logging

import requests

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You clean up raw speech-to-text transcripts. Rules:
- Remove filler words (um, uh, like, you know) and false starts.
- Resolve self-corrections: "let's meet at 5, actually 6" becomes "let's meet at 6".
- Fix punctuation, capitalization, and sentence breaks.
- NEVER change the meaning, add new content, answer questions, or comment on the text.
- Output ONLY the cleaned text, nothing else."""


def _is_reasonable(cleaned: str, source: str) -> bool:
    """ponytail: small local models sometimes answer/rewrite instead of just cleaning;
    a wild length swing is the cheap signal that it drifted from the source."""
    return bool(cleaned) and len(source) * 0.6 <= len(cleaned) <= len(source) * 1.4


def clean(text: str, ollama_cfg: dict) -> str:
    """Return cleaned text, or the raw transcript if Ollama fails or drifts."""
    try:
        resp = requests.post(
            f"{ollama_cfg['url'].rstrip('/')}/api/chat",
            json={
                "model": ollama_cfg["model"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=60,
        )
        resp.raise_for_status()
        cleaned = resp.json()["message"]["content"].strip()
        if not _is_reasonable(cleaned, text):
            log.warning("cleanup output missing or length-deviated from source; pasting raw transcript")
            return text
        return cleaned
    except Exception as e:
        log.warning("Ollama cleanup failed (%s); pasting raw transcript", e)
        return text


def ollama_ready(ollama_cfg: dict) -> tuple[bool, str]:
    """(ok, message). Checks server reachability and that the model is pulled."""
    url = ollama_cfg["url"].rstrip("/")
    try:
        resp = requests.get(f"{url}/api/tags", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        return False, f"Ollama not reachable at {url}: {e}"
    names = [m["name"] for m in resp.json().get("models", [])]
    want = ollama_cfg["model"]
    if not any(n == want or n.split(":")[0] == want for n in names):
        return False, (
            f"Model '{want}' not found in Ollama. Run: ollama pull {want}\n"
            f"Available: {', '.join(names) or '(none)'}"
        )
    return True, "ok"
