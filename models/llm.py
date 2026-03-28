import os
import sys
from typing import List, Optional

from groq import BadRequestError, Groq

from config.config import load_config

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


def _candidate_models(primary: str) -> List[str]:
    ordered = [primary] + [m for m in FALLBACK_MODELS if m != primary]
    return ordered


def get_chatgroq_model():
    """Initialize and return Groq client details, or None if config is missing."""
    try:
        cfg = load_config()
        if not cfg.groq_api_key:
            return None

        client = Groq(api_key=cfg.groq_api_key)
        return {"client": client, "model": cfg.groq_model}
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}") from e


def generate_response(
    model,
    system_prompt: str,
    memory_messages: List[dict],
    user_prompt: str,
    rag_context: Optional[str] = None,
) -> str:
    """Generate a response with optional RAG context and short-term memory."""
    if model is None:
        return "I can help, but no LLM is configured. Add GROQ_API_KEY to enable richer responses."

    messages = [{"role": "system", "content": system_prompt}]
    for item in memory_messages:
        if item.get("role") == "user":
            messages.append({"role": "user", "content": item.get("content", "")})
        elif item.get("role") == "assistant":
            messages.append({"role": "assistant", "content": item.get("content", "")})

    final_prompt = user_prompt
    if rag_context:
        final_prompt = (
            "Use only the following retrieved context when it is relevant. "
            "If context is insufficient, clearly say so.\n\n"
            f"Context:\n{rag_context}\n\n"
            f"User query: {user_prompt}"
        )
    messages.append({"role": "user", "content": final_prompt})

    last_error = None
    for model_name in _candidate_models(model["model"]):
        try:
            response = model["client"].chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
            )
            if model_name != model["model"]:
                model["model"] = model_name
            return response.choices[0].message.content or ""
        except BadRequestError as exc:
            last_error = exc
            err = getattr(exc, "body", {}) or {}
            code = (err.get("error") or {}).get("code")
            if code == "model_decommissioned":
                continue
            return f"LLM request failed: {exc}"
        except Exception as exc:
            return f"LLM request failed: {exc}"

    return (
        "The configured model is unavailable. "
        "Update GROQ_MODEL to a supported model (for example, llama-3.3-70b-versatile)."
    )