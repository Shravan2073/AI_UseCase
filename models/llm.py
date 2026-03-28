import os
import sys
from typing import List, Optional

from groq import Groq

from config.config import load_config

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


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

    response = model["client"].chat.completions.create(
        model=model["model"],
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""