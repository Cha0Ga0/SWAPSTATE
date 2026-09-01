from __future__ import annotations

from typing import Any, Dict, List, Optional


def _extract_messages(req: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """Extract chat messages from an API-style request.

    Supported shapes:
    - {"body": {"messages": [...]}}
    - {"messages": [...]}  (flattened)
    """
    if isinstance(req.get("body"), dict) and isinstance(req["body"].get("messages"), list):
        return req["body"]["messages"]
    if isinstance(req.get("messages"), list):
        return req["messages"]
    return None


def _extract_prompt(req: Dict[str, Any]) -> Optional[str]:
    """Extract a plain prompt from an API-style request."""
    if isinstance(req.get("body"), dict) and isinstance(req["body"].get("prompt"), str):
        return req["body"]["prompt"]
    if isinstance(req.get("prompt"), str):
        return req["prompt"]
    return None


def build_prompt_from_request(tokenizer: Any, req: Dict[str, Any]) -> str:
    """Build a model prompt from one JSONL request object.

    Preference order:
    1) If messages exist, use tokenizer.apply_chat_template(...) when available.
    2) Else, use req.body.prompt or req.prompt.

    The returned string is the *prompt only* (no assistant continuation).
    """
    messages = _extract_messages(req)
    if messages is not None:
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        chat_template = getattr(tokenizer, "chat_template", None)
        if callable(apply_chat_template) and chat_template:
            return apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # fallback: minimal concatenation
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"[{role}] {content}")
        parts.append("[assistant]")
        return "\n".join(parts)

    prompt = _extract_prompt(req)
    if prompt is not None:
        return prompt

    # last resort: stringify
    return str(req)
