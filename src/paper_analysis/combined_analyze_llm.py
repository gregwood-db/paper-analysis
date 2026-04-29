from __future__ import annotations

import json
import os

import anthropic
from openai import OpenAI
from pydantic import ValidationError

from paper_analysis.combined_prompts import COMBINED_JSON_FIX, COMBINED_SYSTEM, COMBINED_USER
from paper_analysis.combined_schemas import CombinedExtractionBatch
from paper_analysis.vision.base import load_json_object_from_model_text

_MAX_TOKENS = 64000


def parse_combined_extraction_batch(text: str) -> CombinedExtractionBatch:
    raw = load_json_object_from_model_text(text)
    return CombinedExtractionBatch.model_validate(raw)


def _collect_anthropic_stream(stream: anthropic.Stream) -> str:
    """Consume a streaming response and return the concatenated text."""
    parts: list[str] = []
    for event in stream:
        if event.type == "content_block_delta" and hasattr(event.delta, "text"):
            parts.append(event.delta.text)
    return "".join(parts)


def run_combined_analysis_anthropic(
    user_content: str,
    *,
    model: str | None = None,
    max_retries: int = 1,
) -> CombinedExtractionBatch:
    m = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    client = anthropic.Anthropic()
    attempts = max_retries + 1
    user_msg = COMBINED_USER + "\n\n--- INPUT ---\n\n" + user_content
    last_err: Exception | None = None
    for attempt in range(attempts):
        text = user_msg if attempt == 0 else user_msg + COMBINED_JSON_FIX
        with client.messages.stream(
            model=m,
            max_tokens=_MAX_TOKENS,
            system=COMBINED_SYSTEM,
            messages=[{"role": "user", "content": text}],
        ) as stream:
            body = _collect_anthropic_stream(stream)
        try:
            return parse_combined_extraction_batch(body)
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            last_err = e
            if attempt >= attempts - 1:
                raise
    raise RuntimeError(f"combined analysis failed: {last_err}")


def run_combined_analysis_openai(
    user_content: str,
    *,
    model: str | None = None,
    max_retries: int = 1,
) -> CombinedExtractionBatch:
    m = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
    client = OpenAI()
    attempts = max_retries + 1
    user_msg = COMBINED_USER + "\n\n--- INPUT ---\n\n" + user_content
    last_err: Exception | None = None
    for attempt in range(attempts):
        text = user_msg if attempt == 0 else user_msg + COMBINED_JSON_FIX
        resp = client.chat.completions.create(
            model=m,
            max_tokens=_MAX_TOKENS,
            messages=[
                {"role": "system", "content": COMBINED_SYSTEM},
                {"role": "user", "content": text},
            ],
        )
        body = resp.choices[0].message.content or ""
        try:
            return parse_combined_extraction_batch(body)
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            last_err = e
            if attempt >= attempts - 1:
                raise
    raise RuntimeError(f"combined analysis failed: {last_err}")


def run_combined_analysis(
    user_content: str,
    *,
    provider: str,
    model: str | None = None,
    max_retries: int = 1,
) -> CombinedExtractionBatch:
    prov = provider.strip().lower()
    if prov == "anthropic":
        return run_combined_analysis_anthropic(user_content, model=model, max_retries=max_retries)
    if prov == "openai":
        return run_combined_analysis_openai(user_content, model=model, max_retries=max_retries)
    raise ValueError(f"Unknown combined LLM provider: {provider!r}")
