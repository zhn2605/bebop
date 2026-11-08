"""Gemini through google-genai: one generate_content per send, the conversation kept client-side.

The turn schema is Claude Code's (a note and one action per tool), adapted to the OpenAPI subset
that response_schema takes. google-genai is imported in start(), so other providers don't need it.
"""

from __future__ import annotations

import json
import math

from brain.providers.base import BackendError, Block, Image, Tool, Turn
from brain.providers.claude_code import turn_schema

DEFAULT_MODEL = "gemini-3.6-flash"
# USD per million tokens (input, output with thinking). Recalled for 2026-11-07, NOT checked against
# https://ai.google.dev/pricing; unknown models cost None.
PRICES = {
    "gemini-3.6-flash": (0.50, 3.00),
    "gemini-3.8-flash": (0.50, 3.00),
    "gemini-3.5-flash-lite": (0.10, 0.40),
}
_THINKING = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH", "xhigh": "HIGH", "max": "HIGH"}


def gemini_schema(schema):
    """schema without what response_schema rejects: const becomes a one-value enum, a type list
    becomes its first type plus nullable, additionalProperties goes, a one-branch anyOf is
    flattened; objects keep their key order as property_ordering."""
    if isinstance(schema, list):
        return [gemini_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for key, value in schema.items():
        if key == "additionalProperties":
            continue
        if key == "const":
            out["enum"] = [value]
            out.setdefault("type", "string")
        elif key == "type" and isinstance(value, list):
            types = [t for t in value if t != "null"]
            out["type"] = types[0] if types else "string"
            if "null" in value:
                out["nullable"] = True
        elif key == "properties":
            out[key] = {name: gemini_schema(s) for name, s in value.items()}
            out["property_ordering"] = list(value)
        else:
            out[key] = gemini_schema(value)
    if len(out.get("anyOf", ())) == 1:
        branch = out.pop("anyOf")[0]
        out = {**branch, **out}
    return out


def cost_usd(model: str, usage) -> float | None:
    if usage is None or model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    out = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    return ((usage.prompt_token_count or 0) * price_in + out * price_out) / 1e6


class GeminiSession:
    """A conversation: each send posts the whole history, so a retry sees the first answer."""

    def __init__(self, client, model: str, config, max_budget_usd: float | None = None):
        self._client, self.model, self.config = client, model, config
        self.max_budget_usd = max_budget_usd
        self._spent = 0.0
        self.history: list = []
        self._closed = False

    def contents(self, blocks: list[Block]):
        """The user Content for blocks: images as inline parts, text as text parts."""
        from google.genai import types

        parts = [
            types.Part.from_bytes(data=b.data, mime_type=b.media_type)
            if isinstance(b, Image) else types.Part(text=b.text)
            for b in blocks
        ]
        return types.Content(role="user", parts=parts)

    def send(self, blocks: list[Block]) -> Turn:
        from google.genai import types

        if self._closed:
            raise BackendError("the gemini session is closed")
        history = [*self.history, self.contents(blocks)]
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=history, config=self.config)
        except Exception as e:  # the SDK raises its own errors, httpx's and timeouts
            raise BackendError(f"gemini failed: {type(e).__name__}: {e}") from e
        raw = response.model_dump(mode="json", exclude_none=True)
        cost = cost_usd(self.model, response.usage_metadata)
        text = response.text
        if not text:
            feedback = response.prompt_feedback
            reason = feedback.block_reason if feedback else None
            if response.candidates:
                reason = reason or response.candidates[0].finish_reason
            raise BackendError(f"gemini returned no text (reason: {reason})", raw)
        try:
            out = json.loads(text)
        except json.JSONDecodeError:
            raise BackendError(f"gemini returned invalid JSON: {text[:300]}", raw) from None
        action = out.get("action") if isinstance(out, dict) else None
        if not (isinstance(action, dict) and isinstance(action.get("args"), dict)):
            raise BackendError(f"missing or malformed action: {text[:300]}", raw)
        # Like claude's --max-budget-usd: the turn that takes the session over it fails.
        self._spent += cost or 0.0
        if self.max_budget_usd is not None and self._spent > self.max_budget_usd:
            raise BackendError(
                f"the session cost ${self._spent:.4f}, over max_budget_usd ${self.max_budget_usd}",
                raw)
        self.history = [*history, types.Content(role="model", parts=[types.Part(text=text)])]
        return Turn(str(out.get("note", "")), str(action.get("tool", "")), action["args"], cost,
                    raw)

    def close(self) -> None:
        self._closed = True


class GeminiBackend:
    """Gemini with temperature 0 and a JSON response schema; the key comes from GOOGLE_API_KEY or
    GEMINI_API_KEY (the SDK prefers GOOGLE_API_KEY)."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        timeout_s: float = 60,
        max_budget_usd: float | None = None,
        effort: str | None = None,
    ):
        for name, value in (("timeout_s", timeout_s), ("max_budget_usd", max_budget_usd)):
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be finite and positive, got {value}")
        if effort is not None and effort not in _THINKING:
            raise ValueError(f"effort must be one of {', '.join(_THINKING)}, got {effort!r}")
        self.model = model
        self.timeout_s = timeout_s
        self.max_budget_usd = max_budget_usd
        self._spent = 0.0
        self.effort = effort

    def config(self, system: str, tools: list[Tool]):
        from google.genai import types

        thinking = None
        if self.effort is not None:
            thinking = types.ThinkingConfig(thinking_level=_THINKING[self.effort])
        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            response_mime_type="application/json",
            response_schema=gemini_schema(turn_schema(tools)),
            thinking_config=thinking,
            http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)),
        )

    def start(self, system: str, tools: list[Tool]) -> GeminiSession:
        try:
            from google import genai
        except ImportError as e:
            raise BackendError("google-genai is not installed: pip install 'google-genai>=2.9,<4'"
                               ) from e
        try:
            client = genai.Client()
        except Exception as e:  # no key: ValueError
            raise BackendError(f"cannot make a Gemini client (set GEMINI_API_KEY): {e}") from e
        return GeminiSession(client, self.model, self.config(system, tools), self.max_budget_usd)
