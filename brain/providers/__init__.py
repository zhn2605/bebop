"""Model providers behind one Backend/Session interface, picked by name.

auto picks Gemini when GEMINI_API_KEY or GOOGLE_API_KEY is set, else Claude Code.
"""

import importlib
import os
import shutil

from brain.providers.base import Backend, BackendError

# Name to module and class. A module is imported only when its provider is picked, so no provider
# needs another's SDK installed.
_BACKENDS = {
    "claude-code": ("claude_code", "ClaudeCodeBackend"),
    "gemini": ("gemini", "GeminiBackend"),
}
NAMES = ("auto", *_BACKENDS)
KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def pick(name: str) -> str:
    """The provider that runs for name; auto is gemini when a key is set, else claude-code."""
    if name == "auto":
        return "gemini" if any(os.environ.get(var) for var in KEY_VARS) else "claude-code"
    if name not in _BACKENDS:
        raise ValueError(f"unknown provider {name!r}; known: {', '.join(NAMES)}")
    return name


def backend(name: str, **options) -> Backend:
    """The provider called name (or picked by auto), made with options such as model."""
    name = pick(name)
    if name == "claude-code" and shutil.which(options.get("claude_bin", "claude")) is None:
        raise BackendError(
            "claude is not on PATH: install Claude Code, then run claude once to log in"
        )
    module, cls = _BACKENDS[name]
    return getattr(importlib.import_module(f"{__name__}.{module}"), cls)(**options)
