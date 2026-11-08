"""Claude Code as the VLM: one long-lived `claude -p` process per session, stream-json both ways."""

import base64
import json
import math
import os
import queue
import subprocess
import threading
import time
from collections import deque
from collections.abc import Mapping

from brain.providers.base import BackendError, Block, Image, Tool, Turn

_STDERR_LINES = 20
_CLOSE_WAIT_S = 5
# What a parent Claude Code session sets for the commands it runs. Inherited, they change how
# claude behaves (CLAUDE_EFFORT=max raises cost and latency) and tie it to the parent (the
# messaging socket and token). Auth and config vars such as CLAUDE_CONFIG_DIR,
# CLAUDE_CODE_OAUTH_TOKEN and ANTHROPIC_* must reach claude, so this is a list, not CLAUDE_*.
_PARENT_SESSION_VARS = {
    "CLAUDECODE", "CLAUDE_EFFORT", "CLAUDE_PID",
    "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_BRIDGE_SESSION_ID", "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_EXECPATH", "CLAUDE_CODE_SESSION_ATTENDED",
    "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN",
}


def turn_schema(tools: list[Tool]) -> dict:
    """One turn's output: a short note and exactly one action, with an anyOf branch per tool."""
    actions = [
        {
            "type": "object",
            "description": tool.description,
            "properties": {"tool": {"const": tool.name}, "args": tool.params},
            "required": ["tool", "args"],
            "additionalProperties": False,
        }
        for tool in tools
    ]
    # "note", not "thought": in a probe on Claude Code 2.1.296, Sonnet 5.5's reasoning_extraction
    # safeguard refused the turns keyed "thought". It was not reproduced later (0 of 10 refused),
    # but "note" is the safer key.
    return {
        "type": "object",
        "properties": {"note": {"type": "string"}, "action": {"anyOf": actions}},
        "required": ["note", "action"],
        "additionalProperties": False,
    }


def parse_result(event: dict, spent: float = 0.0) -> Turn:
    """The Turn in a stream-json result event; BackendError if the turn failed or is malformed.

    total_cost_usd is the session's running total, so the turn costs what it adds to spent.
    """
    if event.get("is_error"):
        # Budget errors give their reason in errors; API errors and refusals give it in result.
        detail = "; ".join(map(str, event.get("errors") or [])) or event.get("result")
        raise BackendError(
            f"claude failed ({event.get('subtype')}): {detail or 'no details'}", event
        )
    out = event.get("structured_output")
    action = out.get("action") if isinstance(out, dict) else None
    if not (
        isinstance(action, dict)
        and isinstance(out.get("note"), str)
        and isinstance(action.get("tool"), str)
        and isinstance(action.get("args"), dict)
    ):
        raise BackendError(
            f"missing or malformed structured_output: {json.dumps(out)[:300]}", event
        )
    total = event.get("total_cost_usd")
    cost = total - spent if isinstance(total, int | float) else None
    return Turn(out["note"], action["tool"], action["args"], cost, event)


def _content(block: Block) -> dict:
    if isinstance(block, Image):
        data = base64.b64encode(block.data).decode()
        source = {"type": "base64", "media_type": block.media_type, "data": data}
        return {"type": "image", "source": source}
    return {"type": "text", "text": block.text}


def _child_env(environ: Mapping[str, str]) -> dict[str, str]:
    """environ without a parent Claude Code session's markers; auth and config vars stay."""
    return {name: value for name, value in environ.items() if name not in _PARENT_SESSION_VARS}


class ClaudeCodeSession:
    """One `claude -p` process, so the conversation keeps its context from send to send.

    A send that gets no result (timeout, EOF, Ctrl+C) kills the process, since a late result would
    be read as the next send's answer; later sends raise BackendError. A failed result does not.
    """

    def __init__(self, cmd: list[str], timeout_s: float):
        self.timeout_s = timeout_s
        try:
            # Its own session keeps a terminal's Ctrl+C away from claude; our caller stops the arm
            # and closes this session.
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
                env=_child_env(os.environ),
                start_new_session=True,
            )
        except OSError as e:
            raise BackendError(f"cannot run {cmd[0]}: {e}") from e
        self._dead: str | None = None  # why sends no longer work
        self._spent = 0.0
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=_STDERR_LINES)
        self._stdout_reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_reader.start()
        self._stderr_reader.start()

    def send(self, blocks: list[Block]) -> Turn:
        if self._dead:
            raise BackendError(f"the claude session is over: {self._dead}")
        content = [_content(block) for block in blocks]
        message = {"type": "user", "message": {"role": "user", "content": content}}
        deadline = time.monotonic() + self.timeout_s
        try:
            self._write(json.dumps(message) + "\n", deadline)
            event = self._read_result(deadline)
        except BaseException as e:
            self._dead = (str(e) or type(e).__name__).splitlines()[0]
            self._kill()
            raise
        spent, total = self._spent, event.get("total_cost_usd")
        if isinstance(total, int | float):
            self._spent = total
        return parse_result(event, spent)

    def close(self) -> None:
        """Close stdin so claude exits, and kill it if it hasn't in a few seconds. Idempotent."""
        self._dead = self._dead or "closed"
        try:
            self._proc.stdin.close()
        except OSError:
            pass  # a dead claude cannot take the last flush
        try:
            self._proc.wait(timeout=_CLOSE_WAIT_S)
        except subprocess.TimeoutExpired:
            self._kill()

    def _write(self, line: str, deadline: float) -> None:
        # A message bigger than the pipe buffer (64 KiB; a webcam JPEG is) blocks here until claude
        # reads it, so a claude that doesn't is killed at the deadline, which breaks the pipe.
        timer = threading.Timer(max(deadline - time.monotonic(), 0), self._proc.kill)
        timer.start()
        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except OSError as e:
            if time.monotonic() >= deadline:
                raise BackendError(
                    f"claude did not read the message in {self.timeout_s:g}s{self._tail()}"
                ) from None
            raise BackendError(f"cannot write to claude: {e}{self._tail()}") from e
        finally:
            timer.cancel()

    def _read_result(self, deadline: float) -> dict:
        while True:
            try:
                line = self._lines.get(timeout=max(deadline - time.monotonic(), 0))
            except queue.Empty:
                raise BackendError(
                    f"no result from claude in {self.timeout_s:g}s{self._tail()}"
                ) from None
            if line is None:
                raise self._exited()
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            # system/init, assistant, user and rate_limit_event come first; result ends the turn.
            if isinstance(event, dict) and event.get("type") == "result":
                return event

    def _exited(self) -> BackendError:
        # stdout closes a moment before the exit code and the last of stderr are in.
        try:
            code = self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            code = None
        self._stderr_reader.join(timeout=2)
        return BackendError(f"claude exited before answering (exit code {code}){self._tail()}")

    def _kill(self) -> None:
        self._proc.kill()
        self._proc.wait()

    def _read_stdout(self) -> None:
        try:
            for line in self._proc.stdout:
                self._lines.put(line)
        finally:
            self._lines.put(None)
            self._proc.stdout.close()

    def _read_stderr(self) -> None:
        # Drained all the time: claude blocks once a full pipe has no reader.
        for line in self._proc.stderr:
            self._stderr.append(line.rstrip("\n"))
        self._proc.stderr.close()

    def _tail(self) -> str:
        lines = list(self._stderr)
        return "\nclaude stderr:\n" + "\n".join(lines) if lines else ""


class ClaudeCodeBackend:
    """Claude Code in print mode; its StructuredOutput tool enforces the turn schema."""

    def __init__(
        self,
        model: str = "sonnet",
        effort: str | None = None,
        max_budget_usd: float | None = None,
        timeout_s: float = 180,
        claude_bin: str = "claude",
    ):
        for name, value in (("timeout_s", timeout_s), ("max_budget_usd", max_budget_usd)):
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be finite and positive, got {value}")
        self.model = model
        self.effort = effort
        self.max_budget_usd = max_budget_usd
        self.timeout_s = timeout_s
        self.claude_bin = claude_bin

    def start(self, system: str, tools: list[Tool]) -> ClaudeCodeSession:
        cmd = [
            self.claude_bin, "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",  # stream-json output needs it
            # No hooks, plugins or CLAUDE.md in the model's context, and no tools but
            # StructuredOutput.
            "--safe-mode", "--tools", "",
            "--no-session-persistence",
            "--model", self.model,
            "--system-prompt", system,
            "--json-schema", json.dumps(turn_schema(tools)),
        ]
        if self.effort is not None:
            cmd += ["--effort", self.effort]
        if self.max_budget_usd is not None:
            # Checked after each turn: the turn that crosses it fails with error_max_budget_usd.
            cmd += ["--max-budget-usd", str(self.max_budget_usd)]
        return ClaudeCodeSession(cmd, self.timeout_s)
