"""Provider-neutral message blocks, tools and turns, and the Backend/Session interface."""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class Image:
    data: bytes
    media_type: str = "image/jpeg"
    label: str = ""  # what it shows, such as a camera's name; the run log names the file after it


Block = Text | Image


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    params: dict  # JSON schema with "type": "object" for the args


@dataclass
class Turn:
    note: str  # short visible caption of what the model sees and does
    tool: str
    args: dict
    cost_usd: float | None = None
    raw: dict = field(default_factory=dict)  # provider payload, for logs


class Session(Protocol):
    def send(self, blocks: list[Block]) -> Turn: ...

    def close(self) -> None: ...


class Backend(Protocol):
    def start(self, system: str, tools: list[Tool]) -> Session: ...


class BackendError(RuntimeError):
    def __init__(self, message: str, raw: dict | None = None):
        super().__init__(message)
        self.raw = raw  # provider payload of the failed turn, if there was one, for logs
