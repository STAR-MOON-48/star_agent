from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from ...protocols import JsonDict, ensure_json_dict


@dataclass(frozen=True)
class ModelMessage:
    """Provider-neutral chat message."""

    role: str
    content: str

    def to_dict(self) -> JsonDict:
        return {"role": self.role, "content": self.content}


@dataclass
class ModelResult:
    """Provider-neutral model response."""

    text: str
    model: Optional[str] = None
    usage: JsonDict = field(default_factory=dict)
    raw: Any = None

    def __post_init__(self) -> None:
        self.usage = ensure_json_dict(self.usage)


class ModelInterface(ABC):
    """Boundary for model access."""

    @abstractmethod
    async def chat(
        self,
        messages: Any,
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResult:
        """Send chat messages to the configured model."""

    async def generate_text(
        self,
        messages: Any,
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        result = await self.chat(messages, model=model, **kwargs)
        return result.text
