from __future__ import annotations

from typing import Any, Optional, Sequence

from ...protocols import JsonDict
from .model import ModelInterface, ModelMessage, ModelResult
from menglong import Model


DEFAULT_MODEL_ID = "xiaomi/mimo-v2.5"


class StarModel(ModelInterface):
    """MengLong-backed model interface."""

    def __init__(
        self,
        *,
        default_model_id: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:

        self.default_model_id = default_model_id or DEFAULT_MODEL_ID
        self._model = Model(default_model_id=self.default_model_id, config_path=config_path)

    async def chat(
        self,
        messages: Any,
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResult:
        response = await self._model.async_chat(
            self._to_menglong_messages(messages),
            model=model,
            **kwargs,
        )
        usage = response.usage.model_dump() if response.usage else {}
        return ModelResult(
            text=response.text or "",
            model=response.model,
            usage=usage,
            raw=response,
        )

    def _to_menglong_messages(self, messages: Any) -> Any:
        if hasattr(messages, "messages"):
            return messages
        if isinstance(messages, str):
            return messages
        if not isinstance(messages, Sequence):
            return messages
        normalized: list[JsonDict] = []
        for message in messages:
            if isinstance(message, ModelMessage):
                normalized.append(message.to_dict())
            else:
                normalized.append(dict(message))
        return normalized
