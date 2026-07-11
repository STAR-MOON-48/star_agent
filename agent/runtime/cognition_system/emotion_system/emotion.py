from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

from ....config import EmotionConfig
from ....protocols import AgentEvent, AgentState, JsonDict, utc_now


EMOTION_VARIABLE_KEY = "emotion_state"
POSITIVE_WORDS = {
    "谢谢",
    "感谢",
    "很好",
    "成功",
    "完成",
    "开心",
    "高兴",
    "喜欢",
    "信任",
    "great",
    "good",
    "success",
    "thanks",
}
NEGATIVE_WORDS = {
    "失败",
    "错误",
    "生气",
    "难过",
    "失望",
    "讨厌",
    "危险",
    "取消",
    "bad",
    "error",
    "failed",
    "angry",
    "sad",
}
HIGH_AROUSAL_WORDS = {"紧急", "立刻", "马上", "危险", "愤怒", "urgent", "now"}


class EmotionSystem:
    """Persistent PAD-style affect state updated from attributed events."""

    def __init__(self, config: EmotionConfig) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def observe_event(self, state: AgentState, event: AgentEvent) -> JsonDict:
        emotion = self._state(state)
        if not self.enabled:
            return emotion
        self._decay(emotion)
        delta_valence, delta_arousal, delta_dominance, reason = self._event_delta(event)
        sensitivity = self.config.sensitivity
        emotion["valence"] = self._clamp(
            float(emotion.get("valence", 0.0)) + delta_valence * sensitivity
        )
        emotion["arousal"] = self._clamp(
            float(emotion.get("arousal", 0.15)) + delta_arousal * sensitivity,
            low=0.0,
        )
        emotion["dominance"] = self._clamp(
            float(emotion.get("dominance", 0.0)) + delta_dominance * sensitivity
        )
        emotion["primary"] = self._primary(emotion)
        emotion["mood"] = self._mood(emotion)
        emotion["intensity"] = round(
            max(abs(float(emotion["valence"])), float(emotion["arousal"])),
            4,
        )
        emotion["updated_at"] = utc_now()
        emotion["last_trigger"] = {
            "event_id": event.event_id,
            "event_type": event.type,
            "reason": reason,
        }
        history = emotion.setdefault("history", [])
        if not isinstance(history, list):
            history = []
        if any(abs(value) > 0.001 for value in (delta_valence, delta_arousal, delta_dominance)):
            history.append(
                {
                    "event_id": event.event_id,
                    "event_type": event.type,
                    "delta": {
                        "valence": delta_valence,
                        "arousal": delta_arousal,
                        "dominance": delta_dominance,
                    },
                    "reason": reason,
                    "created_at": utc_now(),
                }
            )
        emotion["history"] = history[-self.config.history_limit :]
        self._update_social_target(emotion, event, delta_valence)
        return emotion

    def context_view(self, state: AgentState) -> JsonDict:
        emotion = self._state(state)
        self._decay(emotion)
        return {
            key: value
            for key, value in emotion.items()
            if key != "history"
        }

    def _state(self, state: AgentState) -> JsonDict:
        emotion = state.workspace.variables.setdefault(
            EMOTION_VARIABLE_KEY,
            {
                "valence": 0.0,
                "arousal": 0.15,
                "dominance": 0.0,
                "primary": "neutral",
                "mood": "calm",
                "intensity": 0.15,
                "history": [],
                "toward": {},
                "updated_at": utc_now(),
            },
        )
        if not isinstance(emotion, dict):
            emotion = {}
            state.workspace.variables[EMOTION_VARIABLE_KEY] = emotion
        emotion.setdefault("valence", 0.0)
        emotion.setdefault("arousal", 0.15)
        emotion.setdefault("dominance", 0.0)
        emotion.setdefault("history", [])
        emotion.setdefault("toward", {})
        emotion.setdefault("updated_at", utc_now())
        return emotion

    def _decay(self, emotion: JsonDict) -> None:
        updated_at = self._parse_datetime(emotion.get("updated_at"))
        if updated_at is None:
            return
        elapsed = max(0.0, (datetime.now(timezone.utc) - updated_at).total_seconds())
        if elapsed <= 0:
            return
        factor = math.pow(0.5, elapsed / self.config.decay_half_life_seconds)
        emotion["valence"] = round(float(emotion.get("valence", 0.0)) * factor, 4)
        emotion["arousal"] = round(
            0.15 + (float(emotion.get("arousal", 0.15)) - 0.15) * factor,
            4,
        )
        emotion["dominance"] = round(float(emotion.get("dominance", 0.0)) * factor, 4)

    def _event_delta(self, event: AgentEvent) -> tuple[float, float, float, str]:
        if event.type == "action.completed":
            return 0.18, 0.08, 0.12, "action_completed"
        if event.type == "action.failed":
            return -0.28, 0.22, -0.16, "action_failed"
        if event.type == "action.cancelled":
            return -0.12, 0.08, -0.08, "action_cancelled"
        if event.type == "action.internal.completed":
            command = str(event.payload.get("internal_command_type") or "")
            if command == "complete_task":
                return 0.24, 0.06, 0.18, "task_completed"
        text = self._event_text(event).casefold()
        positive = sum(1 for word in POSITIVE_WORDS if word in text)
        negative = sum(1 for word in NEGATIVE_WORDS if word in text)
        high_arousal = sum(1 for word in HIGH_AROUSAL_WORDS if word in text)
        valence = self._clamp((positive - negative) * 0.12, low=-0.35, high=0.35)
        arousal = min(0.3, high_arousal * 0.12 + (positive + negative) * 0.03)
        dominance = self._clamp((positive - negative) * 0.04, low=-0.12, high=0.12)
        understanding = event.payload.get("understanding")
        if isinstance(understanding, dict):
            cues = understanding.get("affect_cues")
            if isinstance(cues, dict):
                valence += self._numeric(cues.get("valence")) * 0.15
                arousal += abs(self._numeric(cues.get("arousal"))) * 0.12
        return valence, arousal, dominance, "linguistic_affect" if text else "neutral_event"

    def _event_text(self, event: AgentEvent) -> str:
        parts = [str(event.payload.get("content") or "")]
        understanding = event.payload.get("understanding")
        if isinstance(understanding, dict):
            parts.append(str(understanding.get("semantic_summary") or ""))
            parts.append(str(understanding.get("affect_cues") or ""))
        return " ".join(parts)

    def _primary(self, emotion: JsonDict) -> str:
        valence = float(emotion.get("valence", 0.0))
        arousal = float(emotion.get("arousal", 0.15))
        dominance = float(emotion.get("dominance", 0.0))
        if abs(valence) < 0.12 and arousal < 0.35:
            return "neutral"
        if valence >= 0.12:
            return "confident" if dominance > 0.2 else "joy"
        if arousal > 0.55:
            return "anger" if dominance > 0 else "anxiety"
        return "sadness"

    def _mood(self, emotion: JsonDict) -> str:
        valence = float(emotion.get("valence", 0.0))
        arousal = float(emotion.get("arousal", 0.15))
        if valence > 0.25:
            return "positive_engaged" if arousal > 0.4 else "content"
        if valence < -0.25:
            return "tense" if arousal > 0.4 else "low"
        return "alert" if arousal > 0.45 else "calm"

    def _update_social_target(
        self,
        emotion: JsonDict,
        event: AgentEvent,
        delta_valence: float,
    ) -> None:
        speaker = event.payload.get("sender") or event.payload.get("speaker_id")
        if not speaker or abs(delta_valence) < 0.001:
            return
        toward = emotion.setdefault("toward", {})
        if not isinstance(toward, dict):
            toward = {}
            emotion["toward"] = toward
        current = toward.get(str(speaker))
        if not isinstance(current, dict):
            current = {"affinity": 0.0, "trust": 0.0}
        current["affinity"] = self._clamp(
            float(current.get("affinity", 0.0)) + delta_valence * 0.2
        )
        current["trust"] = self._clamp(
            float(current.get("trust", 0.0)) + delta_valence * 0.1
        )
        current["updated_at"] = utc_now()
        toward[str(speaker)] = current

    def _numeric(self, value: Any) -> float:
        try:
            return self._clamp(float(value))
        except (TypeError, ValueError):
            return 0.0

    def _parse_datetime(self, value: Any) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _clamp(self, value: float, *, low: float = -1.0, high: float = 1.0) -> float:
        return round(max(low, min(high, value)), 4)
