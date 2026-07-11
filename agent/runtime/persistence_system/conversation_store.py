from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

from ...protocols import (
    ConversationSession,
    ConversationState,
    ConversationTurn,
    JsonDict,
    ensure_json_dict,
    new_id,
    utc_now,
)


class ConversationStore:
    """Durable conversation sessions and turn lifecycle records."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / "conversations"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, agent_id: str) -> Path:
        return self.root / f"{agent_id}.conversation.json"

    def load_state(self, agent_id: str) -> ConversationState:
        path = self._path(agent_id)
        if not path.exists():
            return ConversationState(agent_id=agent_id)
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return (
            ConversationState.from_dict(data)
            if isinstance(data, dict)
            else ConversationState(agent_id=agent_id)
        )

    def save_state(self, state: ConversationState) -> None:
        state.updated_at = utc_now()
        path = self._path(state.agent_id)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(state.to_dict(), file, ensure_ascii=False, indent=2)
        os.replace(temporary, path)

    def create_turn(
        self,
        *,
        agent_id: str,
        conversation_id: str,
        speaker_id: str,
        recipient_id: str,
        channel: str,
        source_event_id: str,
        utterance: str,
        speaker_context: JsonDict | None = None,
        scene_context: JsonDict | None = None,
    ) -> ConversationTurn:
        state = self.load_state(agent_id)
        for existing in state.turns.values():
            if existing.source_event_id == source_event_id:
                return existing
        session = state.sessions.get(conversation_id)
        if session is None:
            session = ConversationSession(
                conversation_id=conversation_id,
                agent_id=agent_id,
                participant_ids=[speaker_id, recipient_id],
            )
            state.sessions[conversation_id] = session
        else:
            for participant_id in (speaker_id, recipient_id):
                if participant_id not in session.participant_ids:
                    session.participant_ids.append(participant_id)

        turn = ConversationTurn(
            turn_id=new_id("turn"),
            conversation_id=conversation_id,
            agent_id=agent_id,
            speaker_id=speaker_id,
            recipient_id=recipient_id,
            channel=channel,
            source_event_id=source_event_id,
            utterance=utterance,
            speaker_context=ensure_json_dict(speaker_context),
            scene_context=ensure_json_dict(scene_context),
        )
        state.turns[turn.turn_id] = turn
        session.turn_ids.append(turn.turn_id)
        session.touch()
        self.save_state(state)
        return turn

    def get_turn(self, agent_id: str, turn_id: str) -> ConversationTurn:
        state = self.load_state(agent_id)
        if turn_id not in state.turns:
            raise KeyError(f"Unknown conversation turn: {turn_id}")
        return state.turns[turn_id]

    def latest_turn(
        self,
        *,
        agent_id: str,
        conversation_id: str,
    ) -> ConversationTurn | None:
        state = self.load_state(agent_id)
        session = state.sessions.get(conversation_id)
        if session is None:
            return None
        for turn_id in reversed(session.turn_ids):
            turn = state.turns.get(turn_id)
            if turn is not None:
                return turn
        return None

    def save_turn(self, turn: ConversationTurn) -> None:
        state = self.load_state(turn.agent_id)
        turn.touch()
        state.turns[turn.turn_id] = turn
        self.save_state(state)

    def recent_turns(
        self,
        *,
        agent_id: str,
        conversation_id: str,
        limit: int,
        before_turn_id: str | None = None,
    ) -> List[JsonDict]:
        state = self.load_state(agent_id)
        session = state.sessions.get(conversation_id)
        if session is None:
            return []
        turn_ids = list(session.turn_ids)
        if before_turn_id in turn_ids:
            turn_ids = turn_ids[: turn_ids.index(before_turn_id)]
        selected = turn_ids[-max(1, limit) :]
        return [
            state.turns[turn_id].to_dict()
            for turn_id in selected
            if turn_id in state.turns
        ]

    def context_turns(
        self,
        *,
        agent_id: str,
        conversation_id: str,
        limit: int,
        verbatim_limit: int,
        compact_limit: int,
        before_turn_id: str | None = None,
    ) -> List[JsonDict]:
        turns = self.recent_turns(
            agent_id=agent_id,
            conversation_id=conversation_id,
            limit=limit,
            before_turn_id=before_turn_id,
        )
        total = len(turns)
        verbatim_start = max(0, total - max(1, verbatim_limit))
        compact_start = max(0, total - max(verbatim_limit, compact_limit))
        context_turns: List[JsonDict] = []
        for index, turn in enumerate(turns):
            if index >= verbatim_start:
                context_turns.append(self._verbatim_context_turn(turn))
            elif index >= compact_start:
                context_turns.append(self._compact_context_turn(turn))
            else:
                context_turns.append(self._summary_context_turn(turn))
        return context_turns

    def search(
        self,
        *,
        agent_id: str,
        conversation_id: str,
        query: str,
        limit: int,
    ) -> List[JsonDict]:
        state = self.load_state(agent_id)
        session = state.sessions.get(conversation_id)
        if session is None:
            return []
        terms = [term for term in query.casefold().split() if term]
        matches: List[JsonDict] = []
        for turn_id in reversed(session.turn_ids):
            turn = state.turns.get(turn_id)
            if turn is None:
                continue
            searchable = json.dumps(turn.to_dict(), ensure_ascii=False).casefold()
            if terms and not all(term in searchable for term in terms):
                continue
            matches.append(turn.to_dict())
            if len(matches) >= max(1, limit):
                break
        return matches

    def _verbatim_context_turn(self, turn: JsonDict) -> JsonDict:
        return {
            "history_tier": "verbatim",
            "turn_id": turn.get("turn_id"),
            "speaker_id": turn.get("speaker_id"),
            "utterance": turn.get("utterance"),
            "response_text": turn.get("response_text"),
            "status": turn.get("status"),
            "created_at": turn.get("created_at"),
        }

    def _compact_context_turn(self, turn: JsonDict) -> JsonDict:
        understanding = turn.get("understanding")
        if not isinstance(understanding, dict):
            understanding = {}
        speech_intent = turn.get("speech_intent")
        if not isinstance(speech_intent, dict):
            speech_intent = {}
        response_intent = speech_intent.get("content")
        if (
            speech_intent.get("kind") == "direct_conversation_response"
            and turn.get("response_text")
        ):
            response_intent = turn.get("response_text")
        return {
            "history_tier": "compact",
            "turn_id": turn.get("turn_id"),
            "speaker_id": turn.get("speaker_id"),
            "utterance_summary": (
                understanding.get("semantic_summary") or turn.get("utterance")
            ),
            "speech_act": understanding.get("speech_act"),
            "intents": understanding.get("intents") or [],
            "agent_response_intent": (
                response_intent or turn.get("response_text")
            ),
            "status": turn.get("status"),
            "created_at": turn.get("created_at"),
        }

    def _summary_context_turn(self, turn: JsonDict) -> JsonDict:
        compact = self._compact_context_turn(turn)
        return {
            "history_tier": "summary",
            "turn_id": compact.get("turn_id"),
            "exchange_summary": {
                "speaker_id": compact.get("speaker_id"),
                "meaning": compact.get("utterance_summary"),
                "speech_act": compact.get("speech_act"),
                "agent_response_intent": compact.get("agent_response_intent"),
                "status": compact.get("status"),
            },
            "created_at": compact.get("created_at"),
        }
