from __future__ import annotations

from agent.protocols import AgentProfile

from .messages import MessagePurpose, NaturalMessage
from .settings import PromptSettings


class PromptCompiler:
    """Build short, purpose-specific prompts from one shared policy."""

    def __init__(self, settings: PromptSettings) -> None:
        self.settings = settings

    def system_prompt(
        self,
        *,
        profile: AgentProfile,
        message: NaturalMessage,
        tools_available: bool,
    ) -> str:
        identity = self._profile_block(profile, message.purpose)
        tool_note = (
            "本轮提供了可调用能力；需要行动时实际调用，不用文字代替，结果出现前不声称完成。"
            if tools_available
            else "本轮不提供能力调用；只能基于已有信息回复。"
        )
        return "\n\n".join(
            part
            for part in (
                f"你是 {profile.name}。",
                identity,
                f"当前职责：{self.settings.description_for(message.purpose)}",
                self.settings.common_rules.strip(),
                tool_note,
            )
            if part
        )

    def _profile_block(
        self,
        profile: AgentProfile,
        purpose: MessagePurpose,
    ) -> str:
        fields = [
            ("身份", profile.identity_profile),
            ("边界", profile.self_boundaries),
        ]
        if purpose == MessagePurpose.UNDERSTANDING:
            fields.extend(
                [
                    ("性格", profile.persona_profile),
                    ("关系", profile.relationship_profile),
                ]
            )
        elif purpose == MessagePurpose.EXPRESSION:
            fields.extend(
                [
                    ("性格", profile.persona_profile),
                    ("声色", profile.voice_profile),
                    ("表达", profile.speech_profile),
                    ("关系", profile.relationship_profile),
                ]
            )
        elif purpose == MessagePurpose.DECISION:
            fields.extend(
                [
                    ("价值", profile.values_profile),
                    ("行为", profile.behavior_profile),
                ]
            )
        else:
            fields.extend(
                [
                    ("背景", profile.background_profile),
                    ("价值", profile.values_profile),
                ]
            )
        return "\n".join(f"{label}：{value}" for label, value in fields if value)
