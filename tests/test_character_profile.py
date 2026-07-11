from __future__ import annotations

import unittest

from agent.protocols import AgentEvent, AgentProfile, AgentState
from agent.runtime.action_systems.actions import ActionRegistry
from agent.runtime.kernel.generator_runtime import GeneratorRuntime
from agent.runtime.kernel.runtime import AgentRuntime
from agent.runtime.state_systems import ContextBuilder
from agent_ling.config import load_agent_config


SELF_MODEL_FIELDS = (
    "identity_profile",
    "background_profile",
    "persona_profile",
    "values_profile",
    "behavior_profile",
    "voice_profile",
    "speech_profile",
    "relationship_profile",
    "self_boundaries",
)


class CharacterProfileTests(unittest.TestCase):
    def test_default_ling_profile_has_a_complete_self_model(self) -> None:
        config = load_agent_config()
        profile = config.profile.to_agent_profile("ling-test")

        self.assertEqual(profile.name, "Ling")
        for field_name in SELF_MODEL_FIELDS:
            with self.subTest(field=field_name):
                self.assertTrue(getattr(profile, field_name).strip())

        self.assertIn("数字", profile.identity_profile)
        self.assertIn("不自动证明当前外部世界", profile.background_profile)
        self.assertIn("身体感受", profile.self_boundaries)

    def test_old_persisted_profile_remains_compatible(self) -> None:
        profile = AgentProfile.from_dict(
            {
                "agent_id": "legacy-agent",
                "name": "Legacy",
                "persona_profile": "旧角色",
            }
        )

        self.assertEqual(profile.name, "Legacy")
        self.assertEqual(profile.persona_profile, "旧角色")
        self.assertEqual(profile.identity_profile, "")
        self.assertEqual(profile.voice_profile, "")

    def test_configured_self_model_refreshes_an_existing_state(self) -> None:
        config = load_agent_config()
        state = AgentState.new("ling-test")
        state.version = 12
        state.profile.name = "Agent"
        state.profile.persona_profile = "早期的简化人格"
        runtime = AgentRuntime.__new__(AgentRuntime)
        runtime.agent_config = config

        runtime._apply_configured_profile(state)

        self.assertEqual(state.profile.name, "Ling")
        self.assertEqual(
            state.profile.persona_profile,
            config.profile.persona_profile,
        )
        self.assertEqual(
            state.profile.speech_profile,
            config.profile.speech_profile,
        )

    def test_every_cognitive_session_receives_the_complete_self_model(self) -> None:
        config = load_agent_config()
        profile = config.profile.to_agent_profile("ling-test")

        for session_id in (
            "decision",
            "wernicke",
            "broca",
            "context_builder",
            "memory_reflection",
            "dmn",
        ):
            with self.subTest(session=session_id):
                prompt = config.generator.prompt_for(session_id)
                rendered = prompt.profile_context_template.format_map(profile.to_dict())
                self.assertIn(profile.identity_profile, rendered)
                self.assertIn(profile.background_profile, rendered)
                self.assertIn(profile.voice_profile, rendered)
                self.assertIn(profile.speech_profile, rendered)
                self.assertNotIn("{identity_profile}", rendered)

    def test_decision_context_carries_the_complete_profile(self) -> None:
        config = load_agent_config()
        state = AgentState.new("ling-test")
        state.profile = config.profile.to_agent_profile(state.agent_id)
        context = ContextBuilder(
            config.generator.prompt_for("decision").context_policy
        ).build(
            state=state,
            event=AgentEvent.make(
                agent_id=state.agent_id,
                type="user.message",
                source="test",
                payload={"content": "你还记得自己是谁吗？"},
            ),
            action_specs=ActionRegistry().list_specs(),
        )

        for field_name in SELF_MODEL_FIELDS:
            with self.subTest(field=field_name):
                self.assertEqual(
                    context["agent"][field_name],
                    getattr(state.profile, field_name),
                )

    def test_self_model_is_in_system_prompt_without_user_payload_duplication(self) -> None:
        config = load_agent_config()
        profile = config.profile.to_agent_profile("ling-test")
        runtime = GeneratorRuntime.__new__(GeneratorRuntime)
        runtime.agent_config = config
        context = {
            "agent": profile.to_dict(),
            "decision": {},
            "runtime": {},
            "workspace": {},
            "tooling": {},
            "focus": {},
            "cognition": {},
            "long_term_memory": {},
            "context_selection": {},
        }

        system_prompt = runtime._build_system_prompt(context, session_id="dmn")
        user_payload = runtime._build_user_payload(context, session_id="dmn")
        decision_payload = runtime._decision_context_markdown(context)

        self.assertIn(profile.background_profile, system_prompt)
        self.assertNotIn(profile.background_profile, user_payload)
        self.assertNotIn(profile.background_profile, decision_payload)


if __name__ == "__main__":
    unittest.main()
