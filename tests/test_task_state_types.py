from __future__ import annotations

import unittest

from agent.config import AgentConfig
from agent.protocols import (
    ActionRun,
    ActionSpec,
    AgentEvent,
    AgentState,
    AgentTask,
    ConversationState,
    ConversationTurn,
    ConversationUnderstanding,
    Workspace,
)
from agent.runtime.action_systems.task_system import TaskSystem
from agent.runtime.cognition_system.decision_system.decision import DecisionSystem
from agent.runtime.interfaces.model import ModelResult
from agent.runtime.persistence_system.memory_store import MemoryRecord


class TaskStructuredStateTests(unittest.TestCase):
    def test_all_persisted_mapping_fields_normalize_scalars(self) -> None:
        event = AgentEvent.make(
            agent_id="mapping-test",
            type="test",
            source="test",
            payload="legacy payload",  # type: ignore[arg-type]
        )
        workspace = Workspace(
            workspace_id="workspace-test",
            variables="legacy variables",  # type: ignore[arg-type]
            transcript="legacy transcript",  # type: ignore[arg-type]
        )
        spec = ActionSpec(
            name="legacy",
            description="legacy schema",
            input_schema="legacy schema",  # type: ignore[arg-type]
            metadata="legacy metadata",  # type: ignore[arg-type]
        )
        understanding = ConversationUnderstanding(
            understanding_id="understanding-test",
            turn_id="turn-test",
            speaker_id="speaker-test",
            semantic_summary="test",
            key_information="legacy information",  # type: ignore[arg-type]
            entities="legacy entity",  # type: ignore[arg-type]
            affect_cues="legacy cues",  # type: ignore[arg-type]
        )
        turn = ConversationTurn(
            turn_id="turn-test",
            conversation_id="conversation-test",
            agent_id="mapping-test",
            speaker_id="speaker-test",
            recipient_id="mapping-test",
            channel="test",
            source_event_id="event-test",
            utterance="hello",
            speaker_context="legacy speaker",  # type: ignore[arg-type]
            scene_context="legacy scene",  # type: ignore[arg-type]
            understanding="legacy understanding",  # type: ignore[arg-type]
            decision="legacy decision",  # type: ignore[arg-type]
            speech_intent="legacy intent",  # type: ignore[arg-type]
            outbound_utterances="legacy outbound",  # type: ignore[arg-type]
            suppressed_speech_intents="legacy suppressed",  # type: ignore[arg-type]
        )
        memory = MemoryRecord(
            memory_id="memory-test",
            agent_id="mapping-test",
            kind="test",
            title="test",
            content="test",
            source_refs="legacy ref",  # type: ignore[arg-type]
        )
        model_result = ModelResult(
            text="test",
            usage="legacy usage",  # type: ignore[arg-type]
        )

        for value in (
            event.payload,
            workspace.variables,
            spec.input_schema,
            spec.metadata,
            understanding.affect_cues,
            turn.speaker_context,
            turn.scene_context,
            turn.understanding,
            turn.decision,
            turn.speech_intent,
            model_result.usage,
        ):
            self.assertIsInstance(value, dict)
        for values in (
            workspace.transcript,
            understanding.key_information,
            understanding.entities,
            turn.outbound_utterances,
            turn.suppressed_speech_intents,
            memory.source_refs,
        ):
            self.assertTrue(all(isinstance(value, dict) for value in values))

    def test_corrupt_top_level_state_containers_are_safely_migrated(self) -> None:
        state = AgentState.from_dict(
            {
                "agent_id": "legacy-state",
                "profile": "legacy profile",
                "workspace": "legacy workspace",
                "tasks": "legacy tasks",
                "action_runs": "legacy runs",
                "processed_event_ids": "legacy event",
            }
        )
        conversations = ConversationState.from_dict(
            {
                "agent_id": "legacy-state",
                "sessions": "legacy sessions",
                "turns": "legacy turns",
            }
        )

        self.assertIsInstance(state.workspace.variables, dict)
        self.assertEqual(state.tasks, {})
        self.assertEqual(state.action_runs, {})
        self.assertEqual(state.processed_event_ids, ["legacy event"])
        self.assertEqual(conversations.sessions, {})
        self.assertEqual(conversations.turns, {})

    def test_model_command_objects_are_normalized_before_runtime(self) -> None:
        decision = DecisionSystem._normalize(  # type: ignore[arg-type]
            None,
            {
                "commands": [
                    {
                        "type": "wait",
                        "condition": "legacy condition",
                        "args": "legacy args",
                    },
                    "legacy command",
                ]
            },
        )
        config = AgentConfig.from_dict(
            {
                "agent": "legacy agent",
                "generator": "legacy generator",
                "entrypoints": "legacy entrypoints",
                "cognition": "legacy cognition",
                "state": "legacy state",
            },
            source="test",
        )

        self.assertEqual(len(decision["commands"]), 1)
        self.assertEqual(
            decision["commands"][0]["condition"],
            {"value": "legacy condition"},
        )
        self.assertEqual(
            decision["commands"][0]["args"],
            {"value": "legacy args"},
        )
        self.assertEqual(config.source, "test")

    def test_update_task_normalizes_structured_scalars_to_objects(self) -> None:
        state = AgentState.new("structured-update-test")
        system = TaskSystem()
        task = system.create_task(
            state,
            title="Keep structured state valid",
            goal="Never persist scalar task objects",
            purpose="regression test",
        )

        update = system.update_task(
            state,
            task.task_id,
            {
                "progress": "working",
                "result": "done",
                "error": "temporary failure",
                "continuation": "resume later",
            },
        )

        self.assertEqual(update["rejected"], {})
        self.assertEqual(task.progress, {"message": "working"})
        self.assertEqual(task.result, {"value": "done"})
        self.assertEqual(task.error, {"message": "temporary failure"})
        self.assertEqual(task.continuation, {"value": "resume later"})

    def test_legacy_task_and_action_run_scalars_are_repaired_on_load(self) -> None:
        task = AgentTask.from_dict(
            {
                "task_id": "task-legacy",
                "agent_id": "legacy-test",
                "title": "Legacy task",
                "goal": "Load safely",
                "purpose": "migration test",
                "scheduling": "legacy scheduling",
                "progress": "legacy progress",
                "result": "legacy result",
                "error": "legacy error",
                "continuation": "legacy continuation",
            }
        )
        run = ActionRun.from_dict(
            {
                "action_run_id": "run-legacy",
                "agent_id": "legacy-test",
                "task_id": task.task_id,
                "action_name": "legacy_action",
                "args": "legacy args",
                "mode": "sync",
                "progress": "legacy run progress",
                "result": "legacy run result",
                "error": "legacy run error",
            }
        )

        for value in (
            task.scheduling,
            task.progress,
            task.result,
            task.error,
            task.continuation,
            run.args,
            run.progress,
            run.result,
            run.error,
        ):
            self.assertIsInstance(value, dict)

    def test_action_completion_repairs_directly_corrupted_progress(self) -> None:
        state = AgentState.new("structured-event-test")
        system = TaskSystem()
        task = system.create_task(
            state,
            title="Handle completion",
            goal="Do not crash on legacy state",
            purpose="ordinary task",
        )
        run = ActionRun(
            action_run_id="run-corrupted",
            agent_id=state.agent_id,
            task_id=task.task_id,
            action_name="external_action",
            args={},
            mode="async",
            status="running",
        )
        state.action_runs[run.action_run_id] = run
        task.active_action_runs.append(run.action_run_id)
        task.progress = "working"  # type: ignore[assignment]

        system.apply_event(
            state,
            AgentEvent.make(
                agent_id=state.agent_id,
                type="action.completed",
                source="test",
                task_id=task.task_id,
                action_run_id=run.action_run_id,
                payload={"result": "done"},
            ),
        )

        self.assertIsInstance(task.progress, dict)
        self.assertEqual(task.result, {})
        self.assertEqual(run.result, {"value": "done"})

    def test_direct_task_commands_repair_legacy_structured_state(self) -> None:
        state = AgentState.new("structured-command-test")
        system = TaskSystem()
        task = system.create_task(
            state,
            title="Repair command input",
            goal="Keep command paths safe",
            purpose="regression test",
        )
        task.scheduling = "legacy scheduling"  # type: ignore[assignment]
        task.progress = "legacy progress"  # type: ignore[assignment]

        system.update_task(state, task.task_id, {"status": "blocked"})
        completion = system.complete_task(
            state,
            task.task_id,
            result="legacy result",  # type: ignore[arg-type]
        )

        self.assertIsInstance(task.scheduling, dict)
        self.assertIsInstance(task.progress, dict)
        self.assertEqual(task.result, {"value": "legacy result"})
        self.assertTrue(completion["completed"])


if __name__ == "__main__":
    unittest.main()
