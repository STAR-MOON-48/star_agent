from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime
import threading
from typing import Any, Callable, Optional
from uuid import uuid4

from rich.markup import escape
from star_protocol.client import EnvironmentClient, HumanClient
from star_protocol.models.payloads import ToolDefinition
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, RichLog, Static


NoticeCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class HumanProfile:
    human_id: str
    display_name: str
    role: str
    background: str
    relationship_to_agent: str
    login_session: str

    def to_context(self) -> dict[str, Any]:
        return {
            "human_id": self.human_id,
            "display_name": self.display_name,
            "role": self.role,
            "background": self.background,
            "relationship_to_agent": self.relationship_to_agent,
            "authenticated": True,
            "authentication_source": "console_ui_scenario",
            "login_session": self.login_session,
        }


@dataclass(frozen=True)
class SceneConfig:
    env_id: str
    title: str
    background: str

    def to_context(self, participants: set[str]) -> dict[str, Any]:
        return {
            "environment_id": self.env_id,
            "title": self.title,
            "background": self.background,
            "participants": sorted(participants),
            "scene_type": "interactive_npc_dialogue_experiment",
        }


class ConsoleEnvironmentClient(EnvironmentClient):
    """Star Environment hosting the interactive NPC dialogue scene."""

    def __init__(
        self,
        *,
        scene: SceneConfig,
        human_profile: HumanProfile,
        notice: NoticeCallback,
        auto_reconnect: bool,
        monitorable: bool,
    ) -> None:
        super().__init__(
            client_id=scene.env_id,
            auto_reconnect=auto_reconnect,
            monitorable=monitorable,
        )
        self.scene = scene
        self.human_profile = human_profile
        self.notice = notice
        self.participants: set[str] = set()

    async def on_discover(self, sender: str) -> list[ToolDefinition]:
        if sender not in self.participants:
            self.participants.add(sender)
            self.notice("participant_joined", {"client_id": sender})
        tools = self._tools()
        self.notice(
            "environment_discover",
            {"sender": sender, "tool_names": [tool["name"] for tool in tools]},
        )
        return tools

    async def on_action(self, sender: str, content: dict[str, Any]) -> None:
        action_name = str(content.get("name") or "unknown")
        action_id = str(content.get("id") or "")
        params = dict(content.get("params") or {})
        self.notice(
            "environment_action",
            {
                "sender": sender,
                "action_name": action_name,
                "action_id": action_id,
                "params": params,
            },
        )

        success = True
        if action_name == "observe_social_scene":
            data = {
                "scene": self.scene.to_context(self.participants),
                "human_public_profile": self.human_profile.to_context(),
                "observation": (
                    f"{self.scene.title}。{self.scene.background}"
                    f" 当前参与者：{', '.join(sorted(self.participants)) or '无'}。"
                ),
            }
        elif action_name == "read_human_profile":
            requested_id = str(params.get("human_id") or self.human_profile.human_id)
            if requested_id != self.human_profile.human_id:
                success = False
                data = {"error": f"场景中没有公开资料属于 {requested_id}"}
            else:
                data = {
                    "profile": self.human_profile.to_context(),
                    "source": "scenario_login_context",
                    "verified_by_environment": True,
                }
        elif action_name == "perform_social_action":
            data = {
                "actor": sender,
                "target": params.get("target") or self.human_profile.human_id,
                "action": params.get("action"),
                "description": params.get("description"),
                "scene": self.scene.env_id,
                "performed": True,
            }
            await self.broadcast_to_env(
                self.scene.env_id,
                "social_action_performed",
                data,
            )
        else:
            success = False
            data = {"error": f"Unknown dialogue-scene action: {action_name}"}

        outcome = {
            "ref_id": action_id,
            "success": success,
            "data": data if success else None,
            "error": None if success else str(data.get("error") or data),
        }
        await self.send_outcome(sender, outcome)
        self.notice(
            "environment_outcome",
            {
                "recipient": sender,
                "action_name": action_name,
                "success": success,
                "data": data,
            },
        )

    async def on_event(self, sender: str, content: dict[str, Any]) -> None:
        self.notice("environment_event", {"sender": sender, "content": content})

    async def on_system_notify(self, event: str, content: dict[str, Any]) -> None:
        await super().on_system_notify(event, content)
        self.notice("environment_system", {"event": event, "content": content})

    async def on_client_joined(self, client_id: str) -> None:
        self.participants.add(client_id)
        self.notice("participant_joined", {"client_id": client_id})

    async def on_client_left(self, client_id: str, reason: str = "leave") -> None:
        self.participants.discard(client_id)
        self.notice(
            "participant_left",
            {"client_id": client_id, "reason": reason},
        )

    async def on_error(self, error: dict[str, Any]) -> None:
        self.notice("error", {"client": "environment", "error": error})

    def _tools(self) -> list[ToolDefinition]:
        return [
            {
                "name": "observe_social_scene",
                "description": "观察当前 NPC 对话实验场景、参与者和可公开的人类身份背景。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "focus": {
                            "type": "string",
                            "description": "希望重点观察的内容，可留空。",
                        }
                    },
                    "additionalProperties": False,
                },
                "tags": ["scene", "social", "observe"],
            },
            {
                "name": "read_human_profile",
                "description": "读取当前场景中已登录 Human 的公开身份、背景和与 Agent 的关系设定。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "human_id": {
                            "type": "string",
                            "description": "Human client id；留空表示当前登录 Human。",
                        }
                    },
                    "additionalProperties": False,
                },
                "tags": ["human", "profile", "social"],
            },
            {
                "name": "perform_social_action",
                "description": "在场景中执行可观察的非语言社会动作，例如点头、挥手、递出物品或转身。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "description": "社会动作名称。"},
                        "target": {"type": "string", "description": "动作对象 client id。"},
                        "description": {"type": "string", "description": "动作的自然语言细节。"},
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
                "tags": ["social", "action", "nonverbal"],
            },
        ]


class ConsoleHumanClient(HumanClient):
    """Logged-in Human identity used by the console input box."""

    def __init__(
        self,
        *,
        profile: HumanProfile,
        notice: NoticeCallback,
        auto_reconnect: bool,
        monitorable: bool,
    ) -> None:
        super().__init__(
            client_id=profile.human_id,
            auto_reconnect=auto_reconnect,
            monitorable=monitorable,
        )
        self.profile = profile
        self.notice = notice

    async def on_event(self, sender: str, content: dict[str, Any]) -> None:
        name = str(content.get("name") or "")
        data = content.get("data")
        event_data = dict(data) if isinstance(data, dict) else {}
        if name == "assistant.message":
            self.notice(
                "assistant_message",
                {
                    "sender": sender,
                    "content": str(event_data.get("content") or ""),
                    "conversation_id": event_data.get("conversation_id"),
                    "turn_id": event_data.get("turn_id"),
                },
            )
            return
        self.notice(
            "human_event",
            {"sender": sender, "name": name, "data": event_data},
        )

    async def on_outcome(self, sender: str, content: dict[str, Any]) -> None:
        self.notice("human_outcome", {"sender": sender, "content": content})

    async def on_stream(self, sender: str, content: dict[str, Any]) -> None:
        self.notice("human_stream", {"sender": sender, "content": content})

    async def on_broadcast_event(self, sender: str, content: dict[str, Any]) -> None:
        self.notice("human_broadcast", {"sender": sender, "content": content})

    async def on_system_notify(self, event: str, content: dict[str, Any]) -> None:
        self.notice("human_system", {"event": event, "content": content})

    async def on_error(self, error: dict[str, Any]) -> None:
        self.notice("error", {"client": "human", "error": error})


class StarNotice(Message):
    def __init__(self, kind: str, payload: dict[str, Any]) -> None:
        super().__init__()
        self.kind = kind
        self.payload = payload


class DialogueConsoleApp(App[None]):
    """Star Protocol Environment + Human client for interactive NPC dialogue."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #connection-status {
        height: 3;
        padding: 0 1;
        background: $surface;
        color: $text;
        border-bottom: solid $primary;
    }

    #main {
        height: 1fr;
    }

    #scene-pane {
        width: 31%;
        min-width: 34;
        border-right: solid $primary;
    }

    #chat-pane {
        width: 1fr;
        min-width: 44;
    }

    #protocol-pane {
        width: 34%;
        min-width: 36;
        border-left: solid $secondary;
    }

    .pane-title {
        height: 3;
        padding: 1;
        text-style: bold;
        background: $panel;
    }

    #scene-info, #human-profile {
        height: auto;
        max-height: 14;
        padding: 1;
        border-bottom: solid $primary-background-lighten-2;
    }

    #participants {
        height: 1fr;
    }

    #conversation {
        height: 1fr;
        padding: 0 1;
    }

    #agent-activity {
        height: 3;
        padding: 0 1;
        background: $surface;
        border-top: solid $secondary;
    }

    #protocol-log {
        height: 1fr;
        padding: 0 1;
    }

    #composer {
        height: 5;
        padding: 1;
        border-top: solid $accent;
    }

    #human-input {
        width: 1fr;
    }

    #send-button {
        width: 10;
        margin-left: 1;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "退出"),
        ("ctrl+l", "clear_logs", "清空显示"),
        ("ctrl+r", "reconnect", "重连"),
    ]

    def __init__(
        self,
        *,
        hub_url: str,
        scene: SceneConfig,
        human_profile: HumanProfile,
        agent_id: str,
        conversation_id: str,
        auto_reconnect: bool,
        monitorable: bool,
        initial_message: Optional[str],
    ) -> None:
        super().__init__()
        self.hub_url = hub_url
        self.scene = scene
        self.human_profile = human_profile
        self.agent_id = agent_id
        self.conversation_id = conversation_id
        self.auto_reconnect = auto_reconnect
        self.monitorable = monitorable
        self._pending_initial_message = initial_message
        self.environment: Optional[ConsoleEnvironmentClient] = None
        self.human: Optional[ConsoleHumanClient] = None
        self.participants: set[str] = set()
        self.connection_state = "未连接"
        self.agent_activity = "等待 Agent 加入场景"
        self._stopping = False
        self._message_sequence = 0
        self._connection_lifetime: Optional[asyncio.Event] = None
        self._star_loop: Optional[asyncio.AbstractEventLoop] = None
        self._ui_thread_id: Optional[int] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("正在连接 Star Hub...", id="connection-status")
        with Horizontal(id="main"):
            with Vertical(id="scene-pane"):
                yield Label("场景与身份", classes="pane-title")
                yield Static(id="scene-info")
                yield Static(id="human-profile")
                yield DataTable(id="participants")
            with Vertical(id="chat-pane"):
                yield Label("Human ↔ NPC Agent", classes="pane-title")
                yield RichLog(id="conversation", wrap=True, markup=True, auto_scroll=True)
                yield Static("等待 Agent 加入场景", id="agent-activity")
            with Vertical(id="protocol-pane"):
                yield Label("Star Protocol", classes="pane-title")
                yield RichLog(id="protocol-log", wrap=True, markup=True, auto_scroll=True)
        with Horizontal(id="composer"):
            yield Input(
                placeholder="以已登录 Human 身份对 Agent 说话，回车发送",
                id="human-input",
                disabled=True,
            )
            yield Button("发送", id="send-button", variant="primary", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self._ui_thread_id = threading.get_ident()
        self.title = self.scene.title
        self.sub_title = f"Human {self.human_profile.display_name} → Agent {self.agent_id}"
        participants = self.query_one("#participants", DataTable)
        participants.cursor_type = "row"
        participants.add_columns("Client", "Role", "State")
        self._render_scene()
        self._render_status()
        self._protocol_line(
            "启动顺序：本 UI 创建 Environment/Human；Agent 需以独立 agent client 加入同一 env。"
        )
        self._protocol_line(
            "Agent 命令："
            f"uv run agent-ling-star --agent-id {self.agent_id} "
            f"--hub-url {self.hub_url} --env-id {self.scene.env_id} --no-startup-objective"
        )
        self.run_worker(
            self._run_star_clients,
            name="star-connect",
            group="star",
            exclusive=True,
            exit_on_error=False,
            thread=True,
        )

    async def action_quit(self) -> None:
        await self._stop_clients()
        self.exit()

    async def action_reconnect(self) -> None:
        await self._stop_clients()
        self._stopping = False
        self.run_worker(
            self._run_star_clients,
            name="star-reconnect",
            group="star",
            exclusive=True,
            exit_on_error=False,
            thread=True,
        )

    def action_clear_logs(self) -> None:
        self.query_one("#conversation", RichLog).clear()
        self.query_one("#protocol-log", RichLog).clear()

    async def on_unmount(self) -> None:
        await self._stop_clients()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        await self._submit_human_text(event.value)
        event.input.value = ""

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "send-button":
            return
        input_widget = self.query_one("#human-input", Input)
        await self._submit_human_text(input_widget.value)
        input_widget.value = ""
        input_widget.focus()

    async def on_star_notice(self, message: StarNotice) -> None:
        kind = message.kind
        payload = message.payload
        if kind == "connecting":
            self.connection_state = "正在连接 Hub"
            self._render_status()
            return
        if kind == "connected":
            self.connection_state = "Environment 与 Human 已连接"
            self._set_composer_enabled(self.agent_id in self.participants)
            self._protocol_line(
                f"Environment={self.scene.env_id} Human={self.human_profile.human_id} 已连接 Hub。",
                style="green",
            )
            if self.agent_id not in self.participants:
                self._protocol_line(
                    f"等待 Agent={self.agent_id} 加入后再开放 Human 输入。",
                    style="yellow",
                )
            await self._send_initial_message_if_ready()
            self._render_status()
            return
        if kind == "participant_joined":
            client_id = str(payload.get("client_id") or "unknown")
            self.participants.add(client_id)
            if client_id == self.agent_id:
                self.agent_activity = "Agent 在线，可以开始交谈"
                self._set_composer_enabled(self.human is not None)
            self._render_participants()
            self._render_status()
            self._protocol_line(f"client_joined: {client_id}", style="green")
            await self._send_initial_message_if_ready()
            return
        if kind == "participant_left":
            client_id = str(payload.get("client_id") or "unknown")
            self.participants.discard(client_id)
            if client_id == self.agent_id:
                self.agent_activity = "Agent 已离开场景"
                self._set_composer_enabled(False)
            self._render_participants()
            self._render_status()
            self._protocol_line(
                f"client_left: {client_id} reason={payload.get('reason')}",
                style="yellow",
            )
            return
        if kind == "human_sent":
            self.agent_activity = "消息已通过 Hub 发送，等待 Agent"
            self._conversation_line(
                self.human_profile.display_name,
                str(payload.get("content") or ""),
                style="cyan",
            )
            self._render_status()
            return
        if kind == "assistant_message":
            self.agent_activity = "Agent 已回复"
            self._conversation_line(
                self.agent_id,
                str(payload.get("content") or ""),
                style="green",
            )
            self._render_status()
            return
        if kind == "environment_action":
            self.agent_activity = f"Agent 正在执行 {payload.get('action_name')}"
            self._render_status()
        if kind == "environment_outcome":
            self.agent_activity = f"环境已返回 {payload.get('action_name')}"
            self._render_status()
        if kind == "human_broadcast":
            content = payload.get("content")
            self._protocol_line(f"broadcast <- {payload.get('sender')}: {content}", style="magenta")
            return
        if kind == "error":
            self.connection_state = "连接或协议错误"
            self._render_status()
            self._protocol_line(str(payload), style="red")
            return
        self._protocol_line(f"{kind}: {payload}")

    def _run_star_clients(self) -> None:
        asyncio.run(self._connect_clients())

    async def _connect_clients(self) -> None:
        self._star_loop = asyncio.get_running_loop()
        self.connection_state = "正在连接 Hub"
        self._notice("connecting", {})
        lifetime = asyncio.Event()
        self._connection_lifetime = lifetime
        try:
            self.environment = ConsoleEnvironmentClient(
                scene=self.scene,
                human_profile=self.human_profile,
                notice=self._notice,
                auto_reconnect=self.auto_reconnect,
                monitorable=self.monitorable,
            )
            await self.environment.connect(self.hub_url)
            await self.environment.start()

            self.human = ConsoleHumanClient(
                profile=self.human_profile,
                notice=self._notice,
                auto_reconnect=self.auto_reconnect,
                monitorable=self.monitorable,
            )
            await self.human.connect(self.hub_url)
            await self.human.start()
            await self.human.join_environment(self.scene.env_id)
            self._notice("connected", {})
            await lifetime.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._notice(
                "error",
                {"client": "startup", "type": type(exc).__name__, "message": str(exc)},
            )

    async def _send_initial_message_if_ready(self) -> None:
        content = self._pending_initial_message
        if not content or self.human is None or self.agent_id not in self.participants:
            return
        self._pending_initial_message = None
        sent = await self._submit_human_text(content)
        if not sent:
            self._pending_initial_message = content

    async def _submit_human_text(self, raw: str) -> bool:
        content = raw.strip()
        star_loop = self._star_loop
        if (
            not content
            or self.human is None
            or star_loop is None
            or not star_loop.is_running()
        ):
            return False
        if self.agent_id not in self.participants:
            self._protocol_line(
                f"消息未发送：Agent={self.agent_id} 尚未加入场景。",
                style="yellow",
            )
            return False
        self._message_sequence += 1
        message_id = f"human-{self._message_sequence}-{uuid4().hex[:8]}"
        params = {
            "content": content,
            "conversation_id": self.conversation_id,
            "message_id": message_id,
            "speaker_context": self.human_profile.to_context(),
            "scene_context": self.scene.to_context(self.participants),
        }
        try:
            send_future = asyncio.run_coroutine_threadsafe(
                self._send_human_action(params),
                star_loop,
            )
            await asyncio.wrap_future(send_future)
        except Exception as exc:
            self._notice(
                "error",
                {
                    "client": "human",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "target": self.agent_id,
                },
            )
            return False
        self._notice("human_sent", {"content": content, "message_id": message_id})
        return True

    async def _send_human_action(self, params: dict[str, Any]) -> None:
        human = self.human
        if human is None:
            raise RuntimeError("Human client is not connected.")
        await human.send_action(
            recipient=self.agent_id,
            action_name="user_message",
            params=params,
        )

    async def _stop_clients(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        self._set_composer_enabled(False)
        star_loop = self._star_loop
        if star_loop is not None and star_loop.is_running():
            stop_future = asyncio.run_coroutine_threadsafe(
                self._stop_clients_on_star_loop(),
                star_loop,
            )
            try:
                await asyncio.wrap_future(stop_future)
            except Exception as exc:
                self._protocol_line(f"Star clients stop error: {exc}", style="red")
            await asyncio.sleep(0)
        else:
            self.human = None
            self.environment = None
            self._connection_lifetime = None
        self._star_loop = None
        self.participants.clear()
        self._render_participants()
        self.connection_state = "已断开"
        self._render_status()

    async def _stop_clients_on_star_loop(self) -> None:
        lifetime = self._connection_lifetime
        self._connection_lifetime = None
        human, environment = self.human, self.environment
        self.human = None
        self.environment = None
        if human is not None:
            try:
                await human.stop()
            except Exception as exc:
                self._notice(
                    "error",
                    {"client": "human", "operation": "stop", "message": str(exc)},
                )
        if environment is not None:
            try:
                await environment.stop()
            except Exception as exc:
                self._notice(
                    "error",
                    {
                        "client": "environment",
                        "operation": "stop",
                        "message": str(exc),
                    },
                )
        if lifetime is not None:
            lifetime.set()

    def _notice(self, kind: str, payload: dict[str, Any]) -> None:
        message = StarNotice(kind, payload)
        if self._ui_thread_id == threading.get_ident():
            self.post_message(message)
            return
        self.call_from_thread(self.post_message, message)

    def _set_composer_enabled(self, enabled: bool) -> None:
        try:
            input_widget = self.query_one("#human-input", Input)
            send_button = self.query_one("#send-button", Button)
        except NoMatches:
            return
        input_widget.disabled = not enabled
        send_button.disabled = not enabled
        if enabled:
            input_widget.focus()

    def _render_scene(self) -> None:
        self.query_one("#scene-info", Static).update(
            "\n".join(
                [
                    f"[bold]{escape(self.scene.title)}[/bold]",
                    f"env_id={escape(self.scene.env_id)}",
                    escape(self.scene.background),
                ]
            )
        )
        profile = self.human_profile
        self.query_one("#human-profile", Static).update(
            "\n".join(
                [
                    f"[bold]已登录 Human[/bold] {escape(profile.display_name)}",
                    f"human_id={escape(profile.human_id)}",
                    f"role={escape(profile.role)}",
                    f"background={escape(profile.background)}",
                    f"relationship={escape(profile.relationship_to_agent)}",
                ]
            )
        )
        self._render_participants()

    def _render_participants(self) -> None:
        try:
            table = self.query_one("#participants", DataTable)
        except NoMatches:
            return
        try:
            table.clear(columns=False)
        except TypeError:
            table.clear()
        known = sorted(self.participants | {self.scene.env_id, self.human_profile.human_id, self.agent_id})
        for client_id in known:
            if client_id == self.scene.env_id:
                role = "environment"
                state = "online" if self.environment else "offline"
            elif client_id == self.human_profile.human_id:
                role = "human"
                state = "joined" if self.human else "offline"
            elif client_id == self.agent_id:
                role = "agent"
                state = "joined" if client_id in self.participants else "waiting"
            else:
                role = "client"
                state = "joined"
            table.add_row(client_id, role, state)

    def _render_status(self) -> None:
        try:
            connection_status = self.query_one("#connection-status", Static)
            agent_activity = self.query_one("#agent-activity", Static)
        except NoMatches:
            return
        connection_status.update(
            " | ".join(
                [
                    f"Hub: {escape(self.hub_url)}",
                    f"连接: {escape(self.connection_state)}",
                    f"Env: {escape(self.scene.env_id)}",
                    f"Human: {escape(self.human_profile.human_id)}",
                    f"Agent: {escape(self.agent_id)}",
                ]
            )
        )
        agent_activity.update(
            f"状态：{escape(self.agent_activity)}"
        )

    def _conversation_line(self, speaker: str, content: str, *, style: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        try:
            conversation = self.query_one("#conversation", RichLog)
        except NoMatches:
            return
        conversation.write(
            f"[dim]{timestamp}[/dim] [bold {style}]{escape(speaker)}[/bold {style}]\n"
            f"{escape(content)}\n"
        )

    def _protocol_line(self, content: str, *, style: str = "dim") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        try:
            protocol_log = self.query_one("#protocol-log", RichLog)
        except NoMatches:
            return
        protocol_log.write(
            f"[dim]{timestamp}[/dim] [{style}]{escape(content)}[/{style}]"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start a Star Protocol Environment + Human console for NPC dialogue.",
    )
    parser.add_argument("--hub-url", default="ws://localhost:8000")
    parser.add_argument("--env-id", default="npc-dialogue-lab")
    parser.add_argument("--agent-id", default="npc_agent", help="Target Agent client id.")
    parser.add_argument("--human-id", default="human_console")
    parser.add_argument("--human-name", default="林舟")
    parser.add_argument("--human-role", default="进入实验场景的访客")
    parser.add_argument(
        "--human-background",
        default="刚结束一天的工作，第一次来到这里，希望认识负责接待的 NPC。",
    )
    parser.add_argument("--relationship", default="与 Agent 初次见面")
    parser.add_argument(
        "--login-session",
        default=None,
        help="Fake authenticated login session id; generated when omitted.",
    )
    parser.add_argument("--scene-title", default="NPC 对话实验室")
    parser.add_argument(
        "--scene-background",
        default="傍晚的安静会客室，Human 与自主 NPC 可以持续交谈和观察彼此反应。",
    )
    parser.add_argument("--conversation-id", default=None)
    parser.add_argument("--initial-message", default=None)
    parser.add_argument("--monitorable", action="store_true")
    parser.add_argument("--no-auto-reconnect", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    login_session = args.login_session or f"login-{uuid4().hex[:12]}"
    conversation_id = args.conversation_id or (
        f"{args.env_id}:{args.human_id}:{args.agent_id}"
    )
    app = DialogueConsoleApp(
        hub_url=args.hub_url,
        scene=SceneConfig(
            env_id=args.env_id,
            title=args.scene_title,
            background=args.scene_background,
        ),
        human_profile=HumanProfile(
            human_id=args.human_id,
            display_name=args.human_name,
            role=args.human_role,
            background=args.human_background,
            relationship_to_agent=args.relationship,
            login_session=login_session,
        ),
        agent_id=args.agent_id,
        conversation_id=conversation_id,
        auto_reconnect=not args.no_auto_reconnect,
        monitorable=args.monitorable,
        initial_message=args.initial_message,
    )
    app.run()


if __name__ == "__main__":
    main()
