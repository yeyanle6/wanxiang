from __future__ import annotations

import inspect
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm_client import DEFAULT_MODEL, LLMClient
from .message import Message, MessageStatus
from .tools import ToolRegistry, ToolResult, ToolSpec

DEFAULT_PERSONA_PROMPT = (
    "Based on the task below, define your optimal approach for this run. "
    "Specify tone, depth, structure, and success criteria in concise bullet points.\n\n"
    "Task intent: {intent}\n"
    "Current sender: {sender}\n"
    "Current status: {status}\n"
    "Current turn: {turn}\n\n"
    "Respond with only the persona definition."
)

ToolEventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


class _SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass(slots=True)
class AgentConfig:
    name: str
    description: str
    base_identity: str
    persona_prompt: str = DEFAULT_PERSONA_PROMPT
    model: str = DEFAULT_MODEL
    max_tokens: int = 1024
    temperature: float = 0.2
    allowed_tools: list[str] = field(default_factory=list)
    native_tools: list[dict[str, Any]] = field(default_factory=list)
    max_tool_rounds: int = 5
    team_context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentConfig:
        file_path = Path(path)
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency/environment specific
            raise RuntimeError("PyYAML is required to load agent configs from YAML.") from exc

        raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid YAML config in {file_path}: root value must be a mapping.")

        base_identity = raw.get("base_identity") or raw.get("system_prompt")
        missing = [
            key
            for key, value in (
                ("name", raw.get("name")),
                ("description", raw.get("description")),
                ("base_identity", base_identity),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required fields in {file_path}: {', '.join(missing)}")

        return cls(
            name=str(raw["name"]).strip(),
            description=str(raw["description"]).strip(),
            base_identity=str(base_identity).strip(),
            persona_prompt=str(raw.get("persona_prompt", DEFAULT_PERSONA_PROMPT)).strip(),
            model=str(raw.get("model", DEFAULT_MODEL)).strip(),
            max_tokens=int(raw.get("max_tokens", 1024)),
            temperature=float(raw.get("temperature", 0.2)),
            allowed_tools=[
                str(name).strip()
                for name in raw.get("allowed_tools", [])
                if str(name).strip()
            ],
            native_tools=[
                {str(key): value for key, value in item.items() if str(key).strip()}
                for item in raw.get("native_tools", [])
                if isinstance(item, dict)
            ],
            max_tool_rounds=max(1, int(raw.get("max_tool_rounds", 5))),
        )

    def render_persona_prompt(self, message: Message) -> str:
        values = _SafeFormatDict(
            intent=message.intent,
            content=message.content,
            sender=message.sender,
            status=message.status.value,
            turn=message.turn,
            context="\n".join(message.context) if message.context else "(empty)",
            parent_id=message.parent_id,
            message_id=message.id,
        )
        return self.persona_prompt.format_map(values)


class BaseAgent:
    def __init__(
        self,
        config: AgentConfig,
        api_key: str | None = None,
        tool_registry: ToolRegistry | None = None,
        on_tool_event: ToolEventCallback | None = None,
        llm_mode: str | None = None,
        usage_recorder: Any = None,
    ) -> None:
        self.config = config
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.tool_registry = tool_registry
        self.on_tool_event = on_tool_event
        self.client = LLMClient(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            api_key=self.api_key,
            mode=llm_mode,
            usage_recorder=usage_recorder,
        )
        self.logger = logging.getLogger(f"wanxiang.agent.{config.name}")

    async def refine_persona(self, message: Message) -> str:
        persona_request = self.config.render_persona_prompt(message)
        planner_system = (
            "You are defining a task-specific persona for an agent execution.\n"
            "Return concise instructions only, no explanation."
        )
        raw = await self.call_llm(
            messages=[{"role": "user", "content": persona_request}],
            system="\n\n".join([self.config.base_identity, planner_system]),
        )
        return raw.strip()

    def build_prompt(self, message: Message, persona: str) -> list[dict[str, Any]]:
        task_input = message.to_prompt()
        # Writer takes precedence over reviewer. Some writer duties include
        # words like "review feedback", which would otherwise misclassify role.
        if self._is_writer_role():
            writer_instructions = (
                "You are the writer.\n"
                "Produce the final article draft directly.\n"
                "Rules:\n"
                "- Output Markdown content only, no meta commentary.\n"
                "- Do not mention tools, permissions, or execution environment.\n"
                "- Keep length around 900-1400 Chinese characters.\n"
                "- If reviewer feedback is provided, explicitly incorporate it.\n"
                "- For research/trend topics, attach source attribution for numeric claims (source + date).\n"
                "- If reliable source is unavailable, avoid precise numbers and use qualitative wording."
            )
            prompt_body = f"{writer_instructions}\n\nHandoff:\n{task_input}"
        elif self._is_reviewer_role():
            review_instructions = (
                "You are the quality gate reviewer.\n"
                "Evaluate the draft and return a decision in this exact format:\n"
                "STATUS: SUCCESS | NEEDS_REVISION | ERROR\n"
                "CONTENT:\n"
                "<actionable feedback or approval summary>\n\n"
                "Decision rules:\n"
                "- If this is the first review pass, default to NEEDS_REVISION and provide at least 3 concrete fixes.\n"
                "- Return SUCCESS only when the draft is publication-ready.\n"
                "- Keep feedback specific and directly applicable by the writer.\n"
                "- Keep feedback concise: no more than 8 bullet points.\n"
                "- On second or later review pass, if major issues are fixed, return SUCCESS."
            )
            capability_block = self._render_team_capability_block()
            sections = [review_instructions]
            if capability_block:
                sections.append(capability_block)
            sections.append(f"Handoff:\n{task_input}")
            prompt_body = "\n\n".join(sections)
        else:
            prompt_body = (
                "Process this handoff and produce the best possible result.\n\n"
                f"{task_input}"
            )
        return [
            {
                "role": "user",
                "content": prompt_body,
            }
        ]

    async def call_llm(self, messages: list[dict[str, Any]], system: str | None = None) -> str:
        return await self.client.generate(messages=messages, system=system)

    async def call_llm_response(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return await self.client.generate_response(messages=messages, system=system, tools=tools)

    def parse_response(self, raw: str, input_message: Message) -> Message:
        status, content = self._extract_status_and_content(raw)
        if status is None:
            status = self._infer_status(raw)
        if not content:
            content = raw.strip()

        return input_message.create_reply(
            intent=input_message.intent,
            content=content,
            sender=self.config.name,
            status=status,
            metadata={"agent": self.config.name, "model": self.config.model},
        )

    async def execute(self, message: Message) -> Message:
        self.logger.info(
            "Executing agent=%s turn=%s input_message_id=%s",
            self.config.name,
            message.turn,
            message.id,
        )
        try:
            persona = await self.refine_persona(message)
            messages = self.build_prompt(message, persona)
            system_prompt = "\n\n".join(
                [
                    self.config.base_identity.strip(),
                    "Task-specific persona:",
                    persona,
                    "Respond with the task output directly unless intent requires a specific format.",
                ]
            )
            tool_specs = self._resolve_tool_specs()
            native_tools = self._resolve_native_tools()
            tool_mode: str | None = None
            if tool_specs or native_tools:
                tool_mode = await self._resolve_tool_mode()
                if tool_mode == "cli" and native_tools:
                    self.logger.warning(
                        "Native tools unsupported in CLI mode; stripping: agent=%s tools=%s",
                        self.config.name,
                        [
                            str(item.get("name", ""))
                            for item in native_tools
                            if isinstance(item, dict)
                        ],
                    )
                    native_tools = []
            if tool_specs or native_tools:
                raw, tool_calls = await self._run_with_tool_loop(
                    messages=messages,
                    system_prompt=system_prompt,
                    tool_specs=tool_specs,
                    native_tools=native_tools,
                    turn=message.turn,
                    mode=tool_mode or await self._resolve_tool_mode(),
                )
            else:
                raw = await self.call_llm(messages=messages, system=system_prompt)
                tool_calls = []
            output = self.parse_response(raw, message)
            output.metadata.setdefault("persona", persona)
            output.metadata.setdefault("input_message_id", message.id)
            output.metadata.setdefault("allowed_tools", list(self.config.allowed_tools))
            output.metadata.setdefault("native_tools", list(self.config.native_tools))
            output.metadata.setdefault("tool_calls", tool_calls)
            output.metadata.setdefault("tool_mode", tool_mode)
            return output
        except Exception as exc:  # pragma: no cover - depends on external API/runtime failures
            self.logger.exception("Agent execution failed: agent=%s", self.config.name)
            return message.create_reply(
                intent=f"Execution failed in {self.config.name}",
                content=str(exc),
                sender=self.config.name,
                status=MessageStatus.ERROR,
                metadata={"error_type": type(exc).__name__},
            )

    def _resolve_tool_specs(self) -> list[ToolSpec]:
        if not self.config.allowed_tools:
            return []
        if self.tool_registry is None:
            raise RuntimeError(
                f"Agent '{self.config.name}' has allowed_tools configured but no ToolRegistry is attached."
            )
        return self.tool_registry.filter_for_agent(self.config.allowed_tools)

    def _resolve_native_tools(self) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        for entry in self.config.native_tools:
            if not isinstance(entry, dict):
                continue
            tool = {str(key): value for key, value in entry.items() if str(key).strip()}
            if tool:
                resolved.append(tool)
        return resolved

    async def _run_with_tool_loop(
        self,
        *,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tool_specs: list[ToolSpec],
        native_tools: list[dict[str, Any]],
        turn: int,
        mode: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        if mode == "api":
            return await self._run_with_tool_loop_api(
                messages=messages,
                system_prompt=system_prompt,
                tool_specs=tool_specs,
                native_tools=native_tools,
                turn=turn,
            )
        if mode == "cli":
            if native_tools:
                raise RuntimeError(
                    "Native tools require API mode. Configure ANTHROPIC_API_KEY or disable native_tools."
                )
            return await self._run_with_tool_loop_cli(
                messages=messages,
                system_prompt=system_prompt,
                tool_specs=tool_specs,
                turn=turn,
            )
        raise RuntimeError(f"Unsupported tool mode: {mode}")

    async def _run_with_tool_loop_api(
        self,
        *,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tool_specs: list[ToolSpec],
        native_tools: list[dict[str, Any]],
        turn: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        tools_payload = [spec.to_claude_tool() for spec in tool_specs] + list(native_tools)
        allowed = {spec.name for spec in tool_specs}
        conversation: list[dict[str, Any]] = list(messages)
        call_log: list[dict[str, Any]] = []
        last_text = ""

        for index in range(self.config.max_tool_rounds):
            tool_round = index + 1
            response = await self.call_llm_response(
                messages=conversation,
                system=system_prompt,
                tools=tools_payload,
            )
            content_blocks = response.get("content", [])
            if not isinstance(content_blocks, list):
                content_blocks = []
            stop_reason = str(response.get("stop_reason", "")).strip().lower()

            await self._emit_native_tool_events_from_blocks(
                content_blocks=content_blocks,
                turn=turn,
                tool_round=tool_round,
            )

            text = self._extract_text_from_blocks(content_blocks)
            if text:
                last_text = text

            tool_uses = [
                block
                for block in content_blocks
                if isinstance(block, dict) and block.get("type") == "tool_use"
            ]
            if stop_reason == "end_turn":
                if text.strip():
                    return text.strip(), call_log
                if last_text.strip():
                    return last_text.strip(), call_log
                raise RuntimeError("LLM ended turn without text output.")

            if stop_reason and stop_reason != "tool_use":
                if text.strip():
                    return text.strip(), call_log
                if last_text.strip():
                    return last_text.strip(), call_log
                if not tool_uses:
                    raise RuntimeError(
                        f"LLM ended with stop_reason='{stop_reason}' but no text output was returned."
                    )

            if not tool_uses:
                if text.strip():
                    return text.strip(), call_log
                if last_text.strip():
                    return last_text.strip(), call_log
                raise RuntimeError("LLM returned no text output after tool rounds.")

            conversation.append({"role": "assistant", "content": content_blocks})

            tool_result_blocks: list[dict[str, Any]] = []
            for block in tool_uses:
                tool_name = str(block.get("name", "")).strip()
                tool_use_id = str(block.get("id", "")).strip()
                raw_args = block.get("input", {})
                arguments = raw_args if isinstance(raw_args, dict) else {}

                await self._emit_tool_event(
                    {
                        "type": "tool_started",
                        "agent": self.config.name,
                        "tool": tool_name,
                        "tool_use_id": tool_use_id,
                        "arguments": arguments,
                        "turn": turn,
                        "tool_round": tool_round,
                    }
                )
                result = await self._execute_tool_with_allowlist(
                    tool_name=tool_name,
                    tool_use_id=tool_use_id,
                    arguments=arguments,
                    allowed=allowed,
                )
                preview_source = result.content if result.success else (result.error or "")
                await self._emit_tool_event(
                    {
                        "type": "tool_completed",
                        "agent": self.config.name,
                        "tool": result.tool_name or tool_name,
                        "tool_use_id": tool_use_id,
                        "turn": turn,
                        "tool_round": tool_round,
                        "success": result.success,
                        "elapsed_ms": result.elapsed_ms,
                        "content_preview": self._preview(preview_source),
                        "error": result.error,
                    }
                )
                call_log.append(
                    {
                        "tool": result.tool_name,
                        "success": result.success,
                        "elapsed_ms": result.elapsed_ms,
                        "error": result.error,
                    }
                )
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result.content
                        if result.success
                        else f"Tool execution failed: {result.error}",
                        "is_error": not result.success,
                    }
                )

            conversation.append({"role": "user", "content": tool_result_blocks})

        raise RuntimeError(
            f"Exceeded max_tool_rounds={self.config.max_tool_rounds} without a final text response."
        )

    async def _emit_native_tool_events_from_blocks(
        self,
        *,
        content_blocks: list[dict[str, Any]],
        turn: int,
        tool_round: int,
    ) -> None:
        server_uses = self._extract_server_tool_uses(content_blocks)
        if not server_uses:
            return

        result_map = self._extract_server_tool_results(content_blocks)
        for use in server_uses:
            tool_name = str(use.get("name", "")).strip()
            tool_use_id = str(use.get("id", "")).strip()
            arguments = use.get("input", {})
            if not isinstance(arguments, dict):
                arguments = {}

            await self._emit_tool_event(
                {
                    "type": "tool_started",
                    "agent": self.config.name,
                    "tool": tool_name,
                    "tool_use_id": tool_use_id,
                    "arguments": arguments,
                    "turn": turn,
                    "tool_round": tool_round,
                }
            )

            result_block = result_map.get(tool_use_id)
            success = True
            error: str | None = None
            preview_source = "Native tool execution completed on server side."
            if isinstance(result_block, dict):
                is_error = bool(result_block.get("is_error", False))
                success = not is_error
                preview_source = self._stringify_server_tool_result(result_block)
                if is_error:
                    error = self._preview(preview_source)

            await self._emit_tool_event(
                {
                    "type": "tool_completed",
                    "agent": self.config.name,
                    "tool": tool_name,
                    "tool_use_id": tool_use_id,
                    "turn": turn,
                    "tool_round": tool_round,
                    "success": success,
                    "elapsed_ms": 0,
                    "content_preview": self._preview(preview_source),
                    "error": error,
                }
            )

    async def _run_with_tool_loop_cli(
        self,
        *,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tool_specs: list[ToolSpec],
        turn: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        allowed = {spec.name for spec in tool_specs}
        conversation: list[dict[str, Any]] = list(messages)
        call_log: list[dict[str, Any]] = []
        protocol_system = self._build_cli_tool_system_prompt(system_prompt, tool_specs)

        for index in range(self.config.max_tool_rounds):
            tool_round = index + 1
            raw = await self.call_llm(messages=conversation, system=protocol_system)
            decision = self._parse_cli_tool_decision(raw)

            if decision is None:
                text = raw.strip()
                if text:
                    return text, call_log
                raise RuntimeError("CLI tool mode returned empty response.")

            action = str(decision.get("action", "")).strip().lower()
            if action == "final":
                content = str(decision.get("content", "")).strip()
                if content:
                    return content, call_log
                text = raw.strip()
                if text:
                    return text, call_log
                raise RuntimeError("CLI tool mode returned empty final content.")

            if action != "tool":
                text = raw.strip()
                if text:
                    return text, call_log
                raise RuntimeError(f"Unsupported CLI tool action: {action}")

            tool_name = str(decision.get("tool", "")).strip()
            arguments = decision.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}

            await self._emit_tool_event(
                {
                    "type": "tool_started",
                    "agent": self.config.name,
                    "tool": tool_name,
                    "tool_use_id": f"cli-tool-{tool_round}",
                    "arguments": arguments,
                    "turn": turn,
                    "tool_round": tool_round,
                }
            )

            result = await self._execute_tool_with_allowlist(
                tool_name=tool_name,
                tool_use_id=f"cli-tool-{tool_round}",
                arguments=arguments,
                allowed=allowed,
            )
            preview_source = result.content if result.success else (result.error or "")
            await self._emit_tool_event(
                {
                    "type": "tool_completed",
                    "agent": self.config.name,
                    "tool": result.tool_name or tool_name,
                    "tool_use_id": f"cli-tool-{tool_round}",
                    "turn": turn,
                    "tool_round": tool_round,
                    "success": result.success,
                    "elapsed_ms": result.elapsed_ms,
                    "content_preview": self._preview(preview_source),
                    "error": result.error,
                }
            )

            call_log.append(
                {
                    "tool": result.tool_name or tool_name,
                    "success": result.success,
                    "elapsed_ms": result.elapsed_ms,
                    "error": result.error,
                }
            )

            conversation.append({"role": "assistant", "content": raw})
            tool_result_text = (
                f"TOOL RESULT\n"
                f"tool: {result.tool_name or tool_name}\n"
                f"success: {result.success}\n"
                f"elapsed_ms: {result.elapsed_ms}\n"
                f"content: {result.content if result.success else (result.error or '')}\n\n"
                "Return your next action JSON."
            )
            conversation.append({"role": "user", "content": tool_result_text})

        raise RuntimeError(
            f"Exceeded max_tool_rounds={self.config.max_tool_rounds} without a final response in CLI mode."
        )

    async def _execute_tool_with_allowlist(
        self,
        *,
        tool_name: str,
        tool_use_id: str,
        arguments: dict[str, Any],
        allowed: set[str],
    ) -> ToolResult:
        if not tool_name:
            return ToolResult(
                tool_name="",
                success=False,
                content="",
                elapsed_ms=0,
                error="Missing tool name in tool_use block.",
            )
        if not tool_use_id:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                content="",
                elapsed_ms=0,
                error="Missing tool_use id in model response.",
            )
        if tool_name not in allowed:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                content="",
                elapsed_ms=0,
                error=f"Tool '{tool_name}' is not allowed for agent '{self.config.name}'.",
            )
        if self.tool_registry is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                content="",
                elapsed_ms=0,
                error="Tool registry is unavailable.",
            )
        return await self.tool_registry.execute(tool_name, arguments)

    def _extract_text_from_blocks(self, blocks: list[dict[str, Any]]) -> str:
        text_parts: list[str] = []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        return "".join(text_parts).strip()

    def _extract_server_tool_uses(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        server_uses: list[dict[str, Any]] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type", "")).strip()
            if block_type == "server_tool_use" or (
                block_type.endswith("_tool_use") and block_type != "tool_use"
            ):
                server_uses.append(block)
        return server_uses

    def _extract_server_tool_results(
        self, blocks: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        result_map: dict[str, dict[str, Any]] = {}
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type", "")).strip()
            if block_type == "tool_result":
                continue
            if block_type == "server_tool_result" or block_type.endswith("_tool_result"):
                tool_use_id = str(block.get("tool_use_id", "")).strip()
                if tool_use_id:
                    result_map[tool_use_id] = block
        return result_map

    def _stringify_server_tool_result(self, block: dict[str, Any]) -> str:
        raw_content = block.get("content")
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, list):
            snippets: list[str] = []
            for item in raw_content:
                if isinstance(item, dict):
                    title = str(item.get("title", "")).strip()
                    url = str(item.get("url", "")).strip()
                    text = str(item.get("text", "")).strip()
                    snippet = " | ".join(part for part in (title, url, text) if part)
                    if snippet:
                        snippets.append(snippet)
            if snippets:
                return "\n".join(snippets)
        try:
            return json.dumps(raw_content, ensure_ascii=False)
        except Exception:
            return str(raw_content)

    async def _emit_tool_event(self, event: dict[str, Any]) -> None:
        if self.on_tool_event is None:
            return
        try:
            maybe_awaitable = self.on_tool_event(event)
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        except Exception:  # pragma: no cover - observability should not break main flow
            self.logger.exception("Tool event callback failed: agent=%s", self.config.name)

    def _preview(self, content: str, limit: int = 200) -> str:
        compact = " ".join(str(content).strip().split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

    async def _resolve_tool_mode(self) -> str:
        resolver = getattr(self.client, "resolve_mode", None)
        if callable(resolver):
            resolved = resolver(require_tools=True)
            if inspect.isawaitable(resolved):
                return str(await resolved)
            return str(resolved)
        return "api"

    def _build_cli_tool_system_prompt(self, base_system: str, tool_specs: list[ToolSpec]) -> str:
        tool_lines = [
            f"- {spec.name}: {spec.description} | schema={spec.input_schema}" for spec in tool_specs
        ]
        protocol = (
            "You are operating in CLI tool mode.\n"
            "You MUST return JSON only, no markdown, no extra text.\n"
            "Two valid outputs:\n"
            '1) {"action":"tool","tool":"<tool_name>","arguments":{...}}\n'
            '2) {"action":"final","content":"<final response>"}\n'
            "Pick action=tool only when a listed tool is needed."
        )
        return "\n\n".join([base_system, protocol, "Available tools:\n" + "\n".join(tool_lines)])

    def _parse_cli_tool_decision(self, raw: str) -> dict[str, Any] | None:
        text = raw.strip()
        if not text:
            return None
        candidate = text
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
            candidate = re.sub(r"\s*```$", "", candidate)
        parsed = self._try_parse_json_object(candidate)
        if parsed is None:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                parsed = self._try_parse_json_object(match.group(0))
        if parsed is None:
            return None
        action = str(parsed.get("action", "")).strip().lower()
        if action not in {"tool", "final"}:
            return None
        if action == "tool":
            parsed["tool"] = str(parsed.get("tool", "")).strip()
            if not isinstance(parsed.get("arguments"), dict):
                parsed["arguments"] = {}
        return parsed

    def _try_parse_json_object(self, text: str) -> dict[str, Any] | None:
        try:
            value = json.loads(text)
        except Exception:
            return None
        if isinstance(value, dict):
            return value
        return None

    def _is_reviewer_role(self) -> bool:
        role_text = f"{self.config.name} {self.config.description} {self.config.base_identity}".lower()
        keywords = ("review", "reviewer", "quality", "critic", "audit", "审核", "评审", "审校")
        return any(word in role_text for word in keywords)

    def _render_team_capability_block(self) -> str:
        team_context = self.config.team_context or {}
        peers = team_context.get("agents")
        if not isinstance(peers, list) or not peers:
            return ""

        effective_mode = str(team_context.get("effective_mode") or "").strip().lower()
        cli_mode = effective_mode == "cli"

        writer_peer: dict[str, Any] | None = None
        for peer in peers:
            if not isinstance(peer, dict):
                continue
            name = str(peer.get("name", "")).lower()
            duty = str(peer.get("duty", "")).lower()
            identity = str(peer.get("base_identity", "")).lower()
            text = f"{name} {duty} {identity}"
            is_writer_like = any(
                word in text
                for word in ("writer", "write", "author", "撰写", "写作", "作者")
            )
            is_reviewer_like = any(
                word in text
                for word in ("review", "reviewer", "critic", "quality", "审核", "评审", "审校")
            )
            if is_writer_like and not is_reviewer_like:
                writer_peer = peer
                break

        # Fallback: Director often names the producer agent things like
        # "analyzer" / "researcher" / "summarizer" that don't match the
        # writer-keyword heuristic. In a review_loop the producer is always
        # the first agent in execution_order — use it as the effective peer
        # so reviewer still receives tool-aware evaluation rules.
        if writer_peer is None:
            execution_order = team_context.get("execution_order")
            if isinstance(execution_order, list) and execution_order:
                producer_name = str(execution_order[0]).strip()
                if producer_name and producer_name != self.config.name:
                    for peer in peers:
                        if (
                            isinstance(peer, dict)
                            and str(peer.get("name", "")).strip() == producer_name
                        ):
                            writer_peer = peer
                            break

        if writer_peer is None:
            return ""

        native_tool_names = [
            str(item.get("name", "")).strip().lower()
            for item in writer_peer.get("native_tools") or []
            if isinstance(item, dict)
        ]
        allowed_tools = [
            str(name).strip().lower()
            for name in writer_peer.get("allowed_tools") or []
            if str(name).strip()
        ]

        has_native_web_search = ("web_search" in native_tool_names) and not cli_mode
        has_any_registry_tool = bool(allowed_tools)

        lines = ["Team tool availability:"]
        lines.append(
            f"- writer (name={writer_peer.get('name', '')}): "
            f"native_web_search={'yes' if has_native_web_search else 'no'}, "
            f"registry_tools={allowed_tools or '(none)'}"
        )
        if cli_mode:
            lines.append("- llm_mode: cli (server-side native tools such as web_search are disabled)")

        lines.append("")
        lines.append("Evaluation rules for THIS run (tool-aware):")
        if has_native_web_search:
            lines.append(
                "- Writer has live web_search. REQUIRE at least one cited source "
                "(publisher + date or URL) per major quantitative or factual claim."
            )
            lines.append(
                "- If citations are missing despite the writer having web_search, "
                "return NEEDS_REVISION with the specific claims that need sourcing."
            )
        else:
            lines.append(
                "- Writer has NO live web_search in this run. Do NOT require specific "
                "citations, URLs, Gartner/IDC/McKinsey reports, or exact market-size numbers."
            )
            lines.append(
                "- Do NOT ask the writer to cite 'Nature AI Systems Review 2025' or similar "
                "specific sources — the writer cannot fetch them and will hallucinate if pressed."
            )
            lines.append(
                "- Evaluate instead: structure, internal logical consistency, clarity, balanced "
                "coverage of perspectives, clear separation of observed vs. predicted, and "
                "presence of an explicit 'no live-search' disclaimer."
            )
            lines.append(
                "- If the draft already acknowledges the offline constraint and is internally "
                "coherent with qualitative analysis, that is sufficient — return SUCCESS."
            )

        if has_any_registry_tool:
            lines.append(
                f"- Writer has registry tools available: {allowed_tools}. "
                "If the task naturally fits these tools and they were not used, flag it."
            )
            lines.append(
                "- Writer's knowledge in this run is BOUNDED by what those registry "
                "tools can surface (e.g. read_text_file only returns what the file "
                "contains). Evaluate the draft on faithfulness to that source "
                "material, structural clarity, and balanced coverage. Do NOT "
                "demand details outside the source's scope — API specs, protocols, "
                "failure modes, scaling numbers, code snippets, etc. — unless the "
                "source itself contains them."
            )
            lines.append(
                "- If the draft explicitly notes which aspects are absent from the "
                "source material (e.g. a 'Limitations' or 'Scope' section), treat "
                "that as proper scoping, NOT a weakness. Return SUCCESS once the "
                "draft accurately reflects the source and is internally coherent."
            )

        return "\n".join(lines)

    def _is_writer_role(self) -> bool:
        role_text = f"{self.config.name} {self.config.description} {self.config.base_identity}".lower()
        keywords = ("writer", "author", "write", "blog", "撰写", "写作", "作者", "博客")
        return any(word in role_text for word in keywords)

    def _extract_status_and_content(self, raw: str) -> tuple[MessageStatus | None, str]:
        text = raw.strip()
        if not text:
            return None, ""

        # JSON extraction (common for structured responses).
        try:
            import json

            data = json.loads(text)
            if isinstance(data, dict):
                raw_status = str(data.get("status", "")).strip().lower()
                content = str(data.get("content", "")).strip()
                status = self._map_status(raw_status)
                if status is not None:
                    return status, content
        except Exception:
            pass

        # Text protocol extraction.
        status_match = re.search(
            r"(?im)^\s*status\s*:\s*(success|needs_revision|error)\s*$",
            text,
        )
        content_match = re.search(r"(?is)^\s*content\s*:\s*(.*)$", text, flags=re.MULTILINE)
        if status_match:
            mapped = self._map_status(status_match.group(1))
            extracted_content = content_match.group(1).strip() if content_match else ""
            return mapped, extracted_content

        return None, ""

    def _infer_status(self, raw: str) -> MessageStatus:
        if not self._is_reviewer_role():
            return MessageStatus.SUCCESS

        lowered = raw.lower()
        negative_tokens = (
            "needs_revision",
            "need revision",
            "requires revision",
            "please revise",
            "needs significant",
            "must revise",
            "requires improvement",
            "需要修改",
            "请修改",
            "需补充",
            "不通过",
        )
        if any(token in lowered for token in negative_tokens):
            return MessageStatus.NEEDS_REVISION

        positive_tokens = (
            "approved",
            "pass",
            "publication-ready",
            "publish-ready",
            "ready for publication",
            "meets publication standards",
            "通过",
            "已通过",
            "可发布",
            "达到发布标准",
        )
        if any(token in lowered for token in positive_tokens):
            return MessageStatus.SUCCESS

        if any(
            token in lowered
            for token in (
                "需要修改",
                "请修改",
                "需补充",
                "不通过",
            )
        ):
            return MessageStatus.NEEDS_REVISION

        # Conservative default for reviewer to keep the quality gate meaningful.
        return MessageStatus.NEEDS_REVISION

    def _map_status(self, value: str) -> MessageStatus | None:
        if value == "success":
            return MessageStatus.SUCCESS
        if value in ("needs_revision", "needs revision"):
            return MessageStatus.NEEDS_REVISION
        if value == "error":
            return MessageStatus.ERROR
        return None
