from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .agent import AgentConfig, BaseAgent
from .llm_client import DEFAULT_MODEL, LLMClient
from .tools import ToolRegistry

SUPPORTED_WORKFLOWS = {"pipeline", "parallel", "review_loop"}
ToolEventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


@dataclass(slots=True)
class AgentSpec:
    name: str
    duty: str
    base_identity: str
    persona_prompt: str | None = None
    allowed_tools: list[str] | None = None
    native_tools: list[dict[str, Any]] | None = None
    max_tool_rounds: int | None = None

    def to_agent_config(self) -> AgentConfig:
        kwargs: dict[str, Any] = {
            "name": self.name,
            "description": self.duty,
            "base_identity": self.base_identity,
        }
        if self.persona_prompt:
            kwargs["persona_prompt"] = self.persona_prompt
        if self.allowed_tools is not None:
            kwargs["allowed_tools"] = list(self.allowed_tools)
        if self.native_tools is not None:
            kwargs["native_tools"] = [
                {str(key): value for key, value in item.items() if str(key).strip()}
                for item in self.native_tools
                if isinstance(item, dict)
            ]
        if self.max_tool_rounds is not None:
            kwargs["max_tool_rounds"] = max(1, int(self.max_tool_rounds))
        return AgentConfig(**kwargs)


@dataclass(slots=True)
class TeamPlan:
    agents: list[AgentSpec]
    workflow: str
    execution_order: list[str]
    max_iterations: int = 1
    rationale: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TeamPlan:
        raw_agents = data.get("agents")
        if not isinstance(raw_agents, list) or not raw_agents:
            raise ValueError("Team plan must include a non-empty 'agents' list.")

        agents: list[AgentSpec] = []
        for entry in raw_agents:
            if not isinstance(entry, dict):
                raise ValueError("Each 'agents' entry must be an object.")
            name = str(entry.get("name", "")).strip()
            duty = str(entry.get("duty") or entry.get("description") or "").strip()
            base_identity = str(entry.get("base_identity") or entry.get("system_prompt") or "").strip()
            persona_prompt = entry.get("persona_prompt")
            allowed_tools = entry.get("allowed_tools")
            native_tools = entry.get("native_tools")
            max_tool_rounds = entry.get("max_tool_rounds")

            if not name or not duty or not base_identity:
                raise ValueError(
                    "Each agent spec requires 'name', 'duty', and 'base_identity'."
                )
            agents.append(
                AgentSpec(
                    name=name,
                    duty=duty,
                    base_identity=base_identity,
                    persona_prompt=str(persona_prompt).strip() if persona_prompt else None,
                    allowed_tools=[
                        str(tool).strip()
                        for tool in (allowed_tools if isinstance(allowed_tools, list) else [])
                        if str(tool).strip()
                    ],
                    native_tools=[
                        {str(key): value for key, value in tool.items() if str(key).strip()}
                        for tool in (native_tools if isinstance(native_tools, list) else [])
                        if isinstance(tool, dict)
                    ],
                    max_tool_rounds=int(max_tool_rounds)
                    if isinstance(max_tool_rounds, int) and max_tool_rounds > 0
                    else None,
                )
            )

        workflow = str(data.get("workflow", "pipeline")).strip().lower()
        if workflow not in SUPPORTED_WORKFLOWS:
            raise ValueError(
                f"Unsupported workflow '{workflow}'. "
                f"Expected one of: {', '.join(sorted(SUPPORTED_WORKFLOWS))}."
            )

        raw_order = data.get("execution_order")
        known_names = [agent.name for agent in agents]
        if isinstance(raw_order, list):
            order = [str(name).strip() for name in raw_order if str(name).strip() in known_names]
        else:
            order = []
        for name in known_names:
            if name not in order:
                order.append(name)

        max_iterations = int(data.get("max_iterations", 1))
        if max_iterations < 1:
            max_iterations = 1

        rationale = str(data.get("rationale", "")).strip()
        return cls(
            agents=agents,
            workflow=workflow,
            execution_order=order,
            max_iterations=max_iterations,
            rationale=rationale,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "max_iterations": self.max_iterations,
            "execution_order": list(self.execution_order),
            "rationale": self.rationale,
            "agents": [
                {
                    "name": spec.name,
                    "duty": spec.duty,
                    "base_identity": spec.base_identity,
                    "persona_prompt": spec.persona_prompt,
                    "allowed_tools": list(spec.allowed_tools or []),
                    "native_tools": list(spec.native_tools or []),
                    "max_tool_rounds": spec.max_tool_rounds,
                }
                for spec in self.agents
            ],
        }


class AgentFactory:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1400,
        temperature: float = 0.1,
        tool_registry: ToolRegistry | None = None,
        llm_mode: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tool_registry = tool_registry
        self.llm_mode = llm_mode
        self.client = LLMClient(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            api_key=self.api_key,
            mode=self.llm_mode,
        )
        self.logger = logging.getLogger("agentforge.factory")

    async def create_team(self, task: str) -> TeamPlan:
        cleaned_task = task.strip()
        if not cleaned_task:
            raise ValueError("Task must be non-empty.")

        messages = [{"role": "user", "content": f"Task:\n{cleaned_task}"}]
        raw = await self._call_planner(messages)
        plan_data = self._parse_json_response(raw)
        plan = TeamPlan.from_dict(plan_data)
        effective_mode = await self._safe_resolve_mode()
        plan = self._apply_planning_policies(cleaned_task, plan, effective_mode=effective_mode)
        self.logger.info(
            "Created team: workflow=%s agents=%s mode=%s",
            plan.workflow,
            [agent.name for agent in plan.agents],
            effective_mode,
        )
        return plan

    async def _safe_resolve_mode(self) -> str | None:
        resolver = getattr(self.client, "resolve_mode", None)
        if not callable(resolver):
            return None
        try:
            result = resolver(require_tools=False)
            if hasattr(result, "__await__"):
                result = await result  # type: ignore[assignment]
            return str(result) if result else None
        except Exception:
            return None

    def instantiate_team(
        self,
        plan: TeamPlan,
        *,
        api_key: str | None = None,
        on_tool_event: ToolEventCallback | None = None,
        effective_mode: str | None = None,
    ) -> dict[str, BaseAgent]:
        team: dict[str, BaseAgent] = {}
        agent_api_key = api_key or self.api_key
        team_snapshot = self._build_team_snapshot(plan, effective_mode)
        for spec in plan.agents:
            config = spec.to_agent_config()
            config.team_context = dict(team_snapshot)
            team[spec.name] = BaseAgent(
                config,
                api_key=agent_api_key,
                tool_registry=self.tool_registry,
                on_tool_event=on_tool_event,
                llm_mode=self.llm_mode,
            )
        return team

    def _build_team_snapshot(
        self, plan: TeamPlan, effective_mode: str | None
    ) -> dict[str, Any]:
        peers: list[dict[str, Any]] = []
        for spec in plan.agents:
            peers.append(
                {
                    "name": spec.name,
                    "duty": spec.duty,
                    "base_identity": spec.base_identity,
                    "allowed_tools": list(spec.allowed_tools or []),
                    "native_tools": [
                        {
                            "name": str(item.get("name", "")).strip(),
                            "type": str(item.get("type", "")).strip(),
                        }
                        for item in (spec.native_tools or [])
                        if isinstance(item, dict)
                    ],
                }
            )
        snapshot: dict[str, Any] = {
            "workflow": plan.workflow,
            "execution_order": list(plan.execution_order),
            "agents": peers,
        }
        if effective_mode:
            snapshot["effective_mode"] = effective_mode
        return snapshot

    async def _call_planner(self, messages: list[dict[str, Any]]) -> str:
        available_tools = (
            ", ".join(self.tool_registry.list_tools()) if self.tool_registry else "(none)"
        )
        available_native_tools = (
            '- web_search: {"type":"web_search_20250305","name":"web_search","max_uses":5}'
        )
        system_prompt = (
            "You are a Director agent that designs execution plans for a multi-agent system.\n"
            "Return JSON only (no markdown) with this schema:\n"
            "{\n"
            '  "workflow": "pipeline|parallel|review_loop",\n'
            '  "max_iterations": 1,\n'
            '  "execution_order": ["agent_name"],\n'
            '  "rationale": "short reason",\n'
            '  "agents": [\n'
            "    {\n"
            '      "name": "writer",\n'
            '      "duty": "what this agent does",\n'
            '      "base_identity": "stable role identity",\n'
            '      "persona_prompt": "optional task-adaptive persona template",\n'
            '      "allowed_tools": ["optional_tool_name"],\n'
            '      "native_tools": [{"type":"web_search_20250305","name":"web_search","max_uses":5}],\n'
            '      "max_tool_rounds": 5\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- Keep the team minimal but effective.\n"
            "- Minimum 2 agents for any task.\n"
            "- Every pipeline MUST include at least one quality reviewer.\n"
            "- For content tasks (blog/article/report/docs), default to review_loop.\n"
            "- For content tasks, typical team is writer + reviewer.\n"
            "- For review_loop, set max_iterations >= 2.\n"
            "- For tasks requiring multiple perspectives, comparisons, or multi-source research, "
            'use workflow "parallel".\n'
            "- In parallel mode, execution_order lists parallel branches first, "
            "with the LAST agent as synthesizer.\n"
            '- Example: ["researcher_a", "researcher_b", "synthesizer"]\n'
            "- Parallel branches all receive the same input and run concurrently.\n"
            "- Ensure execution_order references existing agent names.\n"
            f"- Available tools: {available_tools}\n"
            "Available native tools:\n"
            f"{available_native_tools}\n"
            "- Assign allowed_tools only when needed, and only from available tools.\n"
            "- Keep allowed_tools empty for agents that do not need tools.\n"
            "- For research/parallel tasks, assign web_search native tool to researcher branches.\n"
            "- For content tasks requiring current facts, writer may use web_search native tool.\n"
            "- Reviewer should not receive web_search native tool."
        )
        raw = await self.client.generate(messages=messages, system=system_prompt)
        if not raw:
            raise RuntimeError("Factory planner returned no text content.")
        return raw

    def _parse_json_response(self, raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise ValueError("Unable to parse team plan JSON from model response.") from None
            data = json.loads(match.group(0))

        if not isinstance(data, dict):
            raise ValueError("Team plan response must decode to a JSON object.")
        return data

    def _apply_planning_policies(
        self,
        task: str,
        plan: TeamPlan,
        *,
        effective_mode: str | None = None,
    ) -> TeamPlan:
        content_task = self._is_content_task(task)
        research_task = self._is_research_task(task)
        native_tools_supported = effective_mode != "cli"

        if content_task:
            writer_name = self._ensure_writer(plan)
            reviewer_name = self._ensure_reviewer(plan)
            plan.workflow = "review_loop"
            plan.max_iterations = max(plan.max_iterations, 3 if research_task else 2)
            plan.execution_order = [writer_name, reviewer_name]
            if research_task and native_tools_supported:
                self._ensure_native_web_search_for_agent(plan, writer_name)
            if not native_tools_supported:
                self._strip_all_native_tools(plan)
            self._strip_web_search_from_agent(plan, reviewer_name)
            return plan

        if plan.workflow == "parallel":
            self._ensure_parallel_structure(plan)
            if not native_tools_supported:
                self._strip_all_native_tools(plan)
            return plan

        # Enforce minimum collaboration size.
        if len(plan.agents) < 2:
            self._ensure_reviewer(plan)

        # For non-content pipeline plans, still require reviewer in route.
        if plan.workflow == "pipeline":
            reviewer_name = self._ensure_reviewer(plan)
            self._strip_web_search_from_agent(plan, reviewer_name)
            if reviewer_name not in plan.execution_order:
                plan.execution_order.append(reviewer_name)

        if not native_tools_supported:
            self._strip_all_native_tools(plan)

        return plan

    def _strip_all_native_tools(self, plan: TeamPlan) -> None:
        for spec in plan.agents:
            if spec.native_tools:
                spec.native_tools = []

    def _is_content_task(self, task: str) -> bool:
        text = task.lower()
        keywords = (
            "write",
            "writer",
            "blog",
            "article",
            "post",
            "report",
            "docs",
            "documentation",
            "copywriting",
            "写",
            "博客",
            "文章",
            "文档",
            "报告",
            "稿",
        )
        return any(word in text for word in keywords)

    def _is_research_task(self, task: str) -> bool:
        text = task.lower()
        keywords = (
            "research",
            "trend",
            "latest",
            "analysis",
            "market",
            "report",
            "framework",
            "study",
            "2026",
            "研究",
            "趋势",
            "最新",
            "分析",
            "报告",
            "框架",
            "多角度",
            "多个角度",
        )
        return any(word in text for word in keywords)

    def _ensure_writer(self, plan: TeamPlan) -> str:
        for spec in plan.agents:
            if spec.name == "writer":
                return spec.name

        candidate: AgentSpec | None = None
        for spec in plan.agents:
            role_text = f"{spec.name} {spec.duty}".lower()
            identity_text = spec.base_identity.lower()
            writer_like = self._is_writer_text(role_text) or self._is_writer_text(identity_text)
            reviewer_like = self._is_reviewer_text(role_text) or self._is_reviewer_text(identity_text)
            if writer_like and not reviewer_like:
                candidate = spec
                break

        if candidate is not None:
            writer = AgentSpec(
                name="writer",
                duty=candidate.duty,
                base_identity=candidate.base_identity,
                persona_prompt=candidate.persona_prompt,
                allowed_tools=list(candidate.allowed_tools or []),
                native_tools=[
                    {str(key): value for key, value in item.items() if str(key).strip()}
                    for item in (candidate.native_tools or [])
                    if isinstance(item, dict)
                ],
                max_tool_rounds=candidate.max_tool_rounds,
            )
            plan.agents.insert(0, writer)
            return writer.name

        writer = AgentSpec(
            name="writer",
            duty="Produce the main content draft based on task intent and context.",
            base_identity=(
                "You are a writer agent. Produce clear, technically accurate content. "
                "When revising, incorporate reviewer feedback explicitly."
            ),
        )
        plan.agents.insert(0, writer)
        return writer.name

    def _ensure_reviewer(self, plan: TeamPlan) -> str:
        for spec in plan.agents:
            if spec.name == "reviewer":
                return spec.name

        for spec in plan.agents:
            role_text = f"{spec.name} {spec.duty}".lower()
            identity_text = spec.base_identity.lower()
            reviewer_like = self._is_reviewer_text(role_text) or self._is_reviewer_text(identity_text)
            writer_like = self._is_writer_text(role_text) or self._is_writer_text(identity_text)
            if reviewer_like and not writer_like:
                return spec.name

        reviewer = AgentSpec(
            name="reviewer",
            duty="Review output quality and provide actionable revision feedback.",
            base_identity=(
                "You are a strict reviewer agent. Evaluate structure, accuracy, and clarity. "
                "Return concrete revision points until quality is publication-ready."
            ),
        )
        plan.agents.append(reviewer)
        return reviewer.name

    def _ensure_parallel_structure(self, plan: TeamPlan) -> None:
        synthesizer_name = self._ensure_synthesizer(plan)
        self._ensure_parallel_branches(plan, synthesizer_name, minimum=2)
        self._reorder_parallel_execution(plan, synthesizer_name)

    def _ensure_synthesizer(self, plan: TeamPlan) -> str:
        for spec in plan.agents:
            role_text = f"{spec.name} {spec.duty} {spec.base_identity}".lower()
            if any(
                word in role_text
                for word in ("synth", "merge", "aggregat", "summary", "summarize", "汇总", "整合")
            ):
                return spec.name

        synthesizer = AgentSpec(
            name="synthesizer",
            duty="Merge multiple parallel branch outputs into one coherent final result.",
            base_identity=(
                "You are a synthesizer agent. Merge parallel branch outputs, remove duplication, "
                "resolve conflicts, and produce one high-quality final answer."
            ),
        )
        plan.agents.append(synthesizer)
        return synthesizer.name

    def _ensure_parallel_branches(self, plan: TeamPlan, synthesizer_name: str, minimum: int) -> None:
        def _branch_names() -> list[str]:
            return [spec.name for spec in plan.agents if spec.name != synthesizer_name]

        while len(_branch_names()) < minimum:
            branch_index = len(_branch_names()) + 1
            candidate = f"researcher_{branch_index}"
            existing = {spec.name for spec in plan.agents}
            if candidate in existing:
                suffix = 1
                while f"{candidate}_{suffix}" in existing:
                    suffix += 1
                candidate = f"{candidate}_{suffix}"

            plan.agents.insert(
                0,
                AgentSpec(
                    name=candidate,
                    duty="Research one distinct perspective in the parallel stage.",
                    base_identity=(
                        "You are a research branch agent. Provide one focused perspective with "
                        "clear evidence and concise findings."
                    ),
                ),
            )

    def _reorder_parallel_execution(self, plan: TeamPlan, synthesizer_name: str) -> None:
        agent_names = [spec.name for spec in plan.agents]
        branch_names = [name for name in agent_names if name != synthesizer_name]

        order = [name for name in plan.execution_order if name in agent_names and name != synthesizer_name]
        for name in branch_names:
            if name not in order:
                order.append(name)
        order.append(synthesizer_name)
        plan.execution_order = order

    def _strip_web_search_from_agent(self, plan: TeamPlan, agent_name: str) -> None:
        for spec in plan.agents:
            if spec.name != agent_name:
                continue
            if not spec.native_tools:
                return
            spec.native_tools = [
                item
                for item in spec.native_tools
                if str(item.get("name", "")).strip().lower() != "web_search"
            ]
            return

    def _ensure_native_web_search_for_agent(self, plan: TeamPlan, agent_name: str) -> None:
        for spec in plan.agents:
            if spec.name != agent_name:
                continue
            current = list(spec.native_tools or [])
            has_web_search = any(
                str(item.get("name", "")).strip().lower() == "web_search"
                for item in current
                if isinstance(item, dict)
            )
            if has_web_search:
                return
            current.append(
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 5,
                }
            )
            spec.native_tools = current
            return

    def _is_writer_text(self, text: str) -> bool:
        return any(word in text for word in ("writer", "write", "author", "撰写", "写作", "作者"))

    def _is_reviewer_text(self, text: str) -> bool:
        return any(
            word in text for word in ("review", "reviewer", "critic", "quality", "审核", "评审", "审校")
        )
