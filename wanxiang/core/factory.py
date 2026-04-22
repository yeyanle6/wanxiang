from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any

from .agent import AgentConfig, BaseAgent
from .llm_client import DEFAULT_MODEL, LLMClient
from .plan import AgentSpec, SynthesisRequest, TeamPlan  # re-exported for callers
from .policies import PlanningPolicies
from .tools import ToolRegistry

if False:  # typing only
    from .skill_forge import ForgeResult, SkillForge  # noqa: F401

# Re-export so existing `from .factory import AgentSpec, TeamPlan` keep working.
__all__ = ["AgentFactory", "AgentSpec", "SynthesisRequest", "TeamPlan"]

SUPPORTED_WORKFLOWS = {"pipeline", "parallel", "review_loop"}
ToolEventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


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
        skill_forge: Any = None,
        usage_recorder: Any = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tool_registry = tool_registry
        self.llm_mode = llm_mode
        self.skill_forge = skill_forge
        self.usage_recorder = usage_recorder
        self.synthesis_log: list[dict[str, Any]] = []
        self.client = LLMClient(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            api_key=self.api_key,
            mode=self.llm_mode,
            usage_recorder=usage_recorder,
        )
        self.logger = logging.getLogger("wanxiang.factory")
        self.policies = PlanningPolicies(
            tool_registry=self.tool_registry,
            logger=self.logger,
        )

    async def create_team_probe(
        self,
        task: str,
        *,
        conversation_context: str | None = None,
    ) -> TeamPlan:
        """Zero-LLM bypass that returns a hardcoded 1-agent pipeline plan.

        For autoschool L0/L1 probe tasks where the review-loop overhead
        dwarfs task complexity. Skips the Director LLM call and the
        policy layer's reviewer injection entirely — 1 LLM call per run
        instead of the normal 3+.

        When `conversation_context` is provided (multi-turn dialogue mode),
        the responder gains a different identity: it can choose to ask a
        clarifying question by prefixing its reply with NEEDS_CLARIFICATION:
        and the conversation layer will transition to awaiting_user.
        Without context, the responder behaves as before (direct answer).
        """
        cleaned_task = task.strip()
        if not cleaned_task:
            raise ValueError("Task must be non-empty.")

        ctx = (conversation_context or "").strip()
        if ctx:
            base_identity = (
                "You are a conversational responder in a multi-turn dialogue. "
                "Prior turns are shown below. Decide whether the user's latest "
                "message has enough information for you to act, or whether a "
                "clarifying question is needed.\n\n"
                "If you need more info, reply with a line that STARTS EXACTLY with "
                "the literal token 'NEEDS_CLARIFICATION:' followed by your question. "
                "Example: 'NEEDS_CLARIFICATION: which Python library do you prefer?'\n\n"
                "If you have enough information, just produce the concise answer. "
                "Do not invoke tools unless strictly required by the task.\n\n"
                "--- CONVERSATION HISTORY ---\n"
                f"{ctx}\n"
                "--- END HISTORY ---"
            )
            rationale = "probe-bypass with conversation context (dialogue mode)"
        else:
            base_identity = (
                "You are a direct responder. Produce the exact requested output "
                "concisely. Do not ask for clarification. Do not invoke tools "
                "unless strictly required by the task."
            )
            rationale = "probe-bypass: Director LLM + policies skipped"

        agent = AgentSpec(
            name="responder",
            duty="Answer the probe task directly.",
            base_identity=base_identity,
        )
        plan = TeamPlan(
            agents=[agent],
            workflow="pipeline",
            execution_order=["responder"],
            max_iterations=1,
            rationale=rationale,
        )
        self.logger.info(
            "Created probe team (1 agent, pipeline, %s)",
            "dialogue mode" if ctx else "no context",
        )
        return plan

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

        if plan.needs_synthesis:
            await self._run_synthesis_stage(plan)

        self.logger.info(
            "Created team: workflow=%s agents=%s mode=%s synthesis=%d",
            plan.workflow,
            [agent.name for agent in plan.agents],
            effective_mode,
            len(plan.needs_synthesis),
        )
        return plan

    async def _run_synthesis_stage(self, plan: TeamPlan) -> None:
        if self.skill_forge is None:
            self.logger.warning(
                "Plan requested synthesis but no SkillForge is attached; "
                "skipping %d synthesis request(s).",
                len(plan.needs_synthesis),
            )
            self.synthesis_log.extend(
                {
                    "requirement": req.requirement,
                    "success": False,
                    "error": "no SkillForge attached",
                }
                for req in plan.needs_synthesis
            )
            return

        for request in plan.needs_synthesis:
            try:
                result = await self.skill_forge.forge(request.requirement)
            except Exception:
                self.logger.exception(
                    "SkillForge raised on requirement: %s", request.requirement
                )
                self.synthesis_log.append(
                    {
                        "requirement": request.requirement,
                        "suggested_name": request.suggested_name,
                        "success": False,
                        "error": "forge raised an exception",
                    }
                )
                continue

            entry: dict[str, Any] = {
                "requirement": request.requirement,
                "suggested_name": request.suggested_name,
                "success": bool(result.success),
                "registered": bool(result.registered),
                "attempts": len(result.attempts),
            }
            if not result.success or not result.registered or result.tool_spec is None:
                entry["error"] = result.error or "forge did not register a tool"
                self.logger.info(
                    "Synthesis failed for '%s': %s",
                    request.requirement,
                    entry["error"],
                )
                self.synthesis_log.append(entry)
                continue

            tool_name = result.tool_spec.name
            entry["tool_name"] = tool_name
            self.synthesis_log.append(entry)

            target = request.suggested_name or tool_name
            attached = False
            for spec in plan.agents:
                if target and target in (spec.allowed_tools or []):
                    if tool_name != target:
                        spec.allowed_tools = [
                            tool_name if t == target else t
                            for t in spec.allowed_tools
                        ]
                    elif tool_name not in spec.allowed_tools:
                        spec.allowed_tools.append(tool_name)
                    attached = True

            if not attached:
                self.logger.info(
                    "Synthesized tool '%s' was registered but no agent "
                    "referenced it in allowed_tools; it remains globally "
                    "available in the registry.",
                    tool_name,
                )

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
                usage_recorder=self.usage_recorder,
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
        available_tools = self._format_tools_for_planner()
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
            "\nAvailable registry tools (local + MCP):\n"
            f"{available_tools}\n"
            "\nAvailable native tools (Claude API, server-side):\n"
            f"{available_native_tools}\n\n"
            "Tool assignment rules:\n"
            "- Assign allowed_tools only when needed, and only from available tools.\n"
            "- Keep allowed_tools empty for agents that do not need tools.\n"
            "- IMPORTANT: If `web_search` appears in the registry tools list above, "
            "assign it via allowed_tools (e.g. \"allowed_tools\": [\"web_search\"]). "
            "This works in ALL modes including CLI. Only use native_tools for web_search "
            "if it is NOT listed in the registry tools.\n"
            "- For research/parallel tasks, assign web_search to researcher branches.\n"
            "- For content tasks requiring current facts, writer may use web_search.\n"
            "- Reviewer should not receive web_search.\n"
            "- When a tool has an access restriction (shown as `restricted to [...]`), "
            "assign it only to agents whose name matches one of the listed names. "
            "If the task needs such a tool, name the agent accordingly (e.g., name the "
            "researcher 'researcher' instead of 'analyst').\n"
            "\nRuntime tool synthesis (OPTIONAL):\n"
            "- If the task requires a capability that is clearly NOT satisfiable by "
            "any listed tool, you MAY add a top-level \"needs_synthesis\" array to "
            "request runtime tool creation:\n"
            '  "needs_synthesis": [{"requirement": "what the tool must do",\n'
            '                       "suggested_name": "snake_case_name"}]\n'
            "- The system will try to synthesize and test each tool before execution. "
            "If synthesis succeeds, the tool is attached to whichever agent has "
            "`suggested_name` in its allowed_tools. If it fails, the run proceeds "
            "without that tool — so design the team to degrade gracefully.\n"
            "- Use synthesis only for pure-Python, deterministic capabilities "
            "(numeric conversion, string reshaping, lightweight parsing). Do NOT "
            "request synthesis for tasks needing network, filesystem, or external "
            "services — those belong to MCP servers.\n"
            "- Keep suggested_name unique — do not collide with any listed tool."
        )
        raw = await self.client.generate(messages=messages, system=system_prompt)
        if not raw:
            raise RuntimeError("Factory planner returned no text content.")
        return raw

    def _format_tools_for_planner(self) -> str:
        if self.tool_registry is None:
            return "(none)"
        groups: dict[str, list[Any]] = {}
        for name in self.tool_registry.list_tools():
            spec = self.tool_registry.get(name)
            if spec is None:
                continue
            group_key = spec.group or "builtin"
            groups.setdefault(group_key, []).append(spec)
        if not groups:
            return "(none)"

        ordered_keys = sorted(
            groups.keys(),
            key=lambda k: (0 if k == "builtin" else 1, k),
        )
        lines: list[str] = []
        for key in ordered_keys:
            header = "builtin:" if key == "builtin" else f"{key} (MCP server):"
            lines.append(header)
            for spec in groups[key]:
                restriction = ""
                if spec.allowed_agents:
                    restriction = f" (restricted to {spec.allowed_agents})"
                desc = (spec.description or "").strip().replace("\n", " ")
                if len(desc) > 140:
                    desc = desc[:137] + "..."
                lines.append(f"  - {spec.name}: {desc}{restriction}")
        return "\n".join(lines)

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

    # ---------------------------------------------------------------------------
    # Policy delegation — keep these thin wrappers so existing test call-sites
    # that do `factory._apply_planning_policies(...)` continue to work unchanged.
    # ---------------------------------------------------------------------------

    def _apply_planning_policies(
        self,
        task: str,
        plan: TeamPlan,
        *,
        effective_mode: str | None = None,
    ) -> TeamPlan:
        return self.policies.apply(task, plan, effective_mode=effective_mode)

    def _apply_tool_restrictions(self, plan: TeamPlan) -> TeamPlan:
        return self.policies.apply_tool_restrictions(plan)
