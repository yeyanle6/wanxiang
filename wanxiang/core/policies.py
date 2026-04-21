"""Planning policies — pure decision and mutation logic for TeamPlan objects.

No I/O, no LLM calls. All methods take a TeamPlan and return one, or
perform in-place mutations. Extracted from AgentFactory so the policy
layer can evolve (and be tested) independently.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .tools import ToolRegistry

from .plan import AgentSpec, TeamPlan


MIN_ROUNDS_FOR_TOOL_USERS = 15

_CONTENT_KEYWORDS = (
    "write", "writer", "blog", "article", "post", "report", "docs",
    "documentation", "copywriting",
    "写", "博客", "文章", "文档", "报告", "稿",
)
_RESEARCH_KEYWORDS = (
    "research", "trend", "latest", "analysis", "market", "report",
    "framework", "study", "2026",
    "研究", "趋势", "最新", "分析", "报告", "框架", "多角度", "多个角度",
)
_WRITER_WORDS = ("writer", "write", "author", "撰写", "写作", "作者")
_REVIEWER_WORDS = ("review", "reviewer", "critic", "quality", "审核", "评审", "审校")
_SYNTHESIZER_WORDS = ("synth", "merge", "aggregat", "summary", "summarize", "汇总", "整合")


class PlanningPolicies:
    """Stateless policy engine for TeamPlan validation and mutation.

    Accepts a ToolRegistry for tool-restriction checks and a logger for
    audit messages. Both are optional — pass None for unit tests.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.tool_registry = tool_registry
        self.logger = logger or logging.getLogger("wanxiang.policies")

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def apply(
        self,
        task: str,
        plan: TeamPlan,
        *,
        effective_mode: str | None = None,
    ) -> TeamPlan:
        """Apply all planning policies to a freshly parsed TeamPlan."""
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
            return self.apply_tool_restrictions(plan)

        if plan.workflow == "parallel":
            self._ensure_parallel_structure(plan)
            if not native_tools_supported:
                self._strip_all_native_tools(plan)
            return self.apply_tool_restrictions(plan)

        if len(plan.agents) < 2:
            self._ensure_reviewer(plan)

        if plan.workflow == "pipeline":
            reviewer_name = self._ensure_reviewer(plan)
            self._strip_web_search_from_agent(plan, reviewer_name)
            if reviewer_name not in plan.execution_order:
                plan.execution_order.append(reviewer_name)

        if not native_tools_supported:
            self._strip_all_native_tools(plan)

        return self.apply_tool_restrictions(plan)

    def apply_tool_restrictions(self, plan: TeamPlan) -> TeamPlan:
        """Strip disallowed tools from each agent and enforce min tool rounds."""
        if self.tool_registry is None:
            return plan
        for spec in plan.agents:
            if not spec.allowed_tools:
                continue
            kept: list[str] = []
            for tool_name in spec.allowed_tools:
                tool_spec = self.tool_registry.get(tool_name)
                if tool_spec is None:
                    kept.append(tool_name)
                    continue
                if tool_spec.is_agent_allowed(spec.name):
                    kept.append(tool_name)
                else:
                    self.logger.warning(
                        "Stripped tool '%s' from agent '%s' (allowed_agents=%s)",
                        tool_name,
                        spec.name,
                        tool_spec.allowed_agents,
                    )
            spec.allowed_tools = kept

        for spec in plan.agents:
            has_tools = bool(spec.allowed_tools) or bool(spec.native_tools)
            if not has_tools:
                continue
            current = spec.max_tool_rounds
            if current is None or current < MIN_ROUNDS_FOR_TOOL_USERS:
                self.logger.info(
                    "Lifting max_tool_rounds for tool-using agent '%s': %s → %d",
                    spec.name,
                    current,
                    MIN_ROUNDS_FOR_TOOL_USERS,
                )
                spec.max_tool_rounds = MIN_ROUNDS_FOR_TOOL_USERS
        return plan

    # ---------------------------------------------------------------------------
    # Classification
    # ---------------------------------------------------------------------------

    def _is_content_task(self, task: str) -> bool:
        text = task.lower()
        return any(word in text for word in _CONTENT_KEYWORDS)

    def _is_research_task(self, task: str) -> bool:
        text = task.lower()
        return any(word in text for word in _RESEARCH_KEYWORDS)

    def _is_writer_text(self, text: str) -> bool:
        return any(word in text for word in _WRITER_WORDS)

    def _is_reviewer_text(self, text: str) -> bool:
        return any(word in text for word in _REVIEWER_WORDS)

    # ---------------------------------------------------------------------------
    # Agent injection
    # ---------------------------------------------------------------------------

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

    # ---------------------------------------------------------------------------
    # Parallel workflow enforcement
    # ---------------------------------------------------------------------------

    def _ensure_parallel_structure(self, plan: TeamPlan) -> None:
        synthesizer_name = self._ensure_synthesizer(plan)
        self._ensure_parallel_branches(plan, synthesizer_name, minimum=2)
        self._reorder_parallel_execution(plan, synthesizer_name)

    def _ensure_synthesizer(self, plan: TeamPlan) -> str:
        for spec in plan.agents:
            role_text = f"{spec.name} {spec.duty} {spec.base_identity}".lower()
            if any(word in role_text for word in _SYNTHESIZER_WORDS):
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

    def _ensure_parallel_branches(
        self, plan: TeamPlan, synthesizer_name: str, minimum: int
    ) -> None:
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
                        "You are a research branch agent. Provide one focused perspective "
                        "with clear evidence and concise findings."
                    ),
                ),
            )

    def _reorder_parallel_execution(self, plan: TeamPlan, synthesizer_name: str) -> None:
        agent_names = [spec.name for spec in plan.agents]
        branch_names = [name for name in agent_names if name != synthesizer_name]
        order = [
            name
            for name in plan.execution_order
            if name in agent_names and name != synthesizer_name
        ]
        for name in branch_names:
            if name not in order:
                order.append(name)
        order.append(synthesizer_name)
        plan.execution_order = order

    # ---------------------------------------------------------------------------
    # Native tool mutations
    # ---------------------------------------------------------------------------

    def _strip_all_native_tools(self, plan: TeamPlan) -> None:
        for spec in plan.agents:
            if spec.native_tools:
                spec.native_tools = []

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
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
            )
            spec.native_tools = current
            return
