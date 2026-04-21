"""Team plan data structures — shared between factory and policies.

Kept in a separate module so policies.py can import AgentSpec / TeamPlan
without creating a circular dependency with factory.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_WORKFLOWS = {"pipeline", "parallel", "review_loop"}


@dataclass(slots=True)
class AgentSpec:
    name: str
    duty: str
    base_identity: str
    persona_prompt: str | None = None
    allowed_tools: list[str] | None = None
    native_tools: list[dict[str, Any]] | None = None
    max_tool_rounds: int | None = None

    def to_agent_config(self):  # -> AgentConfig (imported lazily to avoid cycles)
        from .agent import AgentConfig

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
class SynthesisRequest:
    requirement: str
    suggested_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement": self.requirement,
            "suggested_name": self.suggested_name,
        }


@dataclass(slots=True)
class TeamPlan:
    agents: list[AgentSpec]
    workflow: str
    execution_order: list[str]
    max_iterations: int = 1
    rationale: str = ""
    needs_synthesis: list[SynthesisRequest] = field(default_factory=list)

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

        raw_synthesis = data.get("needs_synthesis")
        synthesis: list[SynthesisRequest] = []
        if isinstance(raw_synthesis, list):
            for item in raw_synthesis:
                if isinstance(item, dict):
                    requirement = str(item.get("requirement", "")).strip()
                    if not requirement:
                        continue
                    suggested_raw = item.get("suggested_name")
                    suggested = str(suggested_raw).strip() if suggested_raw else None
                    synthesis.append(
                        SynthesisRequest(
                            requirement=requirement,
                            suggested_name=suggested or None,
                        )
                    )

        return cls(
            agents=agents,
            workflow=workflow,
            execution_order=order,
            max_iterations=max_iterations,
            rationale=rationale,
            needs_synthesis=synthesis,
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
            "needs_synthesis": [s.to_dict() for s in self.needs_synthesis],
        }
