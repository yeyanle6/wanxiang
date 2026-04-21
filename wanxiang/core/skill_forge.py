"""SkillForge — orchestrates a SynthesizerAgent + SandboxExecutor to
synthesize new registry tools at runtime.

Pipeline per requirement:
  1. Ask the synthesizer to emit a JSON spec with handler_code +
     test_code + tool_spec metadata.
  2. Parse that JSON (with a markdown-code-block fallback).
  3. Hand handler_code + test_code to the sandbox and run pytest.
  4. Sandbox passes → exec() the handler_code in an isolated namespace,
     wrap the resulting callable in a ToolSpec, register it with the
     ToolRegistry. Return ForgeResult(success=True, registered=True).
  5. Sandbox fails → feed the error back to the synthesizer as an
     additional user turn. Repeat up to max_retries times.

Every attempt (good or bad) is captured in ForgeResult.attempts so
trace mining / UI can visualize the loop.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import BaseAgent
from .message import Message, MessageStatus
from .sandbox import SandboxExecutor, SandboxResult
from .tier import TierManager
from .tools import ToolRegistry, ToolSpec


DEFAULT_MAX_RETRIES = 3


@dataclass(slots=True)
class ForgeAttempt:
    """One iteration of the synthesizer → sandbox loop."""

    attempt_number: int
    raw_response: str
    tool_name: str | None
    description: str | None
    input_schema: dict[str, Any] | None
    handler_code: str | None
    test_code: str | None
    sandbox_result: SandboxResult | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "raw_response": self.raw_response,
            "tool_name": self.tool_name,
            "description": self.description,
            "input_schema": self.input_schema,
            "handler_code": self.handler_code,
            "test_code": self.test_code,
            "sandbox_result": self.sandbox_result.to_dict()
            if self.sandbox_result
            else None,
            "error": self.error,
        }


@dataclass(slots=True)
class ForgeResult:
    success: bool
    requirement: str
    tool_spec: ToolSpec | None
    attempts: list[ForgeAttempt] = field(default_factory=list)
    registered: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "requirement": self.requirement,
            "tool_name": self.tool_spec.name if self.tool_spec else None,
            "attempts": [a.to_dict() for a in self.attempts],
            "registered": self.registered,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = (
    "tool_name",
    "description",
    "input_schema",
    "handler_code",
    "test_code",
)


def parse_synthesizer_response(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """Try hard to extract a SynthSpec JSON from the LLM's reply.

    Returns (data, error). Exactly one of the two is non-None.
    """
    if not raw or not raw.strip():
        return None, "Synthesizer returned empty output."

    text = raw.strip()
    candidates: list[str] = []

    # 1. Raw text may itself be JSON.
    candidates.append(text)

    # 2. ```json ... ``` fenced block.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1).strip())

    # 3. Greedy {...} match (last resort — multi-line).
    brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace:
        candidates.append(brace.group(0))

    parsed: dict[str, Any] | None = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            parsed = value
            break

    if parsed is None:
        return None, (
            "Synthesizer output is not valid JSON. Reply with a single JSON "
            "object matching the required schema; do not wrap it in prose."
        )

    missing = [key for key in _REQUIRED_KEYS if key not in parsed]
    if missing:
        return None, (
            f"Synthesizer output missing required fields: {missing}. "
            f"Expected keys: {list(_REQUIRED_KEYS)}."
        )

    if not isinstance(parsed.get("input_schema"), dict):
        return None, "Field 'input_schema' must be a JSON object."
    if not isinstance(parsed.get("handler_code"), str):
        return None, "Field 'handler_code' must be a string containing Python source."
    if not isinstance(parsed.get("test_code"), str):
        return None, "Field 'test_code' must be a string containing pytest source."
    if not str(parsed.get("tool_name", "")).strip():
        return None, "Field 'tool_name' must be a non-empty string."
    if not str(parsed.get("description", "")).strip():
        return None, "Field 'description' must be a non-empty string."

    return parsed, None


def materialize_handler(handler_code: str, tool_name: str) -> Any:
    """exec() the handler_code in an isolated namespace and return the callable.

    Convention: handler_code must define either a function named `handler`
    or a function named exactly `tool_name`. We prefer `handler` (which is
    what our test harness imports) so that the same code passes sandbox
    pytest AND can be bound to a ToolSpec here.
    """
    namespace: dict[str, Any] = {}
    exec(handler_code, namespace)  # noqa: S102 — code came through sandbox pytest gate
    for key in ("handler", tool_name):
        candidate = namespace.get(key)
        if callable(candidate):
            return candidate
    raise RuntimeError(
        f"handler_code must define a callable named 'handler' or '{tool_name}'."
    )


# ---------------------------------------------------------------------------
# SkillForge
# ---------------------------------------------------------------------------


class SkillForge:
    def __init__(
        self,
        *,
        sandbox: SandboxExecutor,
        registry: ToolRegistry,
        synthesizer: BaseAgent,
        max_retries: int = DEFAULT_MAX_RETRIES,
        tier_manager: TierManager | None = None,
        skills_dir: Path | None = None,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self.sandbox = sandbox
        self.registry = registry
        self.synthesizer = synthesizer
        self.max_retries = max_retries
        self.tier_manager = tier_manager
        self.skills_dir = skills_dir
        self.logger = logging.getLogger("wanxiang.skill_forge")

    async def forge(self, requirement: str) -> ForgeResult:
        cleaned = (requirement or "").strip()
        if not cleaned:
            return ForgeResult(
                success=False,
                requirement=requirement,
                tool_spec=None,
                error="Requirement must be a non-empty string.",
            )

        result = ForgeResult(success=False, requirement=cleaned, tool_spec=None)
        feedback: str | None = None

        for attempt_number in range(1, self.max_retries + 1):
            raw_response = await self._invoke_synthesizer(cleaned, feedback)

            attempt = ForgeAttempt(
                attempt_number=attempt_number,
                raw_response=raw_response,
                tool_name=None,
                description=None,
                input_schema=None,
                handler_code=None,
                test_code=None,
                sandbox_result=None,
            )

            parsed, parse_error = parse_synthesizer_response(raw_response)
            if parsed is None:
                attempt.error = parse_error
                result.attempts.append(attempt)
                feedback = (
                    "Previous response could not be parsed.\n"
                    f"Error: {parse_error}\n"
                    "Return a single JSON object; do not include any prose."
                )
                self.logger.info(
                    "SkillForge attempt %d: parse error — %s",
                    attempt_number,
                    parse_error,
                )
                continue

            attempt.tool_name = str(parsed["tool_name"]).strip()
            attempt.description = str(parsed["description"]).strip()
            attempt.input_schema = parsed["input_schema"]
            attempt.handler_code = parsed["handler_code"]
            attempt.test_code = parsed["test_code"]

            # Name collision with an existing registered tool is a hard
            # fail — forge must not silently overwrite or shadow.
            if self.registry.get(attempt.tool_name) is not None:
                attempt.error = (
                    f"Tool '{attempt.tool_name}' is already registered. "
                    "Pick a different name that does not collide with builtin/MCP tools."
                )
                result.attempts.append(attempt)
                feedback = (
                    f"The tool name '{attempt.tool_name}' is already taken. "
                    "Choose a different, more specific name and regenerate."
                )
                self.logger.info(
                    "SkillForge attempt %d: name collision on %s",
                    attempt_number,
                    attempt.tool_name,
                )
                continue

            sandbox_result = await self.sandbox.execute(
                attempt.handler_code, attempt.test_code
            )
            attempt.sandbox_result = sandbox_result

            if not sandbox_result.passed:
                attempt.error = sandbox_result.error or "sandbox did not pass"
                result.attempts.append(attempt)
                feedback = self._build_feedback(sandbox_result)
                self.logger.info(
                    "SkillForge attempt %d: sandbox not passed — %s",
                    attempt_number,
                    attempt.error,
                )
                continue

            # Success path — bind handler and register.
            try:
                handler_fn = materialize_handler(
                    attempt.handler_code, attempt.tool_name
                )
            except Exception as exc:
                attempt.error = (
                    f"handler materialization failed: {type(exc).__name__}: {exc}"
                )
                result.attempts.append(attempt)
                feedback = (
                    "The handler code passed pytest but could not be loaded "
                    f"at runtime: {type(exc).__name__}: {exc}. Ensure you "
                    "define a top-level function named 'handler'."
                )
                self.logger.warning(
                    "SkillForge attempt %d: materialization failed — %s",
                    attempt_number,
                    exc,
                )
                continue

            tool_spec = ToolSpec(
                name=attempt.tool_name,
                description=attempt.description,
                input_schema=attempt.input_schema,
                handler=handler_fn,
                group="synthesized",
            )
            try:
                self.registry.register(tool_spec)
            except ValueError as exc:
                attempt.error = f"registration failed: {exc}"
                result.attempts.append(attempt)
                feedback = (
                    f"Registration rejected the spec: {exc}. Adjust and regenerate."
                )
                continue

            if self.tier_manager is not None:
                self.tier_manager.initialize_tool(tool_spec.name, 0)

            self._persist_skill(attempt, tool_spec)

            result.attempts.append(attempt)
            result.tool_spec = tool_spec
            result.registered = True
            result.success = True
            self.logger.info(
                "SkillForge succeeded on attempt %d: registered '%s'",
                attempt_number,
                tool_spec.name,
            )
            return result

        result.error = (
            f"Exceeded max_retries={self.max_retries} without producing a "
            "passing, registerable tool."
        )
        return result

    def _persist_skill(self, attempt: ForgeAttempt, tool_spec: ToolSpec) -> None:
        """Write skills/{name}.json and skills/{name}.py if skills_dir is configured."""
        if self.skills_dir is None:
            return
        try:
            self.skills_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "tool_name": tool_spec.name,
                "description": tool_spec.description,
                "input_schema": tool_spec.input_schema,
                "handler_code": attempt.handler_code or "",
                "tier_level": 0,
                "approved": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            json_path = self.skills_dir / f"{tool_spec.name}.json"
            json_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            py_path = self.skills_dir / f"{tool_spec.name}.py"
            py_path.write_text(
                f"# Auto-generated by SkillForge — approve via POST /api/skills/{tool_spec.name}/approve\n"
                f"{attempt.handler_code or ''}",
                encoding="utf-8",
            )
            self.logger.info("Persisted skill '%s' to %s", tool_spec.name, self.skills_dir)
        except Exception:
            self.logger.exception(
                "Failed to persist skill '%s'; tool remains registered in-memory",
                tool_spec.name,
            )

    async def _invoke_synthesizer(
        self, requirement: str, feedback: str | None
    ) -> str:
        """Run one synthesizer turn. Returns the raw Message content."""
        intent = "synthesize a new registry tool"
        if feedback:
            content = (
                f"Requirement:\n{requirement}\n\n"
                f"Feedback on your previous attempt:\n{feedback}\n\n"
                "Regenerate the JSON spec, fixing the issue above."
            )
        else:
            content = f"Requirement:\n{requirement}"
        message = Message(
            intent=intent,
            content=content,
            sender="skill_forge",
            status=MessageStatus.SUCCESS,
        )
        reply = await self.synthesizer.execute(message)
        return reply.content or ""

    def _build_feedback(self, sandbox_result: SandboxResult) -> str:
        parts: list[str] = []
        if sandbox_result.timed_out:
            parts.append(
                "Your test timed out. Avoid infinite loops and unbounded "
                "computation; keep tests quick."
            )
        if sandbox_result.error:
            parts.append(f"Sandbox summary: {sandbox_result.error}")
        if sandbox_result.stderr:
            parts.append(f"Pytest stderr:\n{sandbox_result.stderr}")
        if sandbox_result.stdout:
            parts.append(f"Pytest stdout:\n{sandbox_result.stdout}")
        if not parts:
            parts.append(
                "Sandbox did not pass but produced no diagnostic output."
            )
        parts.append(
            "Regenerate the JSON spec with handler_code and test_code that "
            "address the failure above."
        )
        return "\n\n".join(parts)
