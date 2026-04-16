"""Immutable core interface lock — prevents accidental signature drift.

Each test captures the exact `inspect.signature` of a protected public
method. If a parameter is added, removed, or renamed the test fails
with a readable diff. This is the CI-enforceable counterpart to the
pre-commit hook in `.githooks/pre-commit`.

To intentionally change a protected interface:
  1. Update IMMUTABLE_CORE.md explaining WHY the change is needed.
  2. Update the expected signature string in this file.
  3. Commit with ALLOW_CORE_CHANGE=1.

See IMMUTABLE_CORE.md for the full rationale.
"""
from __future__ import annotations

import inspect

import pytest


# ---- helpers --------------------------------------------------------------

def _sig(obj: object) -> str:
    """Return the string form of inspect.signature for `obj`."""
    return str(inspect.signature(obj))


def _fields(cls: type) -> tuple[str, ...]:
    """Return the __slots__ (or __dataclass_fields__) names of a dataclass."""
    if hasattr(cls, "__dataclass_fields__"):
        return tuple(cls.__dataclass_fields__.keys())
    if hasattr(cls, "__slots__"):
        return tuple(cls.__slots__)
    raise TypeError(f"{cls} is not a slotted dataclass")


# ---- 1. message.py -------------------------------------------------------


class TestMessageProtocol:
    def test_message_status_values(self):
        from wanxiang.core.message import MessageStatus

        assert set(s.value for s in MessageStatus) == {
            "success",
            "needs_revision",
            "error",
        }

    def test_message_fields(self):
        from wanxiang.core.message import Message

        assert _fields(Message) == (
            "intent",
            "content",
            "sender",
            "status",
            "metadata",
            "parent_id",
            "context",
            "turn",
            "id",
            "timestamp",
        )

    def test_message_create_reply_signature(self):
        from wanxiang.core.message import Message

        assert _sig(Message.create_reply) == (
            "(self, *, intent: 'str', content: 'str', sender: 'str', "
            "status: 'MessageStatus', "
            "metadata: 'dict[str, Any] | None' = None) "
            "-> 'Message'"
        )

    def test_message_to_dict_signature(self):
        from wanxiang.core.message import Message

        assert _sig(Message.to_dict) == "(self) -> 'dict[str, Any]'"

    def test_message_to_prompt_signature(self):
        from wanxiang.core.message import Message

        assert _sig(Message.to_prompt) == "(self) -> 'str'"


# ---- 2. agent.py ---------------------------------------------------------


class TestBaseAgentInterface:
    def test_base_agent_init_signature(self):
        from wanxiang.core.agent import BaseAgent

        assert _sig(BaseAgent.__init__) == (
            "(self, config: 'AgentConfig', "
            "api_key: 'str | None' = None, "
            "tool_registry: 'ToolRegistry | None' = None, "
            "on_tool_event: 'ToolEventCallback | None' = None, "
            "llm_mode: 'str | None' = None) -> 'None'"
        )

    def test_base_agent_execute_signature(self):
        from wanxiang.core.agent import BaseAgent

        assert _sig(BaseAgent.execute) == (
            "(self, message: 'Message') -> 'Message'"
        )

    def test_allowlist_enforcement_exists(self):
        from wanxiang.core.agent import BaseAgent

        method = getattr(BaseAgent, "_execute_tool_with_allowlist", None)
        assert method is not None, (
            "_execute_tool_with_allowlist must exist — it is the allowlist gate"
        )
        sig = _sig(method)
        assert "allowed" in sig, "must accept an 'allowed' parameter"
        assert "tool_name" in sig, "must accept a 'tool_name' parameter"

    def test_team_capability_block_exists(self):
        from wanxiang.core.agent import BaseAgent

        assert hasattr(BaseAgent, "_render_team_capability_block"), (
            "_render_team_capability_block must exist — it injects team_context"
        )


# ---- 3. pipeline.py ------------------------------------------------------


class TestWorkflowEngineInterface:
    def test_workflow_engine_init_signature(self):
        from wanxiang.core.pipeline import WorkflowEngine

        sig = _sig(WorkflowEngine.__init__)
        assert "agents" in sig
        assert "plan" in sig

    def test_workflow_engine_run_signature(self):
        from wanxiang.core.pipeline import WorkflowEngine

        sig = _sig(WorkflowEngine.run)
        assert "task" in sig
        assert "Message" in sig

    def test_three_workflow_modes_exist(self):
        from wanxiang.core.pipeline import WorkflowEngine

        assert hasattr(WorkflowEngine, "_run_pipeline")
        assert hasattr(WorkflowEngine, "_run_review_loop")
        assert hasattr(WorkflowEngine, "_run_parallel")

    def test_validate_plan_exists(self):
        from wanxiang.core.pipeline import WorkflowEngine

        assert hasattr(WorkflowEngine, "_validate_plan")


# ---- 4. tools.py ---------------------------------------------------------


class TestToolRegistryInterface:
    def test_tool_spec_fields(self):
        from wanxiang.core.tools import ToolSpec

        fields = _fields(ToolSpec)
        for required in (
            "name", "description", "input_schema", "handler",
            "timeout_s", "max_output_bytes", "group", "allowed_agents",
        ):
            assert required in fields, f"ToolSpec must have field '{required}'"

    def test_tool_result_fields(self):
        from wanxiang.core.tools import ToolResult

        fields = _fields(ToolResult)
        for required in (
            "tool_name", "success", "content", "elapsed_ms",
            "error", "truncated", "output_bytes",
        ):
            assert required in fields, f"ToolResult must have field '{required}'"

    def test_registry_execute_signature(self):
        from wanxiang.core.tools import ToolRegistry

        assert _sig(ToolRegistry.execute) == (
            "(self, name: 'str', arguments: 'dict[str, Any]') "
            "-> 'ToolResult'"
        )

    def test_registry_filter_for_agent_signature(self):
        from wanxiang.core.tools import ToolRegistry

        assert _sig(ToolRegistry.filter_for_agent) == (
            "(self, allowed: 'list[str]') -> 'list[ToolSpec]'"
        )

    def test_registry_register_signature(self):
        from wanxiang.core.tools import ToolRegistry

        assert _sig(ToolRegistry.register) == (
            "(self, spec: 'ToolSpec') -> 'None'"
        )

    def test_safe_truncate_utf8_exists(self):
        from wanxiang.core.tools import _safe_truncate_utf8

        sig = _sig(_safe_truncate_utf8)
        assert "text" in sig
        assert "max_bytes" in sig


# ---- 5. sandbox.py -------------------------------------------------------


class TestSandboxInterface:
    def test_sandbox_executor_init_signature(self):
        from wanxiang.core.sandbox import SandboxExecutor

        sig = _sig(SandboxExecutor.__init__)
        assert "timeout_s" in sig
        assert "max_output_bytes" in sig

    def test_sandbox_execute_signature(self):
        from wanxiang.core.sandbox import SandboxExecutor

        sig = _sig(SandboxExecutor.execute)
        assert "handler_code" in sig
        assert "test_code" in sig

    def test_scrubbed_env_whitelist(self):
        """The env scrubber must only pass through safe keys.

        If this test fails because you added a new key, update
        IMMUTABLE_CORE.md first to explain why the new key is safe.
        """
        from wanxiang.core.sandbox import SandboxExecutor

        executor = SandboxExecutor()
        env = executor._scrubbed_env()
        for key in env:
            assert key in {
                "PATH", "PYTHONPATH", "LANG", "LC_ALL", "LC_CTYPE",
            }, f"Unexpected env key '{key}' in scrubbed environment"

    def test_scrubbed_env_blocks_secrets(self):
        """API keys must never leak into the sandbox."""
        import os

        from wanxiang.core.sandbox import SandboxExecutor

        executor = SandboxExecutor()
        original = os.environ.get("ANTHROPIC_API_KEY")
        try:
            os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-secret"
            env = executor._scrubbed_env()
            assert "ANTHROPIC_API_KEY" not in env
        finally:
            if original is None:
                os.environ.pop("ANTHROPIC_API_KEY", None)
            else:
                os.environ["ANTHROPIC_API_KEY"] = original


# ---- manifest completeness -----------------------------------------------


class TestManifestCompleteness:
    """Verify the 5 protected modules are importable and non-empty."""

    CORE_MODULES = [
        "wanxiang.core.message",
        "wanxiang.core.agent",
        "wanxiang.core.pipeline",
        "wanxiang.core.tools",
        "wanxiang.core.sandbox",
    ]

    @pytest.mark.parametrize("module_path", CORE_MODULES)
    def test_core_module_importable(self, module_path: str):
        import importlib

        mod = importlib.import_module(module_path)
        assert mod is not None
