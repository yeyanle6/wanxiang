from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from typing import Any

DEFAULT_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
VALID_LLM_MODES = {"auto", "api", "cli"}

# Hard timeout for a single LLM call. Applies uniformly to both API and
# CLI backends. Normal LLM calls rarely exceed 60s; 120s gives a 2x
# buffer for occasional slow responses but bounds silent TPM-rate-limit
# holds that would otherwise stall parallel workflows for 15+ minutes.
DEFAULT_LLM_CALL_TIMEOUT_S = 120.0

# Timeout-only retry ladder. If a single LLM call trips the 120s cap,
# sleep N seconds then retry — TPM holds are typically transient (the
# rate-limit window rotates on the minute). Only asyncio.TimeoutError
# triggers retry; real LLM errors (bad JSON, auth failures) pass through.
# Two retries → three attempts total → worst case 3×120 + 30 + 60 = 450s.
DEFAULT_TIMEOUT_RETRY_WAITS_S: tuple[float, ...] = (30.0, 60.0)

# Claude CLI auto-loads user-level MCP servers (Notion / Gmail / etc.).
# When Wanxiang uses `claude -p` as a text-generation backend we don't
# want that — our own tools/loops are separately orchestrated and having
# Claude CLI drive its own MCP round-trips produces unpredictable hangs.
# The workaround is --strict-mcp-config + an empty config file so
# Claude CLI starts with zero MCP servers.
_WANXIANG_EMPTY_MCP_CONFIG_PATH: str | None = None


async def _terminate_subprocess(process: "asyncio.subprocess.Process") -> None:
    """Best-effort kill of a subprocess and reap it.

    Used when outer wait_for cancels a CLI call — we must not leave
    orphaned `claude -p` processes eating TPM quota after timeout.
    """
    if process.returncode is not None:
        return
    try:
        process.kill()
    except ProcessLookupError:
        return
    except Exception:
        pass
    try:
        await process.wait()
    except Exception:
        pass


def _ensure_empty_mcp_config() -> str:
    global _WANXIANG_EMPTY_MCP_CONFIG_PATH
    path = _WANXIANG_EMPTY_MCP_CONFIG_PATH
    if path and os.path.exists(path):
        return path
    fd, new_path = tempfile.mkstemp(
        prefix="wanxiang-empty-mcp-", suffix=".json"
    )
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"mcpServers": {}}, handle)
    _WANXIANG_EMPTY_MCP_CONFIG_PATH = new_path
    return new_path


class LLMClient:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        api_key: str | None = None,
        oauth_token: str | None = None,
        mode: str | None = None,
        llm_call_timeout_s: float = DEFAULT_LLM_CALL_TIMEOUT_S,
        timeout_retry_waits_s: tuple[float, ...] = DEFAULT_TIMEOUT_RETRY_WAITS_S,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.oauth_token = oauth_token or os.getenv("CLAUDE_CODE_OAUTH_TOKEN")
        resolved_mode = (mode or os.getenv("WANXIANG_LLM_MODE") or "auto").strip().lower()
        if resolved_mode not in VALID_LLM_MODES:
            raise ValueError(
                f"Invalid LLM mode '{resolved_mode}'. Expected one of: {', '.join(sorted(VALID_LLM_MODES))}."
            )
        if llm_call_timeout_s <= 0:
            raise ValueError("llm_call_timeout_s must be positive")
        if any(w < 0 for w in timeout_retry_waits_s):
            raise ValueError("timeout_retry_waits_s entries must be non-negative")
        self.mode = resolved_mode
        self.llm_call_timeout_s = llm_call_timeout_s
        self.timeout_retry_waits_s = tuple(timeout_retry_waits_s)
        self._claude_bin = shutil.which("claude")
        self._cli_auth_cache: bool | None = None
        self.logger = logging.getLogger("wanxiang.llm_client")

    async def generate(self, messages: list[dict[str, Any]], system: str | None = None) -> str:
        response = await self.generate_response(messages=messages, system=system)
        text_parts: list[str] = []
        for block in response.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        final_text = "".join(text_parts).strip()
        if not final_text:
            raise RuntimeError("Claude API returned no text content.")
        return final_text

    async def generate_response(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        mode = await self.resolve_mode(require_tools=bool(tools))
        max_attempts = len(self.timeout_retry_waits_s) + 1
        last_exc: asyncio.TimeoutError | None = None

        for attempt_index in range(max_attempts):
            try:
                return await asyncio.wait_for(
                    self._dispatch_generate_response(
                        mode=mode, messages=messages, system=system, tools=tools
                    ),
                    timeout=self.llm_call_timeout_s,
                )
            except asyncio.TimeoutError as exc:
                last_exc = exc
                if attempt_index < len(self.timeout_retry_waits_s):
                    wait_s = self.timeout_retry_waits_s[attempt_index]
                    self.logger.warning(
                        "LLM call timed out after %.0fs "
                        "(attempt %d/%d, mode=%s); retrying in %.0fs",
                        self.llm_call_timeout_s,
                        attempt_index + 1,
                        max_attempts,
                        mode,
                        wait_s,
                    )
                    if wait_s > 0:
                        await asyncio.sleep(wait_s)
                    continue
                # Out of retries — surface as RuntimeError with a stable
                # "LLM call exceeded" prefix so trace_mining can cluster.
                raise RuntimeError(
                    f"LLM call exceeded {self.llm_call_timeout_s:.0f}s timeout "
                    f"after {max_attempts} attempts. "
                    f"Likely sustained TPM rate-limit hold or service degradation. "
                    f"Mode: {mode}, Model: {self.model}"
                ) from exc

        # Defensive: loop must return or raise; this line protects against
        # future refactors silently dropping a branch.
        raise RuntimeError(
            "LLM call exhausted retries without a final result."
        ) from last_exc

    async def _dispatch_generate_response(
        self,
        *,
        mode: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        if mode == "api":
            return await self._generate_response_via_anthropic_api(
                messages=messages,
                system=system,
                tools=tools,
            )
        if tools:
            raise RuntimeError("Use BaseAgent CLI tool loop in cli mode; native tool blocks are API-only.")
        text = await self._generate_via_claude_cli(messages=messages, system=system)
        return {
            "id": "cli-fallback",
            "model": self.model,
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": text}],
        }

    async def resolve_mode(self, *, require_tools: bool = False) -> str:
        if self.mode == "api":
            if not self.api_key:
                raise RuntimeError("LLM mode is 'api' but ANTHROPIC_API_KEY is not configured.")
            return "api"

        if self.mode == "cli":
            if not self._claude_bin:
                raise RuntimeError("LLM mode is 'cli' but `claude` binary is not installed.")
            if not await self._is_cli_authenticated():
                raise RuntimeError("LLM mode is 'cli' but Claude CLI is not authenticated.")
            return "cli"

        # auto mode
        if self.api_key:
            return "api"
        if self._claude_bin and await self._is_cli_authenticated():
            return "cli"

        if require_tools:
            raise RuntimeError(
                "No usable LLM backend for tool use. Configure ANTHROPIC_API_KEY or login Claude CLI."
            )
        raise RuntimeError(
            "No usable LLM backend. Configure ANTHROPIC_API_KEY or login Claude CLI (`claude auth login`)."
        )

    async def _is_cli_authenticated(self) -> bool:
        if self.oauth_token:
            return True
        if self._cli_auth_cache is not None:
            return self._cli_auth_cache
        if not self._claude_bin:
            self._cli_auth_cache = False
            return False

        try:
            process = await asyncio.create_subprocess_exec(
                self._claude_bin,
                "auth",
                "status",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
        except Exception:
            self._cli_auth_cache = False
            return False

        if process.returncode != 0:
            self._cli_auth_cache = False
            return False

        text = stdout.decode("utf-8", errors="replace").strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            self._cli_auth_cache = False
            return False

        self._cli_auth_cache = bool(payload.get("loggedIn"))
        return self._cli_auth_cache

    async def _generate_response_via_anthropic_api(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        import httpx

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        timeout = httpx.Timeout(60.0, connect=20.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(ANTHROPIC_MESSAGES_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()

    async def _generate_via_claude_cli(
        self, messages: list[dict[str, Any]], system: str | None = None
    ) -> str:
        claude_bin = self._claude_bin
        if not claude_bin:
            raise RuntimeError(
                "No auth credentials found for direct API and `claude` CLI is not installed."
            )

        prompt = self._render_cli_prompt(messages)
        if system:
            prompt = f"SYSTEM:\n{system.strip()}\n\n{prompt}"
        empty_mcp_config = _ensure_empty_mcp_config()
        command = [
            claude_bin,
            "-p",
            "--model",
            self.model,
            "--permission-mode",
            "dontAsk",
            "--strict-mcp-config",
            "--mcp-config",
            empty_mcp_config,
        ]

        env = os.environ.copy()
        if self.oauth_token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = self.oauth_token

        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await process.communicate(input=prompt.encode("utf-8"))
        except asyncio.CancelledError:
            # Outer wait_for cancellation. Kill the orphan so it doesn't
            # keep consuming TPM or leave zombies behind. Re-raise so the
            # wait_for path can convert to a timeout RuntimeError.
            await _terminate_subprocess(process)
            raise

        out_text = stdout.decode("utf-8", errors="replace").strip()
        err_text = stderr.decode("utf-8", errors="replace").strip()
        if process.returncode != 0:
            raise RuntimeError(
                "Claude CLI call failed. "
                "Run `claude auth login` or set `CLAUDE_CODE_OAUTH_TOKEN`. "
                f"Details: {err_text or out_text or f'exit={process.returncode}'}"
            )
        if not out_text:
            json_fallback = await self._generate_via_claude_cli_json_fallback(
                claude_bin=claude_bin, prompt=prompt, env=env
            )
            if json_fallback:
                return json_fallback
            if err_text:
                return err_text
            raise RuntimeError("Claude CLI returned empty output.")

        return out_text

    async def _generate_via_claude_cli_json_fallback(
        self, *, claude_bin: str, prompt: str, env: dict[str, str]
    ) -> str:
        fallback_cmd = [
            claude_bin,
            "-p",
            "--model",
            self.model,
            "--permission-mode",
            "dontAsk",
            "--strict-mcp-config",
            "--mcp-config",
            _ensure_empty_mcp_config(),
            "--output-format",
            "json",
        ]
        process = await asyncio.create_subprocess_exec(
            *fallback_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, _ = await process.communicate(input=prompt.encode("utf-8"))
        except asyncio.CancelledError:
            await _terminate_subprocess(process)
            raise
        if process.returncode != 0:
            return ""

        text = stdout.decode("utf-8", errors="replace").strip()
        if not text:
            return ""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text

        if isinstance(payload, dict):
            result = str(payload.get("result", "")).strip()
            if result:
                return result
        return ""

    def _render_cli_prompt(self, messages: list[dict[str, Any]]) -> str:
        if not messages:
            raise ValueError("messages cannot be empty.")

        lines: list[str] = []
        for item in messages:
            role = str(item.get("role", "user")).upper()
            content = item.get("content", "")
            if isinstance(content, list):
                text_blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_blocks.append(str(block.get("text", "")))
                content_text = "\n".join(text_blocks).strip()
            else:
                content_text = str(content).strip()
            lines.append(f"{role}:\n{content_text}")

        return "\n\n".join(lines).strip()
