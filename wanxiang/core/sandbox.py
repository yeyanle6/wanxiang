"""Isolated Python code executor for the Skill Synthesizer.

Runs LLM-generated `handler.py` + `test_handler.py` inside a temporary
directory under pytest. The sandbox is process-level — no Docker — so
its guarantees are:

- Filesystem isolation via tempfile.TemporaryDirectory; the child can
  only see files we put there.
- Env scrubbing: only PATH (+ optionally PYTHONPATH) is passed through,
  so ANTHROPIC_API_KEY and any user secrets never reach the child.
- cwd pinned to the tempdir.
- Hard timeout via asyncio.wait_for; the process is killed on timeout
  and its exit awaited so there are no zombies.
- Output capped with the same UTF-8-safe truncator as ToolRegistry.

This is enough for Phase 4.1 (Skill Synthesizer writing pure-Python
handlers). Heavier isolation (Docker / nsjail / seccomp) is a separate
decision if we later accept handlers that import untrusted packages or
perform filesystem writes.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .tools import _safe_truncate_utf8


DEFAULT_SANDBOX_TIMEOUT_S = 30.0
DEFAULT_MAX_OUTPUT_BYTES = 20_000


@dataclass(slots=True)
class SandboxResult:
    success: bool
    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    elapsed_ms: int
    timed_out: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "elapsed_ms": self.elapsed_ms,
            "timed_out": self.timed_out,
            "error": self.error,
        }


class SandboxExecutor:
    """Runs pytest on a handler+test pair inside an ephemeral tempdir."""

    def __init__(
        self,
        *,
        timeout_s: float = DEFAULT_SANDBOX_TIMEOUT_S,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        python_executable: str | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        self.timeout_s = timeout_s
        self.max_output_bytes = max_output_bytes
        self.python_executable = python_executable or sys.executable
        self.logger = logging.getLogger("wanxiang.sandbox")

    async def execute(
        self,
        handler_code: str,
        test_code: str,
        *,
        handler_filename: str = "handler.py",
        test_filename: str = "test_handler.py",
        extra_env: Mapping[str, str] | None = None,
    ) -> SandboxResult:
        """Write the two files into a tempdir and run pytest against them.

        Returns SandboxResult whose `passed` is True only when pytest
        reports all tests passing (exit code 0). `success` reflects the
        sandbox infrastructure itself working (process spawned, exited
        within timeout) and is False on timeout or spawn failure.
        """
        if not handler_code.strip():
            raise ValueError("handler_code must be non-empty")
        if not test_code.strip():
            raise ValueError("test_code must be non-empty")

        started = asyncio.get_running_loop().time()
        tempdir = tempfile.mkdtemp(prefix="wanxiang-sandbox-")
        try:
            handler_path = Path(tempdir) / handler_filename
            test_path = Path(tempdir) / test_filename
            handler_path.write_text(handler_code, encoding="utf-8")
            test_path.write_text(test_code, encoding="utf-8")

            env = self._scrubbed_env(extra_env)

            try:
                process = await asyncio.create_subprocess_exec(
                    self.python_executable,
                    "-m",
                    "pytest",
                    test_filename,
                    "-q",
                    "--no-header",
                    "-p",
                    "no:cacheprovider",
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=tempdir,
                    env=env,
                )
            except Exception as exc:
                elapsed_ms = int((asyncio.get_running_loop().time() - started) * 1000)
                self.logger.exception("Sandbox failed to spawn pytest")
                return SandboxResult(
                    success=False,
                    passed=False,
                    exit_code=-1,
                    stdout="",
                    stderr="",
                    stdout_truncated=False,
                    stderr_truncated=False,
                    elapsed_ms=elapsed_ms,
                    timed_out=False,
                    error=f"spawn failed: {type(exc).__name__}: {exc}",
                )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout_s,
                )
                timed_out = False
            except asyncio.TimeoutError:
                timed_out = True
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                try:
                    stdout_bytes, stderr_bytes = await process.communicate()
                except Exception:
                    stdout_bytes, stderr_bytes = b"", b""

            elapsed_ms = int((asyncio.get_running_loop().time() - started) * 1000)
            exit_code = -1 if process.returncode is None else int(process.returncode)

            stdout, stdout_truncated, _ = _safe_truncate_utf8(
                stdout_bytes.decode("utf-8", errors="replace"),
                self.max_output_bytes,
            )
            stderr, stderr_truncated, _ = _safe_truncate_utf8(
                stderr_bytes.decode("utf-8", errors="replace"),
                self.max_output_bytes,
            )

            if timed_out:
                return SandboxResult(
                    success=False,
                    passed=False,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    stdout_truncated=stdout_truncated,
                    stderr_truncated=stderr_truncated,
                    elapsed_ms=elapsed_ms,
                    timed_out=True,
                    error=f"sandbox timed out after {self.timeout_s:.2f}s",
                )

            # pytest exit codes: 0=passed, 1=failed, 2=usage error, 3=internal,
            # 4=usage misconfig, 5=no tests collected. Only 0 is "passed".
            passed = exit_code == 0
            error: str | None = None
            if not passed:
                if exit_code == 1:
                    error = "pytest: some tests failed"
                elif exit_code == 5:
                    error = "pytest: no tests collected"
                else:
                    error = f"pytest exited with code {exit_code}"

            return SandboxResult(
                success=True,
                passed=passed,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
                elapsed_ms=elapsed_ms,
                timed_out=False,
                error=error,
            )
        finally:
            shutil.rmtree(tempdir, ignore_errors=True)

    def _scrubbed_env(
        self, extra_env: Mapping[str, str] | None = None
    ) -> dict[str, str]:
        """Minimal env for the child process.

        Keep PATH so python / pytest resolve their subprocess tools, and
        preserve PYTHONPATH so pytest can import the test file's module
        graph. Everything else (API keys, user credentials) is dropped.
        """
        env: dict[str, str] = {}
        for key in ("PATH", "PYTHONPATH", "LANG", "LC_ALL", "LC_CTYPE"):
            value = os.environ.get(key)
            if value is not None:
                env[key] = value
        if extra_env:
            for key, value in extra_env.items():
                env[str(key)] = str(value)
        return env
