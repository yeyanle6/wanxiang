"""Unit tests for SandboxExecutor.

All tests use pre-written handler/test snippets — no LLM calls. The real
LLM-driven skill synthesis scenarios land with the SkillForge commit.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from wanxiang.core.sandbox import SandboxExecutor


def _run(coro):
    return asyncio.run(coro)


def test_passing_tests_return_success_and_passed() -> None:
    handler = "def add(a, b):\n    return a + b\n"
    test = (
        "from handler import add\n"
        "\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n"
        "\n"
        "def test_add_negative():\n"
        "    assert add(-1, 1) == 0\n"
    )
    sandbox = SandboxExecutor(timeout_s=15.0)
    result = _run(sandbox.execute(handler, test))

    assert result.success is True
    assert result.passed is True
    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.error is None
    assert result.elapsed_ms > 0


def test_failing_tests_return_passed_false_with_pytest_output() -> None:
    handler = "def add(a, b):\n    return a + b + 1  # off by one\n"
    test = (
        "from handler import add\n"
        "\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n"
    )
    sandbox = SandboxExecutor(timeout_s=15.0)
    result = _run(sandbox.execute(handler, test))

    assert result.success is True
    assert result.passed is False
    assert result.exit_code == 1
    assert result.error is not None and "failed" in result.error
    # Pytest traceback/summary should appear somewhere.
    combined = result.stdout + result.stderr
    assert "test_add" in combined


def test_syntax_error_in_handler_yields_collection_failure() -> None:
    handler = "def broken(:\n    return None\n"  # syntax error
    test = (
        "from handler import broken\n"
        "def test_x():\n    assert broken() is None\n"
    )
    sandbox = SandboxExecutor(timeout_s=15.0)
    result = _run(sandbox.execute(handler, test))

    assert result.success is True
    assert result.passed is False
    # pytest collection error exits with a non-zero code (often 2 or 1).
    assert result.exit_code != 0
    combined = result.stdout + result.stderr
    assert "SyntaxError" in combined or "error" in combined.lower()


def test_no_tests_collected_is_reported() -> None:
    handler = "def hi():\n    return 'hi'\n"
    # Test file has no test_ functions.
    test = "from handler import hi\n\nvalue = hi()\n"
    sandbox = SandboxExecutor(timeout_s=15.0)
    result = _run(sandbox.execute(handler, test))

    assert result.success is True
    assert result.passed is False
    assert result.exit_code == 5
    assert "no tests collected" in (result.error or "").lower()


def test_timeout_kills_process_and_marks_timed_out() -> None:
    handler = "def slow():\n    return None\n"
    # Test body sleeps longer than the sandbox timeout.
    test = (
        "import time\n"
        "from handler import slow\n"
        "\n"
        "def test_slow():\n"
        "    time.sleep(5)\n"
        "    assert slow() is None\n"
    )
    sandbox = SandboxExecutor(timeout_s=0.5)
    result = _run(sandbox.execute(handler, test))

    assert result.timed_out is True
    assert result.success is False
    assert result.passed is False
    assert result.error is not None and "timed out" in result.error


def test_env_is_scrubbed_of_sensitive_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stage a secret in the parent env; the sandbox must not expose it to
    # child processes.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("WANXIANG_TEST_SECRET", "do-not-leak")

    handler = "def noop():\n    return None\n"
    # Test inspects its own environment and asserts neither var is present.
    test = (
        "import os\n"
        "\n"
        "def test_env():\n"
        "    assert 'ANTHROPIC_API_KEY' not in os.environ\n"
        "    assert 'WANXIANG_TEST_SECRET' not in os.environ\n"
    )
    sandbox = SandboxExecutor(timeout_s=15.0)
    result = _run(sandbox.execute(handler, test))

    assert result.success is True, (
        f"sandbox itself failed: {result.error} / stderr={result.stderr}"
    )
    assert result.passed is True, (
        f"env scrubbing failed — child saw secrets. "
        f"stdout={result.stdout} stderr={result.stderr}"
    )


def test_extra_env_is_forwarded_to_child() -> None:
    handler = "def noop():\n    return None\n"
    test = (
        "import os\n"
        "\n"
        "def test_has_var():\n"
        "    assert os.environ.get('WANXIANG_ALLOWED') == 'yes'\n"
    )
    sandbox = SandboxExecutor(timeout_s=15.0)
    result = _run(
        sandbox.execute(handler, test, extra_env={"WANXIANG_ALLOWED": "yes"})
    )

    assert result.success is True
    assert result.passed is True


def test_stdout_over_limit_is_truncated() -> None:
    # Write directly to sys.stdout without going through pytest's capture,
    # then fail the test so pytest surfaces the captured output. This is
    # also realistic: when the synthesizer generates a buggy handler,
    # pytest's traceback for a failing test is exactly what we need to
    # truncate.
    handler = "def noop():\n    return None\n"
    big = "X" * 50_000
    test = (
        "import sys\n"
        "\n"
        "def test_flood():\n"
        f"    sys.stdout.write('{big}')\n"
        "    sys.stdout.flush()\n"
        "    assert False, 'force pytest to surface captured stdout'\n"
    )
    sandbox = SandboxExecutor(timeout_s=15.0, max_output_bytes=2_000)
    result = _run(sandbox.execute(handler, test))

    assert result.success is True
    assert result.passed is False  # test was failed on purpose
    assert result.stdout_truncated is True
    assert "Output truncated" in result.stdout


def test_empty_handler_code_raises() -> None:
    sandbox = SandboxExecutor()
    with pytest.raises(ValueError, match="handler_code"):
        _run(sandbox.execute("", "def test_x(): assert True"))


def test_empty_test_code_raises() -> None:
    sandbox = SandboxExecutor()
    with pytest.raises(ValueError, match="test_code"):
        _run(sandbox.execute("def f(): pass", ""))


def test_constructor_validates_arguments() -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        SandboxExecutor(timeout_s=0)
    with pytest.raises(ValueError, match="max_output_bytes"):
        SandboxExecutor(max_output_bytes=0)


def test_tempdir_is_cleaned_up_after_execution() -> None:
    # Spy on tempfile.mkdtemp to capture the path, then assert it's gone.
    import tempfile as _tempfile

    created: list[str] = []
    original_mkdtemp = _tempfile.mkdtemp

    def tracking_mkdtemp(*args, **kwargs):
        path = original_mkdtemp(*args, **kwargs)
        created.append(path)
        return path

    _tempfile.mkdtemp = tracking_mkdtemp  # type: ignore[assignment]
    try:
        sandbox = SandboxExecutor(timeout_s=15.0)
        handler = "def f(): return 1\n"
        test = "from handler import f\ndef test_f(): assert f() == 1\n"
        _run(sandbox.execute(handler, test))
    finally:
        _tempfile.mkdtemp = original_mkdtemp  # type: ignore[assignment]

    assert len(created) == 1
    assert not os.path.exists(created[0]), "sandbox tempdir was not cleaned up"
