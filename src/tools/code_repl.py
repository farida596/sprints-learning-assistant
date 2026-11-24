"""Sandboxed Python REPL tool for learner code assessment."""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from typing import Any

from src.logic import unable_to_verify

BLOCKED_IMPORTS = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "ctypes", "importlib",
    "multiprocessing", "threading", "signal", "resource", "requests", "urllib",
    "http", "ftplib", "smtplib", "pickle", "builtins", "inspect", "gc",
})
BLOCKED_CALLS = frozenset({"open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
                           "exit", "quit", "globals", "locals", "vars", "getattr", "setattr", "delattr"})
MAX_OUTPUT_CHARS = 4000
DEFAULT_TIMEOUT_S = 5.0


class ReplToolError(RuntimeError):
    """Raised when the REPL infrastructure itself fails."""


def find_blocked_operation(code: str) -> str | None:
    """Return a reason if code contains a blocked operation."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BLOCKED_IMPORTS:
                    return f"import of '{alias.name}' is blocked"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in BLOCKED_IMPORTS:
                return f"import from '{node.module}' is blocked"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
            return f"call to '{node.func.id}()' is blocked"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__") and node.attr not in {"__name__", "__init__", "__str__", "__repr__", "__len__"}:
            return f"access to dunder attribute '{node.attr}' is blocked"
    return None


def _limits(timeout_s: float):
    """Return a pre-execution function that applies POSIX resource limits."""
    def apply() -> None:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (int(timeout_s) + 1, int(timeout_s) + 2))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    return apply


def _result(**kw: Any) -> dict[str, Any]:
    base = {"status": "executed", "stdout": "", "stderr": "", "runtime_error": False,
            "timed_out": False, "blocked": False, "exit_code": 0}
    base.update(kw)
    return base


def run_python_snippet(code: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """Execute Python in a separate process and capture stdout and runtime errors."""
    if not isinstance(code, str) or not code.strip():
        return _result(stderr="EmptySubmission: no code was provided", runtime_error=True, exit_code=1)
    try:
        reason = find_blocked_operation(code)
    except SyntaxError as exc:
        return _result(stderr=f"SyntaxError: {exc.msg} (line {exc.lineno})", runtime_error=True, exit_code=1)
    if reason:
        return _result(stderr=f"SecurityError: {reason}", runtime_error=True, blocked=True, exit_code=1)
    try:
        with tempfile.TemporaryDirectory(prefix="sprints_repl_") as workdir:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", "-c", code],
                capture_output=True, text=True, timeout=timeout_s, cwd=workdir,
                env={"PYTHONIOENCODING": "utf-8", "PATH": os.defpath},
                preexec_fn=_limits(timeout_s) if os.name == "posix" else None,
            )
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        return _result(stdout=out[:MAX_OUTPUT_CHARS], stderr=f"TimeoutError: exceeded {timeout_s}s",
                       runtime_error=True, timed_out=True, exit_code=-9)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise ReplToolError(f"{type(exc).__name__}: {exc}") from exc
    return _result(stdout=proc.stdout[:MAX_OUTPUT_CHARS], stderr=proc.stderr[:MAX_OUTPUT_CHARS],
                   runtime_error=proc.returncode != 0, exit_code=proc.returncode)


def safe_run_python(code: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """Return a structured failure if the sandbox infrastructure throws."""
    try:
        return run_python_snippet(code, timeout_s)
    except Exception as exc:
        return unable_to_verify("python_repl", f"{type(exc).__name__}: {exc}")


def get_repl_tool():
    """Return the LangChain tool for running learner Python code."""
    import json
    from langchain_core.tools import tool

    @tool
    def run_python_code(code: str) -> str:
        """Run a learner Python snippet in a sandbox and return JSON execution results."""
        return json.dumps(safe_run_python(code))

    return run_python_code
