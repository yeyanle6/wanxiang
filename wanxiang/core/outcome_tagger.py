"""Outcome tagger — classify a completed run into one of:

  success          — final_status was success AND no infra/gap errors in events
  infra_error      — an INFRA_FAILURE_KEYWORD matched in an error event
  capability_gap   — a CAPABILITY_GAP_KEYWORD matched in an error event
  timeout          — timed out at either run or tool level
  partial          — final_status success but some error signal was present

Pure classification logic. Reads events, returns a string label. No I/O.
Reuses the keyword tuples that trace_mining.py already uses so there's one
canonical taxonomy.
"""
from __future__ import annotations

from typing import Any, Iterable

from .trace_mining import CAPABILITY_GAP_KEYWORDS, INFRA_FAILURE_KEYWORDS

TIMEOUT_KEYWORDS: tuple[str, ...] = (
    "TimeoutError",
    "timed out",
    "Tool timed out",
    "LLM call exceeded",
    "sandbox timed out",
)

OUTCOME_SUCCESS = "success"
OUTCOME_INFRA_ERROR = "infra_error"
OUTCOME_CAPABILITY_GAP = "capability_gap"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_PARTIAL = "partial"
OUTCOME_UNKNOWN = "unknown"


def tag_run(events: Iterable[dict[str, Any]], final_status: str | None) -> str:
    """Return the outcome label for a completed run.

    Decision order (first match wins):
      1. Any timeout signal anywhere       → timeout
      2. Any infra-failure keyword         → infra_error
      3. Any capability-gap keyword        → capability_gap
      4. final_status == 'success'         → success
      5. Otherwise                         → unknown
    """
    error_strings = list(_iter_error_strings(events))

    if _matches_any(error_strings, TIMEOUT_KEYWORDS):
        return OUTCOME_TIMEOUT
    if _matches_any(error_strings, INFRA_FAILURE_KEYWORDS):
        return OUTCOME_INFRA_ERROR
    if _matches_any(error_strings, CAPABILITY_GAP_KEYWORDS):
        return OUTCOME_CAPABILITY_GAP

    status = (final_status or "").strip().lower()
    if status == "success":
        # Success final status but some non-matching error text → partial.
        return OUTCOME_PARTIAL if error_strings else OUTCOME_SUCCESS

    # review_loop exhausted max_iterations with reviewer still unhappy →
    # content was produced but below target quality. This is the 'partial'
    # bucket, not 'unknown'. Without this mapping graduation judges that
    # read review_loop success rates would see most of them as 'unknown'.
    if status == "needs_revision":
        return OUTCOME_PARTIAL

    return OUTCOME_UNKNOWN


def _iter_error_strings(events: Iterable[dict[str, Any]]) -> Iterable[str]:
    for event in events:
        if not isinstance(event, dict):
            continue
        etype = str(event.get("type", ""))
        data = event.get("data")
        if not isinstance(data, dict):
            continue

        # Direct error field on any event.
        err = data.get("error")
        if isinstance(err, str) and err.strip():
            yield err

        # tool_completed with success=False → error text is in content.
        if etype == "tool_completed" and data.get("success") is False:
            content = data.get("content") or data.get("content_preview") or ""
            if isinstance(content, str) and content.strip():
                yield content

        # agent_completed with status=error → error text is top-level content.
        if etype == "agent_completed" and str(data.get("status", "")).lower() == "error":
            content = data.get("content") or data.get("content_preview") or ""
            if isinstance(content, str) and content.strip():
                yield content

        # run_completed carries a trace array; the actual error message
        # lives in trace[].content where trace[].status == "error".
        # final_status of 'error' without this traversal would land as unknown.
        if etype == "run_completed" and str(data.get("final_status", "")).lower() == "error":
            for entry in data.get("trace") or []:
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("status", "")).lower() != "error":
                    continue
                content = entry.get("content")
                if isinstance(content, str) and content.strip():
                    yield content

        # run_started on an init-failed run carries the failure hint in
        # plan.rationale (e.g., "run initialization failed: RuntimeError").
        if etype == "run_started":
            plan = data.get("plan")
            if isinstance(plan, dict):
                rationale = plan.get("rationale")
                if isinstance(rationale, str) and "fail" in rationale.lower():
                    yield rationale


def _matches_any(haystacks: list[str], needles: tuple[str, ...]) -> bool:
    lower_needles = [n.lower() for n in needles]
    for hay in haystacks:
        hay_lower = hay.lower()
        for needle in lower_needles:
            if needle in hay_lower:
                return True
    return False
