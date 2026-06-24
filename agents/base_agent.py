"""
Base agent — Anthropic SDK wrapper with tool-dispatch and post_report pattern.

Every LLM agent in the pipeline:
  1. Receives a system prompt + task message
  2. Calls tools (including post_report) to complete its task
  3. Returns a Report when post_report is called

Usage
-----
class MyAgent(BaseAgent):
    def get_tools(self) -> list[dict]:
        return [MY_TOOL_SCHEMA]

    def call_tool(self, name: str, inputs: dict) -> str:
        if name == "my_tool":
            result = my_function(**inputs)
            return str(result)
        return super().call_tool(name, inputs)   # handles post_report

    def run(self, system_prompt: str, task_msg: str) -> Report:
        return self._run(system_prompt, task_msg, self.get_tools())
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import anthropic

from core.schemas import AgentStatus, ArtifactRef, Report
from core.llm_logger import LLMLogger

log = logging.getLogger(__name__)


# ── post_report tool definition (every agent has this) ────────────────────────

POST_REPORT_SCHEMA: dict = {
    "name": "post_report",
    "description": (
        "Post the final structured report when your task is complete. "
        "Call this EXACTLY ONCE as your last action."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["done", "needs_user_input", "failed"],
                "description": "Completion status.",
            },
            "summary": {
                "type": "string",
                "description": "Brief human-readable summary (1-3 sentences).",
            },
            "produced": {
                "type": "array",
                "description": "Artifact references created by this agent.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id":    {"type": "string"},
                        "type":  {"type": "string"},
                        "store": {"type": "string"},
                    },
                    "required": ["id", "type", "store"],
                },
            },
            "metadata": {
                "type": "object",
                "description": "Agent-specific structured output (model_id, params, etc.).",
            },
            "recommended_next": {
                "type": "string",
                "description": "Optional routing hint for the orchestrator.",
            },
        },
        "required": ["status", "summary"],
    },
}


# ── BaseAgent ─────────────────────────────────────────────────────────────────

class BaseAgent:
    """
    Anthropic SDK wrapper that drives the tool-use loop until post_report is called.

    Subclasses override:
        get_tools()  → list of extra tool schemas
        call_tool()  → dispatch non-post_report tool calls

    The _run() method handles the agentic loop; subclasses expose it as run().
    """

    DEFAULT_MODEL    = "claude-sonnet-4-6"
    MAX_ITERATIONS   = 15
    MAX_TOKENS_REPLY = 4096

    def __init__(
        self,
        model:          Optional[str] = None,
        max_iterations: int = MAX_ITERATIONS,
        api_key:        Optional[str] = None,
        llm_logger:     Optional[LLMLogger] = None,
    ):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set.  "
                "Export it or pass api_key= to the agent constructor."
            )
        self._client      = anthropic.Anthropic(api_key=key)
        self._model       = model or self.DEFAULT_MODEL
        self._max_iter    = max_iterations
        self._llm_logger  = llm_logger

    # ── Override in subclass ──────────────────────────────────────────────────

    def get_tools(self) -> List[dict]:
        """Return extra tool schemas for this agent (post_report is always included)."""
        return []

    def call_tool(self, name: str, inputs: dict) -> str:
        """
        Dispatch a tool call.  Subclass must handle its own tools and call
        super().call_tool() for unrecognised names.
        """
        raise NotImplementedError(f"Tool '{name}' not handled by {self.__class__.__name__}")

    # ── Core loop ─────────────────────────────────────────────────────────────

    def _run(
        self,
        system_prompt: str,
        task_msg: str,
        extra_tools: Optional[List[dict]] = None,
    ) -> Report:
        """
        Drive the tool-use agentic loop until the LLM calls post_report.

        Returns the Report built from the post_report tool call's input.
        """
        tools = [POST_REPORT_SCHEMA] + (extra_tools or self.get_tools())
        messages: List[Dict[str, Any]] = [{"role": "user", "content": task_msg}]

        agent_name = self.__class__.__name__
        _sep = "─" * 72
        log.debug("[%s] %s", agent_name, _sep)
        log.debug("[%s] SYSTEM PROMPT\n%s\n%s", agent_name, system_prompt, _sep)
        log.debug("[%s] TASK MESSAGE\n%s\n%s", agent_name, task_msg, _sep)

        for iteration in range(self._max_iter):
            log.debug("[%s] iteration %d — calling LLM (%s)", agent_name, iteration + 1, self._model)
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self.MAX_TOKENS_REPLY,
                system=system_prompt,
                messages=messages,
                tools=tools,
            )

            # Record the full exchange to the LLM prompt/response log
            if self._llm_logger is not None:
                self._llm_logger.log(
                    agent=agent_name,
                    iteration=iteration + 1,
                    system=system_prompt,
                    messages=messages,
                    model=self._model,
                    tools=tools,
                    response=response,
                )

            # Log any text the LLM emitted before tool calls
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    log.debug("[%s] LLM text:\n%s", agent_name, block.text)

            # Collect tool-use blocks from this turn
            tool_uses = [b for b in response.content if b.type == "tool_use"]

            if not tool_uses:
                log.warning("[%s] LLM returned no tool calls — stopping loop", agent_name)
                break

            # Append assistant turn to history
            messages.append({"role": "assistant", "content": response.content})

            # Process tool calls
            tool_results = []
            final_report: Optional[Report] = None

            for tc in tool_uses:
                log.info("[%s] tool → %s  args=%s",
                         agent_name, tc.name,
                         json.dumps(tc.input, indent=2))
                if tc.name == "post_report":
                    final_report = _build_report(self.__class__.__name__, tc.input)
                    log.debug("[%s] post_report → status=%s summary=%s",
                              agent_name,
                              tc.input.get("status"),
                              tc.input.get("summary"))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": "Report posted successfully.",
                    })
                else:
                    try:
                        result_str = self.call_tool(tc.name, tc.input)
                    except Exception as exc:
                        result_str = f"ERROR: {exc}"
                    log.info("[%s] result ← %s: %s",
                             agent_name, tc.name, result_str[:200])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": result_str,
                    })

            # Feed results back
            messages.append({"role": "user", "content": tool_results})

            if final_report is not None:
                return final_report

        # Exhausted iterations without a report
        return Report(
            agent=self.__class__.__name__,
            status=AgentStatus.FAILED,
            summary="Agent did not call post_report within the allowed iterations.",
        )


# ── Helper ────────────────────────────────────────────────────────────────────

def _build_report(agent_name: str, raw: dict) -> Report:
    """Construct a Report from the post_report tool call's input dict."""
    import json as _json

    status_map = {
        "done":             AgentStatus.DONE,
        "needs_user_input": AgentStatus.NEEDS_USER_INPUT,
        "failed":           AgentStatus.FAILED,
    }
    produced = [
        ArtifactRef(id=a["id"], type=a["type"], store=a["store"])
        for a in raw.get("produced", [])
    ]
    meta = raw.get("metadata", {})
    if isinstance(meta, str):
        try:
            meta = _json.loads(meta)
        except Exception:
            meta = {"raw": meta}
    return Report(
        agent=agent_name,
        status=status_map.get(raw.get("status", "done"), AgentStatus.DONE),
        summary=raw.get("summary", ""),
        produced=produced,
        metadata=meta,
        recommended_next=raw.get("recommended_next"),
    )
