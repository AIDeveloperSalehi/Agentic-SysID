"""
LLM interaction logger — writes a clean JSON array of prompt/response pairs.

Each entry records one API call:
  agent         — which agent made the call
  iteration     — call number within that agent's loop (1-based)
  timestamp     — ISO-8601 seconds
  model         — model ID used
  system        — system prompt (string)
  messages_in   — conversation history sent to the API
  tool_names    — names of tools available (schemas omitted for readability)
  text_out      — LLM reasoning text (may be empty string)
  tool_calls_out — list of {name, input} dicts the LLM requested
  stop_reason   — API stop_reason field

File format: a valid JSON array.  Entries are written incrementally so the
file is readable even if the pipeline crashes mid-run (the array is left
unclosed until close() is called, but each entry is valid JSON on its own
line once the commas are handled).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional


class LLMLogger:
    """
    Write one JSON entry per LLM API call to a .json file.

    Usage
    -----
    with LLMLogger(path) as logger:
        logger.log(agent, iteration, system, messages, model, tools, response)
    """

    def __init__(self, path: "str | Path"):
        self._path  = Path(path)
        self._file  = open(self._path, "w", encoding="utf-8")
        self._count = 0
        self._file.write("[\n")
        self._file.flush()

    # ── Public API ────────────────────────────────────────────────────────────

    def log(
        self,
        agent:     str,
        iteration: int,
        system:    str,
        messages:  list,
        model:     str,
        tools:     list,
        response:  Any,
    ) -> None:
        """Append one LLM exchange to the JSON file."""
        entry = {
            "agent":          agent,
            "iteration":      iteration,
            "timestamp":      datetime.now().isoformat(timespec="seconds"),
            "model":          model,
            "system":         system,
            "messages_in":    _serialize_messages(messages),
            "tool_names":     [t["name"] for t in tools if isinstance(t, dict)],
            "text_out":       _extract_text(response.content),
            "tool_calls_out": _extract_tool_calls(response.content),
            "stop_reason":    getattr(response, "stop_reason", None),
        }
        if self._count > 0:
            self._file.write(",\n")
        self._file.write(json.dumps(entry, ensure_ascii=False, indent=2))
        self._file.flush()
        self._count += 1

    def close(self) -> None:
        """Finalise the JSON array and close the file."""
        self._file.write("\n]\n")
        self._file.close()

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "LLMLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _serialize_messages(messages: list) -> list:
    """
    Convert a messages list (may contain Anthropic SDK block objects) to plain dicts.
    """
    result = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            result.append({"role": msg["role"], "content": content})
        elif isinstance(content, list):
            result.append({"role": msg["role"], "content": _serialize_content(content)})
        else:
            result.append({"role": msg["role"], "content": str(content)})
    return result


def _serialize_content(blocks: list) -> list:
    """Convert a list of content blocks (SDK objects or dicts) to plain dicts."""
    out = []
    for block in blocks:
        if isinstance(block, dict):
            out.append(block)
        elif hasattr(block, "type"):
            if block.type == "text":
                out.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                out.append({
                    "type":  "tool_use",
                    "id":    block.id,
                    "name":  block.name,
                    "input": block.input,
                })
            else:
                out.append({"type": block.type})
        else:
            out.append({"type": "unknown", "raw": str(block)})
    return out


def _extract_text(content: list) -> str:
    """Concatenate all TextBlock texts from the response content."""
    parts = []
    for block in content:
        if hasattr(block, "type") and block.type == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def _extract_tool_calls(content: list) -> List[dict]:
    """Return a list of {name, input} dicts for every ToolUseBlock."""
    return [
        {"name": block.name, "input": block.input}
        for block in content
        if hasattr(block, "type") and block.type == "tool_use"
    ]
