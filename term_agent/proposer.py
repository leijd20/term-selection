from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Protocol

from .state import TermAction, TermState


class Proposer(Protocol):
    def propose(self, state: TermState) -> list[TermAction]:
        ...


class HeuristicProposer:
    """Placeholder proposer until an LLM client is connected."""

    def propose(self, state: TermState) -> list[TermAction]:
        active = set(state.active_terms)
        candidates = [term for term in state.candidate_terms if term not in active]
        actions: list[TermAction] = []

        if "acid1" in candidates and "base1" in candidates:
            actions.append(TermAction(
                type="add_bundle",
                add_terms=["acid1", "base1"],
                rationale="try acid/base synergy",
            ))
        if "normalx" in candidates and "topoVector" in candidates:
            actions.append(TermAction(
                type="add_bundle",
                add_terms=["normalx", "topoVector"],
                rationale="try geometry/vector synergy",
            ))
        for term in candidates:
            actions.append(TermAction(
                type="add_bundle",
                add_terms=[term],
                rationale="single-term exploration",
            ))

        return actions[: state.budget.parallel_k]


class OpenAICompatibleClient:
    """Small OpenAI-compatible chat client using only the standard library."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        temperature: float = 0.2,
    ) -> None:
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "").rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY")
        self.model = model or os.environ.get("LLM_MODEL") or "gpt-4.1-mini"
        self.timeout = timeout
        self.temperature = temperature
        if not self.base_url:
            raise ValueError("LLM_BASE_URL is required for LLMProposer")

    def chat(self, messages: list[dict[str, str]], *, temperature: float | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=data,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        parsed = json.loads(body)
        return parsed["choices"][0]["message"].get("content") or ""

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        return headers


class LLMProposer:
    def __init__(self, client: OpenAICompatibleClient | None = None) -> None:
        self.client = client or OpenAICompatibleClient()

    def propose(self, state: TermState) -> list[TermAction]:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(_observation(state), ensure_ascii=False)},
        ]
        content = self.client.chat(messages)
        actions = _parse_actions(content)
        return actions[: state.budget.parallel_k]


def _observation(state: TermState) -> dict:
    return {
        "active_terms": state.active_terms,
        "candidate_terms": state.candidate_terms,
        "current_metrics": state.current_metrics.to_dict(),
        "budget": state.budget.to_dict(),
        "step": state.step,
        "eval_count": state.eval_count,
        "recent_history": [item.to_dict() for item in state.history[-8:]],
        "instructions": {
            "max_actions": state.budget.parallel_k,
            "max_bundle_size": state.budget.max_bundle_size,
            "allowed_action_types": ["add_bundle", "delete_bundle", "replace_bundle", "stop"],
        },
    }


def _parse_actions(content: str) -> list[TermAction]:
    text = content.strip()
    if not text:
        raise ValueError("LLM returned empty content")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(1))

    raw_actions = data.get("actions", data) if isinstance(data, dict) else data
    if not isinstance(raw_actions, list):
        raise ValueError("LLM response must be a list or an object with actions")

    actions = []
    for item in raw_actions:
        if not isinstance(item, dict):
            continue
        action_type = item.get("type")
        if action_type not in ("add_bundle", "delete_bundle", "replace_bundle", "stop"):
            continue
        actions.append(TermAction(
            type=action_type,
            add_terms=list(item.get("add_terms") or []),
            remove_terms=list(item.get("remove_terms") or []),
            rationale=str(item.get("rationale") or ""),
        ))
    if not actions:
        raise ValueError("LLM did not return any valid actions")
    return actions


_SYSTEM_PROMPT = """\
You are a term selection optimizer. Propose the next actions to evaluate.

Return only JSON in this exact shape:
{
  "actions": [
    {
      "type": "add_bundle | delete_bundle | replace_bundle | stop",
      "add_terms": ["term_name"],
      "remove_terms": ["term_name"],
      "rationale": "short reason"
    }
  ]
}

Rules:
- Propose at most max_actions actions from the observation.
- Do not invent terms. add_terms must come from candidate_terms.
- remove_terms must come from active_terms.
- Keep add_terms + remove_terms within max_bundle_size.
- Prefer bundles when terms may have synergy.
- Avoid repeating failed patterns from recent_history.
- Use stop only when budget is exhausted or no useful action remains.
"""
