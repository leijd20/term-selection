from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .state import EvaluationResult, TermState


class JsonStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.history_path = self.root / "history.jsonl"

    def save_state(self, state: TermState) -> None:
        self.state_path.write_text(
            json.dumps(state.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def append_result(self, result: EvaluationResult) -> None:
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    def write_event(self, event: dict[str, Any]) -> None:
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
