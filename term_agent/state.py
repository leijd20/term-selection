from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ActionType = Literal["add_bundle", "delete_bundle", "replace_bundle", "stop"]
Decision = Literal["pending", "committed", "rejected"]


@dataclass
class Metrics:
    uwrms: float
    validation_uwrms: float | None = None
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Budget:
    max_steps: int = 8
    max_evals: int = 20
    max_bundle_size: int = 3
    parallel_k: int = 3
    min_commit_delta: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TermAction:
    type: ActionType
    add_terms: list[str] = field(default_factory=list)
    remove_terms: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    result_id: str
    action: TermAction
    base_terms: list[str]
    target_terms: list[str]
    base_model_ref: str
    model_ref: str
    base_metrics: Metrics
    metrics: Metrics
    delta_uwrms: float
    coefficients: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    decision: Decision = "pending"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action"] = self.action.to_dict()
        data["base_metrics"] = self.base_metrics.to_dict()
        data["metrics"] = self.metrics.to_dict()
        return data


@dataclass
class TermState:
    active_terms: list[str]
    candidate_terms: list[str]
    current_model_ref: str
    current_metrics: Metrics
    budget: Budget = field(default_factory=Budget)
    step: int = 0
    eval_count: int = 0
    done: bool = False
    stop_reason: str = ""
    history: list[EvaluationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_terms": self.active_terms,
            "candidate_terms": self.candidate_terms,
            "current_model_ref": self.current_model_ref,
            "current_metrics": self.current_metrics.to_dict(),
            "budget": self.budget.to_dict(),
            "step": self.step,
            "eval_count": self.eval_count,
            "done": self.done,
            "stop_reason": self.stop_reason,
            "history": [item.to_dict() for item in self.history],
        }
