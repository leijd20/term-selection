from __future__ import annotations

from .evaluator import Evaluator
from .state import EvaluationResult, TermAction, TermState


class TermTools:
    def __init__(self, evaluator: Evaluator) -> None:
        self.evaluator = evaluator

    def observe(self, state: TermState) -> dict:
        return {
            "active_terms": state.active_terms,
            "candidate_terms": state.candidate_terms,
            "current_model_ref": state.current_model_ref,
            "current_metrics": state.current_metrics.to_dict(),
            "budget": state.budget.to_dict(),
            "step": state.step,
            "eval_count": state.eval_count,
            "recent_history": [item.to_dict() for item in state.history[-8:]],
        }

    def evaluate_action(self, state: TermState, action: TermAction) -> EvaluationResult:
        self._validate_action(state, action)
        target_terms = self._target_terms(state.active_terms, action)
        return self.evaluator.evaluate(
            action=action,
            base_terms=list(state.active_terms),
            target_terms=target_terms,
            base_model_ref=state.current_model_ref,
            base_metrics=state.current_metrics,
        )

    def commit(self, state: TermState, result: EvaluationResult, reason: str) -> None:
        result.decision = "committed"
        result.reason = reason
        state.active_terms = list(result.target_terms)
        state.current_model_ref = result.model_ref
        state.current_metrics = result.metrics

    def reject(self, result: EvaluationResult, reason: str) -> None:
        result.decision = "rejected"
        result.reason = reason

    def stop(self, state: TermState, reason: str) -> None:
        state.done = True
        state.stop_reason = reason

    def _validate_action(self, state: TermState, action: TermAction) -> None:
        if action.type == "stop":
            return
        if state.eval_count >= state.budget.max_evals:
            raise ValueError("evaluation budget exhausted")

        bundle_size = len(action.add_terms) + len(action.remove_terms)
        if bundle_size > state.budget.max_bundle_size:
            raise ValueError("action exceeds max_bundle_size")

        known = set(state.active_terms) | set(state.candidate_terms)
        unknown = (set(action.add_terms) | set(action.remove_terms)) - known
        if unknown:
            raise ValueError("unknown terms: " + ", ".join(sorted(unknown)))

        active = set(state.active_terms)
        missing = set(action.remove_terms) - active
        if missing:
            raise ValueError("cannot remove inactive terms: " + ", ".join(sorted(missing)))

    @staticmethod
    def _target_terms(active_terms: list[str], action: TermAction) -> list[str]:
        terms = set(active_terms)
        if action.type in ("delete_bundle", "replace_bundle"):
            terms -= set(action.remove_terms)
        if action.type in ("add_bundle", "replace_bundle"):
            terms |= set(action.add_terms)
        return sorted(terms)
