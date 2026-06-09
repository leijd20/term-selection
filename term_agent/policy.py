from __future__ import annotations

from .state import EvaluationResult, TermState


class CommitPolicy:
    def choose_commit(self, state: TermState, results: list[EvaluationResult]) -> EvaluationResult | None:
        if not results:
            return None
        best = max(results, key=lambda item: item.delta_uwrms)
        if best.delta_uwrms < state.budget.min_commit_delta:
            return None
        if best.diagnostics.get("overfit_risk"):
            return None
        return best
