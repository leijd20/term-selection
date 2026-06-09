from __future__ import annotations

import hashlib
import random
from abc import ABC, abstractmethod

from .state import EvaluationResult, Metrics, TermAction


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        *,
        action: TermAction,
        base_terms: list[str],
        target_terms: list[str],
        base_model_ref: str,
        base_metrics: Metrics,
    ) -> EvaluationResult:
        raise NotImplementedError


class MockEvaluator(Evaluator):
    """Deterministic evaluator used before a PanGen adapter exists."""

    def evaluate(
        self,
        *,
        action: TermAction,
        base_terms: list[str],
        target_terms: list[str],
        base_model_ref: str,
        base_metrics: Metrics,
    ) -> EvaluationResult:
        key = "|".join(sorted(target_terms))
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:8], 16))

        raw_gain = rng.uniform(-0.04, 0.18)
        synergy = self._synergy_bonus(target_terms)
        complexity_penalty = max(0, len(target_terms) - 3) * 0.025
        delta = round(raw_gain + synergy - complexity_penalty, 4)
        uwrms = round(max(0.001, base_metrics.uwrms - delta), 4)
        validation_uwrms = round(uwrms + rng.uniform(-0.02, 0.08), 4)
        result_id = "eval_" + digest[:10]

        coefficients = {
            term: round(rng.uniform(-0.8, 0.8), 4)
            for term in sorted(target_terms)
        }
        diagnostics = {
            "mock": True,
            "coef_ok": all(abs(value) >= 0.01 for value in coefficients.values()),
            "overfit_risk": validation_uwrms > base_metrics.uwrms,
        }

        return EvaluationResult(
            result_id=result_id,
            action=action,
            base_terms=base_terms,
            target_terms=target_terms,
            base_model_ref=base_model_ref,
            model_ref="models/" + result_id,
            base_metrics=base_metrics,
            metrics=Metrics(uwrms=uwrms, validation_uwrms=validation_uwrms, score=delta),
            delta_uwrms=delta,
            coefficients=coefficients,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _synergy_bonus(terms: list[str]) -> float:
        term_set = set(terms)
        bonus = 0.0
        if {"acid1", "base1"}.issubset(term_set):
            bonus += 0.08
        if {"normalx", "topoVector"}.issubset(term_set):
            bonus += 0.06
        return bonus
