from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .evaluator import Evaluator
from .pangen_minimal.run_term_eval import evaluate_term_set
from .state import EvaluationResult, Metrics, TermAction


class PanGenEvaluator(Evaluator):
    """Evaluator adapter around the minimal PanGen term-set function."""

    def __init__(self, *, work_root: str | Path, case_inputs: dict[str, Any] | None = None) -> None:
        self.work_root = Path(work_root)
        self.case_inputs = case_inputs or {}

    def evaluate(
        self,
        *,
        action: TermAction,
        base_terms: list[str],
        target_terms: list[str],
        base_model_ref: str,
        base_metrics: Metrics,
    ) -> EvaluationResult:
        raw = evaluate_term_set(
            work_root=self.work_root,
            base_model_ref=base_model_ref,
            base_terms=base_terms,
            target_terms=target_terms,
            base_uwrms=base_metrics.uwrms,
            case_inputs=self.case_inputs,
            backend=str(self.case_inputs.get("backend", "auto")),
        )
        uwrms = float(raw["uwrms"])
        delta = round(base_metrics.uwrms - uwrms, 4)
        result_id = "pangen_" + hashlib.sha1(
            "|".join(sorted(target_terms)).encode("utf-8")
        ).hexdigest()[:10]
        diagnostics = dict(raw.get("diagnostics") or {})
        diagnostics["pangen"] = True
        return EvaluationResult(
            result_id=result_id,
            action=action,
            base_terms=base_terms,
            target_terms=target_terms,
            base_model_ref=base_model_ref,
            model_ref=str(raw["model_ref"]),
            base_metrics=base_metrics,
            metrics=Metrics(
                uwrms=uwrms,
                validation_uwrms=raw.get("validation_uwrms"),
                score=delta,
            ),
            delta_uwrms=delta,
            coefficients=dict(raw.get("coefficients") or {}),
            diagnostics=diagnostics,
        )
