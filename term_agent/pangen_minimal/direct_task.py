"""Direct PanGen task for synthetic term-set evaluation.

This file is intended to be used as PanGen's `python_file`.

It deliberately does not read ArcGen's wizard.json and does not import
fit_amc_model.py. The only config input is `term_eval_input.json` in cwd.

PanGen still needs a task file because GA sessions call user-defined lifecycle
functions through pga/psys:

- init_ga_calibrate()
- process_ga_calibrate()
- finalize_ga_calibrate()
- between_rounds_ga_calibrate()

This first version uses a synthetic objective. Its purpose is to verify that a
custom PanGen `python_file` can replace ArcGen's fit_amc_model.py. Replace
`objective()` with real model/RMS calculation later.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any


INPUT_FILE = Path("term_eval_input.json")
RESULT_FILE = Path("term_eval_result.json")
BEST_FILE = Path("best_candidate.json")


def _load_input() -> dict:
    return json.loads(INPUT_FILE.read_text(encoding="utf-8"))


def default_variables(request: dict) -> list[dict]:
    if request.get("variables"):
        return request["variables"]
    base_terms = set(request.get("base_terms", []))
    variables = []
    for term in sorted(set(request.get("target_terms", [])) - base_terms):
        variables.append({"name": f"{term}_coeff", "type": "real", "lower": -1.0, "upper": 1.0})
    variables.append({"name": "threshold", "type": "real", "lower": 0.05, "upper": 0.6})
    return variables


def decode_point(point, variables: list[dict]) -> dict[str, float]:
    params = {}
    for idx, var in enumerate(variables):
        if idx >= len(point):
            break
        value = point[idx]
        if var.get("type", "real") == "integer":
            value = int(round(value))
        params[var["name"]] = float(value)
    return params


def objective(params: dict[str, float], request: dict) -> float:
    """Synthetic uwRMS objective for integration testing."""
    uwrms = float(request.get("base_uwrms", 2.0))
    target_terms = set(request.get("target_terms", []))

    for name, value in params.items():
        if name.endswith("_coeff"):
            term = name[:-6]
            ideal = _ideal_coeff(term)
            uwrms -= max(0.0, 0.10 - abs(value - ideal) * 0.08)
            uwrms += max(0.0, abs(value) - 0.85) * 0.08

    if {"acid1", "base1"}.issubset(target_terms):
        uwrms -= 0.08
    if {"normalx", "topoVector"}.issubset(target_terms):
        uwrms -= 0.06

    threshold = params.get("threshold")
    if threshold is not None:
        uwrms += abs(threshold - 0.35) * 0.15

    complexity = max(0, len(target_terms) - 4)
    uwrms += complexity * 0.025
    return round(max(0.001, uwrms), 4)


def _ideal_coeff(term: str) -> float:
    if term.startswith("acid"):
        return -0.35
    if term.startswith("base"):
        return 0.25
    if term == "normalx":
        return 0.15
    if term == "topoVector":
        return -0.2
    return 0.1


def update_best(params: dict[str, float], uwrms: float, request: dict) -> None:
    current = None
    if BEST_FILE.exists():
        current = json.loads(BEST_FILE.read_text(encoding="utf-8"))
    if current is None or uwrms < float(current["uwrms"]):
        BEST_FILE.write_text(
            json.dumps({
                "uwrms": uwrms,
                "params": params,
                "target_terms": request.get("target_terms", []),
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def write_result(request: dict, backend: str = "direct_pangen_synthetic") -> dict:
    if not BEST_FILE.exists():
        raise FileNotFoundError("best candidate was not written")
    best = json.loads(BEST_FILE.read_text(encoding="utf-8"))
    result = {
        "status": "success",
        "model_ref": "synthetic/" + "_".join(best.get("target_terms", [])),
        "uwrms": best["uwrms"],
        "validation_uwrms": None,
        "coefficients": best["params"],
        "diagnostics": {
            "backend": backend,
            "synthetic": True,
        },
    }
    RESULT_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def run_local_synthetic_ga(request: dict) -> dict:
    """Local fallback used when PanGen is not available."""
    variables = default_variables(request)
    ga_rounds = int(request.get("ga_rounds", 3))
    pop_size = int(request.get("pop_size", 20))
    rng = random.Random(int(request.get("seed", 10)))
    for _ in range(max(1, ga_rounds * pop_size)):
        params = {}
        for var in variables:
            lo = float(var.get("lower", 0.0))
            hi = float(var.get("upper", 1.0))
            if var.get("type", "real") == "integer":
                params[var["name"]] = float(rng.randint(int(lo), int(hi)))
            else:
                params[var["name"]] = rng.uniform(lo, hi)
        update_best(params, objective(params, request), request)
    return write_result(request, backend="local_synthetic_ga")


def init_ga_calibrate():
    """Initialize one term-set tuning session.

    This synthetic implementation only sets GA config and optimization
    variables. It does not build a real model.
    """
    import pga

    request = _load_input()
    pga.set_task_type("ModelCalibration")

    cfg = pga.GA_Config()
    cfg.max_generation = int(request.get("ga_rounds", 3))
    cfg.population_size = int(request.get("pop_size", 20))
    cfg.seed = int(request.get("seed", 10))
    cfg.individual_viscous_mutate_rate = 0.7
    cfg.individual_no_mutate_rate = 0.0
    cfg.selection_method = 1
    cfg.replacement_method = 1
    pga.set_config(cfg)

    opt_vars = []
    for var in default_variables(request):
        if var.get("type", "real") == "integer":
            opt_vars.append(("integer", int(var["lower"]), int(var["upper"]), var["name"]))
        else:
            opt_vars.append(("real", float(var["lower"]), float(var["upper"]), var["name"]))
    pga.set_optimization_variable(opt_vars, 0.3)


def process_ga_calibrate():
    """Evaluate one GA point.

    PanGen gives this function one GA point. We evaluate the synthetic
    objective and return fitness to PanGen.
    """
    import pga

    request = _load_input()
    point = pga.get_current_point()
    params = decode_point(point, default_variables(request))
    uwrms = objective(params, request)
    pga.set_current_point_fitness(-uwrms)
    pga.set_current_point_rms(uwrms)
    update_best(params, uwrms, request)


def finalize_ga_calibrate():
    """Summarize the best model and write term_eval_result.json."""
    write_result(_load_input())


def between_rounds_ga_calibrate():
    """Optional hook called by PanGen between GA generations."""
    return None


def PanGenFlow():
    """PanGen entry point.

    Dispatch by PanGen lifecycle status when available.
    """
    try:
        from pangen.model.pmodelutil import vm_status
    except Exception:
        run_local_synthetic_ga(_load_input())
        return None

    if vm_status == "initialization":
        init_ga_calibrate()
    elif vm_status == "process":
        process_ga_calibrate()
    elif vm_status == "done":
        finalize_ga_calibrate()
    elif vm_status == "between_rounds":
        between_rounds_ga_calibrate()
    return None


if __name__ == "__main__":
    PanGenFlow()
