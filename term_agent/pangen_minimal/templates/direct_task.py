import json
import os
import random


INPUT_FILE = "term_eval_input.json"
RESULT_FILE = "term_eval_result.json"
BEST_FILE = "best_candidate.json"


def _load_input():
    with open(INPUT_FILE, "r") as f:
        return json.load(f)


def default_variables(request):
    if request.get("variables"):
        return request["variables"]
    base_terms = set(request.get("base_terms", []))
    variables = []
    for term in sorted(set(request.get("target_terms", [])) - base_terms):
        variables.append({"name": term + "_coeff", "type": "real", "lower": -1.0, "upper": 1.0})
    variables.append({"name": "threshold", "type": "real", "lower": 0.05, "upper": 0.6})
    return variables


def decode_point(point, variables):
    params = {}
    for idx, var in enumerate(variables):
        if idx >= len(point):
            break
        value = point[idx]
        if var.get("type", "real") == "integer":
            value = int(round(value))
        params[var["name"]] = float(value)
    return params


def _ideal_coeff(term):
    if term.startswith("acid"):
        return -0.35
    if term.startswith("base"):
        return 0.25
    if term == "normalx":
        return 0.15
    if term == "topoVector":
        return -0.2
    return 0.1


def objective(params, request):
    uwrms = float(request.get("base_uwrms", 2.0))
    target_terms = set(request.get("target_terms", []))
    for name, value in params.items():
        if name.endswith("_coeff"):
            term = name[:-6]
            ideal = _ideal_coeff(term)
            uwrms -= max(0.0, 0.10 - abs(value - ideal) * 0.08)
            uwrms += max(0.0, abs(value) - 0.85) * 0.08
    if set(["acid1", "base1"]).issubset(target_terms):
        uwrms -= 0.08
    if set(["normalx", "topoVector"]).issubset(target_terms):
        uwrms -= 0.06
    threshold = params.get("threshold")
    if threshold is not None:
        uwrms += abs(threshold - 0.35) * 0.15
    complexity = max(0, len(target_terms) - 4)
    uwrms += complexity * 0.025
    return round(max(0.001, uwrms), 4)


def update_best(params, uwrms, request):
    current = None
    if os.path.exists(BEST_FILE):
        with open(BEST_FILE, "r") as f:
            current = json.load(f)
    if current is None or uwrms < float(current["uwrms"]):
        with open(BEST_FILE, "w") as f:
            json.dump({
                "uwrms": uwrms,
                "params": params,
                "target_terms": request.get("target_terms", []),
            }, f, indent=2)


def write_result(request, backend):
    if not os.path.exists(BEST_FILE):
        raise IOError("best candidate was not written")
    with open(BEST_FILE, "r") as f:
        best = json.load(f)
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
    with open(RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    return result


def run_local_synthetic_ga(request):
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
    return write_result(request, "local_synthetic_ga")


def init_ga_calibrate():
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
    import pga
    request = _load_input()
    point = pga.get_current_point()
    params = decode_point(point, default_variables(request))
    uwrms = objective(params, request)
    pga.set_current_point_fitness(-uwrms)
    pga.set_current_point_rms(uwrms)
    update_best(params, uwrms, request)


def finalize_ga_calibrate():
    write_result(_load_input(), "direct_pangen_binary_synthetic")


def between_rounds_ga_calibrate():
    return None


def PanGenFlow():
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
