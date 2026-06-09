from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .direct_launcher import DirectPanGenConfig, run_direct_pangen


@dataclass
class TermEvalInput:
    base_model_ref: str
    base_terms: list[str]
    target_terms: list[str]
    base_uwrms: float
    case_inputs: dict[str, Any] = field(default_factory=dict)
    model_inputs: dict[str, Any] = field(default_factory=dict)
    term_specs: dict[str, Any] = field(default_factory=dict)
    variables: list[dict[str, Any]] = field(default_factory=list)
    ga_rounds: int = 3
    pop_size: int = 20
    seed: int = 10


@dataclass
class TermEvalOutput:
    status: str
    model_ref: str
    uwrms: float
    validation_uwrms: float | None = None
    coefficients: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def evaluate_term_set(
    *,
    work_root: str | Path,
    base_model_ref: str,
    base_terms: list[str],
    target_terms: list[str],
    base_uwrms: float,
    case_inputs: dict[str, Any] | None = None,
    keep_workdir: bool = True,
    backend: str = "auto",
) -> dict[str, Any]:
    """Evaluate a target term set through a minimal PanGen boundary.

    This is intentionally a thin function. It owns the transient workspace and
    returns a plain dict so the agent runtime does not need to know PanGen files.

    The current implementation runs a synthetic direct task. With backend=auto,
    it uses PanGen when importable and falls back to a local synthetic GA.
    """
    request = TermEvalInput(
        base_model_ref=base_model_ref,
        base_terms=sorted(base_terms),
        target_terms=sorted(target_terms),
        base_uwrms=base_uwrms,
        case_inputs=(case_inputs or {}).get("case_inputs", {}),
        model_inputs=(case_inputs or {}).get("model_inputs", {}),
        term_specs=(case_inputs or {}).get("term_specs", {}),
        variables=(case_inputs or {}).get("variables", []),
        ga_rounds=int((case_inputs or {}).get("ga_rounds", 3)),
        pop_size=int((case_inputs or {}).get("pop_size", 20)),
        seed=int((case_inputs or {}).get("seed", 10)),
    )
    run_dir = _make_run_dir(Path(work_root), request)
    _write_json(run_dir / "input.json", asdict(request))

    result = run_direct_pangen(
        run_dir=run_dir,
        eval_input=asdict(request),
        config=DirectPanGenConfig(
            backend=backend,
            pangen_path=str((case_inputs or {}).get("pangen_path", "")),
            gateway=str((case_inputs or {}).get("gateway", "")),
            timeout_sec=int((case_inputs or {}).get("timeout_sec", 3600)),
            ga_rounds=request.ga_rounds,
            pop_size=request.pop_size,
            worker_nodes=dict((case_inputs or {}).get("worker_nodes", {})),
            worker_count=int((case_inputs or {}).get("worker_count", 1)),
            preprocess_threads=int((case_inputs or {}).get("preprocess_threads", 4)),
            use_gpu=bool((case_inputs or {}).get("use_gpu", True)),
            remote_shell=str((case_inputs or {}).get("remote_shell", "")),
            session_options=dict((case_inputs or {}).get("session_options", {})),
        ),
    )
    _write_json(run_dir / "result.json", result)

    result["run_dir"] = str(run_dir) if keep_workdir else None
    return result


def _make_run_dir(work_root: Path, request: TermEvalInput) -> Path:
    digest = hashlib.sha1(
        json.dumps(asdict(request), sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    run_dir = work_root / "pangen_runs" / ("eval_" + digest)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--base-model-ref", required=True)
    parser.add_argument("--base-uwrms", type=float, required=True)
    parser.add_argument("--base-terms", nargs="*", default=[])
    parser.add_argument("--target-terms", nargs="+", required=True)
    parser.add_argument("--backend", choices=["auto", "binary", "embedded", "local"], default="auto")
    parser.add_argument("--pangen-path", default="")
    parser.add_argument("--gateway", default="")
    args = parser.parse_args()

    result = evaluate_term_set(
        work_root=args.work_root,
        base_model_ref=args.base_model_ref,
        base_terms=args.base_terms,
        target_terms=args.target_terms,
        base_uwrms=args.base_uwrms,
        case_inputs={"pangen_path": args.pangen_path, "gateway": args.gateway},
        backend=args.backend,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
