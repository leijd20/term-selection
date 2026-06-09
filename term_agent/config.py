from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .state import Budget, Metrics, TermState


ProposerName = Literal["heuristic", "llm"]
EvaluatorName = Literal["mock", "pangen_minimal"]


@dataclass
class LLMConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = "gpt-4.1-mini"
    timeout: float = 60.0
    temperature: float = 0.2

    def with_env(self) -> LLMConfig:
        return LLMConfig(
            base_url=os.environ.get("LLM_BASE_URL", self.base_url),
            api_key=os.environ.get("LLM_API_KEY", self.api_key),
            model=os.environ.get("LLM_MODEL", self.model),
            timeout=float(os.environ.get("LLM_TIMEOUT", self.timeout)),
            temperature=float(os.environ.get("LLM_TEMPERATURE", self.temperature)),
        )


@dataclass
class InitialStateConfig:
    active_terms: list[str] = field(default_factory=lambda: ["ai"])
    candidate_terms: list[str] = field(default_factory=lambda: [
        "acid1", "base1", "normalx", "topoVector", "acid2", "base2"
    ])
    current_model_ref: str = "models/baseline"
    uwrms: float = 2.13
    validation_uwrms: float | None = 2.18


@dataclass
class PanGenMinimalConfig:
    backend: str = "auto"
    pangen_path: str = ""
    gateway: str = ""
    timeout_sec: int = 3600
    ga_rounds: int = 3
    pop_size: int = 20
    seed: int = 10
    worker_nodes: dict[str, str] = field(default_factory=dict)
    worker_count: int = 1
    preprocess_threads: int = 4
    use_gpu: bool = True
    remote_shell: str = ""
    session_options: dict[str, Any] = field(default_factory=dict)
    case_inputs: dict[str, Any] = field(default_factory=dict)
    model_inputs: dict[str, Any] = field(default_factory=dict)
    term_specs: dict[str, Any] = field(default_factory=dict)
    variables: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RuntimeConfig:
    root: str = "."
    proposer: ProposerName = "heuristic"
    evaluator: EvaluatorName = "mock"
    budget: Budget = field(default_factory=Budget)
    initial_state: InitialStateConfig = field(default_factory=InitialStateConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    pangen_minimal: PanGenMinimalConfig = field(default_factory=PanGenMinimalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None) -> RuntimeConfig:
    config = RuntimeConfig()
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        config = _merge_config(config, data)
    config.llm = config.llm.with_env()
    return config


def build_state(config: RuntimeConfig, *, max_steps_override: int | None = None) -> TermState:
    budget = config.budget
    if max_steps_override is not None:
        budget = Budget(**{**asdict(config.budget), "max_steps": max_steps_override})
    initial = config.initial_state
    return TermState(
        active_terms=list(initial.active_terms),
        candidate_terms=list(initial.candidate_terms),
        current_model_ref=initial.current_model_ref,
        current_metrics=Metrics(
            uwrms=initial.uwrms,
            validation_uwrms=initial.validation_uwrms,
            score=0.0,
        ),
        budget=budget,
    )


def _merge_config(config: RuntimeConfig, data: dict[str, Any]) -> RuntimeConfig:
    budget_data = {**asdict(config.budget), **data.get("budget", {})}
    initial_data = {**asdict(config.initial_state), **data.get("initial_state", {})}
    llm_data = {**asdict(config.llm), **data.get("llm", {})}
    pangen_data = {**asdict(config.pangen_minimal), **data.get("pangen_minimal", {})}
    return RuntimeConfig(
        root=data.get("root", config.root),
        proposer=data.get("proposer", config.proposer),
        evaluator=data.get("evaluator", config.evaluator),
        budget=Budget(**budget_data),
        initial_state=InitialStateConfig(**initial_data),
        llm=LLMConfig(**llm_data),
        pangen_minimal=PanGenMinimalConfig(**pangen_data),
    )
