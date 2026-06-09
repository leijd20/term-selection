from __future__ import annotations

import argparse
from pathlib import Path

from .config import RuntimeConfig, build_state, load_config
from .evaluator import Evaluator, MockEvaluator
from .pangen_evaluator import PanGenEvaluator
from .policy import CommitPolicy
from .proposer import HeuristicProposer, LLMProposer, OpenAICompatibleClient, Proposer
from .state import TermState
from .store import JsonStore
from .tools import TermTools


class TermRuntime:
    def __init__(
        self,
        *,
        state: TermState,
        tools: TermTools,
        proposer: Proposer,
        policy: CommitPolicy,
        store: JsonStore,
    ) -> None:
        self.state = state
        self.tools = tools
        self.proposer = proposer
        self.policy = policy
        self.store = store

    def run(self) -> TermState:
        self.store.save_state(self.state)
        while not self.state.done:
            if self.state.step >= self.state.budget.max_steps:
                self.tools.stop(self.state, "max_steps reached")
                break
            if self.state.eval_count >= self.state.budget.max_evals:
                self.tools.stop(self.state, "max_evals reached")
                break

            actions = self.proposer.propose(self.state)
            if not actions:
                self.tools.stop(self.state, "no actions proposed")
                break

            results = []
            for action in actions:
                try:
                    result = self.tools.evaluate_action(self.state, action)
                except ValueError as exc:
                    self.store.write_event({
                        "type": "invalid_action",
                        "action": action.to_dict(),
                        "error": str(exc),
                    })
                    continue
                results.append(result)
                self.state.eval_count += 1

            chosen = self.policy.choose_commit(self.state, results)
            for result in results:
                if chosen is not None and result.result_id == chosen.result_id:
                    self.tools.commit(self.state, result, "best result passed commit policy")
                else:
                    self.tools.reject(result, "not selected by commit policy")
                self.state.history.append(result)
                self.store.append_result(result)

            self.state.step += 1
            self.store.save_state(self.state)

            if chosen is None:
                self.tools.stop(self.state, "no candidate passed commit policy")

        self.store.save_state(self.state)
        return self.state


def build_initial_state(max_steps: int) -> TermState:
    return build_state(RuntimeConfig(), max_steps_override=max_steps)


def build_proposer(config: RuntimeConfig) -> Proposer:
    if config.proposer == "llm":
        return LLMProposer(OpenAICompatibleClient(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
            model=config.llm.model,
            timeout=config.llm.timeout,
            temperature=config.llm.temperature,
        ))
    return HeuristicProposer()


def build_evaluator(config: RuntimeConfig) -> Evaluator:
    if config.evaluator == "pangen_minimal":
        return PanGenEvaluator(
            work_root=Path(config.root),
            case_inputs={
                "backend": config.pangen_minimal.backend,
                "pangen_path": config.pangen_minimal.pangen_path,
                "gateway": config.pangen_minimal.gateway,
                "timeout_sec": config.pangen_minimal.timeout_sec,
                "ga_rounds": config.pangen_minimal.ga_rounds,
                "pop_size": config.pangen_minimal.pop_size,
                "seed": config.pangen_minimal.seed,
                "worker_nodes": config.pangen_minimal.worker_nodes,
                "worker_count": config.pangen_minimal.worker_count,
                "preprocess_threads": config.pangen_minimal.preprocess_threads,
                "use_gpu": config.pangen_minimal.use_gpu,
                "case_inputs": config.pangen_minimal.case_inputs,
                "model_inputs": config.pangen_minimal.model_inputs,
                "term_specs": config.pangen_minimal.term_specs,
                "variables": config.pangen_minimal.variables,
            },
        )
    return MockEvaluator()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--proposer", choices=["heuristic", "llm"], default=None)
    parser.add_argument("--evaluator", choices=["mock", "pangen_minimal"], default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-model", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.root is not None:
        config.root = args.root
    if args.proposer is not None:
        config.proposer = args.proposer
    if args.evaluator is not None:
        config.evaluator = args.evaluator
    if args.llm_base_url is not None:
        config.llm.base_url = args.llm_base_url
    if args.llm_api_key is not None:
        config.llm.api_key = args.llm_api_key
    if args.llm_model is not None:
        config.llm.model = args.llm_model

    store = JsonStore(Path(config.root) / "runs")
    runtime = TermRuntime(
        state=build_state(config, max_steps_override=args.max_steps),
        tools=TermTools(build_evaluator(config)),
        proposer=build_proposer(config),
        policy=CommitPolicy(),
        store=store,
    )
    final_state = runtime.run()
    print("done:", final_state.stop_reason or "completed")
    print("active_terms:", ", ".join(final_state.active_terms))
    print("uwrms:", final_state.current_metrics.uwrms)


if __name__ == "__main__":
    main()
