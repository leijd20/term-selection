import unittest

from term_agent.config import build_state, load_config
from term_agent.evaluator import MockEvaluator
from term_agent.policy import CommitPolicy
from term_agent.proposer import HeuristicProposer, LLMProposer
from term_agent.runtime import TermRuntime, build_initial_state
from term_agent.store import JsonStore
from term_agent.tools import TermTools


class RuntimeTest(unittest.TestCase):
    def test_runtime_smoke(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = TermRuntime(
                state=build_initial_state(max_steps=3),
                tools=TermTools(MockEvaluator()),
                proposer=HeuristicProposer(),
                policy=CommitPolicy(),
                store=JsonStore(root / "runs"),
            )

            state = runtime.run()

            self.assertTrue(state.history)
            self.assertTrue((root / "runs" / "state.json").exists())
            self.assertTrue((root / "runs" / "history.jsonl").exists())


class FakeLLMClient:
    def chat(self, messages, *, temperature=0.2):
        return """
        {
          "actions": [
            {
              "type": "add_bundle",
              "add_terms": ["acid1", "base1"],
              "remove_terms": [],
              "rationale": "try acid/base pair"
            }
          ]
        }
        """


class LLMProposerTest(unittest.TestCase):
    def test_llm_proposer_parses_actions(self):
        state = build_initial_state(max_steps=1)
        proposer = LLMProposer(FakeLLMClient())

        actions = proposer.propose(state)

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].type, "add_bundle")
        self.assertEqual(actions[0].add_terms, ["acid1", "base1"])


class ConfigTest(unittest.TestCase):
    def test_load_config_and_build_state(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                """
                {
                  "proposer": "heuristic",
                  "budget": {"max_steps": 2, "parallel_k": 1},
                  "initial_state": {
                    "active_terms": ["ai", "acid1"],
                    "candidate_terms": ["base1"],
                    "current_model_ref": "m0",
                    "uwrms": 1.5
                  }
                }
                """,
                encoding="utf-8",
            )

            config = load_config(path)
            state = build_state(config)

            self.assertEqual(config.budget.max_steps, 2)
            self.assertEqual(config.budget.parallel_k, 1)
            self.assertEqual(state.active_terms, ["ai", "acid1"])
            self.assertEqual(state.current_metrics.uwrms, 1.5)


class PanGenMinimalTest(unittest.TestCase):
    def test_minimal_pangen_boundary_runs_local_synthetic(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from term_agent.pangen_minimal.run_term_eval import evaluate_term_set

        with TemporaryDirectory() as tmp:
            result = evaluate_term_set(
                work_root=tmp,
                base_model_ref="model0",
                base_terms=["ai"],
                target_terms=["ai", "acid1"],
                base_uwrms=2.0,
                backend="local",
            )

            run_dirs = list((Path(tmp) / "pangen_runs").glob("eval_*"))
            self.assertEqual(len(run_dirs), 1)
            self.assertTrue((run_dirs[0] / "input.json").exists())
            self.assertTrue((run_dirs[0] / "term_eval_result.json").exists())
            self.assertEqual(result["status"], "success")
            self.assertLess(result["uwrms"], 2.0)

    def test_pangen_evaluator_uses_minimal_boundary(self):
        from tempfile import TemporaryDirectory

        from term_agent.pangen_evaluator import PanGenEvaluator
        from term_agent.state import Metrics, TermAction

        with TemporaryDirectory() as tmp:
            evaluator = PanGenEvaluator(
                work_root=tmp,
                case_inputs={"seed": 10, "ga_rounds": 2, "pop_size": 8},
            )
            result = evaluator.evaluate(
                action=TermAction(type="add_bundle", add_terms=["acid1", "base1"]),
                base_terms=["ai"],
                target_terms=["ai", "acid1", "base1"],
                base_model_ref="model0",
                base_metrics=Metrics(uwrms=2.13),
            )

            self.assertEqual(result.diagnostics["pangen"], True)
            self.assertLess(result.metrics.uwrms, 2.13)


if __name__ == "__main__":
    unittest.main()
