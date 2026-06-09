# Term Agent

Minimal Python runtime for term selection experiments.

The agent proposes add/delete/replace actions. The runtime converts each action
into a target term set, validates it, evaluates it through an evaluator, then
commits or rejects the result.

First version uses a deterministic mock evaluator. A PanGen evaluator can later
replace it without changing the agent-facing tools.

## Run

```bash
python -m term_agent.runtime --root . --max-steps 5
```

Use a config file:

```bash
python -m term_agent.runtime --config config.example.json
```

Run through the minimal PanGen boundary. Without PanGen installed, this uses the
local synthetic GA fallback:

```bash
python -m term_agent.runtime --config config.example.json --evaluator pangen_minimal
```

Force PanGen binary mode:

```bash
python -m term_agent.runtime ^
  --config config.example.json ^
  --evaluator pangen_minimal
```

with:

```json
"pangen_minimal": {
  "backend": "binary",
  "pangen_path": "/data/pangen/pangen_2026.04.00.release",
  "gateway": "192.168.18.116:4730",
  "timeout_sec": 3600
}
```

Binary mode writes PanGen process logs into the eval run directory:

- `pangen_stdout.log`
- `pangen_stderr.log`

Non-zero PanGen exit codes and timeout are raised immediately instead of waiting
for `term_eval_result.json` forever.

Use an OpenAI-compatible LLM endpoint:

```bash
set LLM_BASE_URL=http://localhost:8000/v1
set LLM_API_KEY=sk-...
set LLM_MODEL=your-model
python -m term_agent.runtime --config config.example.json --proposer llm
```

You can also pass the values directly:

```bash
python -m term_agent.runtime ^
  --config config.example.json ^
  --proposer llm ^
  --llm-base-url http://localhost:8000/v1 ^
  --llm-api-key sk-... ^
  --llm-model your-model
```

Outputs:

- `runs/state.json`
- `runs/history.jsonl`

## Tool Shape

Agent-facing tools:

- `observe()`
- `evaluate_action(action)`
- `commit(result)`
- `reject(result)`
- `stop(reason)`

Supported actions:

- `add_bundle`
- `delete_bundle`
- `replace_bundle`
- `stop`

## Minimal PanGen Boundary

The PanGen side is intentionally thin:

```python
evaluate_term_set(
    work_root=...,
    base_model_ref=...,
    base_terms=[...],
    target_terms=[...],
    base_uwrms=...,
)
```

It writes one `input.json` under an internal `pangen_runs/eval_*` directory and
expects a runner to return:

```json
{
  "status": "success",
  "model_ref": "...",
  "uwrms": 1.23,
  "validation_uwrms": 1.30,
  "coefficients": {},
  "diagnostics": {}
}
```

The current runner is a stub in
`term_agent/pangen_minimal/run_term_eval.py`. Replace `_run_pangen_stub()` with
the real low-level PanGen calls when the minimum session sequence is settled.

There is also a direct PanGen path:

- `term_agent/pangen_minimal/direct_launcher.py`
- `term_agent/pangen_minimal/pangen_scripts.py`

This path does not use ArcGen `wizard.json` and does not import
`fit_amc_model.py`. The main package runs on modern Python. For PanGen binary
mode, `direct_launcher.py` writes low-version-compatible `direct_pframe.py` and
`direct_task.py` into the run directory, then calls:

```bash
${pangen_path}/bin/pangen -script direct_pframe.py -e "..." -g ${gateway}
```

Only the generated scripts are executed by PanGen's bundled Python.

The first implementation is synthetic:

- `direct_task.py` defines GA variables and a synthetic uwRMS objective.
- `direct_launcher.py` uses PanGen when importable, otherwise falls back to a
  local synthetic GA loop.
- This validates the boundary before replacing `objective()` with real RMS.

## Evaluation Inputs

`pangen_minimal` now carries the real-evaluation input contract, even though the
current objective is synthetic. These fields are written to
`term_eval_input.json` for `direct_task.py`:

- `base_model_ref`
- `base_terms`
- `target_terms`
- `base_uwrms`
- `case_inputs`
- `model_inputs`
- `term_specs`
- `variables`
- `ga_rounds`
- `pop_size`
- `seed`

`case_inputs` is intended for gauge/GDS/layer/TCC/split data. `model_inputs` is
intended for base model, source, mask, optics, and film settings. `term_specs`
describes operations, input channels, and parameter ranges per term. `variables`
can override the default `{term}_coeff + threshold` search space.
