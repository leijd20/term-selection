"""Direct PanGen launcher skeleton.

This launcher avoids ArcGen's pframe/wizard stack. It directly builds PanGen
options and points `python_file` at `direct_task.py`.
"""

from __future__ import annotations

import argparse
import os
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .pangen_scripts import write_pangen_scripts


@dataclass
class DirectPanGenConfig:
    worker_nodes: dict[str, str] = field(default_factory=dict)
    worker_count: int = 1
    preprocess_threads: int = 4
    use_gpu: bool = True
    ga_rounds: int = 3
    pop_size: int = 20
    mode: str = "distributed"
    backend: str = "auto"  # auto | binary | embedded | local
    pangen_path: str = ""
    gateway: str = ""
    timeout_sec: int = 3600


def run_direct_pangen(
    *,
    run_dir: str | Path,
    eval_input: dict[str, Any],
    config: DirectPanGenConfig | None = None,
) -> dict[str, Any]:
    """Prepare cwd and call PanGen directly.

    This function does not read wizard.json. The real PanGen call is isolated in
    `_execute_pangen_sessions()` so it can be implemented once PanGen is
    available in the runtime environment.
    """
    config = config or DirectPanGenConfig()
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    write_pangen_scripts(run_path)

    (run_path / "term_eval_input.json").write_text(
        json.dumps(eval_input, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_path / "direct_pangen_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _execute_pangen_sessions(run_path, config)

    result_path = run_path / "term_eval_result.json"
    if not result_path.exists():
        raise FileNotFoundError("PanGen result was not written: " + str(result_path))
    return json.loads(result_path.read_text(encoding="utf-8"))


def _execute_pangen_sessions(run_dir: Path, config: DirectPanGenConfig) -> None:
    """Call PanGen sessions without ArcGen pframe/wizard.

    If backend is local, run a local synthetic GA without PanGen.
    If backend is binary, call the PanGen binary with generated direct_pframe.py.
    If backend is embedded, import pangen.system in the current interpreter.
    If backend is auto, prefer binary when configured, then embedded, then local.
    """
    if config.backend not in ("auto", "binary", "embedded", "pangen", "local"):
        raise ValueError("backend must be auto, binary, embedded, pangen, or local")

    if config.backend == "local":
        _execute_local_synthetic(run_dir)
        return

    if config.backend == "binary":
        _execute_pangen_binary(run_dir, config)
        return

    if config.backend == "auto" and config.pangen_path and config.gateway:
        _execute_pangen_binary(run_dir, config)
        return

    try:
        import pangen.system as psys
    except ImportError:
        if config.backend in ("pangen", "embedded"):
            raise
        _execute_local_synthetic(run_dir)
        return

    cwd = os.getcwd()
    os.chdir(str(run_dir))
    try:
        options = {
            "python_file": "direct_task.py",
            "use_gpu": "1" if config.use_gpu else "0",
            "worker_count": str(config.worker_count),
            "preprocess_thread_count": str(config.preprocess_threads),
            "ga_need_save_model": "1",
            "ga_total_rounds": str(config.ga_rounds),
            "pop_size": str(config.pop_size),
        }
        psys.execute_session(
            "ga",
            mode=config.mode,
            session_name="ga_calibrate_1",
            options=options,
            worker_nodes=config.worker_nodes,
        )
    finally:
        os.chdir(cwd)


def _execute_local_synthetic(run_dir: Path) -> None:
    import importlib.util

    task_path = run_dir / "direct_task.py"
    spec = importlib.util.spec_from_file_location("direct_task_local", task_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load local direct_task.py")
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    os.chdir(str(run_dir))
    try:
        spec.loader.exec_module(module)
        request = module._load_input()
        module.run_local_synthetic_ga(request)
    finally:
        os.chdir(cwd)


def _execute_pangen_binary(run_dir: Path, config: DirectPanGenConfig) -> None:
    if not config.pangen_path:
        raise ValueError("pangen_path is required for binary backend")
    if not config.gateway:
        raise ValueError("gateway is required for binary backend")

    pangen_root = Path(config.pangen_path)
    pangen_bin = pangen_root / "bin" / ("pangen.exe" if os.name == "nt" else "pangen")
    if not pangen_bin.exists():
        raise FileNotFoundError("PanGen binary not found: " + str(pangen_bin))

    lib_path = pangen_root / "lib"
    python_paths = [
        str(lib_path / "python3.4"),
        str(lib_path / "python3.4" / "lib-dynload"),
        str(lib_path / "python3.4" / "site-packages"),
        str(pangen_root / "script"),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [env.get("PYTHONPATH", "")] + python_paths
    ).strip(os.pathsep)
    env["LD_LIBRARY_PATH"] = str(lib_path)

    auto_setting = pangen_root / "shell" / "auto_setting.bash"
    if os.name != "nt" and auto_setting.exists():
        command = (
            "source {auto_setting} {pangen_root} && "
            "{pangen_bin} -script direct_pframe.py "
            "-e \"PYTHONPATH=$PYTHONPATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH\" "
            "-g {gateway}"
        ).format(
            auto_setting=str(auto_setting),
            pangen_root=str(pangen_root),
            pangen_bin=str(pangen_bin),
            gateway=config.gateway,
        )
        _run_with_logs(
            ["bash", "-lc", command],
            run_dir=run_dir,
            env=env,
            timeout_sec=config.timeout_sec,
        )
        return

    _run_with_logs(
        [
            str(pangen_bin),
            "-script", "direct_pframe.py",
            "-e", "PYTHONPATH={0} LD_LIBRARY_PATH={1}".format(
                env["PYTHONPATH"], env["LD_LIBRARY_PATH"]
            ),
            "-g", config.gateway,
        ],
        run_dir=run_dir,
        env=env,
        timeout_sec=config.timeout_sec,
    )


def _run_with_logs(command: list[str], *, run_dir: Path, env: dict[str, str], timeout_sec: int) -> None:
    stdout_path = run_dir / "pangen_stdout.log"
    stderr_path = run_dir / "pangen_stderr.log"
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, \
            stderr_path.open("w", encoding="utf-8", errors="replace") as stderr:
        try:
            completed = subprocess.run(
                command,
                cwd=str(run_dir),
                env=env,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                "PanGen binary timed out after {0}s. Logs: {1}, {2}".format(
                    timeout_sec, stdout_path, stderr_path
                )
            ) from exc
    if completed.returncode != 0:
        raise RuntimeError(
            "PanGen binary exited with code {0}. Logs: {1}, {2}".format(
                completed.returncode, stdout_path, stderr_path
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--backend", choices=["auto", "binary", "embedded", "local"], default="auto")
    parser.add_argument("--pangen-path", default="")
    parser.add_argument("--gateway", default="")
    parser.add_argument("--timeout-sec", type=int, default=3600)
    args = parser.parse_args()

    eval_input = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    result = run_direct_pangen(
        run_dir=args.run_dir,
        eval_input=eval_input,
        config=DirectPanGenConfig(
            backend=args.backend,
            pangen_path=args.pangen_path,
            gateway=args.gateway,
            timeout_sec=args.timeout_sec,
        ),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
