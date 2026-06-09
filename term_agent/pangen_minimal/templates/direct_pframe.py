import json
import pangen.system as psys


def _load_config():
    with open("direct_pangen_config.json", "r") as f:
        return json.load(f)


def PanGenFlow():
    cfg = _load_config()
    options = {
        "python_file": "direct_task.py",
        "use_gpu": "1" if cfg.get("use_gpu", True) else "0",
        "worker_count": str(cfg.get("worker_count", 1)),
        "preprocess_thread_count": str(cfg.get("preprocess_threads", 4)),
        "ga_need_save_model": "1",
        "ga_total_rounds": str(cfg.get("ga_rounds", 3)),
        "pop_size": str(cfg.get("pop_size", 20)),
    }
    psys.execute_session(
        "ga",
        mode=cfg.get("mode", "distributed"),
        session_name="ga_calibrate_1",
        options=options,
        worker_nodes=cfg.get("worker_nodes", {}),
    )


if __name__ == "__main__":
    PanGenFlow()
