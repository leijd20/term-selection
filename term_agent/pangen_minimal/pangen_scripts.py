"""Copy PanGen-compatible script templates into a run directory."""

from __future__ import annotations

import shutil
from pathlib import Path


TEMPLATE_DIR = Path(__file__).with_name("templates")


def write_pangen_scripts(run_dir: Path) -> None:
    shutil.copy2(TEMPLATE_DIR / "direct_pframe.py", run_dir / "direct_pframe.py")
    shutil.copy2(TEMPLATE_DIR / "direct_task.py", run_dir / "direct_task.py")
