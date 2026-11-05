"""Host integration of local InteractionCore and the independent Agent service."""
import sys
from pathlib import Path

# InteractionCore 以**源码目录**方式引用（`vendor/interactioncore`），
# 不要求 `pip install -e ./vendor/interactioncore`。
#
# ⚠️ 环境里若还装着旧的 editable `interaction`，它的 `__editable__` finder
#    位于 `sys.meta_path`，**先于** `sys.path` 生效 —— 这里插路径也盖不过，
#    必须先 `pip uninstall interaction`。run_orch_*.sh 已经在启动时检查。
_VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "interactioncore"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from .agent import AgentClient  # noqa: E402
from .inprocess import InProcessICClient  # noqa: E402
from .downstream import InteractionDownstream  # noqa: E402

__all__ = ["AgentClient", "InProcessICClient", "InteractionDownstream"]
