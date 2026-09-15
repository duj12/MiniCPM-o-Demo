"""声学延迟的持久化 —— 让每次会话不必从零收敛。

## 为什么需要

参考轨需要**声学路径延迟 D** 才能把 farend 与麦克风里的回声对齐。D 只能
在 TTS 播放窗口内测（那时才有回声），而每次会话开头都要重新收敛 ——
**第一段播报期间用的是不准的 D**，消得不好。

把测得的值存下来，下次会话直接用作初值，第一段播报就是准的。

## 按什么键存

D 由三部分构成：网络往返、浏览器播放调度、设备声学路径。前两者随网络
环境变，第三者随设备变。用 ``client_key``（前端上报的设备/身份标识）
分组，同一设备复用同一份。

首次没有记录时用 ``default_ms``（默认取 ``playback_delay_ms``，因为
那是最主要的已知分量）。
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_PATH = os.environ.get(
    "ORCH_DELAY_STORE", os.path.expanduser("~/.orchestrator_delay.json")
)


class DelayStore:
    """``client_key -> 延迟(ms)`` 的持久化映射。

    线程安全（写入可能来自多个会话的协程）。
    """

    def __init__(self, path: Optional[str] = None,
                 default_ms: float = 250.0) -> None:
        self.path = Path(path or DEFAULT_PATH)
        self.default_ms = default_ms
        self._lock = threading.Lock()
        self._data: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._data = {str(k): v for k, v in raw.items()
                              if isinstance(v, dict)}
            logger.info("加载延迟记录 %d 条（%s）", len(self._data), self.path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("延迟记录加载失败（忽略）: %s", exc)

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self.path)   # 原子替换，避免写坏
        except Exception as exc:  # noqa: BLE001
            logger.warning("延迟记录保存失败（忽略）: %s", exc)

    def get(self, client_key: str) -> tuple:
        """返回 ``(delay_ms, source)``；``source`` ∈ stored|default。"""
        with self._lock:
            rec = self._data.get(client_key)
        if rec and isinstance(rec.get("delay_ms"), (int, float)):
            return float(rec["delay_ms"]), "stored"
        return float(self.default_ms), "default"

    def put(self, client_key: str, delay_ms: float, n_samples: int = 0) -> None:
        """记录一次测量。``n_samples`` 是该值的样本数（越多越可信）。"""
        if delay_ms <= 0 or delay_ms > 5000:
            return                       # 明显异常，不记
        with self._lock:
            prev = self._data.get(client_key) or {}
            # 新值与旧值差很大时不急着覆盖（可能是异常测量）——
            # 用滑动平均，样本多的旧值权重更高
            old = prev.get("delay_ms")
            if isinstance(old, (int, float)):
                delay_ms = 0.7 * float(old) + 0.3 * delay_ms
            self._data[client_key] = {
                "delay_ms": round(float(delay_ms), 1),
                "n_samples": int(prev.get("n_samples", 0)) + max(1, n_samples),
            }
            self._save()

    def snapshot(self) -> Dict[str, dict]:
        with self._lock:
            return dict(self._data)
