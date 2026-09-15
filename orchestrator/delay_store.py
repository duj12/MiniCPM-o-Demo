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

首次没有记录时用 ``default_ms``（默认 250，与
``config.aec_default_delay_ms`` 一致）。
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
        # ⚠️ 0 是有效值（"假定浏览器按约定提前量起播"），不能被 `or` 吞掉
        self.default_ms = 0.0 if default_ms is None else float(default_ms)
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

    # 合理区间：声学+网络延迟实测在 100~500ms 量级。超出这个范围的
    # 测量值多半是互相关选错了峰（对准了语音的次级峰），存下来会持续
    # 污染后续会话 —— 宁可不记。
    MAX_REASONABLE_MS = 700.0
    MIN_REASONABLE_MS = 10.0

    # 新旧值相差超过这个量（毫秒）就判定旧值不可信，直接覆盖而不做滑动
    # 平均。150ms 远大于正常测量的抖动（实测量级 ±20ms），又小于"错值"
    # 与"真值"的典型差距（如 250 默认值 vs 84ms 真值 = 166ms）。
    RESET_DELTA_MS = 150.0

    def put(self, client_key: str, delay_ms: float, n_samples: int = 0) -> bool:
        """记录一次测量。``n_samples`` 是该值的样本数（越多越可信）。

        返回是否采纳。超出合理区间的值**不记录** —— 实测遇到过
        820ms 这种明显选错峰的结果被存下来，之后每次会话都用它，
        回声完全消不掉。
        """
        if not (self.MIN_REASONABLE_MS <= delay_ms <= self.MAX_REASONABLE_MS):
            logger.warning(
                "延迟测量值 %.0fms 超出合理区间 [%.0f, %.0f]，不予记录"
                "（多半是互相关选错峰）",
                delay_ms, self.MIN_REASONABLE_MS, self.MAX_REASONABLE_MS,
            )
            return False
        with self._lock:
            prev = self._data.get(client_key) or {}
            # 新值与旧值接近时用滑动平均（抑制单次测量的抖动）。
            #
            # ⚠️ 但**差得远时必须直接覆盖**：旧值本身可能是错的（早先校准
            # 通道有 3× 采样率错配 + 凭空减 200ms 两个 bug，产出的值系统性
            # 偏移；也可能来自那个"把全程往返当残差"的 250 默认值）。滑动
            # 平均会让这种错值以 0.7 的权重长期把正确的新值拖住 —— 表现为
            # "校准了但没变化"。
            old = prev.get("delay_ms")
            if isinstance(old, (int, float)):
                if abs(float(old) - delay_ms) > self.RESET_DELTA_MS:
                    logger.warning(
                        "延迟记录 %s 旧值 %.0fms 与新值 %.0fms 相差超过 "
                        "%.0fms —— **直接覆盖**（旧值多半来自有缺陷的测量）",
                        client_key, float(old), delay_ms,
                        self.RESET_DELTA_MS,
                    )
                    prev = {**prev, "n_samples": 0}
                else:
                    delay_ms = 0.7 * float(old) + 0.3 * delay_ms
            self._data[client_key] = {
                "delay_ms": round(float(delay_ms), 1),
                "n_samples": int(prev.get("n_samples", 0)) + max(1, n_samples),
            }
            self._save()
            return True

    def snapshot(self) -> Dict[str, dict]:
        with self._lock:
            return dict(self._data)
