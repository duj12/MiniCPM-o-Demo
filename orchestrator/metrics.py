"""会话与全局指标。

**生产环境没有观测就是瞎子** —— 线上出问题时，没有 ERLE / TTFT / 队列
丢弃数就没法定位。本模块提供轻量的计数与延迟统计，不引入 Prometheus
依赖（可后续接 `/metrics` 导出）。

设计：
  · ``LatencyLedger`` —— 按阶段记录延迟（首窗、TTFT、TTS 首帧等），
    用滑动分位数而非全量保留，避免长会话内存增长
  · 计数器是**单调递增**的，便于跨会话聚合
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


class LatencyTracker:
    """滑动窗口的延迟统计（毫秒）。

    保留最近 ``window`` 个样本，算 min/median/p95/max。
    比保留全量省内存，且更能反映**近期**表现（长会话里早期样本无意义）。
    """

    __slots__ = ("name", "_samples", "count", "total_ms")

    def __init__(self, name: str, window: int = 512) -> None:
        self.name = name
        self._samples: Deque[float] = deque(maxlen=window)
        self.count = 0
        self.total_ms = 0.0

    def record(self, ms: float) -> None:
        self._samples.append(ms)
        self.count += 1
        self.total_ms += ms

    def stats(self) -> Dict[str, float]:
        if not self._samples:
            return {"count": 0}
        s = sorted(self._samples)
        n = len(s)
        return {
            "count": self.count,
            "mean_ms": round(self.total_ms / max(self.count, 1), 2),
            "min_ms": round(s[0], 2),
            "p50_ms": round(s[n // 2], 2),
            "p95_ms": round(s[min(int(n * 0.95), n - 1)], 2),
            "max_ms": round(s[-1], 2),
        }


class SessionMetrics:
    """单会话的指标集合。"""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.t_start = time.monotonic()

        # 延迟
        self.aec_first_window = LatencyTracker("aec_first_window")
        self.asr_first_partial = LatencyTracker("asr_first_partial")
        self.asr_final = LatencyTracker("asr_final")
        self.tts_first_frame = LatencyTracker("tts_first_frame")
        self.tts_total = LatencyTracker("tts_total")
        self.omni_ttft = LatencyTracker("omni_ttft")
        self.face_frame = LatencyTracker("face_frame")

        # 计数
        self.counters: Dict[str, int] = {}

    def inc(self, name: str, n: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + n

    def observe(self, name: str, value: float) -> None:
        self.counters[name] = max(self.counters.get(name, 0), int(value))

    @property
    def uptime_s(self) -> float:
        return time.monotonic() - self.t_start

    def snapshot(self) -> Dict:
        return {
            "session_id": self.session_id,
            "uptime_s": round(self.uptime_s, 1),
            "counters": dict(self.counters),
            "latency": {
                "aec_first_window": self.aec_first_window.stats(),
                "asr_first_partial": self.asr_first_partial.stats(),
                "asr_final": self.asr_final.stats(),
                "tts_first_frame": self.tts_first_frame.stats(),
                "tts_total": self.tts_total.stats(),
                "omni_ttft": self.omni_ttft.stats(),
                "face_frame": self.face_frame.stats(),
            },
        }

    def summary_line(self) -> str:
        """一行摘要（日志用）。"""
        c = self.counters
        aec_lat = self.aec_first_window.stats()
        tts_lat = self.tts_total.stats()
        return (
            f"up={self.uptime_s:.0f}s "
            f"audio_in={c.get('audio_chunks', 0)}chk/"
            f"{c.get('audio_samples', 0)/16000:.0f}s "
            f"aec={c.get('aec_segments', 0)}seg"
            f"{'(首窗%.0fms)' % aec_lat['p50_ms'] if aec_lat.get('p50_ms') else ''} "
            f"asr={c.get('asr_partials', 0)}p/{c.get('asr_finals', 0)}f "
            f"omni={c.get('omni_deltas', 0)}d/{c.get('omni_dones', 0)}done "
            f"tts={c.get('tts_calls', 0)}call"
            f"{'(总%.0fms)' % tts_lat['p50_ms'] if tts_lat.get('p50_ms') else ''} "
            f"face={c.get('face_frames', 0)}frm "
            f"drop={c.get('down_dropped', 0)}"
        )


class GlobalMetrics:
    """跨会话聚合。"""

    def __init__(self) -> None:
        self.sessions_started = 0
        self.sessions_ended = 0
        self.sessions_failed = 0
        self.total_audio_s = 0.0
        self.total_tts_s = 0.0
        self._recent: Deque[Dict] = deque(maxlen=20)

    def on_start(self) -> None:
        self.sessions_started += 1

    def on_end(self, m: SessionMetrics, failed: bool = False) -> None:
        self.sessions_ended += 1
        if failed:
            self.sessions_failed += 1
        self.total_audio_s += m.counters.get("audio_samples", 0) / 16000.0
        self.total_tts_s += m.counters.get("tts_audio_s", 0)
        self._recent.append(m.snapshot())

    def snapshot(self) -> Dict:
        return {
            "sessions_started": self.sessions_started,
            "sessions_ended": self.sessions_ended,
            "sessions_failed": self.sessions_failed,
            "total_audio_s": round(self.total_audio_s, 1),
            "total_tts_s": round(self.total_tts_s, 1),
            "recent": list(self._recent)[-5:],
        }


GLOBAL = GlobalMetrics()
