"""人脸工作线程 —— 专用线程 + 有界队列，**绝不阻塞音频路径**。

为什么用专用 ``threading.Thread`` 而不是 ``run_in_executor``：
  · ``queue.Queue.get(timeout=...)`` 提供干净的可取消语义
  · 共享线程池会让一路会话的 200ms 帧饿死另一路，破坏跟踪时序

背压策略：队列 ``maxsize=3``，满则**丢最旧**帧（新帧对唇动状态更有价值，
40ms 前的陈旧帧没有意义），并计数。**永不 await、永不阻塞 mic ingest**。

唇动信号以 **10Hz** 发出（每 100ms 音频窗聚合一次），但 ``False→True``
的说话跳变立即发出（barge-in 边沿）。
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, List, Optional

from .signals import (
    FaceObservation,
    FrameStats,
    IdentityEvent,
    LipEvent,
    WakeEvent,
    aggregate_lip,
)

logger = logging.getLogger(__name__)

# 唇动聚合窗口（对齐音频网格）
LIP_WINDOW_SAMPLES = 1600   # 100ms @16k
SR = 16000


class FaceWorker:
    """在专用线程里跑 G1 人脸库。

    ``provider`` 需实现 ``process(jpeg, t) -> FaceObservation`` 与可选
    ``identify(frames) -> IdentityEvent``。见 ``LocalFaceProvider``。
    """

    def __init__(self, provider, on_wake: Callable[[WakeEvent], None],
                 on_lip: Callable[[LipEvent], None],
                 on_identity: Optional[Callable[[IdentityEvent], None]] = None,
                 on_obs: Optional[Callable[[FaceObservation], None]] = None,
                 on_state: Optional[Callable[[dict], None]] = None,
                 queue_maxsize: int = 3,
                 lip_immediate_edge: bool = True) -> None:
        self.provider = provider
        self.on_wake = on_wake
        self.on_lip = on_lip
        self.on_identity = on_identity
        # on_obs：每帧的原始观测，**仅供 UI 叠加显示**。
        # 不进 downstream（那是控制流，25Hz 会把下游淹没）。
        self.on_obs = on_obs
        # on_state：G1 的每帧 state，**只在刷新时发**（≈208ms 一次，不是每帧）。
        # 与 on_obs 的区别就是这一点 —— 想要"每帧都有"的东西请用 on_obs。
        self.on_state = on_state
        self.lip_immediate_edge = lip_immediate_edge

        self._q: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=queue_maxsize)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.stats = FrameStats()

        # 状态机
        self._was_interacting = False
        self._wake_started_at: Optional[float] = None
        self._peak_conf = 0.0
        self._conf_sum = 0.0
        self._conf_n = 0
        # 唇动聚合
        self._lip_buf: List[FaceObservation] = []
        self._lip_win_start: Optional[int] = None
        self._last_speaking = False
        # 上次发出去的 state_seq，用来判断 state 是否刷新
        self._last_state_seq = -1

    # ------------------------------------------------------------------ #

    def offer(self, jpeg: bytes, t: int) -> None:
        """投递一帧（非阻塞）。队列满则丢**最旧**帧。"""
        try:
            self._q.put_nowait((jpeg, t))
        except queue.Full:
            try:
                self._q.get_nowait()          # 丢最旧
                self._q.put_nowait((jpeg, t))
            except queue.Empty:
                pass
            self.stats.frames_dropped += 1

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="face-worker", daemon=True
        )
        self._thread.start()
        logger.info("人脸线程已启动")

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)   # 唤醒阻塞的 get
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("人脸线程未在 %.1fs 内退出", timeout)
            self._thread = None
        try:
            self.provider.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("provider.close 异常: %s", exc)
        logger.info(
            "人脸线程已停止: in=%d processed=%d dropped=%d wakes=%d identifies=%d",
            self.stats.frames_in, self.stats.frames_processed,
            self.stats.frames_dropped, self.stats.wakes, self.stats.identifies,
        )

    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                self._flush_lip_window()
                continue
            if item is None:
                break
            jpeg, t = item
            self.stats.frames_in += 1
            try:
                obs = self.provider.process(jpeg, t)
            except Exception as exc:  # noqa: BLE001
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("人脸处理失败: %s", exc)
                continue
            if obs is None:
                continue
            self.stats.frames_processed += 1
            self._handle_observation(obs)

        self._flush_lip_window()

    def _handle_observation(self, obs: FaceObservation) -> None:
        # ---- UI 叠加（每帧，仅显示用）----
        if self.on_obs is not None:
            self._safe(self.on_obs, obs)

        # ---- 每帧 state（**只在刷新时发**）----
        # G1 的 state 心跳绑在 landmark 上（≈208ms），其余帧沿用上次快照、
        # `state_seq` 不变。用 seq 去重，避免把同一份 state 重复推 5 次。
        if (self.on_state is not None and obs.state is not None
                and obs.state_seq != self._last_state_seq):
            self._last_state_seq = obs.state_seq
            self._safe(self.on_state, obs.state)

        # ---- 唤醒 ----
        if obs.interacting and not self._was_interacting:
            self._was_interacting = True
            self._wake_started_at = time.monotonic()
            self._peak_conf = obs.score
            self._conf_sum = obs.score
            self._conf_n = 1
            self.stats.wakes += 1
            self._safe(self.on_wake, WakeEvent(
                t=obs.t, phase="begin", dwell_ms=0,
                mean_confidence=obs.score, peak_confidence=obs.score,
                box=obs.box,
            ))
        elif obs.interacting:
            self._peak_conf = max(self._peak_conf, obs.score)
            self._conf_sum += obs.score
            self._conf_n += 1
        elif self._was_interacting:
            self._was_interacting = False
            dwell_ms = int((time.monotonic() - (self._wake_started_at or 0)) * 1000)
            self._safe(self.on_wake, WakeEvent(
                t=obs.t, phase="end", dwell_ms=dwell_ms,
                mean_confidence=(self._conf_sum / self._conf_n) if self._conf_n else 0.0,
                peak_confidence=self._peak_conf, box=obs.box,
            ))

        # ---- 身份（provider 内部攒批/触发，这里只消费）----
        ident = getattr(self.provider, "poll_identity", None)
        if ident is not None:
            try:
                ev = ident(obs.t)
                if ev is not None:
                    self.stats.identifies += 1
                    if self.on_identity:
                        self._safe(self.on_identity, ev)
            except Exception as exc:  # noqa: BLE001
                logger.warning("身份识别失败: %s", exc)

        # ---- 唇动 ----
        self._push_lip(obs)

    def _push_lip(self, obs: FaceObservation) -> None:
        """按 100ms 音频窗聚合唇动状态。"""
        # 说话跳变立即发（barge-in 边沿，100ms 很关键）
        if self.lip_immediate_edge and obs.speaking and not self._last_speaking:
            self._last_speaking = True
            self._flush_lip_window(force_edge=True)
            self._safe(self.on_lip, LipEvent(
                t0=obs.t, t1=obs.t, speaking=True, lip_state=obs.lip_state,
                confidence=obs.score,
            ))
        self._last_speaking = obs.speaking

        w = obs.t // LIP_WINDOW_SAMPLES
        if self._lip_win_start is None:
            self._lip_win_start = w
        if w != self._lip_win_start:
            self._flush_lip_window()
            self._lip_win_start = w
        self._lip_buf.append(obs)

    def _flush_lip_window(self, force_edge: bool = False) -> None:
        if not self._lip_buf:
            return
        buf, self._lip_buf = self._lip_buf, []
        if force_edge:
            return  # 边沿事件已单独发出，不重复
        t0 = (self._lip_win_start or 0) * LIP_WINDOW_SAMPLES
        ev = aggregate_lip(buf, t0, t0 + LIP_WINDOW_SAMPLES)
        if ev is not None:
            self._safe(self.on_lip, ev)

    @staticmethod
    def _safe(fn, arg) -> None:
        try:
            fn(arg)
        except Exception as exc:  # noqa: BLE001
            logger.warning("人脸回调异常: %s", exc)
