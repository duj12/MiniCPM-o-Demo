#!/usr/bin/env python3
"""`_shutdown_session` 的**卡死护栏**验收。

钉住的是一个真实发生过的故障（2026-10-08，105/106 都中过）：

    收尾流程走到 `for t in tasks: t.cancel()` 之后的 `await gather(...)`，
    其中一个后台任务**取消不掉**（阻塞在同步 I/O、吞掉 CancelledError、
    或卡在 finally 的 await），裸 gather 就永远等下去 —— 而 `sess.close()`
    在它后面。于是：

      · 远端人脸会话不释放（105：占着 `G1_FACE_MAX_SESSIONS` 槽位到 300s TTL）
      · 进程内 IC Engine 不关闭
      · `[sid] 会话结束` 永不打印
      · 而且**一行日志都没有** —— `close()` 的第一句才是第一条日志

    实测：105 上 10 个走到「强制关闭」的会话里卡死 2 个；106 旧 grpc 版卡过
    1 个，**且那两次收尾并没有重叠** —— 所以这与 IC 模式、与并发都无关。

为什么必须有这个测试：这个故障**非确定性**（约 1/5），靠真机反复跑根本
证明不了修好了没有 —— 修完跑几轮不出现，可能只是没抽中。这里用一个
**故意取消不掉的任务**把条件钉死：修复前这个测试会永久挂住（所以下面
用 `wait_for` 兜底并判失败），修复后必须能在超时内走完并打印是谁。

    python -m orchestrator.tests.test_session_shutdown
"""
from __future__ import annotations

import asyncio
import logging
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_OK = 0
_FAIL = 0


def check(cond: bool, msg: str) -> None:
    global _OK, _FAIL
    if cond:
        _OK += 1
        print(f"  ✓ {msg}")
    else:
        _FAIL += 1
        print(f"  ✗ {msg}")


class _Capture(logging.Handler):
    """抓 `orchestrator` 的日志，用来断言告警确实打了。"""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def text(self) -> str:
        return "\n".join(r.getMessage() for r in self.records)


class Immortal:
    """一个**取消不掉**的后台任务（模拟真实里那个卡住的任务）。

    `_stopped` 置位后再取消一次才会退出 —— 测试靠它把协程收干净，
    不留悬挂 task（否则 `asyncio.run` 收尾时也会被它挂住）。
    """

    def __init__(self) -> None:
        self._stopped = False

    async def run(self) -> None:
        while True:
            try:
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                if self._stopped:
                    raise
                # 故意不退 —— 这正是被钉住的那个故障
                continue

    def release(self) -> None:
        self._stopped = True


class FakeSess:
    """只具备 `_shutdown_session` 用到的那几个属性/方法。"""

    session_id = "sess-x"
    error = None

    def __init__(self) -> None:
        self.metrics = types.SimpleNamespace(**{
            k: 0 for k in ("ticks", "audio_chunks_in", "video_face_frames")
        })
        self.closed_reason: str | None = None

    async def drain(self, reason: str = "client_stop", **kw) -> None:
        return None

    async def close(self, reason: str = "client_stop") -> None:
        self.closed_reason = reason

    def summary(self) -> str:
        return "fake summary"


def main() -> int:
    from orchestrator import main as om

    cap = _Capture()
    olog = logging.getLogger("orchestrator")
    olog.addHandler(cap)
    olog.setLevel(logging.INFO)

    # 测试要快 —— 把放弃等待压到 0.5s（生产是 5s）
    om._TASK_CANCEL_TIMEOUT_S = 0.5
    cfg = types.SimpleNamespace(drain_timeout_s=0.2)

    print("== 1. 取消不掉的任务不能挡住 sess.close() ==")
    immortal = Immortal()

    async def stuck_case() -> FakeSess:
        bad = asyncio.ensure_future(immortal.run())
        good = asyncio.ensure_future(asyncio.sleep(0))   # 正常任务
        sess = FakeSess()
        # ⚠️ 修复前这里会**永久挂住**，所以用 wait_for 兜底
        await asyncio.wait_for(om._shutdown_session(sess, "sess-x", cfg,
                                                    [bad, good]), timeout=10)
        # 收干净：放它走
        immortal.release()
        bad.cancel()
        await asyncio.gather(bad, good, return_exceptions=True)
        return sess

    try:
        sess = asyncio.run(stuck_case())
    except asyncio.TimeoutError:
        check(False, "10s 内没走完 —— 卡住的任务仍然挡住了 close()（修复无效）")
        print(f"\n结果: {_OK} 通过, {_FAIL} 失败")
        return 1

    check(sess.closed_reason == "client_disconnect",
          f"卡住的任务没能挡住 close()（closed_reason={sess.closed_reason}）")
    txt = cap.text()
    check("取消后未退出" in txt, "打出了「取消后未退出」告警")
    check("Immortal.run" in txt or "immortal" in txt.lower(),
          "告警里**点出了是谁**（任务名从协程取，不是 Task-37）")
    print(f"     告警原文: {[l for l in txt.splitlines() if '取消后未退出' in l][:1]}")

    print("== 2. 任务都正常退出时不告警 ==")
    cap.records.clear()

    async def clean_case() -> FakeSess:
        t1 = asyncio.ensure_future(asyncio.sleep(5))
        t2 = asyncio.ensure_future(asyncio.sleep(5))
        sess = FakeSess()
        await asyncio.wait_for(om._shutdown_session(sess, "sess-y", cfg,
                                                    [t1, t2]), timeout=10)
        return sess

    sess2 = asyncio.run(clean_case())
    check(sess2.closed_reason == "client_disconnect", "正常路径照样 close()")
    check("取消后未退出" not in cap.text(), "没打无谓的告警")

    print("== 3. 任务名可读（卡住时靠它指认）==")

    async def _name_case() -> str:
        # ⚠️ 必须在**运行中的 loop 里**建任务：3.7/3.10 下在 loop 外调
        # `ensure_future` 会去 `get_event_loop()`，直接 RuntimeError。
        t = asyncio.ensure_future(asyncio.sleep(0))
        try:
            return om._describe_task(t)
        finally:
            await t

    name = asyncio.run(_name_case())
    check("sleep" in name, f"_describe_task 取到协程名（{name!r}）")

    olog.removeHandler(cap)
    print(f"\n结果: {_OK} 通过, {_FAIL} 失败")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
