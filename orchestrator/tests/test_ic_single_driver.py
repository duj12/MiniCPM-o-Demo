#!/usr/bin/env python3
"""IC 单一驱动者：并发会话重叠时，只有一个会话在驱动 IC。

**背景**（实测踩过的真实故障）：

  会话**收尾很慢**（等 ASR/Omni drain，实测 10~20 秒），而新会话立刻就
  起来了 —— 这段重叠**必然发生**。而 IC 的服务端状态是**全局一份**、
  没有 session 概念，于是两个会话会同时驱动它：

    · 两者的 `tick()` 一起推进同一个状态机 → `dt` 累加翻倍
    · 各自的 `apply_action` 互相覆盖 `mode` / `barge_hold_ms`
    · 一方写进去的 `turn=HIGH`，可能被另一方的 `apply_action` 先冲掉

  现象：**ASR 识别完美、也送进 IC 了，却没有任何回复**；而 IC 自己的日志里
  明明出过 ANSWER（被另一路的 tick 领走了）。

修法：`InteractionClient` 加「单一驱动者」闸 —— 新会话 `connect()` 时
`activate()` 接管，旧实例被挂起（`apply`/`tick`/`snapshot` 全部短路）。

**测试方式**：每个 client 配一个**独立**的假后端（记录自己被调用了几次）。
「谁在驱动 IC」= 「谁的后端被调到了」—— 这样不用全局变量打标，
也不会把测试写成"看调用方"那种自欺欺人的形式。

    python -m orchestrator.tests.test_ic_single_driver
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_failures: list = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


class _FakeBackend:
    """假 IC 后端：只记录**自己**被驱动了几次。每个 client 一个。"""

    def __init__(self, owner: str) -> None:
        self.owner = owner
        self.ticks: list = []
        self.applies: list = []        # 收到的 apply_* 调用（方法名）
        self.snapshot_calls: int = 0

    def tick(self, dt_ms=0):
        self.ticks.append(dt_ms)
        return None

    # `_run` 用 getattr(self._client, method) 取方法 —— 动态接住 apply_*，
    # 免得到测试里为每个 IC 方法都写一个桩。
    def __getattr__(self, name):
        if name.startswith("apply_"):
            def _rec(**kwargs):
                self.applies.append(name)
            return _rec
        raise AttributeError(name)

    def get_snapshot(self):
        self.snapshot_calls += 1
        return {}

    def close(self):
        pass


def _patch_to_thread() -> None:
    """本地 py3.7 没有 ``asyncio.to_thread``（3.9+）—— 补一个直调版本。

    生产代码用的是真 ``to_thread``（让出事件循环）；测试里假后端是同步的，
    直接调等价。**只改测试进程，不动生产代码。**
    """
    import asyncio
    if not hasattr(asyncio, "to_thread"):
        async def _to_thread(fn, *a, **kw):
            return fn(*a, **kw)
        asyncio.to_thread = _to_thread       # type: ignore[attr-defined]


def _mk_client(owner: str):
    """造一个绕开真实 gRPC 的 InteractionClient，返回 (client, backend)。"""
    _patch_to_thread()
    from orchestrator.interaction.client import InteractionClient
    backend = _FakeBackend(owner)
    c = InteractionClient("fake:0", owner_key=owner)
    c.available = True
    c._client = backend
    return c, backend


def _reset_owner() -> None:
    from orchestrator.interaction import client as cm
    cm._OWNER = None


def test_new_session_takes_over() -> None:
    """新会话 activate 后，旧会话被挂起。"""
    print("\n[单驱动者] 新会话接管，旧会话挂起")
    _reset_owner()
    old, _ = _mk_client("session-A")
    new, _ = _mk_client("session-B")

    old.activate()
    check(not old.suspended, "A 先接管 → 未挂起")
    new.activate()
    check(new.suspended is False, "B 接管后自己未挂起")
    check(old.suspended is True, "A 被挂起")


def test_suspended_apply_is_ignored() -> None:
    """挂起的实例写状态**被丢弃**（不能覆盖新会话的）。"""
    print("\n[单驱动者] 挂起后 apply 被丢弃")
    _reset_owner()
    old, _ = _mk_client("session-A")
    old.activate()

    old.apply("apply_asr", transcript="旧会话的转写")
    check(old._q.qsize() == 1, "未挂起时 apply 正常入队")

    old.suspended = True
    old.apply("apply_asr", transcript="不该进队")
    check(old._q.qsize() == 1, "挂起后 apply **不再入队**（队列没变）")


def test_queued_writes_dropped_after_takeover() -> None:
    """**接管前入队、接管后才被消费**的过期写入必须丢掉 —— 闸门第二道。

    ⚠️ 这是实测踩过的一个真实漏洞：`apply()` 里的检查只管「入队那一刻」，
    而队列是**异步消费**的。旧会话在被接管**之前**投进去的条目还压在队列
    里，接管之后才被消费线程取出 —— 只查入队时刻的话，这些**过期条目照发
    不误**。

    实测现象：两个被挂起的旧会话仍在往 IC 写空转写帧，把新会话刚写进去的
    `text='你好呀，你是谁呀？'` 冲掉 → 新会话拿不到自己的转写 → 永远出不了
    ANSWER。**三个会话全部零决策**。
    """
    print("\n[单驱动者] 接管前入队的写入，接管后被丢弃（闸门第二道）")
    _reset_owner()
    old, backend = _mk_client("old-session")
    old.activate()

    # ① 接管**之前**入队（此时闸门放行）
    old.apply("apply_asr", transcript="接管前投的")
    check(old._q.qsize() == 1, "接管前 apply 正常入队")

    # ② 此时被接管（模拟新会话建立）
    old.suspended = True

    # ③ 起真正的消费线程（生产路径就是它）—— 过期条目不该发到 IC
    import threading
    old._stop.clear()
    old._thread = threading.Thread(target=old._run, name="ic-apply-test",
                                   daemon=True)
    old._thread.start()
    import time
    time.sleep(0.5)                     # 让它把队列里那条消费掉
    old._stop.set()
    try:
        old._q.put_nowait(None)         # 唤醒并退出
    except Exception:                   # noqa: BLE001
        pass
    old._thread.join(timeout=2.0)

    check(backend.applies == [],
          f"过期条目**没发到 IC**（实际 {backend.applies}）")
    check(old.suspended_writes == 1,
          f"记了 1 次「被接管丢弃写」（实际 {old.suspended_writes}）")


def test_suspended_tick_does_not_touch_ic() -> None:
    """挂起的实例 `tick()` **不调到 IC** —— 最关键的一条。

    `tick()` 在 IC 侧会推进状态机（不是只读），两个会话同时 tick 就是
    两个驱动者踩同一个状态机。
    """
    print("\n[单驱动者] 挂起后 tick 不驱动 IC（关键）")
    _reset_owner()
    client, backend = _mk_client("session-A")
    client.activate()

    async def run():
        r1 = await client.tick(50)          # 未挂起 → 真调
        client.suspended = True
        r2 = await client.tick(50)          # 挂起 → 不调
        r3 = await client.tick(50)
        return r1, r2, r3

    r1, r2, r3 = asyncio.run(run())
    check(r1 is None and r2 is None and r3 is None, "tick 返回 None（无决策）")
    check(len(backend.ticks) == 1,
          f"后端只被 tick 1 次（挂起后的 2 次没到）—— 实际 {len(backend.ticks)}")
    check(client.suspended_ticks == 2, "记录了 2 次挂起期间的 tick（诊断用）")


def test_suspended_snapshot_returns_none() -> None:
    """挂起后 snapshot 返回 None（此时 IC 里是**别人的**状态）。"""
    print("\n[单驱动者] 挂起后 snapshot 返回 None")
    _reset_owner()
    client, backend = _mk_client("session-A")
    client.activate()
    check(asyncio.run(client.snapshot()) is not None, "未挂起时能读")

    client.suspended = True
    check(asyncio.run(client.snapshot()) is None, "挂起后返回 None")
    check(backend.snapshot_calls == 1, "挂起后没再调后端")


def test_close_releases_ownership() -> None:
    """close() 释放所有权，且**不自动移交给别人**。"""
    print("\n[单驱动者] close 释放所有权")
    _reset_owner()
    from orchestrator.interaction import client as cm
    client, _ = _mk_client("session-A")
    client.activate()
    check(cm._OWNER is client, "activate 后成为 OWNER")

    client.close()
    check(cm._OWNER is None, "close 后 OWNER 清空（不自动移交）")


def test_two_overlapping_sessions_only_one_drives() -> None:
    """**核心场景**：模拟「旧会话收尾中、新会话起来」的重叠窗口。

    期望：接管之后，只有新会话的 tick 能到达 IC。
    """
    print("\n[单驱动者] 重叠窗口：只有一个会话在驱动 IC")
    _reset_owner()
    old, old_be = _mk_client("old-session")
    new, new_be = _mk_client("new-session")

    # ① 旧会话已在跑（收尾中，但 tick 还没停）
    old.activate()
    old.apply("apply_asr", transcript="旧会话")
    asyncio.run(old.tick(50))
    check(len(old_be.ticks) == 1, "接管前：旧会话能 tick 到 IC")

    # ② 新会话起来 → 接管
    new.activate()
    new.apply("apply_asr", transcript="你好啊。")

    # ③ 之后两个会话各自继续 tick（旧会话的收尾协程还没退）
    async def both_tick():
        await old.tick(50)
        await new.tick(50)
        await old.tick(50)
        await new.tick(50)

    asyncio.run(both_tick())

    check(len(old_be.ticks) == 1,
          f"接管后旧会话**再也 tick 不到 IC**（仍是 1 次，实际 {len(old_be.ticks)}）")
    check(len(new_be.ticks) == 2,
          f"接管后全是新会话在 tick（2 次，实际 {len(new_be.ticks)}）")

    # ④ 新会话的转写正常入队；旧会话的队列不再增长
    queued = []
    while not new._q.empty():
        queued.append(new._q.get_nowait())
    check(len(queued) == 1 and queued[0][1]["transcript"] == "你好啊。",
          "新会话的转写正常入队")
    check(old._q.qsize() == 1,
          f"旧会话队列只有接管前那一条（实际 {old._q.qsize()}）")


def main() -> int:
    print("=" * 68)
    print("IC 单一驱动者验证")
    print("-" * 68)
    test_new_session_takes_over()
    test_suspended_apply_is_ignored()
    test_queued_writes_dropped_after_takeover()
    test_suspended_tick_does_not_touch_ic()
    test_suspended_snapshot_returns_none()
    test_close_releases_ownership()
    test_two_overlapping_sessions_only_one_drives()
    print("=" * 68)
    if _failures:
        print(f"FAILED: {len(_failures)} 项")
        for f in _failures:
            print("  -", f)
        return 1
    print("通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
