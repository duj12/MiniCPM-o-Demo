#!/usr/bin/env python3
"""回归测试：`InProcessICClient.tick()` **不能攥住事件循环**。

## 被测的那个坑（2026-10-13，106 实测）

`Engine.tick()` 判 `YIELD`/`END` 时会经 `_notify_sinks` 调
`ExpressionSink.stop()` —— 同步 `POST /v1/stop`。进程内模式下那个
`/v1/stop` 打的是 **本进程自己**的端口，而唯一的服务者就是事件循环。
在循环上直呼 ⇒ 循环被这一行攥住 ⇒ 没人去握手 ⇒ 只能等满 1s 超时。

实测证据（`orch106-8100.log`）：

  · 84 次 `_ssl.c:1000 handshake operation timed out` ≈ `END(sop=24)` 51 次
    + `YIELD` 30 次 —— 正好是 tick 路径的全部停播调用
  · 每次冻结 0.6~1s，**且是进程级的**（同一毫秒三个会话同时报卡顿）
  · `/v1/stop` 成功 **0** 次 ⇒ IC 的「掐断正在出声的音频」从未生效

## 这个测试怎么搭

把那份结构**原样**搭出来：同一个事件循环上，既跑 `ic.tick()`，又跑一个
`asyncio.start_server` 当 `/v1/stop` 的服务端。然后用两支探针断言：

  1. **心跳** —— tick 期间循环是否还在跳（被攥住就停跳）
  2. **真到达** —— 请求是否真被服务（`sink.stops` / `failures` / 服务端收到）

最后还有一组**反向对照**：绕开 `ic.tick()` 直呼 `engine.tick()`（= 修复前
的写法），必须复现「循环停跳 + 超时报错」—— 否则这个测试证明不了什么。

跑法（远端，需要 `interaction` 包）：

    /home/dujing/miniconda3/envs/py310/bin/python \
        -m orchestrator.tests.test_ic_tick_offloop
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# `interaction` 包在共享树的 interactioncore/ 下 —— 105/106 上 pip editable
# 装过，容器/裸机上没有时补一条相对路径
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "interactioncore"))

_failures: list = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


class _StopProbe:
    """极简 HTTP/1.1 服务端 —— 只为了接住 `POST /v1/stop`。

    ⚠️ 它**必须跑在同一个事件循环上**（`asyncio.start_server`）——
    这正是线上那份自锁的结构：服务端和调用方抢同一个循环。换成
    `http.server` 起在线程里就没有这个耦合了，测试也就测不出东西。
    """

    def __init__(self, delay: float = 0.0) -> None:
        self.requests: list[tuple[str, float]] = []
        #: 回包前先晾一会儿 —— 好让「tick 期间循环仍在跳」这件事**可观测**。
        #: 不晾的话 tick 只要几毫秒，比心跳间隔（20ms）还短，心跳一拍照不到，
        #: 断言就变成了「没意义地必挂」（第一版就是这么挂的）。
        self.delay = delay
        self._server = None
        self.port = 0

    async def _handle(self, reader, writer) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("latin-1").split("\r\n")
            parts = lines[0].split(" ")
            path = parts[1] if len(parts) > 1 else ""
            hdrs = {}
            for ln in lines[1:]:
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    hdrs[k.strip().lower()] = v.strip()
            n = int(hdrs.get("content-length") or 0)
            if n:
                await reader.readexactly(n)
            self.requests.append((path, time.monotonic()))
            if self.delay:
                await asyncio.sleep(self.delay)
            writer.write(b"HTTP/1.1 200 OK\r\n"
                         b"Content-Length: 2\r\n"
                         b"Connection: close\r\n\r\nok")
            await writer.drain()
        except Exception:  # noqa: BLE001 — 对端提前关闭是预期内的
            pass
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass

    async def start(self) -> "_StopProbe":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def stop(self) -> None:
        self._server.close()
        await self._server.wait_closed()


def _arm_end(engine) -> None:
    """把状态推到「policy 必判 END」——`policy.py:409`：
    `mode != IDLE` 且 `absent_ms >= absent_end_ms(4000)`。
    """
    from interaction.state import SessionMode

    engine.state.session.mode = SessionMode.SPEAKING
    engine.state.session.absent_ms = 10_000


async def main() -> None:
    try:
        from interaction.runtime import Engine, ExpressionSink
    except ImportError as exc:
        print(f"[SKIP] 没装 interaction 包：{exc}")
        return

    from orchestrator.interaction.inprocess import InProcessICClient

    print("== 正向：ic.tick() 应当 off-loop ==")
    # 晾 200ms 再回 —— tick 必须在这一整段里**把循环让出来**
    probe = await _StopProbe(delay=0.2).start()
    print(f"  探针服务 127.0.0.1:{probe.port}（与调用方同一个事件循环）")

    sink = ExpressionSink(f"http://127.0.0.1:{probe.port}",
                          session_id="t1", timeout=1.0)
    engine = Engine(expression_sink=sink, agent_sink=None)
    _arm_end(engine)

    ic = InProcessICClient(session_id="t1", expression_url="", agent_url="")
    ic.available = True
    ic._engine = engine

    beats = 0

    async def heartbeat() -> None:
        nonlocal beats
        while True:
            await asyncio.sleep(0.02)
            beats += 1

    hb = asyncio.create_task(heartbeat())
    t0 = time.monotonic()
    action = await ic.tick(dt_ms=50)
    dt = (time.monotonic() - t0) * 1000
    hb.cancel()

    atype = getattr(getattr(action, "type", None), "value", None)
    print(f"  tick → {atype} sop={getattr(action, 'sop', None)}，"
          f"耗时 {dt:.0f}ms，期间心跳 {beats} 拍，"
          f"stops={sink.stops} failures={sink.failures} "
          f"服务端收到={len(probe.requests)}")

    check(getattr(action, "sop", None) == "24", "tick 判出 END(sop=24)")
    check(sink.stops == 1, f"ExpressionSink.stop() 确实被调到（stops={sink.stops}）")
    check(len(probe.requests) == 1,
          f"/v1/stop **真的到达并被执行**（服务端收到 {len(probe.requests)} 个请求）")
    check(sink.failures == 0, f"没有报错（failures={sink.failures}）")
    # ⚠️ 这条是**本测试的核心**：sink 调用在飞的整段时间里，循环必须还在转。
    #    修复前这里恒为 0（循环被攥住），修复后应接近 200ms/20ms ≈ 10 拍。
    check(beats >= 3, f"tick 期间事件循环仍在跑（心跳 {beats} 拍）")
    check(dt >= 150, f"探针的延迟确实生效了（{dt:.0f}ms ≥ 150ms）")

    await probe.stop()

    print("== 反向对照：直呼 engine.tick()（修复前的写法）必须自锁 ==")
    probe2 = await _StopProbe().start()
    sink2 = ExpressionSink(f"http://127.0.0.1:{probe2.port}",
                           session_id="t2", timeout=1.0)
    engine2 = Engine(expression_sink=sink2, agent_sink=None)
    _arm_end(engine2)

    beats2 = 0

    async def heartbeat2() -> None:
        nonlocal beats2
        while True:
            await asyncio.sleep(0.02)
            beats2 += 1

    hb2 = asyncio.create_task(heartbeat2())
    t0 = time.monotonic()
    engine2.tick(50)          # ← 直呼：攥住循环，无 await 点
    dt2 = (time.monotonic() - t0) * 1000
    hb2.cancel()

    print(f"  直呼耗时 {dt2:.0f}ms，心跳 {beats2} 拍，"
          f"failures={sink2.failures}，服务端收到={len(probe2.requests)}")

    check(sink2.failures == 1,
          f"反向对照：直呼确实超时（failures={sink2.failures}）")
    check(beats2 <= 2, f"反向对照：循环确实停跳了（心跳仅 {beats2} 拍）")
    check(dt2 >= 800, f"反向对照：确实卡满超时（{dt2:.0f}ms）")

    await probe2.stop()

    print()
    if _failures:
        print(f"❌ {len(_failures)} 项失败：")
        for f in _failures:
            print(f"   · {f}")
        sys.exit(1)
    print("✅ 全部通过：tick 已 off-loop，/v1/stop 真正生效")


if __name__ == "__main__":
    asyncio.run(main())
