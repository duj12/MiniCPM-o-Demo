#!/usr/bin/env python3
"""进程内 IC 的**并发隔离**验收。

对应 `docs/多会话并发与云服务化-问题与方案.md` 的问题 ③：远端 IC 的
`Engine` / `InteractionState` 是进程级全局一份，两个会话同时驱动会互相
覆盖 `mode` / `barge_hold_ms` / `turn`，现象是「ASR 识别完美却零决策」。
`ORCH_IC_MODE=inprocess` 后每路会话独占一份 Engine，这个类故障应当
**从构造上**不可能发生。

为什么要有这个测试：隔离是「没有发生的事」，靠人工点两下很难证明。
这里把四条**具体可证伪**的性质钉住：

  1. 两路会话拿到的是**不同的 Engine 对象**
  2. 写进 A 的转写**不会**出现在 B（反过来也一样）
  3. tick 只推进**自己**的状态机
  4. `apply_agent` 只落到**本会话**（这是 `ORCH_IC_MODE=inprocess` 的
     硬前置 —— Agent 得能按 callback_ic 投回来）

    python -m orchestrator.tests.test_ic_inprocess_isolation
"""
from __future__ import annotations

import asyncio
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


def _load_interaction():
    """加载 `interaction` 包但**绕过** `interaction/__init__.py`。

    那份 `__init__.py` 会 import grpc（本机开发环境不一定装），而我们要用的
    `runtime` / `state` / `policy` 是纯 Python、不依赖 grpc。
    服务端环境里正常 `import interaction` 即可。
    """
    try:
        import interaction  # noqa: F401
        return
    except ImportError:
        pass
    root = Path(__file__).resolve().parents[3] / "interactioncore" / "interaction"
    pkg = types.ModuleType("interaction")
    pkg.__path__ = [str(root)]
    sys.modules["interaction"] = pkg


def main() -> int:
    _load_interaction()
    from orchestrator.interaction.inprocess import InProcessICClient
    from interaction.state import AgentStatus, Confidence

    print("== 1. 两路会话拿到不同的 Engine ==")
    # 不配 expression/agent 地址 —— 只想验证状态机隔离，
    # 不想真去打 HTTP（那会让测试依赖外部服务）。
    a = InProcessICClient("sess-A")
    b = InProcessICClient("sess-B")
    check(a.connect(), "会话 A 建起来")
    check(b.connect(), "会话 B 建起来")
    check(a.engine is not None and b.engine is not None, "两个 Engine 都非空")
    check(a.engine is not b.engine, "两个 Engine 是**不同对象**（隔离的前提）")

    print("== 2. 状态写入互不可见 ==")
    a.apply("apply_asr", transcript="会话A说的话",
            asr_confidence="HIGH", turn_complete_confidence="HIGH")
    b.apply("apply_asr", transcript="会话B说的话",
            asr_confidence="HIGH", turn_complete_confidence="HIGH")
    ta = a.engine.state.speech.transcript
    tb = b.engine.state.speech.transcript
    check(ta == "会话A说的话", f"A 的转写是自己的（{ta!r}）")
    check(tb == "会话B说的话", f"B 的转写是自己的（{tb!r}）")
    check(ta != tb, "两路转写没有互相覆盖 —— 这正是远端单例 IC 做不到的")

    # 换个人脸/身份再验一次（不同字段路径）
    a.apply("apply_identity", identity_id="uid-A", identity_confidence="HIGH",
            display_name="甲")
    check(b.engine.state.person.identity_id is None,
          "在 A 上写身份，B 的身份仍是 None")

    print("== 3. tick 只推进自己的状态机 ==")
    async def _ticks():
        for _ in range(3):
            await a.tick(50)
        return a.tick_calls, b.tick_calls
    ca, cb = asyncio.run(_ticks())
    check(ca == 3, f"A 的 tick_calls={ca}（应为 3）")
    check(cb == 0, f"B 的 tick_calls={cb}（应为 0 —— 没被 A 带着推进）")

    print("== 4. apply_agent 只落到本会话 ==")
    # ⚠️ 大小写两种写法都必须能落到 **AgentStatus 枚举**上。
    #    IC 里 `AgentStatus` 的值是**小写**（idle/busy/pending_announce），
    #    而 `policy.py` 是拿 `== AgentStatus.PENDING_ANNOUNCE` 判的 ——
    #    存大写字串进去不会报错，只会让 SOP 39 **静默不生效**。
    for written in ("PENDING_ANNOUNCE", "pending_announce", "Pending_Announce"):
        a.apply_agent(status=written)
        got = a.engine.state.agent.status
        check(got is AgentStatus.PENDING_ANNOUNCE,
              f"status={written!r} 归一成枚举成员（而不是裸字符串）")
        check(got == AgentStatus.PENDING_ANNOUNCE,
              f"status={written!r} 后 policy 的 == 能命中")
    check(b.engine.state.agent.status is AgentStatus.IDLE,
          "B 的 agent.status 仍是 IDLE（未被 A 污染）")

    print("== 5. UNSET 语义：省略字段 = 不改 ==")
    a.apply_agent(status="BUSY")
    check(a.engine.state.agent.session_end_pending is False,
          "只写 status 后 end_pending 是默认 False")
    a.apply_agent(session_end_pending=True)
    check(a.engine.state.agent.status is AgentStatus.BUSY,
          "只写 end_pending **不动** status（省略 = 不改）")
    check(a.engine.state.agent.session_end_pending is True, "end_pending 写进去了")

    print("== 6. 非法值被挡住，不炸也不静默改状态 ==")
    before = a.engine.state.agent.status
    a.apply_agent(status="BOGUS")
    check(a.engine.state.agent.status is before,
          "未知 status 被忽略（状态不变、未抛异常）")
    a.apply_agent()           # 全省略
    check(a.engine.state.agent.status is before, "全省略不炸、不改")

    print("== 7. 档位字符串必须落成 Confidence 枚举（否则 tick 会崩）==")
    #
    # ⚠️⚠️ 这一节是**真机回归**补上的，别删。
    #
    # 走 gRPC 时 `grpc_codec.conf_field` 已经把 proto 枚举转成了
    # `state.Confidence` **成员**；进程内路径直接调 `Engine.apply_*`，
    # 拿到的是编排侧 `_conf()` 给的**裸字符串** `'HIGH'`。
    #
    # 危险在于它**看起来是对的**：`Confidence` 是 str 枚举且值是大写的，
    # 所以 `'HIGH' == Confidence.HIGH` 为真、还带 `rank()`；但
    # `policy.py` 有四处是**调方法**的 ——
    # `face_present_confidence.at_least(Confidence.MEDIUM)` —— 裸字符串没有
    # `at_least`，于是 `AttributeError: 'str' object has no attribute
    # 'at_least'`，`tick()` 直接失败 ⇒ `_mark_dead` ⇒ **整个会话无决策**。
    #
    # 为什么原测试没抓到：只 `apply_asr` 就 tick，而那条路上 `at_least`
    # 恰好没被走到。真会话会 `apply_face`/`apply_vad`/`apply_lip`，
    # 一 tick 就炸。所以这里**专门**按 downstream 的真实调法写一遍。
    c = InProcessICClient("sess-conf")
    check(c.connect(), "第三个会话建起来")
    c.apply("apply_face", face_present_confidence="HIGH", bbox_area_ratio=0.12)
    c.apply("apply_track", track_id="7", dwell_ms=2500)
    c.apply("apply_lip", lip_speaking_confidence="LOW")
    c.apply("apply_identity", identity_id="uid-x",
            identity_confidence="HIGH", display_name="甲")
    c.apply("apply_vad", user_speaking_confidence="HIGH",
            barge_in_confidence="MEDIUM")
    c.apply("apply_asr", transcript="你好", asr_confidence="HIGH",
            turn_complete_confidence="HIGH")

    stored = [
        ("person.face_present_confidence",
         c.engine.state.person.face_present_confidence),
        ("person.lip_speaking_confidence",
         c.engine.state.person.lip_speaking_confidence),
        ("person.identity_confidence",
         c.engine.state.person.identity_confidence),
        ("speech.user_speaking_confidence",
         c.engine.state.speech.user_speaking_confidence),
        ("speech.barge_in_confidence",
         c.engine.state.speech.barge_in_confidence),
        ("speech.asr_confidence", c.engine.state.speech.asr_confidence),
    ]
    for label, val in stored:
        check(isinstance(val, Confidence),
              f"{label} 存的是 Confidence 枚举（实际 {type(val).__name__}）")
        check(hasattr(val, "at_least"),
              f"{label} 有 .at_least() —— policy 就是靠它判的")

    # 真正会炸的那一下：按真实写入顺序连 tick 几十拍
    async def _drive():
        return [await c.tick(50) for _ in range(30)]
    acts = asyncio.run(_drive())
    check(c.bad_ticks == 0,
          f"连 tick 30 拍没有失败（bad_ticks={c.bad_ticks} error={c.error}）")
    check(c.error is None, "tick 未把客户端标记为失效")
    kinds = [str(getattr(a.type, "value", a.type)) for a in acts if a]
    check("ANSWER" in kinds,
          f"整条决策链能出 ANSWER（实际出现：{sorted(set(kinds))}）")
    c.close()

    print("== 8. 生命周期 ==")
    check(a.reset_session() is True, "reset_session 可用")
    check(a.end_session() is True, "end_session 可用（含同步 /v1/stop，需 off-loop）")
    a.close()
    b.close()
    check(True, "close 干净退出")

    print(f"\n结果: {_OK} 通过, {_FAIL} 失败")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
