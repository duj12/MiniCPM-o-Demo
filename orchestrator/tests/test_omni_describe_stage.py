#!/usr/bin/env python3
"""VLM 描述随 ASR 一起发给 Agent —— **纯离线端到端验证**（不连任何服务）。

## 验的是什么

发给 Agent 的 `on_answer` 里原本**只有 ASR 转写**。本次改动让它多带一份
OmniLLM 的**画面/语音描述**（`content: {ASR, VLM}`），而 interactioncore
**一行没改** —— 靠的是编排侧给 `AgentSink` 包一层子类覆写 `_send`
（见 `orchestrator/interaction/agent_sink.py`）。

本脚本用**真组件**跑：真 `InteractionDownstream`（inprocess）+ 真 IC `Engine`
+ 真 `VlmAgentSink`（打到本机的捕获端口）。只有 OmniLLM 换成假的
（`FakeOmni`：记下收到的指令，done 由测试手工回灌 —— 真模型跑不动、
也不该在单元测试里跑）。

## 硬指标（本次改动的主指标）

**`AsrFinal`（`2pass-offline`，即 offline 文本）到达时，VLM 描述已经在缓存里**
—— 也就是"**发送的那一刻不生成**"：断句 → Agent 那条链路上**没有**任何等待。
所以这里既验证"该有描述时有描述"，也验证"没有描述时**照样立刻发**"
（只记 `vlm_miss`，不等待）。

    python -m orchestrator.tests.test_omni_describe_stage
"""
from __future__ import annotations

import ast
import asyncio
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_failures: list = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


# ====================================================================== #
#  捕获端口：假装是 Agent 的 /interaction/on_*
# ====================================================================== #

HITS: list = []


class _Handler(BaseHTTPRequestHandler):
    """把收到的 POST 全记下来（含**到达时刻**，用于量端到端时延）。"""

    protocol_version = "HTTP/1.1"          # keep-alive，需自带 Content-Length

    def _record(self, body: str) -> None:
        HITS.append({
            "mono": time.monotonic(),
            "method": self.command,
            "path": self.path,
            "session_header": self.headers.get("X-Session-Id"),
            "body": body,
        })

    def _reply(self) -> None:
        payload = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        self._record(self.rfile.read(n).decode("utf-8", "replace"))
        self._reply()

    def do_GET(self) -> None:  # noqa: N802
        self._record("")
        self._reply()

    def log_message(self, *a) -> None:     # 静音 —— 自己打印
        pass


def _start_capture() -> tuple:
    """起捕获端口（随机空闲端口），返回 (server, url)。"""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _answers() -> list:
    """捕获到的 on_answer（解析后的 body）。"""
    out = []
    for h in HITS:
        if h["path"].endswith("/interaction/on_answer") and h["body"]:
            try:
                out.append((h, json.loads(h["body"])))
            except ValueError:
                pass
    return out


# ====================================================================== #
#  真组件装配
# ====================================================================== #

#: 与 `interaction/downstream.py` 的 `_feed` 完全一致的字段集。
#: ⚠️ 少一个键就会让 IC 的某个档位停在默认值 —— 而症状是"决策不对"，
#:    不是报错。照抄 `_feed` 里的写入。
_FACE = {
    "face_present_confidence": "HIGH", "bbox_area_ratio": 0.12,
    "track_id": 7, "dwell_ms": 2500, "lip_speaking_confidence": "LOW",
    "identity_id": "uid-a", "identity_confidence": "HIGH",
    "display_name": "甲",
}
_ASR_FINAL = {
    "user_speaking_confidence": "HIGH", "barge_in_confidence": "LOW",
    "asr_confidence": "HIGH", "turn_complete_confidence": "HIGH",
}

_TALK = "你好，请介绍一下这个展厅"


def _mk_downstream(agent_url: str, **kw):
    """真 `InteractionDownstream`（进程内 IC + 真 AgentSink）。"""
    from orchestrator.interaction.downstream import InteractionDownstream
    d = InteractionDownstream(
        "inprocess", agent_url, session_id="t-desc",
        ic_mode="inprocess",
        # ⚠️ 关掉 set_target：那会往捕获端口打 /v1/ic/... 之外的东西，
        #    而且真实部署里它是进程级全局状态，测试不该碰。
        agent_set_target=False,
        # 不装 ExpressionSink（没有 /v1/speak），只测 AgentSink 这条链
        ic_expression_url="",
        ic_callback_ic="https://example.invalid:8100/v1/ic",
        **kw)
    return d


def _tick(d, n: int = 10, dt: float = 0.0) -> list:
    """连 tick 若干拍，收集下游产出的动作（= `_dispatch` 的结果）。

    ⚠️ `dt` 是**两拍之间的真实等待** —— 不是可有可无的节奏控制。IC 的
    `dt_ms` 由墙钟算（`interaction/downstream.py:_on_tick`：`now - _last_tick`），
    所以紧循环跑出来的 `dt_ms≈0`，`advance_*`（迎宾时钟、说话保持）**一拍都
    推不动**，policy 永远回不到 ANSWER。只看状态的测试用默认 0 即可；
    `_answer_path` 必须给真值。
    """
    from orchestrator.downstream.interface import Tick

    async def _run():
        out = []
        for _ in range(n):
            if dt:
                await asyncio.sleep(dt)
            out.extend(await d.on_event(Tick(t=0)))
        return out

    return asyncio.run(_run())


def _greet_path(d) -> list:
    """走「先看到人（未开口）」这条路 —— IC 会判 GREET。"""
    from orchestrator.downstream.interface import FaceState
    d._feed(FaceState(t=0, state=dict(_FACE)))
    return _tick(d, 10)


#: offline 之后**下一拍**的 ASR 快照 —— `说/抢` 转 NONE，`完` 与 `transcript`
#: 保留。为什么需要它（这是本测试最容易写错的一处）：
#:
#:   · `2pass-offline` 到达的那一拍，`说/抢` 靠 `_pending_close` **故意**仍报
#:     HIGH（`asr/client.py:641-645`：留一拍缓冲，让 IC 必然观测到"抢"），
#:     `完=HIGH`、`transcript=离线文本`
#:   · **下一拍** tick 调 `expire_if_idle()` 消费那个缓冲 ⇒ `说/抢`→NONE，
#:     而"本轮结论"（`完`/`信`/`transcript`）要等 `TURN_HOLD_MS` 才清
#:   · 编排侧 tick 里 `_push_asr_state_if_changed()` 会把这个变化推给 IC
#:
#: ⚠️ **少了这一条，IC 到不了 ANSWER**。`AsrFinal` 自己那条 `完=HIGH` 会被
#: IC 同拍的 `advance_*` 吃掉（实测 `turn=NONE`、只回 HOLD），必须再有一条
#: "已经安静下来、但本轮已拍板"的快照，policy 的
#: `fresh_turn → ANSWER`（`interaction/policy.py:473-506`）才成立。
_ASR_SETTLED = {
    "user_speaking_confidence": "NONE", "barge_in_confidence": "NONE",
    "asr_confidence": "HIGH", "turn_complete_confidence": "HIGH",
}


def _answer_path(d, text: str = _TALK) -> list:
    """走「见到人 + 已经开口」这条路 —— IC 判 ANSWER（跳过 GREET）。

    ⚠️ ASR 必须在 tick **之前**写进去：`policy` 判 GREET 的前提是
    "用户还没开口"（`user_has_spoken`），先 tick 就会先迎宾。

    ⚠️ tick 必须带**真实间隔**（`dt=0.05`）：IC 的 `dt_ms` 取自墙钟，
    紧循环跑出来恒为 0，`advance_*` 推不动，永远回不出 ANSWER。
    """
    from orchestrator.downstream.interface import (
        AsrFinal, AsrStateUpdate, FaceState)
    d._feed(FaceState(t=0, state=dict(_FACE)))
    d._feed(AsrFinal(t0=0, t1=500, text=text, state=dict(_ASR_FINAL)))
    d._feed(AsrStateUpdate(t=1, state=dict(_ASR_SETTLED, transcript=text)))
    return _tick(d, 12, dt=0.05)


def _flush(d) -> None:
    """排空 sink 的投递队列（POST 在 worker 线程里，断言前必须等它发完）。

    ⚠️ 顺序照 `InProcessICClient.close` 的用法：先 `flush` 再 `close` ——
    `close` 会 join 线程，队列里没发出的东西会被停机哨兵直接丢掉。
    """
    sink = getattr(getattr(d.ic, "engine", None), "agent_sink", None)
    if sink is not None:
        try:
            sink.flush(2.0)
        except Exception:  # noqa: BLE001
            pass


# ====================================================================== #
#  ① 触发点：GREET ⇒ Describe("full")
# ====================================================================== #

def test_greet_triggers_full(srv_url: str) -> None:
    from orchestrator.downstream.interface import Describe

    print("\n[触发] IC 判 GREET ⇒ 派 Describe('full')")
    d = _mk_downstream(srv_url, omni_describe=True,
                       agent_transcript_mode="dual")
    if not check_ic(d):
        return
    acts = _greet_path(d)
    descs = [a for a in acts if isinstance(a, Describe)]
    print(f"      IC 走过的动作：{sorted(d.counts)}")
    check(d.counts.get("GREET", 0) >= 1,
          f"真 IC policy 判出了 GREET（实际动作：{sorted(d.counts)}）")
    check(len(descs) == 1 and descs[0].stage == "full",
          f"GREET 那一拍恰好返回 1 个 Describe(full)（实际 {descs}）")
    # 再连 tick 若干拍，不能重复触发（GREET 会连着好几拍）
    more = _tick(d, 5)
    check(not [a for a in more if isinstance(a, Describe)],
          "GREET 持续期间**不重复**触发全量描述（否则每 50ms 一次全量推理）")
    check(type(d.ic.engine.agent_sink).__name__ == "VlmAgentSink",
          f"IC 的 sink 是包装过的子类（实际 {type(d.ic.engine.agent_sink).__name__}）")
    d.ic.close()


def test_greet_reentry_not_deduped() -> None:
    """**换人后的第二次 GREET 必须再触发一次** —— 判据是「切入 GREET」而不是
    既有的 `fresh`。

    `_last_action_key` 只在非 QUIET 动作时更新，IC 迎宾后会长久停在
    LISTEN/WAIT/HOLD，于是第二次 GREET 的 `(atype, sop, transcript)` 与
    上一次**完全相同** ⇒ 用 `fresh` 会漏掉重迎宾。这里直接喂两个相同的
    GREET 动作，中间夹若干 QUIET 拍。
    """
    from orchestrator.downstream.interface import Describe

    print("\n[触发] 换人后再次 GREET 仍要触发（不能用 fresh 判据）")
    d = _mk_downstream("http://127.0.0.1:1", omni_describe=True,
                       agent_transcript_mode="dual")
    if not check_ic(d):
        return
    greet = SimpleNamespace(type=SimpleNamespace(value="GREET"), sop="15",
                            text="您好", transcript=None)
    quiet = SimpleNamespace(type=SimpleNamespace(value="HOLD"), sop=None,
                            text=None, transcript=None)

    first = [a for a in d._dispatch(greet) if isinstance(a, Describe)]
    for _ in range(5):                       # 迎宾后的稳态
        d._dispatch(quiet)
    second = [a for a in d._dispatch(greet) if isinstance(a, Describe)]
    check(len(first) == 1, f"首次 GREET 触发全量（实际 {len(first)}）")
    check(len(second) == 1,
          f"中间夹了 5 拍 HOLD 之后的又一次 GREET 仍触发（实际 {len(second)}）"
          " —— 若为 0 说明判据退化成了 fresh")
    # 同一拍连着来（GREET 连着好几拍）只算一次
    again = [a for a in d._dispatch(greet) if isinstance(a, Describe)]
    check(not again, "紧接着的重复 GREET 不重复触发")
    d.ic.close()


# ====================================================================== #
#  ② 载荷：on_answer 带 content{ASR,VLM}，transcript 原样保留
# ====================================================================== #

DESC_FULL = "画面里有一位青年男性，穿深蓝色外套、戴黑框眼镜，站在白色展示桌左侧，正看向镜头。"


def test_payload_dual(srv_url: str) -> None:
    from orchestrator.downstream.interface import OmniDescription

    print("\n[载荷] on_answer 带 content{ASR,VLM}，transcript 仍是字符串")
    HITS.clear()
    d = _mk_downstream(srv_url, omni_describe=True,
                       agent_transcript_mode="dual")
    if not check_ic(d):
        return
    # 描述先到（`response.done` → main.py 分流 → `_feed`）
    d._feed(OmniDescription(t=0, stage="full", text=DESC_FULL))
    _answer_path(d)
    _flush(d)
    ans = _answers()
    check(len(ans) == 1, f"抓到恰好 1 个 on_answer（实际 {len(ans)}）")
    if not ans:
        d.ic.close()
        return
    _hit, body = ans[0]
    check(isinstance(body.get("transcript"), str),
          f"transcript 仍是**字符串**（dual = 过渡期双发）"
          f"，实际 {type(body.get('transcript')).__name__}")
    check(body.get("transcript") == _TALK,
          f"transcript 内容不变：{body.get('transcript')!r}")
    check(body.get("content", {}).get("ASR") == _TALK,
          "content.ASR == transcript（两个字段说的是同一件事）")
    check(body.get("content", {}).get("VLM") == DESC_FULL,
          "content.VLM == 缓存的那份描述"
          f"（实际 {body.get('content', {}).get('VLM')!r}）")
    check(body.get("session_id") == "t-desc" and body.get("callback_ic"),
          "会话信封（session_id / callback_ic）仍在 —— 没被覆写挤掉")
    check(body.get("identity_id") == "uid-a"
          and body.get("display_name") == "甲",
          "身份字段仍来自 IC state（没有自己重拼 payload）")
    d.ic.close()


def test_no_truncation(srv_url: str) -> None:
    """**不做本地截断**：多长的描述都原样进 payload。

    早先的方案是"截断到 50 字"。用户否了 —— 要的是**快**（`AsrFinal` 一到
    就有得发），短只是手段。本地砍字会把信息丢掉却不改变生成耗时
    （耗时在生成那一刻就已经花掉了）。
    """
    from orchestrator.downstream.interface import OmniDescription

    print("\n[截断] 长描述**原样**进 payload（本地不砍字）")
    HITS.clear()
    long_desc = DESC_FULL * 6           # ≈300 字
    d = _mk_downstream(srv_url, omni_describe=True,
                       agent_transcript_mode="dual")
    if not check_ic(d):
        return
    d._feed(OmniDescription(t=0, stage="delta", text=long_desc))
    _answer_path(d)
    _flush(d)
    ans = _answers()
    if not ans:
        check(False, "没抓到 on_answer")
        d.ic.close()
        return
    got = ans[0][1].get("content", {}).get("VLM", "")
    check(got == long_desc and len(got) == len(long_desc),
          f"{len(long_desc)} 字原样送达（实际 {len(got)} 字）")
    d.ic.close()


def test_no_change_keeps_previous(srv_url: str) -> None:
    """空 / 「无变化」**不覆盖**缓存 —— 缓存里始终留着一份非空描述。

    否则断句时发出去的是空串，等于白丢了一路的描述。
    """
    from orchestrator.downstream.interface import OmniDescription

    print("\n[覆盖] 空 / 「无变化」不清掉已有的描述")
    HITS.clear()
    d = _mk_downstream(srv_url, omni_describe=True,
                       agent_transcript_mode="dual")
    if not check_ic(d):
        return
    d._feed(OmniDescription(t=0, stage="full", text=DESC_FULL))
    d._feed(OmniDescription(t=1, stage="delta", text="无变化"))
    d._feed(OmniDescription(t=2, stage="delta", text="   "))
    check(d._vlm == DESC_FULL, "「无变化」/空 之后缓存仍是那份有效描述")
    # 但**真变化**的描述（含"无变化"字样）不能被误丢
    mixed = "人物无变化，但桌上多了一个红色的杯子。"
    d._feed(OmniDescription(t=3, stage="delta", text=mixed))
    check(d._vlm == mixed,
          "「人物无变化，但…」这种**有信息**的描述要收下（判据是整句相等，不是 in）")
    _answer_path(d)
    _flush(d)
    ans = _answers()
    check(ans and ans[0][1].get("content", {}).get("VLM") == mixed,
          "送到 Agent 的是最新那份")
    d.ic.close()


# ====================================================================== #
#  ③ 时序：主指标
# ====================================================================== #

def test_ready_before_asr_final(srv_url: str) -> None:
    """(a) 描述**早于** `AsrFinal` 就绪 ⇒ payload 非空，且记下年龄。"""
    from orchestrator.downstream.interface import OmniDescription

    print("\n[时序] 描述早于断句 ⇒ 断句点不生成、直接取用")
    HITS.clear()
    d = _mk_downstream(srv_url, omni_describe=True,
                       agent_transcript_mode="dual")
    if not check_ic(d):
        return
    d._feed(OmniDescription(t=0, stage="full", text=DESC_FULL))
    t_feed = time.monotonic()
    _answer_path(d)
    _flush(d)
    ans = _answers()
    check(d._vlm_miss == 0, f"没有 miss（实际 {d._vlm_miss}）")
    if not ans:
        check(False, "没抓到 on_answer")
        d.ic.close()
        return
    hit, body = ans[0]
    check(body.get("content", {}).get("VLM") == DESC_FULL,
          "断句点取到的是已就绪的描述")
    check(hit["mono"] - t_feed < 0.5,
          "断句 → Agent 收到 POST 用时 "
          f"{(hit['mono'] - t_feed) * 1000:.0f}ms（< 500ms）")
    d.ic.close()


def test_missing_description_never_waits(srv_url: str) -> None:
    """(b) 描述**晚于** `AsrFinal` ⇒ VLM 为空、`vlm_miss`+1，**但绝不等待**。

    这是"发送的那一刻不生成"的**反证**：如果这里为了等描述而变慢，
    说明我们把生成耗时加到了 ASR→Agent 这条链路上 —— 正是硬指标要禁的。
    """
    print("\n[时序] 描述未就绪 ⇒ VLM 空 + vlm_miss，**不等待**")
    HITS.clear()
    d = _mk_downstream(srv_url, omni_describe=True,
                       agent_transcript_mode="dual")
    if not check_ic(d):
        return
    t_feed = time.monotonic()
    _answer_path(d)
    _flush(d)
    ans = _answers()
    check(d._vlm_miss == 1, f"记了 1 次 miss（实际 {d._vlm_miss}）")
    if not ans:
        check(False, "没抓到 on_answer")
        d.ic.close()
        return
    hit, body = ans[0]
    check(body.get("content", {}).get("VLM") == "",
          "VLM 是空串（不是上一轮的、也不是半句话）")
    check(body.get("content", {}).get("ASR") == _TALK, "ASR 照常送达")
    check(hit["mono"] - t_feed < 0.5,
          f"仍按时送达：{(hit['mono'] - t_feed) * 1000:.0f}ms"
          " —— 没有为等描述而阻塞")
    d.ic.close()


# ====================================================================== #
#  ④ 回归：默认态必须与今天逐字节一致
# ====================================================================== #

LEGACY_KEYS = {"session_id", "callback_ic", "identity_id", "display_name",
               "transcript"}


def test_default_is_byte_identical(srv_url: str) -> None:
    """**什么都不设（默认态）⇒ payload 与今天逐字节一致。**

    这条同时是「没碰 106」的保证：编排代码 105/106 共用，默认值就是 106
    下次重启后的行为。`ORCH_AGENT_TRANSCRIPT_MODE` 默认 `legacy`、
    `ORCH_OMNI_DESCRIBE` 默认 0 —— 见 config.py 里那两个开关的说明。
    """
    print("\n[回归] 默认态（legacy）payload 逐字节等于旧版")
    HITS.clear()
    d = _mk_downstream(srv_url)                     # 全部默认
    if not check_ic(d):
        return
    check(type(d.ic.engine.agent_sink).__name__ == "AgentSink",
          f"没包装 sink（实际 {type(d.ic.engine.agent_sink).__name__}）"
          " —— legacy 下连子类都不该存在")
    _answer_path(d)
    _flush(d)
    ans = _answers()
    if not ans:
        check(False, "没抓到 on_answer")
        d.ic.close()
        return
    body = ans[0][1]
    check(set(body) == LEGACY_KEYS,
          f"键集恰好是旧的五个（实际 {sorted(body)}）—— 一个键都不多")
    check("content" not in body, "没有 content 键")
    d.ic.close()


def test_legacy_mode_with_describe_on(srv_url: str) -> None:
    """`omni_describe=1` 但 transcript 模式仍是 legacy ⇒ **也不改 payload**。

    两个开关是**与**关系：描述算出来了也可以不发（Agent 还没改好时
    正是这个状态 —— 先看日志里的 VLM 是否可用，再决定切不切载荷）。
    """
    print("\n[回归] 描述开着 + transcript=legacy ⇒ 仍不改 payload")
    HITS.clear()
    d = _mk_downstream(srv_url, omni_describe=True,
                       agent_transcript_mode="legacy")
    if not check_ic(d):
        return
    check(type(d.ic.engine.agent_sink).__name__ == "AgentSink",
          "legacy 下 sink 不被包装")
    _answer_path(d)
    _flush(d)
    ans = _answers()
    check(ans and set(ans[0][1]) == LEGACY_KEYS,
          f"键集仍是旧的五个（实际 {sorted(ans[0][1]) if ans else '无'}）")
    d.ic.close()


def test_illegal_mode_falls_back_to_legacy(srv_url: str) -> None:
    """非法模式名**归一为 legacy**（安全侧）+ 告警，别静默变成"开了"。"""
    from orchestrator.interaction.agent_sink import normalize_mode
    print("\n[回归] 非法 transcript 模式 → 归一为 legacy（安全侧）")
    # ⚠️ 这里只能放**真的不合法**的值。`"Dict"` 看着像错的，其实用例规整后
    #    是合法值 `dict`（大小写不敏感 + 去空白），拿它当非法用例是在
    #    把正确行为判成失败。
    check(normalize_mode("duall") == "legacy", "拼错 → legacy")
    check(normalize_mode("") == "legacy", "空 → legacy")
    check(normalize_mode(None) == "legacy", "None → legacy")
    check(normalize_mode("dual") == "dual", "合法值原样")
    check(normalize_mode(" DICT ") == "dict", "合法值去空白+小写")
    check(normalize_mode("Dict") == "dict",
          "`Dict` 是合法值（大小写不敏感）—— 别误判成非法")


# ====================================================================== #
#  ⑤ 契约漂移：覆写点必须真的在投递路径上
# ====================================================================== #

def test_contract_drift(srv_url: str) -> None:
    """真 `AgentSink` / `VlmAgentSink` 各打一次，看**键集与投递次数**。

    ⚠️ 这是本方案唯一的结构性风险：`_send` 是 IC 的**私有**方法，被改名或
    改签名后覆写会**静默失效**（基类的 `on_answer` 照常工作，只是不带
    `content`）—— 正是本仓库反复踩的那类"改了、没生效、不报错"。
    这个测试锁住两件事：
      ① `_send` 仍在投递路径上（不在了 ⇒ 下面的 content 断言会失败）
      ② `legacy` 下投递行为与旧版一致（恰好 1 次 POST、键集不变）
    """
    from orchestrator.interaction.agent_sink import make_vlm_agent_sink

    print("\n[契约] 覆写点 `_send` 是否在投递路径上")
    HITS.clear()
    kwargs = dict(timeout=5.0, session_id="drift", callback_ic=None,
                  session_envelope=False)

    def _send_one(sink):
        sink.on_answer(identity_id="u", display_name="n", transcript="转写")
        sink.flush(2.0)
        sink.close(2.0)

    from interaction.runtime import AgentSink
    _send_one(AgentSink(srv_url, **kwargs))
    legacy = _answers()
    check(len(legacy) == 1, f"旧形态恰好 1 次 POST（实际 {len(legacy)}）")
    check(legacy and set(legacy[0][1]) == {"identity_id", "display_name",
                                           "transcript"},
          f"envelope 关闭时键集就是旧的三件套（实际 "
          f"{sorted(legacy[0][1]) if legacy else '无'}）")

    HITS.clear()
    sink = make_vlm_agent_sink(target=srv_url, vlm_getter=lambda: "描述文本",
                               mode="dual", **kwargs)
    _send_one(sink)
    wrapped = _answers()
    check(len(wrapped) == 1,
          f"包装后仍恰好 1 次 POST（实际 {len(wrapped)}）—— 没多发也没漏发")
    if wrapped:
        body = wrapped[0][1]
        check(body.get("content") == {"ASR": "转写", "VLM": "描述文本"},
              f"content 注进去了（实际 {body.get('content')!r}）—— "
              "为 0 说明 `_send` 已不在投递路径上（IC 改版了？）")
        check(body.get("transcript") == "转写",
              "transcript 未被改动（dual 是**追加**一个键）")


# ====================================================================== #
#  ⑥ session 侧：两阶段指令 + 滚动刷新（不重叠、可自愈）
# ====================================================================== #

class FakeOmni:
    """假 OmniLLM：只记下收到的指令，不发网络。"""

    turn_trigger = "asr"

    def __init__(self) -> None:
        self.reqs: list = []
        self.alive = True
        #: 换人设调用记录（真实现是"关掉重连"，这里只记 system_prompt）
        self.switches: list = []
        #: 换人设会不会成功（测失败降级用）
        self.switch_ok = True
        #: 每次 switch 的模拟耗时（秒）—— 用来验"重连期间暂停滚动"
        self.switch_delay = 0.0

    async def trigger_reply(self, text: str, **kw) -> bool:
        if not self.alive:
            return False
        self.reqs.append((time.monotonic(), text))
        return True

    async def switch_system_prompt(self, system_prompt: str) -> bool:
        self.switches.append(system_prompt)
        if self.switch_delay:
            await asyncio.sleep(self.switch_delay)
        return self.switch_ok


def test_two_stage_prompts_and_loop() -> None:
    """`session` 侧：全量/增量指令分开注入；滚动刷新**不重叠、能自愈**。"""
    from orchestrator.omni.describe import DELTA_INSTRUCTION, FULL_INSTRUCTION
    from orchestrator.session import OrchestratorSession

    print("\n[两阶段] 指令注入 + 滚动刷新")
    sess = OrchestratorSession("t-desc-loop",
                               {"omni_describe": True,
                                "omni_delta_interval_s": 0.2})
    omni = FakeOmni()
    sess.omni = omni

    async def _drive() -> str:
        # ① IC 判 GREET ⇒ executor 调它（这里直接调，等价于 Describe("full") 走到了）
        await sess.request_omni_description("full")
        await asyncio.sleep(0.05)
        # ② 模拟 `response.done` 到达（main.on_omni_event 那一步读走 stage）
        stage = sess.take_omni_stage()
        # ⚠️ "读走即清零" 必须**在这里**断言：晚一点滚动循环就发了下一轮
        #    增量、把 `_omni_stage` 又置上了（那时读到 "delta" 是正常的，
        #    不是"没清零"）。
        d_stage2 = sess.take_omni_stage()
        if d_stage2 != "":
            check(False, f"读走即清零（第二次读是空，实际 {d_stage2!r}）")
        # ③ 等滚动循环发增量（间隔 0.2s）
        await asyncio.sleep(0.7)
        n_after_first_delta = len(omni.reqs)
        # ④ 增量那一轮**不回灌 done** ⇒ 在途不消 ⇒ 不能再发（不自激）
        await asyncio.sleep(0.7)
        n_stalled = len(omni.reqs)
        # ⑤ 手工把在途时刻推老 ⇒ 超时分支应放掉并恢复滚动（自愈）
        sess._desc_inflight_at -= 100.0
        await asyncio.sleep(0.7)
        return stage, n_after_first_delta, n_stalled

    stage, n1, n2 = asyncio.run(_drive())
    sess._stop_desc_loop()

    check(stage == "full", f"take_omni_stage 读走的是 full（实际 {stage!r}）")
    reqs = omni.reqs
    check(len(reqs) >= 2, f"发出了 ≥2 次描述请求（实际 {len(reqs)}）")
    check(reqs and reqs[0][1] == FULL_INSTRUCTION,
          "第 1 次注入的是**全量**指令")
    check(any(t == DELTA_INSTRUCTION for _, t in reqs[1:]),
          "之后注入的是**增量**指令（两阶段靠注入指令区分，不重连不重 init）")
    check(n1 == 2 and n2 == 2,
          f"增量在途未完成时**不再发**（{n1} → {n2}）—— 不自激烧 GPU")
    check(len(reqs) > n2,
          f"在途超时后恢复滚动（{n2} → {len(reqs)}）—— 丢了 done 也能自愈")

    # 描述模式下**关掉**对话式触发（拿 ASR 去问一个"描述助手"只会白烧）
    sess.omni = omni
    n = len(omni.reqs)
    sess._trigger_omni("你好")
    asyncio.run(asyncio.sleep(0.05))
    check(len(omni.reqs) == n, "描述模式下 ASR 不再触发对话式回复")
    sess.closed = True


def test_answer_path_starts_desc_without_greet() -> None:
    """IC 直判 ANSWER（**跳过 GREET**）时，首个带文本的 ASR 结果也要点火。

    这是「人一开口就进画」那条路：`GREET` 那一拍根本不出现，而全量的唯一
    触发点原本挂在「切入 GREET」上 ⇒ `_desc_started` 永远是 False ⇒ 整条
    描述链路**含滚动循环**此后再也不启动，`on_answer` 的 VLM 恒为空。
    而现象只是"VLM 是空的"，看不出因果。

    ⚠️ 必须挂**部分结果**，不是最终结果。实测
    `orch-dumps/105/replay-121d10.log`：首个带文本的部分结果 t=12.1s，
    `2pass-offline` 最终结果 t=28.9s —— 中间 17s 提前量，而全量描述只要
    ~1.25s。挂最终结果 = 在 `on_answer` 那一刻才开始生成，必然空。

    ⚠️ `force_listen` 这条别忘了：音视频推送一直是 `force_listen=True`
    （只 prefill 不生成），所以这里缺的从来不是 prefill，而是**该 decode
    一次了的信号** —— `trigger_reply` 就是那次 `force_listen=False`。
    """
    from orchestrator.omni.describe import FULL_INSTRUCTION
    from orchestrator.session import OrchestratorSession

    print("\n[触发] IC 直判 ANSWER（无 GREET）⇒ 首个 ASR 文本点火")
    sess = OrchestratorSession("t-desc-answer",
                               {"omni_describe": True,
                                "omni_delta_interval_s": 5.0})
    omni = FakeOmni()
    sess.omni = omni

    async def _drive() -> int:
        check(sess._desc_started is False, "起点：描述链路未启动")
        sess._start_desc_on_asr("")        # 空文本不点火
        sess._start_desc_on_asr("   ")     # 纯空白也不点火
        await asyncio.sleep(0.05)
        n_blank = len(omni.reqs)
        sess._start_desc_on_asr("好")       # 首个**有文本**的部分结果 ⇒ 点火
        await asyncio.sleep(0.05)
        # 同一句的后续部分结果：绝不能重复点火（否则每 600ms 一次全量推理）
        sess._start_desc_on_asr("好了现在呃呃")
        await asyncio.sleep(0.05)
        return n_blank

    n_blank = asyncio.run(_drive())
    sess._stop_desc_loop()
    check(n_blank == 0, f"空/纯空白的 ASR 文本不点火（实际发了 {n_blank} 次）")
    check(sess._desc_started, "首个有文本的 ASR 结果把链路点起来了")
    reqs = omni.reqs
    check(len(reqs) == 1, f"点火恰好一次（实际 {len(reqs)}）")
    check(reqs and reqs[0][1] == FULL_INSTRUCTION,
          "首份注入的是**全量**指令 —— 没跳过全量")
    sess.closed = True

    # 描述模式关着时，这条路径必须是**死代码**（106 重启后的默认行为）
    sess2 = OrchestratorSession("t-desc-answer-off", {"omni_describe": False})
    omni2 = FakeOmni()
    sess2.omni = omni2
    sess2._start_desc_on_asr("好")
    asyncio.run(asyncio.sleep(0.05))
    check(len(omni2.reqs) == 0 and not sess2._desc_started,
          "描述模式关着 ⇒ 这条路径什么都不做（默认态逐字节不变）")
    sess2.closed = True


def test_describe_prompt_matches_demo() -> None:
    """提示词与 `streaming_chat_demo.py` 的分工**不许漂移**。

    约定（见 `omni/describe.py` 的模块文档）：

    * `DEMO_DESCRIBE_PROMPT` == demo 的 `DESCRIBE_SYSTEM_PROMPT`，**逐字**；
    * `FULL_INSTRUCTION` 以它结尾 ⇒ **全量阶段用的就是 demo 那份原文**；
    * `DESCRIBE_SYSTEM_PROMPT` 的**前两行**逐字取自 demo（身份句 + 要求句），
      但**不带**八类清单与"输出：分条" —— 那两句是格式要求，
      system prompt 每轮重发，待在里头会把增量轮也顶成八类清单
      （2026-10-08 实测：增量轮 311~679 字，注入指令压不回来）。
    """
    print("\n[提示词] 与 demo 的分工不许漂移")
    from orchestrator.omni.describe import (
        DEMO_DESCRIBE_PROMPT, DESCRIBE_SYSTEM_PROMPT, FULL_FORMAT_LINE,
        FULL_INSTRUCTION,
    )

    demo_file = Path(__file__).resolve().parents[2] / "streaming_chat_demo.py"
    if not demo_file.exists():
        check(False, f"找不到 {demo_file} —— 这条断言失去意义，别静默跳过")
        return
    tree = ast.parse(demo_file.read_text(encoding="utf-8"))
    demo = None
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "DESCRIBE_SYSTEM_PROMPT"
                        for t in node.targets)):
            demo = ast.literal_eval(node.value)
    check(demo is not None, "demo 里仍有 DESCRIBE_SYSTEM_PROMPT（改名了？）")
    if demo is None:
        return

    check(DEMO_DESCRIBE_PROMPT == demo,
          f"DEMO_DESCRIBE_PROMPT 与 demo 逐字一致（demo {len(demo)} 字 / "
          f"本模块 {len(DEMO_DESCRIBE_PROMPT)} 字）—— 不一致就是只改了一边")
    check(FULL_INSTRUCTION.endswith(DEMO_DESCRIBE_PROMPT),
          "FULL_INSTRUCTION 以 demo 原文结尾 ⇒ **全量阶段用的就是那份**")
    check(FULL_INSTRUCTION.startswith("直接输出"),
          "全量指令开头是那句针对'模型回好的'的护栏")
    sys_lines = DESCRIBE_SYSTEM_PROMPT.split("\n")
    check(sys_lines[:2] == demo.split("\n")[:2],
          "system prompt 前两行（身份句 + 要求句）逐字取自 demo")
    check("【P0" in DESCRIBE_SYSTEM_PROMPT,
          "system prompt **保留**八类清单 —— 摘掉它全量轮就不分条了（实测）")
    check(FULL_FORMAT_LINE not in DESCRIBE_SYSTEM_PROMPT,
          "system prompt **摘掉**'输出：按上述八类分条' —— 留着增量轮会被顶回长格式")
    check("不要与用户对话" in DESCRIBE_SYSTEM_PROMPT,
          "system prompt 里有'不要对话'护栏（第一版走样的教训）")

    # `DELTA_SYSTEM_PROMPT`（换人设重连后的新人设）—— 它的价值全在
    # **没有什么**上面，所以这几条是断言"缺席"的。
    from orchestrator.omni.describe import (
        DELTA_INSTRUCTION, DELTA_SYSTEM_PROMPT)
    check("【P0" not in DELTA_SYSTEM_PROMPT
          and "【P1" not in DELTA_SYSTEM_PROMPT
          and "【P2" not in DELTA_SYSTEM_PROMPT,
          "DELTA_SYSTEM_PROMPT **不含八类清单** —— 它正是增量轮压不短的"
          "源头（2026-10-08 实测 3 版）；留着它换人设就白换了")
    check("不要与用户对话" in DELTA_SYSTEM_PROMPT,
          "DELTA_SYSTEM_PROMPT 保留'不要对话'护栏（口吻的锚，不是格式的锚）")
    check("视频监控/行为分析助手" in DELTA_SYSTEM_PROMPT,
          "DELTA_SYSTEM_PROMPT 保留身份句 —— 抽象人设会让模型回"
          "'好的，现在开始录音。'（第一版走样的教训）")
    check(len(DELTA_SYSTEM_PROMPT) < len(DESCRIBE_SYSTEM_PROMPT),
          f"DELTA_SYSTEM_PROMPT 比全量那份短"
          f"（{len(DELTA_SYSTEM_PROMPT)} < {len(DESCRIBE_SYSTEM_PROMPT)} 字）")
    check(DELTA_SYSTEM_PROMPT != DESCRIBE_SYSTEM_PROMPT
          and DELTA_INSTRUCTION != DELTA_SYSTEM_PROMPT,
          "三份提示词互不相同（少抄/抄错都会在这里露出来）")


def test_delta_persona_switch() -> None:
    """`ORCH_OMNI_DELTA_PERSONA=1`：全量做完**关掉重连**换短人设。

    为什么非换不可（每轮 prompt 试过三版全压不住）见
    `omni/describe.py` 的模块文档「为什么增量轮压不短」。这里只验编排侧
    那几步**必须**做到位，否则现象会很难归因：

    * 换人设**只在全量之后**（换早了全量就没有八类清单了）；
    * 整个会话**只换一次**（换人设失败也算换过 —— 否则每次全量都会把
      omni 重连打断一遍）；
    * 重连窗口里**一次触发都不许发**（发了也送不出去，白刷
      `omni_desc_failed`）；且窗口结束后在途标记要**主动**放掉 ——
      否则滚动循环空等满超时（10s），这 10s 缓存里描述不更新；
    * 换连接后的**第一条**增量带上「上一轮描述」（新连接 KV 是空的，
      模型不知道"相对什么"在变），**第二条起不带**。
    """
    from orchestrator.omni.describe import (
        DELTA_INSTRUCTION, DELTA_SYSTEM_PROMPT, FULL_INSTRUCTION)
    from orchestrator.session import OrchestratorSession

    print("\n[人设] 全量做完 ⇒ 关掉重连换短人设（首条增量带上一轮描述）")
    sess = OrchestratorSession("t-persona",
                               {"omni_describe": True,
                                "omni_delta_interval_s": 0.2,
                                "omni_delta_persona": True})
    omni = FakeOmni()
    omni.switch_delay = 0.6          # 比间隔长 ⇒ 能验"重连期间暂停滚动"
    sess.omni = omni

    async def _drive():
        # ① IC 判 GREET ⇒ 全量（等价于 Describe("full") 走到 executor）
        await sess.request_omni_description("full")
        await asyncio.sleep(0.05)
        # ② 全量的 `response.done` 到达 —— main.py 那一步（传 text！）
        stage_full = sess.take_omni_stage(DESC_FULL)
        # ③ 重连窗口（0.6s）内不许有触发；此刻只有那次全量
        await asyncio.sleep(0.9)
        n_during, n_switch = len(omni.reqs), len(omni.switches)
        # ④ 等第一条增量
        await asyncio.sleep(0.9)
        first_delta = omni.reqs[1][1] if len(omni.reqs) > 1 else ""
        # ⑤ 回灌它的 done（顺带用「无变化」验它不覆盖参照物）⇒ 第二条增量
        sess.take_omni_stage("无变化")
        await asyncio.sleep(0.8)
        # ⑥ 再来一次"全量 done"（重迎宾）—— 不许再换一次人设
        sess._omni_stage = "full"
        sess.take_omni_stage(DESC_FULL)
        await asyncio.sleep(0.3)
        return stage_full, n_during, n_switch, first_delta

    stage_full, n_during, n_switch, first_delta = asyncio.run(_drive())
    sess._stop_desc_loop()

    check(stage_full == "full", f"take 读走的是 full（实际 {stage_full!r}）")
    check(n_switch == 1, f"恰好换 1 次人设（实际 {n_switch}）")
    check(omni.switches == [DELTA_SYSTEM_PROMPT],
          "换的是 DELTA_SYSTEM_PROMPT（不是全量那份）")
    check(n_during == 1,
          f"重连窗口里**没有**发出任何描述触发（{n_during} 次 —— 应当只有 "
          f"最初那次全量）")
    check(omni.reqs and omni.reqs[0][1] == FULL_INSTRUCTION,
          "第 1 次注入的是**全量**指令")
    check(first_delta.endswith(DELTA_INSTRUCTION)
          and first_delta.startswith("上一轮描述"),
          "换连接后的**第一条**增量带上了「上一轮描述」当参照物")
    check(DESC_FULL in first_delta,
          "……而且带的就是那份全量描述原文")
    reqs = [t for _, t in omni.reqs]
    check(len(reqs) >= 3 and reqs[2] == DELTA_INSTRUCTION,
          f"**第二条**增量不再带参照物（第 3 次注入是否等于裸增量指令："
          f"{len(reqs) >= 3 and reqs[2] == DELTA_INSTRUCTION}）")
    check(sess._desc_last_text == DESC_FULL,
          "「无变化」没把参照物覆盖掉（它意味着上一份仍然成立）")
    sess.closed = True


def test_delta_persona_off_and_fail() -> None:
    """换人设**默认关**；开了但换失败要降级、不能把会话带塌。

    默认关的理由与 `ORCH_OMNI_DESCRIBE` 同款：编排代码 105/106 共用，
    默认值就是 106 下次重启后的行为。
    """
    from orchestrator.config import Settings
    from orchestrator.omni.describe import DELTA_INSTRUCTION
    from orchestrator.session import OrchestratorSession

    print("\n[人设] 默认关 / 换失败降级")
    import os
    for k in ("ORCH_OMNI_DELTA_PERSONA", "ORCH_OMNI_DESCRIBE"):
        os.environ.pop(k, None)
    check(Settings().omni_delta_persona is False,
          "ORCH_OMNI_DELTA_PERSONA 不设时默认 **False**（安全侧）")

    # ---- ① 默认关：不换人设，两阶段仍靠同一条连接上的注入指令区分 ----
    sess = OrchestratorSession("t-persona-off",
                               {"omni_describe": True,
                                "omni_delta_interval_s": 0.2})
    omni = FakeOmni()
    sess.omni = omni

    async def _off():
        await sess.request_omni_description("full")
        await asyncio.sleep(0.05)
        sess.take_omni_stage(DESC_FULL)
        await asyncio.sleep(0.9)
        return omni.switches[:], [t for _, t in omni.reqs]

    switches, reqs = asyncio.run(_off())
    sess._stop_desc_loop()
    sess.closed = True
    check(switches == [], "默认关时**一次都不换**人设")
    check(len(reqs) >= 2 and reqs[1] == DELTA_INSTRUCTION,
          "……增量仍是那条连接的裸增量指令（不带参照物）")

    # ---- ② 开了但换失败：降级成"原人设常驻"，不抛、不重试 ----
    sess2 = OrchestratorSession("t-persona-fail",
                                {"omni_describe": True,
                                 "omni_delta_interval_s": 0.2,
                                 "omni_delta_persona": True})
    omni2 = FakeOmni()
    omni2.switch_ok = False
    sess2.omni = omni2

    async def _fail():
        await sess2.request_omni_description("full")
        await asyncio.sleep(0.05)
        sess2.take_omni_stage(DESC_FULL)
        await asyncio.sleep(0.6)
        sess2._omni_stage = "full"          # 重迎宾 ⇒ 不许重试换人设
        sess2.take_omni_stage(DESC_FULL)
        await asyncio.sleep(0.3)
        return [t for _, t in omni2.reqs]

    reqs2 = asyncio.run(_fail())
    sess2._stop_desc_loop()
    sess2.closed = True
    check(len(omni2.switches) == 1,
          f"换失败也只尝试**一次**（实际 {len(omni2.switches)}）")
    check(sess2._desc_needs_context is False,
          "换失败 ⇒ 不挂参照物（新连接压根没建起来）")
    check(reqs2 and all(t != "" for t in reqs2),
          "换失败后链路照跑（注入的是非空指令）")
    check(len(reqs2) >= 2 and reqs2[1] == DELTA_INSTRUCTION,
          "……增量退回裸增量指令（退化成'原人设常驻'，能跑只是长）")


def check_ic(d) -> bool:

    """连进程内 IC；连不上（没装 interaction 包）就明确报失败。"""
    if d.ic.connect():
        return True
    check(False, f"进程内 IC 不可用：{d.ic.error} —— 本测试需要 interactioncore")
    return False


def main() -> int:
    from orchestrator.interaction.agent_sink import MODES

    print("=" * 68)
    print("VLM 描述（两阶段）随 ASR 发给 Agent —— 离线端到端验证")
    print("-" * 68)
    srv, url = _start_capture()
    try:
        check(MODES == ("legacy", "dual", "dict"),
              f"三种载荷模式（当前 {MODES}）—— 契约变更测试会跟着失败")
        test_greet_triggers_full(url)
        test_greet_reentry_not_deduped()
        test_payload_dual(url)
        test_no_truncation(url)
        test_no_change_keeps_previous(url)
        test_ready_before_asr_final(url)
        test_missing_description_never_waits(url)
        test_default_is_byte_identical(url)
        test_legacy_mode_with_describe_on(url)
        test_illegal_mode_falls_back_to_legacy(url)
        test_contract_drift(url)
        test_describe_prompt_matches_demo()
        test_two_stage_prompts_and_loop()
        test_answer_path_starts_desc_without_greet()
        test_delta_persona_switch()
        test_delta_persona_off_and_fail()
    finally:
        srv.shutdown()
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
