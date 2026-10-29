#!/usr/bin/env python3
"""VLM 描述随 ASR 一起发给 Agent —— **纯离线端到端验证**（不连任何服务）。

## 验的是什么

发给 Agent 的 `on_answer` 类事件里原本**只有 ASR 转写**。本次改动让它多带一份
OmniLLM 的**画面/语音描述**（`content: {ASR, VLM}`）—— 靠的是编排侧给
`AgentSink` 包一层子类覆写 `_send`（见 `orchestrator/interaction/agent_sink.py`）。

⚠️ 后来 IC 把四类事件统一改投 `on_action`（payload 里 `act` 指明动作类型；
四个旧端点保留兼容）—— 那只动了投递层 `_post`，覆写点 `_send` 未变。
本脚本的捕获对两种形态**都兼容**（`_answers()` 同时认 `on_action` + `act`
和旧的 `on_{action_name}` 专用端点），键集断言照现行 IC 行为（§⑤ 契约漂移
测试锁着这一点）。

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
    """捕获到的 on_answer 类事件（解析后的 body）—— **两种投递形态都收**：

    - 统一入口 `/interaction/on_action`（IC 现投这里）⇒ 必须看 `act`，
      只收 `act == "ANSWER"` 的那些；
    - 旧专用端点 `/interaction/on_answer`（原逻辑，旧 IC / 兼容保留）⇒ 路
      径本身就是 ANSWER，payload 带不带 `act` 都算；但若带了别的 `act`
      （同一批捕获里混投时）必须按统一入口同样地过滤掉，否则会错收。
    """
    out = []
    for h in HITS:
        if not h["body"]:
            continue
        path = h["path"]
        unified = path.endswith("/interaction/on_action")
        legacy = path.endswith("/interaction/on_answer")
        if not (unified or legacy):
            continue
        try:
            body = json.loads(h["body"])
        except ValueError:
            continue
        # 统一入口必带 act；旧端点通常不带 —— 带了就必须是 ANSWER
        if body.get("act", "ANSWER") != "ANSWER":
            continue
        out.append((h, body))
    return out


def _is_background_round(body: dict) -> bool:
    """这发 on_answer 类事件是不是「**视觉背景轮**」。

    判据照 Agent 侧的分流（`openai-agent-demo` 的
    `api/interaction_routes.py`）：`content` 在场 且 `ASR` 空 且
    `transcript` 空 ⇒ 走 `handle_visual_background`（**只存不答**），
    不是一次真实话轮。编排侧发它的地方见 `downstream._emit_visual_background`。
    """
    content = body.get("content") or {}
    return (bool(content)
            and not (content.get("ASR") or "").strip()
            and not (body.get("transcript") or "").strip())


def _bg_rounds() -> list:
    """捕获到的视觉背景轮。"""
    return [(h, b) for h, b in _answers() if _is_background_round(b)]


def _talk_rounds() -> list:
    """捕获到的**真实话轮**（排除背景轮）。

    ⚠️ 断言"这一发是第几发"的老测试必须用它：补了背景轮之后，
    `_answers()[0]` 往往**是**背景轮（它在 `_feed(OmniDescription)` 那一刻
    就入队了，早于 `_answer_path`），拿 `[0]` 会静默断言到错的那一发上。
    """
    return [(h, b) for h, b in _answers() if not _is_background_round(b)]


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
#:   · `2pass-offline` 到达的那一拍带**服务端原值**：`说/抢`=HIGH，
#:     `完=HIGH`、`transcript=离线文本`
#:   · **下一拍** tick 调 `expire_if_idle()` ⇒ `说/抢`→NONE，而
#:     `完/信/transcript` 保留（再下一拍才全部清空，见 `docs/asr-state.md`）
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
#  ② 载荷：on_action(act=ANSWER) 带 content{ASR,VLM}，transcript 原样保留
# ====================================================================== #

DESC_FULL = "画面里有一位青年男性，穿深蓝色外套、戴黑框眼镜，站在白色展示桌左侧，正看向镜头。"

#: 8 键契约下的**标准全量描述**（顺序与 `STATE_KEYS` 一致 —— `render_state`
#: 必须逐字还原它，这是"合并成快照"最容易错的地方）。
_STATE_FULL = """ATTENTION: 面向屏幕，注视镜头
ACTION: 无
EMOTION: 平静
FEATURE: 青年女性，灰色短袖上衣
BEHAVIOR: 站立，保持不动
RELATION: 单人
ENV: 室内展厅，光线明亮
POSITION: 站在展台右侧"""


def test_payload_dual(srv_url: str) -> None:
    from orchestrator.downstream.interface import OmniDescription

    print("\n[载荷] on_action(act=ANSWER) 带 content{ASR,VLM}，transcript 仍是字符串")
    HITS.clear()
    d = _mk_downstream(srv_url, omni_describe=True,
                       agent_transcript_mode="dual")
    if not check_ic(d):
        return
    # 描述先到（`response.done` → main.py 分流 → `_feed`）
    d._feed(OmniDescription(t=0, stage="full", text=DESC_FULL))
    _answer_path(d)
    _flush(d)
    ans = _talk_rounds()
    check(len(ans) == 1, f"恰好 1 个**真实话轮**（实际 {len(ans)}；"
                         f"另有 {len(_bg_rounds())} 个背景轮 —— 那条走 "
                         f"`test_visual_background_round` 单独验）")
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
    check(body.get("content", {}).get("VLM") == "",
          "正常话轮**只带 ASR**，content.VLM 为空（VLM 与 ASR 不同步，"
          "视觉状态走单独的视觉轮；实际 "
          f"{body.get('content', {}).get('VLM')!r}）")
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
    d._feed(OmniDescription(t=0, stage="turn", text=long_desc))
    _flush(d)
    bgs = _bg_rounds()
    if not bgs:
        check(False, "没抓到 on_action(act=ANSWER)")
        d.ic.close()
        return
    got = bgs[0][1].get("content", {}).get("VLM", "")
    check(got == long_desc and len(got) == len(long_desc),
          f"{len(long_desc)} 字原样推送（实际 {len(got)} 字）")
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
    d._feed(OmniDescription(t=1, stage="turn", text="无变化"))
    d._feed(OmniDescription(t=2, stage="turn", text="   "))
    check(d._vlm == DESC_FULL, "「无变化」/空 之后缓存仍是那份有效描述")
    # 但**真变化**的描述（含"无变化"字样）不能被误丢
    mixed = "人物无变化，但桌上多了一个红色的杯子。"
    d._feed(OmniDescription(t=3, stage="turn", text=mixed))
    check(d._vlm == mixed,
          "「人物无变化，但…」这种**有信息**的描述要收下（判据是整句相等，不是 in）")
    _flush(d)
    bgs = _bg_rounds()
    check(bgs and bgs[-1][1].get("content", {}).get("VLM") == mixed,
          "推给 Agent 的是最新那份（视觉轮，ASR 空）")
    d.ic.close()


# ====================================================================== #
#  ②′ 视觉背景轮：全量描述额外投一发 ASR 为空的 on_answer 类事件（act=ANSWER）
# ====================================================================== #

def test_visual_background_round(srv_url: str) -> None:
    """全量描述 ⇒ 额外投一发「ASR 为空的背景轮」，Agent 走 `handle_visual_background`。

    没有这一发，Agent 收到的 `scene.visual_change` 就在**引用一个它从没收到过
    的背景**（全量只活在编排侧缓存里、随即被增量覆盖），`scene.visual_background`
    恒空。IC 结构上产生不了这一轮（它的 `on_answer` 只在 ANSWER 时发、
    `transcript` 必非空），所以由编排侧合成。
    """
    from orchestrator.downstream.interface import OmniDescription

    print("\n[背景轮] 全量描述 ⇒ 多一发 on_answer（ASR 空，只存不答）")
    HITS.clear()
    d = _mk_downstream(srv_url, omni_describe=True,
                       agent_transcript_mode="dual")
    if not check_ic(d):
        return
    d._feed(OmniDescription(t=0, stage="full", text=DESC_FULL))
    _flush(d)
    bgs = _bg_rounds()
    check(len(bgs) == 1, f"恰好 1 个视觉轮（实际 {len(bgs)}）")
    if bgs:
        body = bgs[0][1]
        check(body.get("content", {}).get("VLM") == DESC_FULL,
              f"content.VLM 就是那份全量描述（{len(DESC_FULL)} 字，不做截断）")
        check(body.get("content", {}).get("ASR") == ""
              and body.get("transcript") == "",
              "ASR 与 transcript **都空** —— Agent 据此判「会话起点背景」")
        check(isinstance(body.get("content"), dict),
              "content 是 dict（Agent 的分流判据；是字符串就会被当普通话轮）")
        check(body.get("session_id") == "t-desc" and body.get("callback_ic"),
              "会话信封在场 —— 否则 Agent 路由不到本会话的桥接")
        check(body.get("identity_id") is None
              and body.get("display_name") is None,
              "身份字段为空（背景轮不是「某个人说的」）")
    d.ic.close()


def test_turn_pushes_visual_round(srv_url: str) -> None:
    """每轮描述**也**单独推一发视觉轮（ASR 空），正常话轮只带 ASR。

    VLM 与 ASR 不同步（2026-10-28）：描述一生成完就推，不等下一次
    `on_answer` 捎带 —— 断句触发的描述本来就晚于本句，缓存只会让 Agent
    永远看到"上一句那时候"的画面。
    """
    from orchestrator.downstream.interface import OmniDescription

    print("\n[背景轮] 每轮描述也单独推一发（ASR 空），正常话轮 VLM 为空")
    HITS.clear()
    d = _mk_downstream(srv_url, omni_describe=True,
                       agent_transcript_mode="dual")
    if not check_ic(d):
        return
    turn = "EMOTION: 不耐烦\nBEHAVIOR: 站立 6 秒"
    d._feed(OmniDescription(t=0, stage="turn", text=turn))
    _flush(d)
    bgs = _bg_rounds()
    check(len(bgs) == 1, f"每轮描述 ⇒ 1 个视觉轮（实际 {len(bgs)}）")
    if bgs:
        check("EMOTION: 不耐烦" in (bgs[0][1].get("content", {}).get("VLM") or ""),
              "推出去的就是这份描述")
        check(bgs[0][1].get("content", {}).get("ASR") == ""
              and bgs[0][1].get("transcript") == "",
              "ASR 与 transcript 都空（Agent 据此走「只存不答」）")
    _answer_path(d)
    _flush(d)
    ans = _talk_rounds()
    check(len(ans) == 1, f"真实话轮 1 个（实际 {len(ans)}）")
    if ans:
        check(ans[0][1].get("content", {}).get("VLM") == "",
              "正常话轮不带 VLM（视觉状态已经单独推过了）")
        check(ans[0][1].get("content", {}).get("ASR") == _TALK,
              "ASR 照常送达")
    d.ic.close()


def test_visual_push_every_description(srv_url: str) -> None:
    """**每份**描述都推（没有"只发第一份"的闩）；换会话照推。

    旧方案的闩是为「每轮只报变化 + 换短人设重连」设的（第二份 full 只有
    几十字，覆盖 476 字的起点背景是净损失）。现在每份都是 8 键全量快照、
    语义就是"此刻的完整状态" ⇒ 越新越准，留闩反而让 Agent 停在起点那份。
    """
    from orchestrator.downstream.interface import OmniDescription

    print("\n[背景轮] 每份描述都推（无闩）；换会话照推")
    HITS.clear()
    d = _mk_downstream(srv_url, omni_describe=True,
                       agent_transcript_mode="dual")
    if not check_ic(d):
        return
    d._feed(OmniDescription(t=0, stage="full", text=DESC_FULL))
    _flush(d)
    check(len(_bg_rounds()) == 1, "第一份 ⇒ 1 个视觉轮")

    d._feed(OmniDescription(t=1, stage="full", text="画面里换了一位访客。"))
    _flush(d)
    bgs = _bg_rounds()
    check(len(bgs) == 2, f"第二份 **也**推（实际共 {len(bgs)} 个）")
    check(bgs and (bgs[-1][1].get("content", {}).get("VLM") or "").find("换了一位访客") >= 0,
          "Agent 手上换成了**最新**那份（背景 = 当前视觉状态）")

    # 新会话：清场标记复位，之后照推
    asyncio.run(d.on_session_start(None))
    check(d._bg_fail_logged is False, "on_session_start 复位告警去重标记")
    HITS.clear()
    d._feed(OmniDescription(t=2, stage="full", text=DESC_FULL))
    _flush(d)
    check(len(_bg_rounds()) == 1, "新会话的第一份照样推出去")
    d.ic.close()


def test_background_not_sent_in_legacy(srv_url: str) -> None:
    """`transcript=legacy` ⇒ sink 根本没被包装 ⇒ 不发背景轮、也不改 payload。

    这条同时验**鸭子判定**：裸 `AgentSink` 上没有 `emit_visual_background`，
    不能用 `isinstance` 去认（那个子类定义在函数体内、没有模块级名字）。
    """
    from orchestrator.downstream.interface import OmniDescription

    print("\n[背景轮] legacy 载荷下不发（sink 未被包装）")
    HITS.clear()
    d = _mk_downstream(srv_url, omni_describe=True,
                       agent_transcript_mode="legacy")
    if not check_ic(d):
        return
    check(type(d.ic.engine.agent_sink).__name__ == "AgentSink",
          "legacy 下 sink 是裸 AgentSink")
    check(d.ic.send_visual_background() is False,
          "裸 sink ⇒ send_visual_background 返回假（鸭子判定，不抛）")
    d._feed(OmniDescription(t=0, stage="full", text=DESC_FULL))
    _flush(d)
    check(_answers() == [], "一条都没发 —— 视觉轮不该在 legacy 下冒出来")
    check(d._bg_fail_logged is True,
          "交不出去 ⇒ 只告警一次（每题描述都重试，但日志不刷屏）")
    d.ic.close()


# ====================================================================== #
#  ③ 时序：主指标
# ====================================================================== #

def test_ready_before_asr_final(srv_url: str) -> None:
    """(a) 描述**早于** `AsrFinal` ⇒ 它已经单独推过了，ASR 轮不再等。"""
    from orchestrator.downstream.interface import OmniDescription

    print("\n[时序] 描述早于断句 ⇒ 已单独推送；ASR 轮不生成、不等待")
    HITS.clear()
    d = _mk_downstream(srv_url, omni_describe=True,
                       agent_transcript_mode="dual")
    if not check_ic(d):
        return
    d._feed(OmniDescription(t=0, stage="full", text=DESC_FULL))
    _flush(d)
    bgs = _bg_rounds()
    check(len(bgs) == 1
          and bgs[0][1].get("content", {}).get("VLM") == DESC_FULL,
          "描述一到就推给了 Agent（不等 ASR）")
    HITS.clear()
    t_feed = time.monotonic()
    _answer_path(d)
    _flush(d)
    ans = _talk_rounds()
    check(d._vlm_miss == 0, f"没有 miss（实际 {d._vlm_miss}）")
    if not ans:
        check(False, "没抓到 on_action(act=ANSWER)")
        d.ic.close()
        return
    hit, body = ans[0]
    check(body.get("content", {}).get("VLM") == "",
          "ASR 轮只带 ASR（视觉状态已经单独推过）")
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
    ans = _talk_rounds()
    check(d._vlm_miss == 1, f"记了 1 次 miss（实际 {d._vlm_miss}）")
    if not ans:
        check(False, "没抓到 on_action(act=ANSWER)")
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
#  ④ 回归：默认态保持基线形态（旧五键 + act，omni 侧不加料）
# ====================================================================== #

#: 默认态（legacy）话轮事件的键集 = 旧五键 + IC 统一入口注入的 `act`。
LEGACY_KEYS = {"session_id", "callback_ic", "identity_id", "display_name",
               "transcript", "act"}


def test_default_keeps_legacy_shape(srv_url: str) -> None:
    """**什么都不设（默认态）⇒ payload 保持基线形态（旧五键 + `act`）。**

    这条同时是「没碰 106」的保证：编排代码 105/106 共用，默认值就是 106
    下次重启后的行为。`ORCH_AGENT_TRANSCRIPT_MODE` 默认 `legacy`、
    `ORCH_OMNI_DESCRIBE` 默认 0 —— 见 config.py 里那两个开关的说明。
    """
    print("\n[回归] 默认态（legacy）payload = 旧五键 + act（omni 侧不加料）")
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
        check(False, "没抓到 on_action(act=ANSWER)")
        d.ic.close()
        return
    body = ans[0][1]
    check(set(body) == LEGACY_KEYS,
          f"键集恰好是旧五键 + act（实际 {sorted(body)}）—— 一个键都不多")
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
          f"键集仍是旧五键 + act（实际 {sorted(ans[0][1]) if ans else '无'}）")
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
      ② `legacy` 下投递行为不变（恰好 1 次 POST、键集 = 旧三件套 + `act`）
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
    check(len(legacy) == 1, f"恰好 1 次 POST（实际 {len(legacy)}）")
    check(legacy and set(legacy[0][1]) == {"act", "identity_id", "display_name",
                                           "transcript"},
          f"envelope 关闭时键集 = 旧三件套 + act（实际 "
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
        check(body.get("content") == {"ASR": "转写", "VLM": ""},
              f"content 注进去了，且**正常话轮 VLM 为空**"
              f"（实际 {body.get('content')!r}）—— 缺 content 说明 `_send` "
              "已不在投递路径上（IC 改版了？）；VLM 非空说明同步发送的老行为"
              "又回来了")
        check(body.get("transcript") == "转写",
              "transcript 未被改动（dual 是**追加**一个键）")

    # ---- 视觉背景轮：与真实轮共用同一根 sink，两发都要在、按序、互不串味 ----
    #
    # `vlm_getter` 用**可变**缓存：这样能同时证明「背景轮取的是**调用那一刻**
    # 的缓存」—— 那是 `emit_visual_background` 不收文本参数的前提条件。
    HITS.clear()
    cache = {"v": "描述文本"}
    sink = make_vlm_agent_sink(target=srv_url, vlm_getter=lambda: cache["v"],
                               mode="dual", **kwargs)
    sink.emit_visual_background()
    cache["v"] = "后来的描述"
    sink.on_answer(identity_id="u", display_name="n", transcript="转写")
    sink.flush(2.0)
    sink.close(2.0)
    both = _answers()
    check(len(both) == 2, f"背景轮 + 真实轮 = 2 次 POST（实际 {len(both)}）")
    if len(both) == 2:
        check(both[0][1].get("content") == {"ASR": "", "VLM": "描述文本"}
              and both[0][1].get("transcript") == "",
              f"#1 是背景轮（ASR / transcript 都空，VLM = 投递那一刻的缓存）"
              f"，实际 {both[0][1]!r} —— 少 content 说明 `_send` 不在投递路径上")
        check(both[1][1].get("content") == {"ASR": "转写", "VLM": ""},
              f"#2 是真实轮：**只带 ASR**，VLM 留空（视觉状态走单独那一路）"
              f"（实际 {both[1][1].get('content')!r}）")

    # ---- dict 模式：背景轮必被 Agent 的 `transcript: str` 挡掉 ⇒ 明确拒绝 ----
    HITS.clear()
    d_sink = make_vlm_agent_sink(target=srv_url, vlm_getter=lambda: "描述文本",
                                 mode="dict", **kwargs)
    check(d_sink.emit_visual_background() is False,
          "dict 模式拒绝发背景轮（返回假，而不是发一条会被 422 的）")
    d_sink.flush(2.0)
    d_sink.close(2.0)
    check(_answers() == [], "dict 模式下**一条都没发出去**")


# ====================================================================== #
#  ⑥ session 侧：两阶段指令 + 滚动刷新（不重叠、可自愈）
# ====================================================================== #

class FakeOmni:
    """假 OmniLLM：只记下收到的指令，不发网络。"""

    turn_trigger = "asr"

    def __init__(self) -> None:
        self.reqs: list = []
        #: 每次触发的关键字参数（`max_new_tokens` 用它验）
        self.kwargs: list = []
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
        self.kwargs.append(dict(kw))
        return True

    async def switch_system_prompt(self, system_prompt: str) -> bool:
        self.switches.append(system_prompt)
        if self.switch_delay:
            await asyncio.sleep(self.switch_delay)
        return self.switch_ok


def test_two_stage_prompts_and_loop() -> None:
    """`session` 侧：起点/每轮指令分开注入；滚动刷新**不重叠、能自愈**。"""
    from orchestrator.omni.describe import FULL_INSTRUCTION, TURN_INSTRUCTION
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
          "第 1 次注入的是**起点全量**指令")
    check(any(t == TURN_INSTRUCTION for _, t in reqs[1:]),
          "之后注入的是**每轮**指令（两阶段靠注入指令区分，不重连不重 init）")
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


def test_vad_boundary_trigger() -> None:
    """**VAD 断句点 ⇒ 立刻发一次增量描述**（主路径），不排队、不重复。

    三件事必须同时成立，缺一条就会把开销优化变成"要么白烧要么漏"：

    * 断句点**不等心跳**（10s 间隔下也要立刻发 —— 否则每个话轮都要等到
      下一次心跳，描述永远落后一个话轮）；
    * 一次断句**只发一次**（`2pass-offline` 可能来多条）；
    * 在途时**不排队、不补发**（与 `_desc_loop` 同一个不变量：一次只有
      一条 decode 在飞，否则服务端共享队列被占满、反而更慢）。
    """
    from orchestrator.omni.describe import FULL_INSTRUCTION, TURN_INSTRUCTION
    from orchestrator.session import OrchestratorSession

    print("\n[断句] VAD 段关闭 ⇒ 立刻发每轮描述；去重；在途不排队")

    def _mk(**cfg):
        base = {"omni_describe": True, "omni_delta_interval_s": 10.0,
                "omni_delta_max_new_tokens": 96}
        base.update(cfg)
        s = OrchestratorSession("t-vad-boundary", base)
        o = FakeOmni()
        s.omni = o
        s.asr = SimpleNamespace(finals=1)
        return s, o

    async def _drive() -> None:
        sess, omni = _mk()
        # 点火（首个 ASR 文本 ⇒ 全量），再回灌 done 让链路进入"可发增量"
        sess._start_desc_on_asr("你好")
        await asyncio.sleep(0.05)
        sess.take_omni_stage(_STATE_FULL)
        check(omni.reqs[0][1] == FULL_INSTRUCTION, "第 1 次是全量")
        check(omni.kwargs[0].get("max_new_tokens") is None,
              "全量**不设** token 上限")

        n0 = len(omni.reqs)
        sess._on_vad_segment_end()
        await asyncio.sleep(0.05)
        check(len(omni.reqs) == n0 + 1,
              f"断句点**立刻**发了一次每轮描述（不等 10s 心跳，实际 +{len(omni.reqs) - n0}）")
        check(omni.reqs[-1][1] == TURN_INSTRUCTION, "注入的是每轮指令")
        check(omni.kwargs[-1].get("max_new_tokens") == 96,
              f"每轮描述带 token 上限（实际 {omni.kwargs[-1]}）")

        # 同一段（`finals` 没变）再来一条 offline ⇒ 不重复发
        n1 = len(omni.reqs)
        sess._on_vad_segment_end()
        await asyncio.sleep(0.05)
        check(len(omni.reqs) == n1, "同一段关闭的多条 offline 只算一次")

        # 在途时新段到达 ⇒ 跳过，不排队
        sess.asr = SimpleNamespace(finals=2)
        n2 = len(omni.reqs)
        sess._on_vad_segment_end()
        await asyncio.sleep(0.05)
        check(len(omni.reqs) == n2, "在途时**不补发**（一次只允许一条 decode 在飞）")
        check(sess.stats.get("omni_desc_skipped_busy") == 1,
              "跳过计数记上了（可观测）")

        # 上一轮 done ⇒ 新段可以发了
        sess.take_omni_stage("EMOTION: 平静")
        sess.asr = SimpleNamespace(finals=3)
        sess._on_vad_segment_end()
        await asyncio.sleep(0.05)
        check(len(omni.reqs) == n2 + 1, "上一轮结束后，下一个断句点照常发")
        check(sess.stats.get("omni_desc_on_boundary", 0) >= 2,
              f"断句触发计数（实际 {sess.stats.get('omni_desc_on_boundary')}）")
        check(sess.stats.get("omni_desc_last_ms") is not None
              and sess.stats.get("omni_desc_last_chars") == len("EMOTION: 平静"),
              "单次描述成本已埋点（耗时 + 字数）")
        sess._stop_desc_loop()
        sess.closed = True

        # 链路没点火（没有 GREET / 没有 ASR）时，断句点不能凭空烧一次
        sess, omni = _mk()
        sess._on_vad_segment_end()
        await asyncio.sleep(0.05)
        check(len(omni.reqs) == 0, "未点火时断句点不发（省下开场那次白烧）")
        sess._stop_desc_loop()
        sess.closed = True

        # 一键回退到纯定时：`ORCH_OMNI_DESC_ON_TURN_BOUNDARY=0`
        sess, omni = _mk(omni_desc_on_turn_boundary=False)
        sess._start_desc_on_asr("你好")
        await asyncio.sleep(0.05)
        sess.take_omni_stage(_STATE_FULL)
        n = len(omni.reqs)
        sess.asr = SimpleNamespace(finals=9)
        sess._on_vad_segment_end()
        await asyncio.sleep(0.05)
        check(len(omni.reqs) == n, "关掉开关 ⇒ 断句点完全不触发（回到旧行为）")
        sess._stop_desc_loop()
        sess.closed = True

    asyncio.run(_drive())


def test_state_contract_prompts() -> None:
    """8 键契约：prompt 侧 + 解析/渲染侧的断言。

    契约本身在 `omni/describe.py` 的模块文档里。这里只盯**容易悄悄漂移**
    的地方 —— 每一处漂移都不会报错，只会让 Agent 收到的状态缺一项：

    * system prompt 必须逐个点名 8 个键（摘掉键表 ⇒ 全量轮塌成一段话，
      2026-10-08 实测 151 字）；
    * `EMOTION` 必须写明"不参考说话内容" —— 08_2 用例（皱眉却说"满意"）
      全靠这一句判，删了就永远跟着语音走；
    * `ACTION` 必须保留"短暂动作已结束也要报" —— 07 的点头/摇头跨采样窗
      即漏；
    * 旧常量 `DEMO_DESCRIBE_PROMPT` / `FULL_FORMAT_LINE` 已删除，
      不许以任何形式回来。
    """
    print("\n[契约] 8 键提示词与解析器")
    from orchestrator.omni.describe import (
        DESCRIBE_SYSTEM_PROMPT, FULL_INSTRUCTION, STATE_KEYS, TURN_INSTRUCTION,
        parse_state, render_state,
    )
    import orchestrator.omni.describe as desc

    check(STATE_KEYS == ("ATTENTION", "ACTION", "EMOTION", "FEATURE",
                         "BEHAVIOR", "RELATION", "ENV", "POSITION"),
          f"键表与顺序（{STATE_KEYS}）")
    missing = [k for k in STATE_KEYS if k not in DESCRIBE_SYSTEM_PROMPT]
    check(not missing, f"system prompt 逐个点名全部 8 键（缺 {missing}）")
    emo = DESCRIBE_SYSTEM_PROMPT.split("EMOTION")[1].split("\n")[0]
    check("绝不参考" in emo and "说了什么" in emo,
          "EMOTION 那一行写明「只看表情姿态、绝不参考说了什么」（08_2 的判据）")
    check("全部输出" in TURN_INSTRUCTION,
          "每轮指令要求 **8 个字段全部输出**（全量方案的全部价值）")
    check("绝不能写“无变化”" in TURN_INSTRUCTION,
          "每轮指令禁止把「无变化」写进字段值（否则 ATTENTION: 无变化 会污染快照）")
    check("即使已经结束也要报" in TURN_INSTRUCTION,
          "每轮指令要求短暂动作结束了也要报（07 的瞬时动作）")
    check("持续秒数" in TURN_INSTRUCTION,
          "每轮指令要求行为写明持续时间（10 的连续行为）")
    check("越短越好" in TURN_INSTRUCTION and "2~10 个字" in TURN_INSTRUCTION,
          "每轮指令要求值压低（成本口径）")
    # 抽象的一句「尽量短」实测**零效果**（turn 147 字 ≈ full 146 字，2026-10-28 105），
    # 所以改成了逐字段预算 + 一份**可照抄长度的示例块** —— 示例块必须留着，
    # 否则又退回"抽象元指令"，模型不会执行。
    check(TURN_INSTRUCTION.count("\n") >= 8
          and "FEATURE: 青年女性，深色长袖" in TURN_INSTRUCTION,
          "每轮指令带一份「照这个长度写」的具体示例块（抽象指令实测无效）")
    check("格式" in FULL_INSTRUCTION and "全部输出" in FULL_INSTRUCTION,
          "起点指令同样交代「8 个字段全部输出」")
    check("不要与用户对话" in DESCRIBE_SYSTEM_PROMPT
          and "不要与用户对话" in TURN_INSTRUCTION,
          "两套指令都保留『不要对话』护栏（口吻锚，删了模型会聊天）")
    check(not hasattr(desc, "DEMO_DESCRIBE_PROMPT")
          and not hasattr(desc, "FULL_FORMAT_LINE"),
          "旧常量 DEMO_DESCRIBE_PROMPT / FULL_FORMAT_LINE 已删除")
    check(not hasattr(desc, "DELTA_SYSTEM_PROMPT")
          and not hasattr(desc, "DELTA_INSTRUCTION"),
          "增量时代的常量已删除（每轮也是全量，不再换人设）")

    # ---- parse_state：宽容解析 ----
    full_text = "\n".join([
        "ATTENTION: 面向屏幕，注视镜头",
        "ACTION: 无",
        "EMOTION: 平静",
        "FEATURE: 青年女性，灰色短袖上衣",
        "BEHAVIOR: 站立，保持不动",
        "RELATION: 单人",
        "ENV: 室内展厅，光线明亮",
        "POSITION: 站在展台右侧",
    ])
    got = parse_state(full_text)
    check(got == {k: v for k, v in zip(STATE_KEYS, [
        "面向屏幕，注视镜头", "无", "平静", "青年女性，灰色短袖上衣",
        "站立，保持不动", "单人", "室内展厅，光线明亮", "站在展台右侧"])},
          f"完整块逐键解析（{got}）")
    check(parse_state("**EMOTION**：不耐烦") == {"EMOTION": "不耐烦"},
          "粗体 + 中文冒号")
    check(parse_state("- 【ACTION】摇头（3 次）") == {"ACTION": "摇头（3 次）"},
          "列表前缀 + 方括号")
    check(parse_state("好的，我看一下。") == {},
          "纯自由文本 ⇒ 空 dict（调用方据此回退原文）")
    check(parse_state("SCENE: 展厅\nSUMMARY: 两人") == {},
          "未知键一律丢弃（模型自己造的字段不破坏契约）")
    check(parse_state("ACTION:") == {}, "只有键没有值 ⇒ 不算（否则会清空该键）")
    # 模型把「无变化」当字段值写（实测 4/138，全在增量轮）⇒ 跳过该键、保旧值
    check(parse_state("ATTENTION: 无变化\nBEHAVIOR: 没有变化\nEMOTION: 不耐烦")
          == {"EMOTION": "不耐烦"},
          "值是「无变化」的键被丢掉（否则会把字段永久污染成「无变化」）")
    check(parse_state("ACTION: 无") == {"ACTION": "无"},
          "裸「无」是**合法值**（没有动作），不许一起过滤掉")

    # ---- render_state：固定顺序 ----
    shuffled = {"POSITION": "右下", "ATTENTION": "背对", "EMOTION": "不耐烦"}
    check(render_state(shuffled) == "ATTENTION: 背对\nEMOTION: 不耐烦\nPOSITION: 右下",
          "渲染按 STATE_KEYS 的固定顺序，缺的键跳过")
    check(render_state({}) == "", "空快照渲染为空串")


def test_state_merge_snapshot(srv_url: str) -> None:
    """每轮描述 ⇒ 编排侧合并成**完整快照**再送 Agent。

    `content.VLM` 必须是 8 键的完整快照。每轮也是全量，所以正常情况下
    `update` 与 `replace` 等价；这里用**只给 2 个键**的输入验的是兜底：
    某一行被截断/没答时，缺的键保旧值而不是丢掉。同时验三条安全阀：

    * 解析失败 ⇒ 退回原文（一个字节都不丢）；
    * 「无变化」⇒ 快照一字不变；
    * 换会话 ⇒ 快照复位（否则新人会继承上一个人的 FEATURE）。
    """
    from orchestrator.downstream.interface import OmniDescription

    print("\n[合并] 每轮 ⇒ 完整快照；解析失败回退原文；换会话复位")
    d = _mk_downstream(srv_url, omni_describe=True,
                       agent_transcript_mode="dual")
    if not check_ic(d):
        return
    d._feed(OmniDescription(t=0, stage="full", text=_STATE_FULL))
    check(d._vlm_state.get("ATTENTION") == "面向屏幕，注视镜头"
          and len(d._vlm_state) == 8, "全量写满 8 键")
    check(d._vlm == _STATE_FULL.strip(), "渲染结果与全量原文逐字一致（顺序未乱）")

    d._feed(OmniDescription(t=1, stage="turn",
                            text="EMOTION: 不耐烦\nACTION: 摇头（3 次）"))
    check(d._vlm_state.get("EMOTION") == "不耐烦"
          and d._vlm_state.get("ACTION") == "摇头（3 次）",
          "增量覆盖出现过的键")
    check(d._vlm_state.get("FEATURE") == "青年女性，灰色短袖上衣"
          and d._vlm_state.get("ENV") == "室内展厅，光线明亮",
          "没出现的键保留旧值（这就是「合并成快照」的全部含义）")
    check(d._vlm.count("\n") == 7, f"快照仍是 8 行（实际 {d._vlm.count(chr(10)) + 1} 行）")
    check("ACTION: 摇头（3 次）" in d._vlm and "EMOTION: 不耐烦" in d._vlm,
          "快照里是新值")

    d._feed(OmniDescription(t=2, stage="turn", text="无变化"))
    check(d._vlm_state.get("EMOTION") == "不耐烦"
          and "ACTION: 摇头（3 次）" in d._vlm,
          "「无变化」不动快照")

    d._feed(OmniDescription(t=3, stage="turn", text="画面上还是刚才那个人。"))
    check(d._vlm == "画面上还是刚才那个人。",
          "解析失败 ⇒ content.VLM 就是原文（不丢数据）")
    check(d._vlm_state == {}, "同时清掉快照 —— 免得下一条增量合并到过时底座上")

    # 跨会话复位：换人后 FEATURE 不能还是上一个人的
    d._feed(OmniDescription(t=4, stage="full", text=_STATE_FULL))
    asyncio.run(d.on_session_start(None))
    check(d._vlm_state == {} and d._vlm == "",
          "on_session_start 清空快照与缓存")

    # 送出去的是快照 —— 走**视觉轮**（ASR 空），不是正常话轮
    d._feed(OmniDescription(t=5, stage="full", text=_STATE_FULL))
    d._feed(OmniDescription(t=6, stage="turn", text="BEHAVIOR: 持续远离 6 秒"))
    _flush(d)
    bgs = _bg_rounds()
    vlm = bgs[-1][1].get("content", {}).get("VLM") if bgs else ""
    check(bool(vlm) and "BEHAVIOR: 持续远离 6 秒" in vlm and vlm.count("\n") == 7,
          f"推给 Agent 的是 8 行完整快照（实际 {len(vlm)} 字）")
    check(len(bgs) >= 2, f"两份描述 ⇒ 两个视觉轮（实际 {len(bgs)}）")
    d.ic.close()


class _FakeSCC:
    """假 `StreamingChatClient` —— 只模拟「握手窗口」这一段时序。

    `connect()` 里要等 gateway 的 `queue_done`（真实现是 `await ws.recv()`），
    这里用 `await asyncio.sleep()` 顶替，并把「握手期间有没有别人来 recv」
    记下来 —— 那正是 2026-10-08 那个 bug 的判据。
    """

    made: list = []

    def __init__(self, url, ssl_ctx=None, echo=False) -> None:
        self.url = url
        self.session_id = f"fake-{len(_FakeSCC.made)}"
        self.backend = "fake"
        self.closed = False
        self.recv_calls = 0
        #: 握手结束那一刻 `recv_calls` 的值 —— **必须为 0**
        self.recv_at_handshake = -1
        self._stop = asyncio.Event()
        _FakeSCC.made.append(self)

    def handle_event(self, ev: dict) -> bool:      # 真实现是方法，这里可替换
        return True

    async def connect(self) -> None:
        await asyncio.sleep(0.25)                  # ⇐ 握手窗口
        self.recv_at_handshake = self.recv_calls

    async def init(self, **kw) -> dict:
        await asyncio.sleep(0.1)
        self.init_kw = kw
        return {"type": "session.created"}

    async def receive_loop(self) -> None:
        self.recv_calls += 1
        await self._stop.wait()
        self.closed = True

    async def close(self, reason: str = "") -> None:
        self.closed = True
        self._stop.set()

    async def send_input(self, **kw) -> None:
        self.sent = getattr(self, "sent", 0) + 1


def test_connect_publishes_client_last() -> None:
    """`connect()` 必须**最后**才发布 `self.client`（2026-10-08 实测的坑）。

    早发布 ⇒ `recv_loop` 抢在 `connect()`/`init()` 的 `ws.recv()` 之前对
    **同一个 ws** 再 recv 一次 ⇒ websockets 抛 "cannot call recv while
    another coroutine is already waiting" ⇒ 被 `StreamingChatClient.
    receive_loop` 的 `except Exception: break` **静默吞掉** ⇒ recv_loop 看到
    连接代数没变，判成"真断线"（`dead=True` + "接收循环已退出"），而
    `connect()` 随后照样跑完并打出"**已连接**"。

    ⇒ 最坏的一种失效：**日志说连上了，整条链路其实已经死了** —— 之后每次
    触发都被短路（"连接已断，触发被跳过"），描述再也不更新。真机上正是
    这么表现的，而且只在**换人设重连**时才会触发（启动时 `recv_loop` 还没起）。
    """
    import streaming_chat_demo
    from orchestrator.omni.client import OmniClient

    print("\n[重连] connect() 最后才发布 client（否则 recv_loop 与握手抢 recv）")
    real = getattr(streaming_chat_demo, "StreamingChatClient", None)
    if real is None:
        check(False, "streaming_chat_demo 里没有 StreamingChatClient（改名了？）")
        return
    _FakeSCC.made = []
    streaming_chat_demo.StreamingChatClient = _FakeSCC

    async def _drive():
        omni = OmniClient("wss://fake/realtime", system_prompt="A")
        await omni.connect()
        task = asyncio.create_task(omni.recv_loop())
        await asyncio.sleep(0.05)                  # 让它挂到第一条连接上
        after_first = _FakeSCC.made[0].recv_calls
        ok = await omni.switch_system_prompt("B")  # ← 换人设：这里最容易踩
        await asyncio.sleep(0.05)                  # 让 recv_loop 接上新连接
        cur = _FakeSCC.made[-1]
        triggers = await omni.trigger_reply("x")
        alive = not omni.dead
        omni.closed = True
        for c in _FakeSCC.made:
            await c.close()
        task.cancel()
        return after_first, ok, alive, triggers, omni, cur

    try:
        after_first, ok, alive, triggers, omni, cur = asyncio.run(_drive())
    finally:
        streaming_chat_demo.StreamingChatClient = real

    check(len(_FakeSCC.made) == 2,
          f"换人设真的重开了一条连接（实际 {len(_FakeSCC.made)} 条）")
    check(after_first == 1, "recv_loop 挂上了第一条连接")
    check(cur.recv_at_handshake == 0,
          f"**新连接的握手窗口里没有别的 coroutine 抢 recv**"
          f"（实际 {cur.recv_at_handshake} 次 —— 非 0 就是 connect() 发布早了）")
    check(ok and alive,
          "重连后 `dead` 必须是 False —— 为 True 说明 recv_loop 把"
          "「自己换掉的连接」误判成了断线，之后所有触发都会被短路")
    check(triggers, "重连后 trigger_reply 送得出去（链路真的活着）")


def test_desc_loop_rate_limits_failures() -> None:
    """触发送不出去时，滚动循环必须按**间隔**重试，不能按轮询步长空转。

    放掉在途标记是必要的（否则要空等满超时），但**同时必须推进
    `_desc_last_done_at`** —— 否则间隔判据永远成立，循环会以 250ms 的轮询
    步长疯狂重试。真机实测（2026-10-08，omni 连接断掉时）：每 250ms 一对
    `触发增量描述` + `触发未能送出`，四秒上百行，把有用的信息全冲掉。
    """
    from orchestrator.session import OrchestratorSession

    print("\n[限流] 触发失败时按间隔重试（不按 250ms 轮询步长空转）")
    sess = OrchestratorSession("t-rate",
                               {"omni_describe": True,
                                "omni_delta_interval_s": 1.0})
    omni = FakeOmni()
    omni.alive = False                 # 每条触发都送不出去
    sess.omni = omni

    async def _drive() -> int:
        await sess.request_omni_description("full")
        # 起步就排空（全量那次也送不出去）
        sess.take_omni_stage("")
        await asyncio.sleep(2.4)
        return len(omni.reqs)

    # `FakeOmni.trigger_reply` 在 alive=False 时**不记** reqs —— 数不到。
    # 改成数 `omni_desc_failed`（session 每次失败 +1），那才是"真的尝试了"。
    n_calls = asyncio.run(_drive())
    sess._stop_desc_loop()
    sess.closed = True
    failed = sess.stats.get("omni_desc_failed", 0)
    # 1.0s 间隔 / 2.4s 窗口 ⇒ 期望 ≈3 次；没有限流会是 ~10 次（250ms 一次）
    check(failed <= 5,
          f"2.4s 内失败重试 {failed} 次 —— 应按 1.0s 间隔（≈3），"
          f"按轮询步长会是 ~10")
    check(failed >= 2, f"……而且**确实**在重试（{failed} 次，不是 0）")
    check(n_calls == 0, "触发根本没送出去（FakeOmni 已断）")


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
        test_visual_background_round(url)
        test_turn_pushes_visual_round(url)
        test_visual_push_every_description(url)
        test_background_not_sent_in_legacy(url)
        test_ready_before_asr_final(url)
        test_missing_description_never_waits(url)
        test_default_keeps_legacy_shape(url)
        test_legacy_mode_with_describe_on(url)
        test_illegal_mode_falls_back_to_legacy(url)
        test_contract_drift(url)
        test_state_contract_prompts()
        test_state_merge_snapshot(url)
        test_two_stage_prompts_and_loop()
        test_answer_path_starts_desc_without_greet()
        test_vad_boundary_trigger()
        test_connect_publishes_client_last()
        test_desc_loop_rate_limits_failures()
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
