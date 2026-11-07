"""Description staging plus callback-to-Agent protobuf mapping, without network I/O."""
from __future__ import annotations

import asyncio
import json
import queue
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_protocol import agent_pb2 as pb
from orchestrator.downstream.interface import AsrFinal, Describe, OmniDescription
from orchestrator.interaction.downstream import InteractionDownstream


def check(cond: bool, msg: str) -> None:
    assert cond, msg


_STATE_FULL = """ATTENTION: 面向屏幕，注视镜头
ACTION: 无
EMOTION: 平静
FEATURE: 青年女性，灰色短袖上衣
BEHAVIOR: 站立，保持不动
RELATION: 单人
ENV: 室内展厅，光线明亮
POSITION: 站在展台右侧"""


@pytest.fixture
def downstream(monkeypatch):
    monkeypatch.setenv("ORCH_IC_DECISION_PROBE", "0")
    d = InteractionDownstream("inprocess", "127.0.0.1:1", session_id="description",
                              omni_describe=True)
    assert d.ic.connect(), d.ic.error
    yield d
    d.agent.close()
    d.ic.close()


def wire_actions(d):
    async def drain():
        d._drain_events()
        await asyncio.sleep(0)
    asyncio.run(drain())
    actions = []
    while True:
        try:
            item = d.agent._queue.get_nowait()
        except queue.Empty:
            return actions
        if item is not None:
            actions.append(item)
        d.agent._queue.task_done()


def test_greet_entry_and_reentry_trigger_full_description(downstream):
    d = downstream
    greet = SimpleNamespace(type=SimpleNamespace(value="GREET"), sop="15",
                            text="您好", transcript=None)
    quiet = SimpleNamespace(type=SimpleNamespace(value="HOLD"), sop=None,
                            text=None, transcript=None)
    assert [act.stage for act in d._dispatch(greet) if isinstance(act, Describe)] == ["full"]
    assert not [act for act in d._dispatch(greet) if isinstance(act, Describe)]
    d._dispatch(quiet)
    assert [act.stage for act in d._dispatch(greet) if isinstance(act, Describe)] == ["full"]


def test_description_callback_maps_to_vision_then_answer_without_changing_transcript(downstream):
    d = downstream
    d._feed(OmniDescription(t=0, stage="full", text=_STATE_FULL))
    d.ic.engine.agent_sink.on_answer(identity_id="uid-甲", display_name="甲",
                                     transcript="介绍一下展厅", language="zh",
                                     affect={"label": "happy", "confidence": 0.9})
    visual, answer = wire_actions(d)
    assert visual.act == pb.AgentAction.ANSWER and visual.transcript == ""
    assert visual.vlm == _STATE_FULL
    assert answer.act == pb.AgentAction.ANSWER
    assert answer.transcript == "介绍一下展厅" and answer.vlm == ""
    assert answer.identity_id == "uid-甲" and answer.display_name == "甲"
    assert answer.language == "zh"
    assert answer.affect.label == "happy" and answer.affect.confidence == pytest.approx(0.9)
    assert answer.session_id == "description" and answer.epoch > visual.epoch


def test_state_merge_publishes_one_complete_snapshot_per_description(downstream):
    d = downstream
    d._feed(OmniDescription(t=0, stage="full", text=_STATE_FULL))
    first = wire_actions(d)
    assert len(first) == 1
    d._feed(OmniDescription(t=1, stage="turn", text="EMOTION: 不耐烦\nACTION: 摇头（3 次）"))
    second = wire_actions(d)
    assert len(second) == 1
    assert "EMOTION: 不耐烦" in second[0].vlm
    assert "ACTION: 摇头（3 次）" in second[0].vlm
    assert "FEATURE: 青年女性，灰色短袖上衣" in second[0].vlm
    assert second[0].vlm.count("\n") == 7
    d._feed(OmniDescription(t=2, stage="turn", text="无变化"))
    d._feed(OmniDescription(t=3, stage="turn", text=""))
    assert wire_actions(d) == []
    assert d._vlm_state["EMOTION"] == "不耐烦"


def test_unstructured_description_is_preserved_for_agent(downstream):
    d = downstream
    description = "画面里一位青年男性穿深蓝色外套，站在展示桌左侧。" * 80
    d._feed(OmniDescription(t=0, stage="full", text=description))
    messages = wire_actions(d)
    assert len(messages) == 1
    assert messages[0].transcript == "" and messages[0].vlm == description
    assert d._vlm_state == {}


def test_description_delivery_does_not_wait_for_asr(downstream):
    d = downstream
    d._feed(OmniDescription(t=0, stage="full", text=_STATE_FULL))
    visual = wire_actions(d)
    assert len(visual) == 1 and visual[0].transcript == ""
    d._feed(AsrFinal(t0=0, t1=100, text="你好"))
    assert wire_actions(d) == []  # ASR state write does not duplicate the vision notice.


def test_missing_description_does_not_delay_agent_action(downstream):
    d = downstream
    d.ic.engine.agent_sink.on_answer(identity_id=None, display_name=None, transcript="你好")
    messages = wire_actions(d)
    assert len(messages) == 1
    assert messages[0].transcript == "你好" and messages[0].vlm == ""


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


def test_desc_loop_fires_on_interval_while_user_is_speaking() -> None:
    """定时触发**不看语音活动**：用户正在说话也要按间隔触发（2026-11-07 口径）。

    旧口径（`omni_delta_always=0`）里心跳只服务静默，说话期间只有断句点那
    一次；现在默认叠加一个"每 interval 一次"的定时触发。不重叠仍由在途标记
    保证 —— 这里每 50ms 回灌一次 `take_omni_stage` 模拟 `response.done`。
    """
    from orchestrator.session import OrchestratorSession

    print("\n[定时] 说话期间也按间隔触发；关掉开关则回到仅静默")

    async def _drive(always: bool, window: float = 1.2) -> int:
        sess = OrchestratorSession(
            "t-always",
            {"omni_describe": True, "omni_delta_interval_s": 0.2,
             "omni_desc_stale_s": 0.0, "omni_delta_always": always})
        omni = FakeOmni()
        sess.omni = omni
        # 手动点火：不经过 GREET / ASR，只验滚动循环的定时分支
        sess._desc_started = True
        sess._desc_last_done_at = 0.0
        sess._ensure_desc_loop()
        end = time.monotonic() + window
        while time.monotonic() < end:
            # 持续有语音活动 —— 旧口径下这里一条都不会发
            sess._last_speech_at = time.monotonic()
            await asyncio.sleep(0.05)
            # 模拟 response.done。⚠️ 只能**在途时**回灌：`take_omni_stage`
            # 会推进 `_desc_last_done_at`，每 50ms 无条件调一次等于把定时
            # 间隔重置成 50ms，循环永远等不到下一次。
            if sess._desc_inflight:
                sess.take_omni_stage("EMOTION: 平静")
        n = len(omni.reqs)
        sess._stop_desc_loop()
        sess.closed = True
        return n

    on = asyncio.run(_drive(True))
    check(on >= 2, f"说话期间仍按间隔触发（0.2s×1.2s 期望 ≥2，实际 {on}）")
    off = asyncio.run(_drive(False))
    check(off == 0, f"关掉开关 ⇒ 说话期间不发（实际 {off}）")


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



if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
