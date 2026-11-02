#!/usr/bin/env python3
"""新增感知状态的**透传**验证（ASR 语音子集 + VLM 八键 → IC）。

IC 的 P05 给 `apply_asr` 加了感知字段、并新开了 `apply_vision`。本方验证
编排侧那两条新链路，且**两条都必须按 IC 版本自适应**：

  ① 新 IC：`AsrStateUpdate` 里的语种/情绪/事件/SNR/噪声突发/交叠 → `apply_asr`
  ② 新 IC：每份 VLM 描述 → `apply_vision`（**大写键转小写**，不打时间戳）
  ③ 旧 IC（P05 之前）：两块都**整组不发**，基础三个参数照旧 —— 不能抛、
     不能每 50ms 刷一条 TypeError
  ④ 值为 None / 空快照：**不写**（proto 语义是"省略 = 不改"）

用假 IC 客户端（只记账）跑真 `InteractionDownstream` —— 这里要验的是**映射
与版本门**，不是 IC 自己的状态机（那由 interactioncore 的测试覆盖）。

    python -m orchestrator.tests.test_ic_state_passthrough
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


#: P05 的 `apply_asr` 认的全部参数（含基础三件套）
_ASR_P05 = {
    "transcript", "asr_confidence", "turn_complete_confidence",
    "language", "language_prob", "language_switch",
    "emotion", "emotion_prob", "speech_event", "snr_db",
    "noise_burst_detected", "noise_burst_type",
    "overlap_detected", "overlap_prob",
}

#: P05 之前的 `apply_asr`
_ASR_OLD = {"transcript", "asr_confidence", "turn_complete_confidence"}

#: P05 的 `apply_vision`
_VISION_P05 = {
    "attention", "action", "emotion", "feature", "behavior",
    "relation", "env", "position", "updated_at_ms",
}


class FakeIC:
    """记账用的 IC 客户端（鸽鸭接口：available / supports / apply / …）。"""

    def __init__(self, methods: dict) -> None:
        #: {方法名: 它认得的参数名集合}；方法不在表里 = 旧 IC 没这个方法
        self.methods = methods
        self.calls: list = []
        self.available = True
        self.bg_calls = 0
        #: `bind_session` 收到的信封（顺序也要能看出来）
        self.session_binds: list = []
        #: 新版 IC：`apply_vision` 自己会把合并快照当 Vision 通知发给 Agent
        self.owns_vision_round = False

    def ic_owns_vision_round(self) -> bool:
        return self.owns_vision_round

    def reset_session(self) -> bool:
        self.calls.append(("reset_session", {}))
        return True

    def bind_session(self, session_id: str, callback_ic: str) -> bool:
        self.session_binds.append((session_id, callback_ic))
        self.calls.append(("bind_session",
                           {"session_id": session_id, "callback_ic": callback_ic}))
        return True

    def supports(self, method: str, *keys: str) -> bool:
        accepted = self.methods.get(method)
        if accepted is None:
            return False
        return all(k in accepted for k in keys)

    def apply(self, method: str, **kwargs) -> None:
        self.calls.append((method, kwargs))

    def send_visual_background(self) -> bool:
        self.bg_calls += 1
        return False

    def last(self, method: str) -> dict:
        for m, kw in reversed(self.calls):
            if m == method:
                return kw
        return {}

    def count(self, method: str) -> int:
        return sum(1 for m, _ in self.calls if m == method)


def _mk(ic_methods: dict, omni_describe: bool = False):
    """真 `InteractionDownstream` + 假 IC（不连任何服务）。"""
    from orchestrator.interaction.downstream import InteractionDownstream
    d = InteractionDownstream(
        "inprocess", "", session_id="t-pass",
        ic_mode="inprocess", agent_set_target=False,
        ic_expression_url="", ic_callback_ic="",
        # 默认关：不测「视觉轮」，只测两条透传链路。测让路那条要打开。
        omni_describe=omni_describe,
    )
    d.ic = FakeIC(ic_methods)
    return d


#: 一段「刚说完」的服务端快照（含感知字段，形状取自 `AsrState.to_dict()`）
_ASR_STATE = {
    "user_speaking_confidence": "HIGH",
    "barge_in_confidence": "LOW",
    "transcript": "你好呀",
    "asr_confidence": "HIGH",
    "turn_complete_confidence": "HIGH",
    "language": "zh",
    "language_prob": 0.98,
    "language_switch": False,
    "emotion": "happy",
    "emotion_prob": 0.71,
    "speech_event": "speech",
    "noise": {"calibrated": True, "floor_dbfs": -53.8,
              "burst": {"detected": True, "prob": 0.65, "type": "pop"}},
    "overlap": {"detected": False, "prob": 0.098, "regions": []},
    "snr_db": 21.5,
}

#: 一段 8 键全量的 VLM 描述（**大写键**，与 `parse_state` 的输出一致）
_VLM_TEXT = """ATTENTION: 面向屏幕，注视镜头
ACTION: 点头两次
EMOTION: 满意
FEATURE: 青年女性，灰色短袖上衣
BEHAVIOR: 站立，持续 3 秒
RELATION: 单人
ENV: 展厅，明亮
POSITION: 画面中央"""


# ====================================================================== #
#  ① ASR 语音子集
# ====================================================================== #

def test_asr_perception_forwarded() -> None:
    print("\n== 新 IC：ASR 感知字段跟进 apply_asr ==")
    from orchestrator.downstream.interface import AsrStateUpdate

    d = _mk({"apply_asr": _ASR_P05})
    d._feed(AsrStateUpdate(t=1, state=dict(_ASR_STATE)))
    kw = d.ic.last("apply_asr")

    check(kw.get("transcript") == "你好呀", "基础三件套照旧（transcript）")
    for k, v in (("language", "zh"), ("language_prob", 0.98),
                 ("language_switch", False), ("emotion", "happy"),
                 ("emotion_prob", 0.71), ("speech_event", "speech"),
                 ("snr_db", 21.5)):
        check(kw.get(k) == v, f"{k} 透传（期望 {v!r}，实际 {kw.get(k)!r}）")
    check(kw.get("noise_burst_detected") is True,
          f"noise.burst.detected 拆平成 noise_burst_detected"
          f"（实际 {kw.get('noise_burst_detected')!r}）")
    check(kw.get("noise_burst_type") == "pop",
          f"noise.burst.type → noise_burst_type（实际 {kw.get('noise_burst_type')!r}）")
    check(kw.get("overlap_detected") is False,
          "overlap.detected → overlap_detected（**False 也要传**，"
          "它表示'这一段测过，没有交叠'）")
    check(kw.get("overlap_prob") == 0.098, "overlap.prob → overlap_prob")
    check("noise" not in kw and "overlap" not in kw,
          "嵌套字典本身**不能**塞进 kwargs（proto 里没有这两个字段）")


def test_asr_perception_none_skipped() -> None:
    print("\n== 新 IC：没测出来的感知字段（None）不写 ==")
    from orchestrator.downstream.interface import AsrStateUpdate

    d = _mk({"apply_asr": _ASR_P05})
    st = {k: v for k, v in _ASR_STATE.items()
          if k in ("user_speaking_confidence", "barge_in_confidence",
                   "transcript", "asr_confidence", "turn_complete_confidence")}
    d._feed(AsrStateUpdate(t=1, state=st))
    kw = d.ic.last("apply_asr")

    check(kw.get("transcript") == "你好呀", "基础转写照写")
    leaked = [k for k in ("language", "emotion", "speech_event", "snr_db",
                          "noise_burst_detected", "noise_burst_type",
                          "overlap_detected", "overlap_prob") if k in kw]
    check(not leaked, f"None 的感知键一个都不带（实际带了 {leaked}）")


def test_asr_perception_skipped_on_old_ic() -> None:
    print("\n== 旧 IC：整组不发，基础三件套不受影响 ==")
    from orchestrator.downstream.interface import AsrStateUpdate

    d = _mk({"apply_asr": _ASR_OLD})
    d._feed(AsrStateUpdate(t=1, state=dict(_ASR_STATE)))
    kw = d.ic.last("apply_asr")

    check(d.ic.count("apply_asr") == 1, "仍然写一次 apply_asr")
    check(set(kw) == _ASR_OLD,
          f"参数集恰好等于旧 IC 认的三个（实际 {sorted(kw)}）")
    check("asr_perception" in d._cap_warned, "「版本不支持」的告警已记录（只打一次）")


# ====================================================================== #
#  ② VLM 八键
# ====================================================================== #

def test_vision_forwarded() -> None:
    print("\n== 新 IC：VLM 八键 → apply_vision（大写键转小写） ==")
    from orchestrator.downstream.interface import OmniDescription

    d = _mk({"apply_asr": _ASR_P05, "apply_vision": _VISION_P05})
    d._on_vlm_description(OmniDescription(t=0, stage="turn", text=_VLM_TEXT))
    kw = d.ic.last("apply_vision")

    check(d.ic.count("apply_vision") == 1, "发了一次 apply_vision")
    check(kw.get("attention") == "面向屏幕，注视镜头",
          f"ATTENTION → attention（实际 {kw.get('attention')!r}）")
    check(kw.get("action") == "点头两次", "ACTION → action")
    check(kw.get("emotion") == "满意", "EMOTION → emotion")
    check(kw.get("feature") == "青年女性，灰色短袖上衣", "FEATURE → feature")
    check(kw.get("behavior") == "站立，持续 3 秒", "BEHAVIOR → behavior")
    check(kw.get("relation") == "单人", "RELATION → relation")
    check(kw.get("env") == "展厅，明亮", "ENV → env")
    check(kw.get("position") == "画面中央", "POSITION → position")
    check("updated_at_ms" not in kw,
          "时间戳由 IC 自己打（两边墙钟未必同源，传我们的反而更不准）")


def test_vision_skipped_on_old_ic() -> None:
    print("\n== 旧 IC：没有 apply_vision -> 不发，也不抛 ==")
    from orchestrator.downstream.interface import OmniDescription

    d = _mk({"apply_asr": _ASR_OLD})       # 没有 apply_vision
    d._on_vlm_description(OmniDescription(t=0, stage="full", text=_VLM_TEXT))

    check(d.ic.count("apply_vision") == 0, "一次 apply_vision 都不发")
    check(d._vlm_state, "编排侧的 VLM 快照照常保留（Agent 那条链路不受影响）")
    check("vision" in d._cap_warned, "「版本不支持」的告警已记录（只打一次）")


def test_vision_skipped_when_unparsed() -> None:
    print("\n== VLM 没按格式答（解析不出键）-> 不拿空值覆盖 IC ==")
    from orchestrator.downstream.interface import OmniDescription

    d = _mk({"apply_asr": _ASR_P05, "apply_vision": _VISION_P05})
    d._on_vlm_description(
        OmniDescription(t=0, stage="turn", text="好的，一切正常。"))

    check(d.ic.count("apply_vision") == 0,
          "解析失败时 apply_vision 一次都不发（快照已清空，不该覆盖）")
    check(d._vlm == "好的，一切正常。",
          "原文仍留给 Agent（这是本改动唯一的安全阀）")


def test_vision_round_yielded_to_new_ic() -> None:
    print("\n== 新 IC（apply_vision 自己发通知）-> 编排侧不补视觉轮 ==")
    from orchestrator.downstream.interface import OmniDescription

    d = _mk({"apply_asr": _ASR_P05, "apply_vision": _VISION_P05},
            omni_describe=True)
    d.ic.owns_vision_round = True
    d._on_vlm_description(OmniDescription(t=0, stage="turn", text=_VLM_TEXT))

    check(d.ic.count("apply_vision") == 1, "快照照常写进 IC")
    check(d.ic.bg_calls == 0,
          f"编排侧的视觉背景轮**不发**（否则 Agent 每份描述收两轮，"
          f"实际发了 {d.ic.bg_calls} 次）")


def test_vision_round_kept_on_old_ic() -> None:
    print("\n== 旧 IC（不自己发通知）-> 编排侧照旧补视觉轮 ==")
    from orchestrator.downstream.interface import OmniDescription

    d = _mk({"apply_asr": _ASR_P05, "apply_vision": _VISION_P05},
            omni_describe=True)
    d._on_vlm_description(OmniDescription(t=0, stage="turn", text=_VLM_TEXT))

    check(d.ic.count("apply_vision") == 1, "快照照常写进 IC")
    check(d.ic.bg_calls == 1,
          f"旧 IC 不会自己发 ⇒ 编排侧必须补（实际 {d.ic.bg_calls} 次）")


def test_vision_round_kept_when_unparsed_even_on_new_ic() -> None:
    print("\n== 新 IC 但解析失败（没写 apply_vision）-> 原文照样走视觉轮 ==")
    from orchestrator.downstream.interface import OmniDescription

    d = _mk({"apply_asr": _ASR_P05, "apply_vision": _VISION_P05},
            omni_describe=True)
    d.ic.owns_vision_round = True
    d._on_vlm_description(
        OmniDescription(t=0, stage="turn", text="好的，一切正常。"))

    check(d.ic.count("apply_vision") == 0, "没内容可写 ⇒ 一次 apply_vision 都不发")
    check(d.ic.bg_calls == 1,
          "IC 不会发通知 ⇒ 编排侧必须把原文推给 Agent（安全阀不能跟着关）")


# ====================================================================== #
#  ③ 会话启动：reset_session → bind_session（远端 IC 单例的信封）
# ====================================================================== #

def test_session_bound_on_start() -> None:
    print("\n== 会话启动：reset_session 之后把本会话信封绑到 IC ==")
    d = _mk({})
    d.session_id = "t-bind"
    cb = "https://192.168.89.106:8100/v1/ic"
    d.agent.callback_ic = cb
    asyncio.run(d.on_session_start(None))
    check(d.ic.session_binds == [("t-bind", cb)],
          f"bind_session 收到本会话信封（实际 {d.ic.session_binds}）")
    names = [m for m, _ in d.ic.calls]
    check("reset_session" in names and "bind_session" in names
          and names.index("reset_session") < names.index("bind_session"),
          "顺序：先 reset_session 再 bind_session（与远端启动序列一致）")


def test_bind_session_failure_is_not_fatal() -> None:
    print("\n== bind_session：旧 IC / 空 callback_ic 只告警，不阻断会话 ==")
    from orchestrator.interaction.inprocess import InProcessICClient

    ic = InProcessICClient("t-bind-guard")
    ic.available = True
    ic._engine = object()          # 没有 bind_session 的旧 Engine
    check(ic.bind_session("s", "https://192.168.89.106:8100/v1/ic") is False,
          "旧 Engine（无 bind_session）返回 False，不抛")
    check(ic.bind_session("s", "") is False,
          "callback_ic 为空 ⇒ IC 会整次忽略，直接返回 False")


# ====================================================================== #
#  ④ P05 打断探针的装配门禁
# ====================================================================== #

def test_decision_probe_gated_by_engine_signature() -> None:
    print("\n== 打断探针：只装配到认 decision_probe 的 Engine ==")
    try:
        import interaction  # noqa: F401
    except ImportError:
        print("  [SKIP] 本机没装 interactioncore")
        return
    import types
    from orchestrator.interaction.inprocess import InProcessICClient

    class OldEngine:
        def __init__(self, expression_sink=None, agent_sink=None,
                     thresholds=None):
            pass

    class P05Engine:
        def __init__(self, expression_sink=None, agent_sink=None,
                     thresholds=None, decision_probe=None):
            self.decision_probe = decision_probe

    ic = InProcessICClient("t-decision-probe")
    check(ic._make_decision_probe(OldEngine) is None,
          "老 Engine（没有 decision_probe）不应建探针 —— 软打断那条线不走探针")

    made: list = []

    class FakeProbe:
        def __init__(self, *a, **kw):
            self.client = types.SimpleNamespace(base_url="fake://startlux")
            made.append(self)

        def close(self) -> None:
            pass

    fake_mod = types.ModuleType("interaction.decision_probe")
    fake_mod.DecisionProbe = FakeProbe
    saved = sys.modules.get("interaction.decision_probe")
    sys.modules["interaction.decision_probe"] = fake_mod
    try:
        probe = ic._make_decision_probe(P05Engine)
        check(probe is not None and made == [probe],
              "P05 Engine 必须装配打断探针 —— 没有它 barge 永远到不了 HARD，"
              "等于升级后「不打断」")
    finally:
        if saved is None:
            sys.modules.pop("interaction.decision_probe", None)
        else:
            sys.modules["interaction.decision_probe"] = saved


# ====================================================================== #
#  ④ 真客户端的版本探测（防「探测到 Engine 的 **kwargs 壳子」那个坑）
# ====================================================================== #

def test_real_client_probes_state_method() -> None:
    print("\n== 真 InProcessICClient：探测的是 engine.state 上的真方法 ==")
    try:
        from interaction.state import InteractionState  # noqa: F401
    except ImportError:
        print("  [SKIP] 本机没装 interactioncore")
        return
    from orchestrator.interaction.inprocess import InProcessICClient

    ic = InProcessICClient("t-probe", expression_url="", agent_url="")
    if not ic.connect():
        print(f"  [SKIP] 进程内 Engine 起不来：{ic.error}")
        return
    try:
        # ⚠️ Engine.apply_asr 是 `_locked` 包出来的 (*args, **kwargs) ——
        #    拿它探测会**永远**返回"认"，于是旧 IC 上每秒几十条 TypeError。
        #    结论必须与 `engine.state.apply_asr` 的真签名一致。
        import inspect
        state_fn = ic.engine.state.apply_asr
        has_language = "language" in inspect.signature(state_fn).parameters
        check(ic.supports("apply_asr", "language") is has_language,
              f"supports('apply_asr','language') == state 方法认不认"
              f"（state={has_language}，supports="
              f"{ic.supports('apply_asr', 'language')}）")
        check(ic.supports("apply_asr", "transcript") is True,
              "旧参数在新旧 IC 上都认")
        has_vision = hasattr(ic.engine.state, "apply_vision")
        check(ic.supports("apply_vision") is has_vision,
              f"apply_vision 的存在性探测 == state 上有没有该方法"
              f"（state={has_vision}）")
        # 新版 `Engine.apply_vision` 正是 `getattr(agent_sink, "on_vision")`
        # ⇒ 用 sink 上有没有 `on_vision` 判「IC 自己发不发 Vision 通知」。
        from interaction.runtime import AgentSink
        has_notice = callable(getattr(AgentSink, "on_vision", None))
        check(ic.ic_owns_vision_round() is False,
              "没配 agent_url ⇒ 没有 Agent sink ⇒ 谁都不发，判假")
        has_bind = callable(getattr(ic.engine, "bind_session", None))
        bound = ic.bind_session("t-bind", "https://192.168.89.106:8100/v1/ic")
        check(bound is has_bind,
              f"bind_session 结果 == Engine 上有没有该方法"
              f"（Engine={has_bind}，返回值={bound}）—— 旧版只告警不阻断")

        # 真有 Agent sink 时，才轮到「IC 自己发不发」的分叉
        ic2 = InProcessICClient("t-notice", expression_url="",
                                agent_url="http://127.0.0.1:9/interaction")
        if not ic2.connect():
            print(f"  [SKIP] 第二个 Engine 起不来：{ic2.error}")
            return
        try:
            check(ic2.ic_owns_vision_round() is has_notice,
                  f"ic_owns_vision_round() == sink 上有没有 on_vision"
                  f"（on_vision={has_notice}，返回值="
                  f"{ic2.ic_owns_vision_round()}）")
        finally:
            ic2.close()
    finally:
        ic.close()


# ====================================================================== #

def main() -> int:
    print("=" * 68)
    print("感知状态透传（ASR 语音子集 / VLM 八键 → IC）验证")
    print("-" * 68)
    test_asr_perception_forwarded()
    test_asr_perception_none_skipped()
    test_asr_perception_skipped_on_old_ic()
    test_vision_forwarded()
    test_vision_skipped_on_old_ic()
    test_vision_skipped_when_unparsed()
    test_vision_round_yielded_to_new_ic()
    test_vision_round_kept_on_old_ic()
    test_vision_round_kept_when_unparsed_even_on_new_ic()
    test_session_bound_on_start()
    test_bind_session_failure_is_not_fatal()
    test_decision_probe_gated_by_engine_signature()
    test_real_client_probes_state_method()
    print()
    if _failures:
        print(f"失败 {len(_failures)} 项:")
        for f in _failures:
            print(f"  · {f}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
