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


def _mk(ic_methods: dict):
    """真 `InteractionDownstream` + 假 IC（不连任何服务）。"""
    from orchestrator.interaction.downstream import InteractionDownstream
    d = InteractionDownstream(
        "inprocess", "", session_id="t-pass",
        ic_mode="inprocess", agent_set_target=False,
        ic_expression_url="", ic_callback_ic="",
        omni_describe=False,          # 不测「视觉轮」，只测两条透传链路
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


# ====================================================================== #
#  ③ P05 打断探针的装配门禁
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
