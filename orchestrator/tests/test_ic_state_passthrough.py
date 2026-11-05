"""Perception forwarding into the pinned local InteractionCore algorithm."""
from __future__ import annotations

import asyncio
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_failures: list = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    assert cond, msg


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
        #: 新版 IC：`apply_vision` 自己会把合并快照当 Vision 通知发给 Agent
        self.owns_vision_round = False

    def ic_owns_vision_round(self) -> bool:
        return self.owns_vision_round

    def reset_session(self) -> bool:
        self.calls.append(("reset_session", {}))
        return True

    def supports(self, method: str, *keys: str) -> bool:
        accepted = self.methods.get(method)
        if accepted is None:
            return False
        return all(k in accepted for k in keys)

    def apply(self, method: str, **kwargs) -> None:
        self.calls.append((method, kwargs))

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
        ic_mode="inprocess",
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



@pytest.fixture(autouse=True)
def no_external_probe(monkeypatch):
    monkeypatch.setenv("ORCH_IC_DECISION_PROBE", "0")


def test_local_session_start_resets_state_without_remote_binding(monkeypatch):
    from orchestrator.interaction.downstream import InteractionDownstream
    d = InteractionDownstream("inprocess", "127.0.0.1:1", session_id="local-start")
    assert d.ic.connect(), d.ic.error
    monkeypatch.setattr(d.agent, "start", lambda: None)
    try:
        d.ic.apply("apply_asr", transcript="上次文本", asr_confidence="HIGH")
        d._vlm = "旧画面"
        asyncio.run(d.on_session_start(None))
        assert d.ic.engine.state.speech.transcript == ""
        assert d._vlm == "" and d._vlm_state == {}
    finally:
        d.agent.close()
        d.ic.close()


def test_probe_is_owned_and_closed_by_local_adapter(monkeypatch):
    from interaction import decision_probe
    from orchestrator.interaction.inprocess import InProcessICClient
    made = []
    class Probe:
        def __init__(self):
            self.closed = False
            made.append(self)
        def close(self):
            self.closed = True
    monkeypatch.setattr(decision_probe, "DecisionProbe", Probe)
    monkeypatch.setenv("ORCH_IC_DECISION_PROBE", "1")
    client = InProcessICClient("probe-enabled")
    assert client.connect(), client.error
    assert len(made) == 1
    assert client.engine.decision_probe is made[0]
    client.close()
    assert made[0].closed
    monkeypatch.setenv("ORCH_IC_DECISION_PROBE", "0")
    disabled = InProcessICClient("probe-disabled")
    try:
        assert disabled.connect(), disabled.error
        assert disabled.engine.decision_probe is None
        assert len(made) == 1
    finally:
        disabled.close()


def test_real_capability_detection_uses_the_state_signature():
    from orchestrator.interaction.inprocess import InProcessICClient
    events = []
    client = InProcessICClient("caps", on_agent_event=events.append)
    assert client.connect(), client.error
    try:
        assert client.supports("apply_asr", "transcript", "language")
        assert not client.supports("apply_asr", "unknown_keyword")
        assert not client.supports("missing_method", "anything")
        assert client.supports("apply_vision", "attention", "position")
        client.apply("apply_vision", attention="面向屏幕", position="中央")
        assert len(events) == 1
        assert events[0]["type"] == "vision"
        assert events[0]["session_id"] == "caps"
        assert "ATTENTION: 面向屏幕" in events[0]["vlm"]
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
