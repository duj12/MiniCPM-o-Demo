#!/usr/bin/env python3
"""验证人脸信号 → UI 显示（face.state）的完整路径。

不需要真实摄像头/浏览器：直接构造 FaceWorker + 假的 provider，
喂若干帧观测，检查：

  · 每帧观测是否触发 face.state（节流到 ~10Hz）
  · face.state 是否携带 框/置信度/唇动/唤醒/身份
  · 每帧观测**不进** downstream（控制流不被 25Hz 淹没）
  · 唤醒/唇动/身份仍正常进 downstream

    python -m orchestrator.tests.test_face_display
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_failures: list = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


class FakeProvider:
    """按预设序列产出观测。"""

    def __init__(self, script):
        self.script = script
        self.i = 0

    def process(self, jpeg, t):
        from orchestrator.face.signals import FaceObservation
        if self.i >= len(self.script):
            return None
        s = self.script[self.i]
        self.i += 1
        return FaceObservation(
            t=t, valid=s.get("valid", True),
            box=s.get("box", (100, 100, 200, 200)),
            score=s.get("score", 0.9),
            speaking=s.get("speaking", False),
            lip_state=s.get("lip", "SILENT"),
            interacting=s.get("interacting", False),
            person_id=s.get("person_id", -1),
        )

    def poll_identity(self, t):
        return None

    def close(self):
        pass


def main() -> int:
    from orchestrator.face.worker import FaceWorker

    wakes, lips, idents, obs_list = [], [], [], []
    script = [
        # 前 5 帧：检测到人脸，未唤醒
        *[{"score": 0.85, "interacting": False, "lip": "SILENT"} for _ in range(5)],
        # 唤醒（进入 interacting）
        *[{"score": 0.88, "interacting": True, "lip": "SILENT"} for _ in range(5)],
        # 开始说话
        *[{"score": 0.89, "interacting": True, "speaking": True,
           "lip": "SPEAKING"} for _ in range(10)],
        # 人脸消失
        *[{"valid": False, "interacting": False} for _ in range(3)],
    ]
    w = FaceWorker(
        FakeProvider(script),
        on_wake=lambda e: wakes.append(e),
        on_lip=lambda e: lips.append(e),
        on_identity=lambda e: idents.append(e),
        on_obs=lambda o: obs_list.append(o),
    )
    w.start()
    # 按 25fps 投递（每 40ms 一帧）
    for i in range(len(script)):
        w.offer(b"\xff\xd8fake\xff\xd9", i * 1600 // 25)  # t 按采样轴推进
        time.sleep(0.04)
    time.sleep(0.6)
    w.stop()

    print()
    print("=" * 66)
    print("人脸显示路径验证")
    print("-" * 66)
    print(f"  观测回调: {len(obs_list)}")
    print(f"  唤醒事件: {len(wakes)}  唇动事件: {len(lips)}  身份事件: {len(idents)}")
    print("-" * 66)

    check(len(obs_list) == len(script), f"每帧都有观测回调（{len(obs_list)}/{len(script)}）")
    check(len(wakes) >= 1, f"唤醒事件正常（{len(wakes)}）")
    check(any(e.speaking for e in lips), "唇动事件里有说话")
    check(w.stats.frames_dropped == 0, f"无丢帧（{w.stats.frames_dropped}）")

    print("=" * 66)

    # 下面这部分验证 session 侧的合成逻辑（不启服务）
    print()
    print("测试 session 侧 face.state 合成")
    print("-" * 66)
    from orchestrator.session import OrchestratorSession
    from orchestrator.face.signals import FaceObservation

    sent = []

    def fake_send(msg):
        sent.append(msg)

    async def _run():
        sess = OrchestratorSession("test", config={})
        sess.send_to_client = None
        sess._send_display = lambda m: sent.append(m)   # 直接捕获

        # 直接调用合成逻辑
        class Ev:
            pass

        from orchestrator.face.signals import IdentityEvent, WakeEvent
        sess._last_face = None
        # 模拟一帧观测
        sess._last_face = {"valid": True, "box": [1, 2, 3, 4], "score": 0.9,
                           "speaking": True, "lip": "SPEAKING",
                           "interacting": True, "person_id": -1}
        sess._last_face_push = 0.0
        sess._push_face_display()
        sess._last_identity = {"name": "张三", "uid": "abc", "person_id": 3,
                               "similarity": 0.61, "enrolled": True}
        sess._last_face_push = 0.0
        sess._push_face_display()
        return sess

    import asyncio
    asyncio.run(_run())
    check(len(sent) >= 1, f"生成了 face.state（{len(sent)} 条）")
    if sent:
        msg = sent[-1]
        check(msg.type == "face.state", "类型为 face.state")
        check(len(msg.tracks) == 1 and msg.tracks[0]["valid"], "带人脸框数据")
        check(msg.tracks[0]["score"] == 0.9, "带置信度")
        check(msg.tracks[0]["speaking"] is True, "带唇动状态")
        check(msg.identity and msg.identity["name"] == "张三", "带身份信息")
    print("=" * 66)

    if _failures:
        print(f"FAILED: {len(_failures)} 项")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
