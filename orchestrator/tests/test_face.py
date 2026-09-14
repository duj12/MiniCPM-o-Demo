#!/usr/bin/env python3
"""阶段 4 验证：G1 人脸模块（唤醒 / 唇动 / 身份识别）。

用 G1 自带的样例录像回放（含唤醒），验证三个输出。

    # 在 106 上
    export G1_FACE_DEBUG=0
    python -m orchestrator.tests.test_face \
        --so  board-face-and-cloud-infer/G1/lib/libsdk_stream.so \
        --models board-face-and-cloud-infer/G1/models \
        --mjpeg board-face-and-cloud-infer/G1/sample/camera_original.mjpeg \
        --tsv   board-face-and-cloud-infer/G1/sample/camera_capture_timestamps.tsv

    # 加上身份识别（需 buffalo_l + face_db.npz）
    ... --db /data/megastore/Projects/DuJing/models/face/face_db.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_failures: list = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


def read_mjpeg_stream(path: str):
    """按 SOI/EOI 拆分 MJPEG 流为逐帧 JPEG。"""
    data = Path(path).read_bytes()
    frames = []
    i = 0
    n = len(data)
    while i < n - 1:
        if data[i] == 0xFF and data[i + 1] == 0xD8:      # SOI
            j = i + 2
            while j < n - 1:
                if data[j] == 0xFF and data[j + 1] == 0xD9:  # EOI
                    frames.append(data[i:j + 2])
                    i = j + 2
                    break
                j += 1
            else:
                break
        else:
            i += 1
    return frames


def read_tsv(path: str):
    """读时间戳 tsv。

    实际格式（带表头，制表符分隔）::

        capture_index  accepted_frame_index  sequence
        driver_timestamp_us  arrival_us  frame_age_us
        accepted  bytes  width  height  src

    返回 ``accepted=1`` 行的 ``driver_timestamp_us`` 列表。
    """
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return []
    # 定位列（表头可能变，按名字找更稳）
    header = [c.strip().lower() for c in lines[0].split("\t")]
    try:
        i_ts = header.index("driver_timestamp_us")
    except ValueError:
        i_ts = 3
    try:
        i_ok = header.index("accepted")
    except ValueError:
        i_ok = 6
    ts = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) <= max(i_ts, i_ok):
            continue
        try:
            if int(float(parts[i_ok])) != 1:
                continue
            ts.append(int(float(parts[i_ts])))
        except ValueError:
            continue
    return ts


def main() -> None:
    p = argparse.ArgumentParser(description="G1 人脸模块验证")
    p.add_argument("--so", required=True, help="libsdk_stream.so 路径")
    p.add_argument("--models", required=True, help="模型目录（blazeface.onnx 等）")
    p.add_argument("--mjpeg", required=True)
    p.add_argument("--tsv", default=None)
    p.add_argument("--db", default=None, help="face_db.npz（给了才测身份识别）")
    p.add_argument("--fast", action="store_true",
                   help="全速回放（**会大量丢帧**，仅用于调试，不用于验证）")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    from orchestrator.face.signals import FaceObservation, IdentityEvent, LipEvent, WakeEvent
    from orchestrator.face.worker import FaceWorker
    from orchestrator.face.local_provider import LocalFaceProvider, PersonIdMap

    print(f"加载模型: {args.models}")
    svc = None
    if args.db:
        from orchestrator.face.local_provider import load_face_service
        svc, err = load_face_service(args.models, args.db)
        if svc is None:
            print(f"  [!] FaceService 不可用（{err}）—— 跳过身份识别")
        else:
            print("  身份识别已启用")

    provider = LocalFaceProvider(
        args.so, args.models, face_service=svc,
        id_map=PersonIdMap(None), identify_enabled=svc is not None,
        identify_cooldown_s=2.0, debug=args.debug,
    )

    events = {"wake": [], "lip": [], "identity": []}
    worker = FaceWorker(
        provider,
        on_wake=lambda e: events["wake"].append(e),
        on_lip=lambda e: events["lip"].append(e),
        on_identity=lambda e: events["identity"].append(e),
    )

    frames = read_mjpeg_stream(args.mjpeg)
    ts = read_tsv(args.tsv) if args.tsv else []
    if args.max_frames:
        frames = frames[: args.max_frames]
    print(f"回放 {len(frames)} 帧（时间戳 {len(ts)} 条）")

    # ⚠️ 必须按真实帧率投递。全速灌入 120 帧会让 maxsize=3 的队列丢掉
    # 绝大部分 —— 那是**测量方法**的问题，不是产品缺陷（真实的浏览器
    # 就是 25fps 一条一条来的）。有 tsv 用它的时间戳，否则按 25fps 节流。
    worker.start()
    t_wall0 = time.monotonic()
    ts0 = ts[0] if ts else 0
    for i, jpeg in enumerate(frames):
        if not args.fast:
            if i < len(ts):
                target = (ts[i] - ts0) / 1e6
            else:
                target = i / 25.0        # 无时间戳时按 25fps
            d = target - (time.monotonic() - t_wall0)
            if d > 0:
                time.sleep(d)
        # 会话采样轴时刻：用单调墙钟换算（真实场景由 SampleClock 提供）
        t = int((time.monotonic() - t_wall0) * 16000)
        worker.offer(jpeg, t)

    # 等队列排空：不能只看 _q.empty()（刚取出、还在处理时也是空的），
    # 要看 frames_processed + frames_dropped 是否追上了 frames_in。
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        s = worker.stats
        if s.frames_processed + s.frames_dropped >= s.frames_in and s.frames_in > 0:
            break
        time.sleep(0.05)
    time.sleep(0.3)
    worker.stop()

    # ---------------- 断言 ----------------
    print()
    print("=" * 62)
    print("G1 人脸模块验证结果")
    print("-" * 62)
    s = worker.stats
    print(f"  帧: in={s.frames_in} processed={s.frames_processed} dropped={s.frames_dropped}")
    if s.last_error:
        print(f"  last_error: {s.last_error}")

    wakes = events["wake"]
    begins = [w for w in wakes if w.phase == "begin"]
    ends = [w for w in wakes if w.phase == "end"]
    print(f"  唤醒事件: begin={len(begins)} end={len(ends)}")
    for w in begins[:3]:
        print(f"    begin: conf={w.mean_confidence:.3f} box={w.box}")
    for w in ends[:3]:
        print(f"    end:   dwell={w.dwell_ms}ms")

    lips = events["lip"]
    speaking = [e for e in lips if e.speaking]
    print(f"  唇动事件: {len(lips)}（其中说话 {len(speaking)}）")
    states = {}
    for e in lips:
        states[e.lip_state] = states.get(e.lip_state, 0) + 1
    print(f"    lip_state 分布: {states}")

    idents = events["identity"]
    print(f"  身份事件: {len(idents)}")
    for e in idents[:3]:
        print(f"    person_id={e.person_id} name={e.name} sim={e.similarity:.3f}")

    print("-" * 62)
    check(s.frames_processed > 0, "有帧被处理")
    check(s.frames_dropped == 0, f"无丢帧（dropped={s.frames_dropped}）")
    check(len(begins) >= 1, f"至少一次唤醒（{len(begins)} 次）")
    check(len(begins) <= 3, f"唤醒不抖动（{len(begins)} 次 ≤ 3）")
    check(len(lips) > 0, f"有唇动事件（{len(lips)} 条）")
    if args.db and svc is not None:
        check(len(idents) >= 1, f"有身份识别事件（{len(idents)} 条）")
        enrolled = [e for e in idents if e.is_enrolled]
        print(f"    （在库 {len(enrolled)} 条）")
    print("=" * 62)

    if _failures:
        print(f"FAILED: {len(_failures)} 项")
        for f in _failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("通过")


if __name__ == "__main__":
    main()
