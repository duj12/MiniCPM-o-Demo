#!/usr/bin/env python3
"""阶段 4 验证：G1 人脸模块（唤醒 / 唇动 / 身份识别）。

用 G1 自带的样例录像回放（含唤醒），验证三个输出。跑的是**线上那条路径**
（``G1FaceProvider`` → ``FaceWorker``），所以断言通过就等于线上链路没退化。

    # 在 106 上（g1_root 会自动探测到与本仓库同级的 board-face-and-cloud-infer/G1）
    python -m orchestrator.tests.test_face \
        --mjpeg board-face-and-cloud-infer/G1/sample/camera_original.mjpeg \
        --tsv   board-face-and-cloud-infer/G1/sample/camera_capture_timestamps.tsv

    # 加上身份识别（--db 省略时默认用 <g1_root>/models/face_db.npz）
    ... --db /data/megastore/Projects/DuJing/models/face/face_db.npz

⚠️ 别再用全速灌帧（``--fast``）看结果 —— 队列 maxsize=3 会丢掉绝大部分，
那是**测量方法**的问题，不是产品缺陷（真实浏览器就是 25fps 一条一条来的）。
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
    p.add_argument("--g1-root", default=None,
                   help="G1 仓库根（含 g1face/ 与 models/）；不设则自动探测")
    p.add_argument("--models", default=None,
                   help="模型目录（blazeface.onnx 等）；默认 <g1-root>/models")
    p.add_argument("--so", default=None, help="libsdk_stream.so 路径（默认按架构找）")
    p.add_argument("--mjpeg", required=True)
    p.add_argument("--tsv", default=None)
    p.add_argument("--db", default="__default__",
                   help="face_db.npz；默认 <g1-root>/models/face_db.npz，"
                        "传 none 用空库（不测身份）")
    p.add_argument("--wake-dwell", type=int, default=2000,
                   help="唤醒阈值 ms（默认 2000，与 C 侧 wake_ms_high / IC 同值）")
    p.add_argument("--fast", action="store_true",
                   help="全速回放（**会大量丢帧**，仅用于调试，不用于验证）")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    from collections import Counter

    from orchestrator.config import Settings
    from orchestrator.face.g1face_provider import G1FaceProvider, resolve_g1_root
    from orchestrator.face.worker import FaceWorker

    # 复用线上同一套路径解析（不自己拼 parents[3]）
    cfg = Settings()
    cfg.face_g1_root = args.g1_root or cfg.face_g1_root
    cfg.face_model_dir = args.models or cfg.face_model_dir
    cfg.face_lib_path = args.so or cfg.face_lib_path
    g1_root, how = resolve_g1_root(cfg)

    db = args.db
    if db == "__default__":
        cand = g1_root / "models" / "face_db.npz"
        db = str(cand) if cand.is_file() else None
    elif db in ("none", ""):
        db = None
    want_identify = db is not None
    print(f"G1 根: {g1_root}（{how}）")
    print(f"模型目录: {cfg.face_model_dir or g1_root / 'models'}")
    print(f"人脸库: {db or '（无，跳过身份识别）'}")

    provider = G1FaceProvider(
        str(g1_root),
        model_dir=cfg.face_model_dir,
        lib_path=cfg.face_lib_path,
        db_path=db,
        identify=want_identify,
        no_enroll=True,            # ⚠️ 测试绝不写库
        wake_dwell_ms=args.wake_dwell,
        debug=args.debug,
    )
    print(f"身份识别: {'可用' if provider.available else '不可用（降级）'}")

    events = {"wake": [], "lip": [], "identity": []}
    # 每帧观测也收 —— 用来断言 state 刷新帧真的下发了（旧用例只看事件，
    # 漏掉了「state 一直没出」这种静默退化）
    obs_list: list = []
    worker = FaceWorker(
        provider,
        on_wake=lambda e: events["wake"].append(e),
        on_lip=lambda e: events["lip"].append(e),
        on_identity=lambda e: events["identity"].append(e),
        on_obs=obs_list.append,
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
        print(f"    begin: dwell={w.dwell_ms}ms conf={w.mean_confidence:.3f} box={w.box}")
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
        print(f"    uid={e.uid} name={e.name} sim={e.similarity:.3f} "
              f"state={e.identity_state} enrolled={e.is_enrolled} pid={e.person_id}")
    if idents:
        print(f"    identity_state 分布: "
              f"{dict(Counter(e.identity_state for e in idents))}")

    print(f"  输入帧尺寸: {provider._frame_size}")
    print("-" * 62)
    check(s.frames_processed > 0, "有帧被处理")
    check(s.frames_dropped == 0, f"无丢帧（dropped={s.frames_dropped}）")
    check(len(begins) >= 1, f"至少一次唤醒（{len(begins)} 次）")
    check(len(begins) <= 3, f"唤醒不抖动（{len(begins)} 次 ≤ 3）")
    check(len(lips) > 0, f"有唇动事件（{len(lips)} 条）")
    # ---- 本次改造新增的断言（这才是验收标准）----
    # 尺寸是 UI 叠加人脸框的依据，拿不到就会框错位（旧实现用 cv2 且失败静默）
    fs = provider._frame_size
    check(fs is not None and fs[0] > 0 and fs[1] > 0,
          f"拿到输入帧尺寸（{fs}）—— UI 框映射依赖它")
    # state 刷新帧要真的下发（唤醒判据、IC 的 SOP 都靠它）
    n_state = sum(1 for o in obs_list if o.state is not None)
    check(n_state > 0, f"有 state 刷新帧下发（{n_state} 帧）")
    # 唤醒必须真的是「熬够了 dwell」才触发 —— 这是改用 dwell 判据的核心
    bad = [w for w in begins if w.dwell_ms < args.wake_dwell]
    check(not bad,
          f"begin 的 dwell 均 ≥ 阈值 {args.wake_dwell}ms"
          + (f"（有 {len(bad)} 条不满足）" if bad else ""))
    # 身份结果必须带 uid（person_id 在线上是同一个兜底值，认人只能靠 uid）
    if want_identify and provider.available:
        check(len(idents) >= 1, f"有身份识别事件（{len(idents)} 条）")
        check(any(e.uid for e in idents),
              "至少一条身份带 uid（UI 认人靠它，不是 person_id）")
    print("=" * 62)

    if _failures:
        print(f"FAILED: {len(_failures)} 项")
        for f in _failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("通过")


if __name__ == "__main__":
    main()
