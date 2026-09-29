#!/usr/bin/env python3
"""**远端人脸**端到端验收：真开一路会话，把 mjpeg 逐帧推给它。

验的是 `face/remote_provider.py` 那条路在**真实编排进程里**通不通 ——
单独的 provider 单测（`test_face.py --face-service-url`）只测到 provider
本身，测不到「会话把帧喂给 FaceWorker → provider → 远端服务 → 观测回到
session」这一整条。

重点看：

  · 人脸线程真的在处理帧（`in=` / `processed=` 不为 0）
  · `_frame_size` 拿到真实尺寸（UI 画框要用）
  · 唤醒发生在正确的时间（`dwell_ms` 越过阈值）—— 这一条能挡住
    「把音频采样序号当微秒发给服务端」那个坑
  · 全程没有 ERROR

    python -m orchestrator.tests.test_face_remote_live \
        --url http://192.168.89.105:8101 \
        --mjpeg board-face-and-cloud-infer/G1/sample/camera_original.mjpeg \
        --max-frames 120
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
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


def split_mjpeg(raw: bytes) -> list[bytes]:
    out, i = [], 0
    while True:
        s = raw.find(b"\xff\xd8", i)
        if s < 0:
            break
        e = raw.find(b"\xff\xd9", s)
        if e < 0:
            break
        out.append(raw[s:e + 2])
        i = e + 2
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://192.168.89.105:8101")
    ap.add_argument("--mjpeg", required=True)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()

    frames = split_mjpeg(Path(args.mjpeg).read_bytes())[:args.max_frames]
    print(f"素材: {len(frames)} 帧 JPEG")
    if not frames:
        print("没有可用帧")
        return 1

    import ssl as _ssl
    import websockets
    ws_url = args.url.replace("https://", "wss://").replace("http://", "ws://") \
        + "/v1/orchestrator"
    msgs: list[dict] = []
    sid = None
    # 编排服务跑的是**自签证书**（浏览器侧点「继续访问」即可），
    # 测试客户端直接关校验 —— 这是自签证书场景的标准做法，不是偷懒。
    ssl_ctx = (_ssl.create_default_context()
               if args.url.startswith("http://") else
               _ssl._create_unverified_context())

    async with websockets.connect(ws_url, max_size=64 * 1024 * 1024,
                                  ssl=ssl_ctx) as ws:
        await ws.send(json.dumps({"type": "session.start",
                                  "identity": {"worker": "facelive"}}))

        async def rx():
            nonlocal sid
            try:
                while True:
                    m = json.loads(await ws.recv())
                    msgs.append(m)
                    if m.get("type") == "session.ready":
                        sid = m.get("session_id")
            except Exception:
                return

        task = asyncio.create_task(rx())
        for _ in range(400):
            if sid:
                break
            await asyncio.sleep(0.05)
        check(bool(sid), f"会话建立（sid={sid}）")
        if not sid:
            return 1

        print(f"== 推 {len(frames)} 帧（约 {len(frames)/args.fps:.1f}s @ {args.fps}fps）==")
        dt = 1.0 / args.fps
        for k, jp in enumerate(frames):
            await ws.send(json.dumps({
                "type": "video_face",
                "frame_base64": base64.b64encode(jp).decode("ascii"),
                "t_ms": int(k * 1000 / args.fps),
            }))
            await asyncio.sleep(dt)
        # 留时间给人脸线程排空队列 + 出 5Hz state
        await asyncio.sleep(3)

        faces = [m for m in msgs if m.get("type") == "face.state"]
        errs = [m for m in msgs if m.get("type") == "error"]
        print(f"  收到 face.state {len(faces)} 条，error {len(errs)} 条")
        check(len(errs) == 0, "无 error 消息")
        check(len(faces) > 0, "收到了 face.state（人脸线程在出观测）")

        # ⚠️ `face.state` 的字段是**嵌套**的：`tracks[0]` 里才是
        #    valid/box/score/interacting，帧尺寸在 `tracks[0].src_w/src_h`。
        #    早先这里按顶层 `f["valid"]` 取，永远取不到 → 误报「没有人脸」。
        def track0(f: dict) -> dict:
            tr = f.get("tracks") or []
            return tr[0] if tr else {}

        if faces:
            valid = [f for f in faces if track0(f).get("valid")]
            check(len(valid) > 0,
                  f"有人脸被检出（{len(valid)}/{len(faces)} 条 valid）")
            if valid:
                t0 = track0(valid[-1])
                sz = (t0.get("src_w"), t0.get("src_h"))
                check(all(sz), f"_frame_size 有值 {sz}（UI 画框要用）")
                # ⚠️ `dwell_ms` **只在 5Hz 的 state 快照里**（以及 wake 事件里），
                #    `tracks[0]` 里根本没有 —— 会话侧压根没往那儿放。
                #    所以要从 `state` 取。这跟 `interacting`（每帧都算）
                #    是两个不同节奏的东西，别混。
                dwells = [int((f.get("state") or {}).get("dwell_ms") or 0)
                          for f in faces]
                wake_dwells = [int((f.get("wake") or {}).get("dwell_ms") or 0)
                               for f in faces if f.get("wake")]
                best = max(dwells + wake_dwells + [0])
                check(best > 0,
                      f"dwell_ms 在增长（state max={max(dwells)}ms, "
                      f"wake max={max(wake_dwells)}ms）—— "
                      f"这一条能挡住「拿音频采样当微秒」的坑")
                # wake 事件的 dwell 最直接：它就是「熬够多久才唤醒的」证据
                check(max(wake_dwells or [0]) >= 1500,
                      f"唤醒时的 dwell≈阈值（{max(wake_dwells or [0])}ms，"
                      f"阈值 2000ms）")
                woke = [f for f in faces if track0(f).get("interacting")]
                check(len(woke) > 0,
                      f"出现了唤醒（{len(woke)} 条 interacting=True）")
                # 身份：从 state 或顶层 identity 里找
                ids = [f.get("identity") for f in faces if f.get("identity")]
                idents = [s.get("identity_id") for f in faces
                          for s in [f.get("state") or {}] if s.get("identity_id")]
                check(bool(ids or idents),
                      f"身份结果透出来了（identity {len(ids)} 条 / state 里 "
                      f"{len(set(idents))} 个 uid）")

        task.cancel()
        await ws.close()

    # ⚠️ **必须等服务端把远端的会话释放掉。**
    #
    # 服务端 `RemoteFaceProvider.close()` 会打 `POST /face/close`，但那发生在
    # 会话收尾里（drain 之后）。而 105 的 `G1_FACE_MAX_SESSIONS=4` ——
    # 测试跑几轮不放手就会把槽位占满，后面每轮都 503、一条 face.state 都出不来
    # （实测：连跑 4 轮后第 5 轮全失败，现象是「人脸突然坏了」）。
    # 所以这里等一会儿，让服务端把会话关掉。
    await asyncio.sleep(8)
    try:
        import urllib.request
        with urllib.request.urlopen(
                args.url.replace("/v1", "") + "/health", timeout=5) as r:
            h = json.loads(r.read())
        print(f"  人脸服务会话数: {h.get('sessions')}/{h.get('max_sessions')}"
              f"（应为 0 —— 不为 0 说明有泄漏，下轮可能撞上限）")
    except Exception:  # noqa: BLE001
        pass

    print(f"\n结果: {_OK} 通过, {_FAIL} 失败")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
