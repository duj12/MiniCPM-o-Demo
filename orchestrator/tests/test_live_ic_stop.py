#!/usr/bin/env python3
"""**活体验收**：IC 判 `END` 时那条 `/v1/stop` 必须真的打进去。

验的是 2026-10-13 修的那个自锁（见 `orchestrator/interaction/inprocess.py`
模块文档「三个必须 off-loop 的调用」）。`test_ic_tick_offloop.py` 是**单元**
级的（自己搭探针服务），这个打**真实编排服务**，把整条链路串起来验：

    人脸帧 → FaceWorker → 唤醒(dwell≥2000) → IC 判 GREET
      → 编排播欢迎语 → （第二阶段）人走开 → absent_ms ≥ 4000
      → IC 判 END(sop=24)
      → `_notify_sinks` → `ExpressionSink.stop()` → `POST /v1/stop`

## 为什么是 sop=24，而不是看起来更好复现的 sop=23

`policy.py:417` 的 `sop=23`（「欢迎过了、用户一直不开口」）看着更可控
—— 条件里只要 `absent_ms == 0` 且 `greet_elapsed_ms ≥ 10000`。**但它其实
复现不出来**：同一句里还有 `not session.user_spoke_since_greet`，而那个
标记由 `user_opened_since_greet_signal`（`policy.py:173`）置位 —— 它的
第二个分支是「**脸 HIGH 且嘴唇在动**」。实测用的演示视频那人**一直在
讲话**，唇动恒为 HIGH ⇒ 这个标记一进 GREET 就置真 ⇒ sop=23 被永久挡住。

`sop=24` 只需要「人不在场」，是确定性的：推一段**检不出人脸的灰底帧**
（`--blank`）就必然累计 `absent_ms`。这也正是原始 badcase 的判定路径。

## 判据不在这份脚本里

客户端只看得到 `tts.cancel`（而且欢迎语播完就不一定有 cancel 了）。
真正硬的证据在**服务端日志**：跑之前记下游标，跑完看新增行里有没有

    · `IC → END（sop=23）会话收尾`
    · `外部请求停止播报（reason=ic_stop）` + `POST /v1/stop ... 200 OK`
    · **没有** `handshake operation timed out`（修复前每次 END 必有一条）
    · **没有** `⚠️ 事件循环卡顿`

跑法（105 上，用 orch105 环境 —— 它带 websockets）：

    /home/dujing/miniconda3/envs/orch105/bin/python \
        -m orchestrator.tests.test_live_ic_stop \
        --url https://192.168.89.105:8100 \
        --frames /tmp/livetest/frames --blank /tmp/livetest/blank.jpg

⚠️ `--frames` 里是一堆按名排序的 jpg，脚本会**循环推**（素材只有 3.7s，
   但 `sop=23` 要求人「一直在」—— 循环推等效于「人没走」）。
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import ssl
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


def track0(f: dict) -> dict:
    """`face.state` 的字段是**嵌套**的 —— `tracks[0]` 里才是 valid/box/score。"""
    tr = f.get("tracks") or []
    return tr[0] if tr else {}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://192.168.89.105:8100")
    ap.add_argument("--frames", default="/tmp/livetest/frames",
                    help="抽好的 jpg 目录（顺序推、循环推）")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--hold-s", type=float, default=12.0,
                    help="第一阶段（有人的帧）时长。GREET 前要 2s dwell")
    ap.add_argument("--blank", default="/tmp/livetest/blank.jpg",
                    help="第二阶段推的「无人」帧 —— 逼 END(sop=24)")
    ap.add_argument("--absent-s", type=float, default=10.0,
                    help="第二阶段时长。absent_end_ms=4000，留足余量")
    ap.add_argument("--identity", default="live-ic-stop")
    args = ap.parse_args()

    frames = sorted(Path(args.frames).glob("*.jpg"))
    if not frames:
        print(f"[SKIP] {args.frames} 里没有 jpg")
        return 1
    blank = Path(args.blank)
    if not blank.is_file():
        print(f"[SKIP] 没有「无人」帧 {blank}（用 cv2 生成一张同尺寸灰底图）")
        return 1
    blank_b = blank.read_bytes()
    print(f"素材: {len(frames)} 帧有人 + 1 帧无人；"
          f"推 {args.hold_s:.0f}s 有人 → {args.absent_s:.0f}s 无人 @ "
          f"{args.fps}fps")

    import websockets

    ws_url = args.url.replace("https://", "wss://") \
                     .replace("http://", "ws://") + "/v1/orchestrator"
    # 编排服务是**自签证书**，客户端关校验 —— 标准做法，不是偷懒。
    ctx = ssl._create_unverified_context()

    msgs: list[dict] = []
    sid: str | None = None

    async with websockets.connect(ws_url, max_size=64 * 1024 * 1024,
                                  ssl=ctx) as ws:
        await ws.send(json.dumps({"type": "session.start",
                                  "identity": {"worker": args.identity}}))

        async def rx() -> None:
            nonlocal sid
            try:
                while True:
                    m = json.loads(await ws.recv())
                    msgs.append(m)
                    t = m.get("type")
                    if t == "session.ready":
                        sid = m.get("session_id")
                    if t == "tts.audio":
                        # 只报长度，音频本体别往终端刷
                        print(f"    [{time.monotonic() - t0:6.2f}s] tts.audio "
                              f"{len(m.get('audio') or '')}B")
                    elif t in ("tts.cancel", "error", "session.ready"):
                        detail = str(m.get("message") or "")[:60]
                        print(f"    [{time.monotonic() - t0:6.2f}s] {t} {detail}")
            except Exception:  # noqa: BLE001 — 对端关闭是预期内的
                return

        t0 = time.monotonic()
        task = asyncio.create_task(rx())
        for _ in range(400):
            if sid:
                break
            await asyncio.sleep(0.05)
        check(bool(sid), f"会话建立（sid={sid}）")
        if not sid:
            task.cancel()
            return 1

        dt = 1.0 / args.fps
        k = 0

        async def push(payload: bytes, label: str, seconds: float) -> int:
            """推 `seconds` 秒同一张图（按 25fps 循环）。返回推了几帧。"""
            nonlocal k
            n = int(seconds * args.fps)
            for _ in range(n):
                await ws.send(json.dumps({
                    "type": "video_face",
                    "frame_base64": base64.b64encode(payload).decode(),
                    "t_ms": int(k * 1000 / args.fps),
                }))
                k += 1
                if k % int(args.fps * 5) == 0:
                    print(f"    [{time.monotonic() - t0:6.2f}s] {label} "
                          f"已推 {k} 帧（{k / args.fps:.1f}s）")
                await asyncio.sleep(dt)
            return n

        # ① 有人的帧 —— 攒够 dwell(2000ms) 触发 GREET
        for jp in (frames[i % len(frames)]
                   for i in range(int(args.hold_s * args.fps))):
            await ws.send(json.dumps({
                "type": "video_face",
                "frame_base64": base64.b64encode(jp.read_bytes()).decode(),
                "t_ms": int(k * 1000 / args.fps),
            }))
            k += 1
            if k % int(args.fps * 5) == 0:
                print(f"    [{time.monotonic() - t0:6.2f}s] 有人 已推 {k} 帧"
                      f"（{k / args.fps:.1f}s）")
            await asyncio.sleep(dt)

        # ② 无人的帧 —— 逼 absent_ms 越 4000ms ⇒ END(sop=24)
        if not any(m.get("type") == "error" for m in msgs):
            print(f"    [{time.monotonic() - t0:6.2f}s] >>> 切「无人」帧")
            await push(blank_b, "无人", args.absent_s)

        # 再等一会儿收尾消息（END 之后编排要收会话）
        await asyncio.sleep(3)
        task.cancel()

        # ---- 客户端侧能看到的（辅助判据，主判据在服务端日志）----
        kinds: dict[str, int] = {}
        for m in msgs:
            t = m.get("type") or "?"
            kinds[t] = kinds.get(t, 0) + 1
        faces = [m for m in msgs if m.get("type") == "face.state"]
        errs = [m for m in msgs if m.get("type") == "error"]
        valid = [f for f in faces if track0(f).get("valid")]
        wake_dwells = [int((f.get("wake") or {}).get("dwell_ms") or 0)
                       for f in faces if f.get("wake")]

        print(f"\n  收到消息: {kinds}")
        check(len(errs) == 0, f"无 error 消息（{len(errs)} 条）")
        check(len(valid) > 0,
              f"人脸被检出（{len(valid)}/{len(faces)} 条 valid）")
        check(max(wake_dwells or [0]) >= 1500,
              f"发生了唤醒（wake dwell 最大 {max(wake_dwells or [0])}ms，"
              f"阈值 2000ms）⇒ GREET 的前置成立")
        if kinds.get("tts.cancel"):
            print(f"  · 客户端收到 {kinds['tts.cancel']} 条 tts.cancel"
                  f"（IC 的停播命令真的掐断了播放）")

        await ws.close()

    # 让服务端把远端人脸会话释放掉（G1_FACE_MAX_SESSIONS=4，不放会占满）
    await asyncio.sleep(8)

    print(f"\n结果: {_OK} 通过, {_FAIL} 失败")
    print("⚠️ 主判据在服务端日志：见模块文档「判据不在这份脚本里」")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
