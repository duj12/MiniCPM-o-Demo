#!/usr/bin/env python3
"""**两路真会话**并发验收（打真实编排服务，含人脸 / IC / TTS 链路）。

与 `test_ic_inprocess_isolation.py` 的分工：

  · 那个是**纯单元**测试 —— 直接构造两个 `InProcessICClient`，证明 Engine
    互不干扰（快、不依赖服务）。
  · 这个是**端到端**测试 —— 真的开两条 WebSocket 会话，证明编排服务这一层
    也没有把它们串起来（例如显示队列、`_find_session` 路由、人脸 session_id）。

再加一条单元测不出来的：**`/v1/ic/apply_agent` 能按 `X-Session-Id` 精确
路由到指定会话** —— 这是 `ORCH_IC_MODE=inprocess` 的硬前置（Agent 得能
把智脑薄投影投回来）。

    python -m orchestrator.tests.test_ic_multi_session_live \
        --url http://192.168.89.105:8101 --wav assets/asr_input.wav

⚠️ 需要有音频素材；没给 `--wav` 时只验「两路能建起来 + 隔离 + apply_agent
路由」这几条，不推流。
"""
from __future__ import annotations

import argparse
import asyncio
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


#: 编排服务跑**自签证书**（浏览器点「继续访问」即可）。测试客户端关校验 ——
#: 自签场景的标准做法。开着校验时每一个 http 调用都会
#: CERTIFICATE_VERIFY_FAILED，看起来像服务挂了，其实是证书。
_UNVERIFIED = None


def _ssl_ctx(url: str):
    global _UNVERIFIED
    if url.startswith("http://"):
        return None
    if _UNVERIFIED is None:
        import ssl
        _UNVERIFIED = ssl._create_unverified_context()
    return _UNVERIFIED


def _is_rejected(resp: dict) -> bool:
    """判断一次调用是否被**拒了**（404 / ok=false）。

    ⚠️ 不能只查 `"error" in resp`：非 2xx 时 `urlopen` 会抛 `HTTPError`，
    本模块的 `http_post` 把它包成 `{"_error":…, "_body":…}` —— 于是
    错误信息藏在 `_body` 里而不是顶层。早先这里就是这么误判的。
    """
    if resp.get("ok") is False:
        return True
    if "_error" in resp:
        return True
    return False


def _read_wav(path: str):
    """读单声道 float32 波形 + 采样率。

    ⚠️ 优先 `soundfile`，但它在编排服务的运行环境里**不一定装**（orch105 就
    没有）。测试不该为了读一个 wav 去往生产环境装依赖 —— 没有就退回标准库
    `wave`（只支持 16-bit PCM，够用）。
    """
    try:
        import soundfile as sf  # type: ignore
        data, sr = sf.read(path, dtype="float32")
        return data, sr
    except ImportError:
        pass

    import wave

    import numpy as np

    with wave.open(path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise SystemExit(f"只支持 16-bit PCM wav（实际 {sw * 8}-bit）: {path}")
    a = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


async def open_session(ws_url: str, tag: str, sink: dict) -> object:
    """开一条会话，等 `session.ready`，把收到的消息记进 sink。"""
    import websockets

    ws = await websockets.connect(ws_url, max_size=64 * 1024 * 1024,
                                  ssl=_ssl_ctx(ws_url))
    await ws.send(json.dumps({"type": "session.start",
                              "identity": {"worker": tag}}))
    sink["msgs"] = []
    sink["sid"] = None

    async def rx():
        try:
            while True:
                m = json.loads(await ws.recv())
                sink["msgs"].append(m)
                if m.get("type") == "session.ready":
                    sink["sid"] = m.get("session_id")
        except Exception:
            return

    sink["task"] = asyncio.create_task(rx())
    for _ in range(300):
        if sink["sid"]:
            return ws
        await asyncio.sleep(0.05)
    return ws


async def http_post(url: str, body: dict, sid: str | None = None) -> dict:
    import urllib.request

    headers = {"Content-Type": "application/json"}
    if sid:
        headers["X-Session-Id"] = sid
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5,
                                    context=_ssl_ctx(url)) as r:
            return json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        body = ""
        try:
            body = exc.read().decode()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        return {"_error": f"{type(exc).__name__}: {exc}", "_body": body}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://192.168.89.105:8101")
    ap.add_argument("--wav", default="")
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()

    ws_url = args.url.replace("https://", "wss://").replace("http://", "ws://") \
        + "/v1/orchestrator"
    print(f"目标: {args.url}  (ws: {ws_url})")

    # ---- 1. 健康检查 ----
    print("== 1. 服务可达 ==")
    import urllib.request
    try:
        with urllib.request.urlopen(args.url + "/healthz", timeout=5,
                                    context=_ssl_ctx(args.url)) as r:
            hz = json.loads(r.read())
        check(hz.get("status") == "ok", f"/healthz ok（活跃 {hz.get('active_sessions')}）")
    except Exception as exc:  # noqa: BLE001
        check(False, f"/healthz 不可达: {exc}")
        return 1

    # ---- 2. 同时开两路会话 ----
    print("== 2. 两路会话并发建立 ==")
    a, b = {}, {}
    ws_a = await open_session(ws_url, "A", a)
    ws_b = await open_session(ws_url, "B", b)
    check(bool(a.get("sid")), f"会话 A 建立（sid={a.get('sid')}）")
    check(bool(b.get("sid")), f"会话 B 建立（sid={b.get('sid')}）")
    check(a.get("sid") and b.get("sid") and a["sid"] != b["sid"],
          "两路 sid 不同")
    if not (a.get("sid") and b.get("sid")):
        return 1
    sid_a, sid_b = a["sid"], b["sid"]

    # 两路都必须收到 ready（不是"后来者把先来的踢掉"）
    check(any(m.get("type") == "session.ready" for m in a["msgs"]),
          "A 收到 session.ready")
    check(any(m.get("type") == "session.ready" for m in b["msgs"]),
          "B 收到 session.ready")

    # ---- 3. /v1/ic/apply_agent 按 session 精确路由 ----
    print("== 3. apply_agent 按 X-Session-Id 精确路由 ==")
    r1 = await http_post(args.url + "/v1/ic/apply_agent",
                         {"status": "PENDING_ANNOUNCE",
                          "session_end_pending": True}, sid=sid_a)
    check(r1.get("ok") is True, f"投给 A 成功（{r1}）")
    check(r1.get("session_id") == sid_a, "A 的路由命中（返回了自己的 sid）")
    r2 = await http_post(args.url + "/v1/ic/apply_agent",
                         {"status": "BUSY"}, sid=sid_b)
    check(r2.get("ok") is True and r2.get("session_id") == sid_b,
          "B 的路由命中")
    r_bad = await http_post(args.url + "/v1/ic/apply_agent",
                            {"status": "BUSY"}, sid="does-not-exist")
    check(_is_rejected(r_bad),
          f"未知 sid 被拒（不会误改别的会话）—— {r_bad}")
    # 无 sid 且有多路会话时也必须拒（不能瞎猜一路）
    r_nosid = await http_post(args.url + "/v1/ic/apply_agent", {"status": "BUSY"})
    check(_is_rejected(r_nosid),
          "不带 sid 且多路活跃时被拒（不猜）")

    # ---- 4. 推流（可选） ----
    if args.wav:
        print(f"== 4. 推流 {args.seconds}s（两路同时）==")
        import numpy as np
        from orchestrator.protocol import MIC_CHUNK, SR

        data, sr = _read_wav(args.wav)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SR:
            # 简单抽点（测试用，不做重采样质量要求）
            idx = (np.arange(int(len(data) * SR / sr)) * sr / SR).astype(int)
            data = data[np.clip(idx, 0, len(data) - 1)]
        n_chunks = int(args.seconds * SR / MIC_CHUNK)

        async def push(ws, tag):
            import base64
            for i in range(n_chunks):
                seg = data[i * MIC_CHUNK:(i + 1) * MIC_CHUNK]
                if seg.size < MIC_CHUNK:
                    break
                pcm = (np.clip(seg, -1, 1) * 32767).astype("<i2").tobytes()
                await ws.send(json.dumps({
                    "type": "audio",
                    "audio_base64": base64.b64encode(pcm).decode(),
                    "seq": i, "ctx_time": time.monotonic(),
                }))
                await asyncio.sleep(MIC_CHUNK / SR)
            print(f"    {tag}: 推了 {n_chunks} 块")

        await asyncio.gather(push(ws_a, "A"), push(ws_b, "B"))
        await asyncio.sleep(3)
        for tag, s in (("A", a), ("B", b)):
            asr = [m for m in s["msgs"] if m.get("type") == "asr"]
            errs = [m for m in s["msgs"] if m.get("type") == "error"]
            print(f"    {tag}: 消息 {len(s['msgs'])} 条，asr {len(asr)} 条，"
                  f"error {len(errs)} 条")
            check(len(errs) == 0, f"{tag} 无 error 消息")

    # ---- 5. 各自收尾 ----
    print("== 5. 收尾 ==")
    await ws_a.close()
    await ws_b.close()
    await asyncio.sleep(2)
    for s in (a, b):
        t = s.get("task")
        if t:
            t.cancel()
    print(f"\n结果: {_OK} 通过, {_FAIL} 失败")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
